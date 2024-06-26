import torch
import torch.nn as nn
from src.ml_pipeline.models.attention_models.attention_mechansims import *
from src.ml_pipeline.models.late_fusion_models.soft_voting import *


class ModularBCSA(nn.Module):
    NAME = "ModularBCSA"

    def __init__(self, **kwargs):
        required_params = [
            "input_dims",
            "embed_dim",
            "hidden_dim",
            "output_dim",
            "n_head_gen",
            "dropout",
            "n_bcsa",
            "batch_size",
        ]

        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        self.input_dims = kwargs["input_dims"]
        self.embed_dim = kwargs["embed_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.output_dim = kwargs["output_dim"]
        self.n_head = kwargs["n_head_gen"]
        self.dropout = kwargs["dropout"]
        self.n_bcsa = kwargs["n_bcsa"]
        self.batch_size = kwargs["batch_size"]

        super(ModularBCSA, self).__init__()

        self.modalities = nn.ModuleDict()
        self.cross_attention_blocks = nn.ModuleDict()

        for modality in self.input_dims:
            modality_net = nn.ModuleDict(
                {
                    "embedding": nn.Linear(self.input_dims[modality], self.embed_dim),
                    "pos_enc": PositionalEncoding(self.embed_dim),
                    "enc1": SelfAttentionEncoder(
                        self.embed_dim,
                        ffn_hidden=self.hidden_dim,
                        n_head=self.n_head,
                        drop_prob=self.dropout,
                    ),
                    "flatten": nn.Flatten(),
                    "linear": nn.Linear(self.embed_dim * 2, self.hidden_dim),
                    "relu": nn.ReLU(),
                    "dropout_out": nn.Dropout(p=self.dropout),
                    "output": nn.Linear(self.hidden_dim, self.output_dim),
                }
            )
            self.modalities[modality] = modality_net

        modalities = list(self.input_dims.keys())
        for i, modality1 in enumerate(modalities):
            for j, modality2 in enumerate(modalities):
                if i != j:
                    self.cross_attention_blocks[
                        f"{modality1}_to_{modality2}"
                    ] = nn.ModuleList(
                        [
                            CrossAttentionEncoder(
                                self.embed_dim, self.n_head, self.dropout
                            )
                            for _ in range(self.n_bcsa)
                        ]
                    )

        self.relu = nn.ReLU()
        self.dropout_out = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(
            len(self.input_dims) * self.hidden_dim, self.output_dim
        )

    def forward(self, inputs):
        modality_outputs = {modality: [] for modality in self.input_dims}

        # Process each modality input into an embedding
        for modality, x in inputs.items():
            (
                batch_size,
                features,
                seq_len,
            ) = x.shape  # shape is [batch_size, features, seq_len]
            x = x.permute(0, 2, 1)  # Change shape to [batch_size, seq_len, features]
            x_emb = self.modalities[modality]["embedding"](x)
            positional_x = self.modalities[modality]["pos_enc"](x_emb)
            modality_outputs[modality] = positional_x

        # for each BCSA module required, perform bidirectional cross and self attention
        for _ in range(self.n_bcsa):
            updated_outputs = {}
            for modality1 in modality_outputs:
                x1 = modality_outputs[modality1]
                for modality2 in modality_outputs:
                    if modality1 != modality2:
                        x2 = modality_outputs[modality2]
                        x1 = self.cross_attention_blocks[f"{modality2}_to_{modality1}"][
                            _
                        ](x1, x2, x2)
                updated_outputs[modality1] = x1
            modality_outputs = updated_outputs

        final_modality_outputs = []
        for modality, x in modality_outputs.items():
            attn1 = self.modalities[modality]["enc1"](x)
            avg_pool = torch.mean(attn1, 1)
            max_pool, _ = torch.max(attn1, 1)
            concat = torch.cat((avg_pool, max_pool), 1)
            concat_ = self.modalities[modality]["relu"](
                self.modalities[modality]["linear"](concat)
            )
            concat_ = self.modalities[modality]["dropout_out"](concat_)
            modality_output = self.modalities[modality]["output"](concat_)
            final_modality_outputs.append(concat_)

        concat = torch.cat(final_modality_outputs, dim=1)
        concat = self.relu(concat)
        concat = self.dropout_out(concat)

        final_output = self.output_layer(concat)
        return final_modality_outputs, final_output


class MARCONet(nn.Module):
    NAME = "MARCONet"

    def __init__(self, **kwargs):
        required_params = [
            "input_dims",
            "embed_dim",
            "hidden_dim",
            "output_dim",
            "n_head_gen",
            "dropout",
            "n_bcsa",
            "batch_size",
            "token_length",
        ]

        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        self.input_dims = kwargs["input_dims"]
        self.embed_dim = kwargs["embed_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.output_dim = kwargs["output_dim"]
        self.n_head = kwargs["n_head_gen"]
        self.dropout = kwargs["dropout"]
        self.n_bcsa = kwargs["n_bcsa"]
        self.batch_size = kwargs["batch_size"]
        self.token_length = kwargs["token_length"]

        # If val_model is set to True, the model will use cache for the sliding co-attention
        self.val_model = kwargs.get("val_model", False)

        super(MARCONet, self).__init__()

        self.modalities = nn.ModuleDict()
        self.cross_attention_blocks = nn.ModuleDict()
        self.self_attention_blocks = nn.ModuleDict()

        for modality in self.input_dims:
            modality_net = nn.ModuleDict(
                {
                    "embedding": nn.Linear(self.input_dims[modality], self.embed_dim),
                    "pos_enc": PositionalEncoding(self.embed_dim),
                    "predictor": ModularAvgPool(
                        self.embed_dim, self.hidden_dim, self.output_dim, self.dropout
                    ),
                }
            )
            self.modalities[modality] = modality_net
            self.self_attention_blocks[modality] = nn.ModuleList(
                [
                    CachedSlidngSelfAttentionEncoder(
                        self.embed_dim, self.hidden_dim, self.n_head, self.dropout
                    )
                    for _ in range(self.n_bcsa)
                ]
            )

        modalities = list(self.input_dims.keys())
        for i, modality1 in enumerate(modalities):
            for j, modality2 in enumerate(modalities):
                if i != j:
                    self.cross_attention_blocks[
                        f"{modality1}_to_{modality2}"
                    ] = nn.ModuleList(
                        [
                            CachedSlidingCrossAttentionEncoder(
                                d_model=self.embed_dim,
                                ffn_hidden=self.hidden_dim,
                                n_head=self.n_head,
                                drop_prob=self.dropout,
                            )
                            for _ in range(self.n_bcsa)
                        ]
                    )

    def forward(self, inputs):
        # Step 1: Embedding and Positional Encoding for each modality
        modality_features = {}
        for modality, net in self.modalities.items():
            x = inputs[modality]  # Input for the modality
            x = x.permute(
                0, 2, 1
            )  # Change shape to [batch_size, seq_len, features] from [batch_size, features, seq_len]
            x = net["embedding"](x)
            x = net["pos_enc"](x)
            modality_features[modality] = x

        # Step 2: Cross-Attention and Self-Attention Blocks
        for i in range(self.n_bcsa):
            # for modality1 in modality_features:
            #     for modality2 in modality_features:
            #         if modality1 != modality2:
            #             ca_block = self.cross_attention_blocks[f'{modality1}_to_{modality2}'][i]
            #             modality_features[modality1] = ca_block(modality_features[modality1], modality_features[modality2], self.token_length, use_cache=self.val_model)

            for modality, net in self.modalities.items():
                sa_block = self.self_attention_blocks[modality][i]
                modality_features[modality] = sa_block(
                    modality_features[modality],
                    self.token_length,
                    use_cache=self.val_model,
                )

        # Step 3: Merge branches into one tensor and call Predictor
        concatenated_features = torch.cat(list(modality_features.values()), dim=1)
        concatenated_features = concatenated_features.permute(
            0, 2, 1
        )  # change to shape (batch_size, embed_dim, n_branches)
        final_output = net["predictor"](concatenated_features)

        return final_output
