import torch
import torch.nn as nn
from src.ml_pipeline.models.attention_models.attention_mechansims import *
from src.ml_pipeline.models.late_fusion_models.soft_voting import *
from src.ml_pipeline.models.late_fusion_models.hard_voting import *
from src.ml_pipeline.models.late_fusion_models.kalman import * 

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
                    self.cross_attention_blocks[f"{modality1}_to_{modality2}"] = (
                        nn.ModuleList(
                            [
                                CrossAttentionEncoder(
                                    self.embed_dim, self.n_head, self.dropout
                                )
                                for _ in range(self.n_bcsa)
                            ]
                        )
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

class MOSCAN(nn.Module):
    NAME = "MOSCAN"
    def __init__(self, **kwargs):
        super(MOSCAN, self).__init__()
        required_params = [
            "query_cache",
            "input_dims",
            "embed_dim",
            "hidden_dim",
            "output_dim",
            "n_head_gen",
            "dropout",
            "attention_dropout",
            "n_bcsa",
            "batch_size",
            "seq_length",
            "max_seq_length",
            "max_batch_size",
            "active_sensors",
            "predictor",
            "device"
        ]

        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        self.query_cache = kwargs["query_cache"]
        self.input_dims = kwargs["input_dims"]
        self.embed_dim = kwargs["embed_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.output_dim = kwargs["output_dim"]
        self.n_head = kwargs["n_head_gen"]
        self.dropout = kwargs["dropout"]
        self.attention_dropout = kwargs["attention_dropout"]
        self.n_bcsa = kwargs["n_bcsa"]
        self.batch_size = kwargs["batch_size"]
        self.seq_length = kwargs["seq_length"]
        self.max_seq_length = kwargs["max_seq_length"]
        self.max_batch_size = kwargs["max_batch_size"]
        self.active_sensors = kwargs["active_sensors"]
        predictor = kwargs["predictor"]
        self.device = kwargs["device"]
        self.kalman = kwargs.get("kalman", False)

        self.modalities = nn.ModuleDict()
        self.cross_attention_blocks = nn.ModuleDict()
        self.self_attention_blocks = nn.ModuleDict()

        # Initialize the self-attention blocks for each modality
        for modality in self.input_dims:
            modality_net = nn.ModuleDict(
                {
                    "embedding": nn.Linear(self.input_dims[modality], self.embed_dim),
                    "pos_enc": PositionalEncoding(self.embed_dim)
                }
            )
            self.modalities[modality] = modality_net
            self.self_attention_blocks[modality] = nn.ModuleList(
                [
                    CachedSlidingAttentionEncoder(
                        self.embed_dim, self.hidden_dim, self.n_head, self.max_batch_size, self.max_seq_length, self.dropout, self.attention_dropout, query_cache=self.query_cache 
                    )
                    for _ in range(self.n_bcsa)
                ]
            )

        # Initialize the cross-attention blocks for each modality pair
        modalities = list(self.input_dims.keys())
        for i, modality1 in enumerate(modalities):
            for j, modality2 in enumerate(modalities):
                if i != j:
                    self.cross_attention_blocks[f"{modality1}_to_{modality2}"] = (
                        nn.ModuleList(
                            [
                                CachedSlidingAttentionEncoder(
                                    self.embed_dim, self.hidden_dim, self.n_head, self.max_batch_size, self.max_seq_length, self.dropout, self.attention_dropout, query_cache=self.query_cache 
                                )
                                for _ in range(self.n_bcsa)
                            ]
                        )
                    )

        match predictor:
            case "avg_pool":
                self.predictor = ModularPool(self.embed_dim, self.output_dim, self.dropout, pool_type='avg', return_branch_outputs=self.kalman)
            case "max_pool":
                self.predictor = ModularPool(self.embed_dim, self.output_dim, self.dropout, pool_type='max', return_branch_outputs=self.kalman)
            case "weighted_avg_pool":
                self.predictor = ModularWeightedPool(
                    self.embed_dim, self.output_dim, self.dropout, self.active_sensors, pool_type='avg'
                )
            case "weighted_max_pool":
                self.predictor = ModularWeightedPool(
                    self.embed_dim, self.output_dim, self.dropout, self.active_sensors, pool_type='max'
                )
            case "hard_voting":
                self.predictor = ModularHardVoting(
                    self.embed_dim, self.output_dim, self.dropout, self.active_sensors, pool_type='avg'
                )
            case "stacked_avg_pool":
                self.predictor = StackedModularPool(self.embed_dim, self.hidden_dim, self.output_dim, self.dropout)
            case "stacked_max_pool":
                self.predictor = StackedModularPool(self.embed_dim, self.hidden_dim, self.output_dim, self.dropout, pool_type='max')
            case _:
                raise ValueError(f"Predictor {predictor} not supported")
        
        if self.kalman:
            self.kalman_filter = KalmanFilter(self.output_dim, device=self.device, num_branches=len(self.input_dims))

    def forward(self, inputs):
        # Step 1: Embedding and Positional Encoding for each modality
        modality_features = {}
        for modality, net in self.modalities.items():

            # For modularity: only process the modality if it is present in the input
            if modality in inputs:
                x = inputs[modality]  # Input for the modality
                x = x.permute(
                    0, 2, 1
                )  # Change shape to [batch_size, seq_len, features] from [batch_size, features, seq_len]
                x = net["embedding"](x)
                x = net["pos_enc"](x)
                modality_features[modality] = x

        # Step 2: Bidirectional Cross-Attention and Self-Attention Blocks
        for i in range(self.n_bcsa):
            # Bidirectional Cross-Attention (will be skipped if unimodal)
            for modality1 in modality_features:
                for modality2 in modality_features:
                    if modality1 != modality2:
                        ca_block = self.cross_attention_blocks[f'{modality1}_to_{modality2}'][i]
                        modality_features[modality1] = ca_block(modality_features[modality1], modality_features[modality2], modality_features[modality2], use_cache=self.seq_length>1)
        
            # Self Attention
            for modality, net in self.modalities.items():
                sa_block = self.self_attention_blocks[modality][i]
                modality_features[modality] = sa_block(modality_features[modality], modality_features[modality], modality_features[modality], use_cache=self.seq_length > 1)

        # Step 3: Predictor to merge branches and perform late fusion to produce an overall classification or a per branch classification 
        classification = self.predictor(modality_features)

        # Step 4: Optional Kalman Filter expects shape (batch_size, num_branches, output_dim)
        if self.kalman:
            classification = self.kalman_filter.forward(classification)

        return classification

    def reset_attention_cache(self):
        for modality in self.self_attention_blocks:
            for block in self.self_attention_blocks[modality]:
                block.clear_cache()
        for modality in self.cross_attention_blocks:
            for block in self.cross_attention_blocks[modality]:
                block.clear_cache()