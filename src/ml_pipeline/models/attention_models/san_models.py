import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.ml_pipeline.utils import load_generalized_model
from src.ml_pipeline.models.attention_models.attention_mechansims import (
    PositionalEncoding,
    SelfAttentionEncoder,
)


class ModularModalityFusionNet(torch.nn.Module):
    NAME = "ModularModalityFusionNet"

    def __init__(self, **kwargs):
        required_params = [
            "input_dims",
            "embed_dim",
            "hidden_dim",
            "output_dim",
            "n_head_gen",
            "dropout",
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

        super(ModularModalityFusionNet, self).__init__()

        self.modalities = nn.ModuleDict()

        for modality in self.input_dims:
            modality_net = nn.ModuleDict(
                {
                    "embedding": nn.Linear(self.input_dims[modality], self.embed_dim),
                    "pos_enc": PositionalEncoding(self.embed_dim),
                    "enc1": SelfAttentionEncoder(
                        self.embed_dim,
                        ffn_hidden=128,
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

        self.relu = nn.ReLU()
        self.dropout_out = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(
            len(self.input_dims) * self.hidden_dim, self.output_dim
        )

    def forward(self, inputs):
        modality_outputs = []

        for modality, x in inputs.items():
            x = x.permute(
                0, 2, 1
            )  # Change shape to [batch_size, seq_len, features] from [batch_size, features, seq_len]
            x_emb = self.modalities[modality]["embedding"](x)
            positional_x = self.modalities[modality]["pos_enc"](x_emb)
            attn1 = self.modalities[modality]["enc1"](positional_x)
            avg_pool = torch.mean(attn1, 1)
            max_pool, _ = torch.max(attn1, 1)
            concat = torch.cat((avg_pool, max_pool), 1)
            concat_ = self.modalities[modality]["relu"](
                self.modalities[modality]["linear"](concat)
            )
            concat_ = self.modalities[modality]["dropout_out"](concat_)
            modality_output = self.modalities[modality]["output"](concat_)
            modality_outputs.append(concat_)

        concat = torch.cat(modality_outputs, dim=1)
        concat = self.relu(concat)
        concat = self.dropout_out(concat)

        final_output = self.output_layer(concat)
        return modality_outputs, final_output


class PersonalizedModalityFusionNet(nn.Module):
    NAME = "PersonalizedModalityFusionNet"

    def __init__(self, generalized_model_path, model_class, **kwargs):
        super(PersonalizedModalityFusionNet, self).__init__()

        required_params = [
            "input_dims",
            "embed_dim",
            "hidden_dim",
            "output_dim",
            "n_head_gen",
            "n_head_per",
            "dropout",
        ]

        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        self.input_dims = kwargs["input_dims"]
        self.embed_dim = kwargs["embed_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.output_dim = kwargs["output_dim"]
        self.n_head_gen = kwargs["n_head_gen"]
        self.n_head_per = kwargs["n_head_per"]
        self.dropout = kwargs["dropout"]

        # Load the generalized model
        generalized_model = load_generalized_model(
            generalized_model_path, model_class, **kwargs
        )

        self.generalized_modalities = generalized_model.modalities

        # Freeze generalized model parameters
        for param in self.generalized_modalities.parameters():
            param.requires_grad = False

        self.personalized_modalities = nn.ModuleDict()

        for modality in self.input_dims:
            modality_net = nn.ModuleDict(
                {
                    "embedding": nn.Linear(self.input_dims[modality], self.embed_dim),
                    "pos_enc": PositionalEncoding(self.embed_dim),
                    "enc1": SelfAttentionEncoder(
                        self.embed_dim,
                        ffn_hidden=128,
                        n_head=self.n_head_per,
                        drop_prob=self.dropout,
                    ),
                    "flatten": nn.Flatten(),
                    "linear": nn.Linear(self.embed_dim * 2, self.hidden_dim),
                    "relu": nn.ReLU(),
                    "dropout_out": nn.Dropout(p=self.dropout),
                    "output": nn.Linear(self.hidden_dim, self.output_dim),
                }
            )
            self.personalized_modalities[modality] = modality_net

        self.relu = nn.ReLU()
        self.dropout_out = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(
            len(self.input_dims) * self.hidden_dim * 2, self.output_dim
        )

    def forward(self, inputs):
        generalized_modality_outputs = []
        personalized_modality_outputs = []

        for modality, x in inputs.items():
            x = x.permute(
                0, 2, 1
            )  # Change shape to [batch_size, seq_len, features] from [batch_size, features, seq_len]

            # Generalized path
            x_emb_gen = self.modalities[modality]["embedding"](x)
            positional_x_gen = self.generalized_modalities[modality]["pos_enc"](
                x_emb_gen
            )
            attn1_gen = self.generalized_modalities[modality]["enc1"](positional_x_gen)
            avg_pool_gen = torch.mean(attn1_gen, 1)
            max_pool_gen, _ = torch.max(attn1_gen, 1)
            concat_gen = torch.cat((avg_pool_gen, max_pool_gen), 1)
            concat_gen_ = self.generalized_modalities[modality]["relu"](
                self.generalized_modalities[modality]["linear"](concat_gen)
            )
            concat_gen_ = self.generalized_modalities[modality]["dropout_out"](
                concat_gen_
            )
            generalized_modality_outputs.append(concat_gen_)

            # Personalized path
            x_emb_per = self.personalized_modalities[modality]["embedding"](x)
            positional_x_per = self.personalized_modalities[modality]["pos_enc"](
                x_emb_per
            )
            attn1_per = self.personalized_modalities[modality]["enc1"](positional_x_per)
            avg_pool_per = torch.mean(attn1_per, 1)
            max_pool_per, _ = torch.max(attn1_per, 1)
            concat_per = torch.cat((avg_pool_per, max_pool_per), 1)
            concat_per_ = self.personalized_modalities[modality]["relu"](
                self.personalized_modalities[modality]["linear"](concat_per)
            )
            concat_per_ = self.personalized_modalities[modality]["dropout_out"](
                concat_per_
            )
            personalized_modality_outputs.append(concat_per_)

        concat_generalized = torch.cat(generalized_modality_outputs, dim=1)
        concat_personalized = torch.cat(personalized_modality_outputs, dim=1)

        concat = torch.cat((concat_generalized, concat_personalized), dim=1)
        concat = self.relu(concat)
        concat = self.dropout_out(concat)

        final_output = self.output_layer(concat)

        return personalized_modality_outputs, final_output
