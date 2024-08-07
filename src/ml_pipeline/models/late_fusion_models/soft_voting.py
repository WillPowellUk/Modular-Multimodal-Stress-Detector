import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ml_pipeline.models.attention_models.attention_mechansims import (
    AttentionPooling,
)


class StackedModularPool(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, dropout, pool_type="avg"):
        super(StackedModularPool, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Define the average pooling layer
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)

        # Define the first linear layer
        self.linear1 = nn.Linear(embed_dim, hidden_dim)

        # Define the second linear layer
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        # Define dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # Merge branches into one tensor
        concatenated_features = torch.cat(list(x.values()), dim=1)
        concatenated_features = concatenated_features.permute(
            0, 2, 1
        )  # change to shape (batch_size, embed_dim, n_branches)

        # Apply average pooling
        x = self.pool(concatenated_features)

        # Remove the last dimension (which is 1 after avg pooling)
        x = x.squeeze(-1)

        # Apply first linear layer
        x = self.linear1(x)

        # Apply dropout
        x = self.dropout_layer(x)

        # Apply activation function
        x = F.relu(x)

        # Apply second linear layer
        x = self.linear2(x)

        return x


class ModularPool(nn.Module):
    def __init__(
        self,
        embed_dim,
        output_dim,
        dropout,
        pool_type="avg",
        return_branch_outputs=False,
    ):
        super(ModularPool, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.pool_type = pool_type
        self.return_branch_outputs = return_branch_outputs

        # Define the pooling layer based on pool_type
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("pool_type must be either 'avg' or 'max'")

        # Define the linear layer
        self.linear = nn.Linear(embed_dim, output_dim)

        # Define dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        branch_outputs = []
        for key in x:
            branch = x[key]
            # Permute to shape (batch_size, embed_dim, seq_len)
            branch = branch.permute(0, 2, 1)

            # Apply pooling
            branch = self.pool(branch)

            # Remove the last dimension (which is 1 after pooling)
            branch = branch.squeeze(-1)

            # Apply linear layer
            branch = self.linear(branch)

            # Apply dropout
            branch = self.dropout_layer(branch)

            # Apply activation function
            branch = F.relu(branch)

            branch_outputs.append(branch)

        # Stack branch outputs and compute the average
        branch_outputs = torch.stack(
            branch_outputs, dim=1
        )  # Shape: (batch_size, num_branches, output_dim)

        if self.return_branch_outputs:
            return branch_outputs

        # Compute the average of branch outputs
        avg_output = torch.mean(
            branch_outputs, dim=1
        )  # Shape: (batch_size, output_dim)

        return avg_output


class ModularWeightedPool(nn.Module):
    def __init__(self, embed_dim, output_dim, dropout, branch_keys, pool_type="avg"):
        super(ModularWeightedPool, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.pool_type = pool_type

        # Define the pooling layer
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pool_type == "attention":
            self.pool = AttentionPooling(embed_dim, source_seq_length=1)
        else:
            raise ValueError("pool_type must be either 'avg', 'max', or 'attention'")

        # Define the linear layer for branch processing
        self.branch_linear = nn.Linear(embed_dim, output_dim)

        # Define dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize learnable weights for each branch
        self.branch_keys = sorted(branch_keys)
        self.gate_weights = nn.ParameterDict(
            {key: nn.Parameter(torch.randn(1)) for key in self.branch_keys}
        )

    def forward(self, x):
        branch_outputs = []
        branch_gates = []

        for key in sorted(x.keys()):
            branch = x[key]
            if self.pool_type == "attention":
                # AttentionPooling expects input of shape [batch_size, source_source_seq_length, embed_dim]
                branch = self.pool(branch)
            else:
                # Permute to shape (batch_size, embed_dim, seq_len)
                branch = branch.permute(0, 2, 1)

                # Apply pooling
                branch = self.pool(branch)

                # Remove the last dimension (which is 1 after pooling)
                branch = branch.squeeze(-1)

            # Apply linear layer
            branch = self.branch_linear(branch)

            # Apply dropout
            branch = self.dropout_layer(branch)

            # Apply activation function
            branch = F.relu(branch)

            branch_outputs.append(branch)
            branch_gates.append(self.gate_weights[key])

        # Convert gate weights to tensor and apply softmax
        gate_weights = torch.cat(branch_gates, dim=0)
        gate_weights = F.softmax(gate_weights, dim=0)

        # Stack branch outputs and apply the weights
        branch_outputs = torch.stack(
            branch_outputs, dim=1
        )  # Shape: (batch_size, n_branches, output_dim)
        weighted_output = (
            branch_outputs * gate_weights.unsqueeze(0).unsqueeze(-1)
        ).sum(
            dim=1
        )  # Shape: (batch_size, output_dim)

        return weighted_output
