import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class ModularHardVoting(nn.Module):
    def __init__(self, embed_dim, output_dim, dropout, branch_keys, pool_type="avg"):
        super(ModularHardVoting, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.pool_type = pool_type

        # Define the pooling layer
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError("pool_type must be either 'avg' or 'max'")

        # Define the linear layer for branch processing
        self.branch_linear = nn.Linear(embed_dim, output_dim)

        # Define dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize branches keys
        self.branch_keys = sorted(branch_keys)

    def forward(self, x):
        branch_outputs = []

        for key in sorted(x.keys()):
            branch = x[key]
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

            # Apply softmax to get probabilities
            branch = F.softmax(branch, dim=1)

            branch_outputs.append(branch)

        # Stack branch outputs
        branch_outputs = torch.stack(
            branch_outputs, dim=1
        )  # Shape: (batch_size, n_branches, output_dim)

        # Perform hard voting
        batch_size = branch_outputs.size(0)
        final_probabilities = torch.zeros((batch_size, self.output_dim)).to(
            x[self.branch_keys[0]].device
        )

        for i in range(batch_size):
            # Get the predicted class for each branch
            predictions = torch.argmax(branch_outputs[i], dim=1)

            # Count the votes for each class
            vote_counts = Counter(predictions.tolist())

            # Normalize the vote counts to get probabilities
            for class_idx, vote_count in vote_counts.items():
                final_probabilities[i, class_idx] = vote_count / len(self.branch_keys)

        return final_probabilities
