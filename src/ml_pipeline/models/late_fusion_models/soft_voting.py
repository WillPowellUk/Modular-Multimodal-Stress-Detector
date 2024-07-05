import torch
import torch.nn as nn
import torch.nn.functional as F

class OG(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, dropout):
        super(ModularAvgPool, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Define the average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Define the first linear layer
        self.linear1 = nn.Linear(embed_dim, hidden_dim)

        # Define the second linear layer
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        # Define dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # Expects the shape (batch_size, embed_dim, n_branches)

        # Apply average pooling
        x = self.avg_pool(x)

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

class ModularAvgPool(nn.Module):
    def __init__(self, embed_dim, output_dim, dropout):
        super(ModularAvgPool, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Define the average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Define the linear layer
        self.linear = nn.Linear(embed_dim, output_dim)

        # Define dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Define a parameter for the weights
        self.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        branch_outputs = []
        for key in x:
            branch = x[key]
            # Permute to shape (batch_size, embed_dim, seq_len)
            branch = branch.permute(0, 2, 1)
            
            # Apply average pooling
            branch = self.avg_pool(branch)
            
            # Remove the last dimension (which is 1 after avg pooling)
            branch = branch.squeeze(-1)
            
            # Apply linear layer
            branch = self.linear(branch)
            
            # Apply dropout
            branch = self.dropout_layer(branch)
            
            # Apply activation function
            branch = F.relu(branch)
            
            branch_outputs.append(branch)

        # Stack branch outputs and compute weights
        branch_outputs = torch.stack(branch_outputs, dim=1)  # Shape: (batch_size, num_branches, output_dim)
        
        # Compute weights for each branch
        weights = F.softmax(self.weight_scale.expand(branch_outputs.size(1)), dim=0)  # Shape: (num_branches,)

        # Weighted average of branch outputs
        weighted_avg_output = torch.sum(branch_outputs * weights.unsqueeze(0).unsqueeze(2), dim=1)  # Shape: (batch_size, output_dim)

        return weighted_avg_output


class ModularWeightedAvgPool(nn.Module):
    def __init__(self, embed_dim, output_dim, dropout, branch_keys):
        super(ModularWeightedAvgPool, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Define the average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Define the linear layer for branch processing
        self.branch_linear = nn.Linear(embed_dim, output_dim)

        # Define dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize learnable weights for each branch
        self.branch_keys = sorted(branch_keys)
        self.gate_weights = nn.ParameterDict({key: nn.Parameter(torch.randn(1)) for key in self.branch_keys})

    def forward(self, x):
        branch_outputs = []
        branch_gates = []

        for key in sorted(x.keys()):
            branch = x[key]
            # Permute to shape (batch_size, embed_dim, seq_len)
            branch = branch.permute(0, 2, 1)

            # Apply average pooling
            branch = self.avg_pool(branch)

            # Remove the last dimension (which is 1 after avg pooling)
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
        branch_outputs = torch.stack(branch_outputs, dim=1)  # Shape: (batch_size, n_branches, output_dim)
        weighted_output = (branch_outputs * gate_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # Shape: (batch_size, output_dim)

        return weighted_output
