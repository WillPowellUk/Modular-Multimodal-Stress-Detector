import torch
import torch.nn as nn
import torch.nn.functional as F

class ModularAvgPool(nn.Module):
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
        # Assuming x is of shape (batch_size, variable_length, embed_dim)
        # Transpose to (batch_size, embed_dim, variable_length) for avg pooling
        x = x.transpose(1, 2)
        
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
