import torch
import torch.nn as nn
from src.ml_pipeline.models.attention_models.attention_mechansims import PositionalEncoding, BCSAMechanism, EncoderLayer

class ModularBCSA(nn.Module):
    NAME = "ModularBCSA"
    
    def __init__(self, **kwargs):
        required_params = ['input_dims', 'embed_dim', 'hidden_dim', 'output_dim', 'n_head_gen', 'dropout', 'n_bcsa']
        
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f'Missing required parameter: {param}')
        
        self.input_dims = kwargs['input_dims']
        self.embed_dim = kwargs['embed_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.output_dim = kwargs['output_dim']
        self.n_head = kwargs['n_head_gen']
        self.dropout = kwargs['dropout']
        self.n_bcsa = kwargs['n_bcsa']

        super(ModularBCSA, self).__init__()
        
        self.modalities = nn.ModuleDict()
        
        for modality in self.input_dims:
            modality_net = nn.ModuleDict({
                'embedding': nn.Linear(self.input_dims[modality], self.embed_dim),
                'pos_enc': PositionalEncoding(self.embed_dim),
                'bcsas': nn.ModuleList([BCSAMechanism(self.embed_dim, self.n_head, self.dropout) for _ in range(self.n_bcsa)]),
                'enc1': EncoderLayer(self.embed_dim, ffn_hidden=128, n_head=self.n_head, drop_prob=self.dropout),
                'flatten': nn.Flatten(),
                'linear': nn.Linear(self.embed_dim * 2, self.hidden_dim),
                'relu': nn.ReLU(),
                'dropout_out': nn.Dropout(p=self.dropout),
                'output': nn.Linear(self.hidden_dim, self.output_dim)
            })
            self.modalities[modality] = modality_net
        
        self.relu = nn.ReLU()
        self.dropout_out = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(len(self.input_dims) * self.hidden_dim, self.output_dim)

    def forward(self, inputs):
        modality_outputs = []
        
        for modality, x in inputs.items():
            batch_size, seq_len, features = x.shape
            x = x.permute(0, 2, 1)  # Change shape to [batch_size, features, seq_len]
            x = x.reshape(-1, seq_len)  # Flatten to [batch_size * features, seq_len]
            x_emb = self.modalities[modality]['embedding'](x)
            x_emb = x_emb.view(batch_size, features, -1)  # Reshape back to [batch_size, features, embed_dim]
            positional_x = self.modalities[modality]['pos_enc'](x_emb)
            
            # Apply multiple BCSA blocks in series
            for bcsa in self.modalities[modality]['bcsas']:
                positional_x, _ = bcsa(positional_x, positional_x)
                
            attn1 = self.modalities[modality]['enc1'](positional_x)
            avg_pool = torch.mean(attn1, 1)
            max_pool, _ = torch.max(attn1, 1)
            concat = torch.cat((avg_pool, max_pool), 1)
            concat_ = self.modalities[modality]['relu'](self.modalities[modality]['linear'](concat))
            concat_ = self.modalities[modality]['dropout_out'](concat_)
            modality_output = self.modalities[modality]['output'](concat_)
            modality_outputs.append(concat_)
        
        concat = torch.cat(modality_outputs, dim=1)
        concat = self.relu(concat)
        concat = self.dropout_out(concat)
        
        final_output = self.output_layer(concat)
        return modality_outputs, final_output