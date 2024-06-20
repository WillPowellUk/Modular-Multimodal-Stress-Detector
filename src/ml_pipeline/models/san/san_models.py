import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x):
        _x = x
        x, _ = self.attention(x, x, x)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

class ModularModalityFusionNet(torch.nn.Module):
    NAME = "ModularModalityFusionNet"
    def __init__(self, input_dims, embed_dim, hidden_dim, output_dim, n_head=4, dropout=0.1):
        super(ModularModalityFusionNet, self).__init__()
        
        self.modalities = nn.ModuleDict()
        
        for modality in input_dims:
            modality_net = nn.ModuleDict({
                'embedding': nn.Linear(input_dims[modality], embed_dim),
                'pos_enc': PositionalEncoding(embed_dim),
                'enc1': EncoderLayer(embed_dim, ffn_hidden=128, n_head=n_head, drop_prob=dropout),
                'flatten': nn.Flatten(),
                'linear': nn.Linear(embed_dim * 2, hidden_dim),
                'relu': nn.ReLU(),
                'dropout_out': nn.Dropout(p=dropout),
                'output': nn.Linear(hidden_dim, output_dim)
            })
            self.modalities[modality] = modality_net
        
        self.relu = nn.ReLU()
        self.dropout_out = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(len(input_dims) * hidden_dim, output_dim)

    def forward(self, inputs):
        modality_outputs = []
        
        for modality, x in inputs.items():
            batch_size, seq_len, features = x.shape
            x = x.permute(0, 2, 1)  # Change shape to [batch_size, features, seq_len]
            x = x.reshape(-1, seq_len)  # Flatten to [batch_size * features, seq_len]
            x_emb = self.modalities[modality]['embedding'](x)
            x_emb = x_emb.view(batch_size, features, -1)  # Reshape back to [batch_size, features, embed_dim]
            positional_x = self.modalities[modality]['pos_enc'](x_emb)
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

class PersonalizedModalityFusionNet(torch.nn.Module):
    NAME = "PersonalizedModalityFusionNet"
    
    def __init__(self, generalized_model, n_head=1, dropout=0.1):
        super(PersonalizedModalityFusionNet, self).__init__()
        
        self.generalized_modalities = generalized_model.modalities
        
        # Freeze generalized model parameters
        for param in self.generalized_modalities.parameters():
            param.requires_grad = False
        
        self.personalized_modalities = nn.ModuleDict()
        
        # Infer dimensions from generalized model
        input_dims = {modality: mod['embedding'].in_features for modality, mod in self.generalized_modalities.items()}
        embed_dim = next(iter(self.generalized_modalities.values()))['embedding'].out_features
        hidden_dim = next(iter(self.generalized_modalities.values()))['linear'].out_features
        output_dim = next(iter(self.generalized_modalities.values()))['output'].out_features
        
        for modality in input_dims:
            modality_net = nn.ModuleDict({
                'embedding': nn.Linear(input_dims[modality], embed_dim),
                'pos_enc': PositionalEncoding(embed_dim),
                'enc1': EncoderLayer(embed_dim, ffn_hidden=128, n_head=n_head, drop_prob=dropout),
                'flatten': nn.Flatten(),
                'linear': nn.Linear(embed_dim * 2, hidden_dim),
                'relu': nn.ReLU(),
                'dropout_out': nn.Dropout(p=dropout),
                'output': nn.Linear(hidden_dim, output_dim)
            })
            self.personalized_modalities[modality] = modality_net
        
        self.relu = nn.ReLU()
        self.dropout_out = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(len(input_dims) * hidden_dim * 2, output_dim)  # Adjusted for concatenation of both generalized and personalized outputs

    def forward(self, inputs):
        generalized_modality_outputs = []
        personalized_modality_outputs = []
        
        for modality, x in inputs.items():
            batch_size, seq_len, features = x.shape
            x = x.permute(0, 2, 1)  # Change shape to [batch_size, features, seq_len]
            x = x.reshape(-1, seq_len)  # Flatten to [batch_size * features, seq_len]

            # Generalized path
            x_emb_gen = self.generalized_modalities[modality]['embedding'](x)
            x_emb_gen = x_emb_gen.view(batch_size, features, -1)  # Reshape back to [batch_size, features, embed_dim]
            positional_x_gen = self.generalized_modalities[modality]['pos_enc'](x_emb_gen)
            attn1_gen = self.generalized_modalities[modality]['enc1'](positional_x_gen)
            avg_pool_gen = torch.mean(attn1_gen, 1)
            max_pool_gen, _ = torch.max(attn1_gen, 1)
            concat_gen = torch.cat((avg_pool_gen, max_pool_gen), 1)
            concat_gen_ = self.generalized_modalities[modality]['relu'](self.generalized_modalities[modality]['linear'](concat_gen))
            concat_gen_ = self.generalized_modalities[modality]['dropout_out'](concat_gen_)
            generalized_modality_outputs.append(concat_gen_)
            
            # Personalized path
            x_emb_per = self.personalized_modalities[modality]['embedding'](x)
            x_emb_per = x_emb_per.view(batch_size, features, -1)  # Reshape back to [batch_size, features, embed_dim]
            positional_x_per = self.personalized_modalities[modality]['pos_enc'](x_emb_per)
            attn1_per = self.personalized_modalities[modality]['enc1'](positional_x_per)
            avg_pool_per = torch.mean(attn1_per, 1)
            max_pool_per, _ = torch.max(attn1_per, 1)
            concat_per = torch.cat((avg_pool_per, max_pool_per), 1)
            concat_per_ = self.personalized_modalities[modality]['relu'](self.personalized_modalities[modality]['linear'](concat_per))
            concat_per_ = self.personalized_modalities[modality]['dropout_out'](concat_per_)
            personalized_modality_outputs.append(concat_per_)
        
        concat_generalized = torch.cat(generalized_modality_outputs, dim=1)
        concat_personalized = torch.cat(personalized_modality_outputs, dim=1)
        
        concat = torch.cat((concat_generalized, concat_personalized), dim=1)
        concat = self.relu(concat)
        concat = self.dropout_out(concat)
        
        final_output = self.output_layer(concat)
        
        return personalized_modality_outputs, final_output
