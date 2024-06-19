import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# model_weights = torch.load('saved_model/pretrain/SA-AE/checkpoint_50.pth')
# print(model_weights['embedding_enc.weight_ih_l0'])
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

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.enc_dec_attention = nn.MultiheadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc):
        # 1. compute self attention
        _x = dec
        x, _ = self.self_attention(dec, dec, dec)
        
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x, _ = self.enc_dec_attention(x, enc, enc)
            
            # 4. add and norm
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.norm3(x + _x)
        x = self.dropout3(x)
        return x

class SelfAttenModel(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, device, sequence_length, dropout=0.1):
        super(SelfAttenModel, self).__init__()
        # self.pos_enc = PostionalEncoding(input_dim, max_len = sequence_length, device = device)
        # self.embedding = nn.LSTM(
        #                 input_size = input_dim,
        #                 hidden_size = embed_dim,
        #                 num_layers = 1,
        #                 batch_first = True,
        #                 )
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.enc1 = EncoderLayer(embed_dim, ffn_hidden=128, n_head=4, drop_prob=dropout)
        # self.enc2 = EncoderLayer(embed_dim, ffn_hidden=128, n_head=8, drop_prob=dropout)
        # self.enc3 = EncoderLayer(embed_dim, ffn_hidden=128, n_head=8, drop_prob=dropout)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(embed_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_out = nn.Dropout(p=dropout)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x_emb, _ = self.embedding(x)
        # x_emb = self.embedding(x)
        positional_x = self.pos_enc(x_emb)
        
        attn1 = self.enc1(positional_x)
        # attn2 = self.enc2(attn1)
        # attn3 = self.enc3(attn2)
        
        avg_pool = torch.mean(attn1, 1)
        max_pool, _ = torch.max(attn1, 1)
        
        concat = torch.cat((avg_pool, max_pool), 1)
        concat = self.relu(self.linear(concat))
        concat = self.dropout_out(concat)
        
        output = self.output(concat)
        
        return output
    
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

class SelfAttentionAE(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout=0.1):
        super(SelfAttentionAE, self).__init__()
        self.embedding_enc = nn.LSTM(
                        input_size = input_dim,
                        hidden_size = embed_dim,
                        num_layers = 1,
                        batch_first = True,
                        )
        self.pos_enc = PositionalEncoding(embed_dim)
        self.encoder = nn.ModuleList(EncoderLayer(embed_dim, ffn_hidden=128, n_head=8, drop_prob=dropout) \
                        for _ in range(1))
        
        self.embedding_dec = nn.LSTM(
                        input_size = input_dim,
                        hidden_size = embed_dim,
                        num_layers = 1,
                        batch_first = True,
                        )
        self.pos_dec = PositionalEncoding(embed_dim)
        self.decoder = nn.ModuleList(DecoderLayer(embed_dim, ffn_hidden=128, n_head=8, drop_prob=dropout) \
                        for _ in range(1))
        self.linear_dec = nn.Linear(embed_dim, input_dim)
    
    def forward(self, x, target):
        # LSTM embedding & positional encoding
        x_emb, _ = self.embedding_enc(x)
        x = self.pos_enc(x_emb)
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            
        target, _ = self.embedding_dec(target)
        target = self.pos_enc(target)
        for dec_layer in self.decoder:
            target = dec_layer(target, x)
        output = self.linear_dec(target)
        return output
        
class ModalityFusionNet(torch.nn.Module):
    def __init__(self, input_dims, embed_dim, hidden_dim, output_dim, dropout=0.1):
        super(ModalityFusionNet, self).__init__()
        
        self.modalities = nn.ModuleDict()
        
        for modality in input_dims:
            modality_net = nn.ModuleDict({
                'embedding': nn.Linear(input_dims[modality], embed_dim),
                'pos_enc': PositionalEncoding(embed_dim),
                'enc1': EncoderLayer(embed_dim, ffn_hidden=128, n_head=4, drop_prob=dropout),
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

class LSTM_Model(torch.nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        
        self.lstm0 = nn.LSTM(
            input_size = 24,
            hidden_size = 64,
            num_layers = 3,
            batch_first = True,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, 256)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm0(x, None)
        bn_out = F.dropout(self.bn1(lstm_out[:, -1, :]), 0.3)
        fc_out = F.relu(F.dropout(self.fc(bn_out), 0.5))
        out = self.out(fc_out)
        return out
        