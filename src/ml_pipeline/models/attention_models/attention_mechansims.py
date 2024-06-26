import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
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
    
class CachedEncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, token_length):
        super(CachedEncoderLayer, self).__init__()
        self.attention = CachedMultiheadAttention(d_model, n_head, token_length)
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
    
class CachedCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_head, dropout, token_length):
        super(CachedCrossAttentionBlock, self).__init__()
        self.cross_attn = CachedMultiheadAttention(embed_dim, n_head, token_length)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        return self.norm(attn_output + query)

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_head, dropout):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_head, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        return self.norm(attn_output + query)
    
class BidirectionalCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_head, dropout):
        super(BidirectionalCrossAttentionBlock, self).__init__()
        self.cross_attn_A_to_B = nn.MultiheadAttention(embed_dim, n_head, dropout)
        self.cross_attn_B_to_A = nn.MultiheadAttention(embed_dim, n_head, dropout)
        self.norm_A = nn.LayerNorm(embed_dim)
        self.norm_B = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, A, B):
        # Cross attention A to B
        attn_output_A_to_B, _ = self.cross_attn_A_to_B(A, B, B)
        attn_output_A_to_B = self.dropout(attn_output_A_to_B)
        output_A = self.norm_A(attn_output_A_to_B + A)
        
        # Cross attention B to A
        attn_output_B_to_A, _ = self.cross_attn_B_to_A(B, A, A)
        attn_output_B_to_A = self.dropout(attn_output_B_to_A)
        output_B = self.norm_B(attn_output_B_to_A + B)
        
        return output_A, output_B

class CachedMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, token_length, batch_first=True, **kwargs):
        super(CachedMultiheadAttention, self).__init__(embed_dim, num_heads, batch_first=batch_first, **kwargs)
        self.token_length = token_length
        self.batch_first = batch_first
        self.key_cache = None
        self.value_cache = None
        self.query_cache = None
        self.cache_size = 0

    def update_cache(self, query, key, value):
        query = query.detach()
        key = key.detach()
        value = value.detach()
        
        if self.key_cache is None:
            self.key_cache = key
            self.value_cache = value
            self.query_cache = query
        else:
            # Append new keys, values, and queries
            self.key_cache = torch.cat((self.key_cache, key), dim=0).detach()
            self.value_cache = torch.cat((self.value_cache, value), dim=0).detach()
            self.query_cache = torch.cat((self.query_cache, query), dim=0).detach()
            
            # Remove the oldest key, value, and query if necessary
            if self.key_cache.size(0) > self.token_length:
                self.key_cache = self.key_cache[1:].detach()
                self.value_cache = self.value_cache[1:].detach()
                self.query_cache = self.query_cache[1:].detach()

        self.cache_size = self.key_cache.size(0)
        return True

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Update the cache with the latest query, key, and value and only proceed if the cache buffer is full
        if (self.update_cache(query, key, value)):

            # Use the cached keys, values, and queries
            q = self.query_cache
            k = self.key_cache
            v = self.value_cache

            scaling = float(self.head_dim) ** -0.5
            q = q * scaling

            attn_output, attn_output_weights = self._scaled_dot_product_attention(q, k, v, attn_mask, key_padding_mask)

            attn_output = attn_output.transpose(0, 1).contiguous().view(-1, self.embed_dim)
            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(query.size(1), -1, self.embed_dim).transpose(0, 1)

            if self.batch_first:
                attn_output = attn_output.transpose(0, 1)

            if need_weights:
                return attn_output, attn_output_weights
            else:
                return attn_output, None

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(-1, self.num_heads, self.cache_size, self.cache_size)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(-1, self.cache_size, self.cache_size)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        return attn_output, attn_weights
