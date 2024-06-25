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

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
#         super(EncoderLayer, self).__init__()
#         self.attention = nn.MultiheadAttention(d_model, n_head)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(drop_prob)
#         self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(drop_prob)

#     def forward(self, x):
#         _x = x
#         x, _ = self.attention(x, x, x)
#         x = self.norm1(x + _x)
#         x = self.dropout1(x)

#         _x = x
#         x = self.ffn(x)
#         x = self.norm2(x + _x)
#         x = self.dropout2(x)
#         return x

# class CrossAttentionBlock(nn.Module):
#     def __init__(self, embed_dim, n_head, dropout):
#         super(CrossAttentionBlock, self).__init__()
#         self.cross_attn = nn.MultiheadAttention(embed_dim, n_head, dropout)
#         self.norm = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, query, key, value):
#         attn_output, _ = self.cross_attn(query, key, value)
#         attn_output = self.dropout(attn_output)
#         return self.norm(attn_output + query)

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

import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, max_segments):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        self.max_segments = max_segments
        self.cache = {'keys': [], 'queries': [], 'values': []}

    def forward(self, x, use_cache=True):
        # Compute key, query, and value for the new segment
        new_key, new_query, new_value = self.attention.in_proj_q(x), self.attention.in_proj_k(x), self.attention.in_proj_v(x)
        
        if use_cache:
            # Append new tensors to cache
            self.cache['keys'].append(new_key)
            self.cache['queries'].append(new_query)
            self.cache['values'].append(new_value)
            # Ensure cache size does not exceed the maximum number of segments
            if len(self.cache['keys']) > self.max_segments:
                self.cache['keys'].pop(0)
                self.cache['queries'].pop(0)
                self.cache['values'].pop(0)
        else:
            # Reset cache with new tensors
            self.cache = {'keys': [new_key], 'queries': [new_query], 'values': [new_value]}

        # Concatenate cached keys, queries, values
        keys = torch.cat(self.cache['keys'], dim=1)
        queries = torch.cat(self.cache['queries'], dim=1)
        values = torch.cat(self.cache['values'], dim=1)

        # Compute self-attention using cached and new tensors
        _x = x
        x, _ = self.attention(queries, keys, values)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # Feed forward
        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_head, dropout, max_segments):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_head, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_segments = max_segments
        self.cache = {'keys': [], 'queries': [], 'values': []}

    def forward(self, query, key, value, use_cache=True):
        # Compute key, query, and value for the new segment
        new_key, new_query, new_value = self.cross_attn.in_proj_k(key), self.cross_attn.in_proj_q(query), self.cross_attn.in_proj_v(value)

        if use_cache:
            # Append new tensors to cache
            self.cache['keys'].append(new_key)
            self.cache['queries'].append(new_query)
            self.cache['values'].append(new_value)
            # Ensure cache size does not exceed the maximum number of segments
            if len(self.cache['keys']) > self.max_segments:
                self.cache['keys'].pop(0)
                self.cache['queries'].pop(0)
                self.cache['values'].pop(0)
        else:
            # Reset cache with new tensors
            self.cache = {'keys': [new_key], 'queries': [new_query], 'values': [new_value]}

        # Concatenate cached keys, queries, values
        keys = torch.cat(self.cache['keys'], dim=1)
        queries = torch.cat(self.cache['queries'], dim=1)
        values = torch.cat(self.cache['values'], dim=1)

        # Compute cross-attention using cached and new tensors
        attn_output, _ = self.cross_attn(queries, keys, values)
        attn_output = self.dropout(attn_output)
        return self.norm(attn_output + query)
