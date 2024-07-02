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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SelfAttentionEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(SelfAttentionEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
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


class CrossAttentionEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(CrossAttentionEncoder, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, context):
        # Cross-attention
        _x = x
        x, _ = self.cross_attention(x, context, context)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # Feed-forward
        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x


class CachedSlidngSelfAttentionEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(CachedSlidngSelfAttentionEncoder, self).__init__()
        self.attention = MultiheadAttention(
            d_model, n_head, batch_first=True, dropout=drop_prob
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        # Initialize KV cache
        self.kv_cache = None

    def forward(self, x, token_length, use_cache=False):
        if use_cache and self.kv_cache is not None:
            key_cache, value_cache = self.kv_cache
        else:
            key_cache, value_cache = None, None

        # Attention layer
        _x = x
        if key_cache is not None and value_cache is not None:
            keys = torch.cat([key_cache, x], dim=1)
            values = torch.cat([value_cache, x], dim=1)

            # Maintain the sliding window of cache
            if keys.size(1) > token_length:
                keys = keys[:, -token_length:, :]
                values = values[:, -token_length:, :]
        else:
            keys, values = x, x

        x, _ = self.attention(x, keys, values)
        # x = self.attention(x, keys, values, average_attn_weights=False)

        # Update cache
        if use_cache:
            self.kv_cache = (keys.detach(), values.detach())

        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # Feed-forward layer
        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

    def clear_cache(self):
        self.kv_cache = None


class CachedSlidingCrossAttentionEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(CachedSlidingCrossAttentionEncoder, self).__init__()
        self.attention = MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        # Initialize KV cache
        self.kv_cache = None

    def forward(self, query, key_value, token_length, use_cache=False):
        if use_cache and self.kv_cache is not None:
            key_cache, value_cache = self.kv_cache
        else:
            key_cache, value_cache = None, None

        # Cross-attention layer
        _query = query
        if key_cache is not None and value_cache is not None:
            keys = torch.cat([key_cache, key_value], dim=1)
            values = torch.cat([value_cache, key_value], dim=1)

            # Maintain the sliding window of cache
            if keys.size(1) > token_length:
                keys = keys[:, -token_length:, :]
                values = values[:, -token_length:, :]
        else:
            keys, values = key_value, key_value

        query, _ = self.attention(query, keys, values)

        # Update cache
        if use_cache:
            self.kv_cache = (keys.detach(), values.detach())

        query = self.norm1(query + _query)
        query = self.dropout1(query)

        # Feed-forward layer
        _query = query
        query = self.ffn(query)
        query = self.norm2(query + _query)
        query = self.dropout2(query)
        return query

    def clear_cache(self):
        self.kv_cache = None
