import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
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
    def __init__(self, d_model, hidden, ffn_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=ffn_dropout)
        self.linear2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SelfAttentionEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, ffn_dropout):
        super(SelfAttentionEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(ffn_dropout)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, ffn_dropout=ffn_dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(ffn_dropout)

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
    def __init__(self, d_model, ffn_hidden, n_head, ffn_dropout):
        super(CrossAttentionEncoder, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(ffn_dropout)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, ffn_dropout=ffn_dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(ffn_dropout)

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

class SlidingKVQCache(nn.Module):
    """
    Standalone nn.Module containing a kv-cache and optional q-cache to cache past key, queries and values during inference.

    Args:
        max_batch_size (int): maximum batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of num_kv_heads because
            the cache is created after we've expanded the key and value tensors to have the
            same shape as the query tensor. See attention.py for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
        query_cache (bool): whether to cache query tensors
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        query_cache: bool = True
    ) -> None:
        super().__init__()
        cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.query_cache = query_cache
        if self.query_cache:
            self.register_buffer(
                "q_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
            )
        self.current_seq_len = 0
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

    def append_cache(self, key, value, query=None):
        """
        Appends the most recent key and value to the cache. 
        Both tensors are expected with shape [batch_size, num_heads, 1, head_dim].
        The key is appended to the last column. 
        The value is appended to the last row.

        Args:
            key (torch.Tensor): The key tensor to be added.
            value (torch.Tensor): The value tensor to be added.
            query (torch.Tensor, optional): The query tensor to be added, if query_cache is enabled.
        """

        # Remove the oldest kv-cache that corresponds to the oldest token and move the buffer forward.
        # The oldest kv cache is the first column of the key and the first row of the value.
        if self.current_seq_len >= self.max_seq_len:
            # Shift the cache left by one position
            self.k_cache[:, :, :-1, :] = self.k_cache[:, :, 1:, :].clone()
            self.v_cache[:, :, :-1, :] = self.v_cache[:, :, 1:, :].clone()
            if self.query_cache:
                self.q_cache[:, :, :-1, :] = self.q_cache[:, :, 1:, :].clone()
            self.current_seq_len = self.max_seq_len - 1

        batch_size, num_heads, _, head_dim = key.shape
        self.k_cache[:batch_size, :num_heads, self.current_seq_len, :head_dim] = key.squeeze(2)
        self.v_cache[:batch_size, :num_heads, self.current_seq_len, :head_dim] = value.squeeze(2)
        if self.query_cache and query is not None:
            self.q_cache[:batch_size, :num_heads, self.current_seq_len, :head_dim] = query.squeeze(2)
        self.current_seq_len += 1

    def retrieve_cache(self):
        """
        Retrieves the entire kv cache.

        Returns:
            tuple: A tuple containing the key cache, value cache and query cache (if enabled).
        """
        if self.query_cache:
            return self.k_cache, self.v_cache, self.q_cache
        return self.k_cache, self.v_cache


class MultiHeadProjection(nn.Module):
    def __init__(self, embed_dim, head_dim, num_heads):
        super(MultiHeadProjection, self).__init__()
        self.num_heads = num_heads
        
        # Create separate linear layers for each head for query, key, and value
        self.query_layers = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=False) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=False) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(embed_dim, head_dim, bias=False) for _ in range(num_heads)])

    def forward(self, query, key, value):
        '''
        The query, key and values are linearly transformed into distinct matrices for each head.
        Expects query, key, value to be each of shape [batch_size, seq_length, embed_dim] and returns
        query, key, value of shape [batch_size, num_heads, seq_length, head_dim]
        '''
        # Apply each linear layer to the corresponding head
        query_heads = [layer(query) for layer in self.query_layers]
        key_heads = [layer(key) for layer in self.key_layers]
        value_heads = [layer(value) for layer in self.value_layers]
        
        # Stack the results to create a tensor of shape [batch_size, num_heads, seq_length, head_dim]
        query = torch.stack(query_heads, dim=1)
        key = torch.stack(key_heads, dim=1)
        value = torch.stack(value_heads, dim=1)

        return query, key, value
    
class CachedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_batch_size, max_seq_len, dropout=0.1, query_cache=True):
        super(CachedMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.max_seq_len = max_seq_len
        self.scale = self.head_dim ** -0.5

        # Input projection that projects each key, value and query for each number of heads
        self.in_proj = MultiHeadProjection(embed_dim, self.head_dim, num_heads)
        
        # Output projection weight once single headed attention blocks are concatenated
        self.out_proj = nn.Linear(self.head_dim * num_heads, self.embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # KV cache
        self.kv_cache = SlidingKVQCache(max_batch_size, max_seq_len, num_heads, self.head_dim, torch.float32, query_cache=query_cache)

    def scaled_dot_product_attention(self, query, key, value):
        attn_scores = query @ key.transpose(-2,-1) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = attn_weights @ value 
        
        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, embed_dim]
        output = output.transpose(1, 2).contiguous().view(output.size(0), output.size(2), -1)

        return output

    def forward(self, query, key, value, use_cache=True):
        '''
        Performs multi-head attention with caching mechanism using scaled dot product attention.
        Expects query, key, value to each be of shape [batch_size, seq_length, embed_dim]
        '''
        
        # project new query, key and value for each head 
        # this will return tensors each of shape [batch_size, num_heads, seq_length, head_dim]
        query, key, value = self.in_proj(query, key, value)

        if use_cache:
            # Cache latest keys, values, (and queries if enabled) which correspond to latest embedding in the sequence
            if self.kv_cache.query_cache:
                self.kv_cache.append_cache(key, value, query)
                cached_keys, cached_values, cached_queries = self.kv_cache.retrieve_cache()
            else:
                self.kv_cache.append_cache(key, value)
                cached_keys, cached_values = self.kv_cache.retrieve_cache()
        
        else:
            cached_keys, cached_values, cached_queries = key, value, None

        # Perform scaled dot product attention for all attention heads (multi-headed attention)
        if self.kv_cache.query_cache and cached_values is not None:
            out = self.scaled_dot_product_attention(cached_queries, cached_keys, cached_values)
        else:
            out = self.scaled_dot_product_attention(query, cached_keys, cached_values)

        out = self.dropout(out)
        return out

class CachedSlidingAttentionEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, max_batch_size, max_seq_len, ffn_dropout=0.1, attention_dropout=0.1, query_cache=True):
        super(CachedSlidingAttentionEncoder, self).__init__()
        self.attention = CachedMultiHeadAttention(
            d_model, n_head, max_batch_size, max_seq_len, dropout=attention_dropout, query_cache=query_cache
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(ffn_dropout)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, ffn_dropout=ffn_dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(ffn_dropout)

    def forward(self, query, key, value, seq_length, use_cache=True):
        # x is of shape [batch_size, seq_length, features/embedding]

        # Attention mechanism which will remove the oldest token if cache exceeds seq_length if use_cache is True
        x = self.attention(query=query, key=key, value=value, use_cache=use_cache)

        # Add and norm
        x = x + self.dropout1(self.norm1(x))
        
        # Feedforward
        # Add and norm
        x = x + self.dropout1(self.norm1(x))
        
        # Feedforward
        x = self.ffn(x)
        x = x + self.dropout2(self.norm2(x))

        x = x + self.dropout2(self.norm2(x))

        return x


class CachedSlidingCrossAttentionEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, ffn_dropout, attention_dropout=0.1):
        super(CachedSlidingCrossAttentionEncoder, self).__init__()
        self.attention = MultiheadAttention(
            d_model, n_head, batch_first=True, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(ffn_dropout)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, ffn_dropout=ffn_dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(ffn_dropout)

        # Initialize KV cache
        self.kv_cache = None
        self.query_cache = None

    def forward(self, query, key_value, seq_length, use_cache=False):
        if use_cache and self.kv_cache is not None:
            key_cache, value_cache = self.kv_cache
            query_cache = self.query_cache
        else:
            key_cache, value_cache = None, None
            query_cache = None

        # Concatenate new tokens with cached tokens and apply windowing
        if key_cache is not None and value_cache is not None:
            keys = torch.cat([key_cache, key_value], dim=1)
            values = torch.cat([value_cache, key_value], dim=1)
            queries = torch.cat([query_cache, query], dim=1)

            # Maintain the sliding window of cache
            if keys.size(1) > seq_length:
                keys = keys[:, -seq_length:, :]
                values = values[:, -seq_length:, :]
                queries = queries[:, -seq_length:, :]
            if keys.size(1) > seq_length:
                keys = keys[:, -seq_length:, :]
                values = values[:, -seq_length:, :]
                queries = queries[:, -seq_length:, :]

        else:
            keys, values = key_value, key_value
            queries = query

        # Perform cross-attention over all tokens (cached + new)
        x, attn_output_weights = self.attention(queries, keys, values)

        # Update cache
        if use_cache:
            self.kv_cache = (keys.detach(), values.detach())
            self.query_cache = queries.detach()

        # Use the relevant part of the output (last `L` tokens)
        x = x[:, -queries.size(1):, :]

        x = self.norm1(x + queries)
        x = self.dropout1(x)

        # Feed-forward layer
        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

    def clear_cache(self):
        self.kv_cache = None
        self.query_cache = None
