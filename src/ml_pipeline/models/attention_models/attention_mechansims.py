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

class SlidingAttnScoreCache(nn.Module):
    """
    Standalone nn.Module containing a sliding projection and attention score cache.

    Args:
        max_batch_size (int): maximum batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of num_kv_heads because
            the cache is created after we've expanded the key and value tensors to have the
            same shape as the query tensor. See attention.py for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        proj_cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        attn_score_cache_shape = (max_batch_size, num_heads, max_seq_len, max_seq_len)
        self.register_buffer(
            "attn_score_cache", torch.zeros(attn_score_cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "q_cache", torch.zeros(proj_cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "k_cache", torch.zeros(proj_cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(proj_cache_shape, dtype=dtype), persistent=False
        )
        self.current_seq_len = 0
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

    def increment_current_seq_len(self):
        if self.current_seq_len < self.max_seq_len - 1:
            self.current_seq_len += 1

    def append_projections(self, q, k, v):
        """
        Appends the most recent query key and value projections to the cache. 
        The tensors are expected to be of shape [batch_size, num_heads, 1, head_dim].
        Args:
            query (torch.Tensor): The query tensor to be added.
            key (torch.Tensor): The key tensor to be added.
            value (torch.Tensor): The value tensor to be added.
        """
        # Remove the oldest kv-cache that corresponds to the oldest token and move the buffer forward.
        if self.current_seq_len >= self.max_seq_len - 1:
            # Shift the cache left by one position
            self.q_cache[:, :, :-1, :] = self.q_cache[:, :, 1:, :].clone()
            self.k_cache[:, :, :-1, :] = self.k_cache[:, :, 1:, :].clone()
            self.v_cache[:, :, :-1, :] = self.v_cache[:, :, 1:, :].clone()
            self.attn_score_cache[:, :, :-1, :] = self.attn_score_cache[:, :, 1:, :].clone()
            self.attn_score_cache[:, :, :, :-1] = self.attn_score_cache[:, :, :, 1:].clone()
            self.current_seq_len = self.max_seq_len - 1

        # Append the new scaled query-key pair and value to the cache
        self.q_cache[:, :, self.current_seq_len, :] = q[:, :, 0, :]
        self.k_cache[:, :, self.current_seq_len, :] = k[:, :, 0, :]
        self.v_cache[:, :, self.current_seq_len, :] = v[:, :, 0, :]

    def update_attn_score(self, q_t, k_t):
            """
            Appends the most recent scaled query (column) and key (row) pair of the attention score to the cache. 
            Args:
                q_t (torch.Tensor): The query tensor to be added to the last column of size [batch_size, num_heads, 1, seq_length]
                k_t (torch.Tensor): The key tensor to be added to the last row of size [batch_size, num_heads, seq_length, 1]
            """
            if self.current_seq_len >= self.max_seq_len -1:
                # If the sequence length exceeds the maximum, shift the cache left by one position in both dimensions
                self.attn_score_cache[:, :, :-1, :] = self.attn_score_cache[:, :, 1:, :].clone()
                self.attn_score_cache[:, :, :, :-1] = self.attn_score_cache[:, :, :, 1:].clone()

            # Add the new column and row to the attention score cache
            self.attn_score_cache[:, :, self.current_seq_len, :] = q_t[:, :, 0, :]
            self.attn_score_cache[:, :, :, self.current_seq_len] = k_t[:, :, :, 0]

    def retrieve_projections(self):
        """
        Retrieves the projections of the stored embeddings q, k, v.

        Returns:
            tuple: A tuple containing the query-key cache, and value cache
        """
        return self.q_cache, self.k_cache, self.v_cache

    def retrieve_attn_score(self):
        """
        Retrieves the entire QK (attention score) cache

        Returns:
            torch.Tensor: The attention score cache
        """
        return self.attn_score_cache
    
    def clear_cache(self):
        """
        Clears the cache.
        """
        self.attn_score_cache.zero_()
        self.q_cache.zero_()
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_seq_len = 0

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, seq_length=1):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(embed_dim, 1)
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: [batch_size, max_seq_length, embed_dim]
        attention_scores = self.attention(x)  # [batch_size, max_seq_length, 1]
        
        weights = torch.softmax(attention_scores, dim=1)  # [batch_size, max_seq_length, 1]
        pooled = torch.sum(weights * x, dim=1)  # [batch_size, embed_dim]
        
        # Expand to [batch_size, seq_length, embed_dim]
        return pooled.unsqueeze(1).expand(-1, self.seq_length, -1)

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
            
        # Output projection weights for downsampling the context window during caching
        self.attn_pool = AttentionPooling(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # KV cache
        self.sliding_cache = SlidingAttnScoreCache(max_batch_size, max_seq_len, num_heads, self.head_dim, torch.float32)

    def scaled_dot_product_attention(self, query, key, value):
        attn_scores = query @ key.transpose(-2,-1) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = attn_weights @ value 
        
        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, embed_dim]
        output = output.transpose(1, 2).contiguous().view(output.size(0), output.size(2), -1)

        return output
    
    def cached_scaled_dot_product_attention(self, query, key, value, casual_attn=False):
        # Cache latest queries, keys and values which correspond to latest embedding in the sequence so that they can be reattended
        self.sliding_cache.append_projections(query, key, value)

        # Retreive the cached query, key and value projection tensors (old and new projected embeddings)
        q, k, v = self.sliding_cache.retrieve_projections()
        q, k, v = q.detach(), k.detach(), v.detach()

        # Element-wise multiplication to produce last column and last row of attn_score
        q_t = query @ k.transpose(-2, -1) * self.scale  # [batch_size, num_heads, 1, max_seq_length]
        k_t = q @ key.transpose(-2, -1) * self.scale    # [batch_size, num_heads, max_seq_length, 1]

        # Update the attention scores with the new query-key attention scores i.e. the new query and key will reattend to the old tokens
        self.sliding_cache.update_attn_score(q_t, k_t)

        # Move the buffer forward by one position
        self.sliding_cache.increment_current_seq_len()

        # Retrieve the new cache (old and new attention scores)
        attn_scores = self.sliding_cache.retrieve_attn_score()
        attn_scores = attn_scores.detach()

        # Compute the scaled dot product attention 
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = attn_weights @ v
        
        # [batch_size, num_heads, max_seq_length, head_dim] -> [batch_size, max_seq_length, embed_dim]
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
            # Perform scaled dot product attention with the sliding caching mechanism of shape [batch_size, max_seq_length, embed_dim]
            out = self.cached_scaled_dot_product_attention(query, key, value)

            # Attention pooling layer to downsample the context window from max_seq_length back to seq_length
            out = self.attn_pool(out)

        else:
            # Perform scaled dot product attention for all attention heads (multi-headed attention)
            out = self.scaled_dot_product_attention(query, key, value)

        out = self.dropout(out)
        return out
    
    def clear_cache(self):
        self.sliding_cache.clear_cache()

class CachedSlidingAttentionEncoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, max_batch_size, max_seq_len, ffn_dropout=0.1, attention_dropout=0.1, query_cache=True):
        super(CachedSlidingAttentionEncoder, self).__init__()
        # Initialize the multi-head attention layer with caching capabilities
        self.attention = CachedMultiHeadAttention(
            d_model, n_head, max_batch_size, max_seq_len, dropout=attention_dropout, query_cache=query_cache
        )
        # Layer normalization for the output of the attention layer
        self.norm1 = nn.LayerNorm(d_model)
        # Dropout for regularization after the first residual connection
        self.dropout1 = nn.Dropout(ffn_dropout)
        # Position-wise feed-forward network
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, ffn_dropout=ffn_dropout
        )
        # Layer normalization for the output of the feed-forward network
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout for regularization after the second residual connection
        self.dropout2 = nn.Dropout(ffn_dropout)

    def forward(self, query, key, value, use_cache=True):
        ''' 
        Expect query to be of shape [batch_size, seq_length, features/embedding]
        '''

        # Attention mechanism which will remove the oldest token if cache exceeds seq_length if use_cache is True
        attn_output = self.attention(query=query, key=key, value=value, use_cache=use_cache)
        # Apply dropout and add residual connection
        query = query + self.dropout1(attn_output)
        # Apply layer normalization to stabilize the output
        query = self.norm1(query)

        # Apply the position-wise feed-forward network
        ffn_output = self.ffn(query)
        # Apply dropout and add residual connection
        query = query + self.dropout2(ffn_output)
        # Apply layer normalization to stabilize the output
        query = self.norm2(query)

        return query
    
    def clear_cache(self):
        self.attention.clear_cache()
