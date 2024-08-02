import torch


class AttentionTest:
    def __init__(self, batch_size, num_heads, max_seq_len):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.current_seq_len = 0
        self.attn_score_cache = torch.zeros(
            (batch_size, num_heads, max_seq_len, max_seq_len)
        )

    def update_attn_score(self, q_t, k_t):
        """
        Appends the most recent scaled query (column) and key (row) pair of the attention score to the cache.
        Args:
            q_t (torch.Tensor): The query tensor to be added to the last column of size [batch_size, num_heads, 1, seq_length]
            k_t (torch.Tensor): The key tensor to be added to the last row of size [batch_size, num_heads, seq_length, 1]
        """
        if self.current_seq_len >= self.max_seq_len - 1:
            # If the sequence length exceeds the maximum, shift the cache left by one position in both dimensions
            self.attn_score_cache[:, :, :-1, :] = self.attn_score_cache[
                :, :, 1:, :
            ].clone()
            self.attn_score_cache[:, :, :, :-1] = self.attn_score_cache[
                :, :, :, 1:
            ].clone()

        # Add the new column and row to the attention score cache
        self.attn_score_cache[:, :, self.current_seq_len, :] = q_t[:, :, 0, :]
        self.attn_score_cache[:, :, :, self.current_seq_len] = k_t[:, :, :, 0]

        if self.current_seq_len < self.max_seq_len - 1:
            self.current_seq_len += 1


# Parameters for the test
batch_size = 1
num_heads = 2
max_seq_len = 6

# Create an instance of the mock class
attn_test = AttentionTest(batch_size, num_heads, max_seq_len)

# Generating random tensors for q_t and k_t and updating the cache
for i in range(max_seq_len + 5):  # Adding more than max_seq_len to test shifting
    q_t = torch.rand((batch_size, num_heads, 1, max_seq_len))
    k_t = torch.rand((batch_size, num_heads, max_seq_len, 1))
    print(f"Update {i + 1}:")
    attn_test.update_attn_score(q_t, k_t)
    print(attn_test.attn_score_cache)
    print()
