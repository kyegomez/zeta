# Import necessary libraries
import torch
import torch.nn.functional as F
from torch import nn


# Define a utility function for the masked fill to avoid overflows
def mask_fill(value, mask, fill_value):
    return value.masked_fill(mask, fill_value)


# Define the asynchronized softmax function
def asynchronized_softmax(Q, K, V, unified_max_value):
    """
    Perform the asynchronized softmax operation with a unified max value.

    :param Q: Query matrix
    :param K: Key matrix
    :param V: Value matrix
    :param unified_max_value: A scalar value to stabilize the softmax computation
    :return: Weighted attention scores after applying softmax
    """
    # Step 1: Compute attention scores by multiplying Q with the transpose of K
    attention_scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Subtract unified_max_value from attention scores to avoid overflow
    attention_scores_sub_max = attention_scores - unified_max_value

    # Step 3: Asynchronously calculate the exponentials for each element
    exp_attention_scores = torch.exp(attention_scores_sub_max)

    # Step 4: Apply mask to avoid recomputation due to overflow
    attention_mask = (attention_scores_sub_max > unified_max_value) | (
        attention_scores_sub_max < -unified_max_value
    )
    exp_attention_scores = mask_fill(exp_attention_scores, attention_mask, 0.0)

    # Step 5: Compute denominators for softmax
    attention_scores_denominator = torch.sum(
        exp_attention_scores, dim=-1, keepdim=True
    )

    # Step 6: Calculate softmax asynchronously
    attention_softmax = exp_attention_scores / attention_scores_denominator

    # Step 7: Apply softmax to Value matrix
    attention_output = torch.matmul(attention_softmax, V)

    return attention_output


# Define the main class for the attention mechanism
class AsynchronizedAttention(nn.Module):
    def __init__(self, d_model, n_heads, unified_max_value):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.unified_max_value = unified_max_value
        self.head_dim = d_model // n_heads

        # Linear layers for Q, K, V projections
        self.qkv_proj = nn.Linear(d_model, d_model * 3)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Project input to Q, K, V
        qkv = self.qkv_proj(x).view(
            batch_size, seq_length, self.n_heads, 3 * self.head_dim
        )
        Q, K, V = qkv.chunk(3, dim=-1)

        # Apply the asynchronized softmax to compute attention
        attention_output = asynchronized_softmax(
            Q, K, V, self.unified_max_value
        )

        return attention_output


# Example usage
if __name__ == "__main__":
    # Define the parameters
    batch_size, seq_length, d_model, n_heads = 2, 16, 512, 8
    unified_max_value = torch.tensor(
        6.0
    )  # This value should be set based on the dataset/model

    # Create random tensors for Q, K, and V
    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    # Initialize the AsynchronizedAttention module
    attention_module = AsynchronizedAttention(
        d_model, n_heads, unified_max_value
    )

    # Compute the attention output
    attention_output = attention_module(Q)
    print("Attention Output Shape:", attention_output)
