import torch


def kv_cache_with_update(K, V, qt, kt, vt):
    """
    Single-head KV cache update with Dynamic Memory Compression (DMC).

    Parameters:
        K (torch.Tensor): The key matrix (batch, seqlen, dimension).
        V (torch.Tensor): The value matrix (batch, seqlen, dimension).
        qt (torch.Tensor): The current query vector (batch, seqlen, dimension).
        kt (torch.Tensor): The current key vector (batch, seqlen, dimension).
        vt (torch.Tensor): The current value vector (batch, seqlen, dimension).

    Returns:
        tuple: Updated K, V, qt, kt tensors.

    Example:
    """
    # Calculate alpha_t and omega_t using the first element of kt and qt respectively
    # Assume we use the first element of the last dimension for decision and weighting
    alpha_t = torch.round(torch.sigmoid(kt[:, :, 0]))  # Shape (batch, seqlen)
    omega_t = torch.sigmoid(qt[:, :, 0])  # Shape (batch, seqlen)

    # Extend alpha_t and omega_t for element-wise operations
    alpha_t = alpha_t.unsqueeze(-1)  # Shape (batch, seqlen, 1)
    omega_t = omega_t.unsqueeze(-1)  # Shape (batch, seqlen, 1)

    # Initialize z_t if not provided, we'll assume it starts with the initial omega_t values
    zt = omega_t.clone()

    # ACCUMULATE
    # Update keys and values with weighted average only where alpha_t is 1
    accumulate_mask = alpha_t == 1
    K_new = (K * zt + kt * omega_t) / (zt + omega_t)
    V_new = (V * zt + vt * omega_t) / (zt + omega_t)

    # Only update where accumulate condition is met
    K = torch.where(accumulate_mask, K_new, K)
    V = torch.where(accumulate_mask, V_new, V)

    # APPEND
    # Only update where accumulate condition is not met
    append_mask = alpha_t != 1
    K = torch.where(append_mask, kt, K)
    V = torch.where(append_mask, vt, V)

    # Update z_t considering whether to accumulate or just set to omega_t
    zt = torch.where(accumulate_mask, zt + omega_t, omega_t)

    # Reset the first elements used in kt and qt to 0
    kt[:, :, 0] = 0
    qt[:, :, 0] = 0

    return K, V, qt, kt


# # Example of usage:
# batch_size = 2
# seqlen = 5
# dim = 3

# K = torch.randn(batch_size, seqlen, dim)  # Key matrix
# V = torch.randn(batch_size, seqlen, dim)  # Value matrix
# qt = torch.randn(batch_size, seqlen, dim)  # Query vectors
# kt = torch.randn(batch_size, seqlen, dim)  # Key vectors
# vt = torch.randn(batch_size, seqlen, dim)  # Value vectors

# K_updated, V_updated, qt_updated, kt_updated = kv_cache_with_update(
#     K, V, qt, kt, vt
# )
# print("Updated K:", K_updated)
# print("Updated V:", V_updated)
