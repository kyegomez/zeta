import torch
import torch.nn.functional as F


def standard_softmax(tensor):
    return F.softmax(tensor, dim=0)


# selu softmax


def selu_softmax(x):
    """
    selu_softmax works by first applying the scaled exponential linear unit
    (selu) activation function to the input tensor and then applying softmax.

    x: input tensor
    """
    # selu params
    alpha, scale = (
        1.6732632423543772848170429916717,
        1.0507009873554804934193349852946,
    )
    return F.softmax(scale * F.selu(x, alpha), dim=0)


# 2. Sparsemax


def sparsemax(x, k):
    """
    sparsemax works by first sorting the input tensor in descending order and
    then applying the following formula to the sorted tensor:

    sparsemax(z) = max(0, z - tau(z)) where tau(z) = (sum_i=1^k z_i - 1) / k

    z: input tensor
    k: number of elements to keep


    """
    original_size = x.size()
    x = x.view(-1, original_size[-1])
    dim = 1
    number_of_logits = x.size(dim)

    # Check if k is greater than the number of logits
    if k > number_of_logits:
        raise ValueError("k cannot be greater than the number of logits.")

    x = x - torch.max(x, dim=dim, keepdim=True).values
    sorted_x, _ = torch.sort(x, dim=dim, descending=True)
    cumulative_values = torch.cumsum(sorted_x, dim=dim) - 1
    range_values = torch.arange(
        start=1, end=number_of_logits + 1, device=x.device
    )
    bound = (sorted_x - cumulative_values / range_values) > 0
    rho = torch.count_nonzero(bound, dim=dim)

    # Check if k is too large and adjust it
    if k > rho.max():
        k = rho.max().item()

    tau = cumulative_values.gather(dim, rho.unsqueeze(dim) - 1)
    tau /= rho.to(dtype=torch.float32)
    return torch.max(torch.zeros_like(x), x - tau.unsqueeze(dim)).view(
        original_size
    )


# 3. Local Softmax
def local_softmax(tensor, num_chunks: int = 2):
    """
    local softmax works by splitting the input tensor into num_chunks smaller
    tensors and then applying softmax on each chunk. The results are then
    concatenated and returned.

    tensor: input tensor
    num_chunks: number of chunks to split the tensor into


    """
    # split the tensor into num chunks smaller tensor
    tensors = torch.chunk(tensor, num_chunks, dim=0)

    # apply softmax on each chunk and collect the results in a list
    results = [F.softmax(t, dim=0) for t in tensors]

    # concat results
    concated_results = torch.cat(results, dim=0)

    return concated_results


# 4. Fast Softmax


def fast_softmax(tensor):
    """
    LogSumExp trick for numerical stability

    tensor = torch.rand(10, 5)
    result = fast_softmax(tensor)
    print(result)

    """
    shiftx = tensor - torch.max(tensor)

    exps = torch.exp(shiftx)

    return exps / torch.sum(exps)


# 5. Sparse Softmax
def sparse_softmax(z, k: int = 3):
    """
    Sparsemax works by first sorting the input tensor in descending order and
    then applying the following formula to the sorted tensor:

    sparsemax(z) = max(0, z - tau(z)) where tau(z) = (sum_i=1^k z_i - 1) / k

    z: input tensor
    k: number of elements to keep

    """
    _, top_k_indices = z.topk(k, dim=0)
    omega_k = top_k_indices

    # compute sparse softmax transformation
    exp_z = torch.exp(z)
    masked_sum_exp = exp_z[omega_k].sum()
    values = torch.zeros_like(z)
    values[omega_k] = exp_z[omega_k] / masked_sum_exp

    return values


# 6. gumbelmax


def gumbelmax(x, temp=1.0, hard=False):
    """
    Gumbelmax works by adding Gumbel noise to the input tensor x and then
    applying softmax. The hard parameter controls whether the output will
    be one-hot or a probability distribution.

    x: input tensor
    temp: temperature parameter
    hard: if True, the returned tensor will be one-hot, otherwise a probability distribution

    """
    gumbels = -torch.empty_like(x).exponential_().log()
    y = x + gumbels
    y = F.softmax(y / temp, dim=-1)

    if hard:
        y_hard = torch.zeros_like(x).scatter_(
            -1, y.argmax(dim=-1, keepdim=True), 1.0
        )
        y = y_hard - y.detach() + y
    return y


# 7. Softmax with temp


def temp_softmax(x, temp=1.0):
    """
    Temp softmax works by dividing the input tensor by the temperature
    parameter and then applying softmax.

    x: input tensor
    temp: temperature parameter

    """
    return F.softmax(x / temp, dim=-1)


# 8. logit scaled softmax


def logit_scaled_softmax(x, scale=1.0):
    """
    logit scaled softmax works by multiplying the input tensor by the scale
    parameter and then applying softmax.

    x: input tensor
    scale: scale parameter
    """
    return F.softmax(x * scale, dim=-1)


# 9. norm exponential softmax


def norm_exp_softmax(x, scale=1.0):
    """
    norm exponential softmax works by applying the following formula to the
    input tensor:

    norm_exp_softmax(x) = exp(scale * x) / sum(exp(scale * x))

    x: input tensor
    scale: scale parameter
    """
    return torch.exp(scale * x) / torch.exp(scale * x).sum(dim=-1, keepdim=True)
