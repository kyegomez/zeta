import pytest
import torch
from zeta.nn.biases.relative_position_bias import RelativePositionBias


# Helper function to create random data
def create_random_data(shape):
    return torch.randn(shape)


# Test case for initializing RelativePositionBias
def test_relative_position_bias_init():
    bias = RelativePositionBias()
    assert isinstance(bias, RelativePositionBias)


# Test case for _relative_position_bucket method
def test_relative_position_bucket():
    bias = RelativePositionBias()

    relative_position = torch.tensor([[0, 1, -1], [2, -2, 3]])
    bucketed = bias._relative_position_bucket(relative_position)

    expected_result = torch.tensor([[16, 17, 15], [18, 14, 19]])
    assert torch.equal(bucketed, expected_result)


# Test case for computing bias values
def test_compute_bias():
    bias = RelativePositionBias()
    qlen, klen = 3, 4
    values = bias.compute_bias(qlen, klen)

    assert values.shape == (1, 1, qlen, klen)


# Test case for forward pass
def test_forward():
    bias = RelativePositionBias()
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for forward pass with step parameter
def test_forward_with_step():
    bias = RelativePositionBias()
    batch_size, qlen, klen, step = 2, 3, 4, 5
    values = bias.forward(batch_size, qlen, klen, step=step)

    assert values.shape == (batch_size, qlen, klen)


# Test case for bidirectional bias
def test_bidirectional_bias():
    bias = RelativePositionBias(bidirectional=True)
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for different numbers of buckets
def test_different_num_buckets():
    bias = RelativePositionBias(num_buckets=64)
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for different max distances
def test_different_max_distance():
    bias = RelativePositionBias(max_distance=256)
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for multiple heads
def test_multiple_heads():
    bias = RelativePositionBias(num_heads=4)
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for checking if bias values are within a reasonable range
def test_bias_values_range():
    bias = RelativePositionBias()
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)


# Test case for checking if bias values match between different instances of RelativePositionBias
def test_bias_values_equal():
    bias1 = RelativePositionBias()
    bias2 = RelativePositionBias()
    batch_size, qlen, klen = 2, 3, 4
    values1 = bias1.forward(batch_size, qlen, klen)
    values2 = bias2.forward(batch_size, qlen, klen)

    assert torch.equal(values1, values2)


# Test case for batch size of 1
def test_batch_size_1():
    bias = RelativePositionBias()
    batch_size, qlen, klen = 1, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for bidirectional bias with batch size of 1
def test_bidirectional_bias_batch_size_1():
    bias = RelativePositionBias(bidirectional=True)
    batch_size, qlen, klen = 1, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for checking if bias values are consistent across multiple calls with the same parameters
def test_consistent_bias_values():
    bias = RelativePositionBias()
    batch_size, qlen, klen = 2, 3, 4
    values1 = bias.forward(batch_size, qlen, klen)
    values2 = bias.forward(batch_size, qlen, klen)

    assert torch.equal(values1, values2)


# Test case for checking if bias values are different for different batch sizes
def test_different_batch_sizes():
    bias = RelativePositionBias()
    batch_size1, qlen, klen = 2, 3, 4
    batch_size2 = batch_size1 + 1
    values1 = bias.forward(batch_size1, qlen, klen)
    values2 = bias.forward(batch_size2, qlen, klen)

    assert not torch.equal(values1, values2)


# Test case for checking if bias values are different for different qlen and klen
def test_different_qlen_klen():
    bias = RelativePositionBias()
    batch_size, qlen1, klen1 = 2, 3, 4
    qlen2, klen2 = qlen1 + 1, klen1 + 1
    values1 = bias.forward(batch_size, qlen1, klen1)
    values2 = bias.forward(batch_size, qlen2, klen2)

    assert not torch.equal(values1, values2)


# Test case for checking if bias values are different for different steps
def test_different_steps():
    bias = RelativePositionBias()
    batch_size, qlen, klen = 2, 3, 4
    step1, step2 = 0, 1
    values1 = bias.forward(batch_size, qlen, klen, step=step1)
    values2 = bias.forward(batch_size, qlen, klen, step=step2)

    assert not torch.equal(values1, values2)


# Test case for checking if the device of bias values matches the device of the model parameters
def test_device_match():
    bias = RelativePositionBias()
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.device == next(bias.parameters()).device


# Test case for initializing with a different number of buckets
def test_different_num_buckets_init():
    bias = RelativePositionBias(num_buckets=64)
    assert bias.num_buckets == 64


# Test case for initializing with a different max distance
def test_different_max_distance_init():
    bias = RelativePositionBias(max_distance=256)
    assert bias.max_distance == 256


# Test case for initializing with a different number of heads
def test_different_num_heads_init():
    bias = RelativePositionBias(num_heads=4)
    assert bias.num_heads == 4


# Test case for bidirectional bias with different qlen and klen
def test_bidirectional_bias_different_qlen_klen():
    bias = RelativePositionBias(bidirectional=True)
    batch_size, qlen1, klen1 = 2, 3, 4
    qlen2, klen2 = qlen1 + 1, klen1 + 1
    values1 = bias.forward(batch_size, qlen1, klen1)
    values2 = bias.forward(batch_size, qlen2, klen2)

    assert not torch.equal(values1, values2)


# Test case for initializing with bidirectional set to False
def test_bidirectional_false_init():
    bias = RelativePositionBias(bidirectional=False)
    assert not bias.bidirectional


# Test case for initializing with different bidirectional settings
def test_different_bidirectional_init():
    bias1 = RelativePositionBias(bidirectional=True)
    bias2 = RelativePositionBias(bidirectional=False)

    assert bias1.bidirectional
    assert not bias2.bidirectional


# Test case for checking if bias values are different for different bidirectional settings
def test_different_bidirectional_bias_values():
    bias1 = RelativePositionBias(bidirectional=True)
    bias2 = RelativePositionBias(bidirectional=False)
    batch_size, qlen, klen = 2, 3, 4
    values1 = bias1.forward(batch_size, qlen, klen)
    values2 = bias2.forward(batch_size, qlen, klen)

    assert not torch.equal(values1, values2)


# Test case for initializing with negative max distance
def test_negative_max_distance_init():
    with pytest.raises(ValueError):
        RelativePositionBias(max_distance=-128)


# Test case for initializing with negative num buckets
def test_negative_num_buckets_init():
    with pytest.raises(ValueError):
        RelativePositionBias(num_buckets=-32)


# Test case for initializing with a large max distance
def test_large_max_distance_init():
    bias = RelativePositionBias(max_distance=10000)
    assert bias.max_distance == 10000


# Test case for initializing with a large num buckets
def test_large_num_buckets_init():
    bias = RelativePositionBias(num_buckets=64)
    assert bias.num_buckets == 64


# Test case for bidirectional bias with max distance
def test_bidirectional_bias_large_max_distance():
    bias = RelativePositionBias(bidirectional=True, max_distance=1000)
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for large num buckets
def test_large_num_buckets():
    bias = RelativePositionBias(num_buckets=64)
    batch_size, qlen, klen = 2, 3, 4
    values = bias.forward(batch_size, qlen, klen)

    assert values.shape == (batch_size, qlen, klen)


# Test case for bidirectional bias with negative max distance
def test_bidirectional_bias_negative_max_distance():
    with pytest.raises(ValueError):
        RelativePositionBias(bidirectional=True, max_distance=-128)
