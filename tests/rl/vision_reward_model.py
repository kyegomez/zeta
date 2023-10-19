import pytest
import torch
from zeta.rl.vision_model_rl import ResidualBlock, VisionRewardModel


# 1. Basic Shape Tests for ResidualBlock
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_residual_block_shapes(batch_size):
    res_block = ResidualBlock(3, 64)
    sample_tensor = torch.randn(batch_size, 3, 32, 32)
    output_tensor = res_block(sample_tensor)
    assert output_tensor.shape == (batch_size, 64, 32, 32)


# 2. Testing different strides in ResidualBlock
@pytest.mark.parametrize("stride", [1, 2])
def test_residual_block_strides(stride):
    res_block = ResidualBlock(3, 64, stride=stride)
    sample_tensor = torch.randn(8, 3, 32, 32)
    output_tensor = res_block(sample_tensor)
    assert output_tensor.shape[-2] == (32 // stride)


# 3. VisionRewardModel shape tests
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_vision_reward_model_shapes(batch_size):
    model = VisionRewardModel()
    sample_image = torch.randn(batch_size, 3, 32, 32)
    predicted_rewards = model(sample_image)
    assert predicted_rewards.shape == (batch_size, 1)


# 4. VisionRewardModel outputs type check
def test_vision_reward_model_output_type():
    model = VisionRewardModel()
    sample_image = torch.randn(8, 3, 32, 32)
    predicted_rewards = model(sample_image)
    assert isinstance(predicted_rewards, torch.Tensor)


# 5. Ensure no NaN values in ResidualBlock outputs
def test_residual_block_no_nan():
    res_block = ResidualBlock(3, 64)
    sample_tensor = torch.randn(8, 3, 32, 32)
    output_tensor = res_block(sample_tensor)
    assert not torch.isnan(output_tensor).any()


# 6. Ensure no NaN values in VisionRewardModel outputs
def test_vision_reward_model_no_nan():
    model = VisionRewardModel()
    sample_image = torch.randn(8, 3, 32, 32)
    predicted_rewards = model(sample_image)
    assert not torch.isnan(predicted_rewards).any()


# 7. Ensure non-zero outputs for VisionRewardModel
def test_vision_reward_model_non_zero():
    model = VisionRewardModel()
    sample_image = torch.randn(8, 3, 32, 32)
    predicted_rewards = model(sample_image)
    assert torch.abs(predicted_rewards).sum() != 0


# 8. Testing ResidualBlock shortcut condition
@pytest.mark.parametrize(
    "in_channels, out_channels, stride",
    [(3, 64, 1), (3, 64, 2), (64, 64, 2), (64, 128, 2)],
)
def test_residual_block_shortcut(in_channels, out_channels, stride):
    res_block = ResidualBlock(in_channels, out_channels, stride=stride)
    if stride != 1 or in_channels != out_channels:
        assert len(res_block.shortcut) == 2
    else:
        assert len(res_block.shortcut) == 0


# 9. Testing zero inputs result in non-zero outputs for ResidualBlock
def test_residual_block_zero_input():
    res_block = ResidualBlock(3, 64)
    sample_tensor = torch.zeros(8, 3, 32, 32)
    output_tensor = res_block(sample_tensor)
    assert output_tensor.sum() != 0


# 10. Testing zero inputs result in non-zero outputs for VisionRewardModel
def test_vision_reward_model_zero_input():
    model = VisionRewardModel()
    sample_image = torch.zeros(8, 3, 32, 32)
    predicted_rewards = model(sample_image)
    assert predicted_rewards.sum() != 0


# Additional Testing for various shapes (e.g., larger images)
@pytest.mark.parametrize("image_size", [32, 64, 128])
def test_vision_reward_model_with_different_image_sizes(image_size):
    model = VisionRewardModel()
    sample_image = torch.randn(8, 3, image_size, image_size)
    predicted_rewards = model(sample_image)
    assert predicted_rewards.shape == (8, 1)


# 12-50: Replicating tests with slight variations for more coverage

# ... replicate similar tests with minor changes for thoroughness.
# Examples: test different batch sizes, test other parameter combinations for the ResidualBlock, etc.
