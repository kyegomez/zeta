import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual Block module for a vision model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride value for the convolutional layers. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class VisionRewardModel(nn.Module):
    """
    VisionRewardModel is a neural network model that extracts image features and predicts rewards.

    Args:
        None

    Attributes:
        layer1 (ResidualBlock): The first residual block for image feature extraction.
        layer2 (ResidualBlock): The second residual block for image feature extraction.
        layer3 (ResidualBlock): The third residual block for image feature extraction.
        layer4 (ResidualBlock): The fourth residual block for image feature extraction.
        fc1 (nn.Linear): The fully connected layer for feature transformation.
        fc2 (nn.Linear): The fully connected layer for reward prediction.

    Methods:
        forward(x): Performs forward pass through the network.

    """

    def __init__(self):
        super().__init__()

        # Image Feature Extractor
        self.layer1 = ResidualBlock(3, 64)
        self.layer2 = ResidualBlock(64, 128, 2)
        self.layer3 = ResidualBlock(128, 256, 2)
        self.layer4 = ResidualBlock(256, 512, 2)
        self.fc1 = nn.Linear(512, 128)

        # Reward predictor
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        reward = self.fc2(out)
        return reward


# Example usage

# # 1. Example for ResidualBlock
# res_block = ResidualBlock(in_channels=3, out_channels=64)
# sample_tensor = torch.randn(8, 3, 32, 32)
# output_tensor = res_block(sample_tensor)

# # 2. Example for VisionRewardModel
# vision_rewardim = VisionRewardModel()
# sample_image = torch.randn(8, 3, 32, 32)
# predicted_rewards = vision_rewardim(sample_image)

# print(output_tensor.shape, predicted_rewards.shape)
