import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        return F.relu(x)


class AlphaZeroNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.input_conv = nn.Conv2d(12, 64, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(64)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(64) for _ in range(5)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, action_size)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
