"""
Package model provides the neural network model.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StigNet(nn.Module):
    """
    Compact RGB CNN that turns one frame (B, 3, H, W) uint8 into:
        throttle_logits (0: neutral, 1: accelerate, 2: brake)
        steering_logits (0: straight, 1: left, 2: right)

    Parameters:
        size: (Height, Width) of the input frames.
    """

    def __init__(
        self,
        size: Tuple[int, int],
    ):
        super().__init__()

        # Crop top 12% to ignore the HUD lap timer and EXIT button.
        self.crop_top = int(size[0] * 0.12)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )

        self.head = nn.Sequential(
            nn.Linear(64 * 3 * 5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.fc_t = nn.Linear(128, 3)
        self.fc_s = nn.Linear(128, 3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x[:, :, self.crop_top :, :]
        x = x.to(dtype=torch.float32, copy=False) / 255.0
        if x.shape[2] >= 160:
            x = F.avg_pool2d(x, 2)
        z = self.head(self.cnn(x))
        return self.fc_t(z), self.fc_s(z)
