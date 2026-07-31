"""
    Simplified Inception module used as SHARP's feature extractor.
"""

from __future__ import annotations

import torch
from torch import nn

# Per-branch feature-map counts (the "N" in the paper's "N@(ixi)" notation).
BRANCH_CHANNELS_A = 9 
BRANCH_CHANNELS_B = 5
BRANCH_CHANNELS_C = 1

class SimplifiedInceptionModule(nn.Module):
    """
        3-branch Inception-style feature extractor for Doppler traces.
    """

    def __init__(
        self,
        in_channels: int = 1,
        branch_a_channels: int = BRANCH_CHANNELS_A,
        branch_b_channels: int = BRANCH_CHANNELS_B,
        branch_c_channels: int = BRANCH_CHANNELS_C,
    ) -> None:
        """Initializes the 3 parallel branches.

        Args:
            in_channels: Number of input channels.
            branch_a_channels: Output feature maps for branch A.
            branch_b_channels: Output feature maps for branch B.
            branch_c_channels: Output feature maps for branch C.
        """
        super().__init__()
        self.branch_a_channels = branch_a_channels
        self.branch_b_channels = branch_b_channels
        self.branch_c_channels = branch_c_channels

        self.branch_a = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 6, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 9, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.branch_c = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def out_channels(self) -> int:
        """Total output channels after concatenating all 3 branches."""
        return self.branch_a_channels + self.branch_b_channels + self.branch_c_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the 3 branches and concatenates their outputs.

        Args:
            x: Input tensor of shape (batch, in_channels, Nw, ND).

        Returns:
            Concatenated multi-scale feature map of shape
            (batch, out_channels, Nw/2, ND/2).
        """
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        return torch.cat([a, b, c], dim=1)
