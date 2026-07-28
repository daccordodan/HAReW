"""Simplified Inception module used as SHARP's feature extractor.

Source: Paper 2, Sec. 4.1 ("Learning Architecture for HAR") + Fig. 4.

Inspired by the reduction block of Inception-v4, but simplified to 3 parallel
branches combining MAXPOOL and CONV layers at different kernel sizes for
multi-scale feature extraction. Chosen over a full ~43M-parameter Inception-v4
or deep stacked-CNN alternatives (MSDNet, RANet, ELASTIC) specifically for
lightweight, low-cost-device deployability -- an explicit design constraint
in the paper, not just an implementation convenience.

Input: Doppler trace tensor of shape (batch, 1, Nw, ND) = (batch, 1, 340, 100).

NOTE ON EXACT HYPERPARAMETERS: Paper 2 describes the module's topology
(3 branches, MAXPOOL+CONV, 1x1 reduction 15->3 feature maps) but does not
publish exact kernel sizes/strides/padding. This implementation is a
faithful reconstruction of the *described* topology; kernel sizes below
are a reasonable choice validated against the paper's reported total
parameter count (128,535, see sharp_classifier.py) rather than reverse
-engineered from an undisclosed spec. If you have access to the original
SHARP GitHub repo's exact layer definitions, swap them in here directly.
"""

from __future__ import annotations

import torch
from torch import nn


class SimplifiedInceptionModule(nn.Module):
    """3-branch Inception-style feature extractor for Doppler traces.

    Branches (each operating on the same input, concatenated on the channel
    axis at the end):
        Branch A: 1x1 conv -> 3x3 conv          (small receptive field)
        Branch B: 1x1 conv -> 5x5 conv          (medium receptive field)
        Branch C: 3x3 max-pool -> 1x1 conv      (pooled context branch)
    """

    def __init__(self, in_channels: int = 1, branch_channels: int = 5) -> None:
        """Initializes the 3 parallel branches.

        Args:
            in_channels: Number of input channels (1 -- the Doppler trace
                is treated as a single-channel "image", per the
                Presentation's Slide 4 framing).
            branch_channels: Output channels per branch. With 3 branches,
                total concatenated channels = 3 * branch_channels (default
                5 -> 15 channels, matching the paper's stated "15 feature
                maps" before the 1x1 reduction to 3).
        """
        super().__init__()
        self.branch_channels = branch_channels

        self.branch_a = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.branch_c = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    @property
    def out_channels(self) -> int:
        """Total output channels after concatenating all 3 branches."""
        return 3 * self.branch_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the 3 branches and concatenates their outputs.

        Args:
            x: Input tensor of shape (batch, in_channels, Nw, ND).

        Returns:
            Concatenated multi-scale feature map of shape
            (batch, out_channels, Nw, ND).
        """
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        return torch.cat([a, b, c], dim=1)
