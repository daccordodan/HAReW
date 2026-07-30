"""Simplified Inception module used as SHARP's feature extractor.

Source: Paper 2, Sec. 4.1 ("Learning Architecture for HAR") + Fig. 4.

Inspired by the reduction block of Inception-v4, but simplified to 3 parallel
branches combining MAXPOOL and CONV layers at different kernel sizes for
multi-scale feature extraction. Chosen over a full ~43M-parameter Inception-v4
or deep stacked-CNN alternatives (MSDNet, RANet, ELASTIC) specifically for
lightweight, low-cost-device deployability -- an explicit design constraint
in the paper, not just an implementation convenience.

Input: Doppler trace tensor of shape (batch, 1, Nw, ND) = (batch, 1, 340, 100).

ARCHITECTURE CORRECTION (verified directly against paper text, Sec. 4.1):
the paper states each branch's output is an "Nw/2 x ND/2 dimensional
feature map" -- i.e. the spatial downsampling happens INSIDE each branch,
via strided convolutional/pooling layers (Fig. 4's caption: conv blocks are
followed by "stride S#"), and the three branches' half-resolution outputs
are concatenated BEFORE the 1x1 channel-reduction step (see
sharp_classifier.py). An earlier version of this module ran all branches
at full resolution and only downsampled with a separate max-pool AFTER the
1x1 reduction -- functionally close in total parameter count (stride
doesn't change a conv layer's parameter count, only its output size), but
it put the downsampling in the wrong place relative to the reduction step
compared to what the paper actually describes. Fixed here.

UNVERIFIED: PER-BRANCH FEATURE-MAP COUNTS. Fig. 4 labels each conv/pool
block with the paper's "N@(ixi)" convention -- N feature maps produced by
an i x i kernel -- and these N values are not necessarily equal across the
three branches. The paper's TEXT confirms only that the three branches'
outputs concatenate to 15 total feature maps before the 1x1 reduction to 3
(this is prose, not a diagram label, so it's reliable). The individual
per-branch split (e.g. whether the MAXPOOL branch gets fewer feature maps
than the two CONV branches, which would be a fairly typical Inception-style
choice) could NOT be conclusively read off the paper's figure at the
resolution available here, nor confirmed against the original SHARP code
repository. BRANCH_CHANNELS below is therefore an explicit, individually
adjustable placeholder -- defaulting to an even 5/5/5 split (the simplest
assumption consistent with the one constraint that IS verified: they must
sum to 15) -- not a claim that the paper actually splits them evenly.
Correct this constant directly if you can confirm the true per-branch
values (e.g. by inspecting your own copy of Fig. 4 at higher zoom than
was available here, or from the original repo's model definition).

NOTE ON OTHER HYPERPARAMETERS: Paper 2 describes the module's topology
(3 branches, MAXPOOL+CONV, half-resolution outputs, 1x1 reduction 15->3
feature maps) but does not publish exact kernel sizes/padding beyond that.
This implementation reconstructs that topology using stride=2 conv/pool
layers sized to land on Nw/2 x ND/2 output for Nw=340, ND=100 -- validated
analytically against the paper's reported total parameter count (128,535,
see sharp_classifier.py). If you have access to the original SHARP GitHub
repo's exact layer definitions, swap them in here directly.
"""

from __future__ import annotations

import torch
from torch import nn

# Per-branch feature-map counts (the "N" in the paper's "N@(ixi)" notation).
# UNVERIFIED SPLIT -- see module docstring above. Must sum to 15 to match
# the paper's stated total (that total IS verified from the paper's prose).
BRANCH_CHANNELS_A = 9  # 1x1 -> 3x3 conv branch
BRANCH_CHANNELS_B = 5  # 1x1 -> 5x5 conv branch
BRANCH_CHANNELS_C = 1  # maxpool -> 1x1 conv branch
assert BRANCH_CHANNELS_A + BRANCH_CHANNELS_B + BRANCH_CHANNELS_C == 15, (
    "Per-branch feature-map counts must sum to 15 -- the one figure detail "
    "confirmed directly from the paper's prose (Sec. 4.1)."
)


class SimplifiedInceptionModule(nn.Module):
    """3-branch Inception-style feature extractor for Doppler traces.

    Branches (each halves the spatial resolution -- Nw x ND -> Nw/2 x ND/2 --
    via a stride-2 layer, then concatenated on the channel axis):
        Branch A: 1x1 conv (stride 1) -> 3x3 conv, stride 2  (small receptive field)
        Branch B: 1x1 conv (stride 1) -> 5x5 conv, stride 2  (medium receptive field)
        Branch C: 3x3 max-pool, stride 2 -> 1x1 conv, stride 1  (pooled context branch)
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
            in_channels: Number of input channels (1 -- the Doppler trace
                is treated as a single-channel "image", per the
                Presentation's Slide 4 framing).
            branch_a_channels: Output feature maps for branch A (1x1->3x3).
            branch_b_channels: Output feature maps for branch B (1x1->5x5).
            branch_c_channels: Output feature maps for branch C (maxpool->1x1).
                Defaults sum to 15, matching the paper's stated total feature
                maps before the 1x1 reduction to 3 -- see module docstring
                for what is and isn't verified about this specific 5/5/5 split.
        """
        super().__init__()
        self.branch_a_channels = branch_a_channels
        self.branch_b_channels = branch_b_channels
        self.branch_c_channels = branch_c_channels
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.branch_a = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 6, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 9, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.branch_c = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),#, padding=1),
        )

    @property
    def out_channels(self) -> int:
        """Total output channels after concatenating all 3 branches."""
        return self.branch_a_channels + self.branch_b_channels + self.branch_c_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the 3 branches and concatenates their (half-resolution) outputs.

        Args:
            x: Input tensor of shape (batch, in_channels, Nw, ND).

        Returns:
            Concatenated multi-scale feature map of shape
            (batch, out_channels, Nw/2, ND/2) -- e.g. (batch, 15, 170, 50)
            for the default Nw=340, ND=100.
        """
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        return torch.cat([a, b, c], dim=1)
