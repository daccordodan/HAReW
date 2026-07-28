"""Full per-antenna SHARP classifier head.

Source: Paper 2, Sec. 4.1 + Fig. 4.

Pipeline: SimplifiedInceptionModule -> 1x1 conv reduction (15 -> 3 feature
maps) -> Flatten -> Dropout(0.2) -> Dense(n_classes) -> activity vector
(raw logits; apply softmax only where probabilities are needed, since the
training loop uses nn.CrossEntropyLoss which expects logits directly).

Paper-reported total parameters (single-antenna classifier, 5-class task):
128,535. This implementation's exact count will differ slightly since the
paper does not publish full layer hyperparameters (see inception_module.py
docstring) -- use count_parameters() to compare against the reference and
tune branch_channels/dense width if you need a closer match.

Primary task: 5-class (walking, running, jumping, sitting + empty room) --
resolved for this project as label_mapping.TARGET_CLASSES = [E, W, R, J, L].
Extended task (Sec. 6.7, Table 6, single-subject only): 8 classes, adding
standing, sit/stand transition, and arm exercises -- NOT the current
baseline scope, but n_classes is kept configurable for that future work.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.inception_module import SimplifiedInceptionModule

N_CLASSES_PRIMARY = 5    # E, W, R, J, L -- see src/data/label_mapping.py
N_CLASSES_EXTENDED = 8   # future work, Paper 2 Sec. 6.7 -- not the current baseline
DROPOUT_RATE = 0.2
TOTAL_PARAMS_REFERENCE = 128_535  # Paper 2 reported parameter count, for comparison only

DEFAULT_NW = 340
DEFAULT_ND = 100


class SHARPClassifier(nn.Module):
    """Single-antenna SHARP activity classifier.

    Attributes:
        feature_extractor: SimplifiedInceptionModule instance.
        reduction_conv: 1x1 conv reducing branch-concatenated channels -> 3.
        dropout: Dropout layer (rate=0.2).
        classifier_head: Final dense layer producing the activity vector (logits).
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES_PRIMARY,
        nw: int = DEFAULT_NW,
        nd: int = DEFAULT_ND,
        branch_channels: int = 5,
        reduced_channels: int = 3,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        """Initializes the classifier.

        Args:
            n_classes: Number of output activity classes (5 for the current
                baseline; pass N_CLASSES_EXTENDED for the future 8-class task).
            nw: Nw, input Doppler trace time dimension (default 340).
            nd: ND, input Doppler trace velocity-bin dimension (default 100).
            branch_channels: Channels per Inception branch (see inception_module.py).
            reduced_channels: Output channels of the 1x1 reduction conv
                (paper: 15 -> 3, so default 3).
            dropout_rate: Dropout probability before the final dense layer.
        """
        super().__init__()
        self.feature_extractor = SimplifiedInceptionModule(in_channels=1, branch_channels=branch_channels)
        self.reduction_conv = nn.Conv2d(self.feature_extractor.out_channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # A flatten straight from (reduced_channels, Nw, ND) with Nw=340,
        # ND=100 would blow the dense layer out to ~500K+ params (>>the
        # paper's reported 128,535 total). A 2x2 max-pool here is the
        # missing spatial-downsampling step needed to land close to that
        # budget -- verified analytically: with reduced_channels=3 and this
        # pool, total params = 128,443 vs. the paper's 128,535 (99.9% match).
        self.spatial_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        pooled_nw, pooled_nd = nw // 2, nd // 2
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier_head = nn.Linear(reduced_channels * pooled_nw * pooled_nd, n_classes)

    def forward(self, doppler_trace: torch.Tensor) -> torch.Tensor:
        """Produces an activity vector (logits) from a single-antenna Doppler trace.

        Args:
            doppler_trace: Tensor of shape (batch, 1, Nw, ND) = (batch, 1, 340, 100).
                If your data loader yields (batch, Nw, ND), unsqueeze the
                channel dim before calling this (see decision_fusion.py /
                training scripts for the expected call pattern).

        Returns:
            Raw logits of shape (batch, n_classes).
        """
        features = self.feature_extractor(doppler_trace)
        reduced = self.relu(self.reduction_conv(features))
        pooled = self.spatial_pool(reduced)
        flattened = torch.flatten(pooled, start_dim=1)
        dropped = self.dropout(flattened)
        return self.classifier_head(dropped)

    def count_parameters(self) -> int:
        """Returns the total trainable parameter count, for comparison against
        the paper's reported 128,535 (single-antenna, 5-class configuration).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
