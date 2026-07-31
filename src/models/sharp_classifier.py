"""
    Full per-antenna SHARP classifier head.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.inception_module import SimplifiedInceptionModule

N_CLASSES_PRIMARY = 5    # E, W, R, J, L
DROPOUT_RATE = 0.2
TOTAL_PARAMS_REFERENCE = 128_535  # For comparison with the paper

DEFAULT_NW = 340
DEFAULT_ND = 100


class SHARPClassifier(nn.Module):
    """Single-antenna SHARP activity classifier.

    Attributes:
        feature_extractor: SimplifiedInceptionModule instance (internally
            halves Nw x ND to Nw/2 x ND/2 within each branch).
        reduction_conv: 1x1 conv reducing branch-concatenated channels -> 3,
            applied at the already-halved Nw/2 x ND/2 resolution.
        dropout: Dropout layer (rate=0.2).
        classifier_head: Final dense layer producing the activity vector (logits).
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES_PRIMARY,
        nw: int = DEFAULT_NW,
        nd: int = DEFAULT_ND,
        reduced_channels: int = 3,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        """Initializes the classifier.

        Args:
            n_classes: Number of output activity classes.
            nw: Nw, input Doppler trace time dimension.
            nd: ND, input Doppler trace bin dimension.
            reduced_channels: Output channels of the 1x1 reduction conv.
            dropout_rate: Dropout probability before the final dense layer.
        """
        super().__init__()
        self.feature_extractor = SimplifiedInceptionModule(in_channels=1)
        self.reduction_conv = nn.Conv2d(self.feature_extractor.out_channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        pooled_nw, pooled_nd = nw // 2, nd // 2
        self.classifier_head = nn.Linear(reduced_channels * pooled_nw * pooled_nd, n_classes)

    def forward(self, doppler_trace: torch.Tensor) -> torch.Tensor:
        """Produces an activity vector from a Doppler trace of a single antenna.

        Args:
            doppler_trace: Tensor of shape.

        Returns:
            Raw logits of shape (batch, n_classes).
        """
        features = self.feature_extractor(doppler_trace)  # (batch, 15, Nw/2, ND/2)
        reduced = self.relu(self.reduction_conv(features))  # (batch, 3, Nw/2, ND/2)
        flattened = torch.flatten(reduced, start_dim=1)
        dropped = self.dropout(flattened)
        return self.classifier_head(dropped)

    def count_parameters(self) -> int:
        """Returns the total trainable parameter count, for comparison."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
