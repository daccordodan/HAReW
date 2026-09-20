"""Task 1: Cross-subject generalization via contrastive / self-supervised learning.

Source: Presentation, Slide 9 ("Challenges and possible objectives"), item 1.

Goal: improve on the baseline's cross-subject generalization by learning
subject-invariant Doppler-trace representations, e.g. via a contrastive
pretraining stage (SimCLR-style) before the supervised classification head.

STRUCTURAL SEPARATION FROM THE BASELINE (project owner's constraint #3):
    - This module MAY import shared building blocks (SimplifiedInceptionModule)
      from src/models/, but MUST NOT import or depend on
      src/training/train_baseline.py or its checkpoints directly.
    - Its own config lives at config/task1_cross_subject.yaml, entirely
      separate from config/base_config.yaml.
    - Its own checkpoints/logs write to outputs/task1_cross_subject/, never
      to outputs/baseline/.

This file is a scaffold: the contrastive loss and augmentation strategy for
Doppler traces are an open design choice for the extension phase and are
intentionally left unimplemented until that phase is scoped in detail
(see docs/PROJECT_STATUS.md, "Recommended Next Actions").
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.inception_module import SimplifiedInceptionModule


class ContrastiveEncoder(nn.Module):
    """Encoder used for contrastive pretraining on Doppler traces.

    Reuses SimplifiedInceptionModule as the backbone so the pretrained
    weights can later be transplanted into a SHARPClassifier-compatible
    feature extractor, but keeps its own projection head, entirely separate
    from the baseline's classifier_head.
    """

    def __init__(
        self,
        n_classes: int = 5, # E, W, R, J, L
        nw: int = 340,
        nd: int = 100,
        reduced_channels: int = 3,
        dropout_rate: float = 0.2,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes a Doppler trace into the contrastive embedding space.

        Args:
            x: Tensor of shape (batch, 1, Nw, ND).

        Returns:
            Embedding of shape (batch, projection_dim).
        """
        raise NotImplementedError("TODO: implement forward pass once head is defined.")


def train_contrastive_pretraining(config_path: str) -> None:
    """Entry point for the contrastive pretraining stage.

    Args:
        config_path: Path to config/task1_cross_subject.yaml.
    """
    raise NotImplementedError("TODO: implement once augmentation/loss strategy is decided.")
