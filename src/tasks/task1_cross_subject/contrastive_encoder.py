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

    def __init__(self, projection_dim: int = 64) -> None:
        """Initializes the contrastive encoder.

        Args:
            projection_dim: Output dimensionality of the contrastive
                projection head (typical SimCLR-style range: 64-256).
        """
        super().__init__()
        self.backbone = SimplifiedInceptionModule(in_channels=1)
        raise NotImplementedError(
            "TODO: define pooling + projection head, and the augmentation "
            "pipeline appropriate for Doppler traces (e.g. time-window jitter, "
            "velocity-bin masking) once the contrastive strategy is scoped."
        )

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
