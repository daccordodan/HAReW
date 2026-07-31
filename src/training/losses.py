"""
    Loss functions for SHARP classifier training.
"""

from __future__ import annotations

import torch
from torch import nn


def build_loss_fn(n_classes: int = 5) -> nn.Module:
    """Builds the standard cross-entropy loss used to train SHARP.
    This function is here in case of future changes in order to complete the other tasks.

    Args:
        n_classes: Number of activity classes.

    Returns:
        A torch.nn.CrossEntropyLoss instance.
    """
    return nn.CrossEntropyLoss()
