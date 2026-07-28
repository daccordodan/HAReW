"""Loss functions for SHARP classifier training.

Source: Paper 2, Sec. 4.1 -- cross-entropy loss over the activity vector
(5-class primary task, or 8-class in the extended single-subject variant,
Sec. 6.7).
"""

from __future__ import annotations

import torch
from torch import nn


def build_loss_fn(n_classes: int = 5) -> nn.Module:
    """Builds the standard cross-entropy loss used to train SHARP.

    Args:
        n_classes: Number of activity classes (5 primary, 8 extended).

    Returns:
        A torch.nn.CrossEntropyLoss instance.
    """
    return nn.CrossEntropyLoss()
