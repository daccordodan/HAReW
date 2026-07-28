"""Multi-antenna Decision Fusion strategy for SHARP.

Source: Paper 2, Sec. 4.2.

Each of the Nant=4 antennas is independently classified (see
sharp_classifier.py -- one SHARPClassifier forward pass per antenna).
Fusion rule:
    - If >= Nant - 1 antennas agree on the predicted class -> that class wins.
    - Otherwise -> sum activity vectors (softmax probabilities) element-wise
      across antennas, take argmax.

This "decision fusion" approach is explicitly preferred over "data fusion"
(merging antenna data at the network input), which Paper 2's ablation
(Sec. 6.4, Table 5) shows is LESS robust -- especially under NLOS (S4/S5) or
full environment+subject change (S7) -- because data fusion forces equal
weighting of all antennas, including any that are currently in a bad
geometric position relative to the subject.
"""

from __future__ import annotations

from collections import Counter

import torch

DEFAULT_N_ANTENNAS = 4


def majority_vote_fusion(per_antenna_predictions: list[int], n_antennas: int = DEFAULT_N_ANTENNAS) -> int | None:
    """Attempts majority-vote fusion across per-antenna predicted class indices.

    Args:
        per_antenna_predictions: List of predicted class indices, one per antenna.
        n_antennas: Total number of antennas (Nant), default 4.

    Returns:
        The majority-vote class index if the most common prediction is held
        by >= n_antennas - 1 antennas, otherwise None (caller should fall
        back to summed_vector_fusion).
    """
    counts = Counter(per_antenna_predictions)
    top_class, top_count = counts.most_common(1)[0]
    if top_count >= n_antennas - 1:
        return top_class
    return None


def summed_vector_fusion(per_antenna_activity_vectors: torch.Tensor) -> int:
    """Fuses per-antenna activity vectors by element-wise summation + argmax.

    Args:
        per_antenna_activity_vectors: Tensor of shape (Nant, n_classes),
            softmax probabilities (or raw logits -- argmax of the sum is
            invariant to a shared monotonic transform per antenna, but
            probabilities are recommended so no single antenna's raw logit
            scale can dominate the sum).

    Returns:
        Fused predicted class index (argmax of the summed vector).
    """
    summed = per_antenna_activity_vectors.sum(dim=0)
    return int(torch.argmax(summed).item())


def fuse_predictions(
    per_antenna_activity_vectors: torch.Tensor,
    n_antennas: int = DEFAULT_N_ANTENNAS,
) -> int:
    """Full decision-fusion pipeline: majority vote, falling back to summed vectors.

    Args:
        per_antenna_activity_vectors: Tensor of shape (Nant, n_classes) --
            softmax probabilities from each antenna's classifier, for one sample.
        n_antennas: Total number of antennas (Nant), default 4.

    Returns:
        Final fused predicted class index for this sample.
    """
    per_antenna_predictions = torch.argmax(per_antenna_activity_vectors, dim=1).tolist()
    majority_result = majority_vote_fusion(per_antenna_predictions, n_antennas)
    if majority_result is not None:
        return majority_result
    return summed_vector_fusion(per_antenna_activity_vectors)


def fuse_batch(batch_activity_vectors: torch.Tensor, n_antennas: int = DEFAULT_N_ANTENNAS) -> torch.Tensor:
    """Applies fuse_predictions() across a batch.

    Args:
        batch_activity_vectors: Tensor of shape (batch, Nant, n_classes) --
            softmax probabilities from each antenna's classifier.
        n_antennas: Total number of antennas (Nant), default 4.

    Returns:
        LongTensor of shape (batch,) with the fused predicted class per sample.
    """
    fused = [fuse_predictions(batch_activity_vectors[i], n_antennas) for i in range(batch_activity_vectors.shape[0])]
    return torch.tensor(fused, dtype=torch.long)
