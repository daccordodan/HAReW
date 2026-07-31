"""
    Multi-antenna Decision Fusion strategy for SHARP.
"""

from __future__ import annotations

from collections import Counter

import torch

DEFAULT_N_ANTENNAS = 4


def majority_vote_fusion(per_antenna_predictions: list[int], n_antennas: int = DEFAULT_N_ANTENNAS) -> int | None:
    """Attempts majority-vote fusion across per-antenna predicted class indices.

    Args:
        per_antenna_predictions: List of predicted class indices, one per antenna.
        n_antennas: Total number of antennas.

    Returns:
        The majority-vote class index if the most common prediction is held
        by >= n_antennas - 1 antennas, otherwise None.
    """
    counts = Counter(per_antenna_predictions)
    top_class, top_count = counts.most_common(1)[0]
    if top_count >= n_antennas - 1:
        return top_class
    return None


def summed_vector_fusion(per_antenna_activity_vectors: torch.Tensor) -> int:
    """Fuses per-antenna activity vectors by element-wise summation + argmax.

    Args:
        per_antenna_activity_vectors: Tensor of shape (Nant, n_classes) softmax probabilities.

    Returns:
        Fused predicted class index.
    """
    summed = per_antenna_activity_vectors.sum(dim=0)
    return int(torch.argmax(summed).item())


def fuse_predictions(
    per_antenna_activity_vectors: torch.Tensor,
    n_antennas: int = DEFAULT_N_ANTENNAS,
) -> int:
    """Full decision pipeline: majority vote, falling back to summed vectors.

    Args:
        per_antenna_activity_vectors: Tensor of shape (Nant, n_classes) --
            softmax probabilities from each antenna's classifier, for one sample.
        n_antennas: Total number of antennas (Nant).

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
        n_antennas: Total number of antennas (Nant).

    Returns:
        Fused predicted class for each sample of the tensor.
    """
    fused = [fuse_predictions(batch_activity_vectors[i], n_antennas) for i in range(batch_activity_vectors.shape[0])]
    return torch.tensor(fused, dtype=torch.long)
