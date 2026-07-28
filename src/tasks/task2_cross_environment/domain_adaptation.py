"""Task 2: Cross-environment & cross-day robustness beyond the SHARP baseline.

Source: Presentation, Slide 9, item 2; Paper 2 Sec. 6.2-6.7 (S2-S7 results).

Goal: push accuracy on S2-S7 (especially S6/S7 -- new environment) above the
baseline SHARP numbers, e.g. via domain-adaptation techniques (adversarial
domain confusion, test-time batch-norm recalibration, or environment-
conditioned augmentation) applied on top of the reproduced baseline
classifier.

STRUCTURAL SEPARATION FROM THE BASELINE (project owner's constraint #3):
    - Loads a frozen or fine-tunable baseline checkpoint as a STARTING
      POINT only (explicit, one-directional dependency: this module may
      read outputs/baseline/checkpoints/*.pt, but the baseline training
      code must never import anything from src/tasks/).
    - Its own config lives at config/task2_cross_environment.yaml.
    - Its own checkpoints/logs write to outputs/task2_cross_environment/.

This file is a scaffold pending the specific domain-adaptation strategy
being chosen for the extension phase.
"""

from __future__ import annotations

import torch

from src.models.sharp_classifier import SHARPClassifier


def load_baseline_checkpoint_as_starting_point(checkpoint_path: str, device: str) -> SHARPClassifier:
    """Loads a trained baseline checkpoint to use as the starting point for domain adaptation.

    Args:
        checkpoint_path: Path to a checkpoint produced by
            src/training/train_baseline.py (e.g.
            "outputs/baseline/checkpoints/sharp_baseline_best.pt").
        device: "cuda" or "cpu".

    Returns:
        A SHARPClassifier with the baseline's trained weights loaded.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = SHARPClassifier(
        n_classes=len(checkpoint.get("class_names", [])) or config["model"]["n_classes_primary"],
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def train_domain_adaptation(config_path: str) -> None:
    """Entry point for the cross-environment/day robustness extension.

    Args:
        config_path: Path to config/task2_cross_environment.yaml.
    """
    raise NotImplementedError(
        "TODO: implement once the specific domain-adaptation strategy "
        "(e.g. adversarial domain confusion vs. environment-conditioned "
        "augmentation) is chosen for the extension phase."
    )
