"""Unit tests for src/models/ -- inception_module, sharp_classifier, decision_fusion.

Requires torch (see requirements.txt); run with `pytest` in an environment
where torch is installed (e.g. Colab, or after `pip install -r requirements.txt`).
"""

from __future__ import annotations

import torch

from src.models.decision_fusion import fuse_predictions, majority_vote_fusion, summed_vector_fusion
from src.models.inception_module import SimplifiedInceptionModule
from src.models.sharp_classifier import N_CLASSES_PRIMARY, TOTAL_PARAMS_REFERENCE, SHARPClassifier


def test_inception_module_output_shape():
    module = SimplifiedInceptionModule(in_channels=1, branch_channels=5)
    x = torch.randn(2, 1, 340, 100)
    out = module(x)
    assert out.shape == (2, module.out_channels, 340, 100)
    assert module.out_channels == 15  # 3 branches * 5 channels, matches paper's "15 feature maps"


def test_sharp_classifier_forward_shape():
    model = SHARPClassifier(n_classes=N_CLASSES_PRIMARY)
    x = torch.randn(4, 1, 340, 100)
    logits = model(x)
    assert logits.shape == (4, N_CLASSES_PRIMARY)


def test_sharp_classifier_param_count_close_to_paper():
    model = SHARPClassifier(n_classes=N_CLASSES_PRIMARY)
    count = model.count_parameters()
    # Not an exact match (paper doesn't publish full hyperparameters), but
    # should be within a few percent of the reported 128,535.
    assert abs(count - TOTAL_PARAMS_REFERENCE) / TOTAL_PARAMS_REFERENCE < 0.05


def test_majority_vote_fusion_agrees():
    # 3 of 4 antennas agree -> majority wins (Nant-1 = 3 threshold)
    assert majority_vote_fusion([0, 0, 0, 1], n_antennas=4) == 0


def test_majority_vote_fusion_falls_through():
    # Only 2 of 4 agree -> below threshold, caller should fall back
    assert majority_vote_fusion([0, 0, 1, 2], n_antennas=4) is None


def test_summed_vector_fusion():
    probs = torch.tensor(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.6, 0.3, 0.1],
            [0.05, 0.05, 0.9],
        ]
    )
    # class 1 has the highest summed probability across antennas
    assert summed_vector_fusion(probs) == 1


def test_fuse_predictions_uses_majority_when_available():
    probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.1, 0.9],
        ]
    )
    # 3 of 4 antennas predict class 0 -> majority vote wins
    assert fuse_predictions(probs, n_antennas=4) == 0
