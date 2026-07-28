"""Task 3: Person identification, repurposing subject-discriminative features.

Source: Presentation, Slide 9, item 3 ("can subject-discriminative features
[from Task 1] be repurposed to identify WHO is performing the activity?").

Goal: a secondary classification head over the same Doppler-trace feature
extractor, predicting person identity instead of (or alongside) activity.
Explicitly designed to build on Task 1's subject-invariant/discriminative
representations once those exist -- but kept import-isolated for now so
Task 3 can also be developed/tested independently against a simple
from-scratch baseline encoder.

STRUCTURAL SEPARATION FROM THE BASELINE (project owner's constraint #3):
    - Its own config lives at config/task3_person_id.yaml.
    - Its own checkpoints/logs write to outputs/task3_person_id/.
    - Person-ID labels are a NEW label space (see PersonIDClassifier below)
      entirely separate from label_mapping.TARGET_CLASSES (activities) --
      never conflate the two label spaces in one model output layer.

This file is a scaffold: exact person-ID label source (e.g. parsed from a
person-ID suffix in filenames, if present in the person's actual dataset
drop) is not yet confirmed against real files and must be verified before
implementation (see docs/PROJECT_STATUS.md open gaps).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.inception_module import SimplifiedInceptionModule


class PersonIDClassifier(nn.Module):
    """Classifies person identity from a Doppler trace.

    Mirrors SHARPClassifier's topology (same feature extractor design) but
    with an entirely separate output layer/label space -- person IDs, not
    activities.
    """

    def __init__(self, n_persons: int, nw: int = 340, nd: int = 100) -> None:
        """Initializes the person-ID classifier.

        Args:
            n_persons: Number of distinct person identities in scope. NOT
                yet confirmed for this project's actual data -- Paper 2's
                own validation subset uses only 3 subjects (P1-P3), which
                may be too few for a meaningful person-ID task; verify
                against your actual dataset drop before implementing.
            nw: Nw, input Doppler trace time dimension (default 340).
            nd: ND, input Doppler trace velocity-bin dimension (default 100).
        """
        super().__init__()
        self.feature_extractor = SimplifiedInceptionModule(in_channels=1)
        raise NotImplementedError(
            "TODO: define reduction/pooling/head once n_persons and the "
            "person-ID label source are confirmed against real data."
        )

    def forward(self, doppler_trace: torch.Tensor) -> torch.Tensor:
        """Produces person-ID logits from a single-antenna Doppler trace.

        Args:
            doppler_trace: Tensor of shape (batch, 1, Nw, ND).

        Returns:
            Raw logits of shape (batch, n_persons).
        """
        raise NotImplementedError("TODO: implement forward pass once head is defined.")


def train_person_identification(config_path: str) -> None:
    """Entry point for the person-identification extension.

    Args:
        config_path: Path to config/task3_person_id.yaml.
    """
    raise NotImplementedError(
        "TODO: implement once person-ID label source is confirmed against "
        "the actual dataset drop (see docs/PROJECT_STATUS.md open gaps)."
    )
