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

from src.training.train_utils import flatten_antennas

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


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
        nw: int = 340,
        nd: int = 100,
        reduced_channels: int = 3,
        dropout_rate: float = 0.2,
        projection_dim: int = 128
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

        self.reduced_channels=reduced_channels
        self.pooled_nw, self.pooled_nd = nw // 2, nd // 2

        self.projector = nn.Sequential(
            nn.Linear(reduced_channels * self.pooled_nw * self.pooled_nd, reduced_channels * self.pooled_nw * self.pooled_nd),
            nn.ReLU(),
            nn.Linear(reduced_channels * self.pooled_nw * self.pooled_nd, projection_dim)
        )
        

    def forward(self, doppler_trace: torch.Tensor) -> torch.Tensor:
        """Encodes a Doppler trace into the contrastive embedding space.

        Args:
            x: Tensor of shape (batch, 1, Nw, ND).

        Returns:
            Embedding of shape (batch, projection_dim).
        """

        features = self.feature_extractor(doppler_trace)  # (batch, 15, Nw/2, ND/2)
        reduced = self.relu(self.reduction_conv(features))  # (batch, 3, Nw/2, ND/2)
        flattened = torch.flatten(reduced, start_dim=1)
        dropped = self.dropout(flattened)
        return self.projector(dropped)

    def count_parameters(self) -> int:
            """Returns the total trainable parameter count, for comparison."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

def freeze(model):
    for param in model.feature_extractor.branch_a:
        param.requires_grad = False
    for param in model.feature_extractor.branch_b:
        param.requires_grad = False
    for param in model.feature_extractor.branch_c:
        param.requires_grad = False
    for param in model.reduction_conv:
        param.requires_grad = False
    for param in model.relu:
        param.requires_grad = False
    for param in model.dropout:
        param.requires_grad = False

def train_contrastive_pretraining(    
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Runs one epoch of training or evaluation.

    Args:
        model: SHARPClassifier.
        dataloader: Yields (batch_x, batch_y) with batch_x shape (batch, Nant, Nw, ND).
        loss_fn: Used loss function.
        device: "cuda" or "cpu".
        optimizer: runs backward()+step().

    Returns:
        (mean_loss, accuracy) for the epoch.
    """
    model.train()

    total_loss, total_count = 0.0, 0
    context = torch.enable_grad()

    with context:
        for batch_x, batch_y in dataloader:
            print(type(batch_x[0]))
            flattened_x1, flattened_y = flatten_antennas(batch_x[0], batch_y["label"])
            flattened_x2, _ = flatten_antennas(batch_x[1], batch_y["label"])
            flattened_x1, flattened_x2, flattened_y = flattened_x1.to(device), flattened_x2.to(device), flattened_y.to(device)

            logits1 = model(flattened_x1)
            logits2 = model(flattened_x2)
            loss = loss_fn(logits1, logits2, flattened_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * flattened_y.size(0)
            total_count += flattened_y.size(0)

    return total_loss / total_count

def evaluate_encoder(model, val_loader, loss_fn, device):
    total_loss=0

    model.eval()
    with torch.no_grad():
        tqdm(val_loader, desc="Validation", leave=False)
        for inputs, _ in val_loader:
            batch_size, Nant, Nw, ND = inputs.shape
            inputs_flat_1 = inputs[0].view(batch_size * Nant, 1, Nw, ND).to(device)
            inputs_flat_2 = inputs[1].view(batch_size * Nant, 1, Nw, ND).to(device)

            logits_flat_1=model(inputs_flat_1)
            logits_flat_2=model(inputs_flat_2)

            loss = loss_fn(logits_flat_1, logits_flat_2)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss



'''
self.reduction_conv = nn.Conv2d(self.feature_extractor.out_channels, reduced_channels, kernel_size=1)
self.relu = nn.ReLU(inplace=True)
self.dropout = nn.Dropout(p=dropout_rate)
pooled_nw, pooled_nd = nw // 2, nd // 2
self.classifier_head = nn.Linear(reduced_channels * pooled_nw * pooled_nd, n_classes)
'''