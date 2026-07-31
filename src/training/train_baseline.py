"""Baseline SHARP reproduction: training loop.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from src.models.decision_fusion import fuse_batch
from torch.nn.functional import softmax as softmax

from src.data.doppler_trace_dataset import build_train_val_split
from src.data.label_mapping import TARGET_CLASSES
from src.models.sharp_classifier import SHARPClassifier
from src.training.losses import build_loss_fn
from src.utils.colab_utils import get_device
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)


def _flatten_antennas(batch_x: torch.Tensor, batch_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshapes a (batch, Nant, Nw, ND) batch into (batch*Nant, 1, Nw, ND).

    The single shared SHARPClassifier is trained on every antenna's window
    as an independent training example.

    Args:
        batch_x: Tensor of shape (batch, Nant, Nw, ND).
        batch_y: Tensor of shape (batch,).

    Returns:
        (flattened_x, flattened_y)
    """
    batch, n_ant, nw, nd = batch_x.shape
    flattened_x = batch_x.reshape(batch * n_ant, 1, nw, nd)
    flattened_y = batch_y.unsqueeze(1).expand(batch, n_ant).reshape(batch * n_ant)
    return flattened_x, flattened_y


def evaluate_with_fusion(model, val_loader, loss_fn, device):
    """
    Evaluates the model using the SHARP Decision strategy.
    """
    model.eval()
    total_loss = 0.0
    correct_fused = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            batch_size, Nant, Nw, ND = inputs.shape
            inputs_flat = inputs.view(batch_size * Nant, 1, Nw, ND).to(device)
            targets_flat = targets.repeat_interleave(Nant).to(device)

            logits_flat = model(inputs_flat)
            loss = loss_fn(logits_flat, targets_flat)
            total_loss += loss.item()

            probs_flat = softmax(logits_flat, dim=1)
            probs_reshaped = probs_flat.view(batch_size, Nant, -1)
            fused_preds = fuse_batch(probs_reshaped.cpu(), n_antennas=Nant)

            targets_cpu = targets.cpu()
            correct_fused += (fused_preds == targets_cpu).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / len(val_loader)
    fused_acc = correct_fused / total_samples
    
    return avg_loss, fused_acc

def run_epoch(
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
        loss_fn: Cross-entropy loss instance.
        device: "cuda" or "cpu".
        optimizer: runs backward()+step().

    Returns:
        (mean_loss, accuracy) for the epoch.
    """
    model.train()

    total_loss, total_correct, total_count = 0.0, 0, 0
    context = torch.enable_grad()

    with context:
        for batch_x, batch_y in dataloader:
            flattened_x, flattened_y = _flatten_antennas(batch_x, batch_y)
            flattened_x, flattened_y = flattened_x.to(device), flattened_y.to(device)

            logits = model(flattened_x)
            loss = loss_fn(logits, flattened_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * flattened_y.size(0)
            total_correct += (logits.argmax(dim=1) == flattened_y).sum().item()
            total_count += flattened_y.size(0)

    return total_loss / total_count, total_correct / total_count


def main(config_path: str) -> None:
    """Entry point: loads config, builds dataset/model, runs the full training loop.

    Args:
        config_path: Path to config/base_config.yaml.
    """
    config = load_config(config_path)
    logger.info("Loaded baseline config from %s", config_path)

    device = get_device()
    logger.info("Using device: %s", device)

    data_root = Path(config["paths"]["doppler_traces_dir"])
    output_root=Path(config["paths"]["baseline_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _, train_subset, val_subset, _ = build_train_val_split(
        data_root,
        window_size=config["doppler"]["stacked_vectors_nw"],
        stride=config["doppler"].get("window_stride"),
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_subset), len(val_subset))

    train_loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = SHARPClassifier(
        n_classes=len(TARGET_CLASSES),
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
    ).to(device)
    logger.info("Model parameter count: %d (paper reference: 128,535)", model.count_parameters())

    loss_fn = build_loss_fn(n_classes=len(TARGET_CLASSES))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_val_acc = 0.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = evaluate_with_fusion(model, val_loader, loss_fn, device)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, config["training"]["epochs"], train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoints_dir / "sharp_baseline_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "class_names": list(TARGET_CLASSES),
                },
                checkpoint_path,
            )
            logger.info("Saved new best checkpoint (val_acc=%.4f) -> %s", val_acc, checkpoint_path)

    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SHARP baseline classifier on set S1.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
