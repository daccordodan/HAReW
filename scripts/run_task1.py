"""CLI: runs Task 1 (cross-subject generalization) contrastive pretraining.

Usage:
    python scripts/run_task1.py --config config/task1_cross_subject.yaml

NOTE: this task's implementation is currently a scaffold (see
src/tasks/task1_cross_subject/contrastive_encoder.py) pending design
decisions on the augmentation/contrastive strategy -- running this will
raise NotImplementedError until that work is done.
"""

from __future__ import annotations

import argparse

from src.tasks.task1_cross_subject.contrastive_encoder import train_contrastive_pretraining

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.sharp_classifier import SHARPClassifier
from src.data.label_mapping import TARGET_CLASSES
from src.training.train_utils import flatten_antennas, evaluate_with_fusion, load_checkpoint, get_data_loaders, update_checkpoints, plot_train_val_history
from src.utils.colab_utils import get_device
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.models.transform import DopplerTraceTransformation

from pathlib import Path

from huggingface_hub import HfApi

logger = get_logger(__name__)
api = HfApi()

def main(config_path: str) -> None:
    """Entry point: loads config, builds dataset/model, runs the full training loop.

    Args:
        config_path: Path to config/base_config.yaml.
    """
    device = get_device()
    config = load_config(config_path)

    output_root=Path(config["paths"]["task_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_name="sharp_t13_best.pt"
    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / checkpoint_name

    model = SHARPClassifier(
        n_classes=len(TARGET_CLASSES),
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    transform = DopplerTraceTransformation
    print(transform)

    logger.info("Model parameter count: %d (paper reference: 128,535)", model.count_parameters())

    train_loader,val_loader=get_data_loaders(logger, config, "S1", transform)
    start_epoch, best_val_acc, history=load_checkpoint(model, optimizer, config, checkpoint_path)

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = evaluate_with_fusion(model, val_loader, loss_fn, device)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, config["training"]["epochs"], train_loss, train_acc, val_loss, val_acc,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            update_checkpoints(logger, api, model, optimizer, config, epoch, val_acc, history, checkpoint_path, checkpoint_name)

    figure_name="train_validation_over_epoch_t13.png"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_train_val_history(history, figures_dir, figure_name)
    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)

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
            flattened_x, flattened_y = flatten_antennas(batch_x, batch_y["label"])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1: cross-subject contrastive pretraining.")
    parser.add_argument("--config", type=str, default="config/task1_cross_subject.yaml")
    args = parser.parse_args()
    main(args.config)
