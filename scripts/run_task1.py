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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tasks.task1_cross_subject.contrastive_encoder import ContrastiveEncoder, FineTunedModel, evaluate_encoder, train_contrastive_pretraining, freeze
from src.models.transform import dopplerTraceTransformation
from src.models.losses import NTXentLoss
from src.data.label_mapping import TARGET_CLASSES
from src.training.train_utils import evaluate_with_fusion, flatten_antennas, load_checkpoint, get_data_loaders, update_checkpoints, plot_train_val_history
from src.utils.colab_utils import get_device
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

from pathlib import Path
from huggingface_hub import HfApi

logger = get_logger(__name__)
api = HfApi()

def main(config_path: str, local: bool = False) -> None:
    """Entry point: loads config, builds dataset/model, runs the full training loop.

    Args:
        config_path: Path to config/base_config.yaml.
    """
    device = get_device()
    config = load_config(config_path)

    n_classes=config["model"]["n_classes_primary"]

    output_root=Path(config["paths"]["task_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_name="pretraining_t13_best.pt"
    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / checkpoint_name

    contrastive_enc = ContrastiveEncoder(
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
    ).to(device)
    loss_fn = NTXentLoss()
    optimizer = torch.optim.Adam(contrastive_enc.parameters(), lr=config["training"]["learning_rate"])

    transform = dopplerTraceTransformation

    logger.info("Model parameter count: %d (paper reference: 128,535)", contrastive_enc.count_parameters())

    train_loader,val_loader=get_data_loaders(logger, config, "S1", transform)
    start_epoch, best_val_loss, history=load_checkpoint(
        contrastive_enc, optimizer, config, checkpoint_path, local=local
    )

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        train_loss = train_contrastive_pretraining(contrastive_enc, train_loader, loss_fn, device, optimizer)
        val_acc = 0
        val_loss = evaluate_encoder(contrastive_enc, val_loader, loss_fn, device)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch, config["training"]["epochs"], train_loss, val_loss,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            update_checkpoints(
                logger, api, contrastive_enc, optimizer, config, epoch, val_acc, history,
                checkpoint_path, checkpoint_name, local=local
            )

    figure_name="pretrain_validation_over_epoch_t13.png"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_train_val_history(history, figures_dir, figure_name)
    logger.info("Pre - training complete. Best val_loss=%.4f", best_val_loss)

    freeze(contrastive_enc)
    loss_fn=nn.CrossEntropyLoss()
    train_loader,val_loader=get_data_loaders(logger, config, "S1", transform)

    classifier = nn.Linear(contrastive_enc.reduced_channels * contrastive_enc.pooled_nw * contrastive_enc.pooled_nd, n_classes)
    HAReW_model= FineTunedModel(
        contrastive_enc,
        classifier
    )

    checkpoint_name="training_t13_best.pt"
    checkpoint_path = checkpoints_dir / checkpoint_name
    start_epoch, best_val_loss, history=load_checkpoint(
        HAReW_model, optimizer, config, checkpoint_path, local=local
    )

    for epoch in range(start_epoch, config["training"]["finetune_epochs"] + 1):
        logger.info("Starting training epoch %d/%d...", epoch, config["training"]["epochs"])
        train_loss, train_acc = run_epoch(
            HAReW_model, train_loader, loss_fn, device, optimizer, logger=logger
        )
        val_loss, val_acc = evaluate_with_fusion(HAReW_model, val_loader, loss_fn, device)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, config["training"]["finetune_epochs"], train_loss, train_acc, val_loss, val_acc,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            update_checkpoints(
                logger, api, HAReW_model, optimizer, config, epoch, val_acc, history,
                checkpoint_path, checkpoint_name, local=local
            )

    figure_name="train_validation_over_epoch_baseline.png"
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1: cross-subject contrastive pretraining.")
    parser.add_argument("--config", type=str, default="config/task1_cross_subject.yaml")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files instead of Hugging Face.")
    args = parser.parse_args()
    main(args.config, local=args.local)
