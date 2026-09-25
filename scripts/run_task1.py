"""CLI: runs Task 1 (cross-subject generalization) contrastive pretraining.

Usage:
    python scripts/run_task1.py --config config/task1_cross_subject.yaml
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from src.tasks.task1_cross_subject.contrastive_encoder import ContrastiveEncoder, FineTunedModel, evaluate_encoder, train_contrastive_pretraining, freeze
from src.models.transform import dopplerTraceTransformation
from src.models.losses import NTXentLoss
from src.training.train_utils import run_epoch, evaluate_with_fusion, load_checkpoint, get_data_loaders, update_checkpoints, plot_train_val_history
from src.utils.utils import load_config, get_logger

from pathlib import Path
from huggingface_hub import HfApi

logger = get_logger(__name__)
api = HfApi()

def main(config_path: str, local: bool = False) -> None:
    """Entry point: loads config, builds dataset/model, runs the full training loop.

    Args:
        config_path: Path to config/base_config.yaml.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    logger.info("Starting dataset parsing and recording loading...")
    train_loader,val_loader=get_data_loaders(logger, config, "S1", transform)
    logger.info("Dataset preparation complete.")
    start_epoch, best_val_loss, history=load_checkpoint(
        contrastive_enc, optimizer, config, checkpoint_path, local=local
    )
    val_acc = 0

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        logger.info("Starting training epoch %d/%d...", epoch, config["training"]["epochs"])
        train_loss = train_contrastive_pretraining(
            contrastive_enc, train_loader, loss_fn, device, optimizer
        )
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
    train_loader,val_loader=get_data_loaders(logger, config, "S1")

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

    figure_name="train_validation_over_epoch_task1.png"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_train_val_history(history, figures_dir, figure_name)
    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1: cross-subject contrastive pretraining.")
    parser.add_argument("--config", type=str, default="config/task1_cross_subject.yaml")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files instead of Hugging Face.")
    args = parser.parse_args()
    main(args.config, local=args.local)
