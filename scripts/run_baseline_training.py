"""CLI: trains the SHARP baseline classifier on set S1 (PyTorch).

Usage (local or Colab):
    python scripts/run_baseline_training.py --config config/base_config.yaml

    
"""

from __future__ import annotations

import argparse
import torch
from torch import nn

from src.models.sharp_classifier import SHARPClassifier
from src.training.train_utils import run_independent_epochs, evaluate_with_fusion, load_checkpoint, get_data_loaders, update_checkpoints, plot_train_val_history
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
    # torch_num_threads = config["local_hardware"].get("torch_num_threads")
    # torch_num_interop_threads = config["local_hardware"].get("torch_num_interop_threads")
    # if torch_num_threads is not None:
    #     torch.set_num_threads(int(torch_num_threads))
    # if torch_num_interop_threads is not None:
    #     torch.set_num_interop_threads(int(torch_num_interop_threads))
    # logger.info(
    #     "Using device=%s | torch threads=%d | interop threads=%d",
    #     device,
    #     torch.get_num_threads(),
    #     torch.get_num_interop_threads(),
    # )

    output_root=Path(config["paths"]["baseline_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)


    models = [None] * 4
    optimizers = [None] * 4
    loss_fn = nn.CrossEntropyLoss()
    for i in range(config["hardware"]["n_antennas"]):
        models[i] =  SHARPClassifier(
            n_classes=config["model"]["n_classes_primary"],
            nw=config["doppler"]["stacked_vectors_nw"],
            nd=config["doppler"]["velocity_bins_nd"],
        ).to(device)

        optimizers[i] = torch.optim.Adam(models[i].parameters(), lr=config["training"]["learning_rate"])

        checkpoint_name=f"sharp_last_antenna_{i}.pt"
        checkpoints_dir = output_root / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoints_dir / checkpoint_name
        start_epoch, best_val_accuracies, history=load_checkpoint(
            models[i], optimizers[i], config, checkpoint_path, local=local
        )
    
    # logger.info("Model parameter count: %d (paper reference: 128,535)", model.count_parameters())

    logger.info("Starting dataset parsing and recording loading...")
    train_loader,val_loader=get_data_loaders(logger, config, "S1")
    logger.info("Dataset preparation complete.")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        logger.info("Starting training epoch %d/%d...", epoch, config["training"]["epochs"])
        train_losses, training_accuracies = run_independent_epochs(
            models, train_loader, loss_fn, device, optimizers
        )
        val_losses, val_accuracies = evaluate_with_fusion(models, val_loader, loss_fn, device)

        for i in range(4):
            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch, config["training"]["epochs"], train_losses[i], training_accuracies[i], val_losses[i], val_accuracies[i],
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_losses)
        history["val_loss"].append(val_losses)      # not used
        history["val_acc"].append(val_accuracies)   # not used

        for i in range(4):
            if val_accuracies[i] > best_val_accuracies[i]:
                best_val_accuracies[i] = val_accuracies[i]
                checkpoint_name=f"sharp_best_antenna_{i}.pt"
                checkpoints_dir = output_root / "checkpoints"
                checkpoints_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoints_dir / checkpoint_name

                update_checkpoints(
                    logger, api, models[i], optimizers[i], config, epoch, val_accuracies[i], history,
                    checkpoint_path, checkpoint_name, local=local
                )

    figure_name="train_validation_over_epoch_independent_baseline.png"
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_train_val_history(history, figures_dir, figure_name)
    for i in range(4):
        logger.info(f"Training complete. Best val_acc_{i}=%.4f", best_val_accuracies[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SHARP baseline.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files instead of Hugging Face.")
    args = parser.parse_args()
    main(args.config, local=args.local)
