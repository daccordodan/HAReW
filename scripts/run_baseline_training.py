"""CLI: trains the SHARP baseline classifier on set S1 (PyTorch).

Usage (local or Colab):
    python scripts/run_baseline_training.py --config config/base_config.yaml

    
"""

from __future__ import annotations

from tqdm import tqdm
import time

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.sharp_classifier import SHARPClassifier
from src.data.label_mapping import TARGET_CLASSES
from src.training.train_utils import flatten_antennas, evaluate_with_fusion, load_checkpoint, get_data_loaders, update_checkpoints, plot_train_val_history
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
    torch_num_threads = config["local_hardware"].get("torch_num_threads")
    torch_num_interop_threads = config["local_hardware"].get("torch_num_interop_threads")
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))
    if torch_num_interop_threads is not None:
        torch.set_num_interop_threads(int(torch_num_interop_threads))
    logger.info(
        "Using device=%s | torch threads=%d | interop threads=%d",
        device,
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )

    output_root=Path(config["paths"]["baseline_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_name="sharp_baseline_best.pt"
    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / checkpoint_name

    model = SHARPClassifier(
        n_classes=len(TARGET_CLASSES),
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"]) # fused?

    logger.info("Model parameter count: %d (paper reference: 128,535)", model.count_parameters())

    logger.info("Starting dataset parsing and recording loading...")
    train_loader,val_loader=get_data_loaders(logger, config, "S1")
    logger.info("Dataset preparation complete.")
    start_epoch, best_val_acc, history=load_checkpoint(
        model, optimizer, config, checkpoint_path, local=local
    )

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        logger.info("Starting training epoch %d/%d...", epoch, config["training"]["epochs"])
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, device, optimizer, logger=logger
        )
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
            update_checkpoints(
                logger, api, model, optimizer, config, epoch, val_acc, history,
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
    logger=None,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    context = torch.enable_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    t0 = time.perf_counter()
    
    with context:
        for batch_x, batch_y in pbar:
            data_time = time.perf_counter() - t0
            
            # Transfer to GPU
            t1 = time.perf_counter()
            flattened_x, flattened_y = flatten_antennas(batch_x, batch_y["label"])
            flattened_x, flattened_y = flattened_x.to(device, non_blocking=True), flattened_y.to(device, non_blocking=True)
            torch.cuda.synchronize()  # Force CPU to wait for GPU transfer
            transfer_time = time.perf_counter() - t1

            # Forward Pass
            t2 = time.perf_counter()
            logits = model(flattened_x)
            loss = loss_fn(logits, flattened_y)
            torch.cuda.synchronize()
            forward_time = time.perf_counter() - t2

            # Backward Pass
            t3 = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            backward_time = time.perf_counter() - t3

            total_loss += loss.item() * flattened_y.size(0)
            total_correct += (logits.argmax(dim=1) == flattened_y).sum().item()
            total_count += flattened_y.size(0)

            # Update the progress bar with the live times!
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "data(s)": f"{data_time:.2f}",
                "fwd(s)": f"{forward_time:.2f}",
                "bwd(s)": f"{backward_time:.2f}"
            })
            t0 = time.perf_counter()

    return total_loss / total_count, total_correct / total_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SHARP baseline.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files instead of Hugging Face.")
    args = parser.parse_args()
    main(args.config, local=args.local)
