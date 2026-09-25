"""Baseline SHARP reproduction: training loop.
"""

from __future__ import annotations

import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from src.models.decision_fusion import fuse_batch
from torch.nn.functional import softmax as softmax

from src.data.doppler_trace_dataset import build_train_val_split
from src.data.label_mapping import TARGET_CLASSES

from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import matplotlib.pyplot as plt

def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None
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
            flattened_x, flattened_y = flatten_antennas(batch_x, batch_y["label"])
            flattened_x, flattened_y = flattened_x.to(device, non_blocking=True), flattened_y.to(device, non_blocking=True)
            # torch.cuda.synchronize()  # Force CPU to wait for GPU transfer

            # Forward Pass
            t2 = time.perf_counter()
            logits = model(flattened_x)
            loss = loss_fn(logits, flattened_y)
            # torch.cuda.synchronize()
            forward_time = time.perf_counter() - t2

            # Backward Pass
            t3 = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.cuda.synchronize()
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

def flatten_antennas(batch_x: torch.Tensor, batch_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        tqdm(val_loader, desc="Validation", leave=False)
        for inputs, targets in val_loader:
            batch_size, Nant, Nw, ND = inputs.shape
            inputs_flat = inputs.view(batch_size * Nant, 1, Nw, ND).to(device)
            targets_flat = targets["label"].repeat_interleave(Nant).to(device)

            logits_flat = model(inputs_flat)
            loss = loss_fn(logits_flat, targets_flat)
            total_loss += loss.item()

            probs_flat = softmax(logits_flat, dim=1)
            probs_reshaped = probs_flat.view(batch_size, Nant, -1)
            fused_preds = fuse_batch(probs_reshaped.cpu(), n_antennas=Nant)

            targets_cpu = targets["label"].cpu()
            correct_fused += (fused_preds == targets_cpu).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / len(val_loader)
    fused_acc = correct_fused / total_samples
    
    return avg_loss, fused_acc


def load_checkpoint(model, optimizer, config, checkpoint_path, local=False):
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }
    
    if local and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch=checkpoint["epoch"]+1
        val_acc=checkpoint["val_acc"]
        history=checkpoint["history"]
    elif local:
        epoch=1
        val_acc = 0.0
    else:
        try:
            checkpoint = torch.load(hf_hub_download(
                repo_id="danieledaccordo/HAReW",
                filename="checkpoints_dir/"+checkpoint_path.name,
                repo_type="model",
            ), map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch=checkpoint["epoch"]+1
            val_acc=checkpoint["val_acc"]
            history=checkpoint["history"]
            torch.save(checkpoint, checkpoint_path)
        except EntryNotFoundError:
            epoch=1
            val_acc = 0.0

    return epoch, val_acc, history

def get_data_loaders(logger, config, set_id, transform=None):
    hw_config = config.get("hardware", {})
    num_workers = int(hw_config.get("num_workers", 2))
    pin_memory = bool(hw_config.get("pin_memory", True))
    persistent_workers = bool(hw_config.get("persistent_workers", True) and num_workers > 0)

    logger.info(
        "Preparing data loaders: workers=%d, pin_memory=%s, persistent_workers=%s",
        num_workers, pin_memory, persistent_workers
    )

    _, train_subset, val_subset, _ = build_train_val_split(
        Path(config["paths"]["doppler_traces_dir"]),
        set_id,
        window_size=config["doppler"]["stacked_vectors_nw"],
        stride=config["doppler"].get("window_stride"),
        n_antennas=config["hardware"]["n_antennas"],
        logger=logger,
        transform=transform
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader

def update_checkpoints(
    logger, api, model, optimizer, config, epoch, val_acc, history,
    checkpoint_path, checkpoint_name, local=False
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "class_names": list(TARGET_CLASSES),
            "val_acc" : val_acc,
            "history": history
        },
        checkpoint_path,
    )
    logger.info("Saved new best checkpoint (val_acc=%.4f) -> %s", val_acc, checkpoint_path)
    if not local:
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="checkpoints_dir/" + checkpoint_name,
            repo_id="danieledaccordo/HAReW",
            repo_type="model",
        )
        logger.info("Uploaded new best checkpoint to Hugging Face")

def plot_train_val_history(history, figures_dir: Path, figure_name):
    plt.figure(figsize=(8,5))
    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train loss",
        color="blue",
        marker="o"
    )
    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation loss",
        color="red",
        marker="o"
    )
    plt.title("Training and validation loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(figures_dir / figure_name, bbox_inches='tight', dpi=300)
    plt.show()
