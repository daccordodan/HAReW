"""Baseline SHARP reproduction: training loop.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from src.models.decision_fusion import fuse_batch
from torch.nn.functional import softmax as softmax

from src.data.doppler_trace_dataset import build_train_val_split
from src.data.label_mapping import TARGET_CLASSES

from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

import matplotlib.pyplot as plt


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


def load_checkpoint(model, optimizer, config, checkpoint_path):
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }
    
    try:
        checkpoint=torch.load(hf_hub_download(
            repo_id="danieledaccordo/HAReW",
            filename="checkpoints_dir/sharp_baseline_best.pt",
            repo_type="model"
        ))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch=checkpoint["epoch"]+1
        val_acc=checkpoint["val_acc"]
        history=checkpoint["history"]
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
    except EntryNotFoundError:
        epoch=0
        val_acc = 0.0

    return epoch, val_acc, history

def get_data_loaders(logger, config, set_id):
    _, train_subset, val_subset, _ = build_train_val_split(
        Path(config["paths"]["doppler_traces_dir"]),
        set_id,
        window_size=config["doppler"]["stacked_vectors_nw"],
        stride=config["doppler"].get("window_stride"),
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_subset), len(val_subset))

    train_loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["training"]["batch_size"], shuffle=False)

    return train_loader,val_loader

def update_checkpoints(logger, api, model, optimizer, config, epoch, val_acc, history, checkpoint_path, checkpoint_name):
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
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="checkpoints_dir/"+checkpoint_name,
        repo_id="danieledaccordo/HAReW",
        repo_type="model"
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

