"""Baseline SHARP reproduction: zero-shot evaluation across S1-S7.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.data.doppler_trace_dataset import build_train_val_split, build_zero_shot_test_set
from src.data.label_mapping import TARGET_CLASSES
from src.evaluation.metrics import compute_accuracy, compute_confusion_matrix, compute_f1_per_activity
from src.models.decision_fusion import fuse_batch
from src.models.sharp_classifier import SHARPClassifier
from src.utils.colab_utils import get_device
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

S7_REFERENCE_ACCURACY = 0.9599 # For comparison with the paper


@torch.no_grad()
def evaluate_set(model: torch.nn.Module, dataloader: DataLoader, device: str) -> tuple[list[int], list[int]]:
    """Runs fused-prediction inference over a full dataset.

    Args:
        model: Trained SHARPClassifier.
        dataloader: Yields (batch_x, batch_y) with batch_x shape (batch, Nant, Nw, ND).
        device: "cuda" or "cpu".

    Returns:
        (y_true, y_pred): parallel lists of ground-truth and fused-predicted
        class indices for every sample in the dataset.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_x, batch_y in dataloader:
        batch, n_ant, nw, nd = batch_x.shape
        flattened = batch_x.reshape(batch * n_ant, 1, nw, nd).to(device)
        logits = model(flattened)
        probs = torch.softmax(logits, dim=1).reshape(batch, n_ant, -1).cpu()

        fused = fuse_batch(probs, n_antennas=n_ant)
        y_true.extend(batch_y.tolist())
        y_pred.extend(fused.tolist())

    return y_true, y_pred


def main(config_path: str, checkpoint_path: str) -> None:
    """Entry point: loads a trained checkpoint and evaluates across all scenarios.

    Args:
        config_path: Path to configs.
        checkpoint_path: Path to a checkpoint saved by train_baseline.py.
    """
    config = load_config(config_path)
    device = get_device()

    data_root=config["paths"]["doppler_traces_dir"]
    output_root=config["paths"]["baseline_output_dir"]

    model = SHARPClassifier(
        n_classes=len(TARGET_CLASSES),
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d", checkpoint.get("epoch", -1))

    results_by_set: dict[str, float] = {}

    _, _, _, s1_test_subset = build_train_val_split(
        data_root,
        window_size=config["doppler"]["stacked_vectors_nw"],
        stride=config["doppler"].get("window_stride"),
    )
    s1_loader = DataLoader(s1_test_subset, batch_size=config["training"]["batch_size"], shuffle=False)
    y_true, y_pred = evaluate_set(model, s1_loader, device)
    results_by_set["S1"] = compute_accuracy(y_true, y_pred)

    for set_id in ("S2", "S3", "S4", "S5", "S6", "S7"):
        test_dataset = build_zero_shot_test_set(
            data_root,
            set_id=set_id,
            window_size=config["doppler"]["stacked_vectors_nw"],
            stride=config["doppler"].get("window_stride"),
        )
        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
        y_true, y_pred = evaluate_set(model, test_loader, device)
        results_by_set[set_id] = compute_accuracy(y_true, y_pred)

    logger.info("=== Per-set accuracy (fused, decision-level) ===")
    for set_id, acc in results_by_set.items():
        logger.info("  %s: %.4f", set_id, acc)

    if "S7" in results_by_set:
        logger.info(
            "S7 vs. paper reference: measured=%.4f, paper=%.4f, diff=%.4f",
            results_by_set["S7"], S7_REFERENCE_ACCURACY, results_by_set["S7"] - S7_REFERENCE_ACCURACY,
        )

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with open(figures_dir / "per_set_accuracy.txt", "w", encoding="utf-8") as f:
        for set_id, acc in results_by_set.items():
            f.write(f"{set_id}\t{acc:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the SHARP baseline from scenario S1 to S7.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/baseline/checkpoints/sharp_baseline_best.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
