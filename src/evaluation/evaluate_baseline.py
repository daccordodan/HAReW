"""Baseline SHARP reproduction: zero-shot evaluation across S1-S7.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.data.doppler_trace_dataset import build_train_val_split, build_zero_shot_test_set
from src.data.label_mapping import TARGET_CLASSES
from src.evaluation.metrics import compute_accuracy_per_activity, compute_confusion_matrix, compute_f1_per_activity
from src.models.decision_fusion import fuse_batch
from src.models.sharp_classifier import SHARPClassifier
from src.utils.colab_utils import get_device
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

from huggingface_hub import hf_hub_download

S7_REFERENCE_ACCURACY = 0.9599 # For comparison with the paper

logger = get_logger(__name__)

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


def main(config_path: str, checkpoint_name: str) -> None:
    """Entry point: loads a trained checkpoint and evaluates across all scenarios.

    Args:
        config_path: Path to configs.
        checkpoint_name: File name of the checkpoint saved on Hugging Face.
    """
    device = get_device()
    config = load_config(config_path)
    model = load_checkpoint_to_model(config,checkpoint_name,device)

    data_root=Path(config["paths"]["doppler_traces_dir"])
    output_root=Path(config["paths"]["baseline_output_dir"])

    accuracy_by_set: dict[str, float] = {}
    accuracy_by_set_pa: dict[str, dict[str, float]] = {}
    fscore_by_set_pa: dict[str, dict[str, float]] = {}
    conf_mat_by_set: dict[str, np.ndarray] = {}

    logger.info("=== Starting evaluation ===")
    logger.info("Evaluated set S1")
    _, _, _, s1_test_subset = build_train_val_split(
        data_root,
        set_id="S1",
        window_size=config["doppler"]["stacked_vectors_nw"],
        stride=config["doppler"].get("window_stride"),
    )
    s1_loader = DataLoader(s1_test_subset, batch_size=config["training"]["batch_size"], shuffle=False)
    y_true, y_pred = evaluate_set(model, s1_loader, device)
    accuracy_by_set_pa["S1"],  accuracy_by_set["S1"] = compute_accuracy_per_activity(y_true, y_pred,TARGET_CLASSES)
    fscore_by_set_pa["S1"] = compute_f1_per_activity(y_true,y_pred,TARGET_CLASSES)

    set_ids=["S2", "S3", "S4", "S5", "S6", "S7"]
    for set_id in set_ids:
        logger.info(f"Evaluated set {set_id}")
        test_dataset = build_zero_shot_test_set(
            data_root,
            set_id=set_id,
            window_size=config["doppler"]["stacked_vectors_nw"],
            stride=config["doppler"].get("window_stride"),
        )
        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
        y_true, y_pred = evaluate_set(model, test_loader, device)
        accuracy_by_set_pa[set_id], accuracy_by_set[set_id] = compute_accuracy_per_activity(y_true, y_pred,TARGET_CLASSES)
        fscore_by_set_pa[set_id] = compute_f1_per_activity(y_true, y_pred, TARGET_CLASSES)
        conf_mat_by_set[set_id] = compute_confusion_matrix(y_true, y_pred, len(TARGET_CLASSES))

    logger.info("=== Per-set accuracy (fused, decision-level) ===")
    for set_id, acc in accuracy_by_set.items():
        logger.info("  %s: %.4f", set_id, acc)

    if "S7" in accuracy_by_set:
        logger.info(
            "S7 vs. paper reference: measured=%.4f, paper=%.4f, diff=%.4f",
            accuracy_by_set["S7"], S7_REFERENCE_ACCURACY, accuracy_by_set["S7"] - S7_REFERENCE_ACCURACY,
        )

    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    set_ids.insert(0,"S1")
    plot_acc_f1_results_pa(accuracy_by_set_pa,fscore_by_set_pa,figures_dir)
    plot_conf_mat_results_pa(conf_mat_by_set, set_ids, TARGET_CLASSES, figures_dir)

    files_dir = output_root / "text"
    write_report_performances(accuracy_by_set, accuracy_by_set_pa, fscore_by_set_pa, files_dir)


def load_checkpoint_to_model(config,checkpoint_name,device):
    model=SHARPClassifier(
        n_classes=len(TARGET_CLASSES),
        nw=config["doppler"]["stacked_vectors_nw"],
        nd=config["doppler"]["velocity_bins_nd"],
    ).to(device)
    checkpoint = torch.load(hf_hub_download(
        repo_id="danieledaccordo/HAReW",
        filename="checkpoints_dir/"+checkpoint_name,
        repo_type="model"
    ), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d", checkpoint.get("epoch", -1))
    return model


def plot_acc_f1_results_pa(accuracy: dict[str, dict[str, float]], fscore: dict[str, dict[str, float]], figures_dir: Path):
    df_acc = pd.DataFrame(accuracy)
    df_f1 = pd.DataFrame(fscore)

    df_combined = "Acc: " + df_acc.round(2).astype(str) + "\nF1: " + df_f1.round(2).astype(str)
    _, ax = plt.subplots(figsize=(6, 2.5))
    ax.set_title('Accuracy and F1-scores of the baseline')
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df_combined.values,
        rowLabels=df_combined.index,
        colLabels=df_combined.columns,
        loc='center',
        cellLoc='center'
    )
    table.scale(1, 2.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold')

    plt.savefig(figures_dir / "accuracy_f1_pa_table.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_conf_mat_results_pa(confmat, set_ids, class_names, figures_dir: Path):
    for set_id in set_ids:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat[set_id], display_labels=class_names)
        disp.plot(cmap='viridis', ax=ax, xticks_rotation='horizontal')

        ax.set_title(f'Confusion Matrix of the baseline: set {set_id}')

        plt.tight_layout()
        plt.savefig(figures_dir / f"conf_mat_{set_id}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)


def write_report_performances(accuracy_by_set, accuracy_by_set_pa, fscore_by_set_pa, files_dir):
    with open(files_dir / "per_set_accuracy.txt", "w", encoding="utf-8") as f:
        for set_id, acc in accuracy_by_set.items():
            f.write(f"{set_id}\t{acc:.4f}\n")

    with open(files_dir / "per_set_accuracy.txt", "w", encoding="utf-8") as f:
        for set_id, acts in accuracy_by_set_pa.items():
            f.write(f"{set_id}\n")
            for act, acc in acts:
                f.write(f"{act}\t{acc:.4f}\n")

    with open(files_dir / "per_set_f1.txt", "w", encoding="utf-8") as f:
        for set_id, acts in fscore_by_set_pa.items():
            f.write(f"{set_id}\n")
            for act, fs in acts:
                f.write(f"{act}\t{fs:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the SHARP baseline from scenario S1 to S7.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/baseline/checkpoints/sharp_baseline_best.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
