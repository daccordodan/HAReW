"""Baseline SHARP reproduction: zero-shot evaluation across S1-S7.
"""

from __future__ import annotations

from pathlib import Path

import torch

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.data.label_mapping import TARGET_CLASSES
from src.models.sharp_classifier import SHARPClassifier

from huggingface_hub import hf_hub_download

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
