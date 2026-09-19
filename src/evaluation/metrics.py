"""Evaluation metrics matching Paper 2's reported benchmark tables.

Source: Paper 2, Sec. 6.2-6.7, Tables 3-6, Figs. 7-9.

Target correctness gates (Paper 2, Table 3):
    - Mean accuracy > 95% across all test sets (S1-S7).
    - S7 (worst case: new environment + new person) accuracy ~= 95.99%.
    - Dominant confusion pattern: walking misclassified as running (and
      vice versa) -- track this explicitly, not just overall accuracy.

NOTE: this project's baseline uses a 5-class label scope (E, W, R, J, L --
see src/data/label_mapping.py), which differs from Paper 2's original
4-activities+empty-room task only in naming (SHARP's "sitting" == this
project's "L"). Accuracy/F1 numbers are directly comparable; per-class
walking/running confusion tracking below still applies unchanged.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

S7_REFERENCE_ACCURACY = 0.9599  # Paper 2, Table 3 -- worst-case benchmark


def compute_accuracy_per_activity(y_true: list[int], y_pred: list[int], class_names: list[str]) -> float:
    """Computes overall per-activity classification accuracy.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.

    Returns:
        Accuracy as a float in [0, 1]. Returns 0.0 for an empty input
        rather than raising, since evaluate_baseline.py may encounter
        empty test sets for sets not yet uploaded (see docs/PROJECT_STATUS.md).
    """

    accuracy_pa: dict[str, float] = {}
    if len(y_true) == 0:
        return 0.0

    correct=0
    correct_pa=list()
    counts_pa=list()
    for t, p in zip(y_true, y_pred):
        counts_pa[t]+=1
        if t==p:
            correct_pa[t]+=1
            correct+=1
            
    for class_idx, class_name in enumerate(class_names):
        accuracy_pa[class_name]=correct_pa[class_idx]/counts_pa[class_idx]

    correct /= len(y_pred)

    return accuracy_pa, correct


def compute_f1_per_activity(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, float]:
    """Computes per-activity F1 scores.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Ordered list of activity class names (index-aligned
            with the class indices used in y_true/y_pred -- see
            src/data/label_mapping.TARGET_CLASSES).

    Returns:
        Dict mapping activity name -> F1 score.
    """
    f1_scores: dict[str, float] = {}
    for class_idx, class_name in enumerate(class_names):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == class_idx and p == class_idx)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != class_idx and p == class_idx)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == class_idx and p != class_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[class_name] = f1
    return f1_scores


def compute_confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int) -> np.ndarray:
    """Computes the confusion matrix, to inspect the walking/running confusion mode.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        n_classes: Total number of activity classes.

    Returns:
        Confusion matrix of shape (n_classes, n_classes); rows = true class,
        columns = predicted class.
    """
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def report_per_set_accuracy(results_by_set: dict[str, float]) -> str:
    """Formats per-set (S1-S7) accuracy alongside the Paper 2 reference values.

    Args:
        results_by_set: Dict mapping set ID ("S1".."S7") to measured accuracy.

    Returns:
        A human-readable multi-line report string.
    """
    lines = ["Per-set accuracy:"]
    for set_id, acc in results_by_set.items():
        if set_id == "S7":
            lines.append(f"  {set_id}: {acc:.4f}  (paper reference: {S7_REFERENCE_ACCURACY:.4f})")
        else:
            lines.append(f"  {set_id}: {acc:.4f}")
    mean_acc = sum(results_by_set.values()) / len(results_by_set) if results_by_set else 0.0
    lines.append(f"Mean across sets: {mean_acc:.4f}  (paper reference: >0.95)")
    return "\n".join(lines)
