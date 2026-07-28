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

from src.tasks.task1_cross_subject.contrastive_encoder import train_contrastive_pretraining

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1: cross-subject contrastive pretraining.")
    parser.add_argument("--config", type=str, default="config/task1_cross_subject.yaml")
    args = parser.parse_args()
    train_contrastive_pretraining(args.config)
