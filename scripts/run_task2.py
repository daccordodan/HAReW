"""CLI: runs Task 2 (cross-environment/day robustness) domain adaptation.

Usage:
    python scripts/run_task2.py --config config/task2_cross_environment.yaml

NOTE: this task's implementation is currently a scaffold (see
src/tasks/task2_cross_environment/domain_adaptation.py) pending the choice
of domain-adaptation strategy -- running this will raise
NotImplementedError until that work is done.
"""

from __future__ import annotations

import argparse

from src.tasks.task2_cross_environment.domain_adaptation import train_domain_adaptation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 2: cross-environment/day domain adaptation.")
    parser.add_argument("--config", type=str, default="config/task2_cross_environment.yaml")
    args = parser.parse_args()
    train_domain_adaptation(args.config)
