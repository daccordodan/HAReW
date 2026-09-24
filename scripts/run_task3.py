"""CLI: runs Task 3 (person identification) training.

Usage:
    python scripts/run_task3.py --config config/task3_person_id.yaml
"""

from __future__ import annotations

import argparse

from src.tasks.task3_person_id.person_classifier import train_person_identification

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 3: person identification.")
    parser.add_argument("--config", type=str, default="config/task3_person_id.yaml")
    args = parser.parse_args()
    train_person_identification(args.config)
