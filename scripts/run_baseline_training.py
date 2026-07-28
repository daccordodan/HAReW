"""CLI: trains the SHARP baseline classifier on set S1 (PyTorch).

Usage (local or Colab):
    python scripts/run_baseline_training.py --config config/base_config.yaml

    
"""

from __future__ import annotations

import argparse

from src.training.train_baseline import main as train_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SHARP baseline on S1.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    args = parser.parse_args()
    train_main(args.config)
