"""CLI: evaluates the trained SHARP baseline across sets S1-S7 (PyTorch).

Usage (local or Colab):
    python scripts/run_baseline_evaluation.py \\
        --config config/base_config.yaml \\
        --checkpoint outputs/baseline/checkpoints/sharp_baseline_best.pt
"""

from __future__ import annotations

import argparse

from src.evaluation.evaluate_baseline import main as evaluate_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the SHARP baseline across S1-S7.")
    parser.add_argument("--config", type=str, default="config/base_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/baseline/checkpoints/sharp_baseline_best.pt")
    args = parser.parse_args()
    evaluate_main(args.config, args.checkpoint)
