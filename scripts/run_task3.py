"""CLI: runs Task 3 (person identification) training.

Usage:
    python scripts/run_task3.py --config config/task3_person_id.yaml
"""

from __future__ import annotations

import argparse

from src.tasks.task3_person_id.person_classifier import train_person_identification
from src.utils.logger import get_logger
from src.utils.colab_utils import get_device
from src.utils.config_loader import load_config

from pathlib import Path
from huggingface_hub import HfApi

logger = get_logger(__name__)
api = HfApi()

def main(config_path: str, local: bool = False) -> None:
    """Entry point: loads config, builds dataset/model, runs the full training loop.

    Args:
        config_path: Path to config/base_config.yaml.
        local: Specify if it should be run with local AMD GPU 
    """

    device = get_device()
    config = load_config(config_path)

    n_people = config["model"]["n_people"]

    output_root=Path(config["paths"]["task_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name="pretraining_t13_best.pt"
    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / checkpoint_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 3: person identification.")
    parser.add_argument("--config", type=str, default="config/task3_person_id.yaml")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files instead of Hugging Face.")
    args = parser.parse_args()
    train_person_identification(args.config, local=args.local)
