"""Loads and validates the project's YAML configuration file.

Centralizes every paper-derived constant (Nant, M, Nw, ND, N, lambda, MCS,
Tc, etc.) so that source modules never hardcode these values directly --
see config/config.yaml for the canonical defaults and GLOSSARY_AND_VARIABLES.md
for their definitions/sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Loads the YAML config file into a plain dict.

    Args:
        config_path: Path to config/config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
