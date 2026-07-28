"""Google Colab compatibility helpers (project constraint: must run on Colab).

Handles the two things that differ between a local run and a Colab run:
    1. Detecting whether we're actually inside Colab.
    2. Optionally mounting Google Drive so data/doppler_traces/ and
       outputs/ can persist across sessions (Colab's local disk is
       ephemeral and is wiped when the runtime disconnects).

Nothing in src/ or scripts/ should hardcode a Colab-specific path directly
-- always go through get_data_root() / get_output_root() so the same code
runs unmodified locally or in Colab.
"""

from __future__ import annotations

import sys
from pathlib import Path


def in_colab() -> bool:
    """Returns True if running inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def mount_drive(mount_point: str = "/content/drive") -> Path | None:
    """Mounts Google Drive if running in Colab; no-op otherwise.

    Args:
        mount_point: Where to mount Drive inside the Colab VM.

    Returns:
        Path to the mounted Drive root, or None if not in Colab.
    """
    if not in_colab():
        return None
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point)
    return Path(mount_point)


def get_data_root(local_default: str = "data/doppler_traces", drive_subpath: str | None = None) -> Path:
    """Resolves the Doppler-trace data root for the current environment.

    Args:
        local_default: Path used when running locally (relative to repo root).
        drive_subpath: If running in Colab and Drive is mounted, the
            subpath under "My Drive" where the data lives, e.g.
            "nndl_wifi_har/doppler_traces". If None, falls back to
            local_default even inside Colab (e.g. data uploaded directly
            to the Colab VM's local disk instead of Drive).

    Returns:
        Resolved Path to the data root.
    """
    if in_colab() and drive_subpath is not None:
        drive_root = mount_drive()
        if drive_root is not None:
            return drive_root / "My Drive" / drive_subpath
    return Path(local_default)


def get_output_root(local_default: str = "outputs", drive_subpath: str | None = None) -> Path:
    """Resolves the outputs root (checkpoints/logs/figures) for the current environment.

    Args:
        local_default: Path used when running locally.
        drive_subpath: If running in Colab and Drive is mounted, the
            subpath under "My Drive" to persist outputs (recommended, since
            Colab's local disk does not survive a runtime restart).

    Returns:
        Resolved Path to the outputs root.
    """
    if in_colab() and drive_subpath is not None:
        drive_root = mount_drive()
        if drive_root is not None:
            out = drive_root / "My Drive" / drive_subpath
            out.mkdir(parents=True, exist_ok=True)
            return out
    Path(local_default).mkdir(parents=True, exist_ok=True)
    return Path(local_default)


def get_device() -> str:
    """Returns "cuda" if a GPU is available (e.g. Colab GPU runtime), else "cpu"."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
