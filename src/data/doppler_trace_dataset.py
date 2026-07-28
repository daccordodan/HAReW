"""PyTorch Dataset for pre-computed Doppler traces, organized per the SHARP paper.

Source: Paper 2, Sec. 5.2 ("Dataset Acquisition and Organization") + Table 1.

CONFIRMED ON-DISK FORMAT (verified directly against the project owner's
uploaded sample, S1a/S1b):
    - Folder per set+repetition, e.g. "S1a", "S1b" (SetNumber + Letter).
    - Filename pattern: "{Set}{Letter}_{ActivityCode}_stream_{AntennaIdx}.txt"
      e.g. "S1a_W_stream_0.txt".
    - AntennaIdx in {0,1,2,3} -- one file per monitor antenna (Nant=4).
    - Each file is a **pickled NumPy float64 array** of shape
      (n_time_steps, ND), ND=100 (velocity bins). n_time_steps varies per
      activity/recording length and is NOT yet windowed into Nw=340 chunks
      -- that windowing happens here in this Dataset, not upstream.
    - Values are already normalized to roughly [0.06, 1.0].

NO raw-CFR preprocessing is required or performed here (per project
constraint: the owner already has these Doppler traces) -- this module's
only job is: (1) discover files, (2) filter to in-scope activities via
label_mapping.py, (3) window each per-antenna stream into (Nw, ND) chunks,
(4) hand back PyTorch tensors.

Train/test protocol (Paper 2, Sec. 6.1): only set S1 is used for
training/validation (60% / 20% split); the remaining 20% of S1 plus ALL of
S2-S7 are reserved for (zero-shot) testing. This module enforces that split
so no other code path can accidentally leak S2-S7 into training.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.label_mapping import is_in_scope, raw_to_class_index

# Filename pattern confirmed against the real uploaded sample.
_FILENAME_RE = re.compile(
    r"^(?P<set_num>S\d+)(?P<letter>[a-z])_(?P<activity>[A-Za-z0-9]+)_stream_(?P<antenna>\d)\.txt$"
)

DEFAULT_NW = 340   # stacked Doppler vectors per classifier input (~2s window)
DEFAULT_ND = 100   # velocity bins (verified against real files: shape[1] == 100)
DEFAULT_NANT = 4   # monitor antennas (verified: stream_0..stream_3 present)

TRAIN_ONLY_SET = "S1"


@dataclass(frozen=True)
class StreamFileInfo:
    """Parsed metadata for a single per-antenna stream file.

    Attributes:
        path: Full path to the .txt (pickled ndarray) file.
        set_id: e.g. "S1" (set number only, no repetition letter).
        repetition: Repetition letter, e.g. "a", "b".
        activity_code: Raw on-disk activity code, e.g. "W", "J1", "H".
        antenna_idx: Antenna/stream index, 0-3.
    """

    path: Path
    set_id: str
    repetition: str
    activity_code: str
    antenna_idx: int


def parse_filename(path: Path) -> StreamFileInfo | None:
    """Parses a stream filename into structured metadata.

    Args:
        path: Path to a candidate .txt stream file.

    Returns:
        StreamFileInfo if the filename matches the expected pattern,
        otherwise None (caller should skip non-matching files).
    """
    match = _FILENAME_RE.match(path.name)
    if match is None:
        return None
    return StreamFileInfo(
        path=path,
        set_id=match.group("set_num"),
        repetition=match.group("letter"),
        activity_code=match.group("activity"),
        antenna_idx=int(match.group("antenna")),
    )


def discover_stream_files(root_dir: str | Path) -> list[StreamFileInfo]:
    """Walks root_dir and parses every matching stream file, in scope or not.

    Args:
        root_dir: Directory containing set-repetition sub-folders (e.g.
            containing "S1a/", "S1b/", "S6a/", ...).

    Returns:
        List of StreamFileInfo for every recognized file (including
        out-of-scope activities -- filtering happens at Dataset build time
        so counts/logging can still report what was excluded and why).
    """
    root = Path(root_dir)
    infos = []
    for txt_path in sorted(root.rglob("*.txt")):
        info = parse_filename(txt_path)
        if info is not None:
            infos.append(info)
    return infos


def _load_pickled_array(path: Path) -> np.ndarray:
    """Loads one pickled float64 NumPy array from disk.

    Args:
        path: Path to a .txt file containing a pickled ndarray.

    Returns:
        Array of shape (n_time_steps, ND).
    """
    with open(path, "rb") as f:
        arr = pickle.load(f)
    return arr


def _window_stream(stream: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Slides a fixed-size window over the time axis of a Doppler stream.

    Args:
        stream: Array of shape (n_time_steps, ND).
        window_size: Nw, number of stacked Doppler vectors per window.
        stride: Step size between consecutive windows (< window_size gives
            overlapping windows; == window_size gives non-overlapping).

    Returns:
        Array of shape (n_windows, window_size, ND). Trailing samples that
        don't fill a complete window are dropped.
    """
    n_time_steps, nd = stream.shape
    if n_time_steps < window_size:
        return np.empty((0, window_size, nd), dtype=stream.dtype)
    n_windows = 1 + (n_time_steps - window_size) // stride
    windows = np.stack(
        [stream[i * stride : i * stride + window_size] for i in range(n_windows)],
        axis=0,
    )
    return windows


class DopplerTraceDataset(Dataset):
    """PyTorch Dataset over pre-computed, per-antenna Doppler traces.

    Each sample is one (Nant, Nw, ND) tensor -- all 4 antennas' windows for
    the same time slice of the same recording -- paired with a single
    integer class label. Keeping antennas together (rather than flattening
    them into separate samples) is required for decision_fusion.py, which
    fuses per-antenna predictions for the same physical time window.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sets_to_include: tuple[str, ...],
        window_size: int = DEFAULT_NW,
        stride: int | None = None,
        n_antennas: int = DEFAULT_NANT,
    ) -> None:
        """Indexes and windows all in-scope samples for the requested sets.

        Args:
            root_dir: Directory containing set-repetition sub-folders.
            sets_to_include: Which set numbers to load, e.g. ("S1",) for
                training/validation, or ("S1","S2",...,"S7") for full
                zero-shot evaluation. Repetition letters (a/b/c) within an
                included set are all loaded.
            window_size: Nw, default 340 (~2s window, per Paper 2 Table 2).
            stride: Step between windows; defaults to window_size (no
                overlap). Use a smaller stride to oversample short
                recordings (e.g. jumping, which has far fewer raw samples
                than walking/running in the real data).
            n_antennas: Expected number of antenna streams, default 4.
        """
        self.root_dir = Path(root_dir)
        self.sets_to_include = sets_to_include
        self.window_size = window_size
        self.stride = stride or window_size
        self.n_antennas = n_antennas

        self._samples: list[torch.Tensor] = []
        self._labels: list[int] = []
        self._meta: list[dict] = []
        self._excluded_counts: dict[str, int] = {}
        self._corrupt_files: list[str] = []

        self._build_index()

    def _build_index(self) -> None:
        all_files = discover_stream_files(self.root_dir)

        # Group files by (set_id, repetition, activity_code) -> {antenna_idx: path}
        groups: dict[tuple[str, str, str], dict[int, Path]] = {}
        for info in all_files:
            if info.set_id not in self.sets_to_include:
                continue
            if not is_in_scope(info.activity_code):
                self._excluded_counts[info.activity_code] = (
                    self._excluded_counts.get(info.activity_code, 0) + 1
                )
                continue
            key = (info.set_id, info.repetition, info.activity_code)
            groups.setdefault(key, {})[info.antenna_idx] = info.path

        for (set_id, repetition, activity_code), antenna_paths in groups.items():
            '''
            if len(antenna_paths) != self.n_antennas:
                # Incomplete antenna set for this recording -- skip rather
                # than silently zero-pad, since a missing antenna stream is
                # a data problem, not something to paper over.
                continue
            '''
            per_antenna_streams = []
            group_is_corrupt = False
            for i in range(self.n_antennas):
                try:
                    per_antenna_streams.append(_load_pickled_array(antenna_paths[i]))
                except (EOFError, pickle.UnpicklingError, OSError) as exc:
                    # Real-world data issue (e.g. a 0-byte/truncated file from
                    # a partial upload) -- skip this whole recording rather
                    # than crash, but record it so summary()/logs surface it.
                    self._corrupt_files.append(f"{antenna_paths[i]} ({exc})")
                    group_is_corrupt = True
            if group_is_corrupt:
                print("Group corrupted")
                continue

            # All antennas of the same recording should share the same
            # timeline length; trim to the shortest just in case of a
            # trailing-sample mismatch between antennas.
            min_len = min(s.shape[0] for s in per_antenna_streams)
            per_antenna_streams = [s[:min_len] for s in per_antenna_streams]

            per_antenna_windows = [
                _window_stream(s, self.window_size, self.stride) for s in per_antenna_streams
            ]
            n_windows = min(w.shape[0] for w in per_antenna_windows)
            if n_windows == 0:
                continue

            class_idx = raw_to_class_index(activity_code)
            for w in range(n_windows):
                stacked = np.stack(
                    [per_antenna_windows[a][w] for a in range(self.n_antennas)], axis=0
                )  # shape (Nant, Nw, ND)
                self._samples.append(torch.from_numpy(stacked).float())
                self._labels.append(class_idx)
                self._meta.append(
                    {
                        "set_id": set_id,
                        "repetition": repetition,
                        "activity_code": activity_code,
                        "window_idx": w,
                    }
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Returns (doppler_trace, label).

        Args:
            idx: Sample index.

        Returns:
            doppler_trace: FloatTensor of shape (Nant, Nw, ND).
            label: Integer target class index (see label_mapping.TARGET_CLASSES).
        """
        return self._samples[idx], self._labels[idx]

    def summary(self) -> dict:
        """Returns a small report of included/excluded/corrupt counts, for sanity checks."""
        return {
            "root_dir": str(self.root_dir),
            "sets_included": self.sets_to_include,
            "n_samples": len(self._samples),
            "excluded_raw_activity_counts": dict(self._excluded_counts),
            "corrupt_files": list(self._corrupt_files),
        }


def build_train_val_split(
    root_dir: str | Path,
    window_size: int = DEFAULT_NW,
    stride: int | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[DopplerTraceDataset, torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """Builds the S1-only train/val/(held-out) test split, per Paper 2 Sec. 6.1.

    Args:
        root_dir: Directory containing set-repetition sub-folders.
        window_size: Nw, default 340.
        stride: Window stride; defaults to window_size.
        train_frac: Fraction of S1 windows for training (default 0.6).
        val_frac: Fraction of S1 windows for validation (default 0.2).
            The remainder (default 0.2) becomes the S1 held-out test split.
        seed: Random seed for the shuffle (reproducibility).

    Returns:
        (full_s1_dataset, train_subset, val_subset, s1_test_subset).
    """
    s1_dataset = DopplerTraceDataset(
        root_dir, sets_to_include=(TRAIN_ONLY_SET,), window_size=window_size, stride=stride
    )
    n = len(s1_dataset)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_subset = torch.utils.data.Subset(s1_dataset, train_idx)
    val_subset = torch.utils.data.Subset(s1_dataset, val_idx)
    test_subset = torch.utils.data.Subset(s1_dataset, test_idx)
    return s1_dataset, train_subset, val_subset, test_subset


def build_zero_shot_test_set(
    root_dir: str | Path,
    set_id: Literal["S2", "S3", "S4", "S5", "S6", "S7"],
    window_size: int = DEFAULT_NW,
    stride: int | None = None,
) -> DopplerTraceDataset:
    """Builds a test-only dataset for a single zero-shot generalization set.

    Args:
        root_dir: Directory containing set-repetition sub-folders.
        set_id: One of "S2".."S7" -- never "S1" (use build_train_val_split
            for S1, to guarantee the split logic stays in one place).
        window_size: Nw, default 340.
        stride: Window stride; defaults to window_size.

    Returns:
        DopplerTraceDataset restricted to the requested set.
    """
    if set_id == TRAIN_ONLY_SET:
        raise ValueError("S1 must go through build_train_val_split(), not this function.")
    return DopplerTraceDataset(root_dir, sets_to_include=(set_id,), window_size=window_size, stride=stride)
