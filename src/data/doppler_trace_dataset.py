"""PyTorch Dataset for pre-computed Doppler traces, organized using the SHARP paper.

This module's job is: 
(1) discover files, 
(2) maps the different activities via label_mapping.py, 
(3) window each per-antenna stream into (Nw, ND) chunks,
(4) hand back PyTorch tensors.

"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import os, os.path
from torch.utils.data import Dataset

from src.data.label_mapping import is_in_scope, raw_to_class_index

_FILENAME_RE = re.compile(
    r"^(?P<set_num>S\d+)(?P<letter>[a-z])_(?P<activity>[A-Za-z0-9]+)_stream_(?P<antenna>\d)\.txt$"  #set_num,letter,activity,antenna just in case are needed in future
)

DEFAULT_NW = 340   # Doppler vectors per input (~2s)
DEFAULT_NANT = 4   # monitor antennas
DEFAULT_STRIDE = 1 # default stride

TRAIN_ONLY_SET = "S1"


@dataclass(frozen=True)
class StreamFileInfo:
    """Parsed metadata for a single per-antenna stream file.

    Attributes:
        path: Full path to the .txt file.
        set_id: scenario number only.
        repetition: Repetition letter.
        activity_code: Raw activity code.
        antenna_idx: Antenna/stream index, 0-3.
    """

    path: Path
    set_id: str
    repetition: str
    activity_code: str
    antenna_idx: int

def discover_stream_files(root_dir: str | Path) -> list[StreamFileInfo]:
    """Parses every matching stream file in root_dir.

    Args:
        root_dir: Directory containing set-repetition sub-folders.

    Returns:
        List of StreamFileInfo for every recognized file.
    """

    root = Path(root_dir)
    infos = []

    for txt_path in sorted(root.rglob("*.txt")):
        match = _FILENAME_RE.match(txt_path.name)
        if match is not None:
            info = StreamFileInfo(
                path=txt_path,
                set_id=match.group("set_num"),
                repetition=match.group("letter"),
                activity_code=match.group("activity"),
                antenna_idx=int(match.group("antenna")),
            )
            infos.append(info)

    return infos

def count_stream_files(root_dir: str | Path) -> int:
    n=0
    root = Path(root_dir)
    for txt_path in sorted(root.rglob("*.txt")):
        match = _FILENAME_RE.match(txt_path.name)
        if match is not None:
            n+=1

    return n


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


class DopplerTraceDataset(Dataset):
    """PyTorch Dataset over, per-antenna Doppler traces.

    Each sample is one (Nant, Nw, ND) tensor paired with a single
    integer class label. decision_fusion.py will fuse per-antenna 
    predictions for the same time window.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sets_to_include: tuple[str, ...],
        window_size: int = DEFAULT_NW,
        stride: int = DEFAULT_STRIDE,
        n_antennas: int = DEFAULT_NANT,
        temporal_split: Literal["train", "val", "test", "all"] = "all",
    ) -> None:
        """Indexes and windows all in-scope samples for the requested sets
        and creates training, validation and test sets from the original set.

        Args:
            root_dir: Directory containing requested set(s).
            sets_to_include: Which scenario numbers to load.
            window_size: Nw, default 340 (~2 seconds).
            stride: Step between windows.
            n_antennas: Expected number of antenna streams, default 4.
            temporal_split: Requested set to be produced
        """
        self.root_dir = Path(root_dir)
        self.sets_to_include = sets_to_include
        self.window_size = window_size
        self.stride = stride or window_size
        self.n_antennas = n_antennas
        self.temporal_split = temporal_split

        self._recordings: list[np.ndarray] = []
        self._window_indices: list[tuple[int, int, int]] = []

        self._meta: list[dict] = []
        self._excluded_counts: dict[str, int] = {}
        self._corrupt_files: list[str] = []

        self._build_index()

    def _build_index(self) -> None:
        all_files=discover_stream_files(self.root_dir)
        
        groups: dict[tuple[str, str, str], dict[int, Path]] = {} # This was to group files by (set_id, repetition, activity_code) -> {antenna_idx: path}
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
            per_antenna_streams = []
            for i in range(self.n_antennas):
                per_antenna_streams.append(_load_pickled_array(antenna_paths[i]))

            min_len = min(s.shape[0] for s in per_antenna_streams)

            start_idx, end_idx = self.evaluate_temp_split(min_len)

            slice_len = end_idx - start_idx
            if slice_len < self.window_size:
                continue

            per_antenna_streams = [s[start_idx:end_idx] for s in per_antenna_streams]

            stacked_recording = np.stack([s[:min_len] for s in per_antenna_streams], axis=0)
            rec_idx = len(self._recordings)
            self._recordings.append(stacked_recording)

            class_idx = raw_to_class_index(activity_code)
            
            n_windows = 1 + (slice_len - self.window_size) // self.stride           # Calculate how many valid windows can be extracted

            for w in range(n_windows):
                start_time = w * self.stride
                
                self._window_indices.append((rec_idx, start_time, class_idx))
                self._meta.append(          #Need to look at this, if it could be useful for the tasks
                    {
                        "set_id": set_id,
                        "repetition": repetition,
                        "activity_code": activity_code,
                        "window_idx": w,
                        "split": self.temporal_split,
                    }
                )

    def __len__(self) -> int:
        return len(self._window_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Returns (doppler_trace, label), this slices windows on the fly.

        Args:
            idx: Sample index.

        Returns:
            doppler_trace: PyTorch tensor.
            label: Integer target class index.
        """
        rec_idx, start_time, label = self._window_indices[idx]
        end_time = start_time + self.window_size

        window = self._recordings[rec_idx][:, start_time:end_time, :]
        return torch.tensor(window, dtype=torch.float32), label

    def evaluate_temp_split(self,min_len) -> tuple[int, int]:
        """Evaluates the starting and ending index for the requested sets.

        Args:
            min_len: Length of the shortest window.
        """
        if self.temporal_split != "all":
            gap = self.window_size 

            train_end = int(min_len * 0.6)
            val_start = train_end + gap
            val_end = val_start + int(min_len * 0.2)
            test_start = val_end + gap

            if self.temporal_split == "train":
                return 0, train_end
            elif self.temporal_split == "val":
                return val_start, val_end
            elif self.temporal_split == "test":
                return test_start, min_len
        else:
            return 0, min_len

    def summary(self) -> dict:
        """This is used for debug: produces a small report."""
        return {
            "root_dir": str(self.root_dir),
            "sets_included": self.sets_to_include,
            "n_samples": len(self._samples),
            "excluded_raw_activity_counts": dict(self._excluded_counts),
            "corrupt_files": list(self._corrupt_files),
        }


def build_train_val_split(
    root_dir: str | Path,
    set_id: Literal["S1","S2", "S3", "S4", "S5", "S6", "S7"] = TRAIN_ONLY_SET,
    window_size: int = DEFAULT_NW,
    stride: int = DEFAULT_STRIDE,
) -> tuple[DopplerTraceDataset, torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """Builds the train/val/test split.

    Args:
        root_dir: Directory containing set-repetition sub-folders.
        window_size: Nw, default 340.
        stride: Window stride.

    Returns:
        (full_dataset, train_subset, val_subset, s1_test_subset).
    """

    train_dataset = DopplerTraceDataset(
        root_dir, 
        sets_to_include=(set_id,), 
        window_size=window_size, 
        stride=stride,
        temporal_split="train"
    )
    val_dataset = DopplerTraceDataset(
        root_dir, 
        sets_to_include=(set_id,), 
        window_size=window_size, 
        stride=stride,
        temporal_split="val"
    )
    test_dataset = DopplerTraceDataset(
        root_dir, 
        sets_to_include=(set_id,), 
        window_size=window_size, 
        stride=stride,
        temporal_split="test"
    )

    train_subset = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
    val_subset = torch.utils.data.Subset(val_dataset, range(len(val_dataset)))
    test_subset = torch.utils.data.Subset(test_dataset, range(len(test_dataset)))

    full_dataset = DopplerTraceDataset(
        root_dir, 
        sets_to_include=(set_id,), 
        window_size=window_size, 
        stride=stride
    )

    return full_dataset, train_subset, val_subset, test_subset


def build_zero_shot_test_set(
    root_dir: str | Path,
    set_id: Literal["S1","S2", "S3", "S4", "S5", "S6", "S7"],
    window_size: int = DEFAULT_NW,
    stride: int = DEFAULT_STRIDE,
) -> DopplerTraceDataset:
    """Builds a test-only dataset.

    Args:
        root_dir: Directory containing set-repetition sub-folders.
        set_id: Scenario numbers, used in order to undersstand which scenario use to build the test set.
        window_size: Nw, default 340.
        stride: Window stride.

    Returns:
        DopplerTraceDataset restricted to the requested set.
    """

    return DopplerTraceDataset(root_dir, sets_to_include=(set_id,), window_size=window_size, stride=stride)
