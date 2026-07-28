"""Unit tests for src/data/doppler_trace_dataset.py.

Uses small synthetic pickled arrays (not the real uploaded data, which is
not available in CI) to test parsing, windowing, exclusion, and the
corrupt-file handling path added after finding a real 0-byte file in the
project owner's actual upload (S1b_L_stream_3.txt).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from src.data.doppler_trace_dataset import (
    DopplerTraceDataset,
    build_train_val_split,
    discover_stream_files,
    parse_filename,
)


def _write_stream(tmp_path, folder, activity, antenna_idx, n_time_steps, nd=100):
    folder_path = tmp_path / folder
    folder_path.mkdir(exist_ok=True)
    file_path = folder_path / f"{folder}_{activity}_stream_{antenna_idx}.txt"
    arr = np.random.RandomState(0).rand(n_time_steps, nd).astype(np.float64)
    with open(file_path, "wb") as f:
        pickle.dump(arr, f)
    return file_path


@pytest.fixture
def synthetic_root(tmp_path):
    """Builds a small synthetic S1a/S1b tree covering in-scope, excluded,
    and corrupt-file cases."""
    # In-scope activities, all 4 antennas present, enough length for >=1 window
    for activity in ("W", "R", "J1", "J2", "L", "E"):
        for ant in range(4):
            _write_stream(tmp_path, "S1a", activity, ant, n_time_steps=700)

    # Excluded activities (must never appear in the dataset)
    for activity in ("H", "S"):
        for ant in range(4):
            _write_stream(tmp_path, "S1a", activity, ant, n_time_steps=700)

    # S1b: one corrupt (0-byte) file for antenna 3, activity L
    for activity in ("W",):
        for ant in range(4):
            _write_stream(tmp_path, "S1b", activity, ant, n_time_steps=700)
    corrupt_path = tmp_path / "S1b" / "S1b_L_stream_3.txt"
    for ant in range(3):
        _write_stream(tmp_path, "S1b", "L", ant, n_time_steps=700)
    corrupt_path.touch()  # 0 bytes, mirrors the real corrupted file found

    return tmp_path


def test_parse_filename():
    from pathlib import Path

    info = parse_filename(Path("S1a_W_stream_0.txt"))
    assert info is not None
    assert info.set_id == "S1"
    assert info.repetition == "a"
    assert info.activity_code == "W"
    assert info.antenna_idx == 0


def test_parse_filename_rejects_non_matching():
    from pathlib import Path

    assert parse_filename(Path("readme.txt")) is None


def test_discover_stream_files(synthetic_root):
    files = discover_stream_files(synthetic_root)
    # 6 in-scope activities * 4 antennas (S1a) + 2 excluded * 4 antennas (S1a)
    # + 1 activity * 4 antennas (S1b, W) + 1 activity * 4 antennas (S1b, L, incl. corrupt)
    assert len(files) == 6 * 4 + 2 * 4 + 4 + 4


def test_dataset_excludes_h_and_s(synthetic_root):
    ds = DopplerTraceDataset(synthetic_root, sets_to_include=("S1",), window_size=340)
    summary = ds.summary()
    assert summary["excluded_raw_activity_counts"].get("H") == 4
    assert summary["excluded_raw_activity_counts"].get("S") == 4
    assert len(ds) > 0


def test_dataset_skips_corrupt_file_without_crashing(synthetic_root):
    ds = DopplerTraceDataset(synthetic_root, sets_to_include=("S1",), window_size=340)
    summary = ds.summary()
    assert any("S1b_L_stream_3" in f for f in summary["corrupt_files"])


def test_j1_j2_merge_into_single_class(synthetic_root):
    from src.data.label_mapping import TARGET_TO_INDEX

    ds = DopplerTraceDataset(synthetic_root, sets_to_include=("S1",), window_size=340)
    j_index = TARGET_TO_INDEX["J"]
    j_labels = [label for label in ds._labels if label == j_index]
    # Both J1 and J2 windows should be present under the single "J" index.
    assert len(j_labels) > 0


def test_sample_shape(synthetic_root):
    ds = DopplerTraceDataset(synthetic_root, sets_to_include=("S1",), window_size=340)
    x, y = ds[0]
    assert x.shape == (4, 340, 100)
    assert isinstance(y, int)


def test_train_val_split_sizes(synthetic_root):
    full_ds, train_ds, val_ds, test_ds = build_train_val_split(
        synthetic_root, window_size=340, train_frac=0.6, val_frac=0.2
    )
    assert len(train_ds) + len(val_ds) + len(test_ds) == len(full_ds)
