"""Unit tests for src/data/label_mapping.py -- the resolved 5-class decision."""

from __future__ import annotations

import pytest

from src.data.label_mapping import (
    EXCLUDED_RAW_CODES,
    RAW_TO_TARGET,
    TARGET_CLASSES,
    is_in_scope,
    raw_to_class_index,
)


def test_target_classes_are_5():
    assert TARGET_CLASSES == ["E", "W", "R", "J", "L"]


def test_j1_and_j2_merge_into_j():
    assert RAW_TO_TARGET["J1"] == "J"
    assert RAW_TO_TARGET["J2"] == "J"
    assert raw_to_class_index("J1") == raw_to_class_index("J2")


def test_h_and_s_are_excluded():
    assert "H" in EXCLUDED_RAW_CODES
    assert "S" in EXCLUDED_RAW_CODES
    assert not is_in_scope("H")
    assert not is_in_scope("S")


def test_in_scope_codes():
    for code in ("E", "W", "R", "J1", "J2", "L"):
        assert is_in_scope(code)


def test_raw_to_class_index_raises_for_out_of_scope():
    with pytest.raises(KeyError):
        raw_to_class_index("H")
