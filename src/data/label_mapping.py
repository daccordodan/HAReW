"""
    Centralizes the mapping activity-code -> class.
"""

from __future__ import annotations

TARGET_CLASSES: list[str] = ["E", "W", "R", "J", "L"] #index position IS the model's output index.

RAW_TO_TARGET: dict[str, str] = {
    "E": "E",
    "W": "W",
    "R": "R",
    "R1": "R",
    "J1": "J",
    "J2": "J",
    "J3": "J",
    "L": "L",
    "L1": "L",
    "L2": "L",
}

TARGET_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(TARGET_CLASSES)}
INDEX_TO_TARGET: dict[int, str] = {i: name for name, i in TARGET_TO_INDEX.items()}


def is_in_scope(raw_activity_code: str) -> bool:
    """Returns True if a raw activity code belongs to the baseline task.

    Args:
        raw_activity_code: Code parsed from a parsed filename.

    Returns:
        True if the code maps to one of TARGET_CLASSES.
    """
    return raw_activity_code in RAW_TO_TARGET


def raw_to_class_index(raw_activity_code: str) -> int:
    """Maps a raw activity code directly to its target class index.

    Args:
        raw_activity_code: Code parsed from a filename.

    Returns:
        Integer class index into TARGET_CLASSES.
    """
    target_name = RAW_TO_TARGET[raw_activity_code]
    return TARGET_TO_INDEX[target_name]
