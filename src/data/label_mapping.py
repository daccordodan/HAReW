"""Centralizes the raw-activity-code -> target-class mapping decision.

DECISION LOG (resolved with the project owner, see docs/PROJECT_STATUS.md):
    The actual uploaded Doppler-trace data (S1a/S1b) contains 8 raw activity
    codes: E, H, J1, J2, L, R, S, W. Only 5 of these are in scope for the
    baseline recognition task, matching Paper 2's primary SHARP evaluation
    (Sec. 6.2, Table 3: 4 activities + empty room):

        Target classes (5):  E (empty), W (walking), R (running),
                              J (jumping, merging J1+J2), L (sitting still)

        Excluded raw codes:  H, S
            - "S" = standing still, part of Paper 1's full 7-activity letter
              set but NOT part of SHARP's primary 5-class task.
            - "H" has no definition in Paper 1 or Paper 2's letter-code
              table (W/R/J/L/S/C/G/E) -- it is a recording present in this
              dataset drop that is explicitly out of scope for the baseline.
              This is a NEW, still-open naming discrepancy (distinct from --
              but related to -- the previously logged "H in a .h5 filename"
              ambiguity); flagged in docs/PROJECT_STATUS.md.

    J1 and J2 are two recorded variants of "jumping" (e.g. two-footed vs.
    one-footed, or in-place vs. traveling -- not specified in the source
    papers) and are merged into a single "J" class for the baseline, per
    the project owner's explicit decision.

All training/evaluation code MUST route activity-code interpretation through
this module -- never hardcode raw codes elsewhere, so a future change to the
label scope (e.g. moving to the 8-class extended task, Paper 2 Sec. 6.7)
only requires editing this one file.
"""

from __future__ import annotations

# Ordered target class list -- index position IS the model's output index.
# Keep this order stable once training starts; changing it invalidates
# any existing checkpoint's output-layer semantics.
TARGET_CLASSES: list[str] = ["E", "W", "R", "J", "L"]

# Raw on-disk activity code -> target class name.
# Codes not present as keys here are OUT OF SCOPE for the baseline task.
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

# Raw codes present in the dataset but explicitly excluded from the baseline.
EXCLUDED_RAW_CODES: frozenset[str] = frozenset({"H", "S"})

TARGET_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(TARGET_CLASSES)}
INDEX_TO_TARGET: dict[int, str] = {i: name for name, i in TARGET_TO_INDEX.items()}


def is_in_scope(raw_activity_code: str) -> bool:
    """Returns True if a raw activity code belongs to the baseline task.

    Args:
        raw_activity_code: Code parsed from a filename, e.g. "J1", "H".

    Returns:
        True if the code maps to one of TARGET_CLASSES, False if it is in
        EXCLUDED_RAW_CODES (or otherwise unrecognized).
    """
    return raw_activity_code in RAW_TO_TARGET


def raw_to_class_index(raw_activity_code: str) -> int:
    """Maps a raw activity code directly to its target class index.

    Args:
        raw_activity_code: Code parsed from a filename, e.g. "J1", "W".

    Returns:
        Integer class index into TARGET_CLASSES.

    Raises:
        KeyError: If the code is out of scope (check is_in_scope() first).
    """
    target_name = RAW_TO_TARGET[raw_activity_code]
    return TARGET_TO_INDEX[target_name]
