"""Label alignment for windowed data (project document, Section 5.5).

Each sliding window takes the label of the interval it falls in, and
transitional windows (the first and last 10 seconds of every block) are
dropped so the model never trains on ambiguous boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelInterval:
    start_ms: int
    end_ms: int  # exclusive
    activity: str
    task_state: str


def load_labels(rows: list[dict]) -> list[LabelInterval]:
    """Parse ``labels.csv`` rows into sorted, non-overlapping intervals."""
    intervals: list[LabelInterval] = []
    for row in rows:
        intervals.append(
            LabelInterval(
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                activity=str(row["activity"]),
                task_state=str(row["task_state"]),
            )
        )
    intervals.sort(key=lambda iv: (iv.start_ms, iv.end_ms))
    return intervals


def window_label(
    window_start_ms: int,
    window_end_ms: int,
    intervals: list[LabelInterval],
) -> LabelInterval | None:
    """Return the label interval containing the window's centre point.

    If the window spans multiple intervals, the one covering the centre wins.
    Returns ``None`` if no interval covers the window at all.
    """
    centre = (window_start_ms + window_end_ms) / 2.0
    best: LabelInterval | None = None
    best_overlap = 0
    for iv in intervals:
        overlap_start = max(window_start_ms, iv.start_ms)
        overlap_end = min(window_end_ms, iv.end_ms)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = iv
    if best is None or best_overlap <= 0:
        return None
    # Prefer the interval that contains the centre point.
    for iv in intervals:
        if iv.start_ms <= centre < iv.end_ms:
            return iv
    return best


def drop_transition_windows(
    windows: list[tuple[int, int]],
    intervals: list[LabelInterval],
    transition_s: float = 10.0,
) -> list[tuple[int, int, str]]:
    """Return windows that are fully inside a single activity block.

    A window is dropped if its start falls within the first ``transition_s``
    of a block or its end falls within the last ``transition_s`` of a block.
    Returns ``(start_ms, end_ms, task_state)`` for kept windows.
    """
    transition_ms = transition_s * 1000.0
    kept: list[tuple[int, int, str]] = []

    # Index blocks by task_state so transitions are per-block.
    for start_ms, end_ms in windows:
        # Find the block containing the window centre.
        centre = (start_ms + end_ms) / 2.0
        block = None
        for iv in intervals:
            if iv.start_ms <= centre < iv.end_ms:
                block = iv
                break
        if block is None:
            continue

        # Drop if too close to the start or end of this block.
        if start_ms < block.start_ms + transition_ms:
            continue
        if end_ms > block.end_ms - transition_ms:
            continue

        kept.append((start_ms, end_ms, block.task_state))

    return kept


def align_features_with_labels(
    windows: list[tuple[int, int]],
    feature_rows: list[dict],
    intervals: list[LabelInterval],
    transition_s: float = 10.0,
) -> list[dict]:
    """Attach ``task_state`` (and ``activity``) to each feature row.

    Windows without a covering label, or inside transition zones, are dropped.
    """
    kept = drop_transition_windows(windows, intervals, transition_s=transition_s)
    kept_set = {(s, e) for s, e, _ in kept}
    label_by_window = {(s, e): st for s, e, st in kept}

    aligned: list[dict] = []
    for (start_ms, end_ms), feat in zip(windows, feature_rows):
        if (start_ms, end_ms) not in kept_set:
            continue
        row = dict(feat)
        row["task_state"] = label_by_window[(start_ms, end_ms)]
        # Find activity for the same window.
        centre = (start_ms + end_ms) / 2.0
        for iv in intervals:
            if iv.start_ms <= centre < iv.end_ms:
                row["activity"] = iv.activity
                break
        aligned.append(row)
    return aligned
