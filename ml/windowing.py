"""Windowing for IMU streams.

4-second windows with 50% overlap (project document, Section 5.5), then
aggregated into a 30-to-60-second decision window.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Window:
    start_ms: int
    end_ms: int
    start_idx: int  # inclusive index into the sample arrays
    end_idx: int    # exclusive index


def sliding_windows(
    n_samples: int,
    window_s: float,
    sample_rate_hz: float,
    overlap: float = 0.5,
) -> list[Window]:
    """Return every sliding window over ``n_samples`` samples.

    Parameters
    ----------
    n_samples:
        Total number of samples in the stream.
    window_s:
        Window length in seconds (e.g. 4.0).
    sample_rate_hz:
        Nominal sampling rate (used only to convert seconds -> samples).
    overlap:
        Fractional overlap between consecutive windows (0.5 = 50%).
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    window_samples = max(1, int(round(window_s * sample_rate_hz)))
    step = max(1, int(round(window_samples * (1.0 - overlap))))

    windows: list[Window] = []
    start = 0
    while start + window_samples <= n_samples:
        windows.append(
            Window(
                start_ms=0,  # filled by caller from the timestamp array
                end_ms=0,
                start_idx=start,
                end_idx=start + window_samples,
            )
        )
        start += step
    return windows


def assign_timestamps(windows: list[Window], ts_ms: list[int]) -> None:
    """Fill ``start_ms``/``end_ms`` on each window from the sample timestamps."""
    for w in windows:
        w.start_ms = ts_ms[w.start_idx]
        w.end_ms = ts_ms[w.end_idx - 1]


def decision_windows(
    windows: list[Window],
    decision_window_s: float,
) -> list[tuple[int, int]]:
    """Aggregate sliding windows into coarser decision windows.

    Returns a list of ``(start_ms, end_ms)`` tuples spanning the full
    recording in ``decision_window_s``-second chunks (e.g. 30–60 s).
    """
    if not windows:
        return []
    span_ms = windows[-1].end_ms - windows[0].start_ms
    dw = int(round(decision_window_s * 1000))
    out: list[tuple[int, int]] = []
    t = windows[0].start_ms
    while t < windows[-1].end_ms:
        out.append((t, min(t + dw, windows[-1].end_ms)))
        t += dw
    return out
