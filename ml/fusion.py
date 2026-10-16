"""AAC early fusion table (project document, Section 5.5).

Aligns IMU window features with 30-second aggregates from the laptop logger
and webcam vision into a single row per decision window:

    python -m ml.fusion data/processed/ --out fused.csv

For every 30-second decision window:

* IMU features: mean of all feature windows falling inside the window
* Logger (1 Hz): mean of numeric columns, mode of ``category``
* Vision (1 Hz): mean of numeric columns, share of seconds with a face
* Label: majority ``task_state`` among labelled windows inside the window

The result is one row per 30 s, ready for the baseline model or for tuning
the state machine by replay.

Requires numpy (declared in requirements-ml.txt).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------
ID_COLS_FUSION = ["start_ms", "end_ms", "task_state", "n_imu_windows"]

LABELS = ("ON_TASK", "DISTRACTED", "AWAY")


def feature_columns(header: list[str]) -> list[str]:
    """Return feature column names in stable sorted order (non-ID columns)."""
    id_cols = {"start_ms", "end_ms", "window_s", "sample_rate_hz",
               "task_state", "activity"}
    return sorted([c for c in header if c not in id_cols])


# ---------------------------------------------------------------------------
# Decision window
# ---------------------------------------------------------------------------
DECISION_WINDOW_S = 30.0


@dataclass(frozen=True)
class DecisionWindow:
    start_ms: int
    end_ms: int  # exclusive


def decision_windows(
    start_ms: int,
    end_ms: int,
    window_s: float = DECISION_WINDOW_S,
) -> list[DecisionWindow]:
    """Split ``[start_ms, end_ms) into consecutive ``window_s``-second chunks."""
    span_ms = int(round(window_s * 1000))
    if span_ms <= 0:
        raise ValueError("window_s must be positive")
    out: list[DecisionWindow] = []
    t = start_ms
    while t < end_ms:
        out.append(DecisionWindow(start_ms=t, end_ms=min(t + span_ms, end_ms)))
        t += span_ms
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mode(values: list[str]) -> str:
    if not values:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


def _majority(labels: list[str]) -> str:
    """Majority vote over task_state labels; unknown labels are ignored."""
    votes = [l for l in labels if l in LABELS]
    if not votes:
        return "unknown"
    return Counter(votes).most_common(1)[0][0]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Per-stream aggregation into decision windows
# ---------------------------------------------------------------------------
def _aggregate_periodic(
    rows: list[dict[str, str]],
    ts_index: int,
    numeric_cols: list[str],
    category_cols: list[str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Aggregate a periodic stream (logger/vision) into decision-window buckets.

    Returns a dict keyed by window start_ms -> aggregated row dict.
    """
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            ts = int(row["ts_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        bucket_key = ts  # caller groups into decision windows separately
        buckets.setdefault(bucket_key, []).append(row)

    out: dict[int, dict[str, Any]] = {}
    for ts, group in buckets.items():
        agg: dict[str, Any] = {"ts_ms": ts, "n_samples": len(group)}
        for col in numeric_cols:
            vals: list[float] = []
            for r in group:
                try:
                    v = float(r[col])
                    if not (np.isnan(v) or np.isinf(v)):
                        vals.append(v)
                except (KeyError, ValueError, TypeError):
                    pass
            agg[col] = _mean(vals)
        if category_cols:
            for col in category_cols:
                vals = [r[col] for r in group if r.get(col)]
                agg[col] = _mode(vals)
        out[ts] = agg
    return out


def aggregate_logger(
    rows: list[dict[str, str]],
) -> dict[int, dict[str, Any]]:
    """Aggregate 1 Hz logger rows into per-second buckets."""
    return _aggregate_periodic(
        rows,
        ts_index=0,
        numeric_cols=[
            "keys", "mouse_moves", "mouse_clicks", "win_switches",
            "time_since_input_s",
        ],
        category_cols=["category"],
    )


def aggregate_vision(
    rows: list[dict[str, str]],
) -> dict[int, dict[str, Any]]:
    """Aggregate 1 Hz vision rows into per-second buckets."""
    return _aggregate_periodic(
        rows,
        ts_index=0,
        numeric_cols=["head_yaw", "head_pitch", "eye_closure", "conf"],
        category_cols=None,
    )


def _add_face_share(
    vision_buckets: dict[int, dict[str, Any]],
    vision_rows: list[dict[str, str]],
) -> None:
    """Add face_present_share (fraction of seconds with a face) per bucket."""
    present: dict[int, int] = {}
    total: dict[int, int] = {}
    for row in vision_rows:
        try:
            ts = int(row["ts_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        total[ts] = total.get(ts, 0) + 1
        if row.get("face_present", "0").strip() in {"1", "1.0", "true", "True"}:
            present[ts] = present.get(ts, 0) + 1
    for ts, agg in vision_buckets.items():
        agg["face_present_share"] = (
            present.get(ts, 0) / total[ts] if total.get(ts, 0) > 0 else 0.0
        )


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FusedRow:
    start_ms: int
    end_ms: int
    task_state: str
    imu_features: dict[str, float]
    logger_agg: dict[str, Any]
    vision_agg: dict[str, Any]


def fuse_session(
    session_dir: Path,
    decision_window_s: float = DECISION_WINDOW_S,
) -> list[dict[str, Any]]:
    """Produce one fused row per decision window for a single session.

    Reads ``features.csv`` (from ml.pipeline), ``logger.csv``,
    ``vision.csv`` and ``labels.csv`` (optional) from ``session_dir``.
    """
    features_csv = session_dir / "features.csv"
    if not features_csv.exists():
        return []

    # --- read feature rows (output of ml.pipeline) -------------------------
    feature_rows = _read_csv_rows(features_csv)
    if not feature_rows:
        return []
    feats = feature_columns(list(feature_rows[0].keys()))

    # --- read periodic streams ---------------------------------------------
    logger_path = session_dir / "logger.csv"
    vision_path = session_dir / "vision.csv"
    labels_path = session_dir / "labels.csv"

    logger_rows = _read_csv_rows(logger_path) if logger_path.exists() else []
    vision_rows = _read_csv_rows(vision_path) if vision_path.exists() else []
    label_rows = _read_csv_rows(labels_path) if labels_path.exists() else []

    logger_buckets = aggregate_logger(logger_rows)
    vision_buckets = aggregate_vision(vision_rows)
    _add_face_share(vision_buckets, vision_rows)

    # --- label intervals ----------------------------------------------------
    label_intervals: list[tuple[int, int, str]] = []
    for row in label_rows:
        try:
            s = int(row["start_ms"])
            e = int(row["end_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        st = row.get("task_state", "").strip()
        if st and e > s:
            label_intervals.append((s, e, st))

    # --- overall time span --------------------------------------------------
    all_ts: list[int] = []
    for row in feature_rows:
        try:
            all_ts.append(int(row["start_ms"]))
            all_ts.append(int(row["end_ms"]))
        except (KeyError, ValueError, TypeError):
            pass
    for row in logger_rows:
        try:
            all_ts.append(int(row["ts_ms"]))
        except (KeyError, ValueError, TypeError):
            pass
    for row in vision_rows:
        try:
            all_ts.append(int(row["ts_ms"]))
        except (KeyError, ValueError, TypeError):
            pass
    if not all_ts:
        return []
    span_start = min(all_ts)
    span_end = max(all_ts)

    windows = decision_windows(span_start, span_end, window_s=decision_window_s)

    fused: list[dict[str, Any]] = []
    for dw in windows:
        # --- IMU features: mean of feature windows fully inside decision window
        imu_vals: dict[str, list[float]] = {c: [] for c in feats}
        imu_labels: list[str] = []
        for row in feature_rows:
            try:
                ws = int(row["start_ms"])
                we = int(row["end_ms"])
            except (KeyError, ValueError, TypeError):
                continue
            # window fully inside this decision window
            if ws >= dw.start_ms and we <= dw.end_ms:
                for c in feats:
                    try:
                        imu_vals[c].append(float(row[c]))
                    except (KeyError, ValueError, TypeError):
                        pass
                st = row.get("task_state", "").strip()
                if st:
                    imu_labels.append(st)

        imu_features = {c: _mean(v) for c, v in imu_vals.items()}

        # --- logger / vision: aggregate seconds inside decision window ------
        logger_secs: list[dict[str, Any]] = []
        vision_secs: list[dict[str, Any]] = []
        for ts, agg in logger_buckets.items():
            if dw.start_ms <= ts < dw.end_ms:
                logger_secs.append(agg)
        for ts, agg in vision_buckets.items():
            if dw.start_ms <= ts < dw.end_ms:
                vision_secs.append(agg)

        logger_agg = {
            "n_logger_seconds": len(logger_secs),
        }
        if logger_secs:
            for c in ("keys", "mouse_moves", "mouse_clicks", "win_switches",
                      "time_since_input_s"):
                logger_agg[c] = _mean([s.get(c, 0.0) for s in logger_secs])
            cats = [s.get("category", "unknown") for s in logger_secs]
            logger_agg["category"] = _mode(cats)

        vision_agg = {
            "n_vision_seconds": len(vision_secs),
        }
        if vision_secs:
            for c in ("head_yaw", "head_pitch", "eye_closure", "conf",
                      "face_present_share"):
                vision_agg[c] = _mean([s.get(c, 0.0) for s in vision_secs])

        # --- label: majority task_state among IMU windows inside -----------
        task_state = _majority(imu_labels) if imu_labels else "unknown"

        row: dict[str, Any] = {
            "start_ms": dw.start_ms,
            "end_ms": dw.end_ms,
            "task_state": task_state,
            "n_imu_windows": len(imu_labels),
        }
        for c in feats:
            row[f"imu_{c}"] = imu_features.get(c, 0.0)
        for c, v in logger_agg.items():
            row[f"logger_{c}"] = v
        for c, v in vision_agg.items():
            row[f"vision_{c}"] = v
        fused.append(row)

    return fused


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Stable column order: identifiers, label, then imu/logger/vision features.
    id_cols = ["start_ms", "end_ms", "task_state", "n_imu_windows"]
    other_cols = sorted(
        c for c in rows[0].keys() if c not in id_cols
    )
    fieldnames = id_cols + other_cols
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fuse_all(
    root: str | Path,
    decision_window_s: float = DECISION_WINDOW_S,
) -> dict[str, list[dict[str, Any]]]:
    """Fuse every session folder under ``root`` (each containing features.csv)."""
    root = Path(root)
    sessions: dict[str, list[dict[str, Any]]] = {}
    targets: list[Path]
    if (root / "features.csv").exists():
        targets = [root]
    else:
        targets = sorted(p for p in root.iterdir() if p.is_dir())
    for session_dir in targets:
        fused = fuse_session(session_dir, decision_window_s=decision_window_s)
        if fused:
            sessions[session_dir.name] = fused
    return sessions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing session folders with features.csv (+ logger.csv, vision.csv, labels.csv)")
    ap.add_argument("--out", required=True, help="output fused CSV path")
    ap.add_argument("--window-s", type=float, default=DECISION_WINDOW_S,
                    help="decision window length in seconds (default: 30)")
    args = ap.parse_args(argv)

    sessions = fuse_all(args.root, decision_window_s=args.window_s)
    if not sessions:
        print("ERROR: no sessions with features.csv found under", args.root, file=sys.stderr)
        return 2

    all_rows: list[dict[str, Any]] = []
    for session_id, rows in sessions.items():
        for row in rows:
            row["session_id"] = session_id
            all_rows.append(row)

    # Move session_id to front.
    id_cols = ["session_id", "start_ms", "end_ms", "task_state", "n_imu_windows"]
    other_cols = sorted(c for c in all_rows[0].keys() if c not in id_cols)
    fieldnames = id_cols + other_cols
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"fused {len(sessions)} session(s) -> {len(all_rows)} decision-window rows")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
