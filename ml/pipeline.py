"""Feature pipeline — one command, deterministic output (project doc §5.5).

    python -m ml.pipeline data/raw/p01_20261020T141530_laptop --out features.csv

Reads ``imu.csv`` + ``labels.csv`` + ``meta.json`` from a session folder,
windows the IMU stream (4 s @ 50% overlap), computes the feature set,
aligns human labels, drops transition windows, and writes one CSV row per
window. Requires numpy (declared in requirements-ml.txt).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aacdata.columns import IMU, LABELS  # noqa: E402
from aacdata.session import Session  # noqa: E402
from ml.windowing import sliding_windows, assign_timestamps  # noqa: E402
from ml.features import compute_features, FeatureVector  # noqa: E402
from ml.labels import load_labels, align_features_with_labels  # noqa: E402

WINDOW_S = 4.0
OVERLAP = 0.5
TRANSITION_S = 10.0


def read_imu(session_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (accel (n,3), gyro (n,3), ts_ms (n,)) from imu.csv."""
    path = session_dir / IMU.filename
    ts, ax, ay, az, gx, gy, gz = [], [], [], [], [], [], []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts.append(int(row["ts_ms"]))
            ax.append(float(row["ax"]))
            ay.append(float(row["ay"]))
            az.append(float(row["az"]))
            gx.append(float(row["gx"]))
            gy.append(float(row["gy"]))
            gz.append(float(row["gz"]))
    accel = np.column_stack([ax, ay, az]).astype(float)
    gyro = np.column_stack([gx, gy, gz]).astype(float)
    return accel, gyro, np.array(ts, dtype=int)


def read_label_rows(session_dir: Path) -> list[dict]:
    path = session_dir / LABELS.filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def run_pipeline(
    session_dir: str | Path,
    window_s: float = WINDOW_S,
    overlap: float = OVERLAP,
    transition_s: float = TRANSITION_S,
) -> list[dict]:
    """Run the full feature pipeline for one session.

    Returns a list of feature dicts (one per kept window), each containing
    all features plus ``task_state``, ``activity``, ``start_ms``, ``end_ms``.
    """
    session_dir = Path(session_dir)
    accel, gyro, ts_ms = read_imu(session_dir)
    label_rows = read_label_rows(session_dir)
    intervals = load_labels(label_rows)

    # Sample rate from meta.json if available, else 50 Hz default.
    meta_path = session_dir / "meta.json"
    sample_rate_hz = 50.0
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        rates = meta.get("sampling_rate_hz") or {}
        sample_rate_hz = float(rates.get("imu", 50.0))

    # --- windowing ---------------------------------------------------------
    windows = sliding_windows(len(ts_ms), window_s, sample_rate_hz, overlap=overlap)
    assign_timestamps(windows, ts_ms.tolist())
    window_spans = [(w.start_ms, w.end_ms) for w in windows]

    # --- features per window ----------------------------------------------
    feature_rows: list[dict] = []
    dt = 1.0 / sample_rate_hz
    for w in windows:
        a = accel[w.start_idx:w.end_idx]
        g = gyro[w.start_idx:w.end_idx]
        t = ts_ms[w.start_idx:w.end_idx]
        fv = compute_features(a, g, t, sample_rate_hz=sample_rate_hz)
        row = fv.as_dict()
        row["start_ms"] = int(w.start_ms)
        row["end_ms"] = int(w.end_ms)
        row["window_s"] = float(window_s)
        row["sample_rate_hz"] = float(sample_rate_hz)
        feature_rows.append(row)

    # --- label alignment + transition drop --------------------------------
    aligned = align_features_with_labels(
        window_spans, feature_rows, intervals, transition_s=transition_s
    )
    return aligned


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable column order: identifiers first, then features, then labels.
    id_cols = ["start_ms", "end_ms", "window_s", "sample_rate_hz", "task_state", "activity"]
    feature_cols = [c for c in rows[0].keys() if c not in id_cols]
    # Sort feature columns for determinism.
    feature_cols = sorted(feature_cols)
    fieldnames = id_cols + feature_cols
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", help="session folder containing imu.csv + labels.csv + meta.json")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--window-s", type=float, default=WINDOW_S, help="window length in seconds")
    ap.add_argument("--overlap", type=float, default=OVERLAP, help="window overlap fraction")
    ap.add_argument("--transition-s", type=float, default=TRANSITION_S, help="transition drop (s)")
    args = ap.parse_args(argv)

    rows = run_pipeline(args.session_dir, args.window_s, args.overlap, args.transition_s)
    write_csv(rows, Path(args.out))
    print(f"wrote {len(rows)} window rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
