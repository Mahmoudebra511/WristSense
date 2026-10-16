#!/usr/bin/env python
"""AAC session quality-check script (project document, Section 5.5).

Checks one session folder (or every session under a root) for:

* row count, duration, measured sampling rate
* gaps larger than 100 ms between samples
* missing values, duplicate timestamps
* sync-marker presence
* value ranges (sanity limits per column)

Prints a human-readable report and (optionally) JSON. Exit code 1 if any
**error** is found; warnings never fail a run unless --strict is given.

    python scripts/quality_check.py data/raw/p01_20261020T141530_laptop
    python scripts/quality_check.py data/raw --json --strict

Reuses :mod:`aacdata.validate` for the structural checks (columns, meta.json,
time ordering) and adds the range / missing / duplicate / sync analysis here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aacdata.columns import ALL_FILES, IMU, LOGGER, VISION  # noqa: E402
from aacdata.validate import (  # noqa: E402
    Report,
    validate_session,
)

# ---------------------------------------------------------------------------
# Sanity limits per numeric column (warnings only; hardware-dependent).
# ---------------------------------------------------------------------------
RANGE_LIMITS: dict[str, dict[str, tuple[float | None, float | None]]] = {
    "imu.csv": {
        "ax": (-20.0, 20.0),
        "ay": (-20.0, 20.0),
        "az": (-20.0, 20.0),
        "gx": (-2500.0, 2500.0),
        "gy": (-2500.0, 2500.0),
        "gz": (-2500.0, 2500.0),
    },
    "logger.csv": {
        "keys": (0.0, 10000.0),
        "mouse_moves": (0.0, 100000.0),
        "mouse_clicks": (0.0, 10000.0),
        "win_switches": (0.0, 1000.0),
        "time_since_input_s": (0.0, 86400.0),
    },
    "vision.csv": {
        "head_yaw": (-180.0, 180.0),
        "head_pitch": (-180.0, 180.0),
        "eye_closure": (0.0, 1.0),
        "conf": (0.0, 1.0),
    },
}


def _nullish(raw: str) -> bool:
    return raw.strip().lower() in {"", "nan", "na", "null", "none"}


def _coerce_float(raw: str):
    try:
        f = float(raw)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except ValueError:
        return None


def _coerce_int(raw: str):
    try:
        return int(raw)
    except ValueError:
        f = _coerce_float(raw)
        if f is not None and f.is_integer():
            return int(f)
        return None


def analyze_stream(path: Path, stream) -> dict:
    """Return a dict of quality metrics for one periodic CSV stream."""
    info: dict = {
        "file": stream.filename,
        "rows": 0,
        "rate_hz": None,
        "max_gap_ms": None,
        "gaps_over_budget": 0,
        "duplicate_timestamps": 0,
        "missing_per_column": {},
        "ranges": {},
        "out_of_range": [],
    }

    if not path.exists():
        info["missing_file"] = True
        return info

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            info["empty"] = True
            return info
        rows = [r for r in reader if r and any(c.strip() for c in r)]

    info["rows"] = len(rows)
    if not rows:
        return info

    index = {name: header.index(name) for name in header if name in stream.column_names}
    nominal = stream.rate_hz or 1.0
    budget_ms = max(100.0, 1.5 * (1000.0 / nominal))
    missing: dict[str, int] = {c.name: 0 for c in stream.columns}
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    prev_ts: int | None = None
    dup = 0
    gaps: list[int] = []

    for row in rows:
        ts_raw = row[index["ts_ms"]] if "ts_ms" in index and index["ts_ms"] < len(row) else ""
        ts = _coerce_int(ts_raw)
        if ts is not None:
            if prev_ts is not None:
                gap = ts - prev_ts
                if gap == 0:
                    dup += 1
                elif gap < 0:
                    pass  # non-monotonic reported by validator
                else:
                    gaps.append(gap)
            prev_ts = ts

        for cname, cidx in index.items():
            if cname == "ts_ms":
                continue
            col = next((c for c in stream.columns if c.name == cname), None)
            raw = row[cidx] if cidx < len(row) else ""
            if _nullish(raw):
                missing[cname] += 1
                continue
            if col is not None and col.dtype == "float":
                val = _coerce_float(raw)
                if val is None:
                    missing[cname] += 1
                    continue
                mins[cname] = val if cname not in mins else min(mins[cname], val)
                maxs[cname] = val if cname not in maxs else max(maxs[cname], val)
            elif col is not None and col.dtype == "int":
                val = _coerce_int(raw)
                if val is None:
                    missing[cname] += 1
                    continue
                fval = float(val)
                mins[cname] = fval if cname not in mins else min(mins[cname], fval)
                maxs[cname] = fval if cname not in maxs else max(maxs[cname], fval)

    info["missing_per_column"] = {k: v for k, v in missing.items() if v}
    info["duplicate_timestamps"] = dup
    info["gaps_over_budget"] = sum(1 for g in gaps if g > budget_ms)
    info["max_gap_ms"] = max(gaps) if gaps else None

    if ts is not None and prev_ts is not None and len(rows) >= 2:
        span_ms = prev_ts - (int(rows[0][index["ts_ms"]]) if index.get("ts_ms") is not None and index["ts_ms"] < len(rows[0]) else 0)
        # better: use first and last ts
        first_ts = _coerce_int(rows[0][index["ts_ms"]]) if "ts_ms" in index else None
        last_ts = _coerce_int(rows[-1][index["ts_ms"]]) if "ts_ms" in index else None
        if first_ts is not None and last_ts is not None and last_ts > first_ts:
            info["duration_s"] = round((last_ts - first_ts) / 1000.0, 3)
            info["rate_hz"] = round((len(rows) - 1) / ((last_ts - first_ts) / 1000.0), 3)

    limits = RANGE_LIMITS.get(stream.filename, {})
    for cname in sorted(mins):
        lo, hi = limits.get(cname, (None, None))
        info["ranges"][cname] = {"min": mins[cname], "max": maxs[cname]}
        if lo is not None and mins[cname] < lo:
            info["out_of_range"].append(f"{cname} min {mins[cname]} < {lo}")
        if hi is not None and maxs[cname] > hi:
            info["out_of_range"].append(f"{cname} max {maxs[cname]} > {hi}")

    return info


def analyze_labels(path: Path) -> dict:
    info: dict = {"file": "labels.csv", "rows": 0, "intervals": 0,
                  "overlaps": 0, "bad_task_state": 0, "unknown_activity": 0}
    if not path.exists():
        info["missing"] = True
        return info
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            info["empty"] = True
            return info
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    info["rows"] = len(rows)
    if not rows:
        return info

    idx = {name: header.index(name) for name in header if name in header}
    intervals = []
    allowed = {"ON_TASK", "DISTRACTED", "AWAY", "BREAK"}
    from aacdata.enums import PROVISIONAL_ACTIVITIES
    for row in rows:
        def g(name):
            return row[idx[name]] if name in idx and idx[name] < len(row) else ""
        s = _coerce_int(g("start_ms"))
        e = _coerce_int(g("end_ms"))
        state = g("task_state").strip()
        act = g("activity").strip()
        if s is not None and e is not None:
            intervals.append((s, e))
        if state and state not in allowed:
            info["bad_task_state"] += 1
        if act and act not in PROVISIONAL_ACTIVITIES:
            info["unknown_activity"] += 1
    intervals.sort()
    info["intervals"] = len(intervals)
    info["overlaps"] = sum(1 for a, b in zip(intervals, intervals[1:]) if b[0] < a[1])
    return info


def quality_check(session_dir: str | Path, require_labels: bool = False,
                  strict: bool = False) -> Report:
    session_dir = Path(session_dir)
    report = validate_session(session_dir, require_labels=require_labels)

    # Extra per-stream metrics (re-read CSVs; files are small).
    meta_path = session_dir / "meta.json"
    streams_meta: dict = {}
    if meta_path.exists():
        try:
            import json as _json
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            streams_meta = meta.get("streams") if isinstance(meta.get("streams"), dict) else {}
        except Exception:
            streams_meta = {}

    declared = list(streams_meta.items())
    for name, filename in declared:
        stream = ALL_FILES.get(name)
        if stream is None:
            continue
        info = analyze_stream(session_dir / filename, stream)
        report.info[f"{name}_rows"] = info["rows"]
        report.info[f"{name}_rate_hz"] = info.get("rate_hz")
        report.info[f"{name}_max_gap_ms"] = info.get("max_gap_ms")
        report.info[f"{name}_gaps_over_budget"] = info["gaps_over_budget"]
        report.info[f"{name}_duplicate_timestamps"] = info["duplicate_timestamps"]
        if info["missing_per_column"]:
            report.info[f"{name}_missing"] = info["missing_per_column"]
        if info["out_of_range"]:
            for o in info["out_of_range"]:
                report.warn("quality.range", f"{filename}: {o}", filename)
        if info["gaps_over_budget"]:
            report.warn("quality.gap", f"{filename}: {info['gaps_over_budget']} gap(s) > 100 ms", filename)
        if info["duplicate_timestamps"]:
            report.warn("quality.dup_ts", f"{filename}: {info['duplicate_timestamps']} duplicate timestamp(s)", filename)

    # Required IMU stream present?
    if "imu" not in streams_meta and not bool(report.issues and any(i.code == "meta.streams" for i in report.issues)):
        report.warn("quality.missing_stream", "imu stream not declared in meta.json", "meta.json")

    # Labels
    label_info = analyze_labels(session_dir / "labels.csv")
    if label_info.get("missing"):
        report.info["labels"] = "absent"
    else:
        report.info["label_rows"] = label_info["rows"]
        report.info["label_intervals"] = label_info["intervals"]
        report.info["label_overlaps"] = label_info["overlaps"]
        if label_info["bad_task_state"]:
            report.warn("quality.labels", f"{label_info['bad_task_state']} row(s) with invalid task_state", "labels.csv")
        if label_info["unknown_activity"]:
            report.warn("quality.labels", f"{label_info['unknown_activity']} row(s) with unknown activity", "labels.csv")
        if label_info["overlaps"]:
            report.warn("quality.labels", f"{label_info['overlaps']} overlapping label interval(s)", "labels.csv")

    # Sync markers
    import json as _json
    try:
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        markers = meta.get("sync_markers_ms") or []
        report.info["sync_markers"] = len(markers)
        if not markers and not meta.get("partial"):
            report.warn("quality.sync", "no sync markers recorded in meta.json", "meta.json")
    except Exception:
        pass

    # Fail on warnings in strict mode.
    if strict and report.warnings:
        for w in report.warnings:
            report.error(w.code, w.message, w.where)
        report.issues = report.errors  # type: ignore[assignment]

    return report


def _collect(target: Path) -> list[Path]:
    if (target / "meta.json").exists():
        return [target]
    sessions = sorted(p.parent for p in target.glob("*/meta.json"))
    return sessions or [target]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="session folder or a folder containing sessions")
    ap.add_argument("--require-labels", action="store_true", help="missing labels.csv is an error")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors (exit code 1)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        ap.error(f"path does not exist: {target}")

    sessions = _collect(target)
    reports = [quality_check(s, require_labels=args.require_labels, strict=args.strict) for s in sessions]

    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False))
    else:
        for r in reports:
            print(r.format())
            print()

    failed = [r for r in reports if not r.ok]
    if failed:
        print(f"{len(failed)} of {len(reports)} session(s) failed quality check")
        return 1
    print(f"all {len(reports)} session(s) passed quality check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
