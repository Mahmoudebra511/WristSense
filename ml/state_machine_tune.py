"""AAC state machine parameter tuning (project document, Section 5.6).

Grid-searches grace_s, confirm_s, alert_delay_s, away_threshold_s across
recorded sessions and reports the combination that minimizes false alerts
per hour while keeping detection delay acceptable.

    python -m ml.state_machine_tune data/processed/ --fused fused.csv \
        --grace-s 30,60,90,120 --confirm-s 10,20,30 \
        --alert-delay-s 60,90,120,180 --away-threshold-s 20,30,60 \
        --max-delay-s 30 --out reports/tune.json

The tuner replays every session with every parameter combination, computes
false alerts per hour and average detection delay, and selects the best
combination: lowest false-alert rate among configs whose detection delay is
within the allowed maximum.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.replay import replay  # noqa: E402
from ml.state_machine import StateMachineConfig  # noqa: E402

LABELS = ("ON_TASK", "DISTRACTED", "AWAY")


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def rows_to_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert fused CSV rows (or event dicts) into replay events."""
    events: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("task_state", "")).strip()
        ts_ms = row.get("ts_ms")
        if ts_ms is None:
            ts_ms = row.get("start_ms")
        events.append(
            {
                "ts_ms": int(ts_ms),
                "model_state": label if label in LABELS else "ON_TASK",
                "face_present": None,
                "away_signal": label == "AWAY",
                "break_declared": label == "BREAK",
                "ground_truth": label,
            }
        )
    return events


def _detection_delays(
    events: list[dict[str, Any]], alerts: list[dict[str, Any]]
) -> list[float]:
    """Delay from start of each DISTRACTED ground-truth block to first alert inside it."""
    delays: list[float] = []
    in_block = False
    block_start_ms = 0
    for ev in events:
        gt = str(ev.get("ground_truth", "")).strip()
        if gt == "DISTRACTED" and not in_block:
            in_block = True
            block_start_ms = int(ev["ts_ms"])
        if gt != "DISTRACTED" and in_block:
            in_block = False
            block_end_ms = int(ev["ts_ms"])
            block_alerts = [
                a for a in alerts if block_start_ms <= a["ts_ms"] <= block_end_ms
            ]
            if block_alerts:
                first = min(block_alerts, key=lambda a: a["ts_ms"])
                delays.append((first["ts_ms"] - block_start_ms) / 1000.0)
    if in_block and events:
        block_alerts = [a for a in alerts if a["ts_ms"] >= block_start_ms]
        if block_alerts:
            first = min(block_alerts, key=lambda a: a["ts_ms"])
            delays.append((first["ts_ms"] - block_start_ms) / 1000.0)
    return delays


def evaluate_config_on_session(
    rows: list[dict[str, Any]],
    config: StateMachineConfig,
) -> dict[str, Any]:
    events = rows_to_events(rows)
    result = replay(events, config)
    delays = _detection_delays(events, result["alerts"])
    return {
        "false_alerts_per_hour": result["false_alerts_per_hour"],
        "detection_delay_s": round(float(np.mean(delays)), 2) if delays else None,
        "n_alerts": len(result["alerts"]),
        "agreement": result["agreement"],
        "n_windows": result["n_windows"],
    }


def tune(
    sessions: dict[str, list[dict[str, Any]]],
    grace_values: list[float],
    confirm_values: list[float],
    alert_delay_values: list[float],
    away_threshold_values: list[float],
    max_delay_s: float | None = None,
) -> dict[str, Any]:
    """Grid search over state machine parameters."""
    combos = list(
        itertools.product(
            grace_values, confirm_values, alert_delay_values, away_threshold_values
        )
    )
    all_results: list[dict[str, Any]] = []

    for grace, confirm, alert_delay, away in combos:
        config = StateMachineConfig(
            grace_s=grace,
            confirm_s=confirm,
            alert_delay_s=alert_delay,
            away_threshold_s=away,
            adaptive=True,
        )
        session_scores: list[dict[str, Any]] = []
        for sid, rows in sessions.items():
            session_scores.append(
                {"session_id": sid, **evaluate_config_on_session(rows, config)}
            )
        fa_mean = float(np.mean([s["false_alerts_per_hour"] for s in session_scores]))
        delays = [
            s["detection_delay_s"]
            for s in session_scores
            if s["detection_delay_s"] is not None
        ]
        delay_mean = float(np.mean(delays)) if delays else None

        all_results.append(
            {
                "config": config.to_dict(),
                "false_alerts_per_hour": round(fa_mean, 4),
                "detection_delay_s": round(delay_mean, 2) if delay_mean is not None else None,
                "n_alerts": int(sum(s["n_alerts"] for s in session_scores)),
                "agreement": round(
                    float(np.mean([s["agreement"] for s in session_scores])), 4
                ),
                "per_session": session_scores,
            }
        )

    # Selection: minimize false alerts subject to max delay.
    feasible = [
        r
        for r in all_results
        if max_delay_s is None
        or (r["detection_delay_s"] is not None and r["detection_delay_s"] <= max_delay_s)
    ]
    if feasible:
        best = min(feasible, key=lambda r: r["false_alerts_per_hour"])
        selection_note = "lowest false alerts within max delay"
    else:
        with_delay = [r for r in all_results if r["detection_delay_s"] is not None]
        if with_delay:
            best = min(with_delay, key=lambda r: r["detection_delay_s"])
            selection_note = "lowest detection delay (no config met max delay)"
        else:
            best = min(all_results, key=lambda r: r["false_alerts_per_hour"])
            selection_note = "lowest false alerts (no delay data)"

    return {
        "best_config": best["config"],
        "selection_note": selection_note,
        "best_metrics": {
            "false_alerts_per_hour": best["false_alerts_per_hour"],
            "detection_delay_s": best["detection_delay_s"],
            "n_alerts": best["n_alerts"],
            "agreement": best["agreement"],
        },
        "all_results": sorted(
            all_results,
            key=lambda r: (r["false_alerts_per_hour"], r["detection_delay_s"] or 9999),
        ),
    }


def load_sessions(root: str | Path, fused_name: str = "fused.csv") -> dict[str, list[dict[str, Any]]]:
    """Load fused.csv from every session folder under root."""
    root = Path(root)
    sessions: dict[str, list[dict[str, Any]]] = {}
    targets: list[Path]
    if (root / fused_name).exists():
        targets = [root]
    else:
        targets = sorted(p for p in root.iterdir() if p.is_dir())

    for session_dir in targets:
        fused_path = session_dir / fused_name
        if not fused_path.exists():
            continue
        with fused_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            sessions[session_dir.name] = rows
    return sessions


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", help="folder containing session folders with fused.csv")
    ap.add_argument("--fused", default="fused.csv", help="fused CSV filename (default: fused.csv)")
    ap.add_argument(
        "--grace-s",
        default="30,60,90,120",
        help="comma-separated grace values (default: 30,60,90,120)",
    )
    ap.add_argument(
        "--confirm-s",
        default="10,20,30",
        help="comma-separated confirmation values (default: 10,20,30)",
    )
    ap.add_argument(
        "--alert-delay-s",
        default="60,90,120,180",
        help="comma-separated alert delay values (default: 60,90,120,180)",
    )
    ap.add_argument(
        "--away-threshold-s",
        default="20,30,60",
        help="comma-separated away threshold values (default: 20,30,60)",
    )
    ap.add_argument(
        "--max-delay-s",
        type=float,
        default=None,
        help="maximum acceptable detection delay in seconds",
    )
    ap.add_argument("--out", help="save tuning report as JSON")
    args = ap.parse_args(argv)

    sessions = load_sessions(args.root, args.fused)
    if not sessions:
        print("ERROR: no fused.csv found under", args.root, file=sys.stderr)
        return 2

    print(f"loaded {len(sessions)} session(s) with fused data")
    print(f"grid size: {len(_parse_float_list(args.grace_s))} x {len(_parse_float_list(args.confirm_s))} x {len(_parse_float_list(args.alert_delay_s))} x {len(_parse_float_list(args.away_threshold_s))} = {len(list(itertools.product(_parse_float_list(args.grace_s), _parse_float_list(args.confirm_s), _parse_float_list(args.alert_delay_s), _parse_float_list(args.away_threshold_s))))} combos")

    report = tune(
        sessions,
        grace_values=_parse_float_list(args.grace_s),
        confirm_values=_parse_float_list(args.confirm_s),
        alert_delay_values=_parse_float_list(args.alert_delay_s),
        away_threshold_values=_parse_float_list(args.away_threshold_s),
        max_delay_s=args.max_delay_s,
    )

    print("\n=== Best configuration ===")
    print(f"Note: {report['selection_note']}")
    print(f"grace_s       : {report['best_config']['grace_s']}")
    print(f"confirm_s     : {report['best_config']['confirm_s']}")
    print(f"alert_delay_s : {report['best_config']['alert_delay_s']}")
    print(f"away_threshold_s : {report['best_config']['away_threshold_s']}")
    print(f"false alerts/h: {report['best_metrics']['false_alerts_per_hour']}")
    print(f"detection delay: {report['best_metrics']['detection_delay_s']} s")
    print(f"agreement     : {report['best_metrics']['agreement']}")

    if args.out:
        write_report(report, Path(args.out))
        print(f"\nsaved tuning report to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
