"""AAC replay tool (project document, Section 5.5).

Replays a recorded session and shows what the system would have decided,
compare against human labels, and help tune the state machine parameters.

    # replay a session using its fused features + a trained baseline model
    python -m ml.replay data/processed/p01/ --fused fused.csv --model models/baseline.joblib

    # replay with synthetic events (no model needed)
    python -m ml.replay --synthetic --duration 600

    # sweep grace / confirm parameters and print metrics
    python -m ml.replay data/processed/p01/ --fused fused.csv \
        --grace-s 60 --confirm-s 15 --alert-delay-s 90

Outputs a per-window timeline, state transitions, alerts, and evaluation
metrics (false alerts per hour, detection delay, vs-ground-truth agreement).
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

from ml.state_machine import (  # noqa: E402
    StateMachine,
    StateMachineConfig,
    ON_TASK,
    UNCERTAIN,
    DISTRACTED,
    AWAY,
    BREAK,
)

try:  # pragma: no cover - optional joblib
    import joblib

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_fused_events(fused_csv: Path) -> list[dict]:
    """Load a fused CSV (from ml.fusion) into replay events."""
    events: list[dict] = []
    with fused_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Derive model_state from the fused label if present, else infer.
            label = row.get("task_state", "").strip()
            events.append(
                {
                    "ts_ms": int(row["start_ms"]),
                    "model_state": label if label in (ON_TASK, DISTRACTED, AWAY) else ON_TASK,
                    "face_present": None,
                    "away_signal": label == AWAY,
                    "break_declared": label == BREAK,
                    "ground_truth": label,
                }
            )
    return events


def load_model_events(
    features_csv: Path,
    model_path: Path,
) -> list[dict]:
    """Run a trained baseline model over feature windows to produce events."""
    if not _HAS_JOBLIB:
        raise RuntimeError("joblib is required to load a model")

    import joblib

    blob = joblib.load(model_path)
    model = blob["model"]
    feature_names = blob["feature_names"]

    events: list[dict] = []
    with features_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        for row in reader:
            try:
                X = np.array([[float(row[c]) for c in feature_names]], dtype=float)
            except (KeyError, ValueError, TypeError):
                continue
            pred = int(model.predict(X)[0])
            # Map model class index to label string — the baseline classes are
            # always (ON_TASK, DISTRACTED, AWAY) in this order.
            labels = ("ON_TASK", "DISTRACTED", "AWAY")
            model_state = labels[pred] if 0 <= pred < len(labels) else ON_TASK
            events.append(
                {
                    "ts_ms": int(row["start_ms"]),
                    "model_state": model_state,
                    "face_present": None,
                    "away_signal": model_state == AWAY,
                    "break_declared": False,
                    "ground_truth": row.get("task_state", "").strip(),
                }
            )
    return events


def synthetic_events(duration_s: float = 600.0, window_s: float = 30.0,
                     seed: int = 0) -> list[dict]:
    """Generate a scripted session for testing the replay tool."""
    rng = np.random.default_rng(seed)
    n = int(duration_s / window_s)
    events: list[dict] = []
    # Script: ON_TASK → DISTRACTED (phone) → AWAY → ON_TASK → UNCERTAIN
    schedule = [
        (0.0, 180.0, ON_TASK),
        (180.0, 300.0, DISTRACTED),
        (300.0, 420.0, AWAY),
        (420.0, 540.0, ON_TASK),
        (540.0, duration_s, UNCERTAIN),
    ]
    for i in range(n):
        t = i * window_s
        model_state = ON_TASK
        for start, end, st in schedule:
            if start <= t < end:
                model_state = st
                break
        # add occasional noise
        if rng.random() < 0.1:
            model_state = DISTRACTED if model_state == ON_TASK else ON_TASK
        events.append(
            {
                "ts_ms": int(t * 1000),
                "model_state": model_state,
                "face_present": model_state != AWAY,
                "away_signal": model_state == AWAY,
                "break_declared": False,
                "ground_truth": model_state,
            }
        )
    return events


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def replay(
    events: list[dict],
    config: StateMachineConfig,
    ground_truth_key: str = "ground_truth",
) -> dict:
    """Run the state machine and compare against ground truth."""
    sm = StateMachine(config)
    result = sm.run(events)

    # Per-window agreement vs ground truth.
    agree = 0
    total = 0
    per_state: dict[str, dict[str, int]] = {}
    for st, ev in zip(result.states, events):
        gt = ev.get(ground_truth_key, "").strip()
        pred = st["state"]
        total += 1
        if gt and gt == pred:
            agree += 1
        per_state.setdefault(gt, {"total": 0, "correct": 0})
        per_state[gt]["total"] += 1
        if gt == pred:
            per_state[gt]["correct"] += 1

    agreement = agree / total if total else 0.0
    per_state_report = {
        k: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            "support": v["total"]}
        for k, v in per_state.items()
    }

    return {
        "config": config.to_dict(),
        "n_windows": total,
        "agreement": agreement,
        "per_state": per_state_report,
        "transitions": [
            {"ts_ms": t.timestamp_ms, "from": t.from_state, "to": t.to_state,
             "trigger": t.trigger}
            for t in result.transitions
        ],
        "alerts": result.alerts,
        "false_alerts_per_hour": result.false_alerts_per_hour,
        "detection_delay_s": result.detection_delay_s,
        "states": result.states,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", nargs="?", help="session folder with features.csv (+ fused.csv)")
    ap.add_argument("--fused", help="fused CSV from ml.fusion (preferred over features.csv)")
    ap.add_argument("--features", help="features.csv to score with a trained model")
    ap.add_argument("--model", help="trained baseline model (.joblib) for --features")
    ap.add_argument("--synthetic", action="store_true", help="run a scripted synthetic session")
    ap.add_argument("--duration", type=float, default=600.0, help="synthetic duration (s)")
    ap.add_argument("--window-s", type=float, default=30.0, help="decision window (s)")
    ap.add_argument("--seed", type=int, default=0, help="synthetic seed")
    ap.add_argument("--grace-s", type=float, default=90.0, help="grace period (s)")
    ap.add_argument("--confirm-s", type=float, default=20.0, help="confirmation time (s)")
    ap.add_argument("--alert-delay-s", type=float, default=120.0, help="alert delay (s)")
    ap.add_argument("--away-threshold-s", type=float, default=30.0, help="away threshold (s)")
    ap.add_argument("--no-adaptive", action="store_true", help="disable adaptive alert backoff")
    ap.add_argument("--out", help="write replay result as JSON")
    ap.add_argument("--print-states", action="store_true", help="print per-window state timeline")
    args = ap.parse_args(argv)

    config = StateMachineConfig(
        grace_s=args.grace_s,
        confirm_s=args.confirm_s,
        alert_delay_s=args.alert_delay_s,
        away_threshold_s=args.away_threshold_s,
        adaptive=not args.no_adaptive,
    )

    if args.synthetic:
        events = synthetic_events(args.duration, args.window_s, args.seed)
        source = "synthetic"
    elif args.fused:
        events = load_fused_events(Path(args.fused))
        source = args.fused
    elif args.features and args.model:
        events = load_model_events(Path(args.features), Path(args.model))
        source = args.features
    elif args.session_dir:
        session_dir = Path(args.session_dir)
        fused = session_dir / "fused.csv"
        features = session_dir / "features.csv"
        model = Path(args.model) if args.model else session_dir / "baseline.joblib"
        if fused.exists():
            events = load_fused_events(fused)
            source = str(fused)
        elif features.exists() and model.exists():
            events = load_model_events(features, model)
            source = str(features)
        else:
            ap.error(f"no fused.csv or (features.csv + model) found in {session_dir}")
    else:
        ap.error("provide a session_dir, --fused, --features+--model, or --synthetic")

    result = replay(events, config)

    print(f"source : {source}")
    print(f"config : grace={config.grace_s}s confirm={config.confirm_s}s "
          f"alert_delay={config.alert_delay_s}s away={config.away_threshold_s}s "
          f"adaptive={config.adaptive}")
    print(f"windows: {result['n_windows']}")
    print(f"agreement vs ground truth: {result['agreement']:.3f}")
    print(f"false alerts / hour: {result['false_alerts_per_hour']:.2f}")
    print(f"detection delay: {result['detection_delay_s']} s")
    print(f"transitions: {len(result['transitions'])}")
    print(f"alerts: {len(result['alerts'])}")
    print("\nper-state accuracy:")
    for st, rep in result["per_state"].items():
        print(f"  {st:12s} accuracy={rep['accuracy']:.3f} support={rep['support']}")

    if args.print_states:
        print("\nstate timeline (state changes only):")
        prev = None
        for st in result["states"]:
            if st["state"] != prev:
                print(f"  t={st['ts_ms']:>10d}  {st['state']}{'  << ALERT' if st['alert'] else ''}")
                prev = st["state"]

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # states list can be large; write a compact summary plus full states.
        summary = {k: v for k, v in result.items() if k != "states"}
        out_path.write_text(
            json.dumps({"summary": summary, "states": result["states"]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nsaved replay result to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
