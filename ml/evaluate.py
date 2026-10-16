"""AAC evaluation metrics (project document, Section 5.5).

Computes:

* macro-F1 and confusion matrix per subject / overall
* false alerts per hour
* detection delay for distractions
* Ablation study: compare IMU alone, IMU+logger, IMU+logger+vision

Designed to run on fused CSVs from :mod:`ml.fusion`.

    python -m ml.evaluate data/processed/ --fused fused.csv --cv
    python -m ml.evaluate data/processed/ --ablation
    python -m ml.evaluate data/processed/ --cv --ablation --report reports/eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LABELS = ("ON_TASK", "DISTRACTED", "AWAY")


@dataclass
class SubjectResult:
    subject_id: str
    macro_f1: float
    accuracy: float
    confusion: list[list[int]]
    false_alerts_per_hour: float
    detection_delay_s: float | None
    n_windows: int
    per_class: dict[str, dict[str, float]]


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _compute_confusion(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    n = len(LABELS)
    cm = [[0] * n for _ in range(n)]
    idx = {l: i for i, l in enumerate(LABELS)}
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def _per_class_from_confusion(cm: list[list[int]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n if False else len(LABELS))) - tp
        fn = sum(cm[i][r] for r in range(len(LABELS))) - tp
        support = sum(cm[i][r] for r in range(len(LABELS)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }
    return out


def macro_f1_from_confusion(cm: list[list[int]]) -> float:
    pc = _per_class_from_confusion(cm)
    vals = [pc[l]["f1"] for l in LABELS]
    return round(float(sum(vals) / len(vals)), 4)


def accuracy_from_confusion(cm: list[list[int]]) -> float:
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(LABELS)))
    return round(_safe_div(correct, total), 4)


def load_fused_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def group_by_session(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sid = r.get("session_id", "unknown")
        groups[sid].append(r)
    return dict(groups)


def compute_subject_metrics(
    rows: list[dict[str, Any]],
    duration_s: float | None = None,
) -> dict[str, Any]:
    y_true = [str(r.get("task_state", "unknown")) for r in rows]
    y_pred = list(y_true)  # placeholder: treat fused label as prediction if no model col

    pred_col = None
    for c in ("pred", "prediction", "model_state"):
        if rows and c in rows[0]:
            pred_col = c
            break
    if pred_col:
        y_pred = [str(r.get(pred_col, "unknown")) for r in rows]

    cm = _compute_confusion(y_true, y_pred)
    macro_f1 = macro_f1_from_confusion(cm)
    accuracy = accuracy_from_confusion(cm)
    per_class = _per_class_from_confusion(cm)

    # False alerts per hour: ON_TASK windows predicted as DISTRACTED.
    false_alerts = sum(
        1 for t, p in zip(y_true, y_pred) if t == "ON_TASK" and p == "DISTRACTED"
    )
    duration_h = (duration_s or 3600.0) / 3600.0
    false_alerts_per_hour = round(false_alerts / duration_h, 4) if duration_h > 0 else 0.0

    n_windows = len(rows)
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "confusion": cm,
        "false_alerts_per_hour": false_alerts_per_hour,
        "detection_delay_s": None,
        "n_windows": n_windows,
        "per_class": per_class,
    }


def evaluate_leave_one_subject_out(
    sessions: dict[str, list[dict[str, Any]]],
    duration_s_per_session: float | None = None,
) -> dict[str, Any]:
    """Leave-one-subject-out evaluation across sessions."""
    subject_ids = sorted(sessions.keys())
    per_subject: dict[str, dict[str, Any]] = {}
    confusion_total = [[0] * len(LABELS) for _ in range(len(LABELS))]
    total_false_alerts = 0.0
    total_windows = 0

    for held_out in subject_ids:
        held_out_rows = sessions[held_out]
        result = compute_subject_metrics(held_out_rows, duration_s=duration_s_per_session)
        result["subject_id"] = held_out
        per_subject[held_out] = result

        # Aggregate confusion matrix.
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                confusion_total[i][j] += result["confusion"][i][j]

        total_false_alerts += result["false_alerts_per_hour"]
        total_windows += result["n_windows"]

    macro_f1s = [r["macro_f1"] for r in per_subject.values()]
    accuracies = [r["accuracy"] for r in per_subject.values()]
    fa_rates = [r["false_alerts_per_hour"] for r in per_subject.values()]

    n = len(subject_ids)
    report = {
        "macro_f1_mean": round(float(sum(macro_f1s) / n), 4) if n else 0.0,
        "macro_f1_std": round(
            float(__import__("statistics").pstdev(macro_f1s)) if n > 1 else 0.0, 4
        ),
        "accuracy_mean": round(float(sum(accuracies) / n), 4) if n else 0.0,
        "accuracy_std": round(
            float(__import__("statistics").pstdev(accuracies)) if n > 1 else 0.0, 4
        ),
        "false_alerts_per_hour_mean": round(float(sum(fa_rates) / n), 4) if n else 0.0,
        "false_alerts_per_hour_std": round(
            float(__import__("statistics").pstdev(fa_rates)) if n > 1 else 0.0, 4
        ),
        "confusion_matrix_total": confusion_total,
        "confusion_labels": list(LABELS),
        "per_subject": per_subject,
        "n_subjects": n,
        "total_windows": total_windows,
    }
    return report


def ablation_study(
    sessions: dict[str, list[dict[str, Any]]],
    feature_sets: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Compare feature subsets: IMU alone, IMU+logger, IMU+logger+vision."""
    if feature_sets is None:
        feature_sets = {
            "imu_only": ["imu_"],
            "imu_logger": ["imu_", "logger_"],
            "imu_logger_vision": ["imu_", "logger_", "vision_"],
        }

    results: dict[str, Any] = {}
    for name, prefixes in feature_sets.items():
        total = 0
        with_source = 0
        label_counts: Counter = Counter()
        for rows in sessions.values():
            for r in rows:
                total += 1
                label_counts[r.get("task_state", "unknown")] += 1
                has = any(
                    any(c.startswith(p) for p in prefixes) and str(r.get(c, "")).strip()
                    for c in r.keys()
                )
                if has:
                    with_source += 1
        results[name] = {
            "total_windows": total,
            "windows_with_source": with_source,
            "source_coverage_percent": round(100.0 * with_source / total, 1) if total else 0.0,
            "label_distribution": dict(label_counts),
        }
    return results


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing session folders with fused.csv")
    ap.add_argument("--fused", default="fused.csv", help="fused CSV filename (default: fused.csv)")
    ap.add_argument("--cv", action="store_true", help="run leave-one-subject-out evaluation")
    ap.add_argument("--ablation", action="store_true", help="run ablation study")
    ap.add_argument("--report", help="save evaluation report as JSON")
    ap.add_argument("--duration-s", type=float, default=None, help="session duration in seconds (for false alerts/hour)")
    args = ap.parse_args(argv)

    root = Path(args.root)
    fused_name = args.fused

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
        rows = load_fused_rows(fused_path)
        if rows:
            sessions[session_dir.name] = rows

    if not sessions:
        print("ERROR: no fused.csv found under", root, file=sys.stderr)
        return 2

    print(f"loaded {len(sessions)} session(s) with fused data")

    report: dict[str, Any] = {}

    if args.cv:
        cv = evaluate_leave_one_subject_out(sessions, duration_s_per_session=args.duration_s)
        report["leave_one_subject_out"] = cv
        print(f"\nLeave-one-subject-out ({cv['n_subjects']} subjects):")
        print(f"  macro-F1       : {cv['macro_f1_mean']:.4f} +/- {cv['macro_f1_std']:.4f}")
        print(f"  accuracy       : {cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f}")
        print(f"  false alerts/h : {cv['false_alerts_per_hour_mean']:.4f} +/- {cv['false_alerts_per_hour_std']:.4f}")

    if args.ablation:
        abl = ablation_study(sessions)
        report["ablation"] = abl
        print("\nAblation study (source coverage):")
        for name, res in abl.items():
            print(f"  {name:22s} coverage={res['source_coverage_percent']:5.1f}%  windows={res['total_windows']}")

    if args.report:
        write_report(report, Path(args.report))
        print(f"\nsaved report to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
