"""AAC baseline model (project document, Section 5.5).

Trains a classical baseline classifier on windowed IMU features and evaluates
with leave-one-session-out: train on every session except one, test on the
held-out session. Repeated for every session so each one is tested exactly
once on data it has never seen.

    # evaluate on all sessions under data/processed/
    python -m ml.baseline data/processed/ --cv

    # train a final model on all sessions and save it
    python -m ml.baseline data/processed/ --model models/baseline.joblib

Requires numpy + scikit-learn (declared in requirements-ml.txt). LightGBM is
optional and used only when --lightgbm is passed and the package is installed.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.pipeline import FeatureVector  # noqa: E402  (for feature column order)

# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------
ID_COLS = ["start_ms", "end_ms", "window_s", "sample_rate_hz", "task_state", "activity"]

LABELS = ("ON_TASK", "DISTRACTED", "AWAY")  # model classes (BREAK/UNCERTAIN are runtime-only)


def feature_columns(header: list[str]) -> list[str]:
    """Return the feature column names in stable sorted order."""
    return sorted([c for c in header if c not in ID_COLS])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class SessionFeatures:
    session_id: str
    X: np.ndarray          # (n_windows, n_features)
    y: np.ndarray          # (n_windows,) label indices
    labels: np.ndarray     # (n_windows,) label strings
    feature_names: list[str]


def _label_to_index(label: str) -> int:
    try:
        return LABELS.index(label)
    except ValueError:
        return -1  # unknown / BREAK / UNCERTAIN -> dropped


def load_session_features(features_csv: Path) -> SessionFeatures | None:
    """Load one ``features.csv`` produced by :mod:`ml.pipeline`.

    Rows whose task_state is not one of ON_TASK/DISTRACTED/AWAY are dropped
    (BREAK and UNCERTAIN are runtime-only states).
    """
    with features_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return None

    feats = feature_columns(header)
    X_list: list[list[float]] = []
    y_list: list[int] = []
    labels_list: list[str] = []

    for row in rows:
        state = row.get("task_state", "").strip()
        idx = _label_to_index(state)
        if idx < 0:
            continue  # drop runtime-only states
        try:
            X_list.append([float(row[c]) for c in feats])
        except (ValueError, KeyError):
            continue
        y_list.append(idx)
        labels_list.append(state)

    if not X_list:
        return None

    return SessionFeatures(
        session_id=features_csv.parent.name,
        X=np.array(X_list, dtype=float),
        y=np.array(y_list, dtype=int),
        labels=np.array(labels_list),
        feature_names=feats,
    )


def load_all_sessions(root: str | Path) -> dict[str, SessionFeatures]:
    """Load every ``features.csv`` under ``root`` (one per session folder)."""
    root = Path(root)
    sessions: dict[str, SessionFeatures] = {}
    for csv_path in sorted(root.glob("*/features.csv")):
        sess = load_session_features(csv_path)
        if sess is not None:
            sessions[sess.session_id] = sess
    return sessions


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class BaselineModel:
    """A scikit-learn Random Forest baseline with a fixed contract."""

    model: Any
    feature_names: list[str]
    classes: tuple[str, ...] = LABELS

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"model": self.model, "feature_names": self.feature_names, "classes": self.classes}, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        import joblib

        blob = joblib.load(path)
        return cls(model=blob["model"], feature_names=blob["feature_names"], classes=tuple(blob["classes"]))


def train_random_forest(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def train_lightgbm(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Any:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lightgbm is not installed; use --random-forest or pip install lightgbm") from exc

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "multiclass",
        "num_class": len(LABELS),
        "metric": "multi_logloss",
        "verbose": -1,
        "seed": seed,
        "class_weight": "balanced",
    }
    return lgb.train(params, train_data, num_boost_round=300)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    use_lightgbm: bool = False,
    seed: int = 42,
) -> Any:
    if use_lightgbm:
        return train_lightgbm(X, y, seed=seed)
    return train_random_forest(X, y, seed=seed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    session_id: str
    y_true: np.ndarray
    y_pred: np.ndarray
    macro_f1: float
    accuracy: float
    confusion: np.ndarray  # (3, 3)
    n_windows: int


@dataclass
class EvaluationReport:
    folds: list[FoldResult]
    macro_f1_mean: float
    macro_f1_std: float
    accuracy_mean: float
    accuracy_std: float
    confusion_total: np.ndarray  # summed across folds
    per_class: dict[str, dict[str, float]]  # label -> {precision, recall, f1, support}

    def to_dict(self) -> dict[str, Any]:
        per_class = {}
        for label in LABELS:
            pc = self.per_class.get(label, {})
            per_class[label] = {
                "precision": pc.get("precision", 0.0),
                "recall": pc.get("recall", 0.0),
                "f1": pc.get("f1", 0.0),
                "support": pc.get("support", 0),
            }
        return {
            "macro_f1_mean": self.macro_f1_mean,
            "macro_f1_std": self.macro_f1_std,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "confusion_matrix": self.confusion_total.tolist(),
            "confusion_labels": list(LABELS),
            "per_class": per_class,
            "n_folds": len(self.folds),
            "folds": [
                {
                    "session": f.session_id,
                    "macro_f1": f.macro_f1,
                    "accuracy": f.accuracy,
                    "n_windows": f.n_windows,
                }
                for f in self.folds
            ],
        }


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...] = LABELS) -> dict:
    n = len(labels)
    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        support = confusion[i, :].sum()
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

    macro_f1 = float(np.mean([pc["f1"] for pc in per_class.values()]))
    accuracy = _safe_div(confusion.diagonal().sum(), confusion.sum())

    return {
        "confusion": confusion,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class": per_class,
    }


def evaluate_leave_one_session_out(
    sessions: dict[str, SessionFeatures],
    use_lightgbm: bool = False,
    seed: int = 42,
) -> EvaluationReport:
    """Leave-one-session-out cross-validation."""
    session_ids = sorted(sessions.keys())
    if len(session_ids) < 2:
        raise ValueError(
            f"leave-one-session-out needs at least 2 sessions, got {len(session_ids)}"
        )
    folds: list[FoldResult] = []
    confusion_total = np.zeros((len(LABELS), len(LABELS)), dtype=int)

    for held_out in session_ids:
        train_ids = [sid for sid in session_ids if sid != held_out]
        X_train = np.vstack([sessions[sid].X for sid in train_ids])
        y_train = np.concatenate([sessions[sid].y for sid in train_ids])

        model = train_model(X_train, y_train, use_lightgbm=use_lightgbm, seed=seed)

        sess = sessions[held_out]
        y_pred = model.predict(sess.X)
        metrics = _compute_metrics(sess.y, y_pred)

        folds.append(
            FoldResult(
                session_id=held_out,
                y_true=sess.y,
                y_pred=y_pred,
                macro_f1=metrics["macro_f1"],
                accuracy=metrics["accuracy"],
                confusion=metrics["confusion"],
                n_windows=len(sess.y),
            )
        )
        confusion_total += metrics["confusion"]

    macro_f1s = [f.macro_f1 for f in folds]
    accuracies = [f.accuracy for f in folds]

    # Aggregate per-class metrics across all folds (micro-style pooled counts).
    pooled_per_class: dict[str, dict[str, float]] = {}
    for i, label in enumerate(LABELS):
        tp = confusion_total[i, i]
        fp = confusion_total[:, i].sum() - tp
        fn = confusion_total[i, :].sum() - tp
        support = confusion_total[i, :].sum()
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        pooled_per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }

    return EvaluationReport(
        folds=folds,
        macro_f1_mean=float(np.mean(macro_f1s)),
        macro_f1_std=float(np.std(macro_f1s)),
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        confusion_total=confusion_total,
        per_class=pooled_per_class,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _collect_feature_csvs(root: str | Path) -> list[Path]:
    root = Path(root)
    if (root / "features.csv").exists():
        return [root / "features.csv"]
    return sorted(root.glob("*/features.csv"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing session folders with features.csv")
    ap.add_argument("--cv", action="store_true", help="run leave-one-session-out evaluation")
    ap.add_argument("--model", help="save the final trained model to this path")
    ap.add_argument("--report", help="save evaluation report as JSON")
    ap.add_argument("--lightgbm", action="store_true", help="use LightGBM instead of Random Forest")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args(argv)

    sessions = load_all_sessions(args.root)
    if not sessions:
        print("ERROR: no sessions with features.csv found under", args.root, file=sys.stderr)
        return 2

    print(f"loaded {len(sessions)} session(s): {', '.join(sorted(sessions))}")
    n_features = len(next(iter(sessions.values())).feature_names)
    print(f"features per window: {n_features}")

    report: EvaluationReport | None = None
    if args.cv:
        report = evaluate_leave_one_session_out(sessions, use_lightgbm=args.lightgbm, seed=args.seed)
        print(f"\nLeave-one-session-out ({len(sessions)} folds):")
        print(f"  macro-F1  : {report.macro_f1_mean:.3f} +/- {report.macro_f1_std:.3f}")
        print(f"  accuracy  : {report.accuracy_mean:.3f} +/- {report.accuracy_std:.3f}")
        print("\n  per-class (pooled):")
        for label in LABELS:
            pc = report.per_class[label]
            print(
                f"    {label:12s} precision={pc['precision']:.3f} "
                f"recall={pc['recall']:.3f} f1={pc['f1']:.3f} support={pc['support']}"
            )
        print("\n  confusion matrix (rows=true, cols=pred):")
        print("      " + " ".join(f"{l:>9s}" for l in LABELS))
        for i, row in enumerate(report.confusion_total):
            print(f"  {LABELS[i]:9s} " + " ".join(f"{v:9d}" for v in row))

    # Train final model on all sessions and optionally save.
    X_all = np.vstack([s.X for s in sessions.values()])
    y_all = np.concatenate([s.y for s in sessions.values()])
    final_model = train_model(X_all, y_all, use_lightgbm=args.lightgbm, seed=args.seed)

    if args.model:
        import joblib

        joblib.dump(
            {"model": final_model, "feature_names": list(sessions.values())[0].feature_names, "classes": list(LABELS)},
            args.model,
        )
        print(f"\nsaved model to {args.model}")

    if report is not None and args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved report to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
