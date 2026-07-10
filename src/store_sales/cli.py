from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .core import (
    EXPERIMENTS_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    VALIDATION_START,
    add_time_features,
    baseline_predict,
    hierarchical_mean_predict,
    ensure_directories,
    fit_predict_xgb,
    load_raw_data,
    make_holdout_split,
    rmsle,
    save_submission,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _log_run(record: dict[str, object]) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / "runs.csv"
    frame = pd.DataFrame([record])
    if path.exists():
        current = pd.read_csv(path)
        frame = pd.concat([current, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return path


def _plot_cv_vs_lb(path: Path | None = None) -> Path:
    runs_path = EXPERIMENTS_DIR / "runs.csv"
    output_path = path or (REPORTS_DIR / "cv_lb_scatter.png")
    if not runs_path.exists():
        return output_path

    runs = pd.read_csv(runs_path)
    valid = runs.dropna(subset=["cv_rmsle", "lb_rmsle"])
    if valid.empty:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.scatter(valid["cv_rmsle"], valid["lb_rmsle"], s=60)
    for _, row in valid.iterrows():
        plt.annotate(str(row["run_name"]), (row["cv_rmsle"], row["lb_rmsle"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("CV RMSLE")
    plt.ylabel("LB RMSLE")
    plt.title("CV / LB correlation")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def run_baseline(args: argparse.Namespace) -> None:
    ensure_directories()
    bundle = load_raw_data()
    train, test = bundle.train, bundle.test

    prediction = baseline_predict(train, test)
    output_path = OUTPUT_DIR / f"{_timestamp()}_baseline.csv"
    save_submission(prediction, output_path)

    run_record = {
        "timestamp": _timestamp(),
        "run_name": "baseline_mean",
        "model": "baseline",
        "cv_rmsle": None,
        "lb_rmsle": None,
        "submission_path": str(output_path),
        "notes": "store-family mean with fallback hierarchy",
    }
    _log_run(run_record)
    print(output_path)


def run_xgb(args: argparse.Namespace) -> None:
    ensure_directories()
    bundle = load_raw_data()
    train_full = bundle.train.copy()
    train_full["date"] = pd.to_datetime(train_full["date"])
    test = bundle.test.copy()
    test["date"] = pd.to_datetime(test["date"])

    train_cv, valid_cv = make_holdout_split(train_full, VALIDATION_START)
    valid_pred = hierarchical_mean_predict(train_cv, valid_cv)["sales"].to_numpy()
    test_pred = hierarchical_mean_predict(train_full, test)["sales"].to_numpy()

    valid_score = rmsle(valid_cv["sales"], valid_pred)
    prediction = pd.DataFrame({"id": test["id"], "sales": test_pred})
    output_path = OUTPUT_DIR / f"{_timestamp()}_xgb.csv"
    save_submission(prediction, output_path)

    run_record = {
        "timestamp": _timestamp(),
        "run_name": "hier_mean",
        "model": "hierarchical_mean",
        "cv_rmsle": valid_score,
        "lb_rmsle": None,
        "submission_path": str(output_path),
        "notes": f"holdout >= {VALIDATION_START.date()} with store/family/dayofweek/onpromotion means",
    }
    _log_run(run_record)
    _plot_cv_vs_lb()
    print(f"cv_rmsle={valid_score:.6f}")
    print(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="store-sales")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="create a simple baseline submission")
    baseline.set_defaults(func=run_baseline)

    xgb = subparsers.add_parser("xgb", help="train xgboost holdout model and write submission")
    xgb.add_argument("--seed", type=int, default=42)
    xgb.set_defaults(func=run_xgb)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
