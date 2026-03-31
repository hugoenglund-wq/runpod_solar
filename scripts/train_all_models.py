#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import TrainConfig, train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train intraday + D+1/D+2/D+3 solar forecasting models."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for trained models, metrics, predictions, and split datasets.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional train data filter start (origin_time), format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional train data filter end (origin_time), format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--evaluation-fraction",
        type=float,
        default=0.20,
        help="Fraction of issue days to reserve for backtest evaluation across folds.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=8,
        help="Maximum number of seasonal backtest folds.",
    )
    parser.add_argument(
        "--preferred-window-days",
        type=int,
        default=30,
        help="Preferred validation window length per fold in issue days.",
    )
    parser.add_argument(
        "--min-train-issue-days",
        type=int,
        default=60,
        help="Minimum number of issue days required before the first validation fold.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use lighter feature config for faster smoke testing.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=200,
        help="Fail early if a model receives fewer rows than this in train split.",
    )
    parser.add_argument(
        "--min-val-rows",
        type=int,
        default=50,
        help="Fail early if a model receives fewer rows than this in validation split.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainConfig(
        evaluation_fraction=args.evaluation_fraction,
        max_folds=args.max_folds,
        preferred_window_days=args.preferred_window_days,
        min_train_issue_days=args.min_train_issue_days,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        start_date=args.start_date,
        end_date=args.end_date,
        quick_test=args.quick_test,
    )
    metrics = train_all_models(
        artifacts_dir=Path(args.artifacts_dir),
        train_config=config,
        project_root=PROJECT_ROOT,
    )
    print("\nTraining complete.")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
