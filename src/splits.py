from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    validation_days: int = 30
    purge_hours: int = 100


@dataclass(frozen=True)
class BacktestConfig:
    evaluation_fraction: float = 0.20
    preferred_window_days: int = 30
    max_folds: int = 8
    min_train_issue_days: int = 60
    purge_issue_days: int = 1


@dataclass(frozen=True)
class BacktestFold:
    fold_id: int
    fold_name: str
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_issue_end: pd.Timestamp


def time_train_validation_split(
    frame: pd.DataFrame,
    *,
    config: SplitConfig = SplitConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Cannot split an empty frame.")
    if "origin_time" not in frame.columns:
        raise ValueError("Frame must contain origin_time for time split.")

    ordered = frame.sort_values("origin_time").reset_index(drop=True)
    max_origin = ordered["origin_time"].max()
    validation_start = max_origin - pd.Timedelta(days=config.validation_days) + pd.Timedelta(minutes=15)
    train_end = validation_start - pd.Timedelta(hours=config.purge_hours)

    train = ordered.loc[ordered["origin_time"] < train_end].copy()
    val = ordered.loc[ordered["origin_time"] >= validation_start].copy()

    if train.empty or val.empty:
        raise ValueError(
            "Train/validation split produced an empty partition. "
            "Use a longer date range or reduce validation_days/purge_hours."
        )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def build_issue_date_backtest_folds(
    frame: pd.DataFrame,
    *,
    config: BacktestConfig = BacktestConfig(),
) -> list[BacktestFold]:
    if frame.empty:
        raise ValueError("Cannot build backtest folds for an empty frame.")
    if "issue_date" not in frame.columns:
        raise ValueError("Frame must contain issue_date to build backtest folds.")

    issue_dates = pd.Index(pd.Series(frame["issue_date"]).dropna().drop_duplicates().sort_values())
    n_issue_days = len(issue_dates)
    if n_issue_days < 2:
        raise ValueError("At least two distinct issue dates are required for backtesting.")

    total_eval_days = max(1, int(round(n_issue_days * config.evaluation_fraction)))
    if n_issue_days >= 4 * config.preferred_window_days:
        n_folds = min(config.max_folds, max(4, int(round(total_eval_days / config.preferred_window_days))))
    else:
        n_folds = max(1, min(config.max_folds, int(round(total_eval_days / max(1, config.preferred_window_days)))))
    n_folds = max(1, n_folds)
    window_days = max(1, int(round(total_eval_days / n_folds)))

    minimum_required_start = max(config.min_train_issue_days, config.purge_issue_days + 1)
    earliest_start_idx = min(minimum_required_start, max(0, n_issue_days - window_days))
    latest_start_idx = max(earliest_start_idx, n_issue_days - window_days)
    start_indices = np.linspace(earliest_start_idx, latest_start_idx, num=n_folds, dtype=int)

    folds: list[BacktestFold] = []
    seen_ranges: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
    for fold_id, start_idx in enumerate(start_indices, start=1):
        end_idx = min(start_idx + window_days, n_issue_days)
        if end_idx <= start_idx:
            continue

        validation_dates = issue_dates[start_idx:end_idx]
        validation_start = pd.Timestamp(validation_dates[0])
        validation_end = pd.Timestamp(validation_dates[-1])
        key = (validation_start, validation_end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)

        train_end_idx = max(0, start_idx - config.purge_issue_days)
        if train_end_idx == 0:
            continue

        train_issue_end = pd.Timestamp(issue_dates[train_end_idx - 1])
        folds.append(
            BacktestFold(
                fold_id=fold_id,
                fold_name=f"fold_{fold_id:02d}",
                validation_start=validation_start,
                validation_end=validation_end,
                train_issue_end=train_issue_end,
            )
        )

    if not folds:
        raise ValueError(
            "Backtest fold generation failed. Use more issue dates or reduce min_train_issue_days."
        )
    return folds


def split_frame_for_fold(
    frame: pd.DataFrame,
    fold: BacktestFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "issue_date" not in frame.columns:
        raise ValueError("Frame must contain issue_date for fold splitting.")

    train = frame.loc[frame["issue_date"] <= fold.train_issue_end].copy()
    validation = frame.loc[
        (frame["issue_date"] >= fold.validation_start) & (frame["issue_date"] <= fold.validation_end)
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(f"{fold.fold_name}: empty train or validation partition.")
    train["fold_id"] = fold.fold_id
    train["fold_name"] = fold.fold_name
    validation["fold_id"] = fold.fold_id
    validation["fold_name"] = fold.fold_name
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def summarize_season_coverage(frame: pd.DataFrame, *, season_col: str = "target_season") -> dict[str, object]:
    if season_col not in frame.columns:
        return {"season_count": 0, "seasons": []}
    seasons = (
        pd.Series(frame[season_col])
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return {
        "season_count": len(seasons),
        "seasons": seasons,
    }
