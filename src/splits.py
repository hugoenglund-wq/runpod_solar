from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    validation_days: int = 30
    purge_hours: int = 100


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
