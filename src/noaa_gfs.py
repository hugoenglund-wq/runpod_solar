from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests


NOAA_GFS_AWS_BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DEFAULT_VARIABLE_PATTERNS = {
    "temperature_2m": [r":TMP:2 m above ground:"],
    "cloud_cover": [r":TCDC:entire atmosphere:[^:]*fcst:"],
    "shortwave_radiation": [r":DSWRF:surface:[^:]*fcst:"],
}


@dataclass(frozen=True)
class CatalogFile:
    issue_time_utc: datetime
    lead_hours: int
    object_key: str

    @property
    def file_url(self) -> str:
        return f"{NOAA_GFS_AWS_BUCKET}/{self.object_key}"

    @property
    def inv_url(self) -> str:
        return f"{self.file_url}.inv"


def build_daily_issue_times(
    *,
    start_date: date,
    end_date: date,
    issue_cycle_hour_utc: int,
) -> list[datetime]:
    issue_times: list[datetime] = []
    cursor = start_date
    while cursor <= end_date:
        issue_times.append(datetime(cursor.year, cursor.month, cursor.day, issue_cycle_hour_utc, tzinfo=UTC))
        cursor += timedelta(days=1)
    return issue_times


def discover_catalog_files_for_issue(
    session: requests.Session,
    *,
    issue_time_utc: datetime,
) -> list[CatalogFile]:
    ymd = issue_time_utc.strftime("%Y%m%d")
    issue_hour = issue_time_utc.strftime("%H")

    candidate_sets: list[list[CatalogFile]] = []
    for prefix in _candidate_s3_prefixes(ymd=ymd, issue_hour=issue_hour):
        keys = _list_s3_keys(session, prefix=prefix)
        if not keys:
            continue

        files: list[CatalogFile] = []
        for object_key in keys:
            if object_key.endswith(".idx") or ".f" not in object_key:
                continue
            match = re.search(r"\.f(\d{3})$", object_key)
            if not match:
                continue
            files.append(
                CatalogFile(
                    issue_time_utc=issue_time_utc,
                    lead_hours=int(match.group(1)),
                    object_key=object_key,
                )
            )
        if files:
            candidate_sets.append(sorted(files, key=lambda item: item.lead_hours))

    if not candidate_sets:
        return []

    candidate_sets.sort(
        key=lambda items: (
            max(item.lead_hours for item in items),
            len(items),
        ),
        reverse=True,
    )
    return candidate_sets[0]


def _candidate_s3_prefixes(*, ymd: str, issue_hour: str) -> tuple[str, ...]:
    return (
        f"gfs.{ymd}/{issue_hour}/atmos/gfs.t{issue_hour}z.pgrb2.0p25.f",
        f"gfs.{ymd}/{issue_hour}/gfs.t{issue_hour}z.pgrb2.1p00.",
        f"gfs.{ymd}/{issue_hour}/gfs.t{issue_hour}z.pgrb2.0p25.f",
    )


def _list_s3_keys(
    session: requests.Session,
    *,
    prefix: str,
    max_keys: int = 1000,
) -> list[str]:
    url = f"{NOAA_GFS_AWS_BUCKET}?prefix={prefix}&max-keys={max_keys}"
    response = session.get(url, timeout=120)
    if response.status_code != 200:
        return []
    root = ET.fromstring(response.text)
    return [elem.text for elem in root.findall(".//s3:Key", S3_NS) if elem.text]


def required_leads_for_local_day_offsets(
    *,
    issue_time_utc: datetime,
    timezone: str,
    day_offsets: tuple[int, ...],
    max_lead_hours: int = 120,
) -> set[int]:
    tz = ZoneInfo(timezone)
    issue_local = issue_time_utc.astimezone(tz)
    target_dates = {issue_local.date() + timedelta(days=offset) for offset in day_offsets}

    required: set[int] = set()
    for lead in range(0, max_lead_hours + 1):
        valid_local = (issue_time_utc + timedelta(hours=lead)).astimezone(tz)
        if valid_local.date() in target_dates:
            required.add(lead)
    return required


def extract_issue_point_forecast(
    session: requests.Session,
    *,
    catalog_files: list[CatalogFile],
    latitude: float,
    longitude: float,
    timezone: str,
    variable_patterns: dict[str, list[str]] = DEFAULT_VARIABLE_PATTERNS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for catalog_file in catalog_files:
        values = extract_point_values_for_file(
            session,
            file_url=catalog_file.file_url,
            inv_url=catalog_file.inv_url,
            latitude=latitude,
            longitude=longitude,
            variable_patterns=variable_patterns,
        )
        if not values:
            continue

        issue_local = _to_local_naive(catalog_file.issue_time_utc, timezone)
        valid_time_utc = catalog_file.issue_time_utc + timedelta(hours=catalog_file.lead_hours)
        valid_local = _to_local_naive(valid_time_utc, timezone)
        rows.append(
            {
                "forecast_issue_time": issue_local,
                "forecast_valid_time": valid_local,
                "lead_hours": catalog_file.lead_hours,
                **values,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "forecast_issue_time",
                "forecast_valid_time",
                "lead_hours",
                "temperature_2m",
                "cloud_cover",
                "shortwave_radiation",
            ]
        )

    hourly = pd.DataFrame(rows).sort_values(["forecast_issue_time", "forecast_valid_time"]).reset_index(drop=True)
    return resample_issue_frame_to_hourly(hourly)


def resample_issue_frame_to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    resampled_parts: list[pd.DataFrame] = []
    value_columns = [col for col in frame.columns if col not in {"forecast_issue_time", "forecast_valid_time", "lead_hours"}]
    for issue_time, issue_part in frame.groupby("forecast_issue_time", sort=True):
        piece = issue_part.sort_values("forecast_valid_time").drop_duplicates(subset=["forecast_valid_time"])
        piece = piece.set_index("forecast_valid_time")
        full_index = pd.date_range(piece.index.min(), piece.index.max(), freq="1h")
        piece = piece.reindex(full_index)
        piece["forecast_issue_time"] = issue_time
        piece[value_columns] = piece[value_columns].interpolate(method="time", limit_direction="both")
        piece["lead_hours"] = ((piece.index - pd.Timestamp(issue_time)).total_seconds() / 3600.0).astype(int)
        piece = piece.reset_index().rename(columns={"index": "forecast_valid_time"})
        if "shortwave_radiation" in piece.columns:
            piece["shortwave_radiation"] = piece["shortwave_radiation"].clip(lower=0.0)
        if "cloud_cover" in piece.columns:
            piece["cloud_cover"] = piece["cloud_cover"].clip(lower=0.0, upper=100.0)
        resampled_parts.append(piece)

    return pd.concat(resampled_parts, ignore_index=True).sort_values(
        ["forecast_issue_time", "forecast_valid_time"]
    ).reset_index(drop=True)


def extract_point_values_for_file(
    session: requests.Session,
    *,
    file_url: str,
    inv_url: str,
    latitude: float,
    longitude: float,
    variable_patterns: dict[str, list[str]],
) -> dict[str, float]:
    inventory_lines = session.get(inv_url, timeout=120).text.splitlines()
    if not inventory_lines or inventory_lines[0].startswith("<!doctype html"):
        return {}

    values: dict[str, float] = {}
    for variable_name, patterns in variable_patterns.items():
        range_tuple = find_inventory_range(inventory_lines, patterns)
        if range_tuple is None:
            continue
        start_byte, end_byte = range_tuple
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}
        response = session.get(file_url, headers=headers, timeout=120)
        response.raise_for_status()
        values[variable_name] = decode_grib_nearest_value(
            response.content,
            latitude=latitude,
            longitude=longitude,
        )
    return values


def find_inventory_range(lines: list[str], patterns: list[str]) -> tuple[int, int] | None:
    compiled = [re.compile(pattern) for pattern in patterns]
    starts = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        try:
            byte_offset = int(parts[1])
        except ValueError:
            continue
        starts.append((byte_offset, line))

    for idx, (start_byte, line) in enumerate(starts):
        if not any(pattern.search(line) for pattern in compiled):
            continue
        next_start = starts[idx + 1][0] if idx + 1 < len(starts) else None
        end_byte = next_start - 1 if next_start is not None else start_byte + 2_000_000
        return start_byte, end_byte
    return None


def decode_grib_nearest_value(
    content: bytes,
    *,
    latitude: float,
    longitude: float,
) -> float:
    try:
        from eccodes import codes_grib_find_nearest, codes_release
        from gribapi.gribapi import grib_new_from_message
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "NOAA GFS extraction requires the 'eccodes' Python package plus the ecCodes C library."
        ) from exc

    handle = grib_new_from_message(content)
    if handle is None:
        raise RuntimeError("Failed to decode GRIB message for NOAA GFS record.")
    try:
        nearest = codes_grib_find_nearest(handle, latitude, longitude)
        if isinstance(nearest, (list, tuple)) and nearest:
            sample = nearest[0]
            if isinstance(sample, dict) and "value" in sample:
                return float(sample["value"])
        if isinstance(nearest, dict) and "value" in nearest:
            return float(nearest["value"])
        raise RuntimeError("ecCodes did not return a nearest-point value.")
    finally:
        codes_release(handle)


def _to_local_naive(ts_utc: datetime, timezone: str) -> datetime:
    return ts_utc.astimezone(ZoneInfo(timezone)).replace(tzinfo=None)
