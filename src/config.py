from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PANEL_LAT = 57.704
PANEL_LON = 11.771
PANEL_TILT = 27.0
PANEL_AZIMUTH = 180.0
SYSTEM_CAPACITY_W = 11000.0
TIMEZONE = "Europe/Stockholm"
DATA_FREQUENCY = "15min"
FORECAST_HORIZON_HOURS = 100
FORECAST_HORIZON_STEPS = FORECAST_HORIZON_HOURS * 4


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def production_csv(self) -> Path:
        return self.data_dir / "raw" / "production" / "solar_power_15min.csv"

    @property
    def weather_archive_hourly_csv(self) -> Path:
        return self.data_dir / "raw" / "weather" / "archive_hourly" / "weather_hourly_archive.csv"

    @property
    def weather_history_15min_csv(self) -> Path:
        return self.data_dir / "raw" / "weather" / "historical_forecast_15min" / "weather_15min_seamless.csv"

    @property
    def weather_previous_runs_csv(self) -> Path:
        return (
            self.data_dir
            / "raw"
            / "weather"
            / "previous_runs_hourly"
            / "weather_previous_runs_issue_valid.csv"
        )

    @property
    def metadata_json(self) -> Path:
        return self.data_dir / "metadata" / "system_metadata.json"

    @property
    def manifest_json(self) -> Path:
        return self.data_dir / "metadata" / "data_manifest.json"


def default_project_paths(root: str | Path | None = None) -> ProjectPaths:
    base = Path(root) if root is not None else Path.cwd()
    return ProjectPaths(root=base)
