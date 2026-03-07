"""
EPW weather file parser — extracts real 24-hour profiles for simulation.

Parses EnergyPlus Weather (.epw) files and extracts temperature, solar
irradiance, and wind speed for specified dates.  Hourly EPW data is
resampled to 15-minute resolution via linear interpolation.

Data source: London Gatwick TMYx 2009-2023 from climate.onebuilding.org
Reference: ASHRAE/WMO TMYx methodology (Crawley & Lawrie, 2019)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# EPW data rows start after 8 header lines.
# Column indices (0-based) for fields we need:
#   0: Year, 1: Month, 2: Day, 3: Hour (1-24), 4: Minute
#   6: Dry bulb temperature (°C)
#  13: Global horizontal irradiance (Wh/m²)
#  21: Wind speed (m/s)
EPW_HEADER_LINES = 8
COL_YEAR = 0
COL_MONTH = 1
COL_DAY = 2
COL_HOUR = 3
COL_DRY_BULB = 6
COL_GHI = 13
COL_WIND_SPEED = 21


def parse_epw(epw_path: str | Path) -> pd.DataFrame:
    """Parse an EPW file into a tidy DataFrame.

    Parameters
    ----------
    epw_path : str or Path
        Path to the .epw file.

    Returns
    -------
    pd.DataFrame
        Columns: datetime, T_outdoor_C, solar_W_m2, wind_speed_m_s
        8760 rows (hourly, full year).
    """
    epw_path = Path(epw_path)
    rows = []
    with open(epw_path, "r") as f:
        for i, line in enumerate(f):
            if i < EPW_HEADER_LINES:
                continue
            parts = line.strip().split(",")
            year = int(parts[COL_YEAR])
            month = int(parts[COL_MONTH])
            day = int(parts[COL_DAY])
            # EPW hours are 1-24; hour 1 = 00:00-01:00, so timestamp = hour - 1
            hour = int(parts[COL_HOUR]) - 1
            t_db = float(parts[COL_DRY_BULB])
            ghi = float(parts[COL_GHI])
            wind = float(parts[COL_WIND_SPEED])
            rows.append((year, month, day, hour, t_db, ghi, wind))

    df = pd.DataFrame(
        rows, columns=["year", "month", "day", "hour", "T_outdoor_C", "solar_W_m2", "wind_speed_m_s"]
    )
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day"]].assign(hour=df["hour"])
    )
    df = df[["datetime", "T_outdoor_C", "solar_W_m2", "wind_speed_m_s"]]
    logger.info("Parsed EPW: %d rows from %s", len(df), epw_path.name)
    return df


def extract_day_profile(
    epw_df: pd.DataFrame,
    month: int,
    day: int,
    dt_minutes: int = 15,
) -> pd.DataFrame:
    """Extract a single 24-hour profile and resample to simulation resolution.

    Parameters
    ----------
    epw_df : pd.DataFrame
        Full-year EPW data from parse_epw().
    month : int
        Month (1-12).
    day : int
        Day of month.
    dt_minutes : int
        Target output resolution in minutes (default 15).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, T_outdoor_C, solar_W_m2, wind_speed_m_s
        96 rows at 15-minute resolution.
    """
    mask = (epw_df["datetime"].dt.month == month) & (epw_df["datetime"].dt.day == day)
    day_df = epw_df[mask].copy().reset_index(drop=True)

    if len(day_df) < 24:
        raise ValueError(f"Only {len(day_df)} hours found for month={month}, day={day}")

    # Set index to datetime for resampling
    day_df = day_df.set_index("datetime")

    # Resample from hourly to target resolution via interpolation
    target_periods = (24 * 60) // dt_minutes
    new_index = pd.date_range(
        day_df.index[0], periods=target_periods, freq=f"{dt_minutes}min"
    )
    day_resampled = day_df.reindex(day_df.index.union(new_index)).interpolate(method="time")
    day_resampled = day_resampled.loc[new_index]

    # Ensure no negative solar
    day_resampled["solar_W_m2"] = day_resampled["solar_W_m2"].clip(lower=0)

    result = day_resampled.reset_index().rename(columns={"index": "timestamp"})
    t_mean = result["T_outdoor_C"].mean()
    t_min = result["T_outdoor_C"].min()
    t_max = result["T_outdoor_C"].max()
    logger.info(
        "Extracted %02d-%02d: %d pts, T_mean=%.1f°C [%.1f, %.1f]",
        month, day, len(result), t_mean, t_min, t_max,
    )
    return result


def load_weather_profiles(
    epw_path: str | Path,
    weather_days: Dict[str, Dict],
    dt_minutes: int = 15,
) -> Dict[str, pd.DataFrame]:
    """Load all weather scenario profiles from the EPW file.

    Parameters
    ----------
    epw_path : str or Path
        Path to EPW file.
    weather_days : dict
        Mapping of weather ID to {month, day} dict, e.g.
        {"W1": {"month": 3, "day": 19}, ...}
    dt_minutes : int
        Output resolution.

    Returns
    -------
    dict
        {weather_id: DataFrame} with 96-row profiles.
    """
    epw_df = parse_epw(epw_path)
    profiles = {}
    for w_id, day_spec in weather_days.items():
        profiles[w_id] = extract_day_profile(
            epw_df, day_spec["month"], day_spec["day"], dt_minutes
        )
    return profiles
