"""
EoH (Electrification of Heat) trial data loader.

Loads the 30-minute interval cleansed dataset (SN 9209) from the
Energy Systems Catapult EoH demonstration project (2020-2023).
739 monitored ASHP homes with half-hourly electricity consumption,
thermal output, outdoor/indoor temperature, and backup heater usage.

Filters to winter heating season and computes derived fields:
- COP (half-hourly and daily)
- Electrical demand in kW
- Peak demand per property per day
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns we actually need (avoids loading 16 cols x 27M rows)
USE_COLS = [
    "Property_ID",
    "Timestamp",
    "Heat_Pump_Energy_Output",
    "Whole_System_Energy_Consumed",
    "External_Air_Temperature",
    "Internal_Air_Temperature",
    "Back-up_Heater_Energy_Consumed",
    "Immersion_Heater_Energy_Consumed",
]


def load_eoh_half_hourly(
    csv_path: str | Path,
    winter_only: bool = True,
    min_readings_per_property: int = 500,
) -> pd.DataFrame:
    """Load and clean the EoH 30-minute interval dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to eoh_cleaned_half_hourly.csv.
    winter_only : bool
        If True, keep only Nov-Feb (heating season).
    min_readings_per_property : int
        Drop properties with fewer valid readings.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with derived columns.
    """
    logger.info("Loading EoH half-hourly data from %s ...", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=USE_COLS,
        parse_dates=["Timestamp"],
        low_memory=False,
    )
    logger.info("Loaded %d rows, %d properties", len(df), df["Property_ID"].nunique())

    # Filter to winter heating season
    if winter_only:
        df["month"] = df["Timestamp"].dt.month
        df = df[df["month"].isin([11, 12, 1, 2])].copy()
        df.drop(columns=["month"], inplace=True)
        logger.info("Winter filter: %d rows remain", len(df))

    # Drop rows where HP was off (no output and no consumption)
    df = df.dropna(subset=["Whole_System_Energy_Consumed", "External_Air_Temperature"])
    df = df[df["Whole_System_Energy_Consumed"] > 0].copy()

    # Compute derived fields
    # Energy values are kWh per 30-min interval, so kW = kWh * 2
    df["elec_demand_kW"] = df["Whole_System_Energy_Consumed"] * 2 / 1000  # Wh->kWh->kW
    df["hp_thermal_kW"] = df["Heat_Pump_Energy_Output"] * 2 / 1000

    # COP = thermal output / electrical input (only when both > 0)
    mask_valid_cop = (df["Heat_Pump_Energy_Output"] > 0) & (df["Whole_System_Energy_Consumed"] > 0)
    df["COP"] = np.nan
    df.loc[mask_valid_cop, "COP"] = (
        df.loc[mask_valid_cop, "Heat_Pump_Energy_Output"]
        / df.loc[mask_valid_cop, "Whole_System_Energy_Consumed"]
    )
    # Clip implausible COP values (measurement artefacts)
    df.loc[df["COP"] > 8, "COP"] = np.nan
    df.loc[df["COP"] < 0.5, "COP"] = np.nan

    # Backup heater power
    df["Back-up_Heater_Energy_Consumed"] = df["Back-up_Heater_Energy_Consumed"].fillna(0)
    df["backup_kW"] = df["Back-up_Heater_Energy_Consumed"] * 2 / 1000

    # Date and hour
    df["date"] = df["Timestamp"].dt.date
    df["hour"] = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0

    # Drop properties with too few readings
    counts = df.groupby("Property_ID").size()
    valid_props = counts[counts >= min_readings_per_property].index
    df = df[df["Property_ID"].isin(valid_props)].copy()
    logger.info(
        "After quality filter: %d rows, %d properties",
        len(df), df["Property_ID"].nunique(),
    )

    return df


def load_eoh_daily(csv_path: str | Path) -> pd.DataFrame:
    """Load the EoH daily performance dataset.

    Parameters
    ----------
    csv_path : str or Path
        Path to eoh_cleaned_daily.csv.

    Returns
    -------
    pd.DataFrame
        Daily data with COP column.
    """
    logger.info("Loading EoH daily data from %s ...", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["Date"], dayfirst=True, low_memory=False)

    # Filter to winter
    df["month"] = df["Date"].dt.month
    df = df[df["month"].isin([11, 12, 1, 2])].copy()

    # Daily COP
    mask = (df["Heat_Pump_Energy_Output"] > 0) & (df["Whole_System_Energy_Consumed"] > 0)
    df["COP"] = np.nan
    df.loc[mask, "COP"] = (
        df.loc[mask, "Heat_Pump_Energy_Output"]
        / df.loc[mask, "Whole_System_Energy_Consumed"]
    )
    df.loc[df["COP"] > 8, "COP"] = np.nan
    df.loc[df["COP"] < 0.5, "COP"] = np.nan

    logger.info("Loaded %d daily rows, %d properties", len(df), df["Property_ID"].nunique())
    return df


def compute_daily_peaks(eoh_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily peak electricity demand per property.

    Parameters
    ----------
    eoh_df : pd.DataFrame
        Half-hourly EoH data from load_eoh_half_hourly().

    Returns
    -------
    pd.DataFrame
        One row per property-day with peak_kW, mean_T_out, mean_T_in.
    """
    peaks = (
        eoh_df.groupby(["Property_ID", "date"])
        .agg(
            peak_kW=("elec_demand_kW", "max"),
            mean_elec_kW=("elec_demand_kW", "mean"),
            daily_energy_kWh=("Whole_System_Energy_Consumed", lambda x: x.sum() / 1000),
            mean_T_out=("External_Air_Temperature", "mean"),
            min_T_out=("External_Air_Temperature", "min"),
            mean_T_in=("Internal_Air_Temperature", "mean"),
            mean_COP=("COP", "mean"),
            max_backup_kW=("backup_kW", "max"),
            n_readings=("elec_demand_kW", "count"),
        )
        .reset_index()
    )
    # Only keep days with full 48 half-hours (or close)
    peaks = peaks[peaks["n_readings"] >= 40].copy()
    logger.info("Computed daily peaks: %d property-days", len(peaks))
    return peaks
