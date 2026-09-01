"""
validation.py — Validation against EOH field data.

Merged from:
    validation/eoh_loader.py
    validation/cop_validation.py
    validation/peak_validation.py
    validation/archetype_clustering.py
    validation/run_validation.py
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

# Gate for the seven standalone diagnostic figures saved by validate_cop,
# validate_peaks, and validate_archetypes. Default False because their content
# is reproduced (polished) inside fig_eoh_validation in plots.py.
# Computations are unaffected — only the PNG/PDF writes are suppressed.
SAVE_INTERMEDIATE_FIGS = False

# ===== Deduplicated imports =====

import json
import logging
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from plots import (
    apply_style, save_fig, add_grid,
    COLOURS, PALETTE,
    FIG_DOUBLE, FIG_SINGLE, FIG_SINGLE_TALL,
)

# ===== from validation/eoh_loader.py =====

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


# ===== from validation/cop_validation.py =====


def bin_cop_by_temperature(
    eoh_df: pd.DataFrame,
    bin_width: float = 1.0,
) -> pd.DataFrame:
    """Bin real COP values by outdoor temperature.

    Parameters
    ----------
    eoh_df : pd.DataFrame
        Half-hourly EoH data with COP and External_Air_Temperature columns.
    bin_width : float
        Temperature bin width in degrees C.

    Returns
    -------
    pd.DataFrame
        Columns: T_bin_centre, cop_mean, cop_median, cop_std, cop_25, cop_75, n
    """
    df = eoh_df.dropna(subset=["COP", "External_Air_Temperature"]).copy()
    df["T_bin"] = (df["External_Air_Temperature"] / bin_width).round() * bin_width

    binned = (
        df.groupby("T_bin")["COP"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "cop_mean", "median": "cop_median", "std": "cop_std", "count": "n"})
        .reset_index()
        .rename(columns={"T_bin": "T_bin_centre"})
    )

    # Add percentiles
    pcts = df.groupby("T_bin")["COP"].quantile([0.25, 0.75]).unstack()
    pcts.columns = ["cop_25", "cop_75"]
    binned = binned.merge(pcts, left_on="T_bin_centre", right_index=True)

    # Only keep bins with enough data
    binned = binned[binned["n"] >= 50].copy()
    return binned


def modelled_cop_curve(
    T_range: np.ndarray,
    hp_cfg: dict,
) -> np.ndarray:
    """Compute the parametric COP curve over a temperature range.

    Parameters
    ----------
    T_range : np.ndarray
        Outdoor temperatures.
    hp_cfg : dict
        Heat pump config section from YAML.

    Returns
    -------
    np.ndarray
        Modelled COP values.
    """
    cop = hp_cfg["cop_intercept"] + hp_cfg["cop_slope"] * T_range
    defrost_mask = T_range < hp_cfg["defrost_temp_threshold_C"]
    cop[defrost_mask] *= (1.0 - hp_cfg["defrost_efficiency_penalty"])
    return np.clip(cop, hp_cfg["cop_min"], hp_cfg["cop_max"])


def validate_cop(
    eoh_df: pd.DataFrame,
    hp_cfg: dict,
    out_dir: Path | None = None,
) -> dict:
    """Run COP validation and produce comparison figure.

    Parameters
    ----------
    eoh_df : pd.DataFrame
        Half-hourly EoH data.
    hp_cfg : dict
        Heat pump config.
    out_dir : Path or None
        Deprecated; output paths default to TABLES_DIR / FIGURES_DIR.

    Returns
    -------
    dict
        Validation statistics: MAE, RMSE, bias, R-squared.
    """
    tables_out = TABLES_DIR if out_dir is None else out_dir
    figures_out = FIGURES_DIR if out_dir is None else out_dir

    binned = bin_cop_by_temperature(eoh_df)

    T_range = np.linspace(binned["T_bin_centre"].min() - 1, binned["T_bin_centre"].max() + 1, 200)
    model_cop = modelled_cop_curve(T_range, hp_cfg)

    # Fit empirical COP regression to the real binned medians
    T_bins = binned["T_bin_centre"].values
    cop_real = binned["cop_median"].values
    emp_coeffs = np.polyfit(T_bins, cop_real, 1)
    emp_intercept, emp_slope = emp_coeffs[1], emp_coeffs[0]
    emp_cop_at_bins = np.polyval(emp_coeffs, T_bins)

    # Error metrics: model vs real
    model_at_bins = modelled_cop_curve(T_bins.copy(), hp_cfg)
    residuals = cop_real - model_at_bins
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    bias = np.mean(residuals)

    # R² for empirical fit
    emp_residuals = cop_real - emp_cop_at_bins
    ss_res_emp = np.sum(emp_residuals ** 2)
    ss_tot = np.sum((cop_real - cop_real.mean()) ** 2)
    r_squared_empirical = 1 - ss_res_emp / ss_tot if ss_tot > 0 else 0

    offset = float(np.mean(model_at_bins - cop_real))

    stats = {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "bias": round(bias, 3),
        "empirical_intercept": round(emp_intercept, 3),
        "empirical_slope": round(emp_slope, 4),
        "R_squared_empirical": round(r_squared_empirical, 3),
        "systematic_offset": round(offset, 3),
        "n_bins": len(binned),
        "n_observations": int(binned["n"].sum()),
    }
    logger.info(
        "COP validation: MAE=%.3f, bias=%.3f, offset=%.3f (n=%d obs)",
        mae, bias, offset, binned["n"].sum(),
    )
    logger.info(
        "Empirical COP: %.3f + %.4f*T (R\u00b2=%.3f) vs Model: %.1f + %.2f*T",
        emp_intercept, emp_slope, r_squared_empirical,
        hp_cfg["cop_intercept"], hp_cfg["cop_slope"],
    )

    pd.DataFrame([stats]).to_csv(TABLES_DIR / "validation_cop_stats.csv", index=False)
    binned.to_csv(TABLES_DIR / "validation_cop_by_temperature.csv", index=False)

    # ── Figure: COP validation ──
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    # Real data: median with IQR shading
    ax.fill_between(
        binned["T_bin_centre"], binned["cop_25"], binned["cop_75"],
        alpha=0.2, color=COLOURS["primary"],
        label="EoH IQR (25th\u201375th)",
    )
    ax.plot(
        binned["T_bin_centre"], binned["cop_median"],
        "o-", color=COLOURS["primary"], markersize=3, linewidth=1.0,
        label=f"EoH median (n={int(binned['n'].sum()):,})",
    )

    # Model curve
    ax.plot(
        T_range, model_cop, "--", color=COLOURS["secondary"], linewidth=1.2,
        label=f"Model: COP = {hp_cfg['cop_intercept']} + {hp_cfg['cop_slope']}T",
    )

    # Empirical fit
    emp_cop_curve = np.polyval(emp_coeffs, T_range)
    ax.plot(
        T_range, emp_cop_curve, "-.", color=COLOURS["tertiary"], linewidth=1.0,
        label=f"Empirical: COP = {emp_intercept:.2f} + {emp_slope:.3f}T",
    )

    # Defrost threshold
    ax.axvline(
        hp_cfg["defrost_temp_threshold_C"], color=COLOURS["neutral"],
        linestyle=":", linewidth=0.5, alpha=0.5,
    )

    ax.set_xlabel("Outdoor temperature (\u00b0C)")
    ax.set_ylabel("COP")
    ax.legend(fontsize=6)
    add_grid(ax)
    ax.set_xlim(binned["T_bin_centre"].min() - 1, binned["T_bin_centre"].max() + 1)
    ax.set_ylim(0, 6)
    fig.tight_layout()
    if SAVE_INTERMEDIATE_FIGS:
        save_fig(fig, FIGURES_DIR, "fig_validation_cop")

    return stats


# ===== from validation/peak_validation.py =====


def compute_real_peak_stats(daily_peaks: pd.DataFrame) -> pd.DataFrame:
    """Compute peak demand statistics binned by outdoor temperature.

    Parameters
    ----------
    daily_peaks : pd.DataFrame
        Output from compute_daily_peaks().

    Returns
    -------
    pd.DataFrame
        Binned peak demand stats: mean, median, p90, p95, std, n.
    """
    df = daily_peaks.dropna(subset=["peak_kW", "mean_T_out"]).copy()
    df["T_bin"] = (df["mean_T_out"] / 2.0).round() * 2.0

    binned = (
        df.groupby("T_bin")["peak_kW"]
        .agg(
            peak_mean="mean",
            peak_median="median",
            peak_std="std",
            peak_p90=lambda x: np.percentile(x, 90),
            peak_p95=lambda x: np.percentile(x, 95),
            n="count",
        )
        .reset_index()
        .rename(columns={"T_bin": "T_bin_centre"})
    )
    binned = binned[binned["n"] >= 30].copy()
    return binned


def compute_diversity_factor(daily_peaks: pd.DataFrame) -> pd.DataFrame:
    """Compute after-diversity maximum demand (ADMD) by temperature bin.

    Parameters
    ----------
    daily_peaks : pd.DataFrame
        Output from compute_daily_peaks().

    Returns
    -------
    pd.DataFrame
        Diversity factor by temperature bin.
    """
    df = daily_peaks.dropna(subset=["peak_kW", "mean_T_out"]).copy()
    df["T_bin"] = (df["mean_T_out"] / 2.0).round() * 2.0

    result = (
        df.groupby("T_bin")
        .agg(
            mean_individual_peak=("peak_kW", "mean"),
            max_individual_peak=("peak_kW", "max"),
            n_properties=("Property_ID", "nunique"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )
    result["ADMD_kW"] = result["mean_individual_peak"]
    result = result[result["n_properties"] >= 10].copy()
    return result


def validate_peaks(
    daily_peaks: pd.DataFrame,
    sim_results: pd.DataFrame | None,
    hp_cfg: dict,
    out_dir: Path | None = None,
) -> dict:
    """Run peak demand validation and produce comparison figures.

    Parameters
    ----------
    daily_peaks : pd.DataFrame
        Real daily peaks from EoH data.
    sim_results : pd.DataFrame or None
        Simulated results (if available) with peak_kW and weather columns.
    hp_cfg : dict
        Heat pump config.
    out_dir : Path or None
        Deprecated; output paths default to TABLES_DIR / FIGURES_DIR.

    Returns
    -------
    dict
        Validation statistics.
    """
    real_binned = compute_real_peak_stats(daily_peaks)
    diversity = compute_diversity_factor(daily_peaks)

    real_binned.to_csv(TABLES_DIR / "validation_peak_by_temperature.csv", index=False)
    diversity.to_csv(TABLES_DIR / "validation_diversity_by_temperature.csv", index=False)

    overall = daily_peaks["peak_kW"].dropna()
    stats = {
        "real_peak_median_kW": round(float(overall.median()), 3),
        "real_peak_mean_kW": round(float(overall.mean()), 3),
        "real_peak_p90_kW": round(float(np.percentile(overall, 90)), 3),
        "real_peak_p95_kW": round(float(np.percentile(overall, 95)), 3),
        "real_peak_max_kW": round(float(overall.max()), 3),
        "n_property_days": len(daily_peaks),
        "n_properties": int(daily_peaks["Property_ID"].nunique()),
        "rated_capacity_kW": hp_cfg["rated_thermal_capacity_kW"],
    }

    logger.info(
        "Peak validation: median=%.2f kW, p90=%.2f kW, p95=%.2f kW (n=%d property-days)",
        stats["real_peak_median_kW"], stats["real_peak_p90_kW"],
        stats["real_peak_p95_kW"], stats["n_property_days"],
    )

    pd.DataFrame([stats]).to_csv(TABLES_DIR / "validation_peak_stats.csv", index=False)

    # ── Figure 1: Peak demand distribution by temperature (1x2) ──
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # Left: binned peak demand vs temperature
    ax = axes[0]
    ax.fill_between(
        real_binned["T_bin_centre"],
        real_binned["peak_median"] - real_binned["peak_std"],
        real_binned["peak_median"] + real_binned["peak_std"],
        alpha=0.15, color=COLOURS["primary"], label="\u00b11 std",
    )
    ax.plot(
        real_binned["T_bin_centre"], real_binned["peak_median"],
        "o-", color=COLOURS["primary"], markersize=3, linewidth=1.0,
        label=f"Median (n={stats['n_property_days']:,})",
    )
    ax.plot(
        real_binned["T_bin_centre"], real_binned["peak_p95"],
        "^:", color=COLOURS["secondary"], markersize=2, linewidth=0.8,
        label="95th percentile",
    )

    if sim_results is not None and "peak_kW" in sim_results.columns:
        sim_mean = sim_results.groupby("weather")["peak_kW"].mean()
        ax.axhline(
            sim_mean.max(), color=COLOURS["quaternary"], linestyle="--", linewidth=1.0,
            label=f"Model peak: {sim_mean.max():.1f} kW",
        )

    ax.set_xlabel("Mean daily outdoor temperature (\u00b0C)")
    ax.set_ylabel("Peak demand (kW)")
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(fontsize=6)
    add_grid(ax)

    # Right: histogram
    ax = axes[1]
    ax.hist(
        overall, bins=50, density=True, alpha=0.6, color=COLOURS["primary"],
        edgecolor="white", linewidth=0.3,
    )
    ax.axvline(overall.median(), color=COLOURS["quaternary"], linestyle="--", linewidth=0.8,
               label=f"Median: {overall.median():.1f} kW")
    ax.axvline(np.percentile(overall, 95), color=COLOURS["secondary"], linestyle="--", linewidth=0.8,
               label=f"P95: {np.percentile(overall, 95):.1f} kW")
    ax.set_xlabel("Peak demand (kW)")
    ax.set_ylabel("Density")
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(fontsize=6)
    add_grid(ax)

    fig.tight_layout()
    if SAVE_INTERMEDIATE_FIGS:
        save_fig(fig, FIGURES_DIR, "fig_validation_peak")

    # ── Figure 2: ADMD / diversity ──
    fig, ax = plt.subplots(figsize=FIG_SINGLE_TALL)
    ax.plot(
        diversity["T_bin"], diversity["ADMD_kW"],
        "o-", color=COLOURS["primary"], markersize=3, linewidth=1.0,
        label="ADMD",
    )
    ax.plot(
        diversity["T_bin"], diversity["max_individual_peak"],
        "s--", color=COLOURS["secondary"], markersize=2, linewidth=0.8,
        label="Max individual peak",
    )
    ax.set_xlabel("Mean daily outdoor temperature (\u00b0C)")
    ax.set_ylabel("Demand (kW)")
    ax.legend()
    add_grid(ax)
    fig.tight_layout()
    if SAVE_INTERMEDIATE_FIGS:
        save_fig(fig, FIGURES_DIR, "fig_validation_peak_admd")

    logger.info("Saved peak validation figures to %s", FIGURES_DIR)
    return stats


# ===== from validation/archetype_clustering.py =====

PERIODS_PER_DAY = 48


def build_daily_profiles(eoh_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape half-hourly data into one row per property-day with 48 demand columns.

    Parameters
    ----------
    eoh_df : pd.DataFrame
        Half-hourly EoH data with elec_demand_kW, Property_ID, date, hour.

    Returns
    -------
    pd.DataFrame
        Pivoted: rows = (Property_ID, date), columns = period_0..period_47.
    """
    demand_col = "hp_thermal_kW" if "hp_thermal_kW" in eoh_df.columns else "elec_demand_kW"
    df = eoh_df[["Property_ID", "date", "hour", demand_col]].dropna().copy()

    df["period"] = (df["hour"] * 2).astype(int).clip(0, 47)

    pivoted = df.pivot_table(
        index=["Property_ID", "date"],
        columns="period",
        values=demand_col,
        aggfunc="mean",
    )

    pivoted = pivoted.dropna(thresh=44)
    pivoted = pivoted.ffill(axis=1).bfill(axis=1)
    pivoted.columns = [f"period_{int(c)}" for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    return pivoted


def filter_and_normalise_profiles(profiles: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter out flat profiles and normalise to [0, 1] for clustering.

    Parameters
    ----------
    profiles : pd.DataFrame
        Wide-format daily profiles.

    Returns
    -------
    tuple
        (filtered_profiles DataFrame, normalised matrix)
    """
    period_cols = [c for c in profiles.columns if c.startswith("period_")]
    X = profiles[period_cols].values.copy()

    row_range = X.max(axis=1) - X.min(axis=1)
    valid_mask = row_range > 0.05
    profiles_filt = profiles[valid_mask].copy()
    X = X[valid_mask]

    row_min = X.min(axis=1, keepdims=True)
    row_max = X.max(axis=1, keepdims=True)
    denom = row_max - row_min
    denom[denom < 0.01] = 1.0
    X_norm = (X - row_min) / denom

    return profiles_filt, X_norm


def cluster_profiles(
    X_norm: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans, float]:
    """Run k-means clustering on pre-normalised profiles.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalised profile matrix (n_profiles, 48).
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (labels, kmeans_model, silhouette_score)
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_norm)

    sil = silhouette_score(X_norm, labels, sample_size=min(10000, len(X_norm)))
    logger.info("K-means (k=%d): silhouette=%.3f", n_clusters, sil)

    return labels, km, sil


def discover_archetypes(
    eoh_df: pd.DataFrame,
    out_dir: Path | None = None,
    max_clusters: int = 6,
) -> dict:
    """Discover demand archetypes from EoH data and produce figures.

    Parameters
    ----------
    eoh_df : pd.DataFrame
        Half-hourly EoH data.
    out_dir : Path or None
        Deprecated; output paths default to TABLES_DIR / FIGURES_DIR.
    max_clusters : int
        Maximum number of clusters to evaluate.

    Returns
    -------
    dict
        Clustering results: best_k, silhouette scores, cluster sizes.
    """
    logger.info("Building daily demand profiles...")
    profiles = build_daily_profiles(eoh_df)
    logger.info("Built %d daily profiles", len(profiles))

    profiles, X_norm = filter_and_normalise_profiles(profiles)
    logger.info("After filtering flat profiles: %d remain", len(profiles))

    if len(profiles) < 100:
        logger.warning("Too few profiles (%d) for clustering", len(profiles))
        return {"error": "insufficient data", "n_profiles": len(profiles)}

    # Evaluate multiple k values
    sil_scores = {}
    for k in range(2, max_clusters + 1):
        _, _, sil = cluster_profiles(X_norm, n_clusters=k)
        sil_scores[k] = sil

    best_k = max(sil_scores, key=sil_scores.get)
    logger.info("Best k=%d (silhouette=%.3f)", best_k, sil_scores[best_k])

    # Final clustering
    labels, km, sil = cluster_profiles(X_norm, n_clusters=best_k)
    profiles = profiles.copy()
    profiles["cluster"] = labels

    period_cols = [c for c in profiles.columns if c.startswith("period_")]
    centroids_norm = km.cluster_centers_

    X_raw = profiles[period_cols].values
    centroids_kW = []
    for c in range(best_k):
        mask = labels == c
        centroids_kW.append(np.median(X_raw[mask], axis=0))
    centroids_kW = np.array(centroids_kW)

    # Cluster summary
    cluster_summary = []
    for c in range(best_k):
        mask = labels == c
        raw = X_raw[mask]
        norm_c = centroids_norm[c]
        cluster_summary.append({
            "cluster": c,
            "n_profiles": int(mask.sum()),
            "pct_profiles": round(100 * mask.sum() / len(labels), 1),
            "mean_peak_kW": round(float(raw.max(axis=1).mean()), 3),
            "median_mean_kW": round(float(np.median(raw.mean(axis=1))), 3),
            "peak_hour": round(float(norm_c.argmax() / 2), 1),
            "trough_hour": round(float(norm_c.argmin() / 2), 1),
        })

    summary_df = pd.DataFrame(cluster_summary)
    summary_df.to_csv(TABLES_DIR / "validation_archetype_summary.csv", index=False)

    sil_df = pd.DataFrame(list(sil_scores.items()), columns=["k", "silhouette"])
    sil_df.to_csv(TABLES_DIR / "validation_archetype_silhouette.csv", index=False)

    hours = np.arange(PERIODS_PER_DAY) / 2

    # ── Figure 1: Cluster centroid shapes (1x2) ──
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    ax = axes[0]
    for c in range(best_k):
        n = int((labels == c).sum())
        ax.plot(hours, centroids_norm[c], linewidth=1.2, color=PALETTE[c % len(PALETTE)],
                label=f"C{c} (n={n:,})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised demand (0\u20131)")
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(fontsize=6)
    add_grid(ax)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))

    ax = axes[1]
    for c in range(best_k):
        n = int((labels == c).sum())
        ax.plot(hours, centroids_kW[c], linewidth=1.2, color=PALETTE[c % len(PALETTE)],
                label=f"C{c} (n={n:,})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Median thermal output (kW)")
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(fontsize=6)
    add_grid(ax)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))

    fig.tight_layout()
    if SAVE_INTERMEDIATE_FIGS:
        save_fig(fig, FIGURES_DIR, "fig_validation_archetype_centroids")

    # ── Figure 2: Silhouette score vs k ──
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(list(sil_scores.keys()), list(sil_scores.values()), "o-",
            color=COLOURS["primary"], markersize=4)
    ax.axvline(best_k, color=COLOURS["secondary"], linestyle="--", linewidth=0.8, alpha=0.7,
               label=f"Best k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.legend()
    add_grid(ax)
    fig.tight_layout()
    if SAVE_INTERMEDIATE_FIGS:
        save_fig(fig, FIGURES_DIR, "fig_validation_archetype_silhouette")

    # ── Figure 3: Heatmap of cluster centroids ──
    fig, ax = plt.subplots(figsize=(FIG_DOUBLE[0], 1.5 + best_k * 0.5))
    im = ax.imshow(centroids_norm, aspect="auto", cmap="inferno",
                   extent=[0, 24, best_k - 0.5, -0.5])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Cluster")
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"C{c} ({int((labels==c).sum())})" for c in range(best_k)])
    ax.set_xticks(range(0, 25, 4))
    fig.colorbar(im, ax=ax, label="Normalised demand (0\u20131)", shrink=0.8)
    fig.tight_layout()
    if SAVE_INTERMEDIATE_FIGS:
        save_fig(fig, FIGURES_DIR, "fig_validation_archetype_heatmap")

    # ── Figure 4: k=4 forced comparison ──
    if best_k != 4 and 4 in sil_scores:
        labels_4, km_4, sil_4 = cluster_profiles(X_norm, n_clusters=4)
        centroids_4 = km_4.cluster_centers_

        archetype_names = ["Early Riser", "Home All Day", "Late Returner", "Intermittent"]
        peak_hours = [c.argmax() / 2 for c in centroids_4]
        sort_idx = np.argsort(peak_hours)

        fig, ax = plt.subplots(figsize=FIG_DOUBLE)
        for i, idx in enumerate(sort_idx):
            n = int((labels_4 == idx).sum())
            peak_h = peak_hours[idx]
            ax.plot(
                hours, centroids_4[idx], linewidth=1.2,
                color=PALETTE[i % len(PALETTE)],
                label=f"C{i} peak@{peak_h:.0f}h (n={n:,}) \u2014 cf. {archetype_names[i]}",
            )
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Normalised demand (0\u20131)")
        ax.legend(fontsize=6, loc="upper left")
        add_grid(ax)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        fig.tight_layout()
        if SAVE_INTERMEDIATE_FIGS:
            save_fig(fig, FIGURES_DIR, "fig_validation_archetype_k4")

    logger.info("Saved archetype clustering figures to %s", FIGURES_DIR)

    return {
        "best_k": best_k,
        "silhouette_scores": sil_scores,
        "cluster_sizes": {c: int((labels == c).sum()) for c in range(best_k)},
        "n_profiles": len(profiles),
    }


# ===== from validation/run_validation.py =====

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s \u2014 %(message)s",
)


def run_validation() -> None:
    """Orchestrate all validation analyses against EoH field data.

    Produces outputs in outputs/figures/ and outputs/tables/.
    """
    t0 = time.time()

    # Load config
    cfg_path = PROJECT_ROOT / "config" / "simulation_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    hp_cfg = cfg["heat_pump"]

    # Paths
    eoh_hh_path = (
        PROJECT_ROOT / "data" / "external" / "eoh"
        / "UKDA-30min-Interval-9209-csv" / "csv" / "eoh_cleaned_half_hourly.csv"
    )

    # Check data exists
    if not eoh_hh_path.exists():
        # Try alternative paths
        alt_paths = list((PROJECT_ROOT / "data" / "external" / "eoh").rglob("*half_hourly*.csv"))
        if alt_paths:
            eoh_hh_path = alt_paths[0]
            logger.info("Found EoH data at alternative path: %s", eoh_hh_path)
        else:
            logger.error(
                "EoH half-hourly data not found. Expected at:\n  %s\n"
                "Download from UK Data Service (SN 9209) and place in data/external/eoh/",
                eoh_hh_path,
            )
            sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load and clean EoH data ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading EoH half-hourly data")
    logger.info("=" * 60)
    eoh_df = load_eoh_half_hourly(eoh_hh_path)

    # ── Step 2: Compute daily peaks ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Computing daily peaks")
    logger.info("=" * 60)
    daily_peaks = compute_daily_peaks(eoh_df)

    # ── Step 3: COP validation ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: COP validation")
    logger.info("=" * 60)
    cop_stats = validate_cop(eoh_df, hp_cfg)

    # ── Step 4: Peak demand validation ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Peak demand validation")
    logger.info("=" * 60)

    # Try to load simulation results if they exist
    sim_path = PROJECT_ROOT / "outputs" / "simulation_results.csv"
    sim_results = None
    if sim_path.exists():
        sim_results = pd.read_csv(sim_path)
        logger.info("Loaded simulation results for comparison (%d rows)", len(sim_results))

    peak_stats = validate_peaks(daily_peaks, sim_results, hp_cfg)

    # ── Step 5: Archetype clustering ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Archetype clustering")
    logger.info("=" * 60)

    # Subsample by property to keep complete property-days intact
    all_props = eoh_df["Property_ID"].unique()
    max_props = 200  # enough for robust clustering, keeps memory manageable
    if len(all_props) > max_props:
        rng = np.random.RandomState(42)
        sampled_props = rng.choice(all_props, size=max_props, replace=False)
        eoh_sample = eoh_df[eoh_df["Property_ID"].isin(sampled_props)].copy()
        logger.info("Subsampled %d/%d properties for clustering (%d rows)",
                     max_props, len(all_props), len(eoh_sample))
    else:
        eoh_sample = eoh_df

    archetype_stats = discover_archetypes(eoh_sample)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)

    all_stats = {
        "cop_validation": cop_stats,
        "peak_validation": peak_stats,
        "archetype_clustering": archetype_stats,
    }

    # Convert numpy types for JSON serialisation
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary_path = TABLES_DIR / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=convert)

    elapsed = time.time() - t0
    logger.info("All results saved to %s / %s", FIGURES_DIR, TABLES_DIR)
    logger.info("Total time: %.1f seconds", elapsed)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\nCOP Validation (model vs whole-system SPF):")
    print(f"  MAE  = {cop_stats['MAE']:.3f}")
    print(f"  RMSE = {cop_stats['RMSE']:.3f}")
    print(f"  Systematic offset = {cop_stats['systematic_offset']:.3f} (model overestimates)")
    print(f"  Empirical fit: COP = {cop_stats['empirical_intercept']:.3f} + {cop_stats['empirical_slope']:.4f}*T")
    print(f"  Empirical R\u00b2  = {cop_stats['R_squared_empirical']:.3f}")
    print(f"  Bins = {cop_stats['n_bins']}, Obs = {cop_stats['n_observations']:,}")

    print(f"\nPeak Demand Validation:")
    if isinstance(peak_stats, dict) and "real_peak_median_kW" in peak_stats:
        print(f"  Median peak = {peak_stats['real_peak_median_kW']:.2f} kW")
        print(f"  P90 peak    = {peak_stats['real_peak_p90_kW']:.2f} kW")
        print(f"  P95 peak    = {peak_stats['real_peak_p95_kW']:.2f} kW")
        print(f"  N properties = {peak_stats['n_properties']}")

    print(f"\nArchetype Clustering:")
    if isinstance(archetype_stats, dict) and "best_k" in archetype_stats:
        print(f"  Best k = {archetype_stats['best_k']}")
        print(f"  Silhouette scores: {archetype_stats['silhouette_scores']}")
        print(f"  Profiles = {archetype_stats['n_profiles']:,}")

    print(f"\nFull results: figures={FIGURES_DIR}  tables={TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
