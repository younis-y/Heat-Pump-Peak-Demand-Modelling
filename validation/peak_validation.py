"""
Peak demand validation: compare simulated peaks against EoH field trial data.

Bins real half-hourly electricity demand by outdoor temperature and compares
the distribution of measured peak demands against simulated values from
the 2R1C model. Also validates daily demand shapes and diversity factors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_real_peak_stats(daily_peaks: pd.DataFrame) -> pd.DataFrame:
    """Compute peak demand statistics binned by outdoor temperature.

    Parameters
    ----------
    daily_peaks : pd.DataFrame
        Output from eoh_loader.compute_daily_peaks().

    Returns
    -------
    pd.DataFrame
        Binned peak demand stats: mean, median, p90, p95, std, n.
    """
    df = daily_peaks.dropna(subset=["peak_kW", "mean_T_out"]).copy()
    df["T_bin"] = (df["mean_T_out"] / 2.0).round() * 2.0  # 2°C bins

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

    ADMD = mean of individual peaks across all properties for a given day,
    divided by the peak of the aggregated demand. This is a key metric
    for network planning.

    Parameters
    ----------
    daily_peaks : pd.DataFrame
        Output from eoh_loader.compute_daily_peaks().

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
    # ADMD = mean peak (what the network actually sees per customer)
    result["ADMD_kW"] = result["mean_individual_peak"]
    result = result[result["n_properties"] >= 10].copy()
    return result


def validate_peaks(
    daily_peaks: pd.DataFrame,
    sim_results: pd.DataFrame | None,
    hp_cfg: dict,
    out_dir: Path,
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
    out_dir : Path
        Output directory.

    Returns
    -------
    dict
        Validation statistics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    real_binned = compute_real_peak_stats(daily_peaks)
    diversity = compute_diversity_factor(daily_peaks)

    # Save CSVs
    real_binned.to_csv(out_dir / "peak_demand_by_temperature.csv", index=False)
    diversity.to_csv(out_dir / "diversity_factor_by_temperature.csv", index=False)

    # Overall stats
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

    pd.DataFrame([stats]).to_csv(out_dir / "peak_validation_stats.csv", index=False)

    # --- Figure 1: Peak demand distribution by temperature ---
    sns.set_palette("colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: binned peak demand vs temperature
    ax = axes[0]
    ax.fill_between(
        real_binned["T_bin_centre"],
        real_binned["peak_median"] - real_binned["peak_std"],
        real_binned["peak_median"] + real_binned["peak_std"],
        alpha=0.2, color="C0", label="±1 std",
    )
    ax.plot(
        real_binned["T_bin_centre"], real_binned["peak_median"],
        "o-", color="C0", markersize=4, linewidth=1.5,
        label=f"EoH median peak (n={stats['n_property_days']:,})",
    )
    ax.plot(
        real_binned["T_bin_centre"], real_binned["peak_p90"],
        "s--", color="C3", markersize=3, linewidth=1,
        label="EoH 90th percentile",
    )
    ax.plot(
        real_binned["T_bin_centre"], real_binned["peak_p95"],
        "^:", color="C3", markersize=3, linewidth=1,
        label="EoH 95th percentile",
    )

    # Overlay simulated peaks if available
    if sim_results is not None and "peak_kW" in sim_results.columns:
        sim_mean = sim_results.groupby("weather")["peak_kW"].mean()
        ax.axhline(
            sim_mean.max(), color="C1", linestyle="--", linewidth=2,
            label=f"Model peak (worst case): {sim_mean.max():.1f} kW",
        )

    ax.set_xlabel("Mean daily outdoor temperature (°C)")
    ax.set_ylabel("Peak electrical demand (kW)")
    ax.set_title("Peak Demand vs Temperature: EoH Field Trial")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: histogram of peak demands
    ax = axes[1]
    ax.hist(
        overall, bins=50, density=True, alpha=0.7, color="C0",
        edgecolor="white", linewidth=0.5,
    )
    ax.axvline(overall.median(), color="C1", linestyle="--", label=f"Median: {overall.median():.2f} kW")
    ax.axvline(np.percentile(overall, 95), color="C3", linestyle="--", label=f"P95: {np.percentile(overall, 95):.2f} kW")
    ax.set_xlabel("Peak electrical demand (kW)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Daily Peak Demands")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Peak Demand Validation: EoH Field Trial (739 homes, 2020-2023)", fontsize=13)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_peak_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: ADMD / diversity ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        diversity["T_bin"], diversity["ADMD_kW"],
        "o-", color="C0", markersize=5, linewidth=1.5,
        label="After-diversity maximum demand (ADMD)",
    )
    ax.plot(
        diversity["T_bin"], diversity["max_individual_peak"],
        "s--", color="C3", markersize=4, linewidth=1,
        label="Maximum individual peak",
    )
    ax.set_xlabel("Mean daily outdoor temperature (°C)")
    ax.set_ylabel("Demand (kW)")
    ax.set_title("ADMD vs Temperature: EoH Field Trial")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_admd_by_temperature.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved peak validation figures to %s", out_dir)
    return stats
