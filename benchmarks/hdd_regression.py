"""
HDD-based benchmark demand model.

Implements the standard Heating Degree Day regression used by grid planners:
    Demand_peak(kW) = alpha + beta * HDD
    HDD = max(0, T_base - T_mean_daily)

Fits on the multi-archetype average peak (as a planner would) and then
evaluates underestimation error against each archetype's actual peak.

With Monte Carlo replication, each archetype×weather×fabric cell has
multiple peak values, giving the HDD model more training data and
the error analysis proper variance estimates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def compute_hdd(T_mean: float, T_base: float = 15.5) -> float:
    """Compute Heating Degree Days for a single day.

    Parameters
    ----------
    T_mean : float
        Mean daily outdoor temperature (°C).
    T_base : float
        Base temperature (°C), default 15.5 (UK standard).

    Returns
    -------
    float
        HDD value (degree-days).
    """
    return max(0.0, T_base - T_mean)


def fit_hdd_model(
    sim_df: pd.DataFrame,
    hdd_base: float = 15.5,
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """Fit the HDD regression on cross-archetype average peak demand.

    With replication, we get multiple observations per weather×fabric cell,
    giving the OLS fit more data points and proper residual variance.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation results.
    hdd_base : float
        HDD base temperature (°C).

    Returns
    -------
    model : statsmodels OLS results
        Fitted OLS model.
    planner_avg : pd.DataFrame
        Aggregated data used for fitting.
    """
    # Determine grouping columns (include replicate if present)
    has_rep = "replicate" in sim_df.columns
    group_cols = ["weather_scenario", "fabric", "archetype"]
    if has_rep:
        group_cols.append("replicate")

    # Peak demand per individual run
    run_peaks = (
        sim_df.groupby(group_cols)
        .agg(
            peak_kW=("electricity_demand_kW", "max"),
            T_mean_outdoor=("T_outdoor_C", "mean"),
        )
        .reset_index()
    )

    # Average peak across archetypes (planner's view — one value per
    # weather × fabric × replicate)
    avg_cols = ["weather_scenario", "fabric"]
    if has_rep:
        avg_cols.append("replicate")

    planner_avg = (
        run_peaks.groupby(avg_cols)
        .agg(
            peak_kW=("peak_kW", "mean"),
            T_mean_outdoor=("T_mean_outdoor", "mean"),
        )
        .reset_index()
    )

    planner_avg["HDD"] = planner_avg["T_mean_outdoor"].apply(
        lambda t: compute_hdd(t, hdd_base)
    )

    X = sm.add_constant(planner_avg["HDD"])
    y = planner_avg["peak_kW"]
    model = sm.OLS(y, X).fit()

    logger.info(
        "HDD model fitted — R²=%.4f, α=%.3f, β=%.3f  (n=%d)",
        model.rsquared, model.params.iloc[0], model.params.iloc[1], len(y),
    )

    return model, planner_avg


def compute_peak_errors(
    sim_df: pd.DataFrame,
    hdd_model: sm.regression.linear_model.RegressionResultsWrapper,
    hdd_base: float = 15.5,
) -> pd.DataFrame:
    """Compute HDD peak-demand underestimation for every scenario combination.

    Groups across replicates to produce one row per archetype×weather×fabric
    with mean and max peak across replicates.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation results.
    hdd_model : statsmodels OLS results
        Fitted HDD model.
    hdd_base : float
        HDD base temperature.

    Returns
    -------
    pd.DataFrame
        One row per archetype×weather×fabric with underestimation stats.
    """
    has_rep = "replicate" in sim_df.columns
    group_cols = ["archetype", "weather_scenario", "fabric"]
    if has_rep:
        group_cols.append("replicate")

    # Per-run peaks
    run_peaks = (
        sim_df.groupby(group_cols)
        .agg(
            peak_kW=("electricity_demand_kW", "max"),
            T_mean=("T_outdoor_C", "mean"),
        )
        .reset_index()
    )

    # Aggregate across replicates → mean and max peak per cell
    cell_stats = (
        run_peaks.groupby(["archetype", "weather_scenario", "fabric"])
        .agg(
            sim_peak_mean_kW=("peak_kW", "mean"),
            sim_peak_max_kW=("peak_kW", "max"),
            sim_peak_std_kW=("peak_kW", "std"),
            T_mean_outdoor_C=("T_mean", "mean"),
        )
        .reset_index()
    )

    # HDD prediction for each cell
    rows = []
    for _, row in cell_stats.iterrows():
        hdd_val = compute_hdd(row["T_mean_outdoor_C"], hdd_base)
        exog = np.array([[1.0, hdd_val]])
        hdd_peak = float(hdd_model.predict(exog)[0])

        sim_peak = row["sim_peak_mean_kW"]
        underest = (
            (sim_peak - hdd_peak) / sim_peak * 100.0 if sim_peak > 0 else 0.0
        )

        rows.append({
            "archetype": row["archetype"],
            "weather_scenario": row["weather_scenario"],
            "fabric": row["fabric"],
            "sim_peak_kW": round(sim_peak, 3),
            "sim_peak_max_kW": round(row["sim_peak_max_kW"], 3),
            "sim_peak_std_kW": round(row["sim_peak_std_kW"], 3),
            "hdd_peak_kW": round(hdd_peak, 3),
            "underestimation_pct": round(underest, 2),
            "T_mean_outdoor_C": round(row["T_mean_outdoor_C"], 2),
        })

    return pd.DataFrame(rows)


def run_hdd_benchmark(
    sim_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
) -> Tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """Full HDD benchmark pipeline.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    cfg : dict
        Full config (needs hdd.base_temp_C).
    output_dir : Path
        Directory for saving results tables.

    Returns
    -------
    errors_df : pd.DataFrame
        Peak underestimation table.
    model : statsmodels OLS results
        Fitted HDD model.
    """
    hdd_base = cfg["hdd"]["base_temp_C"]

    model, agg_df = fit_hdd_model(sim_df, hdd_base)
    errors_df = compute_peak_errors(sim_df, model, hdd_base)

    out_path = output_dir / "hdd_benchmark.csv"
    logger.info("Saving to %s", out_path)
    errors_df.to_csv(out_path, index=False)

    return errors_df, model
