"""
pipeline.py — Heat-pump analytics & benchmark pipeline.

Merged from:
    analytics/aggregation_montecarlo.py
    analytics/anova_decomposition.py
    analytics/cost_of_underestimation.py
    analytics/demand_surface.py
    analytics/interaction_regression.py
    benchmarks/hdd_regression.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import linear_reset

PROJECT_ROOT = Path(__file__).parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

logger = logging.getLogger(__name__)

# ===== from analytics/aggregation_montecarlo.py =====

# Archetype mix distributions for sensitivity analysis
ARCHETYPE_MIXES: dict[str, dict[str, float]] = {
    "uniform": {"B1": 0.25, "B2": 0.25, "B3": 0.25, "B4": 0.25},
    "mostly_home": {"B1": 0.05, "B2": 0.80, "B3": 0.05, "B4": 0.10},
    "uk_typical": {"B1": 0.30, "B2": 0.25, "B3": 0.25, "B4": 0.20},
    "commuter_heavy": {"B1": 0.35, "B2": 0.05, "B3": 0.50, "B4": 0.10},
}

DEFAULT_N_VALUES = [5, 10, 20, 50, 100, 150, 200, 300, 500]


def extract_demand_profiles(
    sim_df: pd.DataFrame,
    weather_scenario: str = "W3",
    fabric: str | None = None,
) -> dict[str, list[np.ndarray]]:
    """Extract per-run electricity demand arrays grouped by archetype.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation results (23 040 rows).
    weather_scenario : str
        Weather scenario code to filter on (default ``"W3"``).
    fabric : str | None
        Optional fabric code to filter on.  If *None*, all fabrics are
        included (giving more profiles per archetype).

    Returns
    -------
    dict[str, list[np.ndarray]]
        Mapping ``{archetype: [array_of_96_demand_values, ...]}``.
    """
    subset = sim_df[sim_df["weather_scenario"] == weather_scenario]

    if fabric is not None:
        subset = subset[subset["fabric"] == fabric]

    profiles: dict[str, list[np.ndarray]] = {}
    for arch in sorted(subset["archetype"].unique()):
        arch_profiles: list[np.ndarray] = []
        arch_data = subset[subset["archetype"] == arch]

        for (fab, rep), grp in arch_data.groupby(["fabric", "replicate"]):
            demand = grp["electricity_demand_kW"].values.copy()
            if len(demand) > 0:
                arch_profiles.append(demand)

        profiles[arch] = arch_profiles

    logger.info(
        "Extracted profiles: %s (weather=%s, fabric=%s)",
        {k: len(v) for k, v in profiles.items()},
        weather_scenario,
        fabric or "mixed",
    )

    return profiles


def sample_neighbourhood(
    profiles: dict[str, list[np.ndarray]],
    n_homes: int,
    archetype_probs: dict[str, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Sample *n_homes* from archetype pools and return aggregate profile.

    Parameters
    ----------
    profiles
        Output of :func:`extract_demand_profiles`.
    n_homes
        Number of homes to sample.
    archetype_probs
        Probability of each archetype (must sum to ~1).
    rng
        NumPy random generator instance.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(aggregate_profile, aggregate_peak_kW)``.
    """
    archetypes = list(archetype_probs.keys())
    probs = np.array([archetype_probs[a] for a in archetypes])
    probs = probs / probs.sum()

    # Sample archetype labels for each home
    labels = rng.choice(archetypes, size=n_homes, p=probs)

    # Determine profile length from the first available profile
    n_steps = len(next(iter(profiles.values()))[0])
    aggregate = np.zeros(n_steps)

    for label in labels:
        pool = profiles[label]
        idx = rng.integers(0, len(pool))
        aggregate += pool[idx]

    return aggregate, float(aggregate.max())


def compute_admd_curve(
    profiles: dict[str, list[np.ndarray]],
    archetype_probs: dict[str, float],
    n_homes_values: list[int] | None = None,
    n_monte_carlo: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Monte Carlo to compute ADMD as a function of neighbourhood size.

    Parameters
    ----------
    profiles
        Output of :func:`extract_demand_profiles`.
    archetype_probs
        Probability of each archetype.
    n_homes_values
        List of neighbourhood sizes to evaluate.
    n_monte_carlo
        Number of Monte Carlo iterations per size.
    seed
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per neighbourhood size with ADMD statistics.
    """
    if n_homes_values is None:
        n_homes_values = DEFAULT_N_VALUES

    rng = np.random.default_rng(seed)
    rows = []

    for n in n_homes_values:
        peaks = []
        for _ in range(n_monte_carlo):
            _, agg_peak = sample_neighbourhood(profiles, n, archetype_probs, rng)
            peaks.append(agg_peak)

        peaks = np.array(peaks)
        admd = peaks / n

        rows.append({
            "n_homes": n,
            "admd_mean_kW": round(float(admd.mean()), 4),
            "admd_median_kW": round(float(np.median(admd)), 4),
            "admd_p5_kW": round(float(np.percentile(admd, 5)), 4),
            "admd_p95_kW": round(float(np.percentile(admd, 95)), 4),
            "admd_std_kW": round(float(admd.std()), 4),
            "aggregate_peak_mean_kW": round(float(peaks.mean()), 2),
            "individual_peak_mean_kW": round(float(admd.mean()), 4),
        })

        logger.info(
            "N=%d: ADMD=%.3f kW/home (%.3f–%.3f)",
            n, admd.mean(),
            np.percentile(admd, 5), np.percentile(admd, 95),
        )

    return pd.DataFrame(rows)


def run_archetype_sensitivity(
    sim_df: pd.DataFrame,
    weather_scenario: str = "W3",
    fabric: str | None = None,
    n_monte_carlo: int = 1000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Compute ADMD curves for every archetype mix.

    Parameters
    ----------
    sim_df
        Full simulation results.
    weather_scenario
        Weather code (default ``"W3"``).
    fabric
        Optional fabric filter.
    n_monte_carlo
        Monte Carlo draws per size.
    seed
        Random seed.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{mix_name: admd_curve_dataframe}``.
    """
    profiles = extract_demand_profiles(sim_df, weather_scenario, fabric)
    results: dict[str, pd.DataFrame] = {}

    for mix_name, mix_probs in ARCHETYPE_MIXES.items():
        logger.info("Computing ADMD curve for mix: %s", mix_name)
        results[mix_name] = compute_admd_curve(
            profiles, mix_probs,
            n_monte_carlo=n_monte_carlo, seed=seed,
        )

    return results


def run_aggregation_montecarlo(
    sim_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path | None = None,
) -> dict:
    """Master entry point — run all ADMD curves and save outputs.

    Parameters
    ----------
    sim_df
        Full simulation results.
    cfg
        Pipeline configuration dict.
    output_dir
        Directory to write CSV results to.  Defaults to ``TABLES_DIR``.

    Returns
    -------
    dict
        Keys: ``admd_curves``, ``sensitivity_at_100``, ``profiles``.
    """
    out = output_dir if output_dir is not None else TABLES_DIR
    out.mkdir(parents=True, exist_ok=True)

    profiles = extract_demand_profiles(sim_df, weather_scenario="W3", fabric=None)
    admd_curves: dict[str, pd.DataFrame] = {}

    for mix_name, mix_probs in ARCHETYPE_MIXES.items():
        logger.info("ADMD curve: %s", mix_name)
        curve = compute_admd_curve(profiles, mix_probs)
        admd_curves[mix_name] = curve
        curve.to_csv(out / f"admd_curve_{mix_name}.csv", index=False)

    # Extract sensitivity at N=100
    sensitivity_rows = []
    for mix_name, curve in admd_curves.items():
        row_100 = curve[curve["n_homes"] == 100]
        if len(row_100) > 0:
            r = row_100.iloc[0]
            sensitivity_rows.append({
                "mix": mix_name,
                "admd_kW_per_home": r["admd_mean_kW"],
                "admd_p95_kW": r["admd_p95_kW"],
                "total_peak_100_homes_kW": r["aggregate_peak_mean_kW"],
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(out / "admd_sensitivity_at_100.csv", index=False)

    logger.info("Saved aggregation results to %s", out)

    return {
        "admd_curves": admd_curves,
        "sensitivity_at_100": sensitivity_df,
        "profiles": profiles,
    }


# ===== from analytics/anova_decomposition.py =====

def extract_peak_demand(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Compute peak 15-min electricity demand per simulation run.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.

    Returns
    -------
    pd.DataFrame
        One row per run with peak_demand_kW and factor columns.
    """
    group_cols = ["archetype", "weather_scenario", "fabric"]
    if "replicate" in sim_df.columns:
        group_cols.append("replicate")

    peaks = (
        sim_df.groupby(group_cols)
        .agg(peak_demand_kW=("electricity_demand_kW", "max"))
        .reset_index()
    )
    return peaks


def run_anova(
    sim_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the 3-way ANOVA and save results.

    With replication, we use all main effects and two-way interactions
    with the three-way interaction + replicates as residual, giving
    proper degrees of freedom for F-tests.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    output_dir : Path | None
        Directory for results tables.  Defaults to ``TABLES_DIR``.

    Returns
    -------
    pd.DataFrame
        ANOVA table with eta_squared column added.
    """
    out = output_dir if output_dir is not None else TABLES_DIR
    peaks = extract_peak_demand(sim_df)

    # With replication we have enough df for all 2-way interactions
    formula = (
        "peak_demand_kW ~ C(archetype) + C(weather_scenario) + C(fabric) "
        "+ C(archetype):C(weather_scenario) "
        "+ C(archetype):C(fabric) "
        "+ C(weather_scenario):C(fabric)"
    )

    model = ols(formula, data=peaks).fit()
    anova_table = anova_lm(model, typ=2)

    # Compute eta-squared
    ss_total = anova_table["sum_sq"].sum()
    anova_table["eta_squared"] = anova_table["sum_sq"] / ss_total

    # Clean up index names for readability
    anova_table.index = [
        idx.replace("C(", "").replace(")", "").replace(":", " × ")
        for idx in anova_table.index
    ]

    out_path = out / "anova_results.csv"
    logger.info("Saving to %s", out_path)
    anova_table.to_csv(out_path)

    # Log top factor
    non_residual = anova_table[anova_table.index != "Residual"]
    top_factor = non_residual["eta_squared"].idxmax()
    logger.info(
        "Top factor by η²: %s (%.4f)", top_factor, non_residual["eta_squared"].max()
    )

    return anova_table


# ===== from analytics/cost_of_underestimation.py =====

# Unit costs for LV network reinforcement (£)
COST_PARAMS = {
    "planned_GBP_per_kVA": 300,
    "reactive_multiplier": 2.5,
    "headroom_factor": 1.15,
    "power_factor": 1.0,
}

# Diversity factor applied to HDD peak to get ADMD
HDD_DIVERSITY_FACTOR = 0.7

DEVELOPMENT_SIZES = [50, 100, 200, 500]


def compute_hdd_admd(
    errors_df: pd.DataFrame,
    diversity_factor: float = HDD_DIVERSITY_FACTOR,
) -> pd.DataFrame:
    """Compute HDD-based ADMD by weather scenario.

    Parameters
    ----------
    errors_df : pd.DataFrame
        HDD benchmark results (one row per scenario cell).
    diversity_factor : float
        Multiplier to convert individual HDD peak to ADMD.

    Returns
    -------
    pd.DataFrame
        Summary with ``hdd_admd_kW`` per weather scenario.

    Notes
    -----
    This function originates in ``analytics/cost_of_underestimation.py``.
    The task instructions attribute it to ``benchmarks/hdd_regression.py``,
    but it does not appear there; the version here is the sole definition.
    """
    hdd_by_weather = (
        errors_df
        .groupby("weather_scenario")
        .agg(
            hdd_peak_mean_kW=("hdd_peak_kW", "mean"),
            sim_peak_mean_kW=("sim_peak_kW", "mean"),
            mean_temp_C=("T_mean_outdoor_C", "mean"),
        )
        .reset_index()
    )

    hdd_by_weather["hdd_admd_kW"] = (
        hdd_by_weather["hdd_peak_mean_kW"] * diversity_factor
    )

    return hdd_by_weather


def compute_cost_table(
    errors_df: pd.DataFrame,
    admd_curves: dict[str, pd.DataFrame],
    mix_name: str = "uk_typical",
    cost_params: dict | None = None,
    development_sizes: list[int] | None = None,
) -> pd.DataFrame:
    """Build cost comparison table for a given archetype mix.

    Parameters
    ----------
    errors_df : pd.DataFrame
        HDD benchmark results.
    admd_curves : dict[str, pd.DataFrame]
        ADMD curves by mix name from Monte Carlo aggregation.
    mix_name : str
        Which archetype mix to use (default ``"uk_typical"``).
    cost_params : dict | None
        Override cost parameters; defaults to module-level ``COST_PARAMS``.
    development_sizes : list[int] | None
        Number of homes to evaluate; defaults to ``DEVELOPMENT_SIZES``.

    Returns
    -------
    pd.DataFrame
        One row per development size with cost columns.
    """
    if cost_params is None:
        cost_params = COST_PARAMS
    if development_sizes is None:
        development_sizes = DEVELOPMENT_SIZES

    planned_rate = cost_params["planned_GBP_per_kVA"]
    reactive_mult = cost_params["reactive_multiplier"]
    headroom = cost_params["headroom_factor"]
    pf = cost_params["power_factor"]

    # Get HDD-based ADMD for worst-case weather (W3)
    hdd_summary = compute_hdd_admd(errors_df)
    w3 = hdd_summary[hdd_summary["weather_scenario"] == "W3"]
    if len(w3) == 0:
        w3 = hdd_summary.iloc[[-1]]
    hdd_admd = float(w3["hdd_admd_kW"].values[0])

    # Get simulation-based ADMD from the chosen mix curve
    if mix_name not in admd_curves:
        mix_name = list(admd_curves.keys())[0]
    sim_curve = admd_curves[mix_name]

    rows = []
    for n in development_sizes:
        # Look up or interpolate simulation ADMD at this N
        exact = sim_curve[sim_curve["n_homes"] == n]
        if len(exact) > 0:
            sim_admd = float(exact["admd_mean_kW"].values[0])
            sim_p95 = float(exact["admd_p95_kW"].values[0])
        else:
            sim_admd = float(np.interp(
                n, sim_curve["n_homes"].values, sim_curve["admd_mean_kW"].values
            ))
            sim_p95 = float(np.interp(
                n, sim_curve["n_homes"].values, sim_curve["admd_p95_kW"].values
            ))

        # Capacity calculations (kVA)
        hdd_capacity = n * hdd_admd * headroom / pf
        sim_capacity = n * sim_admd * headroom / pf
        sim_p95_capacity = n * sim_p95 * headroom / pf

        shortfall = max(0, sim_capacity - hdd_capacity)
        shortfall_p95 = max(0, sim_p95_capacity - hdd_capacity)

        planned_cost = shortfall * planned_rate
        reactive_cost = shortfall * planned_rate * reactive_mult
        planned_cost_p95 = shortfall_p95 * planned_rate
        reactive_cost_p95 = shortfall_p95 * planned_rate * reactive_mult

        rows.append({
            "n_homes": n,
            "hdd_admd_kW": round(hdd_admd, 3),
            "sim_admd_mean_kW": round(sim_admd, 3),
            "sim_admd_p95_kW": round(sim_p95, 3),
            "shortfall_per_home_kW": round(sim_admd - hdd_admd, 3),
            "total_shortfall_kVA": round(shortfall, 1),
            "planned_cost_GBP": round(planned_cost, 0),
            "reactive_cost_GBP": round(reactive_cost, 0),
            "planned_cost_p95_GBP": round(planned_cost_p95, 0),
            "reactive_cost_p95_GBP": round(reactive_cost_p95, 0),
        })

    return pd.DataFrame(rows)


def run_cost_analysis(
    errors_df: pd.DataFrame,
    admd_curves: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run cost analysis and save results.

    Parameters
    ----------
    errors_df
        HDD benchmark results.
    admd_curves
        ADMD curves from Monte Carlo aggregation.
    output_dir
        Directory to write CSV results to.  Defaults to ``TABLES_DIR``.

    Returns
    -------
    pd.DataFrame
        Cost comparison table.
    """
    out = output_dir if output_dir is not None else TABLES_DIR
    out.mkdir(parents=True, exist_ok=True)

    cost_df = compute_cost_table(errors_df, admd_curves)

    cost_df.to_csv(out / "cost_of_underestimation.csv", index=False)
    logger.info("Saved cost table to %s", out / "cost_of_underestimation.csv")

    # Also save the HDD ADMD summary
    hdd_summary = compute_hdd_admd(errors_df)
    hdd_summary.to_csv(out / "hdd_admd_summary.csv", index=False)

    return cost_df


# ===== from analytics/demand_surface.py =====

def build_demand_surfaces(
    sim_df: pd.DataFrame,
) -> dict:
    """Construct hour × temperature demand matrices per archetype.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output (2304 rows).

    Returns
    -------
    dict
        Keys are archetype IDs. Values are dicts with:
        - 'matrix': 2D np.ndarray (hours × temp bins)
        - 'temp_bins': 1D array of bin centres
        - 'hours': 1D array [0..23]
    """
    sim_df = sim_df.copy()
    sim_df["hour"] = pd.to_datetime(sim_df["timestamp"]).dt.hour

    # Temperature bins
    t_min = sim_df["T_outdoor_C"].min() - 1
    t_max = sim_df["T_outdoor_C"].max() + 1
    temp_edges = np.linspace(t_min, t_max, 20)
    temp_centres = 0.5 * (temp_edges[:-1] + temp_edges[1:])

    sim_df["temp_bin"] = pd.cut(
        sim_df["T_outdoor_C"], bins=temp_edges, labels=temp_centres, include_lowest=True
    )
    sim_df["temp_bin"] = sim_df["temp_bin"].astype(float)

    surfaces = {}
    for arch_id, grp in sim_df.groupby("archetype"):
        pivot = grp.pivot_table(
            index="hour",
            columns="temp_bin",
            values="electricity_demand_kW",
            aggfunc="mean",
        )
        # Ensure full 0–23 hour range
        pivot = pivot.reindex(range(24))
        pivot = pivot.interpolate(axis=0, limit_direction="both")
        pivot = pivot.interpolate(axis=1, limit_direction="both")

        surfaces[arch_id] = {
            "matrix": pivot.values,
            "temp_bins": pivot.columns.values,
            "hours": pivot.index.values,
        }

    logger.info("Built demand surfaces for %d archetypes.", len(surfaces))
    return surfaces


# ===== from analytics/interaction_regression.py =====

def prepare_regression_data(
    sim_df: pd.DataFrame,
    hdd_base: float = 15.5,
) -> pd.DataFrame:
    """Aggregate simulation data to per-run peak demand with predictor columns.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    hdd_base : float
        HDD base temperature (°C).

    Returns
    -------
    pd.DataFrame
        One row per run with peak_demand_kW, T_mean, HDD, archetype, fabric.
    """
    group_cols = ["archetype", "weather_scenario", "fabric"]
    if "replicate" in sim_df.columns:
        group_cols.append("replicate")

    agg = (
        sim_df.groupby(group_cols)
        .agg(
            peak_demand_kW=("electricity_demand_kW", "max"),
            T_mean=("T_outdoor_C", "mean"),
        )
        .reset_index()
    )
    agg["HDD"] = agg["T_mean"].apply(lambda t: max(0.0, hdd_base - t))
    agg["T_mean_sq"] = agg["T_mean"] ** 2
    return agg


def fit_model_a(reg_df: pd.DataFrame):
    """Fit Model A: peak_demand ~ HDD (linear).

    Parameters
    ----------
    reg_df : pd.DataFrame
        Regression dataset.

    Returns
    -------
    statsmodels OLS results
    """
    model = ols("peak_demand_kW ~ HDD", data=reg_df).fit()
    logger.info(
        "Model A  R²=%.4f  AIC=%.2f  BIC=%.2f  (n=%d)",
        model.rsquared, model.aic, model.bic, model.nobs,
    )
    return model


def fit_model_b(reg_df: pd.DataFrame):
    """Fit Model B: peak_demand ~ T_mean + T_mean² + archetype + T_mean:archetype + fabric.

    Parameters
    ----------
    reg_df : pd.DataFrame
        Regression dataset.

    Returns
    -------
    statsmodels OLS results
    """
    formula = (
        "peak_demand_kW ~ T_mean + T_mean_sq + C(archetype) "
        "+ T_mean:C(archetype) + C(fabric)"
    )
    model = ols(formula, data=reg_df).fit()
    logger.info(
        "Model B  R²=%.4f  AIC=%.2f  BIC=%.2f  (n=%d)",
        model.rsquared, model.aic, model.bic, model.nobs,
    )
    return model


def run_reset_test(model_a) -> dict:
    """Run Ramsey RESET test on Model A residuals for nonlinearity.

    Parameters
    ----------
    model_a : statsmodels OLS results
        Fitted Model A.

    Returns
    -------
    dict
        Keys: statistic, p_value.
    """
    reset = linear_reset(model_a, power=3, use_f=True)
    result = {
        "statistic": float(reset.statistic),
        "p_value": float(reset.pvalue),
    }
    logger.info("RESET test: F=%.4f, p=%.6f", result["statistic"], result["p_value"])
    return result


def run_interaction_regression(
    sim_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path | None = None,
) -> dict:
    """Full interaction regression pipeline.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    cfg : dict
        Full config.
    output_dir : Path | None
        Directory for results tables.  Defaults to ``TABLES_DIR``.

    Returns
    -------
    dict
        Keys: model_a, model_b, comparison_df, reset_test, reg_df.
    """
    out = output_dir if output_dir is not None else TABLES_DIR
    hdd_base = cfg["hdd"]["base_temp_C"]
    reg_df = prepare_regression_data(sim_df, hdd_base)

    model_a = fit_model_a(reg_df)
    model_b = fit_model_b(reg_df)

    reset_result = run_reset_test(model_a)

    rmse_a = np.sqrt(np.mean(model_a.resid ** 2))
    rmse_b = np.sqrt(np.mean(model_b.resid ** 2))

    comparison = pd.DataFrame({
        "Model": ["A (HDD-linear)", "B (Full interaction)"],
        "R_squared": [model_a.rsquared, model_b.rsquared],
        "RMSE": [rmse_a, rmse_b],
        "AIC": [model_a.aic, model_b.aic],
        "BIC": [model_a.bic, model_b.bic],
        "n_obs": [int(model_a.nobs), int(model_b.nobs)],
    })

    out_path = out / "model_comparison.csv"
    logger.info("Saving to %s", out_path)
    comparison.to_csv(out_path, index=False)

    # Store residuals for plotting
    reg_df["resid_A"] = model_a.resid.values
    reg_df["fitted_A"] = model_a.fittedvalues.values

    return {
        "model_a": model_a,
        "model_b": model_b,
        "comparison_df": comparison,
        "reset_test": reset_result,
        "reg_df": reg_df,
    }


# ===== from benchmarks/hdd_regression.py =====

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
    output_dir: Path | None = None,
) -> Tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """Full HDD benchmark pipeline.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    cfg : dict
        Full config (needs hdd.base_temp_C).
    output_dir : Path | None
        Directory for saving results tables.  Defaults to ``TABLES_DIR``.

    Returns
    -------
    errors_df : pd.DataFrame
        Peak underestimation table.
    model : statsmodels OLS results
        Fitted HDD model.
    """
    out = output_dir if output_dir is not None else TABLES_DIR
    hdd_base = cfg["hdd"]["base_temp_C"]

    model, agg_df = fit_hdd_model(sim_df, hdd_base)
    errors_df = compute_peak_errors(sim_df, model, hdd_base)

    out_path = out / "hdd_benchmark.csv"
    logger.info("Saving to %s", out_path)
    errors_df.to_csv(out_path, index=False)

    return errors_df, model
