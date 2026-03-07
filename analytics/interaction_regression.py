"""
Interaction regression analysis comparing linear HDD model vs full
interaction model to test H3 (nonlinearity of weather–behaviour interaction).

Model A (HDD-linear):
    Demand = β0 + β1×HDD + ε

Model B (Full interaction):
    Demand = β0 + β1×T_out + β2×T_out² + β3×archetype
             + β4×(T_out × archetype) + β5×fabric + ε

With Monte Carlo replication, each model is fitted on 240 data points
(24 cells × 10 replicates) rather than 24, giving much more robust
R², AIC, and residual diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import linear_reset

logger = logging.getLogger(__name__)


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
    output_dir: Path,
) -> dict:
    """Full interaction regression pipeline.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    cfg : dict
        Full config.
    output_dir : Path
        Directory for results tables.

    Returns
    -------
    dict
        Keys: model_a, model_b, comparison_df, reset_test, reg_df.
    """
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

    out_path = output_dir / "model_comparison.csv"
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
