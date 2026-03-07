"""
Three-way ANOVA variance decomposition on peak electricity demand.

With Monte Carlo replication (N per cell), we now have proper residual
degrees of freedom, enabling reliable F-tests for all main effects and
two-way interactions.

Tests whether occupant behaviour archetype explains more variance in peak
demand than building fabric quality (H2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

logger = logging.getLogger(__name__)


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
    output_dir: Path,
) -> pd.DataFrame:
    """Run the 3-way ANOVA and save results.

    With replication, we use all main effects and two-way interactions
    with the three-way interaction + replicates as residual, giving
    proper degrees of freedom for F-tests.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    output_dir : Path
        Directory for results tables.

    Returns
    -------
    pd.DataFrame
        ANOVA table with eta_squared column added.
    """
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

    out_path = output_dir / "anova_results.csv"
    logger.info("Saving to %s", out_path)
    anova_table.to_csv(out_path)

    # Log top factor
    non_residual = anova_table[anova_table.index != "Residual"]
    top_factor = non_residual["eta_squared"].idxmax()
    logger.info(
        "Top factor by η²: %s (%.4f)", top_factor, non_residual["eta_squared"].max()
    )

    return anova_table
