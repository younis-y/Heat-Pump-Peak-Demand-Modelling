#!/usr/bin/env python3
"""
Heat Pump Peak Demand Modelling — Master Orchestrator.

Runs the complete pipeline:
1. Load config
2. Run all 24 simulations
3. HDD benchmark
4. ANOVA decomposition
5. Interaction regression
6. Demand surfaces
7. Generate all figures
8. Write master results table
9. Run tests
10. Print hypothesis verdicts

Usage:
    python main.py
    python main.py --debug     # enable DEBUG logging
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulation_config.yaml"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results_tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
for d in [DATA_RAW, DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def setup_logging(debug: bool = False) -> None:
    """Configure root logger.

    Parameters
    ----------
    debug : bool
        If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config() -> dict:
    """Load YAML configuration.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    logging.getLogger(__name__).info("Config loaded from %s", CONFIG_PATH)
    return cfg


def build_archetype_names(cfg: dict) -> dict:
    """Extract archetype ID -> display name mapping.

    Parameters
    ----------
    cfg : dict
        Full config.

    Returns
    -------
    dict
        e.g. {"B1": "Early Riser", ...}
    """
    return {k: v["name"] for k, v in cfg["archetypes"].items()}


def write_master_results(
    sim_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Produce the master results table with one row per simulation run.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output (2304 rows).
    errors_df : pd.DataFrame
        HDD benchmark errors (24 rows).
    output_path : Path
        CSV output path.

    Returns
    -------
    pd.DataFrame
        Master results (24 rows).
    """
    logger = logging.getLogger(__name__)

    # First compute per-replicate daily energy, then average across replicates
    group_cols = ["archetype", "weather_scenario", "fabric"]
    if "replicate" in sim_df.columns:
        group_cols_rep = group_cols + ["replicate"]
    else:
        group_cols_rep = group_cols

    per_run = (
        sim_df.groupby(group_cols_rep)
        .agg(
            peak_demand_kW=("electricity_demand_kW", "max"),
            daily_energy_kWh=("electricity_demand_kW", lambda x: x.sum() * 0.25),
            mean_cop=("COP", "mean"),
            max_indoor_temp=("T_indoor_C", "max"),
            min_indoor_temp=("T_indoor_C", "min"),
        )
        .reset_index()
    )

    # Average across replicates → one summary row per cell
    runs = (
        per_run.groupby(["archetype", "weather_scenario", "fabric"])
        .agg(
            peak_demand_kW=("peak_demand_kW", "mean"),
            daily_energy_kWh=("daily_energy_kWh", "mean"),
            mean_cop=("mean_cop", "mean"),
            max_indoor_temp=("max_indoor_temp", "max"),
            min_indoor_temp=("min_indoor_temp", "min"),
        )
        .reset_index()
    )

    # Merge HDD predictions (errors_df has one row per cell already)
    merge_cols = ["archetype", "weather_scenario", "fabric"]
    hdd_cols = merge_cols + ["hdd_peak_kW", "underestimation_pct"]
    master = runs.merge(
        errors_df[hdd_cols],
        on=merge_cols,
        how="left",
    )
    master.insert(0, "run_id", range(1, len(master) + 1))
    master = master.rename(columns={"hdd_peak_kW": "hdd_prediction_kW"})

    logger.info("Saving to %s", output_path)
    master.to_csv(output_path, index=False)
    return master


def print_findings(
    errors_df: pd.DataFrame,
    anova_table: pd.DataFrame,
    reset_result: dict,
    regression_results: dict,
) -> None:
    """Print a human-readable findings summary to stdout.

    Parameters
    ----------
    errors_df : pd.DataFrame
        HDD benchmark errors.
    anova_table : pd.DataFrame
        ANOVA results with eta_squared.
    reset_result : dict
        RESET test result dict (keys: statistic, p_value).
    regression_results : dict
        Contains model_a, model_b, comparison_df.
    """
    print("\n" + "=" * 72)
    print("  FINDINGS SUMMARY")
    print("=" * 72)

    # --- H1: HDD underestimation increases with cold ---
    # HDD fits the cross-archetype mean well; the underestimation manifests
    # when the worst-case archetype is compared to the HDD average prediction.
    print("\n--- H1: HDD underestimation increases at lower outdoor temperatures ---")
    print("  (worst-case archetype peak vs HDD-predicted average)")
    weather_order = ["W1", "W2", "W3"]
    max_errors = []
    for w in weather_order:
        sub = errors_df[errors_df["weather_scenario"] == w]
        # Worst-case = archetype with highest peak that HDD misses most
        worst = sub.loc[sub["underestimation_pct"].idxmax()]
        mean_t = sub["T_mean_outdoor_C"].mean()
        max_err = sub["underestimation_pct"].max()
        max_errors.append(max_err)
        print(f"  {w}: mean T_out = {mean_t:+.1f}°C  →  max underestimation = {max_err:.1f}%"
              f"  (archetype {worst['archetype']})")

    h1_supported = all(max_errors[i] <= max_errors[i + 1] for i in range(len(max_errors) - 1))
    h1_verdict = "SUPPORTED" if h1_supported else "NOT SUPPORTED"
    print(f"  → H1 verdict: {h1_verdict}")

    # --- H2: Archetype > fabric in explaining peak demand ---
    print("\n--- H2: Occupant behaviour is a stronger predictor than fabric ---")
    non_resid = anova_table[anova_table.index != "Residual"]
    top = non_resid["eta_squared"].idxmax()
    print(f"  Top factor by η²: {top} ({non_resid['eta_squared'].max():.4f})")

    # Find archetype and fabric η²
    arch_eta = 0.0
    fabric_eta = 0.0
    for idx in non_resid.index:
        if idx == "archetype":
            arch_eta = non_resid.loc[idx, "eta_squared"]
        elif idx == "fabric":
            fabric_eta = non_resid.loc[idx, "eta_squared"]

    print(f"  archetype η² = {arch_eta:.4f}")
    print(f"  fabric η²    = {fabric_eta:.4f}")
    h2_supported = arch_eta > fabric_eta
    h2_verdict = "SUPPORTED" if h2_supported else "NOT SUPPORTED"
    print(f"  → H2 verdict: {h2_verdict}")

    # --- H3: Nonlinearity not captured by HDD ---
    print("\n--- H3: Weather–behaviour interaction is nonlinear ---")
    p_val = reset_result["p_value"]
    print(f"  RESET test p-value: {p_val:.6f}")
    comp = regression_results["comparison_df"]
    print(f"  Model A (HDD-linear):    R²={comp.iloc[0]['R_squared']:.4f}  AIC={comp.iloc[0]['AIC']:.1f}")
    print(f"  Model B (Full interact): R²={comp.iloc[1]['R_squared']:.4f}  AIC={comp.iloc[1]['AIC']:.1f}")
    h3_supported = p_val < 0.05
    h3_verdict = "SUPPORTED" if h3_supported else "INCONCLUSIVE"
    print(f"  → H3 verdict: {h3_verdict}")

    print("\n" + "=" * 72)
    print(f"  H1 (HDD underestimates more in cold):       {h1_verdict}")
    print(f"  H2 (Behaviour > Fabric for peak demand):    {h2_verdict}")
    print(f"  H3 (Nonlinear interaction, HDD inadequate): {h3_verdict}")
    print("=" * 72 + "\n")


def run_tests() -> bool:
    """Run pytest test suite.

    Returns
    -------
    bool
        True if all tests passed.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running test suite...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        logger.warning("Some tests FAILED.")
        return False
    logger.info("All tests PASSED.")
    return True


# ===================================================================== #
#                              MAIN                                      #
# ===================================================================== #

def main() -> None:
    """Run the full pipeline end-to-end."""
    parser = argparse.ArgumentParser(description="Heat Pump Peak Demand Modelling Pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    # ---- 1. Load config ----
    cfg = load_config()
    archetype_names = build_archetype_names(cfg)
    np.random.seed(cfg["random_seed"])

    # ---- 1b. Save raw weather profiles for reproducibility ----
    from simulation.ochre_runner import generate_weather, run_all_simulations
    from simulation.epw_parser import load_weather_profiles

    epw_rel = cfg.get("epw_file")
    if epw_rel:
        epw_path = PROJECT_ROOT / epw_rel
        weather_days = {}
        for w_id, w_cfg in cfg["weather"].items():
            if "epw_month" in w_cfg and "epw_day" in w_cfg:
                weather_days[w_id] = {"month": w_cfg["epw_month"], "day": w_cfg["epw_day"]}
        if weather_days and epw_path.exists():
            profiles = load_weather_profiles(epw_path, weather_days, cfg["dt_minutes"])
            for w_id, wdf in profiles.items():
                raw_path = DATA_RAW / f"weather_{w_id}.csv"
                logger.info("Saving real EPW profile to %s", raw_path)
                wdf.to_csv(raw_path, index=False)
        else:
            logger.warning("EPW file not found, saving synthetic profiles")
            for w_id, w_cfg in cfg["weather"].items():
                weather_df = generate_weather(
                    w_cfg, timesteps=cfg["timesteps"],
                    dt_minutes=cfg["dt_minutes"], seed=cfg["random_seed"]
                )
                raw_path = DATA_RAW / f"weather_{w_id}.csv"
                logger.info("Saving to %s", raw_path)
                weather_df.to_csv(raw_path, index=False)
    else:
        for w_id, w_cfg in cfg["weather"].items():
            weather_df = generate_weather(
                w_cfg, timesteps=cfg["timesteps"],
                dt_minutes=cfg["dt_minutes"], seed=cfg["random_seed"]
            )
            raw_path = DATA_RAW / f"weather_{w_id}.csv"
            logger.info("Saving to %s", raw_path)
            weather_df.to_csv(raw_path, index=False)

    # ---- 2. Run all simulations ----
    n_rep = cfg.get("replicates", 1)
    n_cells = len(cfg["archetypes"]) * len(cfg["weather"]) * len(cfg["fabric"])
    total = n_cells * n_rep
    logger.info("=== Step 2: Running %d simulations ===", total)
    sim_df = run_all_simulations(cfg, project_root=PROJECT_ROOT)

    sim_path = DATA_PROCESSED / "simulation_results.csv"
    logger.info("Saving to %s", sim_path)
    sim_df.to_csv(sim_path, index=False)

    # ---- 3. HDD benchmark ----
    from benchmarks.hdd_regression import run_hdd_benchmark

    logger.info("=== Step 3: HDD benchmark ===")
    errors_df, hdd_model = run_hdd_benchmark(sim_df, cfg, RESULTS_DIR)

    # ---- 4. ANOVA decomposition ----
    from analytics.anova_decomposition import run_anova

    logger.info("=== Step 4: ANOVA decomposition ===")
    anova_table = run_anova(sim_df, RESULTS_DIR)

    # ---- 5. Interaction regression ----
    from analytics.interaction_regression import run_interaction_regression

    logger.info("=== Step 5: Interaction regression ===")
    regression_results = run_interaction_regression(sim_df, cfg, RESULTS_DIR)

    # ---- 6. Demand surfaces ----
    from analytics.demand_surface import build_demand_surfaces

    logger.info("=== Step 6: Demand surfaces ===")
    surfaces = build_demand_surfaces(sim_df)

    # ---- 7. Generate all figures ----
    from visualisation.plots import generate_all_figures

    logger.info("=== Step 7: Generating figures ===")
    generate_all_figures(
        sim_df=sim_df,
        errors_df=errors_df,
        anova_table=anova_table,
        surfaces=surfaces,
        hp_cfg=cfg["heat_pump"],
        archetype_names=archetype_names,
        out_dir=FIGURES_DIR,
    )

    # ---- 8. Master results table ----
    logger.info("=== Step 8: Master results table ===")
    master_df = write_master_results(sim_df, errors_df, RESULTS_DIR / "master_results.csv")

    # ---- 9. Run tests ----
    logger.info("=== Step 9: Running tests ===")
    run_tests()

    # ---- 10. Print findings ----
    print_findings(errors_df, anova_table, regression_results["reset_test"], regression_results)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
