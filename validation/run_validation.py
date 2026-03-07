"""
Validation runner — orchestrates all validation analyses against EoH field data.

Usage:
    python -m validation.run_validation

Produces:
    outputs/validation/cop/       — COP model vs field data comparison
    outputs/validation/peaks/     — Peak demand distributions and ADMD
    outputs/validation/archetypes/ — Discovered demand archetypes
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from validation.eoh_loader import (
    load_eoh_half_hourly,
    compute_daily_peaks,
)
from validation.cop_validation import validate_cop
from validation.peak_validation import validate_peaks
from validation.archetype_clustering import discover_archetypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("validation")


def main():
    t0 = time.time()

    # Load config
    cfg_path = PROJECT_ROOT / "config" / "simulation_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    hp_cfg = cfg["heat_pump"]

    # Paths
    eoh_hh_path = PROJECT_ROOT / "data" / "external" / "eoh" / "UKDA-30min-Interval-9209-csv" / "csv" / "eoh_cleaned_half_hourly.csv"
    out_base = PROJECT_ROOT / "outputs" / "validation"

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
    cop_stats = validate_cop(eoh_df, hp_cfg, out_base / "cop")

    # ── Step 4: Peak demand validation ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Peak demand validation")
    logger.info("=" * 60)

    # Try to load simulation results if they exist
    sim_path = PROJECT_ROOT / "outputs" / "simulation_results.csv"
    sim_results = None
    if sim_path.exists():
        import pandas as pd
        sim_results = pd.read_csv(sim_path)
        logger.info("Loaded simulation results for comparison (%d rows)", len(sim_results))

    peak_stats = validate_peaks(daily_peaks, sim_results, hp_cfg, out_base / "peaks")

    # ── Step 5: Archetype clustering ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Archetype clustering")
    logger.info("=" * 60)

    # Subsample by property to keep complete property-days intact
    all_props = eoh_df["Property_ID"].unique()
    max_props = 200  # enough for robust clustering, keeps memory manageable
    if len(all_props) > max_props:
        import numpy as np
        rng = np.random.RandomState(42)
        sampled_props = rng.choice(all_props, size=max_props, replace=False)
        eoh_sample = eoh_df[eoh_df["Property_ID"].isin(sampled_props)].copy()
        logger.info("Subsampled %d/%d properties for clustering (%d rows)",
                     max_props, len(all_props), len(eoh_sample))
    else:
        eoh_sample = eoh_df

    archetype_stats = discover_archetypes(eoh_sample, out_base / "archetypes")

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

    import numpy as np

    summary_path = out_base / "validation_summary.json"
    out_base.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=convert)

    elapsed = time.time() - t0
    logger.info("All results saved to %s", out_base)
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
    print(f"  Empirical R²  = {cop_stats['R_squared_empirical']:.3f}")
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

    print(f"\nFull results: {out_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
