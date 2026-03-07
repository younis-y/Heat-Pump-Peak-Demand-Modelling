"""
COP validation: compare modelled COP curve against EoH field trial data.

Bins the real half-hourly COP values by outdoor temperature and compares
against the parametric model COP = 3.5 + 0.12 * T_out (with defrost).
Produces a validation figure and goodness-of-fit statistics.
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
    out_dir: Path,
) -> dict:
    """Run COP validation and produce comparison figure.

    Parameters
    ----------
    eoh_df : pd.DataFrame
        Half-hourly EoH data.
    hp_cfg : dict
        Heat pump config.
    out_dir : Path
        Output directory for figure and stats.

    Returns
    -------
    dict
        Validation statistics: MAE, RMSE, bias, R-squared.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    binned = bin_cop_by_temperature(eoh_df)

    T_range = np.linspace(binned["T_bin_centre"].min() - 1, binned["T_bin_centre"].max() + 1, 200)
    model_cop = modelled_cop_curve(T_range, hp_cfg)

    # Fit empirical COP regression to the real binned medians
    # EoH measures whole-system COP (SPF H4 boundary: compressor + pumps + controls)
    # which is systematically lower than compressor-only COP
    T_bins = binned["T_bin_centre"].values
    cop_real = binned["cop_median"].values
    emp_coeffs = np.polyfit(T_bins, cop_real, 1)  # linear fit
    emp_intercept, emp_slope = emp_coeffs[1], emp_coeffs[0]
    emp_cop_at_bins = np.polyval(emp_coeffs, T_bins)

    # Compute error metrics: model vs real
    model_at_bins = modelled_cop_curve(T_bins.copy(), hp_cfg)
    residuals = cop_real - model_at_bins
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    bias = np.mean(residuals)

    # R² for empirical fit (how well linear model explains temperature-COP)
    emp_residuals = cop_real - emp_cop_at_bins
    ss_res_emp = np.sum(emp_residuals ** 2)
    ss_tot = np.sum((cop_real - cop_real.mean()) ** 2)
    r_squared_empirical = 1 - ss_res_emp / ss_tot if ss_tot > 0 else 0

    # Systematic offset between model and real data
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
        "Empirical COP: %.3f + %.4f*T (R²=%.3f) vs Model: %.1f + %.2f*T",
        emp_intercept, emp_slope, r_squared_empirical,
        hp_cfg["cop_intercept"], hp_cfg["cop_slope"],
    )

    # Save stats
    pd.DataFrame([stats]).to_csv(out_dir / "cop_validation_stats.csv", index=False)
    binned.to_csv(out_dir / "cop_binned_by_temperature.csv", index=False)

    # --- Figure: COP validation ---
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=(9, 6))

    # Real data: median with IQR shading
    ax.fill_between(
        binned["T_bin_centre"], binned["cop_25"], binned["cop_75"],
        alpha=0.25, color="C0", label="EoH field trial IQR (25th-75th)",
    )
    ax.plot(
        binned["T_bin_centre"], binned["cop_median"],
        "o-", color="C0", markersize=4, linewidth=1.5,
        label=f"EoH median COP (n={int(binned['n'].sum()):,})",
    )

    # Model curve (compressor COP)
    ax.plot(
        T_range, model_cop, "--", color="C1", linewidth=2.5,
        label=f"Model (compressor): COP = {hp_cfg['cop_intercept']} + {hp_cfg['cop_slope']}T",
    )

    # Empirical fit to EoH data (whole-system SPF)
    emp_cop_curve = np.polyval(emp_coeffs, T_range)
    ax.plot(
        T_range, emp_cop_curve, "-.", color="C2", linewidth=2,
        label=f"Empirical fit: COP = {emp_intercept:.2f} + {emp_slope:.3f}T"
              f" (R²={r_squared_empirical:.2f})",
    )

    # Defrost threshold
    ax.axvline(
        hp_cfg["defrost_temp_threshold_C"], color="grey", linestyle=":",
        alpha=0.5, label=f"Defrost threshold ({hp_cfg['defrost_temp_threshold_C']} C)",
    )

    ax.set_xlabel("Outdoor temperature (C)")
    ax.set_ylabel("COP")
    ax.set_title("COP Validation: Model vs EoH Field Trial (739 homes, 2020-2023)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(binned["T_bin_centre"].min() - 1, binned["T_bin_centre"].max() + 1)
    ax.set_ylim(0, 6)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_cop_validation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved COP validation figure to %s", out_dir)

    return stats
