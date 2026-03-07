"""
Publication-quality figure generation.

All figures use:
- Font: DejaVu Sans, 10 pt body / 12 pt titles
- Colour palette: seaborn "colorblind" (accessible)
- DPI: 300
- Dual export: PNG + PDF
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
PALETTE = sns.color_palette("colorblind")
DPI = 300

def _apply_style() -> None:
    """Set the global matplotlib style for all figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })
    sns.set_palette("colorblind")


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save figure as PNG and PDF.

    Parameters
    ----------
    fig : matplotlib Figure
    out_dir : Path
        Output directory.
    name : str
        Base filename without extension.
    """
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        logger.info("Saving to %s", p)
        fig.savefig(p, dpi=DPI)
    plt.close(fig)


# ===================================================================== #
#  Fig 1 — Time-series: 4 archetypes under W2, F1                       #
# ===================================================================== #

def fig1_timeseries(
    sim_df: pd.DataFrame,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """Plot electricity demand time-series for all archetypes under design cold / F1.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    archetype_names : dict
        Mapping archetype_id -> display name.
    out_dir : Path
        Figure output directory.
    """
    _apply_style()
    subset = sim_df[
        (sim_df["weather_scenario"] == "W2") & (sim_df["fabric"] == "F1")
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (arch_id, grp) in enumerate(subset.groupby("archetype")):
        hours = np.arange(len(grp)) * 0.25
        ax.plot(hours, grp["electricity_demand_kW"].values,
                label=archetype_names.get(arch_id, arch_id),
                color=PALETTE[i], linewidth=1.2)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Electricity demand (kW)")
    ax.set_title("Fig 1: Heat pump electricity demand — Design Cold (W2), Unimproved fabric (F1)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 24)
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "fig1_timeseries")


# ===================================================================== #
#  Fig 2 — COP curve                                                     #
# ===================================================================== #

def fig2_cop_curve(
    sim_df: pd.DataFrame,
    hp_cfg: dict,
    out_dir: Path,
) -> None:
    """Scatter of simulated COP vs T_out with parametric curve overlay.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    hp_cfg : dict
        Heat pump config section.
    out_dir : Path
        Figure output directory.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter from simulation (sample to avoid over-plotting)
    sample = sim_df.sample(min(500, len(sim_df)), random_state=42)
    ax.scatter(sample["T_outdoor_C"], sample["COP"], alpha=0.4, s=12,
               color=PALETTE[0], label="Simulated")

    # Parametric curve
    t_range = np.linspace(sim_df["T_outdoor_C"].min() - 2, sim_df["T_outdoor_C"].max() + 2, 200)
    cop_curve = hp_cfg["cop_intercept"] + hp_cfg["cop_slope"] * t_range
    # Apply defrost penalty
    defrost_mask = t_range < hp_cfg["defrost_temp_threshold_C"]
    cop_curve[defrost_mask] *= (1.0 - hp_cfg["defrost_efficiency_penalty"])
    cop_curve = np.clip(cop_curve, hp_cfg["cop_min"], hp_cfg["cop_max"])

    ax.plot(t_range, cop_curve, color=PALETTE[1], linewidth=2, label="Parametric COP curve")

    ax.set_xlabel("Outdoor temperature (°C)")
    ax.set_ylabel("COP")
    ax.set_title("Fig 2: Heat pump COP vs outdoor temperature")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "fig2_cop_curve")


# ===================================================================== #
#  Fig 3 — HDD underestimation error vs T_out                           #
# ===================================================================== #

def fig3_hdd_error(
    errors_df: pd.DataFrame,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """Scatter of underestimation % vs mean outdoor temperature, coloured by archetype.

    Parameters
    ----------
    errors_df : pd.DataFrame
        From hdd_benchmark (columns: archetype, T_mean_outdoor_C, underestimation_pct).
    archetype_names : dict
        Mapping archetype_id -> display name.
    out_dir : Path
        Figure output directory.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    archetypes = sorted(errors_df["archetype"].unique())
    for i, arch in enumerate(archetypes):
        sub = errors_df[errors_df["archetype"] == arch]
        ax.scatter(sub["T_mean_outdoor_C"], sub["underestimation_pct"],
                   color=PALETTE[i], s=60, label=archetype_names.get(arch, arch),
                   edgecolors="black", linewidth=0.5, zorder=3)

    # Overall regression line
    from numpy.polynomial.polynomial import polyfit, polyval
    coeffs = polyfit(errors_df["T_mean_outdoor_C"], errors_df["underestimation_pct"], 1)
    t_line = np.linspace(errors_df["T_mean_outdoor_C"].min(), errors_df["T_mean_outdoor_C"].max(), 50)
    ax.plot(t_line, polyval(t_line, coeffs), "--", color="grey", linewidth=1.5, label="Trend")

    ax.set_xlabel("Mean outdoor temperature (°C)")
    ax.set_ylabel("HDD underestimation of peak demand (%)")
    ax.set_title("Fig 3: HDD peak-demand underestimation vs outdoor temperature")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir, "fig3_hdd_error")


# ===================================================================== #
#  Fig 4 — ANOVA η² bar chart                                           #
# ===================================================================== #

def fig4_anova_eta(
    anova_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Bar chart of eta-squared values from ANOVA, ranked by magnitude.

    Parameters
    ----------
    anova_table : pd.DataFrame
        ANOVA results with eta_squared column (index = factor names).
    out_dir : Path
        Figure output directory.
    """
    _apply_style()

    # Exclude Residual
    data = anova_table[anova_table.index != "Residual"].copy()
    data = data.sort_values("eta_squared", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(data.index, data["eta_squared"], color=PALETTE[:len(data)], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("η² (proportion of total variance)")
    ax.set_title("Fig 4: ANOVA variance decomposition of peak demand")
    ax.grid(axis="x", alpha=0.3)

    # Value labels
    for bar, val in zip(bars, data["eta_squared"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    _save(fig, out_dir, "fig4_anova_eta")


# ===================================================================== #
#  Fig 5 — Demand surface heatmap (2×2 archetypes)                      #
# ===================================================================== #

def fig5_demand_surface(
    surfaces: dict,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """2×2 heatmap grid of demand surfaces per archetype.

    Parameters
    ----------
    surfaces : dict
        From demand_surface.build_demand_surfaces(). Keys: archetype IDs.
    archetype_names : dict
        Mapping archetype_id -> display name.
    out_dir : Path
        Figure output directory.
    """
    _apply_style()
    arch_ids = sorted(surfaces.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, arch_id in enumerate(arch_ids):
        ax = axes[idx]
        surf = surfaces[arch_id]
        mat = surf["matrix"]
        temps = surf["temp_bins"]
        hours = surf["hours"]

        im = ax.pcolormesh(
            temps, hours, mat,
            cmap="YlOrRd", shading="auto",
        )
        ax.set_title(archetype_names.get(arch_id, arch_id))
        ax.set_ylabel("Hour of day")
        ax.set_xlabel("Outdoor temperature (°C)")
        fig.colorbar(im, ax=ax, label="Demand (kW)")

    fig.suptitle("Fig 5: Demand surface — electricity demand by hour and outdoor temperature", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir, "fig5_demand_surface_heatmap")


# ===================================================================== #
#  Fig 6 — Peak demand box plots                                        #
# ===================================================================== #

def fig6_peak_boxplots(
    sim_df: pd.DataFrame,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """Box plots of peak demand per archetype × weather, grouped by fabric.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    archetype_names : dict
        Mapping archetype_id -> display name.
    out_dir : Path
        Figure output directory.
    """
    _apply_style()

    # Get per-run peaks (one per replicate if replication is present)
    group_cols = ["archetype", "weather_scenario", "fabric"]
    if "replicate" in sim_df.columns:
        group_cols.append("replicate")
    peaks = (
        sim_df.groupby(group_cols)
        .agg(peak_demand_kW=("electricity_demand_kW", "max"))
        .reset_index()
    )
    peaks["archetype_name"] = peaks["archetype"].map(archetype_names)
    peaks["group"] = peaks["archetype_name"] + " / " + peaks["weather_scenario"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=peaks,
        x="group",
        y="peak_demand_kW",
        hue="fabric",
        ax=ax,
        palette=[PALETTE[0], PALETTE[2]],
    )
    ax.set_xlabel("")
    ax.set_ylabel("Peak electricity demand (kW)")
    ax.set_title("Fig 6: Peak demand by archetype, weather scenario, and fabric condition")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Fabric")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "fig6_peak_boxplots")


# ===================================================================== #
#  Master function — generate all figures                                #
# ===================================================================== #

def generate_all_figures(
    sim_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    anova_table: pd.DataFrame,
    surfaces: dict,
    hp_cfg: dict,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """Generate all six publication-quality figures.

    Parameters
    ----------
    sim_df : pd.DataFrame
        Full simulation output.
    errors_df : pd.DataFrame
        HDD benchmark errors.
    anova_table : pd.DataFrame
        ANOVA results.
    surfaces : dict
        Demand surfaces from demand_surface module.
    hp_cfg : dict
        Heat pump config.
    archetype_names : dict
        Mapping archetype_id -> display name.
    out_dir : Path
        Figure output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_timeseries(sim_df, archetype_names, out_dir)
    fig2_cop_curve(sim_df, hp_cfg, out_dir)
    fig3_hdd_error(errors_df, archetype_names, out_dir)
    fig4_anova_eta(anova_table, out_dir)
    fig5_demand_surface(surfaces, archetype_names, out_dir)
    fig6_peak_boxplots(sim_df, archetype_names, out_dir)

    logger.info("All 6 figures generated in %s", out_dir)
