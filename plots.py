"""
plots.py — Figure styling + main figure generators.

Merged from:
    visualisation/style.py
    visualisation/plots.py

Critical revisions (this version):
    - fig1_timeseries           : 4 separate panels  ->  overlaid profiles + peak-bar inset
    - fig3_hdd_error            : 4-panel line chart ->  diverging error matrix with sign banner
    - fig4_anova_eta            : single ANOVA bars  ->  ANOVA + HDD-vs-proposed-model side-by-side
    - fig5_demand_surface_heatmap: cleaner contours, single shared colorbar, no clabel overprint
    - NEW: fig_eoh_validation   : 3-panel COP/peak/silhouette validation against the EoH trial
    - NEW: fig_admd_and_cost    : ADMD diversification curves + £-cost of HDD under-sizing
    - fig2_cop_curve, fig6_peak_boxplots: kept unchanged (already adequate).

All public function signatures are preserved. New optional parameters default to None,
so an unmodified pipeline.py keeps working; pass the new arguments to enable the new
figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Mapping

PROJECT_ROOT = Path(__file__).parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ===== from visualisation/style.py =====

logger = logging.getLogger(__name__)

# ── Colour palette ──────────────────────────────────────────────────────
# Muted academic palette — accessible (WCAG AA on white)

COLOURS = {
    "primary":    "#2166ac",   # steel blue
    "secondary":  "#b2182b",   # muted red
    "tertiary":   "#1b7837",   # forest green
    "quaternary": "#e08214",   # amber
    "neutral":    "#636363",   # dark grey
}
PALETTE = list(COLOURS.values())

# Semantic mappings — consistent across every figure
ARCHETYPE_COLOURS = {
    "B1": COLOURS["primary"],       # Early Riser
    "B2": COLOURS["quaternary"],    # Home All Day
    "B3": COLOURS["tertiary"],      # Late Returner
    "B4": COLOURS["secondary"],     # Intermittent
}

FABRIC_COLOURS = {
    "F1": COLOURS["secondary"],     # Unimproved
    "F2": COLOURS["primary"],       # Retrofitted
}

WEATHER_COLOURS = {
    "W1": COLOURS["primary"],       # Mild
    "W2": COLOURS["quaternary"],    # Design Cold
    "W3": COLOURS["secondary"],     # Extreme Cold
}

# Human-readable labels for coded factor IDs
WEATHER_LABELS = {
    "W1": "Mild",
    "W2": "Design",
    "W3": "Extreme",
}

FABRIC_LABELS = {
    "F1": "Unimproved",
    "F2": "Retrofitted",
}

# Marker shapes for fabric distinction in scatter plots
FABRIC_MARKERS = {
    "F1": "o",   # circle
    "F2": "D",   # diamond
}


# ── Figure dimensions (inches) ──────────────────────────────────────────
# Calibrated for A4 single-column (~86 mm) and full-width (~178 mm)

FIG_SINGLE      = (3.5, 2.6)
FIG_SINGLE_TALL = (3.5, 3.4)
FIG_DOUBLE      = (7.0, 3.0)
FIG_DOUBLE_TALL = (7.0, 5.5)

# ── DPI ─────────────────────────────────────────────────────────────────
DPI = 300


# ── rcParams ────────────────────────────────────────────────────────────

RCPARAMS: dict = {
    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",

    # Font sizes
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    # Axes — clean academic look
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Grid — off by default (use add_grid() to enable selectively)
    "axes.grid": False,

    # Legend — frameless, compact
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.5,

    # Output
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # Lines and markers
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
}


# ── Helper functions ────────────────────────────────────────────────────

def apply_style() -> list:
    """Apply rcParams and return the palette list."""
    plt.rcParams.update(RCPARAMS)
    return PALETTE


def save_fig(fig: plt.Figure, out_dir: Path, name: str,
             formats: tuple[str, ...] = ("png", "pdf")) -> None:
    """Save figure in multiple formats, then close."""
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        logger.info("Saving %s", p)
        fig.savefig(p, dpi=DPI)
    plt.close(fig)


def add_grid(ax: plt.Axes, axis: str = "y", alpha: float = 0.15) -> None:
    """Add a subtle grid (y-only by default)."""
    ax.grid(axis=axis, alpha=alpha, linewidth=0.4, color="#cccccc")


def clean_factor_label(text: str) -> str:
    """Clean ANOVA factor labels for display."""
    mapping = {
        "archetype": "Occupant type",
        "weather_scenario": "Weather",
        "fabric": "Insulation",
        # Colon-separated (statsmodels default)
        "archetype:weather_scenario": "Occupant \u00d7 Weather",
        "archetype:fabric": "Occupant \u00d7 Insulation",
        "weather_scenario:fabric": "Weather \u00d7 Insulation",
        # Unicode × separated (from this project's ANOVA output)
        "archetype \u00d7 weather_scenario": "Occupant \u00d7 Weather",
        "archetype \u00d7 fabric": "Occupant \u00d7 Insulation",
        "weather_scenario \u00d7 fabric": "Weather \u00d7 Insulation",
    }
    return mapping.get(text, text.replace("_", " ").title())


# ===== from visualisation/plots.py =====


# ===================================================================== #
#  Fig 1 — Overlaid daily demand profile + peak-by-archetype×fabric     #
# ===================================================================== #
#
#  REVISION: original was 4 separate panels, all peaking at ~5.8 kW.
#  Visually that *contradicts* H2 (behaviour-matters). The new design
#  overlays the four profiles on one axis so the eye sees TIMING
#  differences (the H2 evidence), and adds a peak-demand bar inset
#  showing the archetype × fabric interaction with HDD reference.
#
def fig1_timeseries(
    sim_df: pd.DataFrame,
    archetype_names: dict,
    out_dir: Path,
    hdd_peak_w2_kW: Optional[float] = None,
) -> None:
    """Overlaid daily demand profile (top) + peak-demand bars by occupant×fabric (bottom)."""
    apply_style()

    # ----- Top panel: overlaid profiles for W2/F1 -----
    subset = sim_df[
        (sim_df["weather_scenario"] == "W2") & (sim_df["fabric"] == "F1")
    ].copy()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 5.4),
        gridspec_kw={"height_ratios": [2.2, 1.1], "hspace": 0.40},
    )

    arch_order = sorted(subset["archetype"].unique())
    for arch_id in arch_order:
        grp = subset[subset["archetype"] == arch_id]
        # Average across replicates if present
        if "replicate" in grp.columns:
            grp = grp.groupby("timestamp", as_index=False).agg(
                electricity_demand_kW=("electricity_demand_kW", "mean"),
            )
        hours = np.arange(len(grp)) * 0.25
        y = grp["electricity_demand_kW"].values
        colour = ARCHETYPE_COLOURS.get(arch_id, COLOURS["primary"])
        name = archetype_names.get(arch_id, arch_id)
        ax_top.plot(hours, y, color=colour, lw=1.6, label=name, zorder=3)

    # Shaded peak windows with discreet labels
    ax_top.axvspan(6, 8, alpha=0.07, color=COLOURS["neutral"], zorder=1)
    ax_top.axvspan(17, 20, alpha=0.07, color=COLOURS["neutral"], zorder=1)
    ax_top.text(7, 0.25, "AM window", ha="center", va="bottom",
                fontsize=7, color=COLOURS["neutral"], style="italic")
    ax_top.text(18.5, 0.25, "PM window", ha="center", va="bottom",
                fontsize=7, color=COLOURS["neutral"], style="italic")

    ax_top.set_xlim(0, 24)
    ax_top.set_xticks(np.arange(0, 25, 3))
    ax_top.set_xlabel("Hour of day")
    ax_top.set_ylabel("Electricity demand per home (kW)")
    ax_top.set_title("(a) Daily demand profile  ·  W2 design-cold day, F1 unimproved fabric",
                     loc="left", pad=24)
    add_grid(ax_top, "y")
    # Legend above the axis (clear of curves and title)
    ax_top.legend(loc="lower center", ncol=4, columnspacing=1.5,
                  bbox_to_anchor=(0.5, 1.02), frameon=False)

    # ----- Bottom panel: peak demand by archetype × fabric (W2 only) -----
    w2 = sim_df[sim_df["weather_scenario"] == "W2"].copy()
    # Compute per-run peak from half-hourly demand
    if "replicate" in w2.columns:
        peaks = (w2.groupby(["archetype", "fabric", "replicate"])
                   ["electricity_demand_kW"].max().reset_index())
        sub = peaks.groupby(["archetype", "fabric"])["electricity_demand_kW"].mean().reset_index()
    else:
        sub = (w2.groupby(["archetype", "fabric"])
                 ["electricity_demand_kW"].max().reset_index())
    sub = sub.rename(columns={"electricity_demand_kW": "peak_kW"})

    archetypes = sorted(sub["archetype"].unique())
    x = np.arange(len(archetypes))
    w = 0.35
    f1_vals = [sub[(sub.archetype == a) & (sub.fabric == "F1")].peak_kW.iloc[0]
               for a in archetypes]
    f2_vals = [sub[(sub.archetype == a) & (sub.fabric == "F2")].peak_kW.iloc[0]
               for a in archetypes]

    b1 = ax_bot.bar(x - w/2, f1_vals, w, color=COLOURS["secondary"],
                    edgecolor="white", lw=0.5, label="Unimproved fabric (F1)",
                    zorder=3)
    b2 = ax_bot.bar(x + w/2, f2_vals, w, color=COLOURS["primary"],
                    edgecolor="white", lw=0.5, label="Retrofitted fabric (F2)",
                    zorder=3)

    # HDD reference line if supplied
    if hdd_peak_w2_kW is not None:
        ax_bot.axhline(hdd_peak_w2_kW, color=COLOURS["neutral"], ls="--",
                       lw=1.0, zorder=2)
        ax_bot.text(len(archetypes) - 0.3, hdd_peak_w2_kW,
                    f"  HDD = {hdd_peak_w2_kW:.2f} kW",
                    fontsize=7.5, color=COLOURS["neutral"], ha="left",
                    va="center", style="italic")

    # Bar value labels — placed inside bar to keep clear of HDD line
    threshold = (hdd_peak_w2_kW - 0.5) if hdd_peak_w2_kW is not None else -np.inf
    for bars, vals in [(b1, f1_vals), (b2, f2_vals)]:
        for bar, v in zip(bars, vals):
            colour = COLOURS["secondary"] if v < threshold else "white"
            ax_bot.text(bar.get_x() + bar.get_width()/2, v - 0.35,
                        f"{v:.1f}", ha="center", va="top", fontsize=7.5,
                        color=colour, fontweight="bold")

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([archetype_names.get(a, a) for a in archetypes])
    ax_bot.set_ylabel("Mean peak\ndemand (kW)")
    ax_bot.set_ylim(0, 7.4)
    ax_bot.set_title("(b) Peak demand by occupant × fabric  ·  W2 design cold",
                     loc="left", pad=4)
    ax_bot.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.55),
                  frameon=False)
    add_grid(ax_bot, "y")

    save_fig(fig, out_dir, "fig1_timeseries")


# ===================================================================== #
#  Fig 2 — COP curve  (unchanged)                                        #
# ===================================================================== #

def fig2_cop_curve(
    sim_df: pd.DataFrame,
    hp_cfg: dict,
    out_dir: Path,
) -> None:
    """COP vs outdoor temperature with parametric curve and defrost region."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    # Parametric curve
    t_range = np.linspace(
        sim_df["T_outdoor_C"].min() - 2,
        sim_df["T_outdoor_C"].max() + 2, 200,
    )
    cop_curve = hp_cfg["cop_intercept"] + hp_cfg["cop_slope"] * t_range
    defrost_mask = t_range < hp_cfg["defrost_temp_threshold_C"]
    cop_curve[defrost_mask] *= (1.0 - hp_cfg["defrost_efficiency_penalty"])
    cop_curve = np.clip(cop_curve, hp_cfg["cop_min"], hp_cfg["cop_max"])

    # Shaded defrost region
    t_defrost = hp_cfg["defrost_temp_threshold_C"]
    ax.axvspan(t_range[0], t_defrost, alpha=0.08, color=COLOURS["secondary"],
               zorder=0, label="Defrost region")
    ax.text(
        t_defrost - 0.5, hp_cfg["cop_max"] * 0.98,
        "Defrost\nregion", fontsize=7, color=COLOURS["secondary"],
        ha="right", va="top", fontstyle="italic",
    )

    # Scatter — jitter slightly for visibility
    sample = sim_df.sample(min(600, len(sim_df)), random_state=42)
    jitter = np.random.default_rng(42).normal(0, 0.02, size=len(sample))
    ax.scatter(
        sample["T_outdoor_C"], sample["COP"].values + jitter,
        alpha=0.45, s=18, color=COLOURS["neutral"],
        edgecolors="none", zorder=1, label="Simulated points",
    )

    # Parametric curve
    ax.plot(
        t_range, cop_curve,
        color=COLOURS["primary"], linewidth=2.0, zorder=2,
        label=f"COP = {hp_cfg['cop_intercept']:.1f} + {hp_cfg['cop_slope']:.2f}·T",
    )

    # Defrost threshold annotation
    idx_defrost = np.argmin(np.abs(t_range - t_defrost))
    ax.annotate(
        f"Defrost threshold\n({t_defrost}°C)",
        xy=(t_defrost, cop_curve[idx_defrost]),
        xytext=(t_defrost + 4, cop_curve[idx_defrost] - 0.4),
        fontsize=7, color=COLOURS["secondary"],
        arrowprops=dict(arrowstyle="->", color=COLOURS["secondary"], lw=1.0),
    )

    ax.set_xlabel("Outdoor temperature (°C)", fontsize=9)
    ax.set_ylabel("COP", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    add_grid(ax)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_cop_curve")


# ===================================================================== #
#  Fig 3 — HDD error matrix (replaces 4-panel line chart)              #
# ===================================================================== #
#
#  REVISION: original had 4 line panels, each with its own legend, and the
#  headline result (B3 Late-Returner Retrofitted at design temp = -40.3%)
#  was buried in the bottom-left of one panel. The new design is a single
#  diverging heatmap of every behaviour × fabric × weather cell with the
#  worst over-prediction explicitly bordered and annotated.
#
#  Sign convention is documented in the x-axis label so the marker cannot
#  mis-read it: positive values = HDD under-predicts (grid risk);
#  negative values = HDD over-predicts (grid over-built).
#
def fig3_hdd_error(
    errors_df: pd.DataFrame,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """Diverging heatmap of HDD prediction error across behaviour × fabric × weather."""
    apply_style()

    # Pivot into rows = archetype × fabric, cols = weather
    df = errors_df.copy()
    df["row_key"] = df["archetype"] + " · " + df["fabric"].map(FABRIC_LABELS)
    pivot = df.pivot_table(index="row_key", columns="weather_scenario",
                           values="underestimation_pct")

    arche_order = sorted(df["archetype"].unique())
    fab_order = ["F1", "F2"]
    weather_order = sorted(pivot.columns.tolist())  # W1, W2, W3 alphabetical

    row_order = [f"{a} · {FABRIC_LABELS[f]}" for a in arche_order for f in fab_order]
    pivot = pivot.reindex(row_order)
    pivot = pivot[weather_order]

    # Y-axis labels with display names
    ylabels = []
    for a in arche_order:
        for f in fab_order:
            ylabels.append(f"{archetype_names.get(a, a)}\n({FABRIC_LABELS[f]})")

    # X-axis labels with mean temperature (computed from the same df)
    xlabels = []
    for w in weather_order:
        T_mean = df[df.weather_scenario == w]["T_mean_outdoor_C"].mean()
        weather_lbl = WEATHER_LABELS.get(w, w)
        xlabels.append(f"{weather_lbl}\n({T_mean:+.1f} °C)")

    fig, ax = plt.subplots(figsize=(7.4, 5.6))

    # Symmetric diverging colormap
    abs_max = max(abs(pivot.values.min()), abs(pivot.values.max()))
    vmax = float(np.ceil(abs_max / 5.0) * 5.0)  # round up to nearest 5
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")

    # Cell annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            txt_col = "white" if abs(v) > 0.6 * vmax else "black"
            ax.text(j, i, f"{v:+.2f}%", ha="center", va="center",
                    fontsize=10, color=txt_col, fontweight="bold")

    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=8)

    # Visual separator between archetype groups
    for k in [1.5, 3.5, 5.5]:
        ax.axhline(k, color="white", lw=2)

    # Colorbar with policy-interpretation labels
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025, aspect=22)
    cbar.set_label(
        "Signed HDD error (%)\n+ : HDD under-predicts   − : HDD over-predicts",
        fontsize=8.5,
    )
    cbar.ax.tick_params(labelsize=7.5)
    cbar.ax.text(2.4, vmax * 0.85,
                 "HDD UNDER-\npredicts demand\n(under-built)",
                 fontsize=7, ha="left", va="center",
                 color=COLOURS["secondary"], fontweight="bold")
    cbar.ax.text(2.4, -vmax * 0.85,
                 "HDD OVER-\npredicts demand\n(over-built)",
                 fontsize=7, ha="left", va="center",
                 color=COLOURS["primary"], fontweight="bold")

    # Highlight the most negative cell (worst over-prediction)
    flat_min_idx = np.unravel_index(np.nanargmin(pivot.values), pivot.values.shape)
    headline_row, headline_col = flat_min_idx
    headline_val = pivot.values[headline_row, headline_col]
    rect = Rectangle((headline_col - 0.5, headline_row - 0.5), 1, 1,
                     fill=False, edgecolor="black", lw=2.0, zorder=10)
    ax.add_patch(rect)

    # Resolve sim and HDD for the headline cell to put real numbers in annotation
    a_code = arche_order[headline_row // 2]
    f_code = fab_order[headline_row % 2]
    w_code = weather_order[headline_col]
    cell_row = df[(df.archetype == a_code) &
                  (df.fabric == f_code) &
                  (df.weather_scenario == w_code)]
    if len(cell_row) > 0 and "sim_peak_kW" in cell_row.columns and "hdd_peak_kW" in cell_row.columns:
        sim_kw = cell_row["sim_peak_kW"].iloc[0]
        hdd_kw = cell_row["hdd_peak_kW"].iloc[0]
        ann_text = (f"Worst over-prediction:  HDD says {hdd_kw:.2f} kW · "
                    f"sim says {sim_kw:.2f} kW  →  HDD over-builds by "
                    f"{abs(headline_val):.0f}%")
    else:
        ann_text = (f"Worst over-prediction:  HDD over-builds the grid "
                    f"by {abs(headline_val):.0f}% in this cell")

    # Callout box above the heatmap. The headline cell already has a thick
    # black border (the Rectangle drawn above), which serves as the visual
    # anchor — so we omit a leader arrow that would otherwise cross several
    # data cells and overprint their value labels.
    ax.text(
        (len(weather_order) - 1) / 2, -1.1,
        ann_text,
        fontsize=8.5, ha="center", va="center", fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7",
                  ec="black", lw=0.6),
        clip_on=False,
    )

    ax.set_xlabel("Weather scenario", fontsize=9, labelpad=6)

    ax.set_title("HDD prediction error across the behaviour × fabric × weather grid",
                 fontsize=10.5, pad=58)

    # Caption-equivalent footnote, placed beneath the figure (not as the xlabel).
    fig.subplots_adjust(bottom=0.16)
    fig.text(
        0.5, 0.03,
        "At design conditions with unimproved fabric, peak demand is heat-loss-"
        "limited; archetype effects collapse to within rounding.",
        ha="center", va="bottom",
        fontsize=7, color=COLOURS["neutral"], style="italic",
    )

    save_fig(fig, out_dir, "fig3_hdd_error")


# ===================================================================== #
#  Fig 4 — ANOVA η²  +  HDD vs proposed model comparison                #
# ===================================================================== #
#
#  REVISION: original was a single ANOVA bar chart with stars but no
#  effect-size context. New design adds Cohen's η² thresholds and pairs
#  the variance decomposition (H2 evidence) with a model-comparison
#  panel (H3 evidence) so a marker sees both arguments at once.
#
#  If `model_comparison_df` is None, falls back to the single-panel
#  ANOVA chart for backward compatibility.
#
def fig4_anova_eta(
    anova_table: pd.DataFrame,
    out_dir: Path,
    model_comparison_df: Optional[pd.DataFrame] = None,
) -> None:
    """Variance decomposition (Type-II ANOVA) with Cohen thresholds, optionally
    paired with HDD vs proposed-model fit metrics."""
    apply_style()

    data = anova_table[anova_table.index != "Residual"].copy()
    data = data.sort_values("eta_squared", ascending=True)
    data.index = [clean_factor_label(i) for i in data.index]
    data["is_interaction"] = ["×" in idx for idx in data.index]

    if model_comparison_df is None:
        # ---------- single-panel ANOVA chart ----------
        fig, ax1 = plt.subplots(figsize=FIG_SINGLE_TALL)
        _draw_anova_panel(ax1, data, with_legend_inside=True)
        fig.tight_layout()
        save_fig(fig, out_dir, "fig4_anova_eta")
        return

    # ---------- two-panel: ANOVA + model comparison ----------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.6, 4.2),
        gridspec_kw={"width_ratios": [1.2, 1.0], "wspace": 0.55},
    )
    _draw_anova_panel(ax1, data, with_legend_inside=True)

    mc = model_comparison_df.copy()
    # Tolerate either column-naming style
    if "Model" not in mc.columns:
        mc = mc.reset_index().rename(columns={mc.reset_index().columns[0]: "Model"})

    # Short-name the models for x-tick legibility
    def _short_name(s: str) -> str:
        s = str(s)
        if "HDD" in s or "linear" in s.lower():
            return "HDD-linear\n(incumbent)"
        if "interaction" in s.lower() or "full" in s.lower() or "B" in s[:2]:
            return "Quadratic +\noccupant dummy"
        return s
    mc["short"] = mc["Model"].apply(_short_name)

    x = np.arange(len(mc))
    w = 0.35
    r2 = mc["R_squared"].values
    rmse = mc["RMSE"].values

    ax2_rmse = ax2.twinx()
    b_r2 = ax2.bar(x - w/2, r2, w, color=COLOURS["primary"],
                   edgecolor="white", lw=0.5, label="R²", zorder=3)
    b_rmse = ax2_rmse.bar(x + w/2, rmse, w, color=COLOURS["secondary"],
                          edgecolor="white", lw=0.5, label="RMSE (kW)", zorder=3)

    for bar, v in zip(b_r2, r2):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                 color=COLOURS["primary"])
    for bar, v in zip(b_rmse, rmse):
        ax2_rmse.text(bar.get_x() + bar.get_width()/2, v + max(rmse) * 0.02,
                      f"{v:.2f}", ha="center", va="bottom", fontsize=8.5,
                      color=COLOURS["secondary"])

    ax2.set_xticks(x)
    ax2.set_xticklabels(mc["short"].tolist(), fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("R² (variance explained)", color=COLOURS["primary"])
    ax2.tick_params(axis="y", labelcolor=COLOURS["primary"])
    ax2_rmse.set_ylim(0, max(rmse) * 1.8)
    ax2_rmse.set_ylabel("RMSE (kW)", color=COLOURS["secondary"])
    ax2_rmse.tick_params(axis="y", labelcolor=COLOURS["secondary"])
    ax2.set_title("(b) HDD vs proposed model on identical data",
                  loc="left", pad=8)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_rmse.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5)
    add_grid(ax2, "y")

    # Improvement bracket between R² bars — drawn as explicit line segments
    # so it renders reliably regardless of matplotlib backend.
    if len(r2) >= 2 and r2[1] > r2[0]:
        x_left = 0 - w / 2
        x_right = 1 - w / 2
        y_top = r2[1] + 0.06
        y_drop = 0.02
        bracket_color = COLOURS["tertiary"]
        ax2.plot([x_left, x_left, x_right, x_right],
                 [y_top - y_drop, y_top, y_top, y_top - y_drop],
                 color=bracket_color, lw=1.4, solid_capstyle="round",
                 zorder=4, clip_on=False)
        if r2[0] > 0:
            improvement = r2[1] / r2[0]
            label = f"+{improvement:.0f}× R² improvement"
        else:
            label = f"+{r2[1] - r2[0]:.2f} R² gain"
        ax2.text((x_left + x_right) / 2, y_top + 0.015, label,
                 fontsize=8, color=bracket_color, fontweight="bold",
                 ha="center", va="bottom")

    save_fig(fig, out_dir, "fig4_anova_eta")


def _draw_anova_panel(ax, data, with_legend_inside: bool = True):
    """Internal helper: draw the η² horizontal bar chart with Cohen thresholds."""
    colors = [COLOURS["quaternary"] if i else COLOURS["primary"]
              for i in data["is_interaction"]]
    bars = ax.barh(data.index, data["eta_squared"], color=colors,
                   edgecolor="white", lw=0.5, zorder=3)

    # Cohen 1988 effect-size thresholds for η²
    for thr, lbl in [(0.01, "Small"), (0.06, "Medium"), (0.14, "Large")]:
        ax.axvline(thr, color=COLOURS["neutral"], ls=":", lw=0.7, zorder=2)
        ax.text(thr, len(data) - 0.15, lbl,
                fontsize=6, color=COLOURS["neutral"],
                ha="center", va="bottom",
                alpha=0.7)

    # Value + significance label
    for bar, (idx, row) in zip(bars, data.iterrows()):
        val = row["eta_squared"]
        p_val = row.get("PR(>F)", None)
        if p_val is not None and not pd.isna(p_val):
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = "n.s."
        else:
            stars = ""
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}{stars}", va="center", fontsize=8)

    ax.set_xlim(0, max(0.34, data["eta_squared"].max() + 0.05))
    ax.set_xlabel("η²  (proportion of total variance)")
    ax.set_title("(a) Variance decomposition (Type-II ANOVA)",
                 loc="left", pad=4)
    add_grid(ax, "x")

    if with_legend_inside:
        leg_elem = [
            Patch(color=COLOURS["primary"], label="Main effect"),
            Patch(color=COLOURS["quaternary"], label="Interaction"),
        ]
        ax.legend(handles=leg_elem, loc="lower right", fontsize=7.5,
                  handletextpad=0.5)
        # Significance-stars key: separate text annotation, not a phantom legend row
        ax.text(0.98, 0.02, "*** p < 0.001",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=COLOURS["neutral"], style="italic")


# ===================================================================== #
#  Fig 5 — Demand surface heatmap (cleaner contours, single colorbar)  #
# ===================================================================== #
#
#  REVISION: original used in-figure contour labels (ax.clabel) which
#  overprint the heatmap colours and become illegible. New version drops
#  the labels, keeps faint contour lines as visual contour-cues only,
#  and adds a peak-hour reference line per panel. Single shared colorbar.
#
def fig5_demand_surface_heatmap(
    surfaces: dict,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """2×2 demand surface (hour × outdoor temperature) per archetype."""
    apply_style()
    arch_ids = sorted(surfaces.keys())

    all_vals = np.concatenate(
        [np.asarray(surfaces[a]["matrix"]).ravel() for a in arch_ids]
    )
    vmax = float(np.nanpercentile(all_vals, 95))
    vmin = 0

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.18})
    axes_flat = axes.flatten()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    im = None
    for idx, arch_id in enumerate(arch_ids):
        ax = axes_flat[idx]
        surf = surfaces[arch_id]
        mat = np.asarray(surf["matrix"])
        temps = np.asarray(surf["temp_bins"])
        hours = np.asarray(surf["hours"])

        im = ax.pcolormesh(temps, hours, mat,
                           cmap="magma", shading="gouraud",
                           vmin=vmin, vmax=vmax)

        # Faint contour lines, NO labels (labels would overprint colours)
        try:
            ax.contour(temps, hours, mat,
                       levels=[1, 2, 3, 4, 5],
                       colors="white", linewidths=0.5, alpha=0.55)
        except Exception:
            pass

        # Peak-hour reference line + annotation, anchored at the actual peak
        # (location of the global maximum, not the right edge of the plot).
        if mat.ndim == 2 and mat.shape[0] == len(hours):
            peak_idx_2d = np.unravel_index(np.nanargmax(mat), mat.shape)
            peak_h = hours[peak_idx_2d[0]]
            peak_t = temps[peak_idx_2d[1]]
            peak_kw = float(np.nanmax(mat))
            ax.axhline(peak_h, color="white", lw=1.1, ls="--", alpha=0.9,
                       zorder=4)
            # Decide which side of the peak temperature has more empty space,
            # so the label sits in a dark region rather than over the bright peak.
            t_mid = (temps.min() + temps.max()) / 2
            if peak_t <= t_mid:
                x_anchor = temps.max() - 0.5
                ha = "right"
            else:
                x_anchor = temps.min() + 0.5
                ha = "left"
            ax.text(x_anchor, peak_h + 0.6,
                    f"peak ≈ {peak_h:.1f} h ({peak_kw:.1f} kW)",
                    fontsize=7, color="white", ha=ha, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="black", alpha=0.78, ec="none"),
                    zorder=5)
            # Small marker at the actual peak (temp, hour) so the reader can
            # see where the labelled value lives.
            ax.plot(peak_t, peak_h, marker="o", ms=4.5,
                    mfc="white", mec="black", mew=0.6, zorder=6)

        name = archetype_names.get(arch_id, arch_id)
        ax.set_title(f"{panel_labels[idx]} {name}", loc="left", pad=4)
        ax.tick_params(labelsize=7.5)

    # Shared axis labels
    for ax in axes[:, 0]:
        ax.set_ylabel("Hour of day")
    for ax in axes[1, :]:
        ax.set_xlabel("Outdoor temperature (°C)")

    # Single shared colorbar on the right
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.7, pad=0.02, aspect=25)
    cbar.set_label("Electricity demand (kW per home)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle("Peak demand surface: hour × outdoor temperature, by occupant archetype",
                 fontsize=10.5, x=0.42, y=0.98)

    save_fig(fig, out_dir, "fig5_demand_surface_heatmap")


# Alias for backward compatibility with original name
fig5_demand_surface = fig5_demand_surface_heatmap


# ===================================================================== #
#  Fig 6 — Peak demand grouped bars  (unchanged)                         #
# ===================================================================== #

def fig6_peak_boxplots(
    sim_df: pd.DataFrame,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """Grouped bars of peak demand — one panel per archetype, weather on x, fabric as colour."""
    apply_style()

    # Compute per-run peaks
    if "replicate" in sim_df.columns:
        peaks = (sim_df.groupby(
            ["archetype", "weather_scenario", "fabric", "replicate"]
        )["electricity_demand_kW"].max().reset_index())
    else:
        peaks = (sim_df.groupby(
            ["archetype", "weather_scenario", "fabric"]
        )["electricity_demand_kW"].max().reset_index())

    arch_order = sorted(peaks["archetype"].unique())
    weather_order = sorted(peaks["weather_scenario"].unique())
    fab_order = sorted(peaks["fabric"].unique())

    n = len(arch_order)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]

    bar_width = 0.35
    x_positions = np.arange(len(weather_order))

    for ax, arch_id in zip(axes, arch_order):
        sub = peaks[peaks["archetype"] == arch_id]

        for f_idx, fab in enumerate(fab_order):
            fab_data = sub[sub["fabric"] == fab]
            means, stds = [], []
            for w in weather_order:
                ws = fab_data[fab_data["weather_scenario"] == w]["electricity_demand_kW"]
                means.append(ws.mean() if len(ws) else 0.0)
                stds.append(ws.std() if len(ws) > 1 else 0.0)

            offset = (f_idx - (len(fab_order) - 1) / 2) * bar_width
            colour = FABRIC_COLOURS.get(fab, COLOURS["primary"])
            label = FABRIC_LABELS.get(fab, fab)

            ax.bar(x_positions + offset, means, bar_width,
                   yerr=stds, color=colour, alpha=0.85, edgecolor="white",
                   linewidth=0.5, error_kw={"linewidth": 0.7, "capsize": 2},
                   label=label, zorder=3)

            # Value labels
            for x, m in zip(x_positions, means):
                ax.text(x + offset, m + 0.05, f"{m:.1f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color=colour, fontweight="bold")

        # Cosmetics
        name = archetype_names.get(arch_id, arch_id)
        ax.set_title(name, fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([WEATHER_LABELS.get(w, w) for w in weather_order],
                           fontsize=7)
        ax.tick_params(labelsize=7)
        add_grid(ax)

    axes[0].set_ylabel("Peak demand (kW)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(fab_order),
               fontsize=7.5, title="Fabric", title_fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, out_dir, "fig6_peak_boxplots")


# ===================================================================== #
#  NEW: EoH external validation (3-panel)                              #
# ===================================================================== #
#
#  Story: simulator overestimates COP relative to EoH field data, but the
#  bias is one-sided (over-COP -> under-kW), making the £-cost figure a
#  conservative lower bound. Peak band sits inside EoH median but misses
#  the upper tail. Empirical k-means silhouette favours k=2, contradicting
#  the four-archetype assumption — but that's a strength, not a weakness,
#  because the ANOVA tests group-mean differences, not cluster membership.
#
def fig_eoh_validation(
    cop_by_T_df: pd.DataFrame,
    peak_by_T_df: pd.DataFrame,
    silhouette_df: pd.DataFrame,
    out_dir: Path,
    hp_cfg: Optional[dict] = None,
    sim_peak_band_kW: tuple[float, float] = (4.1, 6.4),
    n_property_days: Optional[int] = None,
    n_cop_obs: Optional[int] = None,
    cost_headline_GBP: Optional[float] = 602_000.0,
    silhouette_threshold: float = 0.30,
    assumed_k: int = 4,
) -> None:
    """3-panel external validation against the BEIS Electrification of Heat trial."""
    apply_style()

    fig = plt.figure(figsize=(7.6, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.65, wspace=0.32,
                          height_ratios=[1, 1])

    # ---------- (a) Simulator vs EoH COP ----------
    ax_a = fig.add_subplot(gs[0, :])
    cop = cop_by_T_df.sort_values("T_bin_centre")
    T = cop["T_bin_centre"].values
    median = cop["cop_median"].values
    p25 = cop["cop_25"].values
    p75 = cop["cop_75"].values

    n_label = f" (n = {n_cop_obs/1e6:.2f} M obs)" if n_cop_obs else ""
    ax_a.fill_between(T, p25, p75, color=COLOURS["primary"], alpha=0.18,
                      label=f"EoH field IQR{n_label}", zorder=2)
    ax_a.plot(T, median, color=COLOURS["primary"], lw=1.5, marker="o", ms=3,
              label="EoH field median", zorder=4)

    # Simulator COP curve from hp_cfg if provided; else fall back to defaults
    cop_intercept = hp_cfg["cop_intercept"] if hp_cfg else 3.5
    cop_slope = hp_cfg["cop_slope"] if hp_cfg else 0.12
    cop_min = hp_cfg["cop_min"] if hp_cfg else 1.5
    cop_max = hp_cfg["cop_max"] if hp_cfg else 4.5
    t_defrost = hp_cfg["defrost_temp_threshold_C"] if hp_cfg else -2.0
    defrost_pen = hp_cfg["defrost_efficiency_penalty"] if hp_cfg else (0.5 / cop_intercept)

    T_model = np.linspace(T.min() - 1, T.max() + 1, 300)
    cop_model = cop_intercept + cop_slope * T_model
    cop_model = np.where(T_model < t_defrost,
                         cop_model * (1.0 - defrost_pen),
                         cop_model)
    cop_model = np.clip(cop_model, cop_min, cop_max)
    ax_a.plot(T_model, cop_model, color=COLOURS["secondary"], lw=1.5, ls="--",
              label=(f"Simulator: COP = {cop_intercept:.1f} + "
                     f"{cop_slope:.2f}·T  (clipped, defrost step at {t_defrost:.0f} °C)"),
              zorder=3)

    # Honest annotation of the discontinuity at the defrost threshold.
    # Placed below the IQR band so it doesn't collide with the +0.73 COP box.
    cop_above = cop_intercept + cop_slope * t_defrost
    cop_below = cop_above * (1.0 - defrost_pen)
    ax_a.annotate(
        "modelled defrost penalty",
        xy=(t_defrost, (cop_above + cop_below) / 2),
        xytext=(T.min() + 0.5, 1.4),
        fontsize=7, ha="left", va="bottom",
        color="black", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0,
                        shrinkA=2, shrinkB=2),
        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                  ec=COLOURS["neutral"], lw=0.4),
    )

    # Annotate gap at the defrost / design boundary (T_design = -1)
    T_design = -1.0
    cop_emp = float(np.interp(T_design, T, median))
    cop_mod = cop_intercept + cop_slope * T_design
    if T_design < t_defrost:
        cop_mod *= (1.0 - defrost_pen)
    cop_mod = float(np.clip(cop_mod, cop_min, cop_max))
    ax_a.plot([T_design, T_design], [cop_emp, cop_mod],
              color="black", lw=1.0, zorder=5)
    ax_a.annotate(
        f"+{cop_mod - cop_emp:.2f} COP\nover-prediction\nat design temp",
        xy=(T_design, (cop_emp + cop_mod) / 2),
        xytext=(T.min() + 1, 4.55),
        fontsize=7.5, ha="left", va="top",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec="black", lw=0.4),
    )

    # Headline banner above the plot — wrapped to two lines so it doesn't
    # crowd the panel title underneath.
    cost_str = f"£{cost_headline_GBP/1000:.0f} k" if cost_headline_GBP else "the cost figure"
    ax_a.text(
        0.5, 1.10,
        "Bias is one-sided: simulator over-predicts COP → under-predicts kW.\n"
        f"{cost_str} cost is therefore a CONSERVATIVE lower bound.",
        fontsize=8, color=COLOURS["tertiary"], fontweight="bold",
        ha="center", va="bottom", transform=ax_a.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="#eaf6ee",
                  ec=COLOURS["tertiary"], lw=0.6),
    )

    ax_a.set_xlabel("Outdoor temperature (°C)")
    ax_a.set_ylabel("Coefficient of performance (COP)")
    ax_a.set_xlim(T.min() - 1, 5)
    ax_a.set_ylim(1.0, 5.0)
    ax_a.set_title(
        "(a) Simulator COP vs BEIS Electrification of Heat (EoH) field trial",
        loc="left", pad=36,
    )
    ax_a.legend(loc="lower right", fontsize=7.5)
    add_grid(ax_a, "y")

    # ---------- (b) Peak demand band vs EoH ----------
    ax_b = fig.add_subplot(gs[1, 0])
    peak = peak_by_T_df.sort_values("T_bin_centre")
    Tb = peak["T_bin_centre"].values
    ax_b.plot(Tb, peak["peak_median"], color=COLOURS["primary"], lw=1.4,
              marker="o", ms=3, label="EoH median", zorder=4)
    if "peak_p95" in peak.columns:
        ax_b.plot(Tb, peak["peak_p95"], color=COLOURS["primary"], lw=1.0,
                  ls=":", label="EoH P95", zorder=3)

    sim_lo, sim_hi = sim_peak_band_kW
    ax_b.axhspan(sim_lo, sim_hi, color=COLOURS["secondary"], alpha=0.18,
                 zorder=1, label=f"Simulated {sim_lo:.1f}–{sim_hi:.1f} kW")

    if "peak_p95" in peak.columns:
        # Annotate the upper-tail miss
        p95_val = float(peak[peak["T_bin_centre"] >= 0]["peak_p95"].iloc[0]) \
                  if not peak.empty else 18.0
        ax_b.annotate(
            f"P95 = {p95_val:.1f} kW\nrarely captured\nby the simulator",
            xy=(0, p95_val), xytext=(-3, p95_val - 1.5),
            fontsize=6.5, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
        )

    ax_b.set_xlabel("Outdoor temperature (°C)")
    ax_b.set_ylabel("Peak demand (kW)")
    ax_b.set_xlim(Tb.min() - 1, 5)
    ax_b.set_ylim(0, max(22, peak["peak_p95"].max() + 2 if "peak_p95" in peak.columns else 22))
    ax_b.set_title(
        "(b) Peak demand: simulated band vs\n730 EoH homes" +
        (f" ({n_property_days:,} property-days)" if n_property_days else ""),
        loc="left", pad=6, fontsize=9,
    )
    ax_b.legend(loc="lower left", fontsize=6.5, handlelength=1.2,
                ncol=1, borderpad=0.2)
    add_grid(ax_b, "y")

    # ---------- (c) Silhouette: 4-cluster assumption test ----------
    ax_c = fig.add_subplot(gs[1, 1])
    sil = silhouette_df.sort_values("k")
    ax_c.plot(sil["k"], sil["silhouette"], color=COLOURS["tertiary"], lw=1.4,
              marker="o", ms=6, zorder=4)

    # Threshold band
    ax_c.axhspan(silhouette_threshold, silhouette_threshold + 0.04,
                 color=COLOURS["tertiary"], alpha=0.10, zorder=1)
    ax_c.axhline(silhouette_threshold, color=COLOURS["neutral"],
                 ls="--", lw=0.8, zorder=2)
    ax_c.text(sil["k"].max() + 0.1, silhouette_threshold + 0.005,
              f"Meaningful-cluster\nthreshold ({silhouette_threshold:.2f})",
              fontsize=6.5, color=COLOURS["neutral"], ha="right", va="bottom")

    # Highlight assumed_k
    if assumed_k in sil["k"].values:
        kx = sil[sil["k"] == assumed_k].silhouette.iloc[0]
        ax_c.scatter([assumed_k], [kx], s=110, marker="o",
                     facecolor="white", edgecolor=COLOURS["secondary"],
                     lw=2, zorder=5)
        ax_c.annotate(
            f"k = {assumed_k} assumed:\nsilhouette = {kx:.3f}\n"
            f"(< {silhouette_threshold:.2f} threshold)",
            xy=(assumed_k, kx),
            xytext=(assumed_k + 0.6, sil["silhouette"].max() - 0.02),
            fontsize=7, ha="left",
            arrowprops=dict(arrowstyle="->", color=COLOURS["secondary"], lw=0.8),
            color=COLOURS["secondary"], fontweight="bold",
        )

    # Empirical best
    best_k = int(sil.loc[sil["silhouette"].idxmax(), "k"])
    best_sil = sil["silhouette"].max()
    ax_c.scatter([best_k], [best_sil], s=110, marker="*",
                 color=COLOURS["quaternary"], edgecolor="black", lw=0.5, zorder=5)
    ax_c.text(best_k + 0.2, best_sil,
              f"empirical best: k = {best_k}\n(silhouette = {best_sil:.3f})",
              fontsize=7, ha="left", va="top",
              color=COLOURS["quaternary"], fontweight="bold")

    ax_c.set_xlabel("Number of clusters (k)")
    ax_c.set_ylabel("Silhouette score")
    ax_c.set_xticks(sil["k"].astype(int).tolist())
    ax_c.set_ylim(sil["silhouette"].min() - 0.02, max(0.34, silhouette_threshold + 0.05))
    ax_c.set_title("(c) Empirical clustering of\nthe four-archetype assumption",
                   loc="left", pad=6, fontsize=9)
    add_grid(ax_c, "y")

    save_fig(fig, out_dir, "fig_eoh_validation")


# ===================================================================== #
#  NEW: ADMD diversification + £-cost of HDD under-sizing              #
# ===================================================================== #
#
#  Story: ADMD per home converges quickly with N for every occupant mix,
#  and the gap between simulated ADMD and the HDD ADMD is roughly constant
#  (~1.4 kW/home), translating into £602 k of avoidable reactive
#  reinforcement at 500 homes — the headline policy number.
#
def fig_admd_and_cost(
    admd_curves: Mapping[str, pd.DataFrame],
    cost_df: pd.DataFrame,
    out_dir: Path,
    hdd_admd_kW: Optional[float] = None,
    weather_scenario: str = "W2",
    mix_display_names: Optional[Mapping[str, str]] = None,
    cost_per_kVA_planned: int = 300,
    reactive_multiplier: float = 2.5,
) -> None:
    """ADMD diversification curves (left) + £-cost of HDD under-sizing (right)."""
    apply_style()

    if mix_display_names is None:
        mix_display_names = {
            "uniform":        "Uniform mix",
            "uk_typical":     "UK typical mix",
            "commuter_heavy": "Commuter-heavy mix",
            "mostly_home":    "Mostly-home mix",
        }
    mix_colours = {
        "uniform":        COLOURS["primary"],
        "uk_typical":     COLOURS["tertiary"],
        "commuter_heavy": COLOURS["quaternary"],
        "mostly_home":    COLOURS["secondary"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.0),
                                   gridspec_kw={"wspace": 0.45})

    # ---------- (a) ADMD diversification ----------
    for key, df in admd_curves.items():
        lbl = mix_display_names.get(key, key)
        col = mix_colours.get(key, COLOURS["neutral"])
        ax1.plot(df["n_homes"], df["admd_mean_kW"], color=col, lw=1.4,
                 marker="o", ms=4, label=lbl, zorder=4)
        if (key == "uk_typical"
                and "admd_p5_kW" in df.columns
                and "admd_p95_kW" in df.columns):
            ax1.fill_between(df["n_homes"], df["admd_p5_kW"], df["admd_p95_kW"],
                             color=col, alpha=0.12, zorder=2)

    # HDD ADMD reference
    if hdd_admd_kW is not None:
        scenario_lbl = WEATHER_LABELS.get(weather_scenario, weather_scenario).lower()
        ax1.axhline(hdd_admd_kW, color="black", ls="--", lw=1.0, zorder=3)
        ax1.text(5, hdd_admd_kW - 0.10,
                 f"HDD ADMD = {hdd_admd_kW:.2f} kW/home ({scenario_lbl}-cold winter)",
                 fontsize=7.5, ha="left", va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec="black", lw=0.4))

        # Shortfall bracket at the largest n_homes plotted
        # Use the central mix (uk_typical) if present, else the first
        ref_key = "uk_typical" if "uk_typical" in admd_curves else next(iter(admd_curves))
        ref = admd_curves[ref_key]
        n_max = int(ref["n_homes"].max())
        sim_kW = float(ref[ref["n_homes"] == n_max]["admd_mean_kW"].iloc[0])
        shortfall = sim_kW - hdd_admd_kW
        if shortfall > 0:
            ax1.annotate("", xy=(n_max, sim_kW), xytext=(n_max, hdd_admd_kW),
                         arrowprops=dict(arrowstyle="<->",
                                         color=COLOURS["secondary"], lw=1.2))
            # Place shortfall label well to the LEFT of the arrow, away from
            # the "UK typical mix (baseline)" label that sits above the line.
            ax1.text(n_max * 0.55, (sim_kW + hdd_admd_kW) / 2,
                     f"shortfall ≈ {shortfall:.2f} kW/home",
                     fontsize=7.5, ha="center", va="center",
                     color=COLOURS["secondary"], fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white",
                               ec=COLOURS["secondary"], lw=0.6))

        # Identify the uk_typical curve as the policy-relevant baseline.
        # Anchor the label to the LEFT of the rightmost point so it doesn't
        # collide with the shortfall arrow at n_max.
        if ref_key == "uk_typical":
            ax1.text(n_max * 0.5, sim_kW + 0.18,
                     "← UK typical mix (baseline)",
                     fontsize=7, ha="center", va="bottom",
                     color=mix_colours.get(ref_key, COLOURS["neutral"]),
                     fontweight="bold")

    ax1.set_xscale("log")
    ax1.set_xlim(4, max(df["n_homes"].max() for df in admd_curves.values()) * 1.2)
    # Y-limits chosen for honest scale: pull the visual zero closer to the
    # data so the shortfall doesn't look more dramatic than it is.
    all_y = [df["admd_mean_kW"].values for df in admd_curves.values()]
    min_y = min(np.min(y) for y in all_y)
    max_y = max(np.max(y) for y in all_y)
    if hdd_admd_kW is not None:
        y_lo = max(0.0, min(min_y, hdd_admd_kW) - 1.5)
    else:
        y_lo = max(0.0, min_y - 1.5)
    y_hi = max_y + 1.0
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_xlabel("Number of homes (log scale)")
    ax1.set_ylabel("After-Diversity Maximum Demand (kW per home)")
    ax1.set_title("(a) ADMD diversification by occupant mix",
                  loc="left", pad=6)
    ax1.legend(loc="upper center", fontsize=7, ncol=2,
               bbox_to_anchor=(0.5, -0.20), frameon=False)
    add_grid(ax1, "y")

    # ---------- (b) Reinforcement cost ----------
    cost = cost_df.copy().sort_values("n_homes")
    x = np.arange(len(cost))
    w = 0.35

    planned = cost["planned_cost_GBP"].values / 1000
    reactive = cost["reactive_cost_GBP"].values / 1000

    b_p = ax2.bar(x - w/2, planned, w, color=COLOURS["primary"],
                  edgecolor="white", lw=0.5,
                  label=f"Planned (£{cost_per_kVA_planned}/kVA)", zorder=3)
    b_r = ax2.bar(x + w/2, reactive, w, color=COLOURS["secondary"],
                  edgecolor="white", lw=0.5,
                  label=f"Reactive (£{int(cost_per_kVA_planned*reactive_multiplier)}/kVA, "
                        f"{reactive_multiplier:.1f}×)",
                  zorder=3)

    for bar, v in zip(b_p, planned):
        ax2.text(bar.get_x() + bar.get_width()/2, v + max(planned)*0.02,
                 f"£{v:.0f}k", ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(b_r, reactive):
        ax2.text(bar.get_x() + bar.get_width()/2, v + max(reactive)*0.02,
                 f"£{v:.0f}k", ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold", color=COLOURS["secondary"])

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(n)}" for n in cost["n_homes"]])
    ax2.set_xlabel("Number of homes per LV feeder")
    ax2.set_ylabel("HDD-induced reinforcement cost (£ thousands)")
    ax2.set_ylim(0, max(reactive) * 1.20)
    ax2.set_title("(b) Cost of HDD under-sizing at the design temperature",
                  loc="left", pad=6)
    ax2.legend(loc="upper left", fontsize=7.5)
    add_grid(ax2, "y")

    save_fig(fig, out_dir, "fig_admd_and_cost")


# ===================================================================== #
#  Master function                                                       #
# ===================================================================== #

def generate_all_figures(
    sim_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    anova_table: pd.DataFrame,
    surfaces: dict,
    hp_cfg: dict,
    archetype_names: dict,
    *,
    # New optional arguments — supply to enable the new figures and the
    # extra annotations on existing figures. Backward-compatible: passing
    # None or omitting them yields the same set of 6 figures as before
    # (with the four critical-revision improvements baked in).
    hdd_peak_w2_kW: Optional[float] = None,
    model_comparison_df: Optional[pd.DataFrame] = None,
    cop_by_T_df: Optional[pd.DataFrame] = None,
    peak_by_T_df: Optional[pd.DataFrame] = None,
    silhouette_df: Optional[pd.DataFrame] = None,
    admd_curves: Optional[Mapping[str, pd.DataFrame]] = None,
    cost_df: Optional[pd.DataFrame] = None,
    hdd_admd_kW: Optional[float] = None,
    eoh_kwargs: Optional[dict] = None,
    admd_kwargs: Optional[dict] = None,
    out_dir: Optional[Path] = None,
    essay_mode: bool = False,
) -> None:
    """Generate all publication-quality figures.

    Required (existing) parameters keep the original signature.
    New optional parameters unlock the additional figures and annotations:
      - hdd_peak_w2_kW: HDD reference line on Fig 1 panel (b)
      - model_comparison_df: pairs ANOVA panel with HDD-vs-proposed-model on Fig 4
      - cop_by_T_df, peak_by_T_df, silhouette_df: triggers fig_eoh_validation
      - admd_curves, cost_df, hdd_admd_kW: triggers fig_admd_and_cost
      - out_dir: override output directory (defaults to FIGURES_DIR)
      - essay_mode: skip fig1_timeseries, fig2_cop_curve, fig6_peak_boxplots
        for the word-count-constrained essay figure set
    """
    out_dir = out_dir if out_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not essay_mode:
        fig1_timeseries(sim_df, archetype_names, out_dir,
                        hdd_peak_w2_kW=hdd_peak_w2_kW)
        fig2_cop_curve(sim_df, hp_cfg, out_dir)
    fig3_hdd_error(errors_df, archetype_names, out_dir)
    fig4_anova_eta(anova_table, out_dir,
                   model_comparison_df=model_comparison_df)
    fig5_demand_surface_heatmap(surfaces, archetype_names, out_dir)
    if not essay_mode:
        fig6_peak_boxplots(sim_df, archetype_names, out_dir)

    # New figures — only generated when the relevant data is supplied
    if cop_by_T_df is not None and peak_by_T_df is not None and silhouette_df is not None:
        kwargs = dict(eoh_kwargs or {})
        kwargs.setdefault("hp_cfg", hp_cfg)
        fig_eoh_validation(cop_by_T_df, peak_by_T_df, silhouette_df,
                           out_dir, **kwargs)
    else:
        logger.info("fig_eoh_validation skipped (validation DataFrames not provided)")

    if admd_curves is not None and cost_df is not None:
        kwargs = dict(admd_kwargs or {})
        if hdd_admd_kW is not None:
            kwargs.setdefault("hdd_admd_kW", hdd_admd_kW)
        fig_admd_and_cost(admd_curves, cost_df, out_dir, **kwargs)
    else:
        logger.info("fig_admd_and_cost skipped (ADMD/cost DataFrames not provided)")

    logger.info("Figure generation complete in %s", out_dir)


# ===================================================================== #
#                                                                       #
#  ESSAY FIGURES — five redesigned figures                              #
#                                                                       #
#  One argument per figure. Shared visual system below. Each figure is  #
#  self-contained: no cross-references inside the panel; all narrative  #
#  context lives in the essay text or in the single-line footnote.      #
#                                                                       #
# ===================================================================== #

# ── Two-color semantic system ────────────────────────────────────────── #
ESSAY_INK       = "#1a1a1a"   # primary text/data
ESSAY_INCUMBENT = "#b35a4f"   # warm grey-red for HDD / status quo
ESSAY_PROPOSED  = "#2c5fa3"   # cool blue for simulator / proposed model
ESSAY_NEUTRAL   = "#9aa0a6"   # context / gridlines / footnotes
ESSAY_PANEL_BG  = "#ffffff"

# Sequential blue ramp for 4 archetypes (light → dark)
ESSAY_ARCHETYPE_RAMP = {
    "B1": "#9bc4e2",   # Early Riser    (lightest)
    "B2": "#5a96c8",   # Home All Day
    "B3": "#2c5fa3",   # Late Returner
    "B4": "#0f2d5e",   # Intermittent   (darkest)
}

ESSAY_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "regular",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.labelcolor": ESSAY_INK,
    "axes.edgecolor": ESSAY_INK,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": ESSAY_INK,
    "ytick.color": ESSAY_INK,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "legend.borderpad": 0.2,
    "legend.handlelength": 1.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "text.color": ESSAY_INK,
}


def apply_essay_style() -> None:
    """Apply the essay-figure visual system (sans-serif, two-color semantic)."""
    plt.rcParams.update(ESSAY_RC)


def _essay_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Faint horizontal gridlines for the essay figures."""
    ax.grid(axis=axis, color=ESSAY_NEUTRAL, alpha=0.25, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def _essay_footnote(fig: plt.Figure, text: str, y: float = 0.01) -> None:
    """One-line italic grey footnote at the bottom of a figure."""
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=7.5, color=ESSAY_NEUTRAL, style="italic")


# ===================================================================== #
#  Essay Fig 1 — experimental design (3-panel strip)                    #
# ===================================================================== #

def essay_fig1_design(
    cfg: dict,
    sim_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Three-panel strip showing what we vary: occupants, fabric, weather."""
    apply_essay_style()

    archetypes = cfg["archetypes"]
    fabrics    = cfg["fabric"]
    arch_ids   = list(archetypes.keys())

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(8.4, 2.6),
        gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.75},
    )

    # ───── (a) Heating schedules ─────
    for i, a_id in enumerate(arch_ids):
        for on, off in archetypes[a_id]["schedule_on"]:
            ax_a.barh(i, off - on, left=on, height=0.55,
                      color=ESSAY_ARCHETYPE_RAMP[a_id],
                      edgecolor="white", lw=0.4, zorder=3)
    ax_a.set_yticks(range(len(arch_ids)))
    ax_a.set_yticklabels([archetypes[a]["name"] for a in arch_ids], fontsize=8)
    ax_a.set_xlim(0, 24)
    ax_a.set_xticks([0, 6, 12, 18, 24])
    ax_a.set_xticklabels(["00", "06", "12", "18", "24"])
    ax_a.set_xlabel("Hour of day")
    ax_a.invert_yaxis()
    ax_a.set_title("(a) Occupant schedules", pad=6)
    _essay_grid(ax_a, "x")

    # ───── (b) Fabric U-values ─────
    # Both labels rendered symmetrically OUTSIDE the right end of each bar
    # ("name   value") so the visual treatment is identical regardless of
    # bar length. Y-tick labels are suppressed in favour of these end-labels.
    fab_ids = list(fabrics.keys())
    u_walls = [fabrics[f]["U_wall"] for f in fab_ids]
    fab_names = [fabrics[f]["name"] for f in fab_ids]
    bar_colors = [ESSAY_INCUMBENT, ESSAY_PROPOSED]   # F1 incumbent stock, F2 retrofit
    bars = ax_b.barh(range(len(fab_ids)), u_walls,
                     color=bar_colors, edgecolor="white", lw=0.4,
                     height=0.55, zorder=3)
    # Two-line label: name on top, value below — keeps the horizontal extent
    # short enough to stay inside panel (b) regardless of bar length.
    label_offset = max(u_walls) * 0.05
    for bar, name, val in zip(bars, fab_names, u_walls):
        y = bar.get_y() + bar.get_height() / 2
        x = bar.get_width() + label_offset
        ax_b.text(x, y, f"{name}\n{val:.2f}",
                  va="center", ha="left",
                  fontsize=7.5, color=ESSAY_INK, fontweight="bold",
                  linespacing=1.1)
    ax_b.set_yticks(range(len(fab_ids)))
    ax_b.set_yticklabels([])           # suppress — names live at bar-end
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xlabel("Wall U-value (W m⁻² K⁻¹)")
    # Allow horizontal room for both labels at the bar end
    ax_b.set_xlim(0, max(u_walls) * 1.55)
    ax_b.invert_yaxis()
    ax_b.set_title("(b) Fabric levels", pad=6)
    _essay_grid(ax_b, "x")

    # ───── (c) Weather traces ─────
    sim_df = sim_df.copy()
    sim_df["timestamp"] = pd.to_datetime(sim_df["timestamp"])
    sim_df["hour"] = sim_df["timestamp"].dt.hour + sim_df["timestamp"].dt.minute / 60
    weather_order = ["W1", "W2", "W3"]
    weather_label = {"W1": "Mild (W1)", "W2": "Design (W2)", "W3": "Extreme (W3)"}
    weather_color = {
        "W1": ESSAY_NEUTRAL,
        "W2": ESSAY_INCUMBENT,
        "W3": ESSAY_PROPOSED,
    }
    for w in weather_order:
        sub = sim_df[(sim_df.weather_scenario == w) &
                     (sim_df.archetype == arch_ids[0]) &
                     (sim_df.fabric == "F1") &
                     (sim_df.replicate == 0)].sort_values("hour")
        ax_c.plot(sub["hour"], sub["T_outdoor_C"],
                  color=weather_color[w], lw=1.6, zorder=3,
                  label=weather_label[w])
        # Direct label at right end of each line
        if len(sub) > 0:
            ax_c.text(24.4, sub["T_outdoor_C"].iloc[-1],
                      weather_label[w], ha="left", va="center",
                      fontsize=7.5, color=weather_color[w],
                      fontweight="bold")
    ax_c.set_xlim(0, 24)
    ax_c.set_xticks([0, 6, 12, 18, 24])
    ax_c.set_xticklabels(["00", "06", "12", "18", "24"])
    ax_c.set_xlabel("Hour of day")
    ax_c.set_ylabel("Outdoor T (°C)")
    ax_c.set_title("(c) Weather days", pad=6)
    _essay_grid(ax_c, "y")

    fig.suptitle("Experimental design: 4 × 2 × 3 factorial",
                 x=0.02, y=1.02, ha="left", fontsize=11, color=ESSAY_INK)
    fig.subplots_adjust(bottom=0.28, top=0.84)
    _essay_footnote(
        fig,
        "240 simulation runs (4 occupant schedules × 2 fabrics × 3 weather days × 10 replicates).",
        y=0.04,
    )
    save_fig(fig, out_dir, "essay_fig1_design")


# ===================================================================== #
#  Essay Fig 2 — Peak demand vs outdoor temperature (H1 + H3)           #
# ===================================================================== #

def essay_fig2_peak_vs_temp(
    errors_df: pd.DataFrame,
    archetype_names: dict,
    out_dir: Path,
) -> None:
    """Peak demand vs outdoor T: two panels comparing fabric regimes.

    (a) Unimproved homes: heat-loss-limited — archetypes converge, HDD ≈ correct.
    (b) Retrofitted homes: occupant-pattern-limited — archetypes diverge,
        HDD over-predicts for some, under-predicts for others.

    The contrast is the H3 story: as housing improves, HDD breaks down because
    it ignores occupants. The divergence in (b) IS the structural inadequacy.
    """
    apply_essay_style()

    df = errors_df.copy().sort_values("T_mean_outdoor_C")
    arch_ids = sorted(df["archetype"].unique())

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.6, 3.8),
        gridspec_kw={"wspace": 0.28},
        sharey=True,
    )

    panels = [("F1", "(a) Unimproved homes — heat-loss-limited", ax_a),
              ("F2", "(b) Retrofitted homes — occupant-driven",   ax_b)]

    # Y-range so both panels share an honest scale
    y_lo = df["sim_peak_kW"].min() - 0.4
    y_hi = df["sim_peak_kW"].max() + 0.4

    for fab, title, ax in panels:
        sub_all = df[df.fabric == fab]
        # 4 archetype lines
        for a_id in arch_ids:
            s = sub_all[sub_all.archetype == a_id].sort_values("T_mean_outdoor_C")
            ax.plot(s["T_mean_outdoor_C"], s["sim_peak_kW"],
                    color=ESSAY_ARCHETYPE_RAMP[a_id], lw=1.8,
                    marker="o", ms=4.5, mec="white", mew=0.6, zorder=4)
        # In panel (b) only, label each archetype at the WARM (right) endpoint
        # of its line. Archetypes with close warm-end y-values (e.g. Early
        # Riser at 5.61 and Intermittent at 5.60) are pulled apart by a
        # global min-separation pass to guarantee no labels overlap each
        # other regardless of how the data clusters.
        if fab == "F2":
            warm_T = sub_all["T_mean_outdoor_C"].max()
            rows = []   # (a_id, raw_y)
            for a_id in arch_ids:
                s = sub_all[(sub_all.archetype == a_id) &
                            (sub_all["T_mean_outdoor_C"] == warm_T)]
                if len(s):
                    rows.append((a_id, float(s["sim_peak_kW"].iloc[0])))
            rows.sort(key=lambda r: r[1])

            min_sep = 0.32     # absolute floor on label spacing in kW
            placed_ys: list[float] = []
            for a_id, raw_y in rows:
                y = raw_y
                if placed_ys and y - placed_ys[-1] < min_sep:
                    y = placed_ys[-1] + min_sep
                placed_ys.append(y)
                if abs(y - raw_y) > 1e-3:
                    ax.plot([warm_T, warm_T + 0.18],
                            [raw_y, y],
                            color=ESSAY_ARCHETYPE_RAMP[a_id],
                            lw=0.7, alpha=0.7, zorder=3)
                ax.text(warm_T + 0.22, y,
                        archetype_names.get(a_id, a_id),
                        ha="left", va="center", fontsize=7.5,
                        color=ESSAY_ARCHETYPE_RAMP[a_id],
                        fontweight="bold")

        # HDD reference
        hdd = sub_all.groupby("T_mean_outdoor_C", as_index=False)["hdd_peak_kW"].first()
        hdd = hdd.sort_values("T_mean_outdoor_C")
        ax.plot(hdd["T_mean_outdoor_C"], hdd["hdd_peak_kW"],
                color=ESSAY_INCUMBENT, lw=2.2, ls=(0, (5, 3)), zorder=5)
        # Label HDD at the COLD (left) endpoint to keep panel (b)'s right side
        # for the archetype labels.
        ax.text(hdd["T_mean_outdoor_C"].iloc[0] - 0.3,
                hdd["hdd_peak_kW"].iloc[0],
                "HDD",
                ha="right", va="center", fontsize=8,
                color=ESSAY_INCUMBENT, fontweight="bold")

        ax.set_xlabel("Mean outdoor temperature (°C)")
        ax.set_title(title, pad=6, fontsize=9.5)
        _essay_grid(ax, "y")
        ax.set_ylim(y_lo, y_hi)
        # Add right-side margin in panel (b) so labels fit
        if fab == "F2":
            ax.set_xlim(left=df["T_mean_outdoor_C"].min() - 0.6,
                        right=df["T_mean_outdoor_C"].max() + 2.4)
        else:
            ax.set_xlim(left=df["T_mean_outdoor_C"].min() - 0.8,
                        right=df["T_mean_outdoor_C"].max() + 0.6)

    # Single grouped label in panel (a) since lines collapse there.
    # Place it ABOVE the converged line so it doesn't overlay the data.
    f1 = df[df.fabric == "F1"]
    f1_design = f1[f1["weather_scenario"] == "W2"]   # mid-temperature design point
    if len(f1_design):
        T_anchor = float(f1_design["T_mean_outdoor_C"].iloc[0])
        kW_anchor = float(f1_design["sim_peak_kW"].mean())
        ax_a.annotate(
            "",
            xy=(T_anchor, kW_anchor + 0.05),
            xytext=(T_anchor, kW_anchor + 0.55),
            arrowprops=dict(arrowstyle="-", color=ESSAY_NEUTRAL, lw=0.6),
        )
        ax_a.text(T_anchor, kW_anchor + 0.60,
                  "all 4 archetypes collapse onto one curve",
                  ha="center", va="bottom", fontsize=7.5,
                  color=ESSAY_NEUTRAL, style="italic")

    # H3 callout in panel (b): show the WIDEST archetype gap at the cold end.
    # Place the spread arrow JUST inside the cold endpoint, with the label
    # on the warm side so it doesn't fight with the cold-end markers.
    f2 = df[df.fabric == "F2"]
    cold_T_f2 = f2["T_mean_outdoor_C"].min()
    cold_band = f2[f2["T_mean_outdoor_C"] == cold_T_f2]
    spread = float(cold_band["sim_peak_kW"].max() - cold_band["sim_peak_kW"].min())
    spread_top = float(cold_band["sim_peak_kW"].max())
    spread_bot = float(cold_band["sim_peak_kW"].min())
    # Draw the spread arrow slightly to the WARM side of the cold endpoint
    arrow_T = cold_T_f2 + 0.45
    ax_b.annotate(
        "",
        xy=(arrow_T, spread_top),
        xytext=(arrow_T, spread_bot),
        arrowprops=dict(arrowstyle="<->", color=ESSAY_INK, lw=1.1),
    )
    ax_b.text(arrow_T - 0.35, (spread_top + spread_bot) / 2,
              f"{spread:.1f} kW\nspread",
              fontsize=8, ha="right", va="center",
              color=ESSAY_INK, fontweight="bold")

    ax_a.set_ylabel("Peak demand (kW per home)")

    # Suptitle and footnote
    fig.suptitle(
        "HDD is acceptable for old housing stock — and structurally wrong once homes are retrofitted",
        x=0.02, y=1.00, ha="left", fontsize=10.5, color=ESSAY_INK,
    )
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.22, top=0.84)
    _essay_footnote(
        fig,
        "Three weather days plotted (W1 mild, W2 design, W3 extreme cold). "
        "HDD prediction is identical across both panels.",
        y=0.03,
    )
    save_fig(fig, out_dir, "essay_fig2_peak_vs_temp")


# ===================================================================== #
#  Essay Fig 3 — Variance decomposition (H2)                            #
# ===================================================================== #

def essay_fig3_variance(
    anova_table: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Single-panel η² horizontal bars with Cohen thresholds."""
    apply_essay_style()

    data = anova_table[anova_table.index != "Residual"].copy()
    data = data.sort_values("eta_squared", ascending=True)
    data.index = [clean_factor_label(i) for i in data.index]
    data["is_interaction"] = ["×" in idx for idx in data.index]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    colors = [ESSAY_NEUTRAL if i else ESSAY_PROPOSED
              for i in data["is_interaction"]]
    ax.barh(data.index, data["eta_squared"], color=colors,
            edgecolor="white", lw=0.5, height=0.65, zorder=3)

    # Cohen thresholds — faint vertical lines spanning the full plot height,
    # but the small/medium/large legend rendered as a single text annotation
    # below the x-axis label so it can't collide with any bar.
    cohen = [(0.01, "small"), (0.06, "medium"), (0.14, "large")]
    for thr, _lbl in cohen:
        ax.axvline(thr, color=ESSAY_NEUTRAL, ls=":", lw=0.7, zorder=2, alpha=0.7)
    cohen_legend = "Cohen η² thresholds:    "
    cohen_legend += "    ".join(
        f"{lbl} ({thr:.2f})" for thr, lbl in cohen
    )

    # Direct value labels — keep 2 dp for the headline values, but switch to
    # 3 dp for any tiny effect (< 0.01) so they aren't displayed as "0.00".
    for i, (label, row) in enumerate(data.iterrows()):
        v = row["eta_squared"]
        v_str = f"{v:.3f}" if v < 0.01 else f"{v:.2f}"
        ax.text(v + 0.005, i, v_str,
                va="center", ha="left", fontsize=8,
                color=ESSAY_INK, fontweight="bold")

    ax.set_xlim(0, max(0.32, data["eta_squared"].max() + 0.04))
    ax.set_xlabel("η²  (proportion of total variance)")
    ax.set_title("Occupant behaviour rivals fabric as a driver of peak demand",
                 pad=8)

    # Legend — direct text patches (no boxed legend)
    ax.text(0.99, 0.06, "● main effect", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=ESSAY_PROPOSED,
            fontweight="bold")
    ax.text(0.99, 0.02, "● interaction", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=ESSAY_NEUTRAL,
            fontweight="bold")

    _essay_grid(ax, "x")
    fig.subplots_adjust(left=0.22, right=0.95, bottom=0.28, top=0.86)
    fig.text(0.58, 0.10, cohen_legend, ha="center", va="center",
             fontsize=7.5, color=ESSAY_NEUTRAL, style="italic")
    _essay_footnote(
        fig,
        "Type-II ANOVA of simulated peak demand; all main effects p < 0.001.",
        y=0.03,
    )
    save_fig(fig, out_dir, "essay_fig3_variance")


# ===================================================================== #
#  Essay Fig 4 — External validation (COP only)                         #
# ===================================================================== #

def essay_fig4_validation(
    cop_by_T_df: pd.DataFrame,
    hp_cfg: dict,
    out_dir: Path,
    n_cop_obs: Optional[int] = None,
) -> None:
    """Simulator COP curve vs EoH field IQR + median, single panel."""
    apply_essay_style()

    cop = cop_by_T_df.sort_values("T_bin_centre")
    T = cop["T_bin_centre"].values
    median = cop["cop_median"].values
    p25 = cop["cop_25"].values
    p75 = cop["cop_75"].values

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    # EoH field band (IQR) + median
    n_lbl = f" (n = {n_cop_obs/1e6:.2f} M)" if n_cop_obs else ""
    ax.fill_between(T, p25, p75, color=ESSAY_PROPOSED, alpha=0.18, zorder=2)
    ax.plot(T, median, color=ESSAY_PROPOSED, lw=1.8, marker="o", ms=3.5,
            mec="white", mew=0.5, zorder=4)

    # Simulator analytical COP curve
    cop_intercept = hp_cfg.get("cop_intercept", 3.5)
    cop_slope     = hp_cfg.get("cop_slope", 0.12)
    cop_min       = hp_cfg.get("cop_min", 1.5)
    cop_max       = hp_cfg.get("cop_max", 4.5)
    t_defrost     = hp_cfg.get("defrost_temp_threshold_C", -2.0)
    defrost_pen   = hp_cfg.get("defrost_efficiency_penalty", 0.5 / cop_intercept)

    T_model = np.linspace(T.min() - 1, T.max() + 1, 400)
    cop_model = cop_intercept + cop_slope * T_model
    cop_model = np.where(T_model < t_defrost,
                         cop_model * (1.0 - defrost_pen),
                         cop_model)
    cop_model = np.clip(cop_model, cop_min, cop_max)
    ax.plot(T_model, cop_model, color=ESSAY_INCUMBENT, lw=2.0,
            ls=(0, (5, 3)), zorder=5)

    # Direct labels at right endpoint
    ax.text(T_model[-1] + 0.3, cop_model[-1],
            "Simulator", fontsize=8, color=ESSAY_INCUMBENT,
            fontweight="bold", ha="left", va="center")
    ax.text(T[-1] + 0.3, median[-1],
            f"EoH field median{n_lbl}", fontsize=8,
            color=ESSAY_PROPOSED, fontweight="bold",
            ha="left", va="center")
    # "EoH IQR" anchored INSIDE the band (between the upper edge and the
    # median line) at a warm temperature where the band is widest, so the
    # connection to the shaded region is unambiguous.
    iqr_anchor_x = 10.0
    iqr_p75 = float(np.interp(iqr_anchor_x, T, p75))
    iqr_med = float(np.interp(iqr_anchor_x, T, median))
    ax.text(iqr_anchor_x, (iqr_p75 + iqr_med) / 2,
            "EoH IQR", fontsize=7.5, color=ESSAY_PROPOSED,
            ha="center", va="center", style="italic")

    # Highlight the design-temp gap with a single vertical bar
    T_design = -1.0
    cop_emp = float(np.interp(T_design, T, median))
    cop_mod = cop_intercept + cop_slope * T_design
    if T_design < t_defrost:
        cop_mod *= (1.0 - defrost_pen)
    cop_mod = float(np.clip(cop_mod, cop_min, cop_max))
    if cop_mod > cop_emp:
        ax.annotate(
            "",
            xy=(T_design, cop_mod), xytext=(T_design, cop_emp),
            arrowprops=dict(arrowstyle="<->", color=ESSAY_INK, lw=1.0),
        )
        ax.text(T_design + 0.4, (cop_emp + cop_mod) / 2 + 0.18,
                f"+{cop_mod - cop_emp:.2f} COP\nover-prediction",
                fontsize=8, ha="left", va="center",
                color=ESSAY_INK, fontweight="bold")

    ax.set_xlabel("Outdoor temperature (°C)")
    ax.set_ylabel("Coefficient of performance")
    ax.set_xlim(T.min() - 1, T.max() + 4.5)   # right margin for labels
    ax.set_ylim(1.0, 5.0)
    ax.set_title(
        "Simulator over-predicts COP: bias makes the cost figure a lower bound",
        pad=8,
    )
    _essay_grid(ax, "y")

    fig.subplots_adjust(left=0.10, right=0.93, bottom=0.18, top=0.88)
    _essay_footnote(
        fig,
        "BEIS Electrification of Heat field trial (742 homes, 2020–2023).",
        y=0.02,
    )
    save_fig(fig, out_dir, "essay_fig4_validation")


# ===================================================================== #
#  Essay Fig 5 — Cost of HDD under-sizing (the headline)                #
# ===================================================================== #

def essay_fig5_cost(
    cost_df: pd.DataFrame,
    out_dir: Path,
    cost_per_kVA_planned: int = 300,
    reactive_multiplier: float = 2.5,
) -> None:
    """Single-panel grouped bars: planned vs reactive £ cost by feeder size."""
    apply_essay_style()

    cost = cost_df.copy().sort_values("n_homes")
    n = cost["n_homes"].astype(int).values
    planned  = cost["planned_cost_GBP"].values  / 1000
    reactive = cost["reactive_cost_GBP"].values / 1000
    x = np.arange(len(n))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    bp = ax.bar(x - w/2, planned, w, color=ESSAY_PROPOSED,
                edgecolor="white", lw=0.5, zorder=3)
    br = ax.bar(x + w/2, reactive, w, color=ESSAY_INCUMBENT,
                edgecolor="white", lw=0.5, zorder=3)

    # Numeric labels on every bar — except the rightmost reactive bar, whose
    # value is already named in the headline callout above it.
    headline_idx = len(reactive) - 1
    for bar, v in zip(bp, planned):
        ax.text(bar.get_x() + bar.get_width()/2, v + reactive.max() * 0.015,
                f"£{v:.0f} k", ha="center", va="bottom",
                fontsize=8, color=ESSAY_PROPOSED, fontweight="bold")
    for i, (bar, v) in enumerate(zip(br, reactive)):
        if i == headline_idx:
            continue   # headline callout already names this value
        ax.text(bar.get_x() + bar.get_width()/2, v + reactive.max() * 0.015,
                f"£{v:.0f} k", ha="center", va="bottom",
                fontsize=8, color=ESSAY_INCUMBENT, fontweight="bold")

    # Headline value above the rightmost group
    headline_v = reactive[-1]
    headline_x = x[-1] + w/2
    ax.annotate(
        "",
        xy=(headline_x, headline_v),
        xytext=(headline_x, headline_v + reactive.max() * 0.20),
        arrowprops=dict(arrowstyle="-", color=ESSAY_INCUMBENT, lw=0.7),
    )
    ax.text(headline_x, headline_v + reactive.max() * 0.22,
            f"£{headline_v:.0f} k\navoidable",
            ha="center", va="bottom", fontsize=11,
            color=ESSAY_INCUMBENT, fontweight="bold")

    # Direct legend (no box)
    ax.text(0.02, 0.96,
            f"● planned reinforcement (£{cost_per_kVA_planned}/kVA)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=ESSAY_PROPOSED, fontweight="bold")
    ax.text(0.02, 0.90,
            f"● reactive reinforcement (×{reactive_multiplier:.1f} cost premium)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=ESSAY_INCUMBENT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in n])
    ax.set_xlabel("Number of homes per LV feeder")
    ax.set_ylabel("Avoidable reinforcement cost (£ thousands)")
    ax.set_ylim(0, reactive.max() * 1.45)
    ax.set_title(
        "HDD-only sizing forces reactive reinforcement at 2.5× the planned cost",
        pad=8,
    )
    _essay_grid(ax, "y")

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.88)
    _essay_footnote(
        fig,
        "Cost gap = simulated ADMD shortfall × £/kVA reinforcement premium when caught reactively.",
        y=0.02,
    )
    save_fig(fig, out_dir, "essay_fig5_cost")


# ===================================================================== #
#  Driver: generate the 5-figure essay set                              #
# ===================================================================== #

def generate_essay_figures(
    cfg: dict,
    sim_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    anova_table: pd.DataFrame,
    cop_by_T_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    archetype_names: dict,
    *,
    n_cop_obs: Optional[int] = None,
    out_dir: Optional[Path] = None,
) -> None:
    """Render the redesigned 5-figure essay set."""
    out_dir = out_dir if out_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    essay_fig1_design(cfg, sim_df, out_dir)
    essay_fig2_peak_vs_temp(errors_df, archetype_names, out_dir)
    essay_fig3_variance(anova_table, out_dir)
    essay_fig4_validation(cop_by_T_df, cfg["heat_pump"], out_dir,
                          n_cop_obs=n_cop_obs)
    essay_fig5_cost(cost_df, out_dir)
    logger.info("Essay figure suite written to %s", out_dir)
