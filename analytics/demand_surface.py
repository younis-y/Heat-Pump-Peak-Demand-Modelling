"""
Demand surface heatmap generation.

Produces a 2D surface of electricity demand (kW) as a function of:
- X: outdoor temperature (°C)
- Y: hour of day (0–23)
for each occupant-behaviour archetype, with HDD-predicted demand overlaid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
