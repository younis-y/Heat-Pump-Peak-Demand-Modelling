"""
Placeholder for an EnergyPlus single-archetype benchmark.

In a full study this module would call EnergyPlus via eppy or similar.
Here it simply documents the comparison methodology and provides a
stub that returns the HDD-benchmark result as a conservative proxy.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def note() -> str:
    """Return a description of the EnergyPlus benchmark methodology.

    Returns
    -------
    str
        Methodology note.
    """
    logger.info("EnergyPlus benchmark is a stub — see docstring for methodology.")
    return (
        "EnergyPlus single-archetype benchmark: In production, an EnergyPlus "
        "model of a single 'average' dwelling would be run for each weather "
        "scenario to obtain a peak demand estimate.  This represents the "
        "current best-practice planning approach that still ignores occupant "
        "behaviour diversity.  The HDD regression benchmark already captures "
        "the same conceptual limitation and is used as the primary comparator "
        "in this study."
    )
