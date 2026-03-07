"""
Archetype clustering: discover behavioural demand archetypes from EoH data.

Uses k-means clustering on normalised daily load profiles to identify
dominant heating patterns. Compares discovered clusters against the
4 assumed archetypes (Early Riser, Home All Day, Late Returner, Intermittent).
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# Half-hourly periods in a day
PERIODS_PER_DAY = 48


def build_daily_profiles(eoh_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape half-hourly data into one row per property-day with 48 demand columns.

    Parameters
    ----------
    eoh_df : pd.DataFrame
        Half-hourly EoH data with elec_demand_kW, Property_ID, date, hour.

    Returns
    -------
    pd.DataFrame
        Pivoted: rows = (Property_ID, date), columns = period_0..period_47.
    """
    # Use hp_thermal_kW to capture heating patterns (not masked by base loads)
    demand_col = "hp_thermal_kW" if "hp_thermal_kW" in eoh_df.columns else "elec_demand_kW"
    df = eoh_df[["Property_ID", "date", "hour", demand_col]].dropna().copy()

    # Create half-hour period index (0-47)
    df["period"] = (df["hour"] * 2).astype(int).clip(0, 47)

    # Pivot to wide format
    pivoted = df.pivot_table(
        index=["Property_ID", "date"],
        columns="period",
        values=demand_col,
        aggfunc="mean",
    )

    # Only keep complete days (all 48 periods present)
    pivoted = pivoted.dropna(thresh=44)  # allow up to 4 missing periods
    pivoted = pivoted.ffill(axis=1).bfill(axis=1)

    # Rename columns
    pivoted.columns = [f"period_{int(c)}" for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    return pivoted


def filter_and_normalise_profiles(profiles: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter out flat profiles and normalise to [0, 1] for clustering.

    Parameters
    ----------
    profiles : pd.DataFrame
        Wide-format daily profiles.

    Returns
    -------
    tuple
        (filtered_profiles DataFrame, normalised matrix)
    """
    period_cols = [c for c in profiles.columns if c.startswith("period_")]
    X = profiles[period_cols].values.copy()

    # Filter out flat/near-zero profiles (< 50W variation across day)
    row_range = X.max(axis=1) - X.min(axis=1)
    valid_mask = row_range > 0.05
    profiles_filt = profiles[valid_mask].copy()
    X = X[valid_mask]

    # Min-max normalise each row
    row_min = X.min(axis=1, keepdims=True)
    row_max = X.max(axis=1, keepdims=True)
    denom = row_max - row_min
    denom[denom < 0.01] = 1.0
    X_norm = (X - row_min) / denom

    return profiles_filt, X_norm


def cluster_profiles(
    X_norm: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans, float]:
    """Run k-means clustering on pre-normalised profiles.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalised profile matrix (n_profiles, 48).
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (labels, kmeans_model, silhouette_score)
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_norm)

    sil = silhouette_score(X_norm, labels, sample_size=min(10000, len(X_norm)))
    logger.info("K-means (k=%d): silhouette=%.3f", n_clusters, sil)

    return labels, km, sil


def discover_archetypes(
    eoh_df: pd.DataFrame,
    out_dir: Path,
    max_clusters: int = 6,
) -> dict:
    """Discover demand archetypes from EoH data and produce figures.

    Parameters
    ----------
    eoh_df : pd.DataFrame
        Half-hourly EoH data.
    out_dir : Path
        Output directory.
    max_clusters : int
        Maximum number of clusters to evaluate.

    Returns
    -------
    dict
        Clustering results: best_k, silhouette scores, cluster sizes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building daily demand profiles...")
    profiles = build_daily_profiles(eoh_df)
    logger.info("Built %d daily profiles", len(profiles))

    # Filter out flat profiles and normalise
    profiles, X_norm = filter_and_normalise_profiles(profiles)
    logger.info("After filtering flat profiles: %d remain", len(profiles))

    if len(profiles) < 100:
        logger.warning("Too few profiles (%d) for clustering", len(profiles))
        return {"error": "insufficient data", "n_profiles": len(profiles)}

    # Evaluate multiple k values
    sil_scores = {}
    for k in range(2, max_clusters + 1):
        _, _, sil = cluster_profiles(X_norm, n_clusters=k)
        sil_scores[k] = sil

    best_k = max(sil_scores, key=sil_scores.get)
    logger.info("Best k=%d (silhouette=%.3f)", best_k, sil_scores[best_k])

    # Final clustering with best k
    labels, km, sil = cluster_profiles(X_norm, n_clusters=best_k)
    profiles = profiles.copy()
    profiles["cluster"] = labels

    period_cols = [c for c in profiles.columns if c.startswith("period_")]

    # Normalised centroids from k-means (these capture shape, not magnitude)
    centroids_norm = km.cluster_centers_  # shape (best_k, 48)

    # Also compute raw centroids for magnitude context
    X_raw = profiles[period_cols].values
    centroids_kW = []
    for c in range(best_k):
        mask = labels == c
        centroids_kW.append(np.median(X_raw[mask], axis=0))
    centroids_kW = np.array(centroids_kW)

    # Cluster summary
    cluster_summary = []
    for c in range(best_k):
        mask = labels == c
        raw = X_raw[mask]
        norm_c = centroids_norm[c]
        cluster_summary.append({
            "cluster": c,
            "n_profiles": int(mask.sum()),
            "pct_profiles": round(100 * mask.sum() / len(labels), 1),
            "mean_peak_kW": round(float(raw.max(axis=1).mean()), 3),
            "median_mean_kW": round(float(np.median(raw.mean(axis=1))), 3),
            "peak_hour": round(float(norm_c.argmax() / 2), 1),
            "trough_hour": round(float(norm_c.argmin() / 2), 1),
        })

    summary_df = pd.DataFrame(cluster_summary)
    summary_df.to_csv(out_dir / "archetype_cluster_summary.csv", index=False)

    # Save silhouette scores
    sil_df = pd.DataFrame(list(sil_scores.items()), columns=["k", "silhouette"])
    sil_df.to_csv(out_dir / "silhouette_scores.csv", index=False)

    # --- Figure 1: Normalised cluster centroid shapes ---
    sns.set_palette("colorblind")
    hours = np.arange(PERIODS_PER_DAY) / 2  # 0.0 to 23.5

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: normalised shapes (what k-means actually clustered on)
    ax = axes[0]
    for c in range(best_k):
        n = int((labels == c).sum())
        ax.plot(
            hours, centroids_norm[c],
            linewidth=2.5, label=f"Cluster {c} (n={n:,})",
        )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised demand (0–1)")
    ax.set_title(f"Normalised Heating Profile Shapes (k={best_k})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))

    # Right: raw median demand by cluster
    ax = axes[1]
    for c in range(best_k):
        n = int((labels == c).sum())
        ax.plot(
            hours, centroids_kW[c],
            linewidth=2.5, label=f"Cluster {c} (n={n:,})",
        )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Median thermal output (kW)")
    ax.set_title("Raw Demand Magnitude by Cluster")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))

    fig.suptitle(f"Discovered Heating Archetypes (silhouette={sil:.2f})", fontsize=13)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_archetype_centroids.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Silhouette score vs k ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(sil_scores.keys()), list(sil_scores.values()), "o-", color="C0")
    ax.axvline(best_k, color="C1", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Optimal Number of Demand Archetypes")
    ax.legend()
    ax.grid(alpha=0.3)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_silhouette_scores.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Heatmap of cluster centroids ---
    fig, ax = plt.subplots(figsize=(12, 3 + best_k * 0.8))
    im = ax.imshow(centroids_norm, aspect="auto", cmap="YlOrRd",
                    extent=[0, 24, best_k - 0.5, -0.5])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Cluster")
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"C{c} ({int((labels==c).sum())})" for c in range(best_k)])
    ax.set_xticks(range(0, 25, 2))
    ax.set_title("Demand Profile Heatmap by Cluster")
    fig.colorbar(im, ax=ax, label="Normalised demand (0–1)")

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_archetype_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: k=4 forced comparison (matching assumed 4 archetypes) ---
    if best_k != 4 and 4 in sil_scores:
        labels_4, km_4, sil_4 = cluster_profiles(X_norm, n_clusters=4)
        centroids_4 = km_4.cluster_centers_

        archetype_names = ["Early Riser", "Home All Day", "Late Returner", "Intermittent"]
        # Sort clusters by peak hour to roughly match assumed archetypes
        peak_hours = [c.argmax() / 2 for c in centroids_4]
        sort_idx = np.argsort(peak_hours)

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, idx in enumerate(sort_idx):
            n = int((labels_4 == idx).sum())
            peak_h = peak_hours[idx]
            ax.plot(
                hours, centroids_4[idx],
                linewidth=2.5,
                label=f"C{i} peak@{peak_h:.0f}h (n={n:,}) — cf. {archetype_names[i]}",
            )
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Normalised demand (0–1)")
        ax.set_title(f"Forced k=4 Archetypes vs Assumed Profiles (silhouette={sil_4:.2f})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))

        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"fig_k4_archetypes.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    logger.info("Saved archetype clustering figures to %s", out_dir)

    return {
        "best_k": best_k,
        "silhouette_scores": sil_scores,
        "cluster_sizes": {c: int((labels == c).sum()) for c in range(best_k)},
        "n_profiles": len(profiles),
    }
