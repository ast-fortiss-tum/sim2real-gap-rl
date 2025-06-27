#!/usr/bin/env python3
"""
plot_observation_metrics_scaled.py

Visualise image-comparison metrics produced by
   compare_dataset_images.py
and stored in:
   observations_image_compare_metrics.json

Figures produced
----------------
• metrics_barplot_log.png   – mean ± 1 σ on a log-scaled y-axis
• metrics_boxplot_split.png – two-panel box-plot (large- vs small-scale)

Run:
    python3 plot_observation_metrics_scaled.py \
            Dataset_DC/compare_observations/observations_image_compare_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────  PLOT AESTHETICS  ────────────────────────────
plt.rcParams.update({
    "figure.dpi": 140,
    "axes.titlesize": "x-large",
    "axes.labelsize": "large",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
})
# ─────────────────────────────────────────────────────────────────────────────


def load_metrics(json_path: Path) -> Dict[str, List[float]]:
    """Return {metric_name: [values]} (drops None)."""
    with json_path.open() as jf:
        data = json.load(jf)

    coll: Dict[str, List[float]] = {}
    for row in data["metrics"]:
        mset = row.get("sim_vs_real") or {}
        for m_name, val in mset.items():
            if val is None:
                continue
            coll.setdefault(m_name, []).append(val)
    return coll


# ───────────────────────  BAR PLOT (LOG-SCALE)  ─────────────────────────────
def bar_plot_log(metrics: Dict[str, List[float]], out_path: Path) -> None:
    eps = 1e-8  # for zero / negative protection
    names = list(metrics)
    means = np.array([np.mean(metrics[m]) for m in names])
    stds  = np.array([np.std(metrics[m], ddof=1) for m in names])

    # Handle non-positive means: shift them slightly to plot on log scale
    if (means <= 0).any():
        print("[WARN] Some means ≤ 0 – shifting by ε so they appear on log-scale")
        means = np.where(means > 0, means, eps)
        stds  = np.where(stds  > 0, stds,  eps)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, means, yerr=stds, capsize=6, log=True)
    ax.set_ylabel("mean value (log10 scale)  ±1 σ")
    ax.set_title("Image-comparison metrics – log-scale overview")
    ax.grid(axis="y", which="both")
    ax.set_xticklabels(names, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"saved → {out_path}")


# ───────────────  BOX-PLOT (SPLIT BY MAGNITUDE)  ────────────────────────────
def split_metrics(metrics: Dict[str, List[float]], thresh: float = 1.0
                  ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    big, small = {}, {}
    for k, vals in metrics.items():
        if np.median(vals) >= thresh:
            big[k] = vals
        else:
            small[k] = vals
    return big, small


def box_plot_split(metrics: Dict[str, List[float]], out_path: Path) -> None:
    big, small = split_metrics(metrics)

    n_panels = (1 if not big else 1) + (1 if small else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 + 3 * (n_panels - 1)),
                             sharex=False)
    if n_panels == 1:
        axes = [axes]  # ensure iterable

    # Plot big-scale metrics (median ≥ 1)
    if big:
        ax = axes[0]
        names = list(big)
        ax.boxplot([big[n] for n in names], labels=names, showmeans=True)
        ax.set_title("Metrics with median ≥ 1")
        ax.grid(axis="y")
        ax.set_xticklabels(names, rotation=45, ha="right")

    # Plot small-scale metrics
    if small:
        ax = axes[-1]  # same if only one panel
        names = list(small)
        ax.boxplot([small[n] for n in names], labels=names, showmeans=True)
        ax.set_title("Metrics with median < 1")
        ax.grid(axis="y")
        ax.set_xticklabels(names, rotation=45, ha="right")

    fig.suptitle("Metric distributions (split linear scale)", y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path)
    print(f"saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 plot_observation_metrics_scaled.py <metrics.json>")
        sys.exit(1)

    jpath = Path(sys.argv[1]).expanduser()
    if not jpath.is_file():
        print(f"File not found: {jpath}")
        sys.exit(1)

    metrics = load_metrics(jpath)
    if not metrics:
        print("No numeric metrics detected.")
        sys.exit(0)

    out_dir = jpath.parent
    bar_plot_log(metrics, out_dir / "metrics_barplot_log.png")
    box_plot_split(metrics, out_dir / "metrics_boxplot_split.png")

    plt.show()  # comment out if running headless


if __name__ == "__main__":
    main()
