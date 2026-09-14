#!/usr/bin/env python3
"""
Filter-valid expansion ratio by tier.
Computes valid_ratio = valid_scanned / total_scanned per layer (tier),
grouped by selectivity bucket, and plots as a grouped bar chart.

Layout (2x3):
    Rows: LAION, YTB-Audio
    Cols: Equality, Containment, Overlap
"""

import ctypes, os, sys
from pathlib import Path

def _preload_conda_libstdcpp():
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if not conda_prefix:
        return
    lib_path = Path(conda_prefix) / "lib" / "libstdc++.so.6"
    if lib_path.exists():
        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

_preload_conda_libstdcpp()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]   = ["Times"]
plt.rcParams["text.usetex"]  = True

RESULTS_DIR  = "../expansion_ratio_results"
PLOT_DIR     = "expansion_ratio"
DATASETS     = ["LAION1M", "ytb_audio"]
FILTERS      = ["equality", "containment", "overlap"]
DATASET_NAME = {"LAION1M": "LAION", "ytb_audio": "YTB-Audio"}
FILTER_NAME  = {"equality": "Equality", "containment": "Containment", "overlap": "Overlap"}

# 9 tiers: T1 first bar, T9 last bar
# Purple gradient (plasma, restricted to the purple family — matches the
# paper's bitmap/heatmap palette), light = T1, dark = T9
TIER_CMAP = plt.cm.plasma
TIER_LABELS  = ["T%d" % i for i in range(1, 10)]  # T1..T9
NUM_LAYERS = 9
LAYERS = list(range(NUM_LAYERS))  # 0..8


def load_all():
    """Load per-query rows and aggregate total/valid per layer across all queries."""
    data = {}
    for ds in DATASETS:
        for flt in FILTERS:
            path = os.path.join(RESULTS_DIR, f"expansion_{ds}_{flt}.csv")
            if not os.path.exists(path):
                continue
            # Read CSV, skip the summary section
            raw = pd.read_csv(path, comment="#")
            # Per-query rows: filter_type column is non-empty (has filter name)
            # Summary rows also have filter_type; distinguish by column name
            cols = raw.columns.tolist()
            scanned_cols = [c for c in cols if "total_scanned" in c and not c.startswith("avg_")]
            if not scanned_cols:
                continue
            # Keep only per-query rows (first unlabeled section)
            # Summary rows start after an empty-row separator
            # Keep per-query rows (have ef column) and convert numeric columns
            df = raw.dropna(subset=["ef"])
            if df.empty:
                continue
            for layer in range(NUM_LAYERS):
                for suffix in ["total_scanned", "valid_scanned"]:
                    col = f"layer_{layer}_{suffix}"
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            data[(ds, flt)] = df
    return data


def main():
    data = load_all()
    if not data:
        print("No data found. Run search_expansion_ratio.sh first.")
        sys.exit(1)

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Match plot_building_param.py style
    n_cols, n_rows = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.5, n_rows * 3))
    fig.subplots_adjust(hspace=0.24, wspace=0.22, top=0.92)

    bar_width = 0.7
    x = np.arange(NUM_LAYERS)  # 0..8

    for row_ds, ds in enumerate(DATASETS):
        for col_flt, flt in enumerate(FILTERS):
            ax = axes[row_ds, col_flt]
            df = data.get((ds, flt))
            if df is None:
                continue

            total_by_layer = np.zeros(NUM_LAYERS)
            valid_by_layer = np.zeros(NUM_LAYERS)
            for layer in range(NUM_LAYERS):
                col_t = f"layer_{layer}_total_scanned"
                col_v = f"layer_{layer}_valid_scanned"
                if col_t in df.columns and col_v in df.columns:
                    total_by_layer[layer] = pd.to_numeric(df[col_t]).sum()
                    valid_by_layer[layer] = pd.to_numeric(df[col_v]).sum()

            ratios = np.zeros(NUM_LAYERS)
            colors_r = []
            for pos in range(NUM_LAYERS):
                layer = 8 - pos
                if total_by_layer[layer] > 0:
                    ratios[pos] = valid_by_layer[layer] / total_by_layer[layer]
                # light purple (T1) -> dark purple (T9): plasma 0.6 down to 0.0
                colors_r.append(TIER_CMAP(0.6 - 0.6 * pos / (NUM_LAYERS - 1)))

            ax.bar(x, ratios, bar_width, color=colors_r, edgecolor="white", linewidth=0.3)

            ax.set_xticks(x)
            ax.set_xticklabels(TIER_LABELS, fontsize=22, rotation=45)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.tick_params(axis="y", labelsize=24)

            if row_ds == 0:
                ax.set_title(FILTER_NAME[flt], fontsize=30, fontweight="bold", pad=3)
            if col_flt == 0:
                ax.set_ylabel(f"{DATASET_NAME[ds]}\nValid Ratio", fontsize=30, labelpad=0)
            if row_ds == 1:
                ax.set_xlabel("Tier", fontsize=30, labelpad=0)

    fig.savefig(os.path.join(PLOT_DIR, "expansion_ratio.pdf"),
                dpi=300, bbox_inches="tight")
    print(f"Saved {PLOT_DIR}/expansion_ratio.pdf")


if __name__ == "__main__":
    main()
