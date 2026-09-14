#!/usr/bin/env python3
"""Insertion-drift figures for paper: 3x4 grid + summary table.

Left two columns: Experiment 1 — recall/QPS vs fraction of data inserted
  (B0-B4 batches with varying held-out label fractions).
Right two columns: Experiment 2 — recall/QPS vs FI fraction
  (FI0-FI4 batches with varying frequency-inversion fractions).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]   = ["Times"]
plt.rcParams["text.usetex"]  = True

RESULTS_DIR = "../results/insertion_drift"
PLOT_DIR    = "insertion_drift"

COLOR_IVF     = "#d62728"  # red
COLOR_MINHASH = "#1f77b4"  # blue

SCENARIOS = ["equality", "containment", "overlap"]
DATASETS  = ["LAION1M", "ytb_audio"]
VARIANTS  = ["ivf", "minhash"]

COLD_BATCHES = ["B0", "B1", "B2", "B3", "B4"]
FI_BATCHES   = ["FI0", "FI1", "FI2", "FI3", "FI4"]
FI_FRACTION  = {"FI0": 0.0, "FI1": 0.25, "FI2": 0.50, "FI3": 0.75, "FI4": 1.0}

DATASET_LABEL = {"LAION1M": "LAION", "ytb_audio": "YTB-Audio"}
METHOD_LABEL  = {"ivf": "LSSG-IVF", "minhash": "LSSG-MinHash"}
MARKER        = {"ivf": "o", "minhash": "^"}
COLOR         = {"ivf": COLOR_IVF, "minhash": COLOR_MINHASH}


def load_all():
    data = {}
    for ds in DATASETS:
        for var in VARIANTS:
            path = os.path.join(RESULTS_DIR, f"{ds}_{var}_drift.csv")
            if not os.path.exists(path):
                print(f"WARNING: {path} not found, skipping")
                continue
            data[(ds, var)] = pd.read_csv(path)
    return data


def main():
    data = load_all()
    if not data:
        print("No data found. Run the experiment first.")
        sys.exit(1)

    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(4 * 3.5, 3 * 2.5), dpi=150)
    fig.subplots_adjust(hspace=0.22, wspace=0.65, top=0.92)
    axes_top_left = {}  # (ax1, ax2) per dataset for later title placement

    for row, scenario in enumerate(SCENARIOS):
        for col_ds, dataset in enumerate(DATASETS):
            col_exp1 = col_ds * 2
            col_exp2 = col_ds * 2 + 1

            ax1 = axes[row, col_exp1]
            ax1b = ax1.twinx()
            ax2 = axes[row, col_exp2]
            ax2b = ax2.twinx()

            # ---- Experiment 1 ----
            batch_x = {b: 0.6 + i * 0.1 for i, b in enumerate(COLD_BATCHES)}

            for var in VARIANTS:
                clr = COLOR[var]
                mk = MARKER[var]
                name = METHOD_LABEL[var]
                df = data.get((dataset, var))
                if df is None:
                    continue
                exp1 = df[(df["batch"].isin(COLD_BATCHES)) &
                          (df["scenario"] == scenario)].copy()
                if exp1.empty:
                    continue
                exp1["cum_frac"] = exp1["batch"].map(batch_x)
                exp1 = exp1.sort_values("cum_frac")

                # Recall — solid line on left axis
                ax1.plot(exp1["cum_frac"], exp1["recall"],
                         color=clr, marker=mk, markersize=8,
                         markerfacecolor="none", linewidth=2, zorder=3)

                # QPS — dotted line on right axis
                ax1b.plot(exp1["cum_frac"], exp1["qps"],
                          color=clr, linestyle=":", marker=mk,
                          markersize=5, markerfacecolor="none",
                          linewidth=1.2, alpha=0.5, zorder=2)

                # Init baseline
                init_row = df[(df["batch"] == "init") & (df["scenario"] == scenario)]
                if not init_row.empty:
                    ax1.axhline(y=init_row["recall"].values[0],
                                color=clr, linestyle="--", linewidth=1.5,
                                alpha=0.7, zorder=1)

            ax1.set_xlim(0.58, 1.02)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.tick_params(axis="y", which="major", labelsize=16, labelrotation=45)
            ax1b.tick_params(axis="y", which="major", labelsize=16, colors="black", labelrotation=45)

            # ---- Experiment 2 ----
            for var in VARIANTS:
                clr = COLOR[var]
                mk = MARKER[var]
                name = METHOD_LABEL[var]
                df = data.get((dataset, var))
                if df is None:
                    continue
                exp2 = df[(df["batch"].isin(FI_BATCHES)) &
                          (df["scenario"] == scenario)].copy()
                if exp2.empty:
                    continue
                exp2["fi_frac"] = exp2["batch"].map(FI_FRACTION)
                exp2 = exp2.sort_values("fi_frac")

                ax2.plot(exp2["fi_frac"], exp2["recall"],
                         color=clr, marker=mk, markersize=8,
                         markerfacecolor="none", linewidth=2, zorder=3)

                ax2b.plot(exp2["fi_frac"], exp2["qps"],
                          color=clr, linestyle=":", marker=mk,
                          markersize=5, markerfacecolor="none",
                          linewidth=1.2, alpha=0.5, zorder=2)

                init_row = df[(df["batch"] == "init") & (df["scenario"] == scenario)]
                if not init_row.empty:
                    ax2.axhline(y=init_row["recall"].values[0],
                                color=clr, linestyle="--", linewidth=1.5,
                                alpha=0.7, zorder=1)

            ax2.set_xlim(-0.05, 1.05)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.tick_params(axis="y", which="major", labelsize=16, labelrotation=45)
            ax2b.tick_params(axis="y", which="major", labelsize=16, colors="black", labelrotation=45)

            # X ticks
            ax1.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
            ax1.set_xticklabels([".6", ".7", ".8", ".9", "1.0"], fontsize=20)
            ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax2.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=20)

            # Subplot titles (top row: experiment type)
            if row == 0:
                ax1.set_title("Held-Out Labels", fontsize=22, fontweight="bold", pad=6)
                ax2.set_title("Frequency Shift", fontsize=22, fontweight="bold", pad=6)

            # Store top-row axes for dataset name placement
            if row == 0:
                axes_top_left.setdefault(dataset, (ax1, ax2))

            # Y labels
            if col_ds == 0:
                sc_name = {"equality": "Equality", "containment": "Containment",
                           "overlap": "Overlap"}[scenario]
                ax1.set_ylabel(f"{sc_name}\nRecall@10", fontsize=22)
            if col_ds == 1:
                ax2b.set_ylabel("QPS", fontsize=19, color="black")

            # X labels (bottom row)
            if row == 2:
                ax1.set_xlabel("Fraction of data inserted", fontsize=22, fontweight="bold")
                ax2.set_xlabel("Inversion fraction", fontsize=22, fontweight="bold")

    # Legend: 2 rows (IVF, MinHash) × 3 cols (Recall, Init. Recall, QPS)
    handles = [
        plt.Line2D([0], [0], color=COLOR_IVF, marker=MARKER["ivf"],
                   markersize=8, markerfacecolor="none", linewidth=2,
                   label="LSSG-IVF (Recall@10)"),
        plt.Line2D([0], [0], color=COLOR_MINHASH, marker=MARKER["minhash"],
                   markersize=8, markerfacecolor="none", linewidth=2,
                   label="LSSG-MinHash (Recall@10)"),
        plt.Line2D([0], [0], color=COLOR_IVF, linestyle="--",
                   linewidth=1.5, label="LSSG-IVF (Init. Recall)"),
        plt.Line2D([0], [0], color=COLOR_MINHASH, linestyle="--",
                   linewidth=1.5, label="LSSG-MinHash (Init. Recall)"),
        plt.Line2D([0], [0], color=COLOR_IVF, linestyle=":",
                   marker=MARKER["ivf"], markersize=5,
                   markerfacecolor="none", linewidth=1.2, alpha=0.5,
                   label="LSSG-IVF (QPS)"),
        plt.Line2D([0], [0], color=COLOR_MINHASH, linestyle=":",
                   marker=MARKER["minhash"], markersize=5,
                   markerfacecolor="none", linewidth=1.2, alpha=0.5,
                   label="LSSG-MinHash (QPS)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               fontsize=23, frameon=False,
               bbox_to_anchor=(0.5, 1.15), handletextpad=0.3,
               columnspacing=0.8, handlelength=1.5)

    # subplots_adjust controls spacing; remove tight_layout to avoid override

    # Dataset names above each column pair (after tight_layout for correct coords)
    for dataset, (a1, a2) in axes_top_left.items():
        b1 = a1.get_position()
        b2 = a2.get_position()
        x_mid = (b1.x0 + b2.x1) / 2
        fig.text(x_mid, b1.y1 + 0.04, DATASET_LABEL[dataset],
                 fontsize=22, fontweight="bold", ha="center", va="bottom")

    fig.savefig(os.path.join(PLOT_DIR, "insertion_drift.pdf"),
                dpi=300, bbox_inches="tight")
    print(f"Saved {PLOT_DIR}/insertion_drift.pdf")

    # ---- LaTeX Table ----
    print("\n% LaTeX summary table for insertion-drift experiment")
    print(r"\begin{tabular}{llrrrrrrrr}")
    print(r"\toprule")
    print(r"Dataset & Variant & $\Delta R^{\rm cold}_{\rm cont}$ & $\Delta R^{\rm cold}_{\rm eq}$ & $\Delta R^{\rm cold}_{\rm ov}$ & $\Delta R^{\rm fi}_{\rm cont}$ & $\Delta R^{\rm fi}_{\rm eq}$ & $\Delta R^{\rm fi}_{\rm ov}$ & $\Delta Q^{\rm cold}$ & $\Delta Q^{\rm fi}$ \\")
    print(r"\midrule")
    for ds in DATASETS:
        for var in VARIANTS:
            df = data.get((ds, var))
            if df is None:
                continue
            dr_cold, dr_fi = [], []
            dq_cold, dq_fi = 0.0, 0.0
            for sc in SCENARIOS:
                b0 = df[(df["batch"] == "B0") & (df["scenario"] == sc)]
                b4 = df[(df["batch"] == "B4") & (df["scenario"] == sc)]
                fi0 = df[(df["batch"] == "FI0") & (df["scenario"] == sc)]
                fi4 = df[(df["batch"] == "FI4") & (df["scenario"] == sc)]
                dr_cold.append((b4["recall"].values[0] - b0["recall"].values[0]) if (not b0.empty and not b4.empty) else float("nan"))
                dr_fi.append((fi4["recall"].values[0] - fi0["recall"].values[0]) if (not fi0.empty and not fi4.empty) else float("nan"))
                dq_cold += (b4["qps"].values[0] - b0["qps"].values[0]) if (not b0.empty and not b4.empty) else 0
                dq_fi   += (fi4["qps"].values[0] - fi0["qps"].values[0]) if (not fi0.empty and not fi4.empty) else 0
            print(f"{DATASET_LABEL[ds]} & {METHOD_LABEL[var]} & "
                  f"{dr_cold[0]:+.3f} & {dr_cold[1]:+.3f} & {dr_cold[2]:+.3f} & "
                  f"{dr_fi[0]:+.3f} & {dr_fi[1]:+.3f} & {dr_fi[2]:+.3f} & "
                  f"{dq_cold/3:+.0f} & {dq_fi/3:+.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
