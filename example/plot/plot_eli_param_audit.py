#!/usr/bin/env python3
import argparse
import ctypes
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def _preload_conda_libstdcpp():
    """Ensure Conda's libstdc++ is loaded before C-extension imports (e.g., pandas)."""
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if not conda_prefix:
        return

    lib_path = Path(conda_prefix) / "lib" / "libstdc++.so.6"
    if not lib_path.exists():
        return

    # Load globally so dependent extension modules can resolve GLIBCXX symbols.
    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


_preload_conda_libstdcpp()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True
plt.rcParams["axes.unicode_minus"] = False

METHOD_NAME = "ELI"
DATASET_ORDER = ["sift1m", "LAION1M", "YTB-Audio"]
DATASET_LABEL = {"sift1m": "SIFT", "LAION1M": "LAION", "YTB-Audio":"YTB-Audio"}
SEM_ORDER = ["equality", "containment", "overlap"]
SEM_LABEL = {"equality": "Equality", "containment": "Containment", "overlap": "Overlap"}

CORE_M = 16
CORE_EFC = 200
SUGGESTED_ELASTIC = 0.2

ELASTIC_STYLES = {
    0.2: {"color": "#b6042a", "linestyle": "-",  "marker": "s"},
    0.4: {"color": "#f27b50", "linestyle": "--", "marker": "o"},
    0.6: {"color": "#5e62a9", "linestyle": "-.", "marker": "^"},
}

DATASET_ALIASES = {
    "sift1m": "sift1m",
    "laion1m": "LAION1M",
    "LAION1M": "LAION1M",
    "ytb_audio": "YTB-Audio",
    "ytb-audio": "YTB-Audio",
    "YTB-Audio": "YTB-Audio",
}


def normalize_dataset_name(name: str):
    return DATASET_ALIASES.get(name)

def extract_archive(archive_path, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(out_dir)
    return out_dir / "LabelANN"

def load_eli_query_results(query_dir):
    pat = re.compile(r"search_(equality|containment|overlap)_M(\d+)_efc(\d+)_elastic([0-9.]+)\.csv$")
    rows = []
    for path in sorted(Path(query_dir).rglob("*.csv")):
        m = pat.match(path.name)
        if not m:
            continue
        dataset = normalize_dataset_name(path.parent.name)
        if dataset not in DATASET_ORDER:
            continue
        sem = m.group(1)
        M = int(m.group(2))
        efc = int(m.group(3))
        elastic = float(m.group(4))
        df = pd.read_csv(path).sort_values("recall")
        for _, row in df.iterrows():
            rows.append({
                "dataset_name": dataset,
                "sem_name": sem,
                "M": M,
                "efc": efc,
                "elastic": elastic,
                "search_ef": row["ef"],
                "recall_pct": float(row["recall"]) * 100.0,
                "qps": float(row["qps"]),
                "distance_computations_per_query": float(row.get("distance_computations_per_query", float("nan"))),
            })
    if not rows:
        raise RuntimeError(f"No ELI query CSV files found under {query_dir}")
    return pd.DataFrame(rows)

def make_eli_query_plot(query, output_pdf):
    n_rows = len(SEM_ORDER)
    n_cols = len(DATASET_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18.5, 10.8), sharex=False, sharey=False)
    if n_rows == 1:
        axes = [axes]
    for r, sem in enumerate(SEM_ORDER):
        for c, dataset in enumerate(DATASET_ORDER):
            ax = axes[r][c]
            sub = query[(query["dataset_name"] == dataset) & (query["sem_name"] == sem)]
            if sub.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue

            for (M, efc, elastic), g in sub.groupby(["M", "efc", "elastic"]):
                g = g.sort_values("recall_pct")
                ax.plot(g["recall_pct"], g["qps"], color="#9e9e9e", alpha=0.6, linewidth=1.5, zorder=1)

            for elastic, st in ELASTIC_STYLES.items():
                g = sub[
                    (sub["M"] == CORE_M) &
                    (sub["efc"] == CORE_EFC) &
                    (sub["elastic"] == elastic)
                ].sort_values("recall_pct")
                if g.empty:
                    continue
                ax.plot(g["recall_pct"], g["qps"], color=st["color"], linestyle=st["linestyle"],
                        linewidth=4.0, marker=st["marker"], markersize=11,
                        markerfacecolor="none", markevery=max(1, len(g)//6), zorder=3)

            if r == 0:
                # Dataset name inside the subplot (matches plot_ivf_budget style)
                ax.text(0.05, 0.05, f"\\textbf{{{DATASET_LABEL[dataset]}}}",
                        transform=ax.transAxes, fontsize=40, ha="left", va="bottom")
            if c == 0:
                if r == 1:
                    # Method name + semantics + QPS stacked on the containment row
                    ax.set_ylabel(f"\\textbf{{{METHOD_NAME}}}\n{SEM_LABEL[sem]}\nQPS",
                                  fontsize=36, labelpad=3)
                else:
                    ax.set_ylabel("QPS", fontsize=36, labelpad=3)
                    ax.text(-0.28, 0.5, SEM_LABEL[sem], transform=ax.transAxes,
                            rotation=90, ha="center", va="center", fontsize=40, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Recall@10", fontsize=38, labelpad=3)

            ax.set_yscale("log")
            xmin = max(0, math.floor(float(sub["recall_pct"].min()) / 5) * 5)
            xmax = min(100, math.ceil(float(sub["recall_pct"].max()) / 5) * 5)
            if xmax - xmin < 10:
                xmin = max(0, xmax - 10)
            ax.set_xlim(xmin, xmax)
            ax.tick_params(axis="both", labelsize=32, pad=1, width=0.8, length=3)
            ax.grid(True, alpha=0.25, linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#444444")

    handles = [Line2D([0], [0], color="#9e9e9e", lw=2.5, alpha=0.8, label="All swept configs")]
    for elastic, st in ELASTIC_STYLES.items():
        label = fr"$M=16,\mathrm{{efc}}=200,e={elastic:g}$"
        handles.append(Line2D([0], [0], color=st["color"], lw=4.0, linestyle=st["linestyle"],
                              marker=st["marker"], markersize=11, markerfacecolor="none",
                              label=label))

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.08),
               ncol=2, frameon=False, fontsize=44, handlelength=1.2, columnspacing=0.6, handletextpad=0.2)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.085, top=0.875, wspace=0.22, hspace=0.28)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")

def make_eli_build_table(build_csv, output_tex, output_csv):
    build = pd.read_csv(build_csv)
    build = build.loc[:, ~build.columns.str.startswith("Unnamed")].copy()
    build["dataset"] = build["dataset"].astype(str).map(normalize_dataset_name)
    build = build[build["dataset"].notna()].copy()
    build = build[build["dataset"].isin(DATASET_ORDER)].copy()

    rows = []
    for ds in DATASET_ORDER:
        b = build[build["dataset"] == ds]
        if b.empty:
            continue
        suggested = b[(b["max_degree"] == CORE_M) & (b["efc"] == CORE_EFC) & (b["elastic"] == SUGGESTED_ELASTIC)]
        suggested_size = float("nan")
        suggested_time = float("nan")
        if not suggested.empty:
            suggested_size = suggested["index size(MB)"].iloc[0]
            suggested_time = suggested["indexing time(s)"].iloc[0]
        else:
            print(f"Warning: Suggested ELI setting not found for {ds}; writing NaN in suggested columns")
        rows.append({
            "Dataset": DATASET_LABEL[ds],
            "#Cfg.": int(len(b)),
            "Grid size min (MB)": b["index size(MB)"].min(),
            "Grid size med (MB)": b["index size(MB)"].median(),
            "Grid size max (MB)": b["index size(MB)"].max(),
            "Suggested size (MB)": suggested_size,
            "Grid time min (s)": b["indexing time(s)"].min(),
            "Grid time med (s)": b["indexing time(s)"].median(),
            "Grid time max (s)": b["indexing time(s)"].max(),
            "Suggested time (s)": suggested_time,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_csv, index=False)

    def f1(x):
        if pd.isna(x):
            return "N/A"
        return f"{x:.1f}"

    # Build a pastable metric-by-column table (same style as UNG write-up)
    target_datasets = ["sift1m", "LAION1M", "YTB-Audio"]
    stats = {}
    for ds in target_datasets:
        b = build[build["dataset"] == ds]
        if b.empty:
            stats[ds] = None
            continue
        core = b[(b["max_degree"] == CORE_M) & (b["efc"] == CORE_EFC)]
        stats[ds] = {
            "# Configurations": int(len(b)),
            "Grid size min (MB)": b["index size(MB)"].min(),
            "Grid size med (MB)": b["index size(MB)"].median(),
            "Grid size max (MB)": b["index size(MB)"].max(),
            "Core size range (MB)": (
                core["index size(MB)"].min() if not core.empty else float("nan"),
                core["index size(MB)"].max() if not core.empty else float("nan"),
            ),
            "Grid time min (s)": b["indexing time(s)"].min(),
            "Grid time med (s)": b["indexing time(s)"].median(),
            "Grid time max (s)": b["indexing time(s)"].max(),
            "Core time range (s)": (
                core["indexing time(s)"].min() if not core.empty else float("nan"),
                core["indexing time(s)"].max() if not core.empty else float("nan"),
            ),
        }

    def get_stat(ds, key):
        if stats.get(ds) is None:
            return "N/A"
        return stats[ds][key]

    def f_range(val):
        lo, hi = val
        if pd.isna(lo) or pd.isna(hi):
            return "N/A"
        return f"{lo:.1f}--{hi:.1f}"

    tex = """\\begin{table}[t]
\\centering
\\caption{ELI construction-side parameter audit}
\\label{tab:eli_build_param_audit}
\\begin{tabular}{@{}lccc@{}}
\\toprule
\\textbf{Metric} & \\textbf{SIFT} & \\textbf{LAION} & \\textbf{YTB-Audio} \\\\
\\midrule
"""

    tex += (
        f"\\# Configurations & {get_stat('sift1m', '# Configurations')} & "
        f"{get_stat('LAION1M', '# Configurations')} & "
        f"{get_stat('YTB-Audio', '# Configurations')} \\\\\n"
    )
    tex += "\\midrule\n"
    tex += (
        f"Grid size min (MB) & {f1(get_stat('sift1m', 'Grid size min (MB)'))} & "
        f"{f1(get_stat('LAION1M', 'Grid size min (MB)'))} & "
        f"{f1(get_stat('YTB-Audio', 'Grid size min (MB)'))} \\\\\n"
    )
    tex += (
        f"Grid size med (MB) & {f1(get_stat('sift1m', 'Grid size med (MB)'))} & "
        f"{f1(get_stat('LAION1M', 'Grid size med (MB)'))} & "
        f"{f1(get_stat('YTB-Audio', 'Grid size med (MB)'))} \\\\\n"
    )
    tex += (
        f"Grid size max (MB) & {f1(get_stat('sift1m', 'Grid size max (MB)'))} & "
        f"{f1(get_stat('LAION1M', 'Grid size max (MB)'))} & "
        f"{f1(get_stat('YTB-Audio', 'Grid size max (MB)'))} \\\\\n"
    )
    tex += (
        f"Core size range (MB) & {f_range(get_stat('sift1m', 'Core size range (MB)'))} & "
        f"{f_range(get_stat('LAION1M', 'Core size range (MB)'))} & "
        f"{f_range(get_stat('YTB-Audio', 'Core size range (MB)'))} \\\\\n"
    )
    tex += "\\midrule\n"
    tex += (
        f"Grid time min (s) & {f1(get_stat('sift1m', 'Grid time min (s)'))} & "
        f"{f1(get_stat('LAION1M', 'Grid time min (s)'))} & "
        f"{f1(get_stat('YTB-Audio', 'Grid time min (s)'))} \\\\\n"
    )
    tex += (
        f"Grid time med (s) & {f1(get_stat('sift1m', 'Grid time med (s)'))} & "
        f"{f1(get_stat('LAION1M', 'Grid time med (s)'))} & "
        f"{f1(get_stat('YTB-Audio', 'Grid time med (s)'))} \\\\\n"
    )
    tex += (
        f"Grid time max (s) & {f1(get_stat('sift1m', 'Grid time max (s)'))} & "
        f"{f1(get_stat('LAION1M', 'Grid time max (s)'))} & "
        f"{f1(get_stat('YTB-Audio', 'Grid time max (s)'))} \\\\\n"
    )
    tex += (
        f"Core time range (s) & {f_range(get_stat('sift1m', 'Core time range (s)'))} & "
        f"{f_range(get_stat('LAION1M', 'Core time range (s)'))} & "
        f"{f_range(get_stat('YTB-Audio', 'Core time range (s)'))} \\\\\n"
    )
    tex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    Path(output_tex).write_text(tex + "\n")
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-archive", default="/root/work/wow/example/plot/labelann_sweep_result/")
    ap.add_argument("--build-csv", default="/root/work/wow/example/Extended-WoW building - ELI-grid.csv")
    ap.add_argument("--workdir", default="/tmp/eli_param_audit")
    ap.add_argument("--output-pdf", default="eli_query_param_audit.pdf")
    ap.add_argument("--output-build-tex", default="eli_build_param_audit_table.tex")
    ap.add_argument("--output-build-csv", default="eli_build_param_audit_summary.csv")
    args = ap.parse_args()

    query_dir = args.query_archive
    query = load_eli_query_results(query_dir)
    make_eli_query_plot(query, args.output_pdf)
    make_eli_build_table(args.build_csv, args.output_build_tex, args.output_build_csv)

if __name__ == "__main__":
    main()
