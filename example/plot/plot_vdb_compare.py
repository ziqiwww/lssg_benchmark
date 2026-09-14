#!/usr/bin/env python3
"""System-level comparison: LSSG vs FAISS/VSAG/Milvus/PGVector (recall vs QPS).
Layout: 3 rows × N cols (Equality, Containment, Overlap × datasets).
Style: matches plot_label_cmp_recall.py exactly.
"""

import ctypes, os, sys
from pathlib import Path

def _preload_libstdcpp():
    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if conda_prefix:
        lib = Path(conda_prefix) / "lib" / "libstdc++.so.6"
        if lib.exists():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
_preload_libstdcpp()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]   = ["Times"]
plt.rcParams["text.usetex"]  = True

VDB_DIR   = "vdb_compare"
PLOT_DIR  = "vdb_compare"

# ============================================================
# Configure datasets here: edit DATASETS to include/exclude.
# DS_INFO maps each dataset key to:
#   (display_label, lssg_dir, faiss_vsag_filename, milvus_pgvector_dir)
# ============================================================
# DATASETS = ["SIFT1M", "TripClick", "YTB-Video", "YFCC"]
DATASETS = ["GIST1M", "LAION1M", "YTB-Audio", "Wikipedia-5m"]

DS_INFO = {
    "GIST1M":       ("GIST",       "gist",       "GIST1M",       "gist1m"),
    "LAION1M":      ("LAION",      "laion",      "LAION1M",      "LAION1M"),
    "YTB-Audio":    ("YTB-Audio",  "ytb_audio",  "YTB-Audio",    "ytb_audio"),
    "Wikipedia-5m": ("Wikipedia",  "wikipedia",  "Wikipedia-5m", "wikipedia-5m"),
    "SIFT1M":       ("SIFT",       "sift",       "SIFT1M",       "sift1m"),
    "TripClick":    ("TripClick",  "tripclick",  "TripClick",    "tripclick"),
    "YTB-Video":    ("YTB-Video",  "ytb_video",  "YTB-Video",    "ytb_video"),
    "YFCC":         ("YFCC",       "yfcc",       "YFCC",         "yfcc"),
}

FILTERS   = ["equality", "containment", "overlap"]
FL_LABEL  = {"equality": "Equality", "containment": "Containment", "overlap": "Overlap"}

COLOR_MAP = {
    "LSSG-IVF":      "#b60033",
    "LSSG-MinHash":  "#ff7f0e",
    "FAISS-HNSW":    "#08519c",
    "FAISS-IVF":     "#6baed6",
    "VSAG-HNSW":     "#54278f",
    "VSAG-IVF":      "#9e9ac8",
    "Milvus-HNSW":   "#006d2c",
    "Milvus-IVF":    "#74c476",
    "PGVector-HNSW": "#ae017e",
    "PGVector-IVF":  "#f768a1",
}
MARKER_MAP = {
    "LSSG-IVF":      "o",  "LSSG-MinHash":  "o",
    "FAISS-HNSW":    "^",  "FAISS-IVF":     "^",
    "VSAG-HNSW":     "D",  "VSAG-IVF":      "D",
    "Milvus-HNSW":   "s",  "Milvus-IVF":    "s",
    "PGVector-HNSW": "v",  "PGVector-IVF":  "v",
}

METHOD_ORDER = ["LSSG-IVF", "LSSG-MinHash",
                "FAISS-IVF", "FAISS-HNSW",
                "VSAG-IVF", "VSAG-HNSW",
                "Milvus-IVF", "Milvus-HNSW",
                "PGVector-IVF", "PGVector-HNSW"]
VDB_METHODS   = {m for m in METHOD_ORDER if m not in ("LSSG-IVF", "LSSG-MinHash")}


def load_lssg(ds, flt, variant):
    """Load LSSG result CSV (no header: ef,recall,qps,dist_comp,hops)."""
    _, ds_dir, _, _ = DS_INFO[ds]
    suffix = "" if variant == "LSSG-IVF" else "_minhash"
    path = os.path.join(ds_dir, "label", f"poindex{suffix}_{flt}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=None, names=["ef", "recall", "qps", "dist_comp", "hops"],
                     on_bad_lines="skip")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["recall", "qps"])


def load_vdb(name, ds, flt):
    """Load a vdb method result CSV."""
    _, _, faiss_vsag_name, mvpg_dir = DS_INFO[ds]
    if name == "FAISS-HNSW":
        path = os.path.join(VDB_DIR, "faiss", f"faiss_{faiss_vsag_name}_hnsw_{flt}.csv")
    elif name == "FAISS-IVF":
        path = os.path.join(VDB_DIR, "faiss", f"faiss_{faiss_vsag_name}_ivf_{flt}.csv")
    elif name == "VSAG-HNSW":
        path = os.path.join(VDB_DIR, "vsag", f"vsag_{faiss_vsag_name}_hgraph_{flt}.csv")
    elif name == "VSAG-IVF":
        path = os.path.join(VDB_DIR, "vsag", f"vsag_{faiss_vsag_name}_ivf_{flt}.csv")
    elif name == "Milvus-HNSW":
        path = os.path.join(VDB_DIR, "milvus", f"{mvpg_dir}_{flt}", f"milvus-hnsw-{flt}.csv")
    elif name == "Milvus-IVF":
        path = os.path.join(VDB_DIR, "milvus", f"{mvpg_dir}_{flt}", f"milvus-ivf-{flt}.csv")
    elif name == "PGVector-HNSW":
        path = os.path.join(VDB_DIR, "pgvetor", f"{mvpg_dir}_{flt}", f"pgvector-hnsw-{flt}.csv")
    elif name == "PGVector-IVF":
        path = os.path.join(VDB_DIR, "pgvetor", f"{mvpg_dir}_{flt}", f"pgvector-ivf-{flt}.csv")
    else:
        return None
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace('\r', '')
    df = df.rename(columns={"efSearch": "ef", "recall@10": "recall"})
    for c in ["ef", "recall", "qps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["recall"] > 0]


def main():
    n_rows, n_cols = 3, len(DATASETS)  # rows=filters, cols=datasets
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.0, n_rows * 2.8), dpi=150)
    fig.subplots_adjust(hspace=0.18, wspace=0.16)

    seen_methods = set()

    for row_flt, flt in enumerate(FILTERS):
        for col_ds, ds in enumerate(DATASETS):
            ax = axes[row_flt, col_ds]

            for name in METHOD_ORDER:
                if name in VDB_METHODS:
                    df = load_vdb(name, ds, flt)
                else:
                    df = load_lssg(ds, flt, name)
                if df is None or df.empty:
                    continue
                df = df.dropna(subset=["recall", "qps"])
                df = df.sort_values("recall")

                seen_methods.add(name)
                is_lssg = name in ("LSSG-IVF", "LSSG-MinHash")
                is_ivf = name.endswith("-IVF")
                is_filled = (not is_ivf) and (name != "LSSG-IVF")
                ax.plot(df["recall"], df["qps"],
                        color=COLOR_MAP[name], marker=MARKER_MAP[name],
                        markersize=8,
                        markerfacecolor=COLOR_MAP[name] if is_filled else "none",
                        linestyle="--" if is_ivf else "-",
                        linewidth=2.5 if is_lssg else 1.5,
                        label=name, zorder=5 if is_lssg else 2)

            ax.set_xlim(0.5, 1.01)
            ax.set_yscale("log")
            ax.set_xticks([0.6, 0.8, 1.0])
            ax.set_xticklabels(["0.6", "0.8", "1.0"])
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.tick_params(axis="both", which="major", labelsize=18)
            ax.tick_params(axis="x", which="major", labelsize=20)

            if row_flt == 0:
                ax.set_title(DS_INFO[ds][0], fontsize=28)
            if col_ds == 0:
                ax.set_ylabel(f"{FL_LABEL[flt]}\nQPS", fontsize=26, labelpad=0)
            if row_flt == 2:
                ax.set_xlabel("Recall@10", fontsize=26)

    # Legend
    handles = []
    for name in METHOD_ORDER:
        if name in seen_methods:
            is_ivf = name.endswith("-IVF")
            is_filled = (not is_ivf) and (name != "LSSG-IVF")
            handles.append(plt.Line2D([0], [0], color=COLOR_MAP[name],
                                       marker=MARKER_MAP[name], markersize=8,
                                       markerfacecolor=COLOR_MAP[name] if is_filled else "none",
                                       linestyle="--" if is_ivf else "-",
                                       linewidth=2, label=name))
    fig.legend(handles=handles, loc="upper center", ncol=5,
               fontsize=24, frameon=False, bbox_to_anchor=(0.48, 1.06),
               handletextpad=0.2, handlelength=1.2, columnspacing=0.5)

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, "vdb_compare.pdf"), dpi=300, bbox_inches="tight")
    print(f"Saved {PLOT_DIR}/vdb_compare.pdf")


if __name__ == "__main__":
    main()
