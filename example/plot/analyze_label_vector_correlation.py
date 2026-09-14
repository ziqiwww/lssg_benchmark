#!/usr/bin/env python3
"""
Label-Vector Distance Correlation Analysis

Directly measures the correlation between label-set distance (Jaccard) and
vector distance (L2/IP) for each dataset, stratified by label-set size.

Generates a single tight grid figure: rows = size buckets + overall (5 rows),
columns = datasets.

Usage:
  python analyze_label_vector_correlation.py [--datasets SIFT1M ...]
      [--sample_size 50000] [--output_dir ./correlation_analysis]
"""

import os, argparse, multiprocessing as mp
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]   = ["Times"]
plt.rcParams["text.usetex"]  = True

# ============================================================
DATASETS = {
    "SIFT1M": {
        "name": "SIFT",
        "base_vec": "../data/vecs/sift1m/sift_base.fvecs",
        "base_labelset": "./attrdata/label/sift1m_base_n1000000_l12_zipf.txt",
        "space": "l2",
    },
    "GIST1M": {
        "name": "GIST",
        "base_vec": "../data/vecs/gist1m/gist_base.fvecs",
        "base_labelset": "./attrdata/label/gist1m_base_n1000000_l12_zipf.txt",
        "space": "l2",
    },
    "TripClick": {
        "name": "TripClick",
        "base_vec": "~/work/data/filterbenchmark/tripclick/tripclick_base.fvecs",
        "base_labelset": "~/work/data/filterbenchmark/tripclick/label_base.txt",
        "space": "l2",
    },
    "LAION1M": {
        "name": "LAION",
        "base_vec": "~/work/data/filterbenchmark/LAION1M/LAION1M_base.fvecs",
        "base_labelset": "./attrdata/label/LAION1M_base_n1000448_l30_zipf.txt",
        "space": "l2",
    },
    "YTB-Video": {
        "name": "YTB-Video",
        "base_vec": "~/work/data/filterbenchmark/ytb_video/ytb_video_base.fvecs",
        "base_labelset": "~/work/data/filterbenchmark/ytb_video/label_base.txt",
        "space": "l2",
    },
    "YTB-Audio": {
        "name": "YTB-Audio",
        "base_vec": "~/work/data/filterbenchmark/ytb_audio/ytb_audio_base.fvecs",
        "base_labelset": "~/work/data/filterbenchmark/ytb_audio/label_base.txt",
        "space": "l2",
    },
    "wikipedia-5m": {
        "name": "Wikipedia",
        "base_vec": "/root/work/data/vecs/wikipedia/wikipedia_5m_base.fvecs",
        "base_labelset": "/root/work/data/vecs/wikipedia/wikipedia_5m_base_labels.txt",
        "space": "l2",
    },
    "YFCC":{
        "name": "YFCC-1M",
        "base_vec": "~/work/data/filterbenchmark/yfcc/yfcc_base.fvecs",
        "base_labelset": "~/work/data/filterbenchmark/yfcc/label_base.txt",
        "space": "l2",
    },
    
}

N_ROWS  = 5    # overall + 4 size-quartile buckets
N_COLS  = 8    # max datasets
DPI     = 200  # publication quality
FONTSIZE_TITLE = 16
FONTSIZE_LABEL = 14
FONTSIZE_ANNOT = 11
FONTSIZE_TICK  = 10

# ============================================================
# DATA LOADING
# ============================================================

def load_fvecs(filepath: str) -> Tuple[np.ndarray, int]:
    path = os.path.expanduser(filepath)
    with open(path, "rb") as f:
        dim = int.from_bytes(f.read(4), "little", signed=True)
        f.seek(0, 2)
        n = f.tell() // (4 + dim * 4)
    data = np.fromfile(path, dtype=np.int32)
    expected = n * (1 + dim)
    if len(data) < expected:
        n = len(data) // (1 + dim)
        data = data[:n * (1 + dim)]
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        offset = i * (1 + dim)
        vectors[i] = data[offset + 1:offset + 1 + dim].view(np.float32)
    print(f"  Loaded {n} vectors dim={dim}")
    return vectors, dim

def load_labelsets(filepath: str) -> List[np.ndarray]:
    path = os.path.expanduser(filepath)
    labelsets = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            if tokens:
                labelsets.append(np.array(sorted(set(int(t) for t in tokens)), dtype=np.int32))
    print(f"  Loaded {len(labelsets)} label sets")
    return labelsets

# ============================================================
# PARALLEL PAIR COMPUTATION (memmap — zero-copy, no pickling)
# ============================================================

import tempfile, atexit

# Temp dir for memmap files (cleaned on exit)
_memmap_dir = tempfile.mkdtemp(prefix="corr_mmap_")
atexit.register(lambda: __import__("shutil").rmtree(_memmap_dir, ignore_errors=True))

# Module-level globals set by _init_worker
_w_vecs  = None   # memmap: vectors
_w_ldata = None   # memmap: flat label data
_w_loff  = None   # memmap: label offsets

def _init_worker(vec_path, vec_shape, vec_dtype,
                 ldata_path, ldata_shape, ldata_dtype,
                 loff_path, loff_shape, loff_dtype):
    """Open memmap files (zero-copy — only metadata crosses process boundary)."""
    global _w_vecs, _w_ldata, _w_loff
    _w_vecs  = np.memmap(vec_path,  dtype=vec_dtype,  mode="r", shape=vec_shape)
    _w_ldata = np.memmap(ldata_path, dtype=ldata_dtype, mode="r", shape=ldata_shape)
    _w_loff  = np.memmap(loff_path,  dtype=loff_dtype,  mode="r", shape=loff_shape)

def _compute_chunk(pairs):
    """Compute Jaccard+L2 distances for a chunk of pairs."""
    jd = np.empty(len(pairs), dtype=np.float32)
    vd = np.empty(len(pairs), dtype=np.float32)
    off = _w_loff; data = _w_ldata; vecs = _w_vecs; no = len(off)
    for k, (i, j) in enumerate(pairs):
        si, ei = off[i], off[i + 1] if i + 1 < no else len(data)
        sj, ej = off[j], off[j + 1] if j + 1 < no else len(data)
        sa, sb = data[si:ei], data[sj:ej]
        la, lb = len(sa), len(sb)
        if la == 0 and lb == 0: jd[k] = 0.0
        elif la == 0 or lb == 0: jd[k] = 1.0
        else:
            inter, ia, ib = 0, 0, 0
            while ia < la and ib < lb:
                if sa[ia] == sb[ib]: inter += 1; ia += 1; ib += 1
                elif sa[ia] < sb[ib]: ia += 1
                else: ib += 1
            jd[k] = 1.0 - inter / (la + lb - inter)
        diff = vecs[i].astype(np.float64) - vecs[j].astype(np.float64)
        vd[k] = float(np.sqrt(np.dot(diff, diff)))
    return jd, vd

def sample_and_compute(labelsets, vectors, sample_size, n_workers=None, seed=42):
    """Sample pairs stratified by size quartiles, compute distances in parallel."""
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 2)

    rng = np.random.RandomState(seed)
    n   = len(labelsets)

    # --- flatten labelsets & save as memmap files ---
    total_l = sum(len(ls) for ls in labelsets)
    ldata = np.memmap(f"{_memmap_dir}/ldata", dtype=np.int32, mode="w+", shape=(total_l,))
    loff  = np.memmap(f"{_memmap_dir}/loff",  dtype=np.int64,  mode="w+", shape=(n + 1,))
    loff[0] = 0
    for i, ls in enumerate(labelsets):
        loff[i + 1] = loff[i] + len(ls)
        ldata[loff[i]:loff[i + 1]] = ls
    ldata.flush(); loff.flush()

    # --- save vectors as memmap ---
    vec_path = f"{_memmap_dir}/vecs"
    vec_mmap = np.memmap(vec_path, dtype=vectors.dtype, mode="w+", shape=vectors.shape)
    np.copyto(vec_mmap, vectors)
    vec_mmap.flush()

    initargs = (vec_path, vectors.shape, vectors.dtype,
                f"{_memmap_dir}/ldata", ldata.shape, ldata.dtype,
                f"{_memmap_dir}/loff", loff.shape, loff.dtype)

    # --- size-based quartile binning ---
    sizes = loff[1:] - loff[:-1]
    sorted_idx = np.argsort(sizes)
    n_pb = n // 4
    bucket_idx = np.zeros(n, dtype=np.int32)
    for b in range(3):
        bucket_idx[sorted_idx[(b + 1) * n_pb:]] = b + 1

    # --- sample pairs ---
    per_bucket = max(250, sample_size // 4)
    all_pairs = []  # (i, j, bucket)
    for b in range(4):
        indices = np.where(bucket_idx == b)[0]
        if len(indices) < 2: continue
        n_samp = min(per_bucket, len(indices) * (len(indices) - 1) // 2, sample_size // 4)
        for _ in range(n_samp):
            i, j = rng.choice(indices, size=2, replace=False)
            all_pairs.append((int(i), int(j), b))

    print(f"  Sampled {len(all_pairs)} pairs, computing with {n_workers} workers...")

    # --- chunk and run ---
    cs = max(2000, len(all_pairs) // (n_workers * 8))
    chunks = [[(p[0], p[1]) for p in all_pairs[s:s + cs]]
              for s in range(0, len(all_pairs), cs)]

    with mp.Pool(n_workers, initializer=_init_worker, initargs=initargs) as pool:
        results = pool.map(_compute_chunk, chunks)

    # Per-bucket size range for annotation
    bucket_size_ranges = {}
    for b in range(4):
        mask = bucket_idx == b
        if mask.sum() > 0:
            bucket_size_ranges[b] = (int(sizes[mask].min()), int(sizes[mask].max()))

    all_jd     = np.concatenate([r[0] for r in results])
    all_vd     = np.concatenate([r[1] for r in results])
    all_bucket = np.array([p[2] for p in all_pairs], dtype=np.int32)
    return all_jd, all_vd, all_bucket, bucket_size_ranges


# ============================================================
# UNIFIED FIGURE
# ============================================================

def plot_unified_figure(all_results: Dict, dataset_names: Dict, output_path: str):
    """Tight grid: 5 rows × N columns. Each panel uses its own L2 range."""
    ds_keys = list(all_results.keys())
    n_cols  = len(ds_keys)
    n_rows  = N_ROWS
    row_labels = ["Overall", "Q1 (small)", "Q2", "Q3", "Q4 (large)"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.6 + 1.2, n_rows * 1.75 + 0.5),
        dpi=DPI, squeeze=False,
        gridspec_kw={"wspace": 0.18, "hspace": 0.10,
                     "left": 0.045, "right": 0.995, "top": 0.92, "bottom": 0.07}
    )

    for ci, ds_key in enumerate(ds_keys):
        res      = all_results[ds_key]
        jd_all   = res["jd"]
        vd_all   = res["vd"]
        buckets  = res["bucket"]
        bkt_info = res.get("bucket_sizes", {})
        sz_all   = res.get("all_sizes", None)

        # Overall size stats for row-0 annotation
        if sz_all is not None and len(sz_all) > 0:
            overall_stats = {"min": int(np.min(sz_all)),
                             "median": int(np.median(sz_all)),
                             "max": int(np.max(sz_all)),
                             "p90": int(np.percentile(sz_all, 90)),
                             "p95": int(np.percentile(sz_all, 95)),
                             "p99": int(np.percentile(sz_all, 99)),
                             "n": len(sz_all)}
        else:
            overall_stats = None

        axes[0, ci].set_title(dataset_names.get(ds_key, ds_key),
                              fontsize=FONTSIZE_TITLE, fontweight="bold", pad=4)

        # Row 0: overall — show full size stats
        _draw_panel(axes[0, ci], jd_all, vd_all, overall_stats)

        # Rows 1-4: per quartile — show size range
        for bi in range(4):
            mask = buckets == bi
            if mask.sum() >= 30:
                _draw_panel(axes[bi + 1, ci], jd_all[mask], vd_all[mask],
                            bkt_info.get(bi, None))
            else:
                axes[bi + 1, ci].set_visible(False)

        if ci == 0:
            for ri in range(n_rows):
                axes[ri, 0].set_ylabel(row_labels[ri], fontsize=FONTSIZE_LABEL,
                                       fontweight="bold", labelpad=1)

    for ri in range(n_rows):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            if not ax.get_visible(): continue
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.tick_params(labelsize=FONTSIZE_TICK, pad=2)
            if ri < n_rows - 1: ax.set_xticklabels([])

    fig.text(0.52, 0.03, "Jaccard Distance", ha="center", fontsize=FONTSIZE_LABEL + 4)
    fig.text(0.017, 0.5, "L2 Distance", va="center", rotation="vertical",
             fontsize=FONTSIZE_LABEL + 4)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"\nSaved: {output_path}")
    plt.close(fig)


def _draw_panel(ax, jd, vd, size_range):
    """Draw a panel with per-panel L2 range, trend line, ρ, and size annotation."""
    # Per-panel L2 range
    vd_min, vd_max = np.percentile(vd, [1, 99])
    m = (vd_max - vd_min) * 0.05
    vd_min -= m; vd_max += m

    ax.hist2d(jd, vd, bins=25, cmap="Blues", cmin=1, range=[[0, 1], [vd_min, vd_max]])

    # Mean trend
    bins = np.linspace(0, 1, 16)
    centers = (bins[:-1] + bins[1:]) / 2
    arr = np.array([vd[(jd >= lo) & (jd < hi)].mean() if (vd[(jd >= lo) & (jd < hi)]).size > 0 else np.nan
                    for lo, hi in zip(bins[:-1], bins[1:])])
    valid = ~np.isnan(arr)
    if valid.sum() >= 2:
        ax.plot(centers[valid], arr[valid], color="#c0392b", linewidth=1.5, alpha=0.9)

    # Spearman ρ (top-right)
    if len(jd) >= 30:
        rho, pval = stats.spearmanr(jd, vd)
        stars = "" if pval > 0.05 else "*" if pval > 0.01 else "**" if pval > 0.001 else "***"
        ax.text(0.97, 0.97, f"$\\rho={rho:.3f}{stars}$",
                transform=ax.transAxes, fontsize=FONTSIZE_ANNOT, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0.3))

    # Label-set size annotation (center-bottom)
    if size_range is not None:
        small_ano_size=16
        if isinstance(size_range, dict):
            # Overall row: two-line summary
            s = (f"$|L|$: min={size_range['min']},  med={size_range['median']},  max={size_range['max']}\n"
                 f"P90={size_range['p90']},  P95={size_range['p95']},  P99={size_range['p99']}")
            ana_size=small_ano_size
        else:
            lo, hi = size_range
            s = f"$|L| \\in [{lo}, {hi}]$" if lo != hi else f"$|L| = {lo}$"
            ana_size=FONTSIZE_ANNOT
        ax.text(0.5, 0.01, s, transform=ax.transAxes, fontsize=ana_size,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, linewidth=0.3))

    ax.set_xlim(0, 1); ax.set_ylim(vd_min, vd_max)
    ax.xaxis.set_major_locator(FixedLocator([0, 0.5, 1]))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{v:.1f}'.rstrip('0').rstrip('.') if v != 0 else '0'))


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary(all_results, all_sizes, dataset_names):
    print("\n" + "=" * 120)
    print("LABEL-VECTOR DISTANCE CORRELATION — SUMMARY")
    print("=" * 120)
    hdr = f"{'Dataset':<14} {'Med|Sz':>6} {'P90|Sz':>6} {'ρ(overall)':>11} {'ρ(Q1)':>9} {'ρ(Q2)':>9} {'ρ(Q3)':>9} {'ρ(Q4)':>9}  {'n_pairs':>8}"
    print(hdr); print("-" * 120)
    for ds in all_results:
        name = dataset_names.get(ds, ds)
        sz = all_sizes[ds]
        r = all_results[ds]
        jd, vd, buckets = r["jd"], r["vd"], r["bucket"]
        rho_all = stats.spearmanr(jd, vd)[0]
        rhos = []
        for b in range(4):
            m = buckets == b
            rhos.append(stats.spearmanr(jd[m], vd[m])[0] if m.sum() >= 30 else float("nan"))
        print(f"{name:<14} {np.median(sz):6.0f} {np.percentile(sz,90):6.0f} "
              f"{rho_all:10.4f} {rhos[0]:9.4f} {rhos[1]:9.4f} {rhos[2]:9.4f} {rhos[3]:9.4f}  {len(jd):8,}")
    print("=" * 120)
    print("Q1=smallest 25% label-set sizes, Q4=largest 25%.  * p<0.05, ** p<0.01, *** p<0.001\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--sample_size", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="./correlation_analysis")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.datasets: selected = {k: DATASETS[k] for k in args.datasets if k in DATASETS}
    else: selected = DATASETS

    print(f"Datasets: {list(selected.keys())}  |  sample_size={args.sample_size}  |  workers={args.workers or mp.cpu_count()-2}")
    all_results = {}
    all_sizes   = {}
    dataset_names = {}

    for db_key, cfg in selected.items():
        print(f"\n{'='*50}\n{cfg['name']} ({db_key})\n{'='*50}")
        dataset_names[db_key] = cfg["name"]

        vectors, dim = load_fvecs(cfg["base_vec"])
        labelsets = load_labelsets(cfg["base_labelset"])
        n = min(len(labelsets), len(vectors))
        labelsets, vectors = labelsets[:n], vectors[:n]

        sizes = np.array([len(ls) for ls in labelsets])
        all_sizes[db_key] = sizes
        print(f"  Sizes: med={np.median(sizes):.0f}  P90={np.percentile(sizes,90):.0f}  max={np.max(sizes)}")

        jd, vd, buckets, bkt_sizes = sample_and_compute(
            labelsets, vectors, args.sample_size, args.workers, args.seed)
        all_results[db_key] = {"jd": jd, "vd": vd, "bucket": buckets,
                               "bucket_sizes": bkt_sizes,
                               "all_sizes": sizes}

        rho = stats.spearmanr(jd, vd)[0]
        print(f"  Spearman ρ(overall) = {rho:.4f}")

    # Unified figure
    if all_results:
        plot_unified_figure(all_results, dataset_names,
                            os.path.join(args.output_dir, "label_vector_correlation.pdf"))
        print_summary(all_results, all_sizes, dataset_names)


if __name__ == "__main__":
    main()
