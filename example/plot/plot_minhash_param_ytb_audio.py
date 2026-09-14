import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

DATA_DIR = "/root/work/lssg/example/plot/minhash_param/ytb_audio"
INDEXING_FILE = os.path.join(DATA_DIR, "indexing_information.txt")

# Rotation flag: if True, transpose the layout (rows become columns)
rotate = False


def load_csv_values(file_path):
    """Load CSV values with flexible header handling. Returns (ef, recall, qps) arrays."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        header_idx = -1
        for i, line in enumerate(lines):
            if "ef" in line and "recall" in line and "qps" in line:
                header_idx = i
                break
        data_lines = lines[header_idx + 1:] if header_idx >= 0 else lines
        data_text = "\n".join([ln.strip() for ln in data_lines if ln.strip()])
        data = np.genfromtxt(data_text.split("\n"), delimiter=",", dtype=float)
        if data.size == 0:
            return None
        if data.ndim == 1:
            data = data.reshape(1, -1)
        ef = data[:, 0]
        recall = data[:, 1]
        qps = data[:, 2]
        return ef, recall, qps
    except Exception:
        return None


def collect_heatmap_for_scenario(scenario, ef_target):
    """Collect recall and QPS heatmap matrices for a given scenario at ef=ef_target.
    Returns (tau_values, beta_values, recall_grid, qps_grid)."""
    # Discover files
    files = [fn for fn in os.listdir(DATA_DIR) if fn.startswith(f"poindex_{scenario}_") and fn.endswith(".csv")]
    tau_set = set()
    beta_set = set()
    entries = {}
    for fn in files:
        base = os.path.splitext(fn)[0]
        parts = base.split("_")
        # Expect: poindex, scenario, tau, beta
        if len(parts) < 4:
            continue
        try:
            tau = int(parts[2])
            beta = int(parts[3])
        except ValueError:
            continue
        tau_set.add(tau)
        beta_set.add(beta)
        fp = os.path.join(DATA_DIR, fn)
        vals = load_csv_values(fp)
        if vals is None:
            continue
        ef, recall, qps = vals
        # pick ef == ef_target or nearest
        idx = np.argmin(np.abs(ef - ef_target))
        entries[(tau, beta)] = (recall[idx], qps[idx])
    if not tau_set or not beta_set:
        return [], [], None, None
    tau_values = sorted(tau_set)
    beta_values = sorted(beta_set)
    recall_grid = np.full((len(tau_values), len(beta_values)), np.nan)
    qps_grid = np.full_like(recall_grid, np.nan, dtype=float)
    for i, tau in enumerate(tau_values):
        for j, beta in enumerate(beta_values):
            if (tau, beta) in entries:
                r, q = entries[(tau, beta)]
                recall_grid[i, j] = r
                qps_grid[i, j] = q
    return tau_values, beta_values, recall_grid, qps_grid


def parse_indexing_matrix():
    """Parse indexing_information.txt structured matrix into tau, beta, size, time grids."""
    if not os.path.exists(INDEXING_FILE):
        return [], [], None, None
    with open(INDEXING_FILE, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    # First line: header with beta values (tab-separated)
    header = lines[0].split("\t")
    beta_values = [int(x) for x in header[1:]]
    tau_values = []
    time_grid = []
    size_grid = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        try:
            tau = int(parts[0])
        except ValueError:
            continue
        tau_values.append(tau)
        time_row = []
        size_row = []
        for cell in parts[1:]:
            cell = cell.strip()
            if cell == "-":
                time_row.append(np.nan)
                size_row.append(np.nan)
                continue
            # cell: time,index_size,minhash_size
            try:
                t_str, s_str, _ = cell.split(",")
                time_row.append(float(t_str))
                size_row.append(float(s_str))
            except Exception:
                time_row.append(np.nan)
                size_row.append(np.nan)
        time_grid.append(time_row)
        size_grid.append(size_row)
    time_grid = np.array(time_grid, dtype=float)
    size_grid = np.array(size_grid, dtype=float)
    return tau_values, beta_values, size_grid, time_grid


def plot_minhash_param_ytb_audio(ef_target=1000, output_file="plot_minhash_param_ytb_audio.pdf"):
    # Collect heatmaps for scenarios
    tau_c, beta_c, recall_c, qps_c = collect_heatmap_for_scenario("containment", ef_target)
    tau_o, beta_o, recall_o, qps_o = collect_heatmap_for_scenario("overlap", ef_target)

    # Parse indexing matrices
    tau_i, beta_i, size_grid, time_grid = parse_indexing_matrix()

    # Figure layout: 3x2 by default, 2x3 when rotated
    # Rows: containment, overlap, indexing (or cols when rotated)
    # Cols: Recall, QPS (or rows when rotated)
    if rotate:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
        fig.subplots_adjust(hspace=0.25, wspace=0.01, top=0.95)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.subplots_adjust(hspace=0.30, wspace=0.01, top=0.95)

    # Unified colormap (use first column palette across all)
    base_cmap = plt.get_cmap('viridis').copy()
    base_cmap.set_bad(color='#e0e0e0')  # color for NaN cells

    def format_compact(val, kind):
        """Format values compactly depending on kind: 'recall', 'qps', 'size', 'time'."""
        if kind == 'recall':
            # Format as .xxxx (4 decimals without leading zero)
            return f"{val:.4f}"[1:]
        if kind in ('size', 'time'):
            # Use full integer, no suffix and no decimals (truncate)
            return f"{int(val)}"
        # For qps, keep compact suffix
        absval = abs(val)
        if absval >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if absval >= 1_000:
            return f"{val/1_000:.1f}k"
        return f"{val:.2f}"

    def format_colorbar_label(val, kind):
        """Format colorbar min/max labels: full integers for size/time, compact otherwise."""
        if kind == 'recall':
            return f"{val:.2f}"
        if kind in ('size', 'time'):
            return f"{int(val)}"
        absval = abs(val)
        if absval >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if absval >= 1_000:
            return f"{val/1_000:.1f}k"
        return f"{int(val)}"

    def annotate_cells(ax, im, grid, kind):
        """Annotate each cell with its value, choosing contrasting text color."""
        if grid is None:
            return
        norm = im.norm if hasattr(im, 'norm') else Normalize()
        cmap = im.cmap if hasattr(im, 'cmap') else base_cmap
        nrows, ncols = np.shape(grid)
        for i in range(nrows):
            for j in range(ncols):
                val = grid[i, j]
                if np.isnan(val):
                    text_str = r"$\mathrm{NaN}$"
                    # Use dark text on light NaN cell color
                    color = 'black'
                else:
                    text_str = format_compact(val, kind)
                    rgba = cmap(norm(val))
                    # Perceived luminance
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    color = 'white' if lum < 0.5 else 'black'
                ax.text(j, i, text_str, ha='center', va='center', fontsize=24, color=color)
    
    tick_size = 26
    # Helper to plot heatmap with concise labels
    def plot_heat(ax, grid, tau_vals, beta_vals, title, row_idx, col_idx, kind):
        
        if grid is None or tau_vals is None or beta_vals is None or len(tau_vals) == 0 or len(beta_vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=tick_size)
            ax.axis("off")
            return None
        # Ensure masked NaNs render with set_bad color
        grid_masked = np.ma.masked_invalid(np.array(grid, dtype=float))
        # Use plasma colormap for indexing info (row 2 when not rotated, or col 2 when rotated)
        is_indexing = (row_idx == 2 and not rotate) or (col_idx == 2 and rotate)
        cmap = plt.get_cmap('plasma').copy() if is_indexing else base_cmap
        cmap.set_bad(color='#e0e0e0')
        im = ax.imshow(grid_masked, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(beta_vals)))
        ax.set_xticklabels([str(b) for b in beta_vals], fontsize=tick_size)
        ax.set_yticks(range(len(tau_vals)))
        ax.set_yticklabels([str(t) for t in tau_vals], fontsize=tick_size)
        # Concise axis labels: x only on last row, y only on first column
        # Adjust based on rotation
        if rotate:
            # When rotated: x-label on last row (row 1), y-label on first column (col 0)
            if row_idx == 1:
                ax.set_xlabel(r"$\beta$", fontsize=28)
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel(r"$\tau$", fontsize=28)
            else:
                ax.set_ylabel("")
        else:
            # Original layout
            if row_idx == 2:
                ax.set_xlabel(r"$\beta$", fontsize=28)
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel(r"$\tau$", fontsize=28)
            else:
                ax.set_ylabel("")  # Hide ylabel but keep ytick labels
        ax.set_title(title, fontsize=30, pad=8)
        annotate_cells(ax, im, np.array(grid, dtype=float), kind)
        return im

    # Create plots based on rotation
    if rotate:
        # Rotated: 2 rows (Recall/QPS) x 3 cols (containment/overlap/indexing)
        # Row 0: Recall metrics, Row 1: QPS/performance metrics
        # Col 0: Containment, Col 1: Overlap, Col 2: Indexing
        
        im00 = plot_heat(axes[0, 0], recall_c, tau_c, beta_c, r"Containment: Recall@10", row_idx=0, col_idx=0, kind='recall')
        if im00:
            cbar00 = fig.colorbar(im00, ax=axes[0, 0], pad=0.02)
            cbar00.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(recall_c, dtype=float)))
                vmax = float(np.nanmax(np.array(recall_c, dtype=float)))
                cbar00.ax.text(0.5, 1.0, f"{vmax:.2f}", transform=cbar00.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar00.ax.text(0.5, -0.02, f"{vmin:.2f}", transform=cbar00.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        im01 = plot_heat(axes[0, 1], recall_o, tau_o, beta_o, r"Overlap: Recall@10", row_idx=0, col_idx=1, kind='recall')
        if im01:
            cbar01 = fig.colorbar(im01, ax=axes[0, 1], pad=0.02)
            cbar01.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(recall_o, dtype=float)))
                vmax = float(np.nanmax(np.array(recall_o, dtype=float)))
                cbar01.ax.text(0.5, 1.0, f"{vmax:.2f}", transform=cbar01.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar01.ax.text(0.5, -0.02, f"{vmin:.2f}", transform=cbar01.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass
        
        im02 = plot_heat(axes[0, 2], size_grid, tau_i, beta_i, r"Index Size (MB)", row_idx=0, col_idx=2, kind='size')
        if im02:
            cbar02 = fig.colorbar(im02, ax=axes[0, 2], pad=0.02)
            cbar02.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(size_grid, dtype=float)))
                vmax = float(np.nanmax(np.array(size_grid, dtype=float)))
                cbar02.ax.text(0.5, 1.0, format_colorbar_label(vmax, 'size'), transform=cbar02.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar02.ax.text(0.5, -0.02, format_colorbar_label(vmin, 'size'), transform=cbar02.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        im10 = plot_heat(axes[1, 0], qps_c, tau_c, beta_c, r"Containment: QPS", row_idx=1, col_idx=0, kind='qps')
        if im10:
            cbar10 = fig.colorbar(im10, ax=axes[1, 0], pad=0.02)
            cbar10.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(qps_c, dtype=float)))
                vmax = float(np.nanmax(np.array(qps_c, dtype=float)))
                cbar10.ax.text(0.5, 1.0, format_colorbar_label(vmax, 'qps'), transform=cbar10.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar10.ax.text(0.5, -0.02, format_colorbar_label(vmin, 'qps'), transform=cbar10.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        im11 = plot_heat(axes[1, 1], qps_o, tau_o, beta_o, r"Overlap: QPS", row_idx=1, col_idx=1, kind='qps')
        if im11:
            cbar11 = fig.colorbar(im11, ax=axes[1, 1], pad=0.02)
            cbar11.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(qps_o, dtype=float)))
                vmax = float(np.nanmax(np.array(qps_o, dtype=float)))
                cbar11.ax.text(0.5, 1.0, format_colorbar_label(vmax, 'qps'), transform=cbar11.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar11.ax.text(0.5, -0.02, format_colorbar_label(vmin, 'qps'), transform=cbar11.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        im12 = plot_heat(axes[1, 2], time_grid, tau_i, beta_i, r"Indexing Time (s)", row_idx=1, col_idx=2, kind='time')
        if im12:
            cbar12 = fig.colorbar(im12, ax=axes[1, 2], pad=0.02)
            cbar12.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(time_grid, dtype=float)))
                vmax = float(np.nanmax(np.array(time_grid, dtype=float)))
                cbar12.ax.text(0.5, 1.0, format_colorbar_label(vmax, 'time'), transform=cbar12.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar12.ax.text(0.5, -0.02, format_colorbar_label(vmin, 'time'), transform=cbar12.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass
    
    else:
        # Original: Row 0: containment recall (col 0) and QPS (col 1)
        im00 = plot_heat(axes[0, 0], recall_c, tau_c, beta_c, r"Containment: Recall@10", row_idx=0, col_idx=0, kind='recall')
        if im00:
            cbar00 = fig.colorbar(im00, ax=axes[0, 0], pad=0.02)
            cbar00.set_ticks([])
            # Position min/max labels outside the colorbar
            try:
                vmin = float(np.nanmin(np.array(recall_c, dtype=float)))
                vmax = float(np.nanmax(np.array(recall_c, dtype=float)))
                cbar00.ax.text(0.5, 1.0, f"{vmax:.2f}", transform=cbar00.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar00.ax.text(0.5, -0.02, f"{vmin:.2f}", transform=cbar00.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        im01 = plot_heat(axes[0, 1], qps_c, tau_c, beta_c, r"Containment: QPS", row_idx=0, col_idx=1, kind='qps')
        if im01:
            cbar01 = fig.colorbar(im01, ax=axes[0, 1], pad=0.02)
            cbar01.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(qps_c, dtype=float)))
                vmax = float(np.nanmax(np.array(qps_c, dtype=float)))
                cbar01.ax.text(0.5, 1.0, format_colorbar_label(vmax, 'qps'), transform=cbar01.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar01.ax.text(0.5, -0.02, format_colorbar_label(vmin, 'qps'), transform=cbar01.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        # Row 1: overlap recall (col 0) and QPS (col 1)
        im10 = plot_heat(axes[1, 0], recall_o, tau_o, beta_o, r"Overlap: Recall@10", row_idx=1, col_idx=0, kind='recall')
        if im10:
            cbar10 = fig.colorbar(im10, ax=axes[1, 0], pad=0.02)
            cbar10.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(recall_o, dtype=float)))
                vmax = float(np.nanmax(np.array(recall_o, dtype=float)))
                cbar10.ax.text(0.5, 1.0, f"{vmax:.2f}", transform=cbar10.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar10.ax.text(0.5, -0.02, f"{vmin:.2f}", transform=cbar10.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        im11 = plot_heat(axes[1, 1], qps_o, tau_o, beta_o, r"Overlap: QPS", row_idx=1, col_idx=1, kind='qps')
        if im11:
            cbar11 = fig.colorbar(im11, ax=axes[1, 1], pad=0.02)
            cbar11.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(qps_o, dtype=float)))
                vmax = float(np.nanmax(np.array(qps_o, dtype=float)))
                cbar11.ax.text(0.5, 1.0, format_colorbar_label(vmax, 'qps'), transform=cbar11.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar11.ax.text(0.5, -0.02, format_colorbar_label(vmin, 'qps'), transform=cbar11.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        # Row 2: indexing (size in col 0, time in col 1)
        im20 = plot_heat(axes[2, 0], size_grid, tau_i, beta_i, r"Index Size (MB)", row_idx=2, col_idx=0, kind='size')
        if im20:
            cbar20 = fig.colorbar(im20, ax=axes[2, 0], pad=0.02)
            cbar20.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(size_grid, dtype=float)))
                vmax = float(np.nanmax(np.array(size_grid, dtype=float)))
                cbar20.ax.text(0.5, 1.0, format_colorbar_label(vmax, 'size'), transform=cbar20.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar20.ax.text(0.5, -0.02, format_colorbar_label(vmin, 'size'), transform=cbar20.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

        im21 = plot_heat(axes[2, 1], time_grid, tau_i, beta_i, r"Indexing Time (s)", row_idx=2, col_idx=1, kind='time')
        if im21:
            cbar21 = fig.colorbar(im21, ax=axes[2, 1], pad=0.02)
            cbar21.set_ticks([])
            try:
                vmin = float(np.nanmin(np.array(time_grid, dtype=float)))
                vmax = float(np.nanmax(np.array(time_grid, dtype=float)))
                cbar21.ax.text(0.5, 1.0, format_colorbar_label(vmax, 'time'), transform=cbar21.ax.transAxes,
                              ha='center', va='bottom', fontsize=tick_size)
                cbar21.ax.text(0.5, -0.02, format_colorbar_label(vmin, 'time'), transform=cbar21.ax.transAxes,
                              ha='center', va='top', fontsize=tick_size)
            except Exception:
                pass

    # Add a small super title
    # fig.suptitle(r"YTB-Audio MinHash Params: $ef=1000$", fontsize=26, y=0.995)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Figure saved as {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MinHash params heatmaps for YTB-Audio")
    parser.add_argument("--ef", type=int, default=1000, help="ef value to slice recall/QPS (default 1000)")
    parser.add_argument("--out", default="plot_minhash_param_ytb_audio.pdf", help="Output PDF filename")
    parser.add_argument("--rotate", action="store_true", help="Transpose the subplot layout (rows become columns)")
    args = parser.parse_args()
    
    rotate = args.rotate
    plot_minhash_param_ytb_audio(ef_target=args.ef, output_file=args.out)
