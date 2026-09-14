import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# Datasets
datasets = ["laion", "ytb_audio"]

# Budget values for φ
budgets = [50, 500, 5000, 50000, 500000]

# Scenarios
scenarios = ["containment", "overlap"]

# Rotation flag: if True, transpose the layout (rows become columns)
rotate = False

# Color palette for budget values (single gradient from light to dark)
# budget_colors = {
#     50: "#e5c185",      # Lighter
#     500: "#cc7a3d",
#     5000: "#c7522a",
#     50000: "#963e20",
#     500000: "#642915"   # Darker
# }

# budget_colors = {
#     50: "#9eafbf",      # Lighter
#     500: "#9eafbf",
#     5000: "#7f96aa",
#     50000: "#607e96",
#     500000: "#406682"   # Darker
# }


budget_colors = {
    50: "#f27b50",      # Lighter
    500: "#b6042a",
    5000: "#730220",
    50000: "#88a2c4", 
    500000: "#5e62a9"   # Darker
}

# Markers for budgets
budget_markers = {
    50: "o",       # Circle
    500: "s",      # Square
    5000: "^",     # Triangle
    50000: "D",    # Diamond
    500000: "v"    # Inverted triangle
}

# Dataset display names
dataset_display_map = {
    "laion": "LAION",
    "ytb_audio": "YTB-Audio",
    "ytb-audio": "YTB-Audio"
}

def get_dataset_display_name(dataset):
    """Get display name for a dataset"""
    return dataset_display_map.get(dataset, dataset)

def load_query_data(dataset, scenario, budget, metric="qps"):
    """
    Load query performance data from ivf_budget dataset directory
    Format: ef,recall,qps,dist_comp,hops (after debug headers, or no header at all)
    Files are named: poindex_{scenario}_{budget}.csv
    
    metric: "qps" (default) or "cmps" (dist_comp)
    """
    if dataset.lower() == "ytb_audio" or dataset == "ytb-audio":
        file_path = f"/root/work/lssg/example/plot/ivf_budget/ytb_audio/poindex_{scenario}_{budget}.csv"
    else:
        file_path = f"/root/work/lssg/example/plot/ivf_budget/LAION/poindex_{scenario}_{budget}.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        # Skip debug lines, find the header line
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (should contain "ef")
        header_idx = -1
        for i, line in enumerate(lines):
            if 'ef' in line and 'recall' in line and 'qps' in line:
                header_idx = i
                break
        
        # If no header found, assume data starts from first numeric line
        if header_idx == -1:
            header_idx = -1  # No header, data starts at line 0
        
        # Parse data after header
        if header_idx >= 0:
            data_lines = lines[header_idx + 1:]
        else:
            data_lines = lines
        
        data_text = ''.join(data_lines)
        
        # Parse CSV, skip empty lines
        data_text = '\n'.join([line for line in data_text.split('\n') if line.strip()])
        data = np.genfromtxt(data_text.split('\n'), delimiter=',', skip_header=0)
        
        if data.size == 0:
            print(f"No data found in {file_path}")
            return None
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Columns: ef, recall, qps, dist_comp, hops
        # Index: 0,  1,      2,   3,         4
        recall = data[:, 1]
        
        if metric.lower() == "qps":
            perf = data[:, 2]
        elif metric.lower() == "cmps":
            perf = data[:, 3]
        else:
            perf = data[:, 2]
        
        # Sort by recall
        sort_idx = np.argsort(recall)
        recall = recall[sort_idx]
        perf = perf[sort_idx]
        
        return recall, perf
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_indexing_data():
    """
    Load indexing information from indexing_information.txt
    Returns dict: {dataset: {budget: (time_s, size_mb)}}
    """
    file_path = "/root/work/lssg/example/plot/ivf_budget/indexing_information.txt"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    data = {}
    current_dataset = None
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a dataset line
                if line.lower() in ["laion", "ytb-audio", "ytb_audio"]:
                    current_dataset = line.lower()
                    if current_dataset == "ytb-audio":
                        current_dataset = "ytb_audio"
                    data[current_dataset] = {}
                    continue
                
                # Parse budget line
                if current_dataset and ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            budget = int(parts[0])
                            time_str = parts[1].strip().rstrip('s')
                            time_s = float(time_str)
                            size_str = parts[2].strip().rstrip('M').replace(',', '')
                            size_mb = float(size_str)
                            
                            data[current_dataset][budget] = (time_s, size_mb)
                        except (ValueError, IndexError):
                            pass
    
    except Exception as e:
        print(f"Error loading indexing data: {e}")
        return None
    
    return data

def plot_query_performance_dataset(ax, dataset, scenario, metric="qps"):
    """Plot query performance (QPS vs Recall or Cmps vs Recall) for one dataset+scenario"""
    
    for budget in budgets:
        result = load_query_data(dataset, scenario, budget, metric=metric)
        if result is None:
            continue
        
        recall, perf = result
        
        # Plot line with marker (matching plot_labeldiver.py styling)
        ax.plot(recall, perf, 
                color=budget_colors[budget],
                marker=budget_markers[budget],
                linestyle='-',
                linewidth=2.5,
                markersize=12,
                markerfacecolor="none",
                label=f"$\\phi$={budget:,}")
    
    ax.set_xlabel("Recall", fontsize=26, labelpad=0)
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 1.0)

def plot_indexing_bars(ax, indexing_data, metric="size"):
    """
    Plot indexing metrics as grouped bars with budget colors
    metric: "size" or "time"
    
    Layout: Two groups (one for LAION, one for YTB-Audio),
            each group has 5 bars for budgets [50, 500, 5000, 50000, 500000]
            Bars colored by budget phi value
    """
    
    # Group width and bar width
    group_width = 0.8  # Reduced from 1.5 to make groups closer
    bar_width = 0.15
    
    # Extract data for each dataset and budget
    laion_data = {b: 0 for b in budgets}
    ytb_data = {b: 0 for b in budgets}
    
    for budget in budgets:
        if indexing_data and "laion" in indexing_data and budget in indexing_data["laion"]:
            time_s, size_mb = indexing_data["laion"][budget]
            if metric == "size":
                laion_data[budget] = size_mb
            else:
                laion_data[budget] = time_s
        
        if indexing_data and "ytb_audio" in indexing_data and budget in indexing_data["ytb_audio"]:
            time_s, size_mb = indexing_data["ytb_audio"][budget]
            if metric == "size":
                ytb_data[budget] = size_mb
            else:
                ytb_data[budget] = time_s
    
    # Create positions for bars within each group
    # For each group, we have 5 bars (one for each budget)
    laion_x_base = np.arange(len(budgets)) * bar_width - (len(budgets) - 1) * bar_width / 2
    ytb_x_base = laion_x_base + group_width
    
    # Plot bars for each budget with corresponding colors
    for i, budget in enumerate(budgets):
        laion_x = laion_x_base[i]
        ytb_x = ytb_x_base[i]
        
        # Plot LAION bar
        ax.bar(laion_x, laion_data[budget], bar_width, 
               color=budget_colors[budget], edgecolor='black', linewidth=0.5,
               label=f"LAION $\\phi$={budget}" if i == 0 else "")
        
        # Plot YTB-Audio bar
        ax.bar(ytb_x, ytb_data[budget], bar_width,
               color=budget_colors[budget], edgecolor='black', linewidth=0.5,
               label=f"YTB-Audio $\\phi$={budget}" if i == 0 else "")
    
    # Labels and formatting
    if metric == "size":
        ax.set_ylabel("Size (MB)", fontsize=26, labelpad=0)
    else:
        ax.set_ylabel("Time (s)", fontsize=26, labelpad=0)
    
    # Set x-ticks at group centers
    group_centers = [0, group_width]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([])  # Remove default tick labels
    
    # Manually add dataset names as text below x-axis
    ax.text(0, -0.04, r"LAION", transform=ax.get_xaxis_transform(), 
           fontsize=30, ha='center', va='top')
    ax.text(group_width, -0.04, r"YTB-Audio", transform=ax.get_xaxis_transform(), 
           fontsize=30, ha='center', va='top')
    
    ax.tick_params(axis='y', labelsize=24, pad=0)
    plt.setp(ax.get_yticklabels(), rotation=45, ha='right')
    
    # Create legend with phi values
    handles = [plt.Rectangle((0, 0), 1, 1, fc=budget_colors[b], edgecolor='black', linewidth=0.5)
              for b in budgets]
    legend_labels = [f"$\\phi$={b:,}" for b in budgets]
    if metric == "size":
        ax.legend(handles, legend_labels, fontsize=24, loc="upper left", frameon=False, ncol=1, labelspacing=0.2, handletextpad=0.3)
    
    ax.set_ylim(bottom=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["qps", "cmps"], default="qps",
                       help="Query metric to plot (default: qps)")
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="If set, plot only the four query subplots (2×2) and skip indexing bars",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Transpose the subplot layout (rows become columns)",
    )
    args = parser.parse_args()
    
    global rotate
    rotate = args.rotate

    # Load indexing data only when needed for bar plots
    indexing_data = None if args.query_only else load_indexing_data()

    # Figure layout: default 3×2 (with bars); 2×2 when --query-only is set
    # With rotate: dimensions are transposed
    if args.query_only:
        if rotate:
            fig = plt.figure(figsize=(8, 16))
            gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.18, top=0.92)
        else:
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.12, top=0.92)
    else:
        if rotate:
            # Rotated: 2 rows (datasets) × 3 cols (2 scenarios + 1 indexing)
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.20, wspace=0.25, top=0.92)
        else:
            # Original: 3 rows × 2 cols
            fig = plt.figure(figsize=(16, 13))
            gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.20, top=0.92)
    
    # Create subplots based on rotation
    if rotate:
        # Rotated layout: 2 rows (datasets) × 2 cols (scenarios) or × 3 cols (with indexing)
        # Row 0: LAION, Row 1: YTB-Audio
        # Col 0: Containment, Col 1: Overlap, Col 2: Indexing (if not query_only)
        
        ax00 = fig.add_subplot(gs[0, 0])
        plot_query_performance_dataset(ax00, "laion", "containment", metric=args.metric)
        ax00.set_title(r"Containment", fontsize=30)
        ax00.set_ylabel(r"\textbf{LAION}" + "\n" + "QPS", fontsize=28)
        ax00.set_xlabel("Recall@10", fontsize=28, labelpad=2)
        ax00.set_yscale("log")
        
        ax01 = fig.add_subplot(gs[0, 1])
        plot_query_performance_dataset(ax01, "laion", "overlap", metric=args.metric)
        ax01.set_title(r"Overlap", fontsize=30)
        ax01.set_ylabel("QPS", fontsize=28)
        ax01.set_xlabel("Recall@10", fontsize=28, labelpad=2)
        ax01.set_yscale("log")
        
        ax10 = fig.add_subplot(gs[1, 0])
        plot_query_performance_dataset(ax10, "ytb_audio", "containment", metric=args.metric)
        ax10.set_ylabel(r"\textbf{YTB-Audio}" + "\n" + "QPS", fontsize=28)
        ax10.set_xlabel("Recall@10", fontsize=28, labelpad=2)
        ax10.set_yscale("log")
        
        ax11 = fig.add_subplot(gs[1, 1])
        plot_query_performance_dataset(ax11, "ytb_audio", "overlap", metric=args.metric)
        ax11.set_ylabel("QPS", fontsize=28)
        ax11.set_xlabel("Recall@10", fontsize=28, labelpad=2)
        ax11.set_yscale("log")
        
        # Indexing plots in column 2 (if not query_only)
        if not args.query_only:
            ax02 = fig.add_subplot(gs[0, 2])
            plot_indexing_bars(ax02, indexing_data, metric="size")
            ax02.set_title(r"Indexing", fontsize=30)
            
            ax12 = fig.add_subplot(gs[1, 2])
            plot_indexing_bars(ax12, indexing_data, metric="time")
    
    else:
        # Original layout: Row 0: LAION dataset (Containment and Overlap), etc.
        ax00 = fig.add_subplot(gs[0, 0])
        plot_query_performance_dataset(ax00, "laion", "containment", metric=args.metric)
        ax00.set_ylabel(r"Containment" + "\n" + "QPS", fontsize=28)
        ax00.set_yscale("log")
        ax00.set_xlabel("Recall@10", fontsize=28, labelpad=2)
        ax00.text(0.05, 0.05, r"\textbf{LAION}", transform=ax00.transAxes, fontsize=30, 
                 va='bottom', ha='left')
        
        ax01 = fig.add_subplot(gs[0, 1])
        plot_query_performance_dataset(ax01, "ytb_audio", "containment", metric=args.metric)
        ax01.set_ylabel("QPS", fontsize=28)
        ax01.set_yscale("log")
        ax01.set_xlabel("Recall@10", fontsize=28, labelpad=2)
        ax01.text(0.05, 0.05, r"\textbf{YTB-Audio}", transform=ax01.transAxes, fontsize=30, 
                 va='bottom', ha='left')
        
        # Row 1: LAION and YTB-Audio overlap
        ax10 = fig.add_subplot(gs[1, 0])
        plot_query_performance_dataset(ax10, "laion", "overlap", metric=args.metric)
        ax10.set_ylabel(r"Overlap" + "\n" + "QPS", fontsize=28)
        ax10.set_xlabel("Recall@10", fontsize=28, labelpad=2)
        ax10.set_yscale("log")
        
        ax11 = fig.add_subplot(gs[1, 1])
        plot_query_performance_dataset(ax11, "ytb_audio", "overlap", metric=args.metric)
        ax11.set_xlabel("Recall@10", fontsize=28, labelpad=2)
        ax11.set_ylabel("QPS", fontsize=28)
        ax11.set_yscale("log")
        
        # Row 2: Indexing information (only when not query-only)
        if not args.query_only:
            ax20 = fig.add_subplot(gs[2, 0])
            plot_indexing_bars(ax20, indexing_data, metric="size")

            ax21 = fig.add_subplot(gs[2, 1])
            plot_indexing_bars(ax21, indexing_data, metric="time")
    
    # Add legend for query performance plots
    handles = [plt.Line2D([0], [0], color=budget_colors[b], marker=budget_markers[b],
                         linestyle='-', linewidth=2.5, markersize=12, markerfacecolor="none",
                         label=f"$\\phi$={b:,}")
              for b in budgets]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=30, 
              bbox_to_anchor=(0.5, 0.985), frameon=False, columnspacing=1.0, handletextpad=0.3)
    
    # Save figure
    output_file = "plot_ivf_budget.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
