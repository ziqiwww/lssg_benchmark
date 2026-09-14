import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# Datasets
datasets = ["laion", "ytb-audio"]

# Methods
methods = ["poindex", "poindex_wo", "poindex_minhash", "poindex_minhash_wo"]
scenarios = ["containment", "overlap"]

# Colors for scenarios - use color palettes with different shades for with/without
method_colors = {
    # Containment palette (reds)
    ("containment", "poindex"): "#5e62a9",              # Dark red
    ("containment", "poindex_wo"): "#88a2c4",          # Light red
    ("containment", "poindex_minhash"): "#526a40",     # Darker red
    ("containment", "poindex_minhash_wo"): "#b5ea8c",  # Lighter red
    # Overlap palette (blues)
    ("overlap", "poindex"): "#730220",                 # Dark blue
    ("overlap", "poindex_wo"): "#f27b50",              # Light blue
    ("overlap", "poindex_minhash"): "#b6042a",         # Darker blue
    ("overlap", "poindex_minhash_wo"): "#f29f05",      # Lighter blue
}

# Markers for methods (same marker for poindex/poindex_wo, same for minhash variants)
method_markers = {
    "poindex": "o",            # Circle
    "poindex_wo": "o",         # Circle (same)
    "poindex_minhash": "s",    # Square
    "poindex_minhash_wo": "s"  # Square (same)
}

# Line styles for with/without
linestyle_map = {
    "poindex": "-",            # Solid
    "poindex_wo": "--",        # Dashed
    "poindex_minhash": "-",    # Solid
    "poindex_minhash_wo": "--" # Dashed
}

# Display names
method_display_map = {
    "poindex": "IVF",
    "poindex_wo": "IVF w/o",
    "poindex_minhash": "MinHash",
    "poindex_minhash_wo": "MinHash w/o"
}

# Bar plot colors
bar_colors = {
    "poindex": ("#730220", 1.0),              # Red, full opacity
    "poindex_wo": ("#f27b50", 1.0),           # Red, 50% opacity
    "poindex_minhash": ("#5e62a9", 1.0),      # Blue, full opacity
    "poindex_minhash_wo": ("#88a2c4", 1.0)    # Blue, 50% opacity
}

# Dataset display names
dataset_display_map = {
    "laion": "LAION",
    "ytb-audio": "YTB-Audio",
    "ytb_audio": "YTB-Audio"
}

# Rotation flag: if True, transpose the layout (rows become columns)
rotate = False

def get_method_display_name(method):
    """Get display name for a method"""
    return method_display_map.get(method, method)

def get_dataset_display_name(dataset):
    """Get display name for a dataset"""
    return dataset_display_map.get(dataset, dataset)

def load_query_data(dataset, scenario, method):
    """
    Load query performance data from labeldiver dataset directory
    Format: L, Recall, QPS, Cmps, hops (no header, 5 columns)
    Files are named: {base_method}_{scenario}_wo.csv or {base_method}_{scenario}.csv
    """
    # Determine if this is a "_wo" variant
    if method.endswith("_wo"):
        base_method = method[:-3]  # Remove "_wo" suffix
        file_path = f"/root/work/lssg/example/plot/labeldiver/{dataset}/{base_method}_{scenario}_wo.csv"
    else:
        file_path = f"/root/work/lssg/example/plot/labeldiver/{dataset}/{method}_{scenario}.csv"
    
    if not os.path.exists(file_path):
        return None
    
    try:
        # First, try to load directly (no header)
        data = np.loadtxt(file_path, delimiter=",")
    except (ValueError, UnicodeDecodeError):
        # If that fails, skip non-numeric lines (debug headers)
        try:
            with open(file_path, 'r') as f:
                lines = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Check if line starts with a digit (actual data)
                    if line[0].isdigit():
                        # Also check if it has exactly 5 comma-separated values
                        parts = line.split(',')
                        if len(parts) == 5:
                            try:
                                # Try to parse all values as floats
                                float(parts[0])
                                float(parts[1])
                                lines.append(line)
                            except ValueError:
                                continue
                
                # Convert to numpy array
                if not lines:
                    return None
                data = np.loadtxt(lines, delimiter=",")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    result = np.zeros(data.shape[0], dtype=[
        ('L', np.int32),
        ('Recall', np.float64),
        ('QPS', np.float64),
        ('Cmps', np.float64),
        ('hops', np.float64)
    ])
    result['L'] = data[:, 0].astype(np.int32)
    result['Recall'] = data[:, 1]
    result['QPS'] = data[:, 2]
    result['Cmps'] = data[:, 3]
    result['hops'] = data[:, 4]
    return result

def load_indexing_data():
    """
    Load indexing data from labeldiver/indexing.csv
    Format: db, method, time, size (with header)
    """
    file_path = "/root/work/lssg/example/plot/labeldiver/indexing.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_labeldiver_ablation(metric="QPS"):
    """
    Plot ablation study results with 4 subfigures:
    - Original layout (rotate=False): 
      - Top-left: LAION dataset (both scenarios)
      - Top-right: YTB-audio dataset (both scenarios)
      - Bottom-left: Index size bar plot
      - Bottom-right: Index time bar plot
    - Rotated layout (rotate=True):
      - Top-left: LAION dataset (both scenarios)
      - Top-right: Index size bar plot
      - Bottom-left: YTB-audio dataset (both scenarios)
      - Bottom-right: Index time bar plot
    """
    
    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.22, top=0.80)
    
    # Plot 1: LAION dataset
    ax1 = fig.add_subplot(gs[0, 0])
    
    tick_size=26
    label_size=28
    
    if rotate:
        # Rotated layout: datasets on rows, indexing metrics on columns
        # Top-left: LAION dataset
        plot_query_performance_dataset(ax1, "laion", metric)
        ax1.set_xlabel("Recall@10", fontsize=label_size, labelpad=0)
        ax1.set_ylabel(metric, fontsize=label_size)
        ax1.tick_params(axis='both', labelsize=tick_size)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")
        ax1.set_xlim(0.8, 1.0)
        ax1.text(0.05, 0.05, r"\textbf{LAION}", transform=ax1.transAxes, fontsize=30, 
                va='bottom', ha='left')
        
        # Top-right: Index size bar plot
        ax2 = fig.add_subplot(gs[0, 1])
        plot_indexing_bar(ax2, "size")
        ax2.set_ylabel("Size (MB)", fontsize=label_size,labelpad=0)
        ax2.tick_params(axis='both', labelsize=tick_size, rotation=45, pad=1)
        
        # Bottom-left: YTB-audio dataset
        ax3 = fig.add_subplot(gs[1, 0])
        plot_query_performance_dataset(ax3, "ytb-audio", metric)
        ax3.set_xlabel("Recall@10", fontsize=label_size, labelpad=0)
        ax3.set_ylabel(metric, fontsize=label_size)
        ax3.tick_params(axis='both', labelsize=tick_size)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")
        ax3.set_xlim(0.8, 1.0)
        ax3.text(0.05, 0.05, r"\textbf{YTB-Audio}", transform=ax3.transAxes, fontsize=30, 
                va='bottom', ha='left')
        
        # Bottom-right: Index time bar plot
        ax4 = fig.add_subplot(gs[1, 1])
        plot_indexing_bar(ax4, "time")
        ax4.set_ylabel("Time (s)", fontsize=label_size,labelpad=0)
        ax4.tick_params(axis='both', labelsize=tick_size, rotation=45, pad=1)
    else:
        # Original layout: datasets on top row, indexing on bottom row
        # Top-left: LAION dataset
        plot_query_performance_dataset(ax1, "laion", metric)
        ax1.set_xlabel("Recall@10", fontsize=label_size, labelpad=0)
        ax1.set_ylabel(metric, fontsize=label_size)
        ax1.tick_params(axis='both', labelsize=tick_size)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")
        ax1.set_xlim(0.8, 1.0)
        ax1.text(0.05, 0.05, r"\textbf{LAION}", transform=ax1.transAxes, fontsize=30, 
                va='bottom', ha='left')
        
        # Top-right: YTB-audio dataset
        ax2 = fig.add_subplot(gs[0, 1])
        plot_query_performance_dataset(ax2, "ytb-audio", metric)
        ax2.set_xlabel("Recall@10", fontsize=label_size, labelpad=0)
        ax2.set_ylabel(metric, fontsize=label_size)
        ax2.tick_params(axis='both', labelsize=tick_size)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")
        ax2.set_xlim(0.8, 1.0)
        ax2.text(0.05, 0.05, r"\textbf{YTB-Audio}", transform=ax2.transAxes, fontsize=30, 
                va='bottom', ha='left')
        
        # Bottom-left: Index size bar plot
        ax3 = fig.add_subplot(gs[1, 0])
        plot_indexing_bar(ax3, "size")
        ax3.set_ylabel("Size (MB)", fontsize=label_size,labelpad=0)
        ax3.tick_params(axis='both', labelsize=tick_size, rotation=45, pad=1)
        
        # Bottom-right: Index time bar plot
        ax4 = fig.add_subplot(gs[1, 1])
        plot_indexing_bar(ax4, "time")
        ax4.set_ylabel("Time (s)", fontsize=label_size,labelpad=0)
        ax4.tick_params(axis='both', labelsize=tick_size, rotation=45, pad=1)
    
    # Create unified legend at top without box
    handles = []
    labels = []
    
    # Containment curves
    for method in methods:
        color = method_colors[("containment", method)]
        marker = method_markers[method]
        ls = linestyle_map[method]
        
        line, = ax1.plot([], [], 
                color=color,
                marker=marker,
                markersize=12,
                markerfacecolor="none",
                linewidth=2.5,
                linestyle=ls)
        handles.append(line)
        labels.append(f"Cont-{get_method_display_name(method)}")
    
    # Overlap curves
    for method in methods:
        color = method_colors[("overlap", method)]
        marker = method_markers[method]
        ls = linestyle_map[method]
        
        line, = ax2.plot([], [], 
                color=color,
                marker=marker,
                markersize=12,
                markerfacecolor="none",
                linewidth=2.5,
                linestyle=ls)
        handles.append(line)
        labels.append(f"Ovlp-{get_method_display_name(method)}")
    
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.78), 
              ncol=4, fontsize=26, frameon=False, handlelength=1.8, handletextpad=0.3, columnspacing=0.6, labelspacing=0.2)
    
    # Add bar plot legend to appropriate plot (ax2 if rotated, ax3 if not)
    bar_handles = []
    bar_labels = []
    for method in methods:
        # Get color and alpha from bar_colors dictionary
        color, alpha = bar_colors[method]
        
        bar = plt.Rectangle((0,0),1,1, fc=color, alpha=alpha, edgecolor='black', linewidth=1.5)
        bar_handles.append(bar)
        bar_labels.append(get_method_display_name(method))
    
    # Place legend on the first indexing plot (ax2 if rotated, ax3 if not)
    legend_ax = ax2 if rotate else ax3
    legend_ax.legend(bar_handles, bar_labels, loc="upper left", fontsize=24, frameon=False,
              handletextpad=0.2, labelspacing=0.25)
    
    plt.savefig("plot_labeldiver.pdf", dpi=150, bbox_inches="tight")
    print("Figure saved as plot_labeldiver.pdf")
    plt.show()

def plot_query_performance_dataset(ax, dataset, metric):
    """Plot query performance curves for both scenarios on a single dataset"""
    
    for scenario in scenarios:
        for method in methods:
            data = load_query_data(dataset, scenario, method)
            if data is None:
                continue
            
            # Filter to recall >= 0.8
            mask = data['Recall'] >= 0.8
            if not any(mask):
                continue
            data = data[mask]
            
            # Sort by recall
            indices = np.argsort(data['Recall'])
            data = data[indices]
            
            x = data['Recall']
            if metric == "QPS":
                y = data['QPS']
            elif metric == "Cmps":
                y = data['Cmps']
            else:
                y = data['QPS']
            
            # Get line style, marker, and color
            ls = linestyle_map[method]
            marker = method_markers[method]
            color = method_colors[(scenario, method)]
            
            # Plot
            line, = ax.plot(x, y, 
                          label=f"{scenario}-{get_method_display_name(method)}",
                          color=color,
                          marker=marker,
                          markersize=12,
                          markerfacecolor="none",
                          linewidth=2.5,
                          ls=ls)

def plot_indexing_bar(ax, metric_type):
    """Plot indexing bar chart (time or size)"""
    
    df = load_indexing_data()
    if df is None:
        return
    
    # Get unique datasets in sorted order
    datasets_unique = sorted(df['db'].unique())
    
    bar_width = 0.2
    x_positions = []
    x_labels = []
    group_x = 0
    
    for dataset_idx, dataset in enumerate(datasets_unique):
        dataset_data = df[df['db'] == dataset]
        
        # Plot bars for each method in order
        for method_idx, method in enumerate(methods):
            method_data = dataset_data[dataset_data['method'] == method]
            if len(method_data) > 0:
                if metric_type == "time":
                    value = method_data['time'].values[0]
                else:  # size
                    value = method_data['size'].values[0]
                
                # Get color and alpha from bar_colors dictionary
                color, alpha = bar_colors[method]
                
                # Plot bar
                ax.bar(group_x + method_idx * bar_width, value, bar_width,
                      color=color, alpha=alpha, edgecolor='black', linewidth=1.5)
        
        # Add x-axis label and position
        x_positions.append(group_x + 1.5 * bar_width)
        x_labels.append(get_dataset_display_name(dataset))
        
        # Move to next dataset group position
        group_x += 4 * bar_width + 0.2
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([])  # Remove default tick labels
    
    # Manually add dataset names as text below x-axis
    for pos, label in zip(x_positions, x_labels):
        ax.text(pos, -0.04, label, transform=ax.get_xaxis_transform(), 
               fontsize=30, ha='center', va='top')
    
    ax.tick_params(axis='y', labelsize=24)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot labeldiver ablation study")
    parser.add_argument("--metric", default="QPS", choices=["QPS", "Cmps"],
                       help="Metric to plot (QPS or Cmps)")
    parser.add_argument("--rotate", action="store_true",
                       help="Transpose the subplot layout (rows become columns)")
    args = parser.parse_args()
    
    rotate = args.rotate
    
    plot_labeldiver_ablation(metric=args.metric)
