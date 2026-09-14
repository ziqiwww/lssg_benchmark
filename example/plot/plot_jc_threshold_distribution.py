import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# Configuration
datasets = ["laion", "ytb-audio"]
scenarios = ["containment", "overlap"]
methods = ["ivf", "minhash"]
threshold_divisions = ["uniform", "logarithm", "exponential", "quadratic"]

# Colors for threshold divisions
division_colors = {
    "uniform": "#5e62a9",       # Blue
    "logarithm": "#88a2c4",     # Light blue
    "exponential": "#730220",   # Dark red
    "quadratic": "#f27b50"      # Orange
}

# Markers for threshold divisions
division_markers = {
    "uniform": "o",       # Circle
    "logarithm": "s",     # Square
    "exponential": "^",   # Triangle
    "quadratic": "D"      # Diamond
}

# Display names
division_display_map = {
    "uniform": "Uniform",
    "logarithm": "Logarithmic",
    "exponential": "Exponential",
    "quadratic": "Quadratic"
}

dataset_display_map = {
    "laion": "LAION",
    "ytb-audio": "YTB-Audio"
}

method_display_map = {
    "ivf": "LSSG-IVF",
    "minhash": "LSSG-MinHash"
}

scenario_display_map = {
    "containment": "Containment",
    "overlap": "Overlap"
}

def get_division_display_name(division):
    """Get display name for a threshold division"""
    return division_display_map.get(division, division)

def get_dataset_display_name(dataset):
    """Get display name for a dataset"""
    return dataset_display_map.get(dataset, dataset)

def get_method_display_name(method):
    """Get display name for a method"""
    return method_display_map.get(method, method)

def get_scenario_display_name(scenario):
    """Get display name for a scenario"""
    return scenario_display_map.get(scenario, scenario)

def load_query_data(dataset, division, method, scenario):
    """
    Load query performance data from jc_threshold_distribution directory
    Format: L, Recall, QPS, Cmps, hops (no header, 5 columns)
    Files are named: {method}_poindex_{scenario}.csv
    May contain debug headers that need to be skipped
    """
    file_path = f"/root/work/lssg/example/plot/jc_threshold_distribution/{dataset}/{division}/{method}_poindex_{scenario}.csv"
    
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
                    if line[0].isdigit() or (line[0] == '-' and len(line) > 1 and line[1].isdigit()):
                        # Check if it has exactly 5 comma-separated values
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
            print(f"Warning: Could not load {file_path}: {e}")
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

def plot_jc_threshold_distribution(metric="QPS"):
    """
    Plot Jaccard threshold distribution comparison with 4x2 subfigures:
    Rows:
      - Row 0: IVF Containment
      - Row 1: IVF Overlap
      - Row 2: MinHash Containment
      - Row 3: MinHash Overlap
    Columns: 
      - Col 0: LAION
      - Col 1: YTB-Audio
    Each plot shows 4 curves for different threshold divisions
    """
    
    fig = plt.figure(figsize=(14, 18), dpi=150)
    gs = fig.add_gridspec(4, 2, hspace=0.15, wspace=0.14, top=0.85)
    
    # Create 4x2 grid of plots
    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],  # Row 0: IVF Containment
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],  # Row 1: IVF Overlap
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],  # Row 2: MinHash Containment
        [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]   # Row 3: MinHash Overlap
    ]
    
    # Plot data for each method, scenario, and dataset combination
    for method_idx, method in enumerate(methods):
        for scenario_idx, scenario in enumerate(scenarios):
            # Calculate row index: IVF containment (row 0), IVF overlap (row 1), MinHash containment (row 2), MinHash overlap (row 3)
            row = method_idx * 2 + scenario_idx
            
            for dataset_idx, dataset in enumerate(datasets):
                ax = axes[row][dataset_idx]
                
                # Plot curves for each threshold division
                for division in threshold_divisions:
                    data = load_query_data(dataset, division, method, scenario)
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
                    
                    # Get color and marker
                    color = division_colors[division]
                    marker = division_markers[division]
                    
                    # Plot
                    line, = ax.plot(x, y,
                                  label=get_division_display_name(division),
                                  color=color,
                                  marker=marker,
                                  markersize=10,
                                  markerfacecolor="none",
                                  linewidth=2.5,
                                  ls="-")
                
                # Configure subplot
                # Only set x-label for bottom row (row == 3)
                if row == 3:
                    ax.set_xlabel("Recall@10", fontsize=28)
                
                # Only set y-label for first column (dataset_idx == 0)
                if dataset_idx == 0:
                    scenario_label = get_scenario_display_name(scenario)
                    ax.set_ylabel(f"{scenario_label}\n{metric}", fontsize=26)
                
                ax.tick_params(axis='both', labelsize=24)
                ax.grid(True, alpha=0.3)
                ax.set_yscale("log")
                ax.set_xlim(0.8, 1.0)
                
                # Add dataset name at top for first row only (not bold)
                if row == 0:
                    dataset_label = get_dataset_display_name(dataset)
                    ax.set_title(dataset_label, fontsize=30, pad=3)
                
                # Add method name as bold text at bottom-left
                method_label = get_method_display_name(method)
                ax.text(0.05, 0.05, r"\textbf{" + method_label + "}", transform=ax.transAxes,
                       fontsize=28, va='bottom', ha='left')
    
    # Create unified legend
    handles = []
    labels = []
    for division in threshold_divisions:
        color = division_colors[division]
        marker = division_markers[division]
        
        line, = axes[0][0].plot([], [],
                              color=color,
                              marker=marker,
                              markersize=10,
                              markerfacecolor="none",
                              linewidth=2.5,
                              ls="-")
        handles.append(line)
        labels.append(get_division_display_name(division))
    
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
              ncol=4, fontsize=32, frameon=False, handletextpad=0.2, columnspacing=0.4, labelspacing=0.2, handlelength=1.5)
    
    plt.savefig("/root/work/lssg/example/plot/plot_jc_threshold_distribution.pdf", dpi=150, bbox_inches="tight")
    print("Figure saved as plot_jc_threshold_distribution.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Jaccard threshold distribution comparison")
    parser.add_argument("--metric", default="QPS", choices=["QPS", "Cmps"],
                       help="Metric to plot (QPS or Cmps)")
    args = parser.parse_args()
    
    plot_jc_threshold_distribution(metric=args.metric)
