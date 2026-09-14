import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# Scenarios and methods
scenarios = ["containment", "overlap"]
methods = ["poindex", "poindex_minhash"]
distance_metrics = ["jaccard", "cosine", "dice", "overlap"]

# Colors for distance metrics
metric_colors = {
    "jaccard": "#5e62a9",     # Red (swapped with overlap)
    "cosine": "#88a2c4",      # Orange
    "dice": "#730220",        # Green
    "overlap": "#f27b50"      # Blue (swapped with jaccard)
}

# Markers for distance metrics
metric_markers = {
    "jaccard": "o",    # Circle
    "cosine": "s",     # Square
    "dice": "^",       # Triangle
    "overlap": "D"     # Diamond
}

# Display names
metric_display_map = {
    "jaccard": "Jaccard",
    "cosine": "Cosine",
    "dice": "Dice",
    "overlap": "Overlap"
}

method_display_map = {
    "poindex": "LSSG-IVF",
    "poindex_minhash": "LSSG-MinHash"
}

scenario_display_map = {
    "containment": "Containment",
    "overlap": "Overlap"
}

def get_metric_display_name(metric):
    """Get display name for a distance metric"""
    return metric_display_map.get(metric, metric)

def get_method_display_name(method):
    """Get display name for a method"""
    return method_display_map.get(method, method)

def get_scenario_display_name(scenario):
    """Get display name for a scenario"""
    return scenario_display_map.get(scenario, scenario)

def load_query_data(scenario, method, metric):
    """
    Load query performance data from setdistance directory
    Format: L, Recall, QPS, Cmps, hops (no header, 5 columns)
    Files are named: {method}_{scenario}_{metric}.csv
    """
    file_path = f"/root/work/lssg/example/plot/setdistance/LAION/{method}_{scenario}_{metric}.csv"
    
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

def plot_setdistance(metric="QPS"):
    """
    Plot setdistance ablation study with 4 subfigures:
    - Top-left: Containment + poindex
    - Top-right: Containment + poindex_minhash
    - Bottom-left: Overlap + poindex
    - Bottom-right: Overlap + poindex_minhash
    """
    
    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.14, top=0.80)
    
    # Create 2x2 grid of plots
    axes = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],  # Row 0: containment
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]   # Row 1: overlap
    ]
    
    # Plot data for each scenario and method combination
    for scenario_idx, scenario in enumerate(scenarios):
        for method_idx, method in enumerate(methods):
            ax = axes[scenario_idx][method_idx]
            
            # Plot curves for each distance metric
            for metric_name in distance_metrics:
                data = load_query_data(scenario, method, metric_name)
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
                color = metric_colors[metric_name]
                marker = metric_markers[metric_name]
                
                # Plot
                line, = ax.plot(x, y,
                              label=get_metric_display_name(metric_name),
                              color=color,
                              marker=marker,
                              markersize=12,
                              markerfacecolor="none",
                              linewidth=2.5,
                              ls="-")
            
            # Configure subplot
            # Only set x-label for bottom row (scenario_idx == 1)
            if scenario_idx == 1:
                ax.set_xlabel("Recall@10", fontsize=28)
            
            # Only set y-label for first column (method_idx == 0)
            if method_idx == 0:
                scenario_label = get_scenario_display_name(scenario)
                ax.set_ylabel(f"{scenario_label}\n{metric}", fontsize=28)
            
            ax.tick_params(axis='both', labelsize=26)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
            ax.set_xlim(0.8, 1.0)
            
            # Add method name as bold text at bottom-left
            title_text = get_method_display_name(method)
            ax.text(0.05, 0.05, r"\textbf{" + title_text + "}", transform=ax.transAxes,
                   fontsize=32, va='bottom', ha='left')
    
    # Create unified legend
    handles = []
    labels = []
    for metric_name in distance_metrics:
        color = metric_colors[metric_name]
        marker = metric_markers[metric_name]
        
        line, = axes[0][0].plot([], [],
                              color=color,
                              marker=marker,
                              markersize=12,
                              markerfacecolor="none",
                              linewidth=2.5,
                              ls="-")
        handles.append(line)
        labels.append(get_metric_display_name(metric_name))
    
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.88),
              ncol=4, fontsize=32, frameon=False, handletextpad=0.3, columnspacing=0.8, labelspacing=0.2)
    
    plt.savefig("/root/work/lssg/example/plot/plot_setdistance.pdf", dpi=150, bbox_inches="tight")
    print("Figure saved as plot_setdistance.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot setdistance ablation study")
    parser.add_argument("--metric", default="QPS", choices=["QPS", "Cmps"],
                       help="Metric to plot (QPS or Cmps)")
    args = parser.parse_args()
    
    plot_setdistance(metric=args.metric)
