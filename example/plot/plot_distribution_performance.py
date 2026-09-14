import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# Datasets
# datasets = ["LAION", "tripclick"]
datasets = ["LAION"]

# Scenarios
scenarios = ["equality", "containment", "overlap"]

# Distributions
distributions = ["zipf","uniform", "poisson", "multi_normial"]

# Methods
methods = ["poindex", "poindex_minhash", "ung", "eli"]

# Colors and markers for each method
color_map = {
    "poindex": "#b60033",  # Red
    "poindex_minhash": "#ff7f0e",  # Orange
    "ung": "#27b9d6",      # Blue
    "eli": "#2ca02c",      # Green
}

marker_map = {
    "poindex": "o",  # Circle
    "poindex_minhash": "o",  # Circle
    "ung": "^",      # Triangle
    "eli": "s",      # Square
}

# Method name mapping for display
method_display_map = {
    "poindex_minhash": "LSSG-MinHash",
    "poindex": "LSSG-IVF",
    "ung": "UNG",
    "eli": "ELI",
}


def get_method_display_name(method):
    """Get the display name for a method"""
    return method_display_map.get(method, method)


def load_distribution_data(dataset, distribution, scenario, method):
    """
    Load data from distribution result files
    Returns structured array with Recall, QPS, Cmps fields
    """
    # Map method to directory name
    if method == "poindex":
        method_dir = "ivf"
        file_name = f"poindex_{scenario}.csv"
    elif method == "poindex_minhash":
        method_dir = "minhash"
        file_name = f"poindex_{scenario}.csv"
    elif method == "ung":
        method_dir = "ung"
        file_name = f"result_{scenario}.csv"
    elif method == "eli":
        method_dir = "eli"
        file_name = f"result_{scenario}.csv"
    else:
        return None
    
    # Construct file path
    file_path = f"/root/work/lssg/example/plot/distribution/{method_dir}_{dataset}_{distribution}/{file_name}"
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return None
    
    try:
        if method == "poindex" or method == "poindex_minhash":
            # Format: L,Recall,QPS,Cmps,hops (no header)
            data = np.loadtxt(file_path, delimiter=",")
            if data.size == 0:
                return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32),
                ('Recall', np.float64),
                ('QPS', np.float64),
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0].astype(np.int32)
            result['Recall'] = data[:, 1]
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 3]
            return result
        
        elif method == "ung":
            # Format: L,Cmps,QPS,Recall (with header, Recall in percentage)
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            if data.size == 0:
                return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Convert recall from percentage to decimal
            recall = data[:, 3]
            if np.any(recall > 1):
                recall = recall / 100.0
            
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32),
                ('Recall', np.float64),
                ('QPS', np.float64),
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0].astype(np.int32)
            result['Recall'] = recall
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 1]
            return result
        
        elif method == "eli":
            # Format: L,Recall,QPS,Cmps (no header)
            data = np.loadtxt(file_path, delimiter=",")
            if data.size == 0:
                return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32),
                ('Recall', np.float64),
                ('QPS', np.float64),
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0].astype(np.int32)
            result['Recall'] = data[:, 1]
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 3]
            return result
        
        else:
            return None
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def dataset_beautify(dataset):
    """Convert dataset names to more readable format"""
    if dataset == "LAION":
        return "LAION"
    elif dataset == "tripclick":
        return "TripClick"
    else:
        return dataset.upper()


def scenario_beautify(scenario):
    """Convert scenario names to more readable format"""
    if scenario == "equality":
        return "Equality"
    elif scenario == "containment":
        return "Containment"
    elif scenario == "overlap":
        return "Overlap"
    else:
        return scenario.capitalize()


def distribution_beautify(distribution):
    """Convert distribution names to more readable format"""
    if distribution == "uniform":
        return "Uniform"
    elif distribution == "poisson":
        return "Poisson"
    elif distribution == "multi_normial":
        return "Multinomial"
    elif distribution == "zipf":
        return "Zipf"
    else:
        return distribution.capitalize()


def method_zorder(method):
    """Determine z-order for plotting methods"""
    z_order = list(methods)
    if method in z_order:
        return len(z_order) - z_order.index(method)
    return 0


def create_distribution_plot(metric="QPS", minimum_recall=0.5):
    """
    Create QPS-Recall or Cmps-Recall curves for different distributions
    Layout: 6 rows × 3 columns
    Rows: dataset-scenario combinations (LAION-containment, LAION-equality, LAION-overlap,
                                          TripClick-containment, TripClick-equality, TripClick-overlap)
    Columns: distributions (uniform, poisson, multinomial)
    """
    # Create dataset-scenario combinations
    dataset_scenario_pairs = [(ds, sc) for ds in datasets for sc in scenarios]
    
    # Create figure with subplots - reduced height per subplot
    fig, axs = plt.subplots(
        len(dataset_scenario_pairs), len(distributions),
        figsize=(len(distributions) * 3.5, len(dataset_scenario_pairs) * 2.6),  # Reduced height from 2.8 to 2.2
        dpi=150
    )
    
    # Handle case of single subplot (shouldn't happen here but for safety)
    if len(dataset_scenario_pairs) == 1 and len(distributions) == 1:
        axs = np.array([[axs]])
    elif len(dataset_scenario_pairs) == 1:
        axs = np.array([axs])
    elif len(distributions) == 1:
        axs = np.array([[ax] for ax in axs])
    
    handles = []
    pushed_methods = []
    has_data = False
    
    # For each dataset-scenario pair and distribution
    for i, (dataset, scenario) in enumerate(dataset_scenario_pairs):
        for j, distribution in enumerate(distributions):
            ax = axs[i, j]
            
            # Plot each method
            for method in methods:
                data = load_distribution_data(dataset, distribution, scenario, method)
                if data is None:
                    continue
                
                # Filter by minimum recall
                mask = data['Recall'] >= minimum_recall
                if not any(mask):
                    continue
                data = data[mask]
                
                # Sort by recall (descending) for cleaner plots
                indices = np.argsort(data['Recall'])[::-1]
                data = data[indices]
                
                # Extract x and y values
                x = data['Recall']
                if metric == "QPS":
                    y = data['QPS']
                elif metric == "Cmps":
                    y = data['Cmps']
                else:
                    raise ValueError(f"Metric {metric} not recognized")
                
                # Plot the curve
                line, = ax.plot(
                    x, y,
                    label=method.upper(),
                    color=color_map.get(method, "black"),
                    marker=marker_map.get(method, "o"),
                    markersize=8,
                    markerfacecolor="none",
                    linewidth=2,
                    zorder=method_zorder(method)
                )
                
                # Add to legend if not already added
                method_display = get_method_display_name(method)
                if method_display not in pushed_methods:
                    pushed_methods.append(method_display)
                    handles.append(line)
                
                has_data = True
            
            # Configure subplot
            # Title: distribution name (top row only) - increased font size
            if i == 0:
                ax.set_title(f"{distribution_beautify(distribution)}", fontsize=26)
            
            # Y-axis label: Scenario + Metric (left column only) - removed dataset name
            if j == 0:
                print_metric = metric
                if metric == "Cmps":
                    print_metric = "DC"
                ax.set_ylabel(f"{scenario_beautify(scenario)}\n{print_metric}", 
                             fontsize=24, labelpad=0)
            
            # Add dataset name as large bold text at bottom-left of first subplot of each dataset
            # This appears for rows 0 (LAION) and 3 (TripClick), column 0
            # if j == 0 and i % 3 == 0:
            #     # Add text at bottom-left inside the subplot
            #     ax.text(0.02, 0.02, dataset_beautify(dataset), 
            #            transform=ax.transAxes,
            #            fontsize=22, fontweight='bold', 
            #            va='bottom', ha='left',
            #            bbox=dict(boxstyle='round', facecolor='none', alpha=0.8, edgecolor='none'))
            
            # X-axis label (bottom row only)
            if i == len(dataset_scenario_pairs) - 1:
                ax.set_xlabel(f"Recall@10", fontsize=24, labelpad=0)
            
            # Set y-axis to logarithmic scale
            if metric in ["QPS", "Cmps"]:
                ax.set_yscale("log")
            
            # Set x-axis limits
            ax.set_xlim(minimum_recall, 1.01)
            
            # Configure ticks
            ax.tick_params(axis="both", which="major", labelsize=20)
            ax.tick_params(axis="x", which="major", labelsize=22)
            
            # Add grid for readability
            ax.grid(True, linestyle="--", alpha=0.7)
    
    if not has_data:
        plt.close(fig)
        print(f"No data available for plotting. Please check your data files.")
        return
    
    # Add legend at the top
    if handles:
        # Reorder handles to match methods list
        method_order = {get_method_display_name(m): i for i, m in enumerate(methods)}
        sorted_pairs = sorted(zip(pushed_methods, handles), key=lambda x: method_order.get(x[0], 999))
        pushed_methods_sorted, handles_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        fig.legend(
            handles=handles_sorted,
            labels=pushed_methods_sorted,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(pushed_methods_sorted) if pushed_methods_sorted else 1,
            fontsize=28,
            frameon=False,
            handletextpad=0.3,
            borderpad=0,
            labelspacing=0,
        )
    
    # Adjust layout - shorter gaps between subplots
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.20, wspace=0.17)  # Reduced from 0.3 and 0.25
    
    # Save figure
    output_pdf = f"distribution_{metric.lower()}_recall.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    print(f"Plot saved as {output_pdf}")


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QPS-Recall or Cmps-Recall curves for different label distributions.')
    parser.add_argument('--metric', type=str, choices=['QPS', 'Cmps'], default='QPS',
                        help='Metric to plot (QPS or Cmps, default: QPS)')
    parser.add_argument('--min-recall', type=float, default=0.5,
                        help='Minimum recall value to display (default: 0.5)')
    parser.add_argument('--methods', type=str, nargs='+', default=methods,
                        help='Methods to include in the plot')
    parser.add_argument('--datasets', type=str, nargs='+', default=datasets,
                        help='Datasets to include in the plot')
    parser.add_argument('--distributions', type=str, nargs='+', default=distributions,
                        help='Distributions to include in the plot')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename prefix (default: distribution_{metric}_recall)')
    
    args = parser.parse_args()
    
    # Update global variables
    methods = args.methods
    datasets = args.datasets
    distributions = args.distributions
    
    # Print configuration
    print(f"Plotting {args.metric}-Recall curves for different label distributions")
    print(f"Datasets: {datasets}")
    print(f"Distributions: {distributions}")
    print(f"Methods: {methods}")
    print(f"Minimum recall: {args.min_recall}")
    
    # Create the plot
    create_distribution_plot(metric=args.metric, minimum_recall=args.min_recall)
    
    # Rename output if specified
    if args.output:
        import shutil
        src = f"distribution_{args.metric.lower()}_recall.pdf"
        dst = f"{args.output}.pdf"
        shutil.move(src, dst)
        print(f"Plot renamed to {dst}")
