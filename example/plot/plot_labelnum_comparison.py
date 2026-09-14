import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# List of label numbers to compare (default for sift1m) $|A|$
label_nums = [256, 512, 1024, 2048, 4096, 8192]
label_nums = [128, 512, 2048, 8192]
# corresponding label set num $|\mathcal{L}|$=[610621, 754394, 852615, 914985, 951901, 972650]

# Mapping from label_num to label_set_num for different datasets
label_set_mapping = {
    "sift1m": {
        256: 610621,
        512: 754394,
        1024: 852615,
        2048: 914985,
        4096: 951901,
        8192: 972650
    },
    "LAION": {
        128: 422427,
        256: 610621,  # Update with actual LAION values if different
        512: 754394,
        1024: 852615,
        2048: 914985,
        4096: 951901,
        8192: 972650
    }
}

# List of query types
query_types = ["equality", "containment", "overlap"]

# Methods to compare
methods = ["poindex", "poindex_minhash","ung", "eli", ]

# Dataset to use
dataset = "LAION"  # Can be "sift1m" or "LAION"

# Colors and markers for each method
color_map = {
    "poindex": "#b60033",  # Red
    "poindex_minhash": "#ff7f0e",  # Orange
    "ung": "#27b9d6",      # Blue
    "eli": "#2ca02c",      # Green
}

marker_map = {
    "poindex": "o",  # Circle
    "poindex_minhash": "s",  # Square
    "ung": "^",      # Triangle
    "eli": "D",      # Diamond
}

# Default configuration
metric = "QPS"  # Can be changed to "Cmps" for distance computations
minimum_recall = 0.85  # Minimum recall to display

# Method name mapping for display in plots
method_display_map = {
    "poindex_minhash": "LSSG-MinHash",
    "poindex": "LSSG-IVF",
    "ung": "UNG",
    "eli": "ELI",
}


def get_method_display_name(method):
    """
    Get the display name for a method
    """
    return method_display_map.get(method, method)


def load_data(label_num, query_type, method):
    """
    Load data from the result files
    """
    if method == "poindex":
        # poindex format: L,Recall,QPS,Cmps,hops (may have debug headers)
        folder_name = f"ivf_{dataset}_{label_num}"
        file_path = f"/root/work/lssg/example/plot/labelnum/{folder_name}/poindex_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            # Read file and check if it has debug/log header lines
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find where actual numeric data starts (5 columns for poindex)
            data_start = 0
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                # Skip empty lines
                if not line_stripped:
                    continue
                parts = line_stripped.split(',')
                # Check if line has exactly 5 columns and all are numeric
                if len(parts) == 5:
                    try:
                        # Try to parse all values as numbers
                        float(parts[0])
                        float(parts[1])
                        float(parts[2])
                        float(parts[3])
                        float(parts[4])
                        # If successful, data starts here
                        data_start = i
                        break
                    except (ValueError, IndexError):
                        # This line is not valid numeric data, continue
                        continue
            
            # Load data starting from data_start line
            data = np.loadtxt(file_path, delimiter=",", skiprows=data_start)
            
            # Convert to structured array with named fields
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64), 
                ('hops', np.float64)
            ])
            result['L'] = data[:, 0]
            result['Recall'] = data[:, 1]
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 3]
            result['hops'] = data[:, 4]
            return result
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    elif method == "poindex_minhash":
        # poindex_minhash format: L,Recall,QPS,Cmps,hops (may have debug headers)
        folder_name = f"minhash_{dataset}_{label_num}"
        file_path = f"/root/work/lssg/example/plot/labelnum/{folder_name}/poindex_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            # Read file and check if it has log/debug header lines
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find where actual numeric data starts (5 columns for poindex_minhash)
            data_start = 0
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                # Skip empty lines and known header lines
                if not line_stripped or line_stripped.startswith("ef,recall,qps"):
                    continue
                parts = line_stripped.split(',')
                # Check if line has exactly 5 columns and all are numeric
                if len(parts) == 5:
                    try:
                        # Try to parse all values as numbers
                        float(parts[0])
                        float(parts[1])
                        float(parts[2])
                        float(parts[3])
                        float(parts[4])
                        # If successful, data starts here
                        data_start = i
                        break
                    except (ValueError, IndexError):
                        # This line is not valid numeric data, continue
                        continue
            
            # Load data starting from data_start line
            data = np.loadtxt(file_path, delimiter=",", skiprows=data_start)
            
            # Convert to structured array with named fields
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64), 
                ('hops', np.float64)
            ])
            result['L'] = data[:, 0]
            result['Recall'] = data[:, 1]
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 3]
            result['hops'] = data[:, 4]
            return result
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    elif method == "eli":
        # eli format: L,recall,qps,distance_computations (no header)
        folder_name = f"eli_{dataset}_{label_num}"
        # File naming pattern: LAION1M_base_n1000448_l{label_num}_zipf.txt-{query_type}-0.20.log
        # file_path = f"/root/work/lssg/example/plot/labelnum/{folder_name}/{dataset}1M_base_n1000448_l{label_num}_zipf.txt-{query_type}-0.20.log"
        file_path = f"/root/work/lssg/example/plot/labelnum/{folder_name}/{dataset}1M_base_n1000448_l{label_num}_zipf.txt-{query_type}-0.20.log"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            data = np.loadtxt(file_path, delimiter=",")
            # Convert to structured array with named fields
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0]
            result['Recall'] = data[:, 1]
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 3]  # Distance computations
            return result
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    elif method == "ung":
        # ung format: L,Cmps,QPS,Recall (with header)
        folder_name = f"ung_{dataset}_{label_num}"
        file_path = f"/root/work/lssg/example/plot/labelnum/{folder_name}/result_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
            
        try:
            # Skip header row
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            # Convert to structured array with named fields in the same order as poindex
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0]
            # Convert recall from percentage (0-100) to decimal (0-1)
            recall_values = data[:, 3]
            if np.any(recall_values > 1):  # Check if values are percentages
                recall_values = recall_values / 100.0
            result['Recall'] = recall_values  # Recall is in 4th column for ung
            result['QPS'] = data[:, 2]        # QPS is in 3rd column for ung
            result['Cmps'] = data[:, 1]       # Cmps is in 2nd column for ung
            return result
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    else:
        print(f"Warning: Unknown method {method}")
        return None


def query_type_beautify(query_type):
    """
    Convert query types to more readable format
    """
    if query_type == "equality":
        return "Equality"
    elif query_type == "containment":
        return "Containment"
    elif query_type == "overlap":
        return "Overlap"
    else:
        return query_type.capitalize()


def method_zorder(method):
    """
    Determine z-order for plotting methods
    """
    z_order = list(methods)
    if method in z_order:
        return len(z_order) - z_order.index(method)
    return 0


def create_plot(metric="QPS"):
    """
    Create the QPS-Recall or Cmps-Recall plot with label numbers on x-axis
    """
    # Create a figure with (num_query_types x num_label_nums) subplots
    fig, axs = plt.subplots(
        len(query_types), len(label_nums), 
        figsize=(len(label_nums) * 3.5, len(query_types) * 2.6), 
        dpi=150
    )
    
    local_min_recall = minimum_recall
    
    # Handle case of single subplot
    if len(query_types) == 1 and len(label_nums) == 1:
        axs = np.array([[axs]])
    elif len(query_types) == 1:
        axs = np.array([axs])
    elif len(label_nums) == 1:
        axs = np.array([[ax] for ax in axs])

    handles = []
    pushed_methods = []
    has_data = False  # Flag to track if any data was plotted

    # For each query type and label number
    for i, query_type in enumerate(query_types):
        for j, label_num in enumerate(label_nums):
            
            local_min_recall = 0.5
            
            ax = axs[i, j]
            
            # Load and plot data for each method
            for method in methods:
                data = load_data(label_num, query_type, method)
                if data is None:
                    continue
                
                # Keep only data points with recall >= minimum_recall
                mask = data['Recall'] >= local_min_recall
                if not any(mask):
                    continue
                data = data[mask]
                
                # Sort by recall (descending) to make the plot more readable
                indices = np.argsort(data['Recall'])[::-1]
                data = data[indices]
                
                # Extract x and y coordinates
                x = data['Recall']
                if metric == "QPS":
                    y = data['QPS']
                elif metric == "Cmps":
                    y = data['Cmps']
                else:
                    raise ValueError(f"Metric {metric} not recognized")
                
                # Plot the data
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
                
                # Add to legend handles if not already added
                method_display = get_method_display_name(method)
                
                if method_display not in pushed_methods:
                    pushed_methods.append(method_display)
                    handles.append(line)
                
                # Set flag indicating we have data
                has_data = True
            
            # Configure subplot
            if i == 0:
                # Get label set count for this label number
                label_set_count = label_set_mapping.get(dataset, {}).get(label_num, "?")
                # ax.set_title(f"$|A|={label_num}$, $|\\mathcal{{L}}|={label_set_count}$", fontsize=26)
                ax.set_title(f"$|A|={label_num:,}$", fontsize=26)
            
            if j == 0:
                print_metric = metric
                if metric == "Cmps":
                    print_metric = "DC"
                ax.set_ylabel(f"{query_type_beautify(query_type)}\n{print_metric}", fontsize=24, labelpad=0)
            
            if i == len(query_types) - 1:
                ax.set_xlabel(f"Recall@10", fontsize=24, labelpad=0)
            
            # Set y-axis to logarithmic scale if using QPS or Cmps
            if metric in ["QPS", "Cmps"]:
                ax.set_yscale("log")
            
            # Set x-axis limits based on minimum recall value
            ax.set_xlim(local_min_recall, 1.01)
            
            # Set explicit x-ticks for better granularity
            ax.set_xticks([0.5, 0.7, 0.9, 1.0])
            
            # Configure ticks
            ax.tick_params(axis="both", which="major", labelsize=20)
            ax.tick_params(axis="x", which="major", labelsize=22)
            # Hide minor tick labels on y-axis
            ax.tick_params(axis="y", which="minor", labelleft=False)
            
            # Add grid for readability
            ax.grid(True, linestyle="--", alpha=0.7)

    if not has_data:
        plt.close(fig)
        print("No data available for plotting. Please check your data files.")
        return

    # Add legend at the top
    if handles:
        # Reorder handles and labels to match the methods list order
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
            columnspacing=0.8,
            handletextpad=0.3,
            borderpad=0,
            labelspacing=0,
        )

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.25, wspace=0.18)

    # Save figure (PDF only)
    output_pdf = f"labelnum_{metric.lower()}_recall.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    print(f"Plot saved as {output_pdf}")


# Main function to parse arguments and create plots
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QPS-Recall or Cmps-Recall plots for different label numbers.')
    parser.add_argument('--metric', type=str, choices=['QPS', 'Cmps'], default='QPS',
                        help='Metric to plot on y-axis (QPS or Cmps)')
    parser.add_argument('--min-recall', type=float, default=0.7,
                        help='Minimum recall value to display')
    parser.add_argument('--methods', type=str, nargs='+', default=methods,
                        help='Methods to include in the plot')
    parser.add_argument('--label-nums', type=int, nargs='+', default=label_nums,
                        help='Label numbers to include in the plot')
    parser.add_argument('--query-types', type=str, nargs='+', default=query_types,
                        help='Query types to include (default: equality containment overlap)')
    parser.add_argument('--dataset', type=str, default='LAION',
                        help='Dataset to use (default: LAION)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename prefix (default: labelnum_{metric}_recall)')
    
    args = parser.parse_args()
    
    # Update global variables based on arguments
    metric = args.metric
    minimum_recall = args.min_recall
    methods = args.methods
    label_nums = args.label_nums
    query_types = args.query_types
    dataset = args.dataset
    
    # Print configuration for user information
    print(f"Plotting {metric}-Recall curves")
    print(f"Dataset: {dataset}")
    print(f"Label numbers: {label_nums}")
    print(f"Query types: {query_types}")
    print(f"Methods: {methods}")
    print(f"Minimum recall: {minimum_recall}")
    
    # Create the plot
    create_plot(metric=metric)
    
    # If output name specified, rename file
    # if args.output:
    #     output_base = args.output
    #     import shutil
    #     shutil.move(f"labelnum_{metric.lower()}_recall.pdf", f"{output_base}.pdf")
    #     print(f"Plot renamed to {output_base}.pdf")
