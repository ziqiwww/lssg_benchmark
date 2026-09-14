import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# List of datasets (all 7 from plot_label_qps_recall.py)
# Note: Use the actual directory names for quantile data
db_list = ["sift1m", "gist1m", "tripclick", "LAION1M", "ytb_video", "ytb_audio", "yfcc"]
# db_list = ["gist1m", "LAION1M", "ytb_video", "ytb_audio"]
db_list = ["sift1m", "tripclick", "yfcc"]

# List of quantiles
quantiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
percentiles = [10, 30, 50, 70, 90]  # Corresponding percentiles

# Scenarios
scenarios = ["equality", "containment", "overlap"]

# List of methods
methods = ["poindex", "poindex_minhash", "acorn", "ung", "eli"]

# Colors and markers for each method
color_map = {
    "poindex": "#b60033",  # Red
    "poindex_minhash": "#ff7f0e",  # Orange
    "ung": "#27b9d6",      # Blue
    "eli": "#2ca02c",      # Green
    "oracle": "#797877",   # Gray
    "acorn": "#6b3194",    # Purple
    "packing": "#afd613",  # Yellow-green
    "rwalks": "#EC81E5FF",  # Pink
    "prefilter": "#FFC800",  # Gold/Yellow
}

marker_map = {
    "poindex": "o",  # Circle
    "poindex_minhash": "o",  # Circle (same shape, different color)
    "ung": "^",      # Triangle
    "eli": "s",      # Square
    "oracle": "D",   # Diamond
    "acorn": "v",    # Inverted triangle
    "packing": "p",  # Pentagon
    "rwalks": "X",   # X
    "prefilter": "*",  # Star
}

# Method name mapping for display in plots
method_display_map = {
    "poindex_minhash": "LSSG-MinHash",
    "poindex": "LSSG-IVF",
    "ung": "UNG",
    "eli": "ELI",
    "oracle": "Oracle",
    "acorn": "ACORN",
    "packing": "Packing",
    "rwalks": "RWalks",
    "prefilter": "Pre-filter",
}


def get_method_display_name(method):
    """Get the display name for a method"""
    return method_display_map.get(method, method)


def get_method_plot_order():
    """
    Get the plotting order for methods.
    Methods are plotted in the order they appear in this list.
    Later methods are drawn on top of earlier methods.
    
    IMPORTANT: To ensure a method appears on top of another:
    - Place it LATER in the list
    - Example: poindex appears after poindex_minhash, so poindex is on top
    
    Returns:
        list: Ordered list of methods for plotting
    """
    # Plot order: earlier methods are drawn first (bottom layer)
    #             later methods are drawn last (top layer)
    return [
        "acorn",            # Bottom layer
        "ung",
        "eli",
        "poindex_minhash",  # Draw MinHash before IVF
        "poindex",          # IVF on top (drawn last, appears on top)
    ]


def load_quantile_data(db, scenario, quantile, method):
    """
    Load data from quantile result files
    Returns all data points (recall, qps, cmps) for plotting curves
    
    Directory structure: {method_dir}_{dataset}_{scenario}/
    - ivf = poindex
    - minhash = poindex_minhash
    - ung has subdirectories per quantile
    """
    # Map methods to their directory names
    if method == "poindex":
        method_dir = "ivf"
        file_name = f"poindex_{scenario}_{quantile}.csv"
    elif method == "poindex_minhash":
        method_dir = "minhash"
        file_name = f"poindex_{scenario}_{quantile}.csv"
    elif method == "ung":
        # UNG has a different structure with subdirectories
        method_dir = "ung"
        base_path = f"/root/work/lssg/example/plot/quantile/{method_dir}_{db}_{scenario}"
        if not os.path.exists(base_path):
            return None
        subdirs = [d for d in os.listdir(base_path) if d.startswith(quantile)]
        if not subdirs:
            return None
        file_path = os.path.join(base_path, subdirs[0], "result.csv")
    elif method == "eli":
        # method_dir = "eli"
        # file_name = f"{quantile}-labelann-{scenario}-0.20.csv"
        method_dir = "eli"
        file_name = f"{quantile}-labelann-{scenario}-0.2.csv"
    elif method == "acorn":
        method_dir = "acorn"
        file_name = f"{quantile}-acorn-{scenario}-M16_Mb32_gamma30.csv"
    else:
        return None
    
    # Construct full file path (except for ung which was already set)
    if method != "ung":
        file_path = f"/root/work/lssg/example/plot/quantile/{method_dir}_{db}_{scenario}/{file_name}"
    
    # Check if file exists
    if not os.path.exists(file_path):
        return None
    
    try:
        if method == "poindex" or method == "poindex_minhash":
            # Format: L,Recall,QPS,Cmps,hops
            # IVF files: no header
            # MinHash files: debug output + header line "ef,recall,qps,dist_comp,hops"
            
            # Read file and skip non-numeric lines (debug output and headers)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find where the actual CSV data starts
            data_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Check if line starts with a number (actual data)
                if line[0].isdigit():
                    # Skip the ground truth line (has many comma-separated IDs)
                    parts = line.split(',')
                    # Data lines should have exactly 5 columns
                    if len(parts) == 5:
                        try:
                            # Verify it's numeric data
                            float(parts[0])
                            float(parts[1])
                            data_lines.append(line)
                        except ValueError:
                            continue
            
            if not data_lines:
                return None
            
            # Parse the numeric data
            data = np.array([[float(x) for x in line.split(',')] for line in data_lines])
            
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
            result['L'] = data[:, 0]
            result['Recall'] = data[:, 1]
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 3]
            return result
        
        elif method == "ung":
            # Format: L,Cmps,QPS,Recall (with header)
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            if data.size == 0:
                return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            recall = data[:, 3]
            if np.any(recall > 1):
                recall = recall / 100.0
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0]
            result['Recall'] = recall
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 1]
            return result
        
        elif method == "eli":
            # Format: ef,recall,qps,distance_computations_per_query (with header)
              
            # Read file and skip non-numeric lines (debug output and headers)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find where the actual CSV data starts
            data_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Check if line starts with a number (actual data)
                if line[0].isdigit():
                    parts = line.split(',')
                    # Data lines should have exactly 4 columns
                    if len(parts) == 4:
                        try:
                            # Verify it's numeric data
                            float(parts[0])
                            float(parts[1])
                            data_lines.append(line)
                        except ValueError:
                            continue
            
            if not data_lines:
                return None
            
            # Parse the numeric data
            data = np.array([[float(x) for x in line.split(',')] for line in data_lines])
            
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
        
        elif method == "acorn":
            # Format: efSearch,recall@10,QPS,QPS_with_filter,dist_comps_per_query (with header)
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
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
            result['Cmps'] = data[:, 4]
            return result
        
        else:
            return None
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_selectivity_data(db, scenario):
    base_path = "/root/work/lssg/example/quantile_partitions"
    file_path = os.path.join(base_path, f"{db}-quantile-{scenario}.csv")
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_best_qps_at_recall(data, min_recall=0.95):
    """
    Get the highest QPS where recall >= min_recall
    Returns (qps, actual_recall_used) tuple
    Returns (None, None) if no data available
    """
    if data is None:
        return None, None
    
    # Filter by minimum recall
    mask = data['Recall'] >= min_recall
    if not any(mask):
        return None, None
    
    filtered_data = data[mask]
    
    # Find the point with highest QPS
    max_qps_idx = np.argmax(filtered_data['QPS'])
    return filtered_data['QPS'][max_qps_idx], min_recall


def find_adaptive_recall_threshold(db, scenario, quantiles, methods_to_check, default_min_recall=0.95):
    """
    Find the minimum recall threshold that allows plotting at all percentiles for ALL specified methods.
    Start from default and decrease as soon as ANY method lacks data at ANY percentile.
    Try recall thresholds from default down to 0.70 in steps of 0.05.
    Returns the adjusted threshold that gives data for all percentiles for all methods, or default if all methods work.
    """
    # First check if default threshold works for all methods at all percentiles
    default_works = True
    for method in methods_to_check:
        for quantile in quantiles:
            data = load_quantile_data(db, scenario, quantile, method)
            qps, _ = get_best_qps_at_recall(data, default_min_recall)
            if qps is None:
                default_works = False
                break
        if not default_works:
            break
    
    # If default works, return it immediately
    if default_works:
        return default_min_recall
    
    # Default doesn't work, so find the lowest threshold that does work
    test_thresholds = list(np.arange(0.90, 0.75, -0.05))
    
    for threshold in test_thresholds:
        all_methods_have_data = True
        
        for method in methods_to_check:
            all_have_data = True
            for quantile in quantiles:
                data = load_quantile_data(db, scenario, quantile, method)
                qps, _ = get_best_qps_at_recall(data, threshold)
                if qps is None:
                    all_have_data = False
                    break
            
            if not all_have_data:
                all_methods_have_data = False
                break
        
        if all_methods_have_data:
            return threshold
    
    # If standard thresholds don't work, find the actual maximum recall achievable
    # This handles cases where recall is below 0.80
    max_min_recall = 0.0
    for method in methods_to_check:
        method_min_recall = 1.0
        for quantile in quantiles:
            data = load_quantile_data(db, scenario, quantile, method)
            if data is not None and len(data) > 0:
                quantile_max_recall = np.max(data['Recall'])
                method_min_recall = min(method_min_recall, quantile_max_recall)
        max_min_recall = max(max_min_recall, method_min_recall)
    
    # Return the maximum recall achievable (rounded down to nearest 0.05)
    if max_min_recall > 0:
        adjusted = np.floor(max_min_recall * 20) / 20  # Round down to nearest 0.05
        return max(adjusted, 0.70)  # Don't go below 0.70
    
    # If no data at all, return default
    return default_min_recall


def dbname_beautify(db):
    """Convert database names to more readable format"""
    if db == "sift" or db == "sift1m":
        return "SIFT"
    elif db == "gist" or db == "gist1m":
        return "GIST"
    elif db == "laion" or db == "laion1m" or db == "LAION1M":
        return "LAION"
    elif db == "tripclick":
        return "TripClick"
    elif db == "ytb_video":
        return "YTB-Video"
    elif db == "ytb_audio":
        return "YTB-Audio"
    elif db == "yfcc":
        return "YFCC-1M"
    else:
        return db.upper()


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


from matplotlib.ticker import LogFormatterMathtext
class CustomLogFormatter(LogFormatterMathtext):
    def __call__(self, x, pos=None):
        if x == 1:
            return "1"
        return super().__call__(x, pos)

def minus_x_formatter(x, pos=None):
    if x == 1:
        return "0"
    try:
        exp = int(np.log10(x))
    except Exception:
        return ""
    return f"{exp}" if x != 0 else "0"

def plot_selectivity_statistics(ax, db, show_ylabel=False, show_xlabel=False):
    quantiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    percentiles = [10, 30, 50, 70, 90]
    scenarios = ["equality", "containment", "overlap"]
    scenario_colors = {
        "equality": "#730220",
        "containment": "#f27b50",
        "overlap": "#5e62a9",
    }
    scenario_display_map = {
        "equality": "Equality",
        "containment": "Containment",
        "overlap": "Overlap",
    }
    bar_width = 0.25
    x_positions = np.arange(len(percentiles))
    data_by_scenario = {}
    for scenario in scenarios:
        df = load_selectivity_data(db, scenario)
        if df is not None and len(df) == len(quantiles):
            data_by_scenario[scenario] = df
    if not data_by_scenario:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    for i, scenario in enumerate(scenarios):
        if scenario not in data_by_scenario:
            continue
        df = data_by_scenario[scenario]
        avg_values = df['avg_spec'].values/100
        low_values = df['low_spec'].values/100
        high_values = df['high_spec'].values/100
        yerr_lower = avg_values - low_values
        yerr_upper = high_values - avg_values
        yerr = np.array([yerr_lower, yerr_upper])
        positions = x_positions + (i - 1) * bar_width
        ax.bar(
            positions,
            avg_values,
            bar_width,
            label=scenario_display_map[scenario],
            color=scenario_colors[scenario],
            edgecolor='black',
            linewidth=1.5,
            yerr=yerr,
            capsize=4,
            error_kw={'linewidth': 1.5, 'ecolor': 'black'}
        )
    # Only show y label on first column
    if show_ylabel:
        ax.set_ylabel("Avg. Filter Ratio", fontsize=24, labelpad=0)
    # Only show x label on last row
    if show_xlabel:
        ax.set_xlabel("Percentile", fontsize=24, labelpad=0)
    # Title for last row can be omitted for consistency
    # X axis format same as previous rows
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p}\\%" for p in percentiles], fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, linestyle="--", alpha=0.7, axis='y')
    ax.set_axisbelow(True)
    all_values = []
    for df in data_by_scenario.values():
        all_values.extend(df['avg_spec'].values)
    if len(all_values) > 0:
        min_val = min(all_values)
        max_val = max(all_values)
        if max_val / min_val > 10:
            ax.set_yscale('log')
    
    # Format y-ticks as plain numbers and rotate 45 degrees
    from matplotlib.ticker import FuncFormatter
    def plain_fmt(x, pos):
        if x >= 1:
            return f"{int(x)}"
        else:
            return f"{x:.4f}".rstrip('0').rstrip('.')
    ax.yaxis.set_major_formatter(FuncFormatter(plain_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(minus_x_formatter))
    ax.tick_params(axis='y', labelsize=15, pad=0)
    for label in ax.get_yticklabels():
        label.set_rotation(45)
    # Use custom log formatter for y-ticks
    ax.yaxis.set_major_formatter(CustomLogFormatter())
    # ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_rotation(45)


def create_specificity_qps_plot(min_recall=0.95):
    scenario_colors = {"equality": "#730220", "containment": "#f27b50", "overlap": "#5e62a9"}
    scenario_display_map = {"equality": "Equality", "containment": "Containment", "overlap": "Overlap"}
    import matplotlib.patches as mpatches
    bar_handles = [mpatches.Patch(facecolor=scenario_colors[sc], edgecolor='black', label=scenario_display_map[sc], linewidth=1.5) for sc in ["equality", "containment", "overlap"]]
    # Create figure with subplots
    if len(db_list) == 3:
        figsize = (len(db_list) * 4, 4 * 2.4)
    else:
        figsize = (len(db_list) * 3.5, 4 * 2.8)
    fig, axs = plt.subplots(
        4, len(db_list),
        figsize=figsize,
        dpi=150
    )
    
    # Handle case of single subplot
    if len(scenarios) == 1 and len(db_list) == 1:
        axs = np.array([[axs]])
    elif len(scenarios) == 1:
        axs = np.array([axs])
    elif len(db_list) == 1:
        axs = np.array([[ax] for ax in axs])
    
    handles = []
    pushed_methods = []
    has_data = False
    
    # Track adaptive thresholds for annotations - one per subplot
    adaptive_thresholds = {}  # Key: (scenario, db), Value: threshold (single value)
    
    # First pass: determine adaptive thresholds for each subplot
    for i, scenario in enumerate(scenarios):
        for j, db in enumerate(db_list):
            # Check which poindex methods are present
            alps_methods = [m for m in ["poindex", "poindex_minhash"] if m in methods]
            
            if alps_methods:
                # Find threshold that works for all Alps methods in this subplot
                threshold = find_adaptive_recall_threshold(db, scenario, quantiles, alps_methods, min_recall)
                if scenario == "overlap" and db == "ytb_audio":
                    threshold = 0.8
                if threshold < min_recall:
                    adaptive_thresholds[(scenario, db)] = threshold
    
    # For each scenario and dataset
    for i, scenario in enumerate(scenarios):
        for j, db in enumerate(db_list):
            ax = axs[i, j]
            
            # Determine the effective recall threshold for this subplot (same for all methods)
            effective_recall = adaptive_thresholds.get((scenario, db), min_recall)
            
            # Get plotting order (methods plotted later appear on top)
            plot_order = get_method_plot_order()
            
            # Filter to only include methods that are in the current methods list
            methods_to_plot = [m for m in plot_order if m in methods]
            
            # Plot each method in the specified order
            for method in methods_to_plot:
                qps_values = []
                valid_percentiles = []
                
                # Collect QPS values for each quantile using the subplot's effective recall
                for quantile, percentile in zip(quantiles, percentiles):
                    data = load_quantile_data(db, scenario, quantile, method)
                    best_qps, _ = get_best_qps_at_recall(data, effective_recall)
                    
                    if best_qps is not None:
                        qps_values.append(best_qps)
                        valid_percentiles.append(percentile)
                
                # Plot if we have data
                if len(qps_values) > 0:
                    line, = ax.plot(
                        valid_percentiles, qps_values,
                        label=get_method_display_name(method),
                        color=color_map.get(method, "black"),
                        marker=marker_map.get(method, "o"),
                        markersize=12,
                        linewidth=4,
                        markerfacecolor="white",
                        markeredgewidth=2
                    )
                    
                    # Add to legend if not already added
                    method_display = get_method_display_name(method)
                    if method_display not in pushed_methods:
                        pushed_methods.append(method_display)
                        handles.append(line)
                    
                    has_data = True
            
            # Add annotation if using adaptive threshold (single annotation per subplot)
            if (scenario, db) in adaptive_thresholds:
                threshold = adaptive_thresholds[(scenario, db)]
                annotation_text = f"R$=${threshold:.2f}"
                
                # Add annotation in top-right corner with lighter background
                ax.text(0.98, 0.98, annotation_text,
                       transform=ax.transAxes,
                       fontsize=18,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.3))
            
            # Configure subplot
            # Title only on top row
            if i == 0:
                ax.set_title(f"{dbname_beautify(db)}", fontsize=24)
            
            # Y-axis label only on leftmost column
            if j == 0:
                ax.set_ylabel(f"{scenario_beautify(scenario)}\nQPS", fontsize=24, labelpad=0)
            
            # X-axis label only on bottom row
            if i == len(scenarios) - 1:
                ax.set_xlabel("Percentile", fontsize=24, labelpad=0)
            
            # Set y-axis to logarithmic scale
            ax.set_yscale("log")
            
            # Keep only major ticks on y-axis
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            
            # Set x-axis limits and ticks
            ax.set_xlim(5, 95)
            ax.set_xticks(percentiles)
            ax.set_xticklabels([f"{p}\\%" for p in percentiles], fontsize=18)
            
            # Configure ticks
            ax.tick_params(axis="both", which="major", labelsize=18)
            
            # Add grid for readability
            ax.grid(True, linestyle="--", alpha=0.7)
    
    # Add selectivity statistics in the 4th row
    for j, db in enumerate(db_list):
        show_ylabel = (j == 0)
        show_xlabel = True
        plot_selectivity_statistics(axs[3, j], db, show_ylabel=show_ylabel, show_xlabel=show_xlabel)
    
    if not has_data:
        plt.close(fig)
        print(f"No data available. Please check your data files.")
        return
    
    # Add legend at the top
    if handles:
        # Reorder handles to match methods list
        method_order = {get_method_display_name(m): i for i, m in enumerate(methods)}
        sorted_pairs = sorted(zip(pushed_methods, handles), key=lambda x: method_order.get(x[0], 999))
        pushed_methods_sorted, handles_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        # Combine method and bar handles/labels
        all_handles = list(handles_sorted) + bar_handles
        all_labels = list(pushed_methods_sorted) + [scenario_display_map[sc] for sc in ["equality", "containment", "overlap"]]
        # columns = len(all_labels) if all_labels else 1
        columns = len(all_labels) / 2 if methods else 1
        fig.legend(
            handles=all_handles,
            labels=all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.07),
            ncol=columns,
            fontsize=26,
            frameon=False,
            handletextpad=0.2,
            borderpad=0,
            labelspacing=0,
        )
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.15, wspace=0.15)
    
    # Save figure
    output_pdf = f"specificity_qps_recall{int(min_recall*100)}.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    print(f"Plot saved as {output_pdf}")


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QPS vs Specificity plots across scenarios and datasets.')
    parser.add_argument('--min-recall', type=float, default=0.95,
                        help='Minimum recall threshold for selecting QPS points (default: 0.95)')
    parser.add_argument('--methods', type=str, nargs='+', default=methods,
                        help='Methods to include in the plot')
    parser.add_argument('--datasets', type=str, nargs='+', default=db_list,
                        help='Datasets to include in the plot')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    # Update global variables
    methods = args.methods
    db_list = args.datasets
    
    # Print configuration
    print(f"Plotting QPS vs Specificity")
    print(f"Scenarios: {scenarios}")
    print(f"Datasets: {db_list}")
    print(f"Methods: {methods}")
    print(f"Minimum recall threshold: {args.min_recall}")
    
    # Create the plot
    create_specificity_qps_plot(min_recall=args.min_recall)
    
    # Rename output if specified
    if args.output:
        import shutil
        src = f"specificity_qps_recall{int(args.min_recall*100)}.pdf"
        dst = f"{args.output}.pdf"
        shutil.move(src, dst)
        print(f"Plot renamed to {dst}")
