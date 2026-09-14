import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# List of datasets
# db_list = ["sift", "gist", "laion", "tripclick",  "ytb_video", "ytb_audio","yfcc"]
# db_list = [ "gist", "laion", "ytb_video", "ytb_audio"]
db_list = [ "sift", "tripclick", "yfcc"]

# List of query types
query_types = ["equality", "containment", "overlap"]

# List of methods
methods = ["poindex", "poindex_minhash", "acorn", "packing", "rwalks", "ung", "eli",  "oracle", "prefilter"]
# methods = ["poindex", "ung", "eli", "packing", "rwalks", "acorn", "oracle", "prefilter"]

# Colors and markers for each method
color_map = {
    "poindex": "#b60033",  # Red
    "poindex_minhash": "#ff7f0e",  # Dark Red
    "ung": "#27b9d6",      # Blue
    "eli": "#2ca02c",      # Green
    "oracle": "#797877",   # Orange
    "acorn": "#6b3194",  # Brown
    "packing": "#afd613",  # Crimson
    "rwalks": "#EC81E5FF",  # Green
    "prefilter": "#FFC800",  # Gold/Yellow
    # Add more methods as needed
}

marker_map = {
    "poindex": "o",  # Circle
    "poindex_minhash": "o",  # Circle (same shape, different color)
    "ung": "^",      # Triangle
    "eli": "s",      # Square
    "oracle": "D",   # Diamond
    "acorn": "v",  # Inverted triangle (same shape, different color)
    "packing": "p",  # Pentagon
    "rwalks": "X",  # X
    "prefilter": "*",  # Star
    # Add more methods as needed
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
    "oracle": "Oracle",
    "acorn": "ACORN",
    "packing": "Packing",
    "rwalks": "RWalks",
    "prefilter": "Pre-filter",
    # Add more custom mappings as needed
}


def get_method_display_name(method):
    """
    Get the display name for a method
    """
    return method_display_map.get(method, method)


def load_data(db, query_type, method):
    """
    Load data from the result files
    """
    if method == "poindex":
        # poindex format: L,Recall,QPS,Cmps,hops (no header)
        file_path = f"/root/work/lssg/example/plot/{db}/label/{method}_{query_type}.csv"
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
        # poindex_minhash format: L,Recall,QPS,Cmps,hops (no header)
        file_path = f"/root/work/lssg/example/plot/{db}/label/{method}_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            data = np.loadtxt(file_path, delimiter=",")
            # Convert to structured array with named fields (same format as poindex)
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
            
    elif method == "ung":
        # ung format: L,Cmps,QPS,Recall (with header)
        file_path = f"/root/work/lssg/example/plot/{db}/label/ung_{query_type}.csv"
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
            
    elif method == "eli":
        # eli format: L,recall,qps,distance_computations (no header)
        # file_path = f"/root/work/lssg/example/plot/{db}/label/esi_{query_type}.csv"
        file_path = f"/root/work/lssg/example/plot/{db}/label/esi_{query_type}.csv"
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
    elif method == "oracle":
        # oracle format: ef,recall,qps,avg_index_size,avg_distance_computations (with header)
        file_path = f"/root/work/lssg/example/plot/{db}/label/oracle_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            # Skip header row
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            # Convert to structured array with named fields
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64),
                ('IndexSize', np.float64)
                
            ])
            result['L'] = data[:, 0].astype(np.int32)  # ef_search parameter
            result['Recall'] = data[:, 1]
            result['QPS'] = data[:, 2]
            result['Cmps'] = data[:, 3]  # Average distance computations
            result['IndexSize'] = data[:, 4]  # Average index size (filtered vectors)
            return result
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    elif method == "acorn":
        # acorn format: efSearch,recall@10,QPS,QPS_with_filter,dist_comps_per_query (with header)
        file_path = f"/root/work/lssg/example/plot/{db}/label/acorn_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            # Skip header row
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            
            # Check if file is empty or has no data after header
            if data.size == 0:
                print(f"Warning: File {file_path} contains no data")
                return None
            
            # Handle case where only one row exists (becomes 1D array)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Convert to structured array with named fields
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0].astype(np.int32)  # efSearch parameter
            result['Recall'] = data[:, 1]  # recall@10
            result['QPS'] = data[:, 2]  # Use QPS (3rd column)
            result['Cmps'] = data[:, 4]  # dist_comps_per_query
            return result
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            return None
    
    elif method == "acorn-filter":
        # acorn-filter format: efSearch,recall@10,QPS,QPS_with_filter,dist_comps_per_query (with header)
        # This uses QPS_with_filter column instead of QPS
        file_path = f"/root/work/lssg/example/plot/{db}/label/acorn_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            # Skip header row
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            
            # Check if file is empty or has no data after header
            if data.size == 0:
                print(f"Warning: File {file_path} contains no data")
                return None
            
            # Handle case where only one row exists (becomes 1D array)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Convert to structured array with named fields
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0].astype(np.int32)  # efSearch parameter
            result['Recall'] = data[:, 1]  # recall@10
            result['QPS'] = data[:, 3]  # Use QPS_with_filter (4th column)
            result['Cmps'] = data[:, 4]  # dist_comps_per_query
            return result
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            return None
    
    elif method == "packing":
        # packing format: ef,recall,qps,dist_comp (with header)
        file_path = f"/root/work/lssg/example/plot/{db}/label/packing_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            # Skip header row
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            
            # Check if file is empty or has no data after header
            if data.size == 0:
                print(f"Warning: File {file_path} contains no data")
                return None
            
            # Handle case where only one row exists (becomes 1D array)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Convert to structured array with named fields
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64)
            ])
            result['L'] = data[:, 0].astype(np.int32)  # ef parameter
            result['Recall'] = data[:, 1]  # recall
            result['QPS'] = data[:, 2]  # qps
            result['Cmps'] = data[:, 3]  # dist_comp
            return result
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            return None
    
    elif method == "prefilter":
        # prefilter format: single QPS value (no header, no recall - always 1.00)
        file_path = f"/root/work/lssg/example/plot/{db}/label/prefilter_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            # Read single QPS value
            qps_value = np.loadtxt(file_path, delimiter=",")
            
            # Create single-row result with recall = 1.0
            result = np.zeros(1, dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64)
            ])
            result['L'] = 0  # N/A for prefilter
            result['Recall'] = 1.0  # Always perfect recall
            result['QPS'] = qps_value
            result['Cmps'] = 0  # N/A for prefilter
            return result
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            return None
    
    elif method == "rwalks":
        # rwalks format: ef,recall,qps,dist_comp,hops (with header)
        file_path = f"/root/work/lssg/example/plot/{db}/label/rwalks_{query_type}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            # Skip header row
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            
            # Check if file is empty or has no data after header
            if data.size == 0:
                print(f"Warning: File {file_path} contains no data")
                return None
            
            # Handle case where only one row exists (becomes 1D array)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # Convert to structured array with named fields
            result = np.zeros(data.shape[0], dtype=[
                ('L', np.int32), 
                ('Recall', np.float64), 
                ('QPS', np.float64), 
                ('Cmps', np.float64),
                ('hops', np.float64)
            ])
            result['L'] = data[:, 0].astype(np.int32)  # ef parameter
            result['Recall'] = data[:, 1]  # recall
            result['QPS'] = data[:, 2]  # qps
            result['Cmps'] = data[:, 3]  # dist_comp
            result['hops'] = data[:, 4]  # hops
            return result
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
            return None
    
    else:
        print(f"Warning: Unknown method {method}")
        return None


def dbname_beautify(db):
    """
    Convert database names to more readable format
    """
    if db == "sift" or db == "sift1m":
        return "SIFT"
    elif db == "gist" or db == "gist1m":
        return "GIST"
    elif db == "wiki":
        return "Wikidata"
    elif db == "deep10m":
        return "Deep10M"
    elif db == "laion" or db == "laion1m":
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
    Create the QPS-Recall or Cmps-Recall plot
    """
    # Create a figure with (num_query_types x num_datasets) subplots
    fig, axs = plt.subplots(
        len(query_types), len(db_list), 
        figsize=(len(db_list) * 4, len(query_types) * 2.8), 
        dpi=150
    )
    
    local_min_reacall=minimum_recall
    
    # Handle case of single subplot
    if len(query_types) == 1 and len(db_list) == 1:
        axs = np.array([[axs]])
    elif len(query_types) == 1:
        axs = np.array([axs])
    elif len(db_list) == 1:
        axs = np.array([[ax] for ax in axs])

    handles = []
    pushed_methods = []
    has_data = False  # Flag to track if any data was plotted

    # For each query type and dataset
    for i, query_type in enumerate(query_types):
        for j, db in enumerate(db_list):
            
            local_min_reacall = 0.5
            
            ax = axs[i, j]
            
            # Load and plot data for each method
            for method in methods:
                data = load_data(db, query_type, method)
                if data is None:
                    continue
                
                if metric == "QPS" and method == "oracle":
                    continue
                
                # Skip prefilter for Cmps metric (no distance computation data)
                if metric == "Cmps" and method == "prefilter":
                    continue
                
                # Keep only data points with recall >= minimum_recall
                
                mask = data['Recall'] >= local_min_reacall
                if not any(mask):
                    continue
                data = data[mask]
                
                # Sort by recall (descending) to make the plot more readable
                # Use argsort to get indices sorted by Recall
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
                # For prefilter, use scatter plot with larger star marker
                if method == "prefilter":
                    line = ax.scatter(
                        x, y,
                        label=method.upper(),
                        color=color_map.get(method, "black"),
                        marker=marker_map.get(method, "*"),
                        s=300,  # Larger size for visibility
                        edgecolors=color_map.get(method, "black"),
                        linewidths=2,
                        zorder=method_zorder(method) + 10  # Draw on top
                    )
                else:
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
                ax.set_title(dbname_beautify(db), fontsize=24)
            
            if j == 0:
                print_metric=metric
                if metric == "Cmps":
                    print_metric="DC"
                ax.set_ylabel(f"{query_type_beautify(query_type)}\n{print_metric}", fontsize=24, labelpad=0)
            
            if i == len(query_types) - 1:
                ax.set_xlabel(f"Recall@10", fontsize=24)
            
            # Set y-axis to logarithmic scale if using QPS or Cmps
            if metric in ["QPS", "Cmps"]:
                ax.set_yscale("log")
            
            # Set x-axis limits based on minimum recall value
            ax.set_xlim(local_min_reacall, 1.01)
            
            # Configure ticks
            ax.tick_params(axis="both", which="major", labelsize=20)
            ax.tick_params(axis="x", which="major", labelsize=22)
            
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
            bbox_to_anchor=(0.5, 1.09),
            ncol=len(pushed_methods_sorted) / 2 if pushed_methods_sorted else 1,
            fontsize=26,
            frameon=False,
            # columnspacing=0.5,
            handletextpad=0,
            borderpad=0,
            labelspacing=0,
        )

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.17, wspace=0.17)

    # Save figure (PDF only)
    output_pdf = f"label_{metric.lower()}_recall.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    print(f"Plot saved as {output_pdf}")


# Main function to parse arguments and create plots
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QPS-Recall or Cmps-Recall plots for label filtering methods.')
    parser.add_argument('--metric', type=str, choices=['QPS', 'Cmps'], default='QPS',
                        help='Metric to plot on y-axis (QPS or Cmps)')
    parser.add_argument('--min-recall', type=float, default=0.7,
                        help='Minimum recall value to display')
    parser.add_argument('--methods', type=str, nargs='+', default=methods,
                        help='Methods to include in the plot (default: poindex ung)')
    parser.add_argument('--datasets', type=str, nargs='+', default=db_list,
                        help='Datasets to include in the plot (default: sift gist)')
    parser.add_argument('--query-types', type=str, nargs='+', default=query_types,
                        help='Query types to include (default: equality containment overlap)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename prefix (default: label_{metric}_recall)')
    
    args = parser.parse_args()
    
    # Update global variables based on arguments
    metric = "Cmps"
    minimum_recall = args.min_recall
    methods = args.methods
    db_list = args.datasets
    query_types = args.query_types
    
    # Print configuration for user information
    print(f"Plotting {metric}-Recall curves")
    print(f"Datasets: {db_list}")
    print(f"Query types: {query_types}")
    print(f"Methods: {methods}")
    print(f"Minimum recall: {minimum_recall}")
    
    # Create the plot
    create_plot(metric=metric)
    
    # If output name specified, rename file
    if args.output:
        output_base = args.output
        import shutil
        shutil.move(f"label_{metric.lower()}_recall.pdf", f"{output_base}.pdf")
        print(f"Plot renamed to {output_base}.pdf")