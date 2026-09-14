import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# List of datasets
db_list = ["sift1m", "gist1m", "LAION1M", "tripclick", "ytb_video", "ytb_audio", "yfcc"]

# Quantiles and percentiles
quantiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
percentiles = [10, 30, 50, 70, 90]

# Scenarios
scenarios = ["equality", "containment", "overlap"]

# Colors for each scenario
scenario_colors = {
    "equality": "#b60033",      # Red
    "containment": "#27b9d6",   # Blue
    "overlap": "#2ca02c",       # Green
}

# Scenario name mapping for display
scenario_display_map = {
    "equality": "Equality",
    "containment": "Containment",
    "overlap": "Overlap",
}


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


def load_selectivity_data(db, scenario):
    """
    Load selectivity statistics from quantile partition files
    Returns: DataFrame with columns [quantile, avg_spec, low_spec, high_spec]
    """
    base_path = "/root/work/lssg/example/quantile_partitions"
    file_path = os.path.join(base_path, f"{db}-quantile-{scenario}.csv")
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def create_selectivity_plot():
    """
    Create selectivity statistics plot
    - First row: First 3 datasets (sift1m, gist1m, LAION1M)
    - Second row: Last 4 datasets (tripclick, ytb_video, ytb_audio, yfcc)
    - X-axis: Percentiles (10%, 30%, 50%, 70%, 90%)
    - Y-axis: Average filter ratio (specificity)
    - Three bars per percentile (one for each scenario)
    - Error bars showing min and max values
    """
    # Split datasets into two rows
    row1_datasets = db_list[:3]  # sift1m, gist1m, LAION1M
    row2_datasets = db_list[3:]  # tripclick, ytb_video, ytb_audio, yfcc
    
    # Create figure with 2 rows
    fig, axs = plt.subplots(
        2, max(len(row1_datasets), len(row2_datasets)),
        figsize=(12, 6),
        dpi=150
    )
    
    # Ensure axs is 2D
    if len(row1_datasets) == 1 and len(row2_datasets) == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    
    # Plot first row
    for j, db in enumerate(row1_datasets):
        ax = axs[0, j]
        plot_dataset_selectivity(ax, db, j == 0, True)
    
    # Hide unused subplots in first row
    for j in range(len(row1_datasets), axs.shape[1]):
        axs[0, j].axis('off')
    
    # Plot second row
    for j, db in enumerate(row2_datasets):
        ax = axs[1, j]
        plot_dataset_selectivity(ax, db, j == 0, False)
    
    # Hide unused subplots in second row (if any)
    for j in range(len(row2_datasets), axs.shape[1]):
        axs[1, j].axis('off')
    
    # Add legend at the top
    handles = []
    labels = []
    for scenario in scenarios:
        from matplotlib.patches import Rectangle
        handle = Rectangle((0, 0), 1, 1, fc=scenario_colors[scenario], edgecolor='black', linewidth=1.5)
        handles.append(handle)
        labels.append(scenario_display_map[scenario])
    
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(scenarios),
        fontsize=20,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.92)
    
    # Save figure
    output_pdf = "/root/work/lssg/example/plot/selectivity_stats.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", dpi=300)
    print(f"Plot saved as {output_pdf}")


def plot_dataset_selectivity(ax, db, show_ylabel, show_title):
    """
    Plot selectivity statistics for a single dataset
    """
    # Bar width and positions
    bar_width = 0.25
    x_positions = np.arange(len(percentiles))
    
    # Collect data for all scenarios
    data_by_scenario = {}
    for scenario in scenarios:
        df = load_selectivity_data(db, scenario)
        if df is not None and len(df) == len(quantiles):
            data_by_scenario[scenario] = df
    
    if not data_by_scenario:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    # Plot bars for each scenario
    for i, scenario in enumerate(scenarios):
        if scenario not in data_by_scenario:
            continue
        
        df = data_by_scenario[scenario]
        
        # Extract values
        avg_values = df['avg_spec'].values
        low_values = df['low_spec'].values
        high_values = df['high_spec'].values
        
        # Calculate error bars (distance from average to min/max)
        yerr_lower = avg_values - low_values
        yerr_upper = high_values - avg_values
        yerr = np.array([yerr_lower, yerr_upper])
        
        # Bar positions for this scenario
        positions = x_positions + (i - 1) * bar_width
        
        # Plot bars with error bars
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
    
    # Configure subplot
    if show_title:
        ax.set_title(dbname_beautify(db), fontsize=20)
    else:
        ax.set_xlabel(dbname_beautify(db), fontsize=18)
    
    if show_ylabel:
        ax.set_ylabel("Avg. Filter Ratio", fontsize=18)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{p}\\%" for p in percentiles], fontsize=14)
    
    # Configure ticks
    ax.tick_params(axis="both", which="major", labelsize=14)
    
    # Add grid for readability
    ax.grid(True, linestyle="--", alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis to log scale if values span multiple orders of magnitude
    if data_by_scenario:
        all_values = []
        for df in data_by_scenario.values():
            all_values.extend(df['avg_spec'].values)
        
        if len(all_values) > 0:
            min_val = min(all_values)
            max_val = max(all_values)
            if max_val / min_val > 10:  # If ratio > 10, use log scale
                ax.set_yscale('log')


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate selectivity statistics plots across scenarios and datasets.')
    parser.add_argument('--datasets', type=str, nargs='+', default=db_list,
                        help='Datasets to include in the plot')
    parser.add_argument('--output', type=str, default='selectivity_stats',
                        help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    # Update global variables if specified
    if args.datasets:
        db_list = args.datasets
    
    # Print configuration
    print(f"Plotting Selectivity Statistics")
    print(f"Scenarios: {scenarios}")
    print(f"Datasets: {db_list}")
    print(f"Quantiles: {quantiles}")
    
    # Create the plot
    create_selectivity_plot()
    
    # Rename output if specified and different from default
    if args.output != 'selectivity_stats':
        import shutil
        src = "selectivity_stats.pdf"
        dst = f"{args.output}.pdf"
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Plot renamed to {dst}")
