import os
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# Color palette (matching plot_ivf_budget.py style)
colors = {
    "k1": "#f27b50",
    "k10": "#b6042a",
    "k50": "#730220",
    "k100": "#88a2c4",
    "k500": "#5e62a9",
    "thread1": "#f27b50",
    "thread2": "#b6042a",
    "thread4": "#730220",
    "thread8": "#88a2c4",
    "thread16": "#5e62a9"
}

markers = {
    "k1": "o",
    "k10": "s",
    "k50": "^",
    "k100": "D",
    "k500": "v",
    "thread1": "o",
    "thread2": "s",
    "thread4": "^",
    "thread8": "D",
    "thread16": "v"
}

# Parameter display names
k_params = ["k1", "k10", "k50", "k100", "k500"]
thread_params = ["thread1", "thread2", "thread4", "thread8", "thread16"]

def extract_param_value(filename, param_type):
    """Extract parameter value from filename"""
    if param_type == "k":
        match = re.search(r'k(\d+)', filename)
        if match:
            return f"k{match.group(1)}"
    elif param_type == "thread":
        match = re.search(r'thread(\d+)', filename)
        if match:
            return f"thread{match.group(1)}"
    return None

def load_data(directory, scenario, param_type):
    """
    Load QPS vs Recall data for all parameter values
    Returns dict: {param_value: (recall_array, qps_array)}
    """
    data = {}
    
    # Find all matching files for this scenario and param_type
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return data
    
    for filename in os.listdir(directory):
        if not filename.startswith("poindex_minhash_") or not filename.endswith(".csv"):
            continue
        
        # Check if it matches the scenario
        if f"_{scenario}_" not in filename:
            continue
        
        # Extract parameter value
        param = extract_param_value(filename, param_type)
        if param is None:
            continue
        
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find header line
            header_idx = -1
            for i, line in enumerate(lines):
                if 'ef' in line and 'recall' in line and 'qps' in line:
                    header_idx = i
                    break
            
            # Parse data
            if header_idx >= 0:
                data_lines = lines[header_idx + 1:]
            else:
                data_lines = lines
            
            data_text = '\n'.join([line.strip() for line in data_lines if line.strip()])
            parsed_data = np.genfromtxt(data_text.split('\n'), delimiter=',', skip_header=0)
            
            if parsed_data.size == 0:
                continue
            
            if parsed_data.ndim == 1:
                parsed_data = parsed_data.reshape(1, -1)
            
            # Columns: ef, recall, qps, dist_comp, hops
            recall = parsed_data[:, 1]
            qps = parsed_data[:, 2]
            
            # Sort by recall
            sort_idx = np.argsort(recall)
            recall = recall[sort_idx]
            qps = qps[sort_idx]
            
            data[param] = (recall, qps)
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return data

def plot_scenario(ax, directory, scenario, param_type, params):
    """Plot QPS vs Recall for one scenario and parameter type"""
    
    data = load_data(directory, scenario, param_type)
    
    if not data:
        print(f"No data found for {scenario} with param_type {param_type}")
        return
    
    # Plot curves for each parameter value
    for param in params:
        if param in data:
            recall, qps = data[param]
            
            # Extract numeric value for label
            if param_type == "k":
                label_val = param.replace("k", "")
            else:  # thread
                label_val = param.replace("thread", "")
            
            ax.plot(recall, qps,
                   color=colors[param],
                   marker=markers[param],
                   linestyle='-',
                   linewidth=2.5,
                   markersize=12,
                   markerfacecolor="none",
                   label=f"${param_type}$={label_val}")
    
    ax.set_xlabel("Recall", fontsize=26, labelpad=0)
    ax.set_ylabel("QPS", fontsize=26, labelpad=0)
    ax.tick_params(axis='both', labelsize=26, pad=1)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(0.8, 1.0)
    
    # Legend in two bottom-aligned columns: first column (3 items), second column (2 items)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 5:
        # First column: 3 legends
        leg1 = ax.legend(handles[:3], labels[:3], fontsize=24, loc="lower left",
                         frameon=False, bbox_to_anchor=(0, -0.05), borderaxespad=0.1, labelspacing=0.3, handletextpad=0.3)
        ax.add_artist(leg1)
        # Second column: 2 legends, bottom-aligned via same 'lower left' with x offset
        ax.legend(handles[3:], labels[3:], fontsize=24, loc="lower left",
                  frameon=False, bbox_to_anchor=(0.38, -0.05), borderaxespad=0.1,
                  labelspacing=0.3, handletextpad=0.3)
    else:
        # Fallback: single legend
        ax.legend(fontsize=24, loc="lower left", frameon=False)

def main():
    # Create 1×2 subplot figure with reduced height
    fig = plt.figure(figsize=(16, 3.8))
    gs = fig.add_gridspec(1, 2, hspace=0.2, wspace=0.17, top=0.92)
    
    # Subplot 0: Varying k (LAION dataset, containment scenario)
    ax0 = fig.add_subplot(gs[0, 0])
    plot_scenario(ax0, "/root/work/lssg/example/plot/hyperparam/LAION/k", 
                 "containment", "k", k_params)
    # X label: Recall@$k$
    ax0.set_xlabel(r"Recall@$k$", fontsize=30)
    
    # Subplot 1: Varying thread number (LAION dataset, containment scenario)
    ax1 = fig.add_subplot(gs[0, 1])
    plot_scenario(ax1, "/root/work/lssg/example/plot/hyperparam/LAION/thread",
                 "containment", "thread", thread_params)
    # X label: Recall@10
    ax1.set_xlabel("Recall@10", fontsize=30)
    
    # Remove titles as requested (ensure none set)
    ax0.set_title("Varying $k$", fontsize=32,pad=10)
    ax1.set_title("Varying Thread Number", fontsize=32,pad=10)
    
    # Save figure
    output_file = "plot_k_thread_query.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
