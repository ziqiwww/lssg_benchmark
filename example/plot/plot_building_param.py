import os
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True

# Color palette for different parameter values
colors_map = {
    "T": ["#f27b50", "#b6042a", "#730220", "#88a2c4", "#5e62a9"],
    "efc": ["#f27b50", "#b6042a", "#730220", "#88a2c4", "#5e62a9"],
    "m": ["#f27b50", "#b6042a", "#730220", "#88a2c4", "#5e62a9"]
}

markers_map = {
    "T": ["o", "s", "^", "D", "v"],
    "efc": ["o", "s", "^", "D", "v"],
    "m": ["o", "s", "^", "D", "v"]
}

def get_param_latex_label(param_type):
    """Convert parameter type to LaTeX label"""
    latex_map = {
        "T": r"$T$",
        "efc": r"$\omega_c$",
        "m": r"$m$"
    }
    return latex_map.get(param_type, param_type)

def extract_param_from_filename(filename, param_type):
    """Extract parameter value from filename"""
    if param_type == "T":
        match = re.search(r'T(\d+)', filename)
        if match:
            return int(match.group(1))
    elif param_type == "efc":
        match = re.search(r'efc(\d+)', filename)
        if match:
            return int(match.group(1))
    elif param_type == "m":
        match = re.search(r'm(\d+)', filename)
        if match:
            return int(match.group(1))
    return None

def load_query_data(directory, scenario, param_type):
    """
    Load QPS vs Recall data for containment scenario
    Returns list of tuples: [(param_val, recall, qps), ...]
    """
    data = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return data
    
    for filename in os.listdir(directory):
        if not filename.startswith("poindex_minhash_") or not filename.endswith(".csv"):
            continue
        if f"_{scenario}_" not in filename:
            continue
        
        param_val = extract_param_from_filename(filename, param_type)
        if param_val is None:
            continue
        
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find header
            header_idx = -1
            for i, line in enumerate(lines):
                if 'ef' in line and 'recall' in line and 'qps' in line:
                    header_idx = i
                    break
            
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
            
            recall = parsed_data[:, 1]
            qps = parsed_data[:, 2]
            
            # Sort by recall
            sort_idx = np.argsort(recall)
            recall = recall[sort_idx]
            qps = qps[sort_idx]
            
            data.append((param_val, recall, qps))
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return data

def parse_indexing_info(indexing_file, param_type):
    """
    Parse indexing_information.txt for the given parameter type
    Returns list: [(param_val, time_s, size_mb), ...]
    """
    data = []
    
    try:
        with open(indexing_file, 'r') as f:
            lines = f.readlines()
        
        # Find section for this parameter
        section_idx = -1
        section_name = "M" if param_type == "m" else param_type
        
        for i, line in enumerate(lines):
            if line.strip() == section_name:
                section_idx = i
                break
        
        if section_idx == -1:
            return data
        
        # Skip to "Minhash" subsection
        minhash_idx = -1
        for i in range(section_idx + 1, len(lines)):
            if "Minhash" in lines[i]:
                minhash_idx = i
                break
        
        if minhash_idx == -1:
            return data
        
        # Parse data lines until next section
        for i in range(minhash_idx + 1, len(lines)):
            line = lines[i].strip()
            if not line or line in ["T", "efc", "M", "k", "thread", "ivf"]:
                break
            
            if ',' in line:
                try:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        param_val = int(parts[0].strip())
                        time_str = parts[1].strip().rstrip('s')
                        time_s = float(time_str)
                        size_str = parts[2].strip().rstrip('M').rstrip('#')
                        size_mb = float(size_str)
                        data.append((param_val, time_s, size_mb))
                except (ValueError, IndexError):
                    pass
    
    except Exception as e:
        print(f"Error parsing indexing info: {e}")
    
    return data

def plot_qps_recall_row(fig, row_idx, param_types, directories):
    """Plot QPS vs Recall curves for all three parameter types"""
    
    param_name_map = {
        "T": r"$T$",
        "efc": r"$\omega_c$",
        "m": r"$m$"
    }
    
    for col_idx, (param_type, directory) in enumerate(zip(param_types, directories)):
        ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1)
        
        # Add title for first row
        ax.set_title(f"Varying {param_name_map[param_type]}", fontsize=30, fontweight='bold', pad=3)
        
        # Load data
        data = load_query_data(directory, "containment", param_type)
        
        if not data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(get_param_latex_label(param_type), fontsize=22)
            ax.set_ylabel("QPS", fontsize=22)
            continue
        
        # Sort by parameter value
        data.sort(key=lambda x: x[0])
        
        # Plot curves
        colors = colors_map[param_type]
        markers = markers_map[param_type]
        
        for idx, (param_val, recall, qps) in enumerate(data):
            color_idx = idx % len(colors)
            
            # Get LaTeX label for this parameter
            param_label = get_param_latex_label(param_type).strip('$')  # Remove outer $
            
            ax.plot(recall, qps,
                   color=colors[color_idx],
                   marker=markers[color_idx],
                   linestyle='-',
                   linewidth=2.5,
                   markersize=10,
                   markerfacecolor="none",
                   label=f"${param_label}$={param_val}")
        
        ax.set_xlabel("Recall@$k$", fontsize=26, labelpad=0)
        ax.set_ylabel("QPS", fontsize=26, labelpad=0)
        ax.tick_params(axis='both', labelsize=24, pad=1)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        ax.set_xlim(0.8, 1.0)
        
        # Adjust y-limits for T parameter to avoid legend overlap
        if param_type == "T":
            ax.set_ylim(bottom=10)
        
        # Legend in two bottom-aligned columns: first column (3 items), second column (2 items)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) >= 5:
            # First column: 3 legends
            leg1 = ax.legend(handles[:3], labels[:3], fontsize=22, loc="lower left",
                             frameon=False, bbox_to_anchor=(0, -0.05), borderaxespad=0.1, labelspacing=0.3, handletextpad=0.3, handlelength=1.5)
            ax.add_artist(leg1)
            # Second column: 2 legends, bottom-aligned via same 'lower left' with x offset
            ax.legend(handles[3:], labels[3:], fontsize=22, loc="lower left",
                      frameon=False, bbox_to_anchor=(0.38, -0.05), borderaxespad=0.1,
                      labelspacing=0.3, handletextpad=0.3, handlelength=1.5)
        else:
            # Fallback: single legend
            ax.legend(fontsize=22, loc="lower left", frameon=False, bbox_to_anchor=(0, -0.05))

def plot_indexing_bar_row(fig, row_idx, param_types, directories, indexing_file):
    """
    Plot combined bar charts with index size (upper) and indexing time (lower)
    X-axis is uniformly spaced (not by actual parameter values)
    Size grows upward, time grows downward (inverted)
    """
    
    for col_idx, (param_type, directory) in enumerate(zip(param_types, directories)):
        ax = fig.add_subplot(2, 3, row_idx * 3 + col_idx + 1)
        
        # Load indexing data
        indexing_data = parse_indexing_info(indexing_file, param_type)
        
        if not indexing_data:
            ax.text(0.5, 0.5, "No indexing data", ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(get_param_latex_label(param_type), fontsize=22)
            continue
        
        # Sort by parameter value
        indexing_data.sort(key=lambda x: x[0])
        
        # Extract values
        param_vals = [x[0] for x in indexing_data]
        times = [x[1] for x in indexing_data]
        sizes = [x[2] for x in indexing_data]
        
        # Create uniform x positions
        n_params = len(param_vals)
        x_pos = np.arange(n_params)
        bar_width = 0.6
        
        # Color for bars
        colors = colors_map[param_type]
        bar_colors = [colors[i % len(colors)] for i in range(n_params)]
        
        # Find max values
        max_size = max(sizes) if sizes else 1
        max_time = max(times) if times else 1
        
        # Scale data separately to fit upper and lower halves
        # Scale to [0, 1] for each half, then set limits with margin
        size_scale = 1.0
        time_scale = 1.0
        margin = 0.2  # 20% margin above/below tallest bars
        # if "T", margin=0.25
        if param_type == "efc":
            margin = 0.3
        
        # Scaled values (normalized to 0-1 range)
        sizes_scaled = [(s / max_size) * size_scale for s in sizes]
        times_scaled = [-(t / max_time) * time_scale for t in times]
        
        # Y-limits with margin to prevent bars from touching border
        y_limit = 1.0 * (1 + margin)
        ax.set_ylim(-y_limit, y_limit)
        
        # Plot bars for size (upper half, grows upward)
        ax.bar(x_pos, sizes_scaled, bar_width, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Plot bars for time (lower half, grows downward)
        ax.bar(x_pos, times_scaled, bar_width, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Set x-axis with uniform spacing and parameter labels
        ax.set_xticks(x_pos)
        param_labels = [str(v) for v in param_vals]
        ax.set_xticklabels(param_labels, fontsize=16)
        
        ax.set_xlabel(get_param_latex_label(param_type), fontsize=30, labelpad=0)
        # Remove y label - use only the text labels inside
        ax.set_ylabel("", fontsize=18)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Custom y-axis ticks based on actual values (maps to normalized positions)
        upper_tick_vals = np.linspace(0, max_size, 4)
        lower_tick_vals = np.linspace(0, max_time, 4)  # positive values

        # Map ticks using the same normalization as bars (not stretched by margin)
        upper_ticks_pos = (upper_tick_vals / max_size) * size_scale
        lower_ticks_pos = -(lower_tick_vals / max_time) * time_scale

        # Drop duplicated zero by skipping it in the upper ticks only
        all_ticks = np.concatenate([lower_ticks_pos, upper_ticks_pos[1:]])
        tick_labels = [f"{v:.0f}" for v in lower_tick_vals] + [f"{v:.0f}" for v in upper_tick_vals[1:]]

        ax.set_yticks(all_ticks)
        ax.set_yticklabels(tick_labels)
        
        ax.tick_params(axis='both', labelsize=24, pad=1)
        
        
        # Add text labels on top left and bottom left inside plots
        # "Size (MB)" label at top left
        ax.text(0.02, 0.98, "Size (MB)", transform=ax.transAxes,
               fontsize=24, ha='left', va='top')
        
        # "Time (s)" label at bottom left
        ax.text(0.02, 0.02, "Time (s)", transform=ax.transAxes,
               fontsize=24, ha='left', va='bottom')

def main():
    # Directories for each parameter type
    base_path = "/root/work/lssg/example/plot/hyperparam/LAION"
    param_types = ["T", "efc", "m"]
    directories = [
        os.path.join(base_path, "T"),
        os.path.join(base_path, "efc"),
        os.path.join(base_path, "m")
    ]
    indexing_file = os.path.join(base_path, "indexing_information.txt")
    
    # Create 2×3 figure
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(2, 3, wspace=0.3)
    
    # Row 0: QPS vs Recall curves
    plot_qps_recall_row(fig, 0, param_types, directories)
    
    # Row 1: Combined indexing bar charts
    plot_indexing_bar_row(fig, 1, param_types, directories, indexing_file)
    
    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0.28,wspace=0.22)
    
    # Save figure
    output_file = "plot_building_param.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
