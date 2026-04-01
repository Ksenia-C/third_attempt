import re
import ast
import json
from pathlib import Path
import matplotlib.pyplot as plt

def parse_history(history_str):
    """Extract loss list and metrics dict from the history string."""
    lines = history_str.splitlines()
    
    # 1. Extract loss rounds and values
    loss_rounds = []
    loss_values = []
    loss_pattern = re.compile(r"round\s+(\d+):\s+([0-9.]+)")
    for line in lines:
        match = loss_pattern.search(line)
        if match:
            loss_rounds.append(int(match.group(1)))
            loss_values.append(float(match.group(2)))
    
    # 2. Find the metrics dictionary part
    #    It starts after "History (metrics, distributed, evaluate):"
    metrics_start = None
    for i, line in enumerate(lines):
        if "History (metrics, distributed, evaluate):" in line:
            metrics_start = i + 1
            break
    
    if metrics_start is None:
        raise ValueError("Metrics section not found")
    
    # Collect all following lines until the end (or until an empty line)
    metrics_lines = []
    for line in lines[metrics_start:]:
        if line.strip() == "":
            continue
        metrics_lines.append(line.strip())
    
    # Join them into a single string that should be a valid Python dict
    metrics_dict_str = " ".join(metrics_lines)
    # ast.literal_eval expects a valid Python literal – it works with tuples and dicts
    metrics_dict = ast.literal_eval(metrics_dict_str)
    
    # Convert list of tuples to separate round/value lists for convenience
    acc_rounds, acc_values = zip(*metrics_dict['accuracy']) if metrics_dict['accuracy'] else ([], [])
    f1_rounds, f1_values = zip(*metrics_dict['f1']) if metrics_dict['f1'] else ([], [])
    evil_global_weighted_f1_rounds, evil_global_weighted_f1_values = zip(*metrics_dict['evil_global_weighted_f1']) if metrics_dict['evil_global_weighted_f1'] else ([], [])
    lss_ce_rounds, lss_ce_values = zip(*metrics_dict['loss_cross_entropy']) if metrics_dict['loss_cross_entropy'] else ([], [])
    
    return {
        'loss': {'rounds': loss_rounds, 'values': loss_values},
        'accuracy': {'rounds': list(acc_rounds), 'values': list(acc_values)},
        'f1': {'rounds': list(f1_rounds), 'values': list(f1_values)},
        'evil_global_weighted_f1': {'rounds': list(evil_global_weighted_f1_rounds), 'values': list(evil_global_weighted_f1_values)},
        'loss_cross_entropy': {'rounds': list(lss_ce_rounds), 'values': list(lss_ce_values)}
    }

def save_parsed_data(data, directory):
    """Save the parsed metrics as JSON files."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in data.items():
        file_path = directory / f"{name}.json"
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
    print(f"Saved JSON files to {directory}")

def plot_server_metrics(data, directory):
    """Create a figure with loss, accuracy, f1 vs round, annotated."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [('loss', 'Loss'), ('accuracy', 'Accuracy'), ('f1', 'F1 Score')]
    for ax, (key, title) in zip(axes, metrics):
        rounds = data[key]['rounds']
        values = data[key]['values']
        ax.plot(rounds, values, marker='o', linestyle='-', label=title)
        ax.set_title(title)
        ax.set_xlabel('Round')
        ax.set_ylabel(title)
        ax.grid(True)
        # Annotate each point with its value
        for r, v in zip(rounds, values):
            ax.annotate(f'{v:.3f}', (r, v), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)
    
    # extra - evil global weighted f1
    rounds = data['evil_global_weighted_f1']['rounds']
    values = data['evil_global_weighted_f1']['values']
    axes[-1].plot(rounds, values, marker='o', linestyle='-', label='global_w_f1')
    for r, v in zip(rounds, values):
        axes[-1].annotate(f'{v:.3f}', (r, v), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)
    

    
    # extra - simple cross entropy along focal (train) loss
    rounds = data['loss_cross_entropy']['rounds']
    values = data['loss_cross_entropy']['values']
    axes[0].plot(rounds, values, marker='o', linestyle='-', label='cross_entropy_loss')
    for r, v in zip(rounds, values):
        axes[0].annotate(f'{v:.3f}', (r, v), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)

    # draw final
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    save_path = directory / "server_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved server metrics plot to {save_path}")

# ----------------------------------------------------------------------
# Client training losses: read from client_*/train_losses.txt and plot
def read_client_losses(base_dir):
    """Find all client directories and read their train_losses.txt files."""
    client_dirs = sorted(base_dir.glob("client_*"))
    client_losses = {}
    
    for client_dir in client_dirs:
        loss_file = client_dir / "train_losses.txt"
        if not loss_file.exists():
            print(f"Warning: {loss_file} not found, skipping")
            continue
        
        losses = []
        with open(loss_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Expecting format: "client X: value"
                    parts = line.split(':')
                    if len(parts) == 2:
                        try:
                            losses.append(float(parts[1].strip()))
                        except ValueError:
                            pass  # ignore malformed lines
        if losses:
            client_losses[client_dir.name] = losses
    
    return client_losses

def plot_client_losses(client_losses, directory):
    """Plot all client loss curves on one graph."""
    plt.figure(figsize=(10, 6))
    for client_name, losses in client_losses.items():
        steps = list(range(1, len(losses) + 1))
        plt.plot(steps, losses, marker='.', linestyle='-', label=client_name)
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Per‑Client Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = directory / "client_losses.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved client losses plot to {save_path}")



import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_std_data(data, save_dir, show_plot=False, save_plot=True):
    """
    Create a scatter plot with connecting lines from the given std data.
    Saves the raw data as a human‑readable JSON file.

    Parameters
    ----------
    data : dict
        Dictionary with keys 'size_std', 'local_std', and 'class_std'.
        'class_std' must be a dict mapping class labels (any type) to std values.
    save_dir : str
        Directory where the raw data (as .json) and the plot (as .png) will be saved.
    show_plot : bool, optional
        If True, call plt.show() (default True).
    save_plot : bool, optional
        If True, save the figure as 'std_plot.png' in save_dir (default True).
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Save raw data in human‑readable JSON format
    # ------------------------------------------------------------------
    # Convert NumPy types to Python native types and ensure keys are strings
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    json_compatible_data = convert_for_json(data)
    json_path = os.path.join(save_dir, 'std_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_compatible_data, f, indent=2)
    print(f"Raw data saved to: {json_path}")

    # ------------------------------------------------------------------
    # 2. Extract values and build the plotting sequence
    # ------------------------------------------------------------------
    size_std = float(data['size_std'])          # ensure Python float
    local_std = float(data['local_std'])
    class_std_dict = data['class_std']

    # Order: first size_std, then local_std, then all class entries sorted by key
    class_keys = sorted(class_std_dict.keys())
    x_labels = ['size_std', 'local_std'] + [f'class_{k}' for k in class_keys]
    y_values = [size_std, local_std] + [float(class_std_dict[k]) for k in class_keys]

    # x positions are simply the indices (0,1,2,...)
    x_pos = list(range(len(x_labels)))

    # ------------------------------------------------------------------
    # 3. Create the plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(max(8, len(x_labels)*0.6), 5))
    
    # Plot lines (subtle) and markers
    plt.plot(x_pos, y_values, marker='o', linestyle='-', 
             linewidth=0.8, color='gray', markersize=6, markerfacecolor='steelblue')

    # Add value annotations near each point
    for i, (x, y) in enumerate(zip(x_pos, y_values)):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                     xytext=(0, 8), ha='center', fontsize=8, color='darkred')

    # Customize axes
    plt.xticks(ticks=x_pos, labels=x_labels, rotation=45, ha='right', fontsize=9)
    plt.ylabel('Standard Deviation', fontsize=10)
    plt.title('Standard Deviation Values', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 4. Save and/or show
    # ------------------------------------------------------------------
    if save_plot:
        plot_path = os.path.join(save_dir, 'std_plot.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()
