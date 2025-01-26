import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
import argparse
from matplotlib.table import Table
from matplotlib.gridspec import GridSpec

def load_data_from_json(json_path):
    """
    Load chain iptm data and other metrics from a JSON file.

    Parameters:
    - json_path: str, path to the JSON file.

    Returns:
    - chain_iptm_data: 2D NumPy array of chain iptm values.
    - metrics: Dictionary containing the values for "fraction_disordered", "has_clash", 
               "iptm", "num_recycles", "ptm", and "ranking_score".
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        chain__pair_iptm_data = np.array(data['chain_pair_iptm'])
        chain_pair_pae_min_data = np.array(data['chain_pair_pae_min'])
        chain_iptm_data = np.array(data['chain_iptm'])
        chain_ptm_data = np.array(data['chain_ptm'])

        metrics = {
            "fraction_disordered": data.get("fraction_disordered"),
            "has_clash": data.get("has_clash"),
            "iptm": data.get("iptm"),
            "num_recycles": data.get("num_recycles"),
            "ptm": data.get("ptm"),
            "ranking_score": data.get("ranking_score")
        }
        return chain__pair_iptm_data, metrics, chain_pair_pae_min_data, chain_iptm_data, chain_ptm_data
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading JSON file: {e}")
        return None, None



def plot_heatmap_with_table(data1, data2, data3, data4, labels, metrics):
    """
    Plot two heatmaps and two linear heatmaps with different plot heights.

    Parameters:
    - data1: 2D NumPy array for the first heatmap.
    - data2: 2D NumPy array for the second heatmap.
    - data3: 1D NumPy array for the third heatmap (horizontal).
    - data4: 1D NumPy array for the fourth heatmap (horizontal).
    - labels: List of labels for the heatmap axes.
    - metrics: Dictionary containing additional metrics.
    """
    # Create a figure with GridSpec
    fig = plt.figure(figsize=(26, 18))
    spec = GridSpec(4, 4, figure=fig, height_ratios=[2, 0.2, 0.2, 0.4])  # Adjusted for more space below

    # Define axes
    ax1 = fig.add_subplot(spec[0, 0:2])  # Top-left: spans 2 columns
    ax2 = fig.add_subplot(spec[0, 2:4])  # Top-right: spans 2 columns
    ax3 = fig.add_subplot(spec[1, 1:3])  # Bottom-left: smaller and centered
    ax4 = fig.add_subplot(spec[2, 1:3])  # Bottom-right: smaller and centered
    ax_table = fig.add_subplot(spec[3, 1:3])  # Space for the table below

    # Plot first heatmap
    sns.heatmap(data1, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'iptm Score'}, vmin=0, vmax=1, ax=ax1)
    ax1.set_title("Chain pair iptm")
    ax1.set_xlabel("Chain")
    ax1.xaxis.set_label_position('top')
    ax1.set_ylabel("Chain")
    ax1.xaxis.set_ticks_position('top')

    # Plot second heatmap
    sns.heatmap(data2, annot=True, cmap="Greens_r", xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Predicted Aligned Error (Å)'}, vmin=0, vmax=32, ax=ax2)
    ax2.set_title("Chain pair PAE min")
    ax2.set_xlabel("Chain")
    ax2.xaxis.set_label_position('top')
    ax2.set_ylabel("Chain")
    ax2.xaxis.set_ticks_position('top')

    # Plot third heatmap
    im = sns.heatmap(np.array(data3).reshape(1, -1), annot=True, cmap="coolwarm_r", xticklabels=labels, cbar = False,
                     cbar_kws={'label': 'Chain iptm'}, vmin=0, vmax=1, ax=ax3)
    ax3.set_title("Heatmap of chain iptm")
    ax3.set_yticks([])  # Remove y-axis ticks for linear data

    # Plot fourth heatmap
    sns.heatmap(np.array(data4).reshape(1, -1), annot=True, cmap="coolwarm_r", xticklabels=labels, cbar = False,
                cbar_kws={'label': 'Chain ptm'}, vmin=0, vmax=1, ax=ax4)
    ax4.set_title("Heatmap of chain ptm")
    ax4.set_yticks([])  # Remove y-axis ticks for linear data

    # Colorbar for third and fourth heatmaps
    mappable = im.get_children()[0]
    plt.colorbar(mappable, ax=[ax3, ax4], orientation='horizontal')

    # Add table
    ax_table.axis('off')  # Turn off the axis for the table
    table = Table(ax_table, bbox=[0, 0, 1, 1])  # Adjust bbox to fit the table within the axis

    table.auto_set_column_width([0.5, 0.5]) 
    # Display additional metrics as a table
    # Add rows to the table
    for i, (key, value) in enumerate(metrics.items()):
        table.add_cell(i, 0, 0.2, 0.3, text=key, loc='center', edgecolor='black', facecolor='lightgray')
        table.add_cell(i, 1, 0.2, 0.3, text=value, loc='center', edgecolor='black')

    # Add header row
    table.add_cell(-1, 0, 0.2, 0.3, text="Metric", loc='center', edgecolor='black', facecolor='lightblue')
    table.add_cell(-1, 1, 0.2, 0.3, text="Value", loc='center', edgecolor='black', facecolor='lightblue')

    ax_table.add_table(table)


def process_folder(folder_path, output_dir):
    """
    Process all JSON files in the folder matching the pattern *summary_confidences_*.json.

    Parameters:
    - folder_path: str, path to the folder containing JSON files.
    - output_dir: str, path to save the plots.
    """
    json_files = glob.glob(f"{folder_path}/*summary_confidences_*.json")
    if not json_files:
        print(f"No matching JSON files found in folder: {folder_path}")
        return

    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for json_file in json_files:
        json_file_name = Path(json_file).stem
        print(f"Processing file: {json_file}")
        chain__pair_iptm_data, metrics, chain_pair_pae_min_data, chain_iptm_data, chain_ptm_data = load_data_from_json(json_file)

        if chain__pair_iptm_data is not None:
            # Create labels for the columns (alphabetical order starting with 'A')
            labels = [chr(i) for i in range(65, 65 + chain__pair_iptm_data.shape[1])]

            # Generate output file name based on the JSON file name
            output_file = Path(output_dir) / f"{json_file_name}_summary.png"

            # Create a figure with 3 subplots
            fig, axes = plt.subplots(2, 2, figsize=(24, 16))
            
            # Unpack axes for easier handling
            (ax1, ax2), (ax3, ax4) = axes

            # Plot and save the heatmap with the table
            plot_heatmap_with_table(chain__pair_iptm_data, chain_pair_pae_min_data, chain_iptm_data,chain_ptm_data, labels, metrics)

            # Save the plot to the output file
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
        else:
            print(f"Skipping file due to error: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch process JSON files and create chain iptm plots.")
    parser.add_argument("folder_path", help="Path to the folder containing chain iptm JSON files.")
    parser.add_argument("--output", "-o", help="Path to save the plots.", default=None)

    args = parser.parse_args()

    # Set output folder
    output_dir = args.output if args.output else os.path.join(args.folder_path, "summary_plots")

    # Process all matching JSON files in the folder
    process_folder(args.folder_path, output_dir)


if __name__ == "__main__":
    main()
