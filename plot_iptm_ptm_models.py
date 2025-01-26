import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
import argparse
from matplotlib.table import Table
from matplotlib.cm import get_cmap
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

        chain_iptm_data = np.array(data['chain_iptm'])
        chain_ptm_data = np.array(data['chain_ptm'])

        return chain_iptm_data, chain_ptm_data
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading JSON file: {e}")
        return None, None



def plot_boxplot(data, title, labels, output_dir, filename):
    """
    Create a boxplot of the aggregated data and save it.

    Parameters:
    - data: List of 1D NumPy arrays containing the chain data (either IPTM or PTM).
    - title: Title of the boxplot.
    - labels: List of labels corresponding to the data.
    - output_dir: str, directory to save the plot.
    - filename: Name of the file to save the plot as.
    """
    # Create a figure for the boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    num_chains = len(labels)

    # Choose a colormap with enough colors
    if num_chains <= 10:
        color_palette = get_cmap("tab10")  # A colormap with 10 colors
    elif num_chains <= 20:
        color_palette = get_cmap("tab20")  # A colormap with 20 colors
    else:
        color_palette = get_cmap("tab20c")  # A colormap with a larger range of colors

        # Generate colors for each chain
    colors = [color_palette(i / num_chains) for i in range(num_chains)]


    # Plot the boxplot
    sns.boxplot(data=data, ax=ax, palette=colors)
    sns.stripplot(data=data, size=4, color=".3")

    # Set the labels and title
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel('Values')
    ax.spines[["right", "top"]].set_visible(False)

    # Save the plot
    output_file = os.path.join(output_dir, filename)
    plt.savefig(output_file)
    plt.close()




def process_folder(folder_path, output_dir):
    """
    Process all JSON files in the folder matching the pattern *summary_confidences_*.json,
    aggregate data, and plot a boxplot for each dataset (chain_iptm_data and chain_ptm_data).

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

    # Initialize lists to store data for each chain (per chain, aggregated across files)
    aggregated_chain_iptm_data = []
    aggregated_chain_ptm_data = []
    all_labels = []

    # Iterate over each JSON file
    for json_file in json_files:
        json_file_name = Path(json_file).stem
        print(f"Processing file: {json_file}")

        chain_iptm_data, chain_ptm_data = load_data_from_json(json_file)

        if chain_iptm_data is not None and chain_ptm_data is not None:
            # Create labels for the chains (using the length of the data)
            labels = [chr(i) for i in range(65, 65 + len(chain_iptm_data))]  # Corrected for 1D data
            all_labels = labels  # Assuming labels are consistent across all files

            # Append each chain's data across all files (aggregating by chain)
            if not aggregated_chain_iptm_data:
                # Initialize the aggregated lists with empty lists for each chain
                aggregated_chain_iptm_data = [[] for _ in range(len(chain_iptm_data))]
                aggregated_chain_ptm_data = [[] for _ in range(len(chain_ptm_data))]

            # Aggregate the data for each chain (column-wise)
            for i in range(len(chain_iptm_data)):
                aggregated_chain_iptm_data[i].append(chain_iptm_data[i])
                aggregated_chain_ptm_data[i].append(chain_ptm_data[i])
        else:
            print(f"Skipping file due to error: {json_file}")

    if aggregated_chain_iptm_data and aggregated_chain_ptm_data:
        # Now plot separate boxplots for chain_iptm_data and chain_ptm_data
        plot_boxplot(aggregated_chain_iptm_data, 'Chain IPTM Data', all_labels, output_dir, 'chain_iptm_data.png')
        plot_boxplot(aggregated_chain_ptm_data, 'Chain PTM Data', all_labels, output_dir, 'chain_ptm_data.png')
    else:
        print("No valid data to plot.")



def main():
    parser = argparse.ArgumentParser(description="Batch process JSON files and create chain iptm plots.")
    parser.add_argument("folder_path", help="Path to the folder containing chain iptm JSON files.")
    parser.add_argument("--output", "-o", help="Path to save the plots.", default=None)

    args = parser.parse_args()

    # Set output folder
    output_dir = args.output if args.output else os.path.join(args.folder_path, "iptm_ptm_plots")

    # Process all matching JSON files in the folder
    process_folder(args.folder_path, output_dir)


if __name__ == "__main__":
    main()
