import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import glob
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib.lines import Line2D


def load_data_from_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        pae_matrix = np.array(data.get("pae", []))
        plddt_scores = data.get("atom_plddts", [])
        token_chain_ids = data.get("token_chain_ids", [])
        token_res_ids = data.get("token_res_ids", [])
        contact_proba_data = np.array(data.get("contact_probs", []))

        if pae_matrix.size == 0 or not plddt_scores:
            raise ValueError("Missing required keys ('pae', 'plddt') in JSON file.")
        
        return pae_matrix, plddt_scores, token_chain_ids, token_res_ids, contact_proba_data
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error loading JSON file: {e}")
        return None, None, None, None, None, None



def plot_matrix(matrix, type_matrix, ax=None):
    """
    Plot the PAE matrix on the given axis.

    Parameters:
    - pae_matrix: 2D NumPy array of PAE values.
    - ax: Matplotlib axis object, optional. If None, a new axis is created.

    Returns:
    - ax: Matplotlib axis object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    if type_matrix == "PAE":    
        ax.set_facecolor("white")
        cmap = ax.imshow(
            matrix,
            vmin=0,
            vmax=32,
            cmap="Greens_r"
        )
        ax.set_xlabel("Scored Residue")
        ax.set_ylabel("Aligned Residue")
        return ax, cmap
    if type_matrix == "contact":
        cmap = ax.imshow(
            matrix,
            vmin=0,
            vmax=1,
            cmap="viridis"
        )
        ax.set_xlabel("Residue i")
        ax.set_ylabel("Residue j")
        return ax, cmap
        

def add_chain_boundaries(token_chain_ids, token_res_ids, ax):
    """
    Add chain boundaries and labels using token_chain_ids and token_res_ids.

    Parameters:
    - token_chain_ids: list of chain IDs for each token.
    - token_res_ids: list of residue numbers for each token.
    - ax: Matplotlib axis object.
    """
    chain_ranges = []
    current_chain = None
    start = 0

    for i, (chain_id, res_id) in enumerate(zip(token_chain_ids, token_res_ids)):
        if current_chain is None:
            current_chain = chain_id

        if chain_id != current_chain or i == len(token_chain_ids) - 1:
            end = i - 1 if i == len(token_chain_ids) - 1 else i
            chain_ranges.append((current_chain, start, end))
            current_chain = chain_id
            start = i

            
    # Define a color palette with enough colors for all chains
    # Define the number of chains
    num_chains = len(chain_ranges)

    # Choose a colormap with enough colors
    if num_chains <= 10:
        color_palette = colormaps["tab10"]  # A colormap with 10 colors
    elif num_chains <= 20:
        color_palette = colormaps["tab20"]  # A colormap with 20 colors
    else:
        color_palette = colormaps["tab20c"]  # A colormap with a larger range of colors

    # Generate colors for each chain
    colors = [color_palette(i / num_chains) for i in range(num_chains)]


    # Create custom legend entries
    legend_elements = []

    for idx, (chain_id, start, end) in enumerate(chain_ranges):
        ax.axhline(y=start, color="black", linestyle="--", linewidth=1)
        ax.axvline(x=start, color="black", linestyle="--", linewidth=1)
        ax.axhline(y=end, color="black", linestyle="--", linewidth=1)
        ax.axvline(x=end, color="black", linestyle="--", linewidth=1)
        
        # Calculate mid position
        mid_pos = (start + end) / 2
        ax.text(mid_pos, -40, chain_id, ha="center", fontsize=10, color="black")
        
        # Select a color for the chain
        color = color_palette(idx % color_palette.N)  # Handle more chains than available colors
        # Draw the chain bottom border with the selected color
        ax.plot([start, end], [-20, -20], color=color, linewidth=2)

        # Create legend entry for this chain
        legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f"Chain {chain_id}"))
        
        ax.text(-45, mid_pos, chain_id, ha="center", fontsize=10, color="black", rotation=90)
        ax.plot([-15, -15], [start, end], color=color, linewidth=2) 
        ax.tick_params(axis='both', which='major', pad=10)


def plot_combined(pae_matrix, plddt_scores, token_chain_ids, token_res_ids, output_dir, json_file_name, contact_proba_data):
    """
    Plot combined pLDDT, PAE, and Contact Probability in a single figure and save as a PNG file.

    Parameters:
    - pae_matrix: 2D NumPy array of PAE values.
    - plddt_scores: list of pLDDT scores.
    - token_chain_ids: list of chain IDs for each token.
    - token_res_ids: list of residue IDs for each token.
    - output_dir: str, directory to save the plot.
    - json_file_name: str, base name of the plot file.
    - contact_proba_data: 2D NumPy array of contact probability values.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.4, 1], width_ratios=[1, 1])

    # Plot pLDDT
    ax1 = fig.add_subplot(gs[0, :])
    total_residues = len(plddt_scores)
    ax1.add_patch(Rectangle((0, 90), total_residues, 10, color="#024fcc", alpha=0.3, label="Very high (pLDDT > 90)"))
    ax1.add_patch(Rectangle((0, 70), total_residues, 20, color="#60c2e8", alpha=0.3, label="High (90 > pLDDT > 70)"))
    ax1.add_patch(Rectangle((0, 50), total_residues, 20, color="#f37842", alpha=0.3, label="Low (70 > pLDDT > 50)"))
    ax1.add_patch(Rectangle((0, 0), total_residues, 50, color="#f9d613", alpha=0.3, label="Very low (pLDDT < 50)"))
    sns.lineplot(x=np.arange(1, total_residues + 1), y=plddt_scores, ax=ax1, color="black", linewidth=0.3)
    ax1.set_xlabel("Residue")
    ax1.set_ylabel("pLDDT Score")
    ax1.set_ylim(0, 100)
    ax1.set_title("Predicted Local Distance Difference Test (pLDDT) Scores")
    ax1.legend(title="pLDDT confidence", loc="upper left", bbox_to_anchor=(1, 1))
    ax1.spines[["right", "top"]].set_visible(False)
    ax1.set_xlim(0, total_residues + 1)  # Ensure x-axis matches residue range
    ax1.margins(0)  # Disable any additional margins

    # Plot PAE
    ax2 = fig.add_subplot(gs[1, 0])
    ax2, cmap = plot_matrix(pae_matrix, "PAE", ax2)
    add_chain_boundaries(token_chain_ids, token_res_ids, ax2)
    fig.colorbar(cmap, ax=ax2, orientation="vertical", label="Predicted Aligned Error (Å)", shrink=0.5)
    ax2.set_title("Predicted Aligned Error (PAE)", pad=20)

    # Plot Contact Probability Matrix
    ax3 = fig.add_subplot(gs[1, 1])
    ax3, cmap = plot_matrix(contact_proba_data, "contact", ax3)
    add_chain_boundaries(token_chain_ids, token_res_ids, ax3)
    fig.colorbar(cmap, ax=ax3, orientation="vertical", label="Predicted contact", shrink=0.5)
    ax3.set_title("Contact Probability Matrix")
    ax3.set_xlabel("Residue i")
    ax3.set_ylabel("Residue j")

    # Adjust spacing and layout
    plt.tight_layout()

    # Save the plot
    output_file = Path(output_dir) / f"{json_file_name}_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined plot saved to {output_file}")



def process_folder(folder_path, output_dir):
    """
    Process all JSON files in the folder matching the pattern *full_data_*.json.

    Parameters:
    - folder_path: str, path to the folder containing JSON files.
    - output_dir: str, path to save the plots.
    """
    json_files = glob.glob(f"{folder_path}/*full_data_*.json")
    if not json_files:
        print(f"No matching JSON files found in folder: {folder_path}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for json_file in json_files:
        print(f"Processing file: {json_file}")
        pae_matrix, plddt_scores, token_chain_ids, token_res_ids, contact_proba_data = load_data_from_json(json_file)
        if pae_matrix is not None:
            json_file_name = Path(json_file).stem
            plot_combined(pae_matrix, plddt_scores, token_chain_ids, token_res_ids, output_dir, json_file_name, contact_proba_data)


def main():
    parser = argparse.ArgumentParser(description="Batch process PAE JSON files and create combined pLDDT and PAE plots.")
    parser.add_argument("folder_path", help="Path to the folder containing PAE JSON files.")
    parser.add_argument("--output", "-o", help="Path to save the plots. Default is a subfolder 'PAE_plots' in the input folder.", default=None)

    args = parser.parse_args()

    output_dir = args.output if args.output else os.path.join(args.folder_path, "pae_plddt_contacts_plots")
    process_folder(args.folder_path, output_dir)


if __name__ == "__main__":
    main()
