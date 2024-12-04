from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import string
import glob
import os
import argparse

def find_json_files(folder_path):
    """Find all JSON files matching the pattern in the specified folder."""
    matching_files = glob.glob(os.path.join(folder_path, "*_job_request.json"))
    if not matching_files:
        raise FileNotFoundError(f"No files matching the pattern 'job_request.json' were found in '{folder_path}'")
    return matching_files  # Return all matches


def generate_alphafold_pdf_report(json_file_path, output_pdf_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        print(f"Processing JSON file: {json_file_path}")

    
    # Function to generate chain labels based on the count starting from the given index
    def generate_chain_labels(start_index, count):
        """Generate chain labels based on the count starting from the given index."""
        labels = []
        alphabet = string.ascii_uppercase
        for i in range(count):
            label = ''
            n = start_index + i
            while n >= 0:
                label = alphabet[n % 26] + label
                n = n // 26 - 1
            labels.append(label)
        return labels
    
    # Assign labels and add to each sequence
    last_used_index = 0
    for entry in data:
        for sequence in entry["sequences"]:
            if "proteinChain" in sequence:
                seq = sequence["proteinChain"]
                chain_labels = generate_chain_labels(last_used_index, seq['count'])
                # Assign the labels to the sequence
                sequence["chainLabels"] = chain_labels
                last_used_index += seq['count']
            elif "dnaSequence" in sequence:
                seq = sequence["dnaSequence"]
                chain_labels = generate_chain_labels(last_used_index, seq['count'])
                # Assign the labels to the sequence
                sequence["chainLabels"] = chain_labels
                last_used_index += seq['count']
    
    # Initialize the PDF canvas
    pdf = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter
    
    # Font settings
    pdf.setFont("Helvetica", 10)
    margin = 50
    line_spacing = 14
    y_position = height - margin  # Start near the top of the page
    
    # Maximum width for text wrapping
    max_line_length = 60  # Maximum number of characters per line for sequences
    
    def wrap_long_text(text, max_length):
        """Split text into lines of specified max length."""
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    for entry in data:
        # Entry title
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(margin, y_position, f"{entry['name']}")
        y_position -= line_spacing
        
        pdf.setFont("Helvetica", 10)
        
        # Print model seeds
        model_seeds = "Model Seeds: " + ", ".join(entry["modelSeeds"])
        pdf.drawString(margin, y_position, model_seeds)
        y_position -= line_spacing
        
        # Print sequence details
        pdf.drawString(margin, y_position, "Sequences:")
        y_position -= line_spacing
        
        for sequence in entry["sequences"]:
            # Print chain labels and sequence details
            chain_labels = ", ".join(sequence["chainLabels"])
            if "proteinChain" in sequence:
                seq = sequence["proteinChain"]
                y_position -= line_spacing
                header = f"Protein Sequence Chains {chain_labels} (Count = {seq['count']}, Length = {len(seq['sequence'])}):"
                pdf.drawString(margin + 20, y_position, header)
                y_position -= line_spacing
                
                # Wrap and print the sequence
                sequence_lines = wrap_long_text(seq["sequence"], max_line_length)
                for line in sequence_lines:
                    pdf.drawString(margin + 40, y_position, line)
                    y_position -= line_spacing
                
            elif "dnaSequence" in sequence:
                seq = sequence["dnaSequence"]
                y_position -= line_spacing
                header = f"DNA Sequence Chains {chain_labels} (Count = {seq['count']}, Length = {len(seq['sequence'])}):"
                pdf.drawString(margin + 20, y_position, header)
                y_position -= line_spacing
                
                # Wrap and print the sequence
                sequence_lines = wrap_long_text(seq["sequence"], max_line_length)
                for line in sequence_lines:
                    pdf.drawString(margin + 40, y_position, line)
                    y_position -= line_spacing
            
            # Check if we need a new page
            if y_position < margin:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y_position = height - margin
        
        # Add space between entries
        y_position -= line_spacing
        if y_position < margin:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y_position = height - margin
    
    # Save the PDF
    pdf.save()



def main():
    parser = argparse.ArgumentParser(description="Generate AlphaFold PDF reports from JSON files.")
    parser.add_argument("folder_path", help="Folder path containing the JSON files.")
    parser.add_argument("--output", "-o", help="Path to save the PDFs.", default=None)
    
    args = parser.parse_args()

    try:
        
        # Find all JSON files in the specified folder
        json_file_paths = find_json_files(args.folder_path)
        print(f"Found JSON files: {json_file_paths}")

        # Set output folder
        output_dir = args.output if args.output else os.path.join(args.folder_path, "reports")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        for index, json_file_path in enumerate(json_file_paths, start=1):
            # Generate a unique output path for each PDF
            output_pdf_path = os.path.join(output_dir, f"af_report_{index}.pdf")
            print(f"Generating PDF for: {json_file_path} -> {output_pdf_path}")
            generate_alphafold_pdf_report(json_file_path, output_pdf_path)
            print(f"Generated PDF: {output_pdf_path}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

