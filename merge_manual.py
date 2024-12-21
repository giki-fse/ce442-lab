import os
import json
from glob import glob

def merge_ipynb_files(folder_path, output_file):
    """
    Merge all .ipynb files in a folder into a single notebook.

    Args:
        folder_path (str): Path to the folder containing .ipynb files.
        output_file (str): Path for the output merged notebook.
    """
    # Get all .ipynb files in the folder
    notebook_files = sorted(glob(os.path.join(folder_path, "*.ipynb")))
    if not notebook_files:
        print("No .ipynb files found in the specified folder.")
        return

    # Initialize the structure of the merged notebook with provided metadata
    merged_notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.4"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Loop through each notebook file and add its cells to the merged notebook
    for notebook_file in notebook_files:
        print(f"Processing {notebook_file}...")
        with open(notebook_file, "r", encoding="utf-8") as f:
            try:
                notebook_data = json.load(f)
                if "cells" in notebook_data:
                    merged_notebook["cells"].extend(notebook_data["cells"])
                else:
                    print(f"Warning: No cells found in {notebook_file}")
            except json.JSONDecodeError:
                print(f"Error: {notebook_file} is not a valid .ipynb file")

    # Save the merged notebook
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_notebook, f, indent=2)

    print(f"All notebooks have been merged into '{output_file}'.")

# Specify the folder containing the .ipynb files and output file name
folder_path = "./manual/"  # Change this to your folder path
output_file = "manual.ipynb"  # Output file name

# Call the function to merge the files
merge_ipynb_files(folder_path, output_file)
