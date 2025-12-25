import json
import os
from typing import List, Dict, Any
import argparse

def load_notebook(file_path: str) -> Dict[str, Any]:
    """Load a Jupyter notebook from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        print(f"✓ Successfully loaded: {file_path}")
        return notebook
    except FileNotFoundError:
        print(f"✗ Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"✗ Error: Invalid JSON in file - {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading {file_path}: {str(e)}")
        return None

def merge_notebooks(notebook_paths: List[str], output_path: str, 
                   add_separators: bool = True, preserve_metadata: bool = True) -> bool:
    """
    Merge multiple Jupyter notebooks into a single notebook.
    
    Args:
        notebook_paths: List of paths to notebook files to merge
        output_path: Path where the merged notebook will be saved
        add_separators: Whether to add markdown separators between notebooks
        preserve_metadata: Whether to preserve metadata from the first notebook
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    if len(notebook_paths) < 2:
        print("✗ Error: Need at least 2 notebooks to merge")
        return False
    
    notebooks = []
    
    # Load all notebooks
    for path in notebook_paths:
        if not os.path.exists(path):
            print(f"✗ Error: File does not exist - {path}")
            return False
        
        notebook = load_notebook(path)
        if notebook is None:
            return False
        notebooks.append((path, notebook))
    
    print(f"\n🔄 Merging {len(notebooks)} notebooks...")
    
    # Start with the structure of the first notebook
    merged_notebook = {
        "cells": [],
        "metadata": notebooks[0][1].get("metadata", {}) if preserve_metadata else {},
        "nbformat": notebooks[0][1].get("nbformat", 4),
        "nbformat_minor": notebooks[0][1].get("nbformat_minor", 4)
    }
    
    # Merge cells from all notebooks
    for i, (path, notebook) in enumerate(notebooks):
        filename = os.path.basename(path)
        
        # Add separator markdown cell before each notebook (except the first)
        if add_separators:
            if i > 0:
                separator_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"\n---\n\n# Content from: {filename}\n\n---\n"
                    ]
                }
                merged_notebook["cells"].append(separator_cell)
            else:
                # Add title for the first notebook
                title_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# Merged Notebook\n\n## Content from: {filename}\n"
                    ]
                }
                merged_notebook["cells"].append(title_cell)
        
        # Add all cells from current notebook
        cells = notebook.get("cells", [])
        for cell in cells:
            # Clear execution count and outputs for code cells to avoid conflicts
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            
            merged_notebook["cells"].append(cell)
        
        print(f"  ✓ Added {len(cells)} cells from {filename}")
    
    # Save the merged notebook
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_notebook, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Successfully merged notebooks!")
        print(f"📁 Output saved to: {output_path}")
        print(f"📊 Total cells: {len(merged_notebook['cells'])}")
        return True
        
    except Exception as e:
        print(f"✗ Error saving merged notebook: {str(e)}")
        return False

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Merge multiple Jupyter notebooks into one")
    parser.add_argument("notebooks", nargs="+", help="Paths to notebook files to merge")
    parser.add_argument("-o", "--output", default="merged_notebook.ipynb", 
                       help="Output file path (default: merged_notebook.ipynb)")
    parser.add_argument("--no-separators", action="store_true", 
                       help="Don't add markdown separators between notebooks")
    parser.add_argument("--no-metadata", action="store_true",
                       help="Don't preserve metadata from first notebook")
    
    args = parser.parse_args()
    
    print("🔧 Jupyter Notebook Merger")
    print("=" * 50)
    
    success = merge_notebooks(
        notebook_paths=args.notebooks,
        output_path=args.output,
        add_separators=not args.no_separators,
        preserve_metadata=not args.no_metadata
    )
    
    if success:
        print("\n🎉 Merge completed successfully!")
    else:
        print("\n❌ Merge failed!")
        exit(1)

# Example usage function for interactive use
def merge_three_notebooks_example():
    """
    Example function showing how to merge 3 specific notebooks.
    Modify the file paths as needed.
    """
    
    # Define the three notebook files you want to merge
    notebook_files = [
        "comprehensive_analysis.ipynb",
        "cse440project-Glove-with-7-models.ipynb", 
        "cse440project-skipgram-with-7-models.ipynb"
    ]
    
    # Output file name
    output_file = "merged_comprehensive_analysis.ipynb"
    
    print("🔧 Merging three notebooks example")
    print("=" * 50)
    
    # Check if files exist in current directory
    existing_files = []
    for file in notebook_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✓ Found: {file}")
        else:
            print(f"✗ Not found: {file}")
    
    if len(existing_files) >= 2:
        success = merge_notebooks(
            notebook_paths=existing_files,
            output_path=output_file,
            add_separators=True,
            preserve_metadata=True
        )
        
        if success:
            print(f"\n🎉 Successfully created {output_file}!")
        else:
            print("\n❌ Failed to merge notebooks!")
    else:
        print(f"\n⚠️  Need at least 2 notebooks to merge. Found {len(existing_files)} notebooks.")
        print("Available .ipynb files in current directory:")
        for file in os.listdir("."):
            if file.endswith(".ipynb"):
                print(f"  - {file}")

if __name__ == "__main__":
    # Uncomment the next line to run the example function instead of command line
    # merge_three_notebooks_example()
    
    # Run command line interface
    main()
