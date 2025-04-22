#!/usr/bin/env python3
"""Script to clear and force reindexing of ChromaDB cache."""

import os
import shutil
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from nomad_agent import config
except ImportError:
    print("ERROR: Could not import config. Make sure you are running this script from the project root directory.")
    sys.exit(1)

def clear_chroma_cache():
    """Safely removes the ChromaDB persistent cache directory."""
    cache_dir = config.CHROMA_CACHE_DIR
    print(f"Attempting to remove ChromaDB cache directory: {cache_dir}")

    if not os.path.exists(cache_dir):
        print("Cache directory does not exist. Nothing to remove.")
        return

    if not os.path.isdir(cache_dir):
        print(f"Error: Expected a directory at {cache_dir}, but found something else.")
        return

    try:
        # --- SAFETY CONFIRMATION ---
        confirm = input(f"Are you sure you want to permanently delete '{cache_dir}' and all its contents? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return
        # --- END SAFETY CONFIRMATION ---

        print(f"Deleting directory: {cache_dir}...")
        shutil.rmtree(cache_dir)
        print("✅ ChromaDB cache directory successfully removed.")
        print("The agent will re-index data on its next run.")

    except PermissionError:
        print(f"Error: Permission denied. Could not remove {cache_dir}.")
        print("  Please check file/directory permissions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("--- ChromaDB Cache Reset Script ---")
    clear_chroma_cache()
    print("--- Script Finished ---") 