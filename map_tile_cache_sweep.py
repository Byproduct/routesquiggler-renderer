#!/usr/bin/env python3
"""
Map Tile Cache Sweep Script

This script cleans the map tile cache created by cartopy in npy format,
detecting and removing blank tiles that contain only white pixels.

The script looks for cache directories in two locations:
1. Local: './map tile cache' (with progress bar)
2. Remote: '/mnt/storage-box/map tile cache' (without progress bar)
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List

# Try to import tqdm for progress bar (only needed for local directory)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")


def is_blank_or_erroneous_tile(npy_file_path: str) -> bool:
    """
    Check if an npy file contains only white pixels or is erroneous (corrupted/malformed).
    
    Args:
        npy_file_path: Path to the .npy file
        
    Returns:
        True if the tile contains only white pixels or is erroneous, False otherwise
    """
    try:
        # Load the numpy array
        tile_data = np.load(npy_file_path)
        
        # Handle different array shapes and data types
        if tile_data.dtype == np.uint8:
            # For uint8, white is 255
            white_value = 255
        elif tile_data.dtype == np.float32 or tile_data.dtype == np.float64:
            # For float, white is typically 1.0
            white_value = 1.0
        else:
            # For other types, assume normalized to 1.0
            white_value = 1.0
        
        # Check if all pixels are white
        # Handle both RGB and RGBA formats
        if len(tile_data.shape) == 3:
            # RGB/RGBA image
            if tile_data.shape[2] >= 3:  # At least RGB channels
                # Check if all RGB channels are white
                rgb_channels = tile_data[:, :, :3]
                return np.all(rgb_channels == white_value)
            else:
                # Single channel or unexpected format
                return np.all(tile_data == white_value)
        elif len(tile_data.shape) == 2:
            # Grayscale image
            return np.all(tile_data == white_value)
        else:
            # Unexpected format, assume not blank
            return False
            
    except Exception:
        # Any error (reshape errors, empty files, etc.) means the tile is erroneous
        # and should be deleted
        return True


def find_npy_files(directory: str) -> List[str]:
    """
    Recursively find all .npy files in the given directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of paths to .npy files
    """
    npy_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files


def process_cache_directory(cache_dir: str, use_progress_bar: bool = False, test_mode: bool = False) -> Tuple[int, int, List[str]]:
    """
    Process a cache directory and count good vs blank/erroneous tiles.
    
    Args:
        cache_dir: Path to the cache directory
        use_progress_bar: Whether to show progress bar
        test_mode: If True, only report without deleting files
        
    Returns:
        Tuple of (good_tiles_count, blank_or_erroneous_tiles_count, list_of_blank_tile_paths)
    """
    print(f"Scanning directory: {cache_dir}")
    
    # Find all .npy files
    npy_files = find_npy_files(cache_dir)
    total_files = len(npy_files)
    
    if total_files == 0:
        print("No .npy files found in the directory.")
        return 0, 0
    
    print(f"Found {total_files} .npy files to process")
    
    good_tiles = 0
    blank_or_erroneous_tiles = 0
    blank_tile_paths = []
    
    # Set up progress tracking
    if use_progress_bar and TQDM_AVAILABLE:
        iterator = tqdm(npy_files, desc="Processing tiles", unit="files")
    else:
        iterator = npy_files
        if not use_progress_bar:
            print("Processing files (no progress bar for remote directory)...")
    
    # Process each file
    for npy_file in iterator:
        if is_blank_or_erroneous_tile(npy_file):
            blank_or_erroneous_tiles += 1
            blank_tile_paths.append(npy_file)
            if not test_mode:
                os.remove(npy_file)
        else:
            good_tiles += 1
    
    return good_tiles, blank_or_erroneous_tiles, blank_tile_paths


def format_duration(seconds: float) -> str:
    """
    Format duration in minutes and seconds.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2m 34s"
    """
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    else:
        return f"{remaining_seconds}s"


def main():
    """Main function to run the cache sweep."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Clean map tile cache of blank and erroneous tiles')
    parser.add_argument('mode', nargs='?', default='production', 
                       help='Mode to run in: "test" for reporting only, or "production" for actual deletion (default)')
    args = parser.parse_args()
    
    test_mode = args.mode.lower() == 'test'
    
    print("Map Tile Cache Sweep - Starting...")
    if test_mode:
        print("Running in TEST MODE - no files will be deleted")
    else:
        print("Running in PRODUCTION MODE - files will be deleted")
    print("=" * 50)
    
    start_time = time.time()
    
    # Determine which cache directory to use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_cache = os.path.join(script_dir, "map tile cache")
    remote_cache1 = "/mnt/storage-box/map tile cache"
    remote_cache2 = "/var/cache/map_tile_cache"
    
    cache_dir = None
    use_progress_bar = False
    
    # Check for local cache directory first
    if os.path.exists(local_cache) and os.path.isdir(local_cache):
        cache_dir = local_cache
        use_progress_bar = True
        print(f"Using local cache directory: {cache_dir}")
    elif os.path.exists(remote_cache1) and os.path.isdir(remote_cache1):
        cache_dir = remote_cache1
        use_progress_bar = False
        print(f"Using remote cache directory: {cache_dir}")
    elif os.path.exists(remote_cache2) and os.path.isdir(remote_cache2):
        cache_dir = remote_cache2
        use_progress_bar = False
        print(f"Using remote cache directory: {cache_dir}")
    else:
        print("Error: No cache directory found!")
        print(f"Looked for:")
        print(f"  - {local_cache}")
        print(f"  - {remote_cache1}")
        print(f"  - {remote_cache2}")
        sys.exit(1)
    
    # Process the cache directory
    try:
        good_tiles, blank_or_erroneous_tiles, blank_tile_paths = process_cache_directory(cache_dir, use_progress_bar, test_mode)
        
        # Calculate and display results
        end_time = time.time()
        duration = end_time - start_time
        total_tiles = good_tiles + blank_or_erroneous_tiles
        
        print("\n" + "=" * 50)
        print("RESULTS:")
        print(f"Total tiles processed: {total_tiles}")
        print(f"Good tiles (kept): {good_tiles}")
        if test_mode:
            print(f"Blank/erroneous tiles (would be deleted): {blank_or_erroneous_tiles}")
            if blank_tile_paths:
                print("\nPaths of blank/erroneous tiles:")
                for path in blank_tile_paths:
                    print(f"  - {path}")
        else:
            print(f"Blank/erroneous tiles (deleted): {blank_or_erroneous_tiles}")
        
        print(f"\nOperation completed in: {format_duration(duration)}")
        if test_mode:
            print("\nNOTE: This was a test run - no files were actually deleted.")
            print("Run without 'test' parameter to perform actual deletion.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
