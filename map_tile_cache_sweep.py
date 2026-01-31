#!/usr/bin/env python3
"""
Map Tile Cache Sweep Script

This script cleans the map tile cache created by cartopy in npy format,
detecting and removing blank tiles that contain only white pixels.

The script looks for cache directories in two locations:
1. Local: './map tile cache' (with progress bar)
2. Remote: '/mnt/storage-box/map tile cache' (without progress bar)
"""

# Standard library imports
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Third-party imports
import numpy as np

# Try to import tqdm for progress bar (only needed for local directory)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Local imports
from write_log import write_debug_log, write_log


def is_blank_or_erroneous_tile(npy_file_path: str) -> bool:
    """
    Triple check: tile must (1) not be completely white, (2) have average pixel
    value between 10 and 245 on 0-255 scale. Toner schemes (GeoapifyToner,
    StadiaToner) are excluded from the average check but must still pass (1).

    Args:
        npy_file_path: Path to the .npy file

    Returns:
        True if the tile is corrupt or erroneous, False otherwise
    """
    try:
        tile_data = np.load(npy_file_path)

        # --- Check 1: Not completely empty (all white) ---
        if tile_data.dtype == np.uint8:
            white_value = 255
        elif tile_data.dtype == np.float32 or tile_data.dtype == np.float64:
            white_value = 1.0
        else:
            white_value = 1.0

        if len(tile_data.shape) == 3:
            if tile_data.shape[2] >= 3:
                rgb_channels = tile_data[:, :, :3]
                if np.all(rgb_channels == white_value):
                    return True  # corrupt: completely white
            else:
                if np.all(tile_data == white_value):
                    return True
        elif len(tile_data.shape) == 2:
            if np.all(tile_data == white_value):
                return True
        # else: unexpected shape, treat as not completely white

        # --- Check 2: Average pixel value 10..245 (skip for black/white toner schemes) ---
        is_toner = "GeoapifyToner" in npy_file_path or "StadiaToner" in npy_file_path
        if is_toner:
            return False  # passed completely-empty check; toner excluded from average check

        # Normalize to 0-255 scale
        if tile_data.dtype == np.uint8:
            data_255 = tile_data
        elif tile_data.dtype == np.float32 or tile_data.dtype == np.float64:
            data_255 = (tile_data * 255).astype(np.float64)
        else:
            data_255 = np.asarray(tile_data, dtype=np.float64) * 255

        if len(tile_data.shape) == 3 and tile_data.shape[2] >= 3:
            rgb = data_255[:, :, :3]
            avg_value = float(np.mean(rgb))
        elif len(tile_data.shape) == 2:
            avg_value = float(np.mean(data_255))
        else:
            return False  # unexpected format, pass

        # Corrupt if average outside valid range (max 255)
        if avg_value < 10 or avg_value > 245:
            return True
        return False

    except Exception:
        # Any error (reshape errors, empty files, etc.) means the tile is erroneous
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


def process_cache_directory(cache_dir: str, use_progress_bar: bool = False, test_mode: bool = False) -> Tuple[int, int, int, List[str]]:
    """
    Process a cache directory and count good vs blank/erroneous tiles.
    Uses a verified tiles list to skip already-checked files for better performance.
    
    Args:
        cache_dir: Path to the cache directory
        use_progress_bar: Whether to show progress bar
        test_mode: If True, only report without deleting files
        
    Returns:
        Tuple of (skipped_tiles_count, good_tiles_count, blank_or_erroneous_tiles_count, list_of_blank_tile_paths)
    """
    write_debug_log(f"Scanning directory: {cache_dir}")
    
    # Path to verified tiles file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    verified_tiles_file = os.path.join(script_dir, "utils", "verified_map_tiles.txt")
    
    # Load already verified tiles
    verified_tiles = set()
    if os.path.exists(verified_tiles_file):
        try:
            with open(verified_tiles_file, "r", encoding="utf-8") as f:
                verified_tiles = set(line.strip() for line in f if line.strip())
            write_debug_log(f"Loaded {len(verified_tiles)} already verified tiles from cache")
        except Exception as e:
            write_debug_log(f"Warning: Could not load verified tiles list: {e}")
            verified_tiles = set()
    else:
        write_debug_log("No verified tiles file found, starting from scratch.")
    
    # Find all .npy files
    npy_files = find_npy_files(cache_dir)
    total_files = len(npy_files)
    
    if total_files == 0:
        write_debug_log("No .npy files found in the directory.")
        return 0, 0, 0, []
    
    write_debug_log(f"Found {total_files} .npy files to process")
    
    skipped_tiles = 0
    good_tiles = 0
    blank_or_erroneous_tiles = 0
    blank_tile_paths = []
    newly_verified_tiles = []
    
    # Set up progress tracking
    if use_progress_bar and TQDM_AVAILABLE:
        iterator = tqdm(npy_files, desc="Processing tiles", unit="files")
    else:
        iterator = npy_files
        if not use_progress_bar:
            write_debug_log("Processing files (no progress bar for remote directory)")
    
    # Process each file
    for npy_file in iterator:
        # Skip if already verified
        if npy_file in verified_tiles:
            # Verify file still exists (might have been deleted externally)
            if os.path.exists(npy_file):
                skipped_tiles += 1
                continue
            else:
                # File was deleted, remove from verified set
                verified_tiles.discard(npy_file)
        
        # Check the tile
        if is_blank_or_erroneous_tile(npy_file):
            blank_or_erroneous_tiles += 1
            blank_tile_paths.append(npy_file)
            # Remove from verified set if it was there (tile became bad)
            verified_tiles.discard(npy_file)
            if not test_mode:
                os.remove(npy_file)
        else:
            good_tiles += 1
            newly_verified_tiles.append(npy_file)
    
    # Append newly verified tiles to the file
    if newly_verified_tiles:
        try:
            # Ensure utils directory exists
            os.makedirs(os.path.dirname(verified_tiles_file), exist_ok=True)
            with open(verified_tiles_file, "a", encoding="utf-8") as f:
                for tile_path in newly_verified_tiles:
                    f.write(f"{tile_path}\n")
            write_debug_log(f"Added {len(newly_verified_tiles)} new verified tiles to cache")
        except Exception as e:
            write_debug_log(f"Warning: Could not save verified tiles list: {e}")
    
    return skipped_tiles, good_tiles, blank_or_erroneous_tiles, blank_tile_paths


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
    
    write_debug_log("Running map tile cache sweep.")
    if test_mode:
        write_debug_log("Map tile sweep running in test mode - files will not be actually deleted.")
    
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
        write_debug_log(f"Using local cache directory: {cache_dir}")
    elif os.path.exists(remote_cache1) and os.path.isdir(remote_cache1):
        cache_dir = remote_cache1
        use_progress_bar = False
        write_debug_log(f"Using remote cache directory: {cache_dir}")
    elif os.path.exists(remote_cache2) and os.path.isdir(remote_cache2):
        cache_dir = remote_cache2
        use_progress_bar = False
        write_debug_log(f"Using remote cache directory: {cache_dir}")
    else:
        write_log("Error: No cache directory found!")
        write_log(f"Looked for:")
        write_log(f"  - {local_cache}")
        write_log(f"  - {remote_cache1}")
        write_log(f"  - {remote_cache2}")
        sys.exit(1)
    
    # Process the cache directory
    try:
        skipped_tiles, good_tiles, blank_or_erroneous_tiles, blank_tile_paths = process_cache_directory(cache_dir, use_progress_bar, test_mode)
        
        # Calculate and display results
        end_time = time.time()
        duration = end_time - start_time
        total_tiles = skipped_tiles + good_tiles + blank_or_erroneous_tiles
        
        # Output summary in one line
        if test_mode:
            write_log(f"Skipped {skipped_tiles} already verified tiles. Total tiles processed: {total_tiles}. Good tiles (kept): {good_tiles}. Blank/erroneous tiles (would be deleted): {blank_or_erroneous_tiles}")
        else:
            write_log(f"Skipped {skipped_tiles} already verified tiles. Total tiles processed: {total_tiles}. Good tiles (kept): {good_tiles}. Blank/erroneous tiles (deleted): {blank_or_erroneous_tiles}")

        # Always print full file names of corrupt tiles to console (even when debug log is off)
        if blank_tile_paths:
            write_log("Corrupt map tiles (full paths):")
            for path in blank_tile_paths:
                write_log(path)
        
        write_debug_log(f"Operation completed in: {format_duration(duration)}")
        if test_mode:
            write_debug_log("NOTE: This was a test run - no files were actually deleted.")
            write_debug_log("Run without 'test' parameter to perform actual deletion.")
        
    except KeyboardInterrupt:
        write_debug_log("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        write_log(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
