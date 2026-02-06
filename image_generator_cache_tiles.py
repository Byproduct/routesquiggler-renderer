"""
Image generation map tiles caching for the Route Squiggler render client.
This module handles pre-caching map tiles needed for image generation before multiprocessing.
"""

# Standard library imports
import os
import time
from typing import List, Tuple, Optional

# Third-party imports
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# Local imports
from image_generator_maptileutils import (
    calculate_tile_count, 
    create_map_tiles, 
    set_cache_directory
)
from video_generator_cache_map_tiles import is_tile_cached

# Rate limiting delay (0.03 seconds minimum between tile requests)
# If a request takes longer than this, the wait is skipped
TILE_DOWNLOAD_DELAY = 0.03  # seconds


def pre_cache_map_tiles_for_images(
    zoom_levels: List[int],
    map_bounds: Tuple[float, float, float, float],
    map_style: str,
    resolution_x: int,
    resolution_y: int,
    log_callback=None,
    debug_callback=None,
    progress_callback=None
) -> bool:
    """
    Pre-cache map tiles for all zoom levels required for image generation.
    This should be called BEFORE multiprocessing starts to ensure all tiles are downloaded
    with proper rate limiting and the map tile lock can be released.
    
    Args:
        zoom_levels: List of zoom levels to cache tiles for
        map_bounds: Tuple (lon_min, lon_max, lat_min, lat_max)
        map_style: Map style to use
        resolution_x: Image width in pixels (for calculating padded bounds)
        resolution_y: Image height in pixels (for calculating padded bounds)
        log_callback: Optional callback for logging messages
        debug_callback: Optional callback for debug messages
        progress_callback: Optional callback for progress updates
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if debug_callback:
            debug_callback("Starting tile pre-caching for image generation")
        
        # Set up cache directory
        set_cache_directory(map_style)
        
        # Get the cache directory
        import cartopy
        cache_dir = cartopy.config.get('cache_dir', '')
        if not cache_dir:
            if log_callback:
                log_callback("Error: Could not determine cache directory")
            return False
        
        if debug_callback:
            debug_callback(f"Using cache directory: {cache_dir}")
        
        # Calculate padded bounds for each zoom level (same as image generation does)
        from image_generator_utils import ImageGenerator
        generator = ImageGenerator()
        target_aspect_ratio = resolution_x / resolution_y
        
        # Phase 1: Gather all required tiles for all zoom levels
        if debug_callback:
            debug_callback("Phase 1: Gathering all required tiles for all zoom levels")
        
        all_required_tiles = set()  # Use set to automatically remove duplicates
        
        for zoom_level in zoom_levels:
            # Calculate padded bounds for this zoom level (same as image generation)
            lon_min_padded, lon_max_padded, lat_min_padded, lat_max_padded = generator.calculate_aspect_ratio_bounds(
                map_bounds, target_aspect_ratio=target_aspect_ratio
            )
            
            # Convert coordinates to tile coordinates
            def deg2num(lat_deg, lon_deg, zoom):
                """Convert latitude/longitude to tile coordinates."""
                import math
                lat_rad = math.radians(lat_deg)
                n = 2.0 ** zoom
                x = int((lon_deg + 180.0) / 360.0 * n)
                y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
                return x, y
            
            # Get tile coordinates for the corners
            northwest = deg2num(lat_max_padded, lon_min_padded, zoom_level)
            southeast = deg2num(lat_min_padded, lon_max_padded, zoom_level)
            
            # Extract min/max tile coordinates
            x_min, y_min = northwest[0], northwest[1]
            x_max, y_max = southeast[0], southeast[1]
            
            # Ensure proper ordering
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)
            
            # Add all tiles for this zoom level to the set
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    # Validate tile coordinates
                    if zoom_level < 0 or zoom_level > 20:
                        continue
                    max_coord = 2 ** zoom_level
                    if not (0 <= x < max_coord and 0 <= y < max_coord):
                        continue
                    
                    all_required_tiles.add((x, y, zoom_level))
        
        total_required_tiles = len(all_required_tiles)
        if debug_callback:
            debug_callback(f"Found {total_required_tiles} unique tiles required across all zoom levels")
        
        # Phase 2: Filter out existing tiles
        if debug_callback:
            debug_callback("Phase 2: Checking which tiles are already cached")
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 0, "Checking cached tiles")
        
        tiles_to_download = []
        tiles_already_cached = 0
        
        for tile_index, tile_coords in enumerate(all_required_tiles):
            x, y, zoom = tile_coords
            
            if is_tile_cached(cache_dir, x, y, zoom, map_style):
                tiles_already_cached += 1
            else:
                tiles_to_download.append(tile_coords)
            
            # Update progress every 100 tiles
            if (tile_index + 1) % 100 == 0 and progress_callback:
                progress = int((tile_index + 1) / total_required_tiles * 50)  # First 50% for checking
                progress_callback("progress_bar_tiles", progress, f"Checking tiles: {tile_index + 1}/{total_required_tiles}")
        
        total_to_download = len(tiles_to_download)
        
        if debug_callback:
            debug_callback(f"Tile analysis complete: {tiles_already_cached} already cached, {total_to_download} need downloading")
        
        # Phase 3: Download missing tiles
        if total_to_download == 0:
            if debug_callback:
                debug_callback("All tiles are already cached! No downloads needed.")
            if progress_callback:
                progress_callback("progress_bar_tiles", 100, "All tiles already cached")
            return True
        
        if debug_callback:
            debug_callback(f"Phase 3: Downloading {total_to_download} missing tiles")
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 50, f"Downloading {total_to_download} tiles")
        
        # Create map tiles object
        map_tiles = create_map_tiles(map_style)
        
        # Track progress and statistics
        total_tiles_cached = 0
        consecutive_errors = 0
        max_consecutive_errors = 20
        error_types = {}
        
        # Track last request time for smart rate limiting
        last_request_time = None
        
        # Download all missing tiles with rate limiting
        for tile_index, tile_coords in enumerate(tiles_to_download):
            x, y, zoom = tile_coords
            
            # Apply rate limiting: only sleep if less than TILE_DOWNLOAD_DELAY has passed
            # This means if a request takes longer than the delay, we skip the wait
            if last_request_time is not None:
                current_time = time.time()
                time_since_last = current_time - last_request_time
                if time_since_last < TILE_DOWNLOAD_DELAY:
                    sleep_time = TILE_DOWNLOAD_DELAY - time_since_last
                    time.sleep(sleep_time)
            
            # Update progress
            if progress_callback and (tile_index + 1) % 10 == 0:
                progress = 50 + int((tile_index + 1) / total_to_download * 50)  # Second 50% for downloading
                progress_callback("progress_bar_tiles", progress, f"Downloading: {tile_index + 1}/{total_to_download}")
            
            try:
                # Use the cartopy API method to cache the tile
                tile_location = (x, y, zoom)
                img = map_tiles.get_image(tile_location)
                total_tiles_cached += 1
                consecutive_errors = 0  # Reset consecutive error counter on success
            except Exception as img_error:
                # Handle any errors from CartoPy
                error_str = str(img_error)
                
                if log_callback:
                    log_callback(f"Error fetching tile image with get_image: {img_error}")
                
                consecutive_errors += 1
                
                # Record error type
                error_type = type(img_error).__name__
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1
            finally:
                # Update last request time after the request completes (success or failure)
                # This ensures we track actual time between requests, not just sleeps
                last_request_time = time.time()
            
            # If too many consecutive errors, pause longer
            if consecutive_errors >= max_consecutive_errors:
                if log_callback:
                    log_callback(f"Too many consecutive errors ({consecutive_errors}). Pausing for 10 seconds")
                time.sleep(10)  # Longer pause when many errors
                consecutive_errors = 0
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 100, f"Downloaded {total_tiles_cached} tiles")
        
        if debug_callback:
            debug_callback(f"Tile caching completed: {total_tiles_cached} tiles downloaded, {tiles_already_cached} already cached")
            if error_types:
                debug_callback(f"Error summary: {error_types}")
        
        # Consider it successful if we downloaded most tiles (allow some errors)
        success_rate = total_tiles_cached / total_to_download if total_to_download > 0 else 1.0
        if success_rate >= 0.95:  # 95% success rate threshold
            return True
        else:
            if log_callback:
                log_callback(f"Warning: Only {total_tiles_cached}/{total_to_download} tiles downloaded successfully ({success_rate*100:.1f}%)")
            # Still return True if we got most tiles - some errors are acceptable
            return True
            
    except Exception as e:
        if log_callback:
            log_callback(f"Error in tile pre-caching: {str(e)}")
        import traceback
        if debug_callback:
            debug_callback(f"Traceback: {traceback.format_exc()}")
        return False
