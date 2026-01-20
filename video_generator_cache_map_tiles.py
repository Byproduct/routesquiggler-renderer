"""
Video generation map tiles caching for the Route Squiggler render client.
This module handles caching map tiles needed for video frame generation.
"""

# Standard library imports
import json
import os
import threading
import time
from pathlib import Path

# Third-party imports
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import numpy as np

# Local imports
from image_generator_maptileutils import calculate_tile_count, create_map_tiles, detect_zoom_level, set_cache_directory
from sync_map_tiles import sync_map_tiles
from video_generator_calculate_bounding_boxes import calculate_route_time_per_frame, calculate_unique_bounding_boxes

STADIA_API_KEY = "2413a338-8b10-4302-a96c-439cb795b285"

# Debug flag to control console output
MAP_TILE_CACHING_DEBUG = False

# Global variable to store storage box credentials for syncing
STORAGE_BOX_CREDENTIALS = None

def set_storage_box_credentials(credentials):
    """Set storage box credentials for background syncing."""
    global STORAGE_BOX_CREDENTIALS
    STORAGE_BOX_CREDENTIALS = credentials

def trigger_background_sync(log_callback=None, debug_callback=None):
    """Trigger background tile syncing if new tiles were cached during video generation."""
    global STORAGE_BOX_CREDENTIALS
    
    if not STORAGE_BOX_CREDENTIALS:
        if log_callback:
            log_callback("No storage box credentials available for background sync")
        return
    
    def sync_worker():
        """Background worker to perform tile syncing."""
        try:
            if debug_callback:
                debug_callback("Starting background tile sync...")
            
            success, uploaded_count, downloaded_count = sync_map_tiles(
                storage_box_address=STORAGE_BOX_CREDENTIALS['address'],
                storage_box_user=STORAGE_BOX_CREDENTIALS['user'],
                storage_box_password=STORAGE_BOX_CREDENTIALS['password'],
                local_cache_dir="map tile cache",
                log_callback=log_callback,  # Use the provided log callback
                progress_callback=lambda msg: None,  # Don't show progress in UI
                sync_state_callback=lambda state: None,  # Don't change UI state
                max_workers=5,  # Use fewer workers for background sync
                dry_run=False,
                upload_only=False  # Full sync to upload new tiles
            )
            
            if success:
                if debug_callback:
                    debug_callback(f"Background tile sync completed: {uploaded_count} uploaded, {downloaded_count} downloaded")
            else:
                if log_callback:
                    log_callback("Background tile sync failed")
                    
        except Exception as e:
            if log_callback:
                log_callback(f"Background tile sync error: {str(e)}")
    
    # Start background sync thread
    sync_thread = threading.Thread(target=sync_worker, daemon=True)
    sync_thread.start()
    
    if debug_callback:
        debug_callback("Background tile sync started (will continue in background)")


def is_tile_cached(cache_dir, x, y, zoom, map_style):
    """
    Check if a tile is already cached in the file system.
    
    Args:
        cache_dir (str): Base cache directory
        x (int): Tile x coordinate
        y (int): Tile y coordinate
        zoom (int): Zoom level
        map_style (str): Map style (used to determine file extension)
    
    Returns:
        bool: True if tile is cached, False otherwise
    """
    try:
        # CartoPy creates nested subdirectories. Check both possible locations:
        # 1. Direct in cache directory: {cache_dir}/{x}_{y}_{zoom}.npy
        # 2. For Stadia maps: {cache_dir}/CustomStadiaTiles/{x}_{y}_{zoom}.npy
        # 3. For other maps: {cache_dir}/{subdir}/{x}_{y}_{zoom}.npy
        
        # First try direct location
        tile_file = os.path.join(cache_dir, f"{x}_{y}_{zoom}.npy")
        
        # If not found, try nested subdirectory based on map style
        if not os.path.exists(tile_file):
            if map_style.startswith('stadia'):
                # For Stadia maps, check the CustomStadiaTiles subdirectory
                tile_file = os.path.join(cache_dir, 'CustomStadiaTiles', f"{x}_{y}_{zoom}.npy")
            else:
                # For non-Stadia maps, check if there's a nested subdirectory
                # This handles cases like OSM/OSM structure
                style_subdir_mapping = {
                    'osm': 'OSM',
                    'otm': 'OpenTopoMap'
                }
                subdir = style_subdir_mapping.get(map_style, map_style.upper())
                tile_file = os.path.join(cache_dir, subdir, f"{x}_{y}_{zoom}.npy")
        
        # Debug output
        if MAP_TILE_CACHING_DEBUG:
            print(f"Looking for tile file: {tile_file}")
            print(f"  - File exists: {os.path.exists(tile_file)}")
            if os.path.exists(tile_file):
                print(f"  - File size: {os.path.getsize(tile_file)} bytes")
        
        # Check if file exists and has reasonable size (> 0 bytes)
        if os.path.exists(tile_file) and os.path.getsize(tile_file) > 0:
            return True
        
        return False
    except Exception as e:
        # If any error occurs, assume tile is not cached
        if MAP_TILE_CACHING_DEBUG:
            print(f"Error checking tile cache: {e}")
        return False


def pre_cache_map_tiles_for_video(unique_bounding_boxes, json_data, progress_callback=None, log_callback=None, debug_callback=None):
    """
    Pre-cache map tiles for all unique bounding boxes required for video generation.
    
    Args:
        unique_bounding_boxes (set): Set of unique bounding boxes as tuples (lon_min, lon_max, lat_min, lat_max)
        json_data (dict): Job data containing video parameters
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting tile pre-caching for video generation")
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 0, "Calculating bounding boxes")
        
        # Get video parameters
        video_length = float(json_data.get('video_length', 30))
        video_fps = float(json_data.get('video_fps', 30))
        map_style = json_data.get('map_style', 'osm')
        
        # Set up cache directory and map tiles
        set_cache_directory(map_style)
        map_tiles = create_map_tiles(map_style)
        
        # Get the current cache directory that was set
        cache_dir = os.environ.get('CARTOPY_CACHE_DIR', '')
        if not cache_dir:
            # Fallback to checking cartopy config
            import cartopy
            cache_dir = cartopy.config.get('cache_dir', '')
        
        if debug_callback:
            debug_callback(f"Using cache directory: {cache_dir}")
        
        # Convert bounding boxes to list for iteration
        bbox_list = list(unique_bounding_boxes)
        total_bboxes = len(bbox_list)
        
        if debug_callback:
            debug_callback(f"Processing {total_bboxes} unique bounding boxes")
        
        # Determine max tiles based on map style
        max_tiles_config = {
            'osm': 100,
            'otm': 100,
            'stadia_light': 100,
            'stadia_dark': 100,
            'stadia_outdoors': 100,
            'stadia_toner': 100,
            'stadia_watercolor': 100,
        }
        max_tiles = max_tiles_config.get(map_style, 100)
        
        # Phase 1: Gather all required tiles
        if debug_callback:
            debug_callback("Phase 1: Gathering all required tiles...")
        
        all_required_tiles = set()  # Use set to automatically remove duplicates
        
        for bbox_index, bbox in enumerate(bbox_list):
            lon_min, lon_max, lat_min, lat_max = bbox
            
            # Detect the best zoom level for this bounding box
            map_bounds = (lon_min, lon_max, lat_min, lat_max)
            zoom_levels = detect_zoom_level(map_bounds, min_tiles=10, max_tiles=max_tiles, map_style=map_style)
            
            if zoom_levels:
                # Use the highest zoom level (most detailed) that fits within max_tiles
                zoom_level = max(zoom_levels)
            else:
                # If no suitable zoom found, use a conservative zoom level
                zoom_level = 17  # Default fallback (changed from 10)
                if log_callback:
                    log_callback(f"Warning: No suitable zoom found for bbox {bbox_index+1}, using zoom {zoom_level}")
            
            # Cap zoom level to 17 for video mode (tiles above zoom 17 don't exist in most providers)
            if zoom_level > 17:
                if debug_callback:
                    debug_callback(f"Zoom level {zoom_level} too high for video mode, capping to 17")
                zoom_level = 17
            
            # Convert coordinates to tile coordinates
            def deg2num(lat_deg, lon_deg, zoom):
                """Convert latitude/longitude to tile coordinates."""
                import math
                lat_rad = math.radians(lat_deg)
                n = 2.0 ** zoom
                x = int((lon_deg + 180.0) / 360.0 * n)
                # Use the correct formula that matches the original working code
                y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
                return x, y
            
            # Get tile coordinates for the corners of our region (using original working approach)
            northwest = deg2num(lat_max, lon_min, zoom_level)
            southeast = deg2num(lat_min, lon_max, zoom_level)
            
            # Extract min/max tile coordinates
            x_min, y_min = northwest[0], northwest[1]
            x_max, y_max = southeast[0], southeast[1]
            
            # Ensure proper ordering (northwest/southeast can be flipped in some cases)
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)
            
            # Add all tiles for this bounding box to the set
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
            debug_callback(f"Found {total_required_tiles} unique tiles required across all bounding boxes")
        
        # Phase 2: Filter out existing tiles
        if debug_callback:
            debug_callback("Phase 2: Checking which tiles are already cached...")
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 0, "Determining files to download")
        
        # Debug: Show cache directory
        if MAP_TILE_CACHING_DEBUG:
            print(f"Cache directory being used: {cache_dir}")
            print(f"Cache directory exists: {os.path.exists(cache_dir)}")
            if os.path.exists(cache_dir):
                print(f"Cache directory contents:")
                try:
                    for item in os.listdir(cache_dir)[:10]:  # Show first 10 items
                        item_path = os.path.join(cache_dir, item)
                        if os.path.isfile(item_path):
                            print(f"  File: {item} ({os.path.getsize(item_path)} bytes)")
                        else:
                            print(f"  Dir: {item}")
                except Exception as e:
                    print(f"Error listing cache directory: {e}")
        
        tiles_to_download = []
        tiles_already_cached = 0
        
        for tile_index, tile_coords in enumerate(all_required_tiles):
            x, y, zoom = tile_coords
            
            # Only show debug output for first few tiles to avoid spam
            if MAP_TILE_CACHING_DEBUG and tile_index < 3:
                print(f"\nChecking tile {tile_index + 1}: x={x}, y={y}, zoom={zoom}")
            
            if is_tile_cached(cache_dir, x, y, zoom, map_style):
                tiles_already_cached += 1
                if MAP_TILE_CACHING_DEBUG and tile_index < 3:
                    print(f"  -> Tile {tile_index + 1} is CACHED")
            else:
                tiles_to_download.append(tile_coords)
                if MAP_TILE_CACHING_DEBUG and tile_index < 3:
                    print(f"  -> Tile {tile_index + 1} needs DOWNLOADING")
            
            # Stop debug output after first few tiles
            if MAP_TILE_CACHING_DEBUG and tile_index == 2:
                print(f"\n... (showing first 3 tiles only)")
                # Continue checking all tiles, just stop debug output
        
        total_to_download = len(tiles_to_download)
        
        if debug_callback:
            debug_callback(f"Tile analysis complete: {tiles_already_cached} already cached, {total_to_download} need downloading")
        
        # Phase 3: Download missing tiles
        if total_to_download == 0:
            if debug_callback:
                debug_callback("All tiles are already cached! No downloads needed.")
            if progress_callback:
                progress_callback("progress_bar_tiles", 100, "All tiles already cached")
            return {
                'total_tiles_cached': 0,
                'total_tiles_skipped': total_required_tiles,
                'total_tiles_to_cache': total_required_tiles,
                'total_bboxes': total_bboxes,
                'error_types': {},
                'success': True
            }
        
        if debug_callback:
            debug_callback(f"Phase 3: Downloading {total_to_download} missing tiles...")
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 0, f"Downloading {total_to_download} tiles")
        
        # Add rate limiting based on map style
        if map_style == 'osm':
            # OSM is more restrictive - add delay between requests
            request_delay = 0.1  # 100ms delay for OSM
        else:
            # Other services are generally more permissive
            request_delay = 0.02  # 20ms delay for others
        
        # Track progress and statistics
        total_tiles_cached = 0
        consecutive_errors = 0
        max_consecutive_errors = 20
        error_types = {}
        
        # Download all missing tiles
        for tile_index, tile_coords in enumerate(tiles_to_download):
            x, y, zoom = tile_coords
            
            # Apply rate limiting (except for first request)
            if tile_index > 0:
                time.sleep(request_delay)
            
            # Debug: Show what we're trying to download
            if MAP_TILE_CACHING_DEBUG:
                print(f"\nDownloading tile {tile_index + 1}/{total_to_download}: x={x}, y={y}, zoom={zoom}")
            
            try:
                # Use the cartopy API method to cache the tile
                tile_location = (x, y, zoom)
                if MAP_TILE_CACHING_DEBUG:
                    print(f"  Trying CartoPy get_image for: {tile_location}")
                img = map_tiles.get_image(tile_location)
                total_tiles_cached += 1
                consecutive_errors = 0  # Reset consecutive error counter on success
                if MAP_TILE_CACHING_DEBUG:
                    print(f"  -> CartoPy get_image SUCCESS")
            except Exception as img_error:
                # Handle any errors from CartoPy
                error_str = str(img_error)
                if MAP_TILE_CACHING_DEBUG:
                    print(f"  -> CartoPy get_image FAILED: {error_str}")
                
                if log_callback:
                    log_callback(f"Error fetching tile image with get_image: {img_error}")
                
                consecutive_errors += 1
                
                # Record error type
                error_type = type(img_error).__name__
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1
            
            # If too many consecutive errors, pause longer
            if consecutive_errors >= max_consecutive_errors:
                if log_callback:
                    log_callback(f"Too many consecutive errors ({consecutive_errors}). Pausing for 10 seconds...")
                time.sleep(10)  # Longer pause when many errors
                consecutive_errors = 0
            
            # Update progress based on downloaded tiles only
            download_progress = (tile_index + 1) / total_to_download
            progress_percent = int(download_progress * 100)
            
            # Create progress text for the progress bar
            progress_text = f"Downloading tile {tile_index + 1}/{total_to_download}"
            
            # Update the progress bar with both percentage and text
            if progress_callback:
                progress_callback("progress_bar_tiles", progress_percent, progress_text)
        
        # Phase 4: Second pass to catch any failed downloads
        if debug_callback:
            debug_callback("Running second tile cache pass.")
        
        # Check for any tiles that might have failed to download in the first pass
        tiles_still_missing = []
        for tile_coords in tiles_to_download:
            x, y, zoom = tile_coords
            if not is_tile_cached(cache_dir, x, y, zoom, map_style):
                tiles_still_missing.append(tile_coords)
        
        if tiles_still_missing:
            if debug_callback:
                debug_callback(f"Second pass: Found {len(tiles_still_missing)} tiles still missing, retrying")
            
            # Reset error tracking for second pass
            consecutive_errors = 0
            
            # Retry downloading missing tiles
            for tile_index, tile_coords in enumerate(tiles_still_missing):
                x, y, zoom = tile_coords
                
                # Apply rate limiting (except for first request)
                if tile_index > 0:
                    time.sleep(request_delay)
                
                # Debug: Show what we're trying to download
                if MAP_TILE_CACHING_DEBUG:
                    print(f"\nSecond pass - Downloading tile {tile_index + 1}/{len(tiles_still_missing)}: x={x}, y={y}, zoom={zoom}")
                
                try:
                    # Use the cartopy API method to cache the tile
                    tile_location = (x, y, zoom)
                    if MAP_TILE_CACHING_DEBUG:
                        print(f"  Trying CartoPy get_image for: {tile_location}")
                    img = map_tiles.get_image(tile_location)
                    total_tiles_cached += 1
                    consecutive_errors = 0  # Reset consecutive error counter on success
                    if MAP_TILE_CACHING_DEBUG:
                        print(f"  -> CartoPy get_image SUCCESS (second pass)")
                except Exception as img_error:
                    # Handle any errors from CartoPy
                    error_str = str(img_error)
                    if MAP_TILE_CACHING_DEBUG:
                        print(f"  -> CartoPy get_image FAILED (second pass): {error_str}")
                    
                    if log_callback:
                        log_callback(f"Second pass error fetching tile image with get_image: {img_error}")
                    
                    consecutive_errors += 1
                    
                    # Record error type
                    error_type = type(img_error).__name__
                    if error_type not in error_types:
                        error_types[error_type] = 0
                    error_types[error_type] += 1
                
                # If too many consecutive errors, pause longer
                if consecutive_errors >= max_consecutive_errors:
                    if log_callback:
                        log_callback(f"Second pass: Too many consecutive errors ({consecutive_errors}). Pausing for 10 seconds")
                    time.sleep(10)  # Longer pause when many errors
                    consecutive_errors = 0
        else:
            if debug_callback:
                debug_callback("Second pass: All tiles successfully cached in first pass.")
        
        # Ensure the progress bar is at 100% when caching is complete
        if progress_callback:
            progress_callback("progress_bar_tiles", 100, "Tile caching complete")
        
        # Summarize error types if any
        if error_types and log_callback:
            log_callback("Error summary by type:")
            for error_type, count in error_types.items():
                log_callback(f"  - {error_type}: {count} occurrences")
        
        if debug_callback:
            debug_callback(f"Tile caching complete: {total_tiles_cached} downloaded, {tiles_already_cached} already cached, {total_required_tiles} total")
        
        # Note: Background sync removed to preserve tile quality - sync only on startup/exit
        
        return {
            'total_tiles_cached': total_tiles_cached,
            'total_tiles_skipped': tiles_already_cached,
            'total_tiles_to_cache': total_required_tiles,
            'total_bboxes': total_bboxes,
            'error_types': error_types,
            'success': True
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in map tile pre-caching: {str(e)}")
        return None


def cache_map_tiles(json_data=None, combined_route_data=None, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None):
    """
    Cache map tiles needed for video generation.
    
    Args:
        json_data (dict, optional): Job data containing video parameters
        combined_route_data (dict, optional): Combined route data containing gpx_time_per_video_time
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting map tile caching")
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 0, "Starting map tile caching")
        
        # Load job data if not provided
        if json_data is None:
            data_file = Path("data.json")
            if not data_file.exists():
                if log_callback:
                    log_callback("Error: data.json not found")
                return None
            
            with open(data_file, 'r') as f:
                json_data = json.load(f)
        
        # Use gpx_time_per_video_time from combined_route_data if available, otherwise calculate it
        if combined_route_data and 'gpx_time_per_video_time' in combined_route_data:
            gpx_time_per_video_time = combined_route_data['gpx_time_per_video_time']
            video_fps = float(json_data.get('video_fps', 30))
            route_time_per_frame = gpx_time_per_video_time / video_fps
            
            if debug_callback:
                # debug_callback(f"Using gpx_time_per_video_time from combined_route_data: {gpx_time_per_video_time}")
                # debug_callback(f"Calculated route_time_per_frame: {route_time_per_frame:.6f} seconds")
                pass
        else:
            # Fallback to calculating route time per frame
            route_time_per_frame = calculate_route_time_per_frame(json_data, log_callback)
            
            if route_time_per_frame is None:
                if log_callback:
                    log_callback("Error: Could not calculate route time per frame")
                return None
            
            if debug_callback:
                debug_callback(f"Route time per frame: {route_time_per_frame:.4f} seconds")
        
        # Calculate unique bounding boxes across all frames
        unique_bounding_boxes = calculate_unique_bounding_boxes(json_data, route_time_per_frame, log_callback, max_workers, combined_route_data, debug_callback)
        
        if unique_bounding_boxes is None:
            if log_callback:
                log_callback("Error: Could not calculate unique bounding boxes")
            return None
        
        if debug_callback:
            debug_callback(f"Found {len(unique_bounding_boxes)} unique bounding boxes")
        
        # Pre-cache map tiles for all unique bounding boxes
        cache_result = pre_cache_map_tiles_for_video(unique_bounding_boxes, json_data, progress_callback, log_callback, debug_callback)
        
        if cache_result is None:
            if log_callback:
                log_callback("Error: Could not cache map tiles")
            return None
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 100, "Map tile caching complete")
        
        return {
            'route_time_per_frame': route_time_per_frame,
            'unique_bounding_boxes': unique_bounding_boxes,
            'cache_result': cache_result,
            'success': True
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in map tiles caching: {str(e)}")
        return None