"""
Video generation map image caching for the Route Squiggler render client.
This module handles caching map images for each unique bounding box needed for video frame generation.
"""

import json
import os
import time
import cartopy.crs as ccrs
from pathlib import Path
from multiprocessing import Pool, Manager
from video_generator_calculate_bounding_boxes import calculate_route_time_per_frame, calculate_unique_bounding_boxes
from video_generator_coordinate_encoder import encode_coords
from image_generator_maptileutils import create_map_tiles, detect_zoom_level, set_cache_directory
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for map image caching

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from video_generator_create_single_frame import _gps_to_web_mercator, _convert_bbox_to_web_mercator


def create_map_image_worker(args):
    """
    Worker function to create a map image for a single bounding box.
    
    Args:
        args (tuple): Contains (bbox_index, bbox, json_data, shared_progress_dict, shared_map_cache)
    
    Returns:
        dict: Result information for this bounding box
    """
    bbox_index, bbox, json_data, shared_progress_dict, shared_map_cache = args
    
    try:
        # Get bounding box coordinates
        lon_min, lon_max, lat_min, lat_max = bbox
        
        # Extract parameters from json_data
        map_style = json_data.get('map_style', 'osm')
        map_transparency = float(json_data.get('map_transparency', 0))  # Transparency parameter
        map_opacity = 100 - map_transparency  # Convert transparency to opacity
        video_resolution_x = int(json_data.get('video_resolution_x', 1920))
        video_resolution_y = int(json_data.get('video_resolution_y', 1080))
        max_map_tiles = 100  # As specified in requirements
        
        # Set up map image cache directory with subfolder for the current map style and opacity
        # Note: No longer creating disk directories since we're using shared memory exclusively

        # Generate a unique filename for this bounding box using the coordinate encoder
        encoded_bbox = encode_coords(lon_min, lon_max, lat_min, lat_max)
        
        # Check if image already exists in shared cache
        if shared_map_cache is not None and encoded_bbox in shared_map_cache:
            # Update shared progress counter
            with shared_progress_dict['lock']:
                shared_progress_dict['completed'] += 1
                current_progress = shared_progress_dict['completed']
                total_work = shared_progress_dict['total']
            
            return {
                'bbox_index': bbox_index,
                'bbox': bbox,
                'success': True,
                'current_progress': current_progress,
                'total_work': total_work,
                'message': f"Map image already exists in shared cache for bbox {bbox_index + 1}",
                'was_cached': True
            }

        # Video resolution and DPI settings
        width, height = video_resolution_x, video_resolution_y
        dpi = 100  # Hardcoded as specified
        figsize = (width / dpi, height / dpi)
        
        # Set up the cartopy cache directory
        set_cache_directory(map_style)
        
        # Get and display the current cache directory being used
        import cartopy
        current_cache_dir = cartopy.config.get('cache_dir', 'Not set')
        
        # Write debug info to debug.log since print doesn't work in multiprocess
        debug_msg = f"Map image creation for bbox {bbox_index + 1}: Using map tile cache directory: {current_cache_dir}\n"
        
        # Show expected vs actual cache path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        expected_base_cache = os.path.join(current_directory, 'map tile cache')
        debug_msg += f"  Expected base cache directory: {expected_base_cache}\n"
        debug_msg += f"  Map style: {map_style}\n"
        
        # Verify cache directory exists and show some stats
        if os.path.exists(current_cache_dir):
            try:
                cache_contents = os.listdir(current_cache_dir)
                tile_files = [f for f in cache_contents if f.endswith('.npy')]
                debug_msg += f"  Cache directory exists with {len(cache_contents)} items ({len(tile_files)} tile files)\n"
                
                # Show first few tile files as examples
                if tile_files:
                    debug_msg += f"  Example tile files: {tile_files[:3]}\n"
                
                # Show subdirectories if any
                subdirs = [f for f in cache_contents if os.path.isdir(os.path.join(current_cache_dir, f))]
                if subdirs:
                    debug_msg += f"  Subdirectories found: {subdirs}\n"
                    # Check first subdirectory for tile files
                    first_subdir = os.path.join(current_cache_dir, subdirs[0])
                    if os.path.exists(first_subdir):
                        subdir_contents = os.listdir(first_subdir)
                        subdir_tiles = [f for f in subdir_contents if f.endswith('.npy')]
                        debug_msg += f"  Subdirectory '{subdirs[0]}' contains {len(subdir_tiles)} tile files\n"
                        if subdir_tiles:
                            debug_msg += f"  Example tiles in subdirectory: {subdir_tiles[:3]}\n"
                        
            except Exception as e:
                debug_msg += f"  Error checking cache directory contents: {e}\n"
        else:
            debug_msg += f"  WARNING: Cache directory does not exist!\n"
            # Check if base cache directory exists
            if os.path.exists(expected_base_cache):
                debug_msg += f"  Base cache directory exists: {expected_base_cache}\n"
                try:
                    base_contents = os.listdir(expected_base_cache)
                    debug_msg += f"  Base cache contains: {base_contents}\n"
                except Exception as e:
                    debug_msg += f"  Error listing base cache: {e}\n"
            else:
                debug_msg += f"  Base cache directory does not exist: {expected_base_cache}\n"
        
        # Write debug info to debug.log file
        try:
            # Use absolute path to ensure debug.log is created in project root
            project_root = os.path.dirname(os.path.abspath(__file__))
            debug_log_path = os.path.join(project_root, 'debug.log')
            with open(debug_log_path, 'a', encoding='utf-8') as debug_file:
                debug_file.write(debug_msg)
                debug_file.flush()
        except Exception as e:
            pass  # Ignore debug file write errors
        
        # Map opacity from percentage to alpha value
        map_alpha = map_opacity / 100.0
        
        # Create map tiles with the specified style
        map_tiles = create_map_tiles(map_style)
        
        # Detect appropriate zoom level for this bounding box
        zoom_levels = detect_zoom_level((lon_min, lon_max, lat_min, lat_max), max_tiles=max_map_tiles, map_style=map_style)
        
        if zoom_levels:
            # Use the highest zoom level (most detailed) that fits within max_tiles
            zoom_level = max(zoom_levels)
        else:
            # If no suitable zoom found, use a conservative zoom level
            zoom_level = 10  # Default fallback
        
        # Create the figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=map_tiles.crs)
        
        # Set the map extent explicitly
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Force the aspect ratio
        ax.set_aspect('auto')
        
        # Add map tiles to the map with calculated zoom level and apply opacity preference
        ax.add_image(map_tiles, zoom_level, alpha=map_alpha)
        
        # Apply tight layout with zero padding to ensure content fits within the figure
        plt.tight_layout(pad=0)
        
        # Explicitly set the figure size in inches to ensure correct dimensions
        fig.set_size_inches(figsize)
        
        # Convert figure to numpy array directly (much faster than PNG)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)
        
        # Convert RGBA to RGB (remove alpha channel)
        img_array = img_array[:, :, :3]
        
        # Store in shared memory cache (no disk saving for better performance)
        if shared_map_cache is not None:
            shared_map_cache[encoded_bbox] = img_array
        
        plt.close(fig)
        
        # Extra cleanup to prevent memory bloat
        plt.clf()
        plt.close('all')
        
        # Update shared progress counter
        with shared_progress_dict['lock']:
            shared_progress_dict['completed'] += 1
            current_progress = shared_progress_dict['completed']
            total_work = shared_progress_dict['total']
        
        return {
            'bbox_index': bbox_index,
            'bbox': bbox,
            'success': True,
            'current_progress': current_progress,
            'total_work': total_work,
            'message': f"Created map image for bbox {bbox_index + 1}",
            'was_cached': False
        }
        
    except Exception as e:
        # Clean up matplotlib in case of error
        plt.close('all')
        
        # Update shared progress counter even on error
        with shared_progress_dict['lock']:
            shared_progress_dict['completed'] += 1
            current_progress = shared_progress_dict['completed']
            total_work = shared_progress_dict['total']
        
        return {
            'bbox_index': bbox_index,
            'bbox': bbox,
            'success': False,
            'error': str(e),
            'current_progress': current_progress,
            'total_work': total_work,
            'message': f"Error creating map image for bbox {bbox_index + 1}: {str(e)}",
            'was_cached': False
        }


def cache_map_images_for_video(unique_bounding_boxes, json_data, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None):
    """
    Cache map images for all unique bounding boxes required for video generation.
    Uses shared memory cache exclusively for maximum performance.
    
    Args:
        unique_bounding_boxes (set): Set of unique bounding boxes as tuples (lon_min, lon_max, lat_min, lat_max)
        json_data (dict): Job data containing video parameters
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for storing map images (required)
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting map image caching for video generation")
        
        # Validate that shared map cache is provided
        if shared_map_cache is None:
            if log_callback:
                log_callback("Error: Shared map cache is required for map image caching")
            return None
        
        if progress_callback:
            progress_callback("progress_bar_map_images", 0, "Preparing map image caching")
        
        # Convert bounding boxes to list for processing
        bbox_list = list(unique_bounding_boxes)
        total_bboxes = len(bbox_list)
        
        if debug_callback:
            debug_callback(f"Processing {total_bboxes} unique bounding boxes for map images")
        
        # Determine number of workers to use
        if max_workers is None:
            max_workers = min(4, total_bboxes)  # Default to 4 workers or number of bboxes, whichever is smaller
        
        # Limit workers to available work
        workers_to_use = min(max_workers, total_bboxes)
        
        if debug_callback:
            debug_callback(f"Using {workers_to_use} worker processes for map image generation")
        
        # Handle case where there are no bounding boxes
        if total_bboxes == 0:
            if debug_callback:
                debug_callback("No bounding boxes to process")
            if progress_callback:
                progress_callback("progress_bar_map_images", 100, "No map images needed")
            return {
                'total_images_created': 0,
                'total_bboxes': 0,
                'success': True,
                'results': []
            }
        
        # Create shared progress tracking
        with Manager() as manager:
            shared_progress_dict = manager.dict()
            shared_progress_dict['completed'] = 0
            shared_progress_dict['total'] = total_bboxes
            shared_progress_dict['lock'] = manager.Lock()
            
            # Create shared memory cache for map images
            if shared_map_cache is None:
                shared_map_cache = manager.dict()
            
            # Prepare work arguments for each worker
            work_args = []
            for bbox_index, bbox in enumerate(bbox_list):
                work_args.append((bbox_index, bbox, json_data, shared_progress_dict, shared_map_cache))
            
            # Start multiprocessing
            results = []
            successful_images = 0
            
            if workers_to_use == 1:
                # Single-threaded execution for debugging or when only one worker
                if debug_callback:
                    debug_callback("Using single-threaded execution")
                
                for work_arg in work_args:
                    result = create_map_image_worker(work_arg)
                    results.append(result)
                    
                    if result['success']:
                        successful_images += 1
                    
                    # Update progress
                    progress_percent = int((result['current_progress'] / result['total_work']) * 100)
                    progress_text = f"Creating map image {result['current_progress']}/{result['total_work']}"
                    
                    if progress_callback:
                        progress_callback("progress_bar_map_images", progress_percent, progress_text)
                    
                    # if log_callback:
                    #    log_callback(result['message'])
            else:
                # Multi-threaded execution
                if debug_callback:
                    debug_callback(f"Using multi-threaded execution with {workers_to_use} workers")
                
                with Pool(processes=workers_to_use) as pool:
                    # Submit all work
                    if progress_callback:
                        progress_callback("progress_bar_map_images", 0, "Starting map image creation")
                    
                    # Process results as they complete
                    for result in pool.imap_unordered(create_map_image_worker, work_args):
                        results.append(result)
                        
                        if result['success']:
                            successful_images += 1
                        
                        # Update progress
                        progress_percent = int((result['current_progress'] / result['total_work']) * 100)
                        progress_text = f"Creating map image {result['current_progress']}/{result['total_work']}"
                        
                        if progress_callback:
                            progress_callback("progress_bar_map_images", progress_percent, progress_text)
                        
                        # if log_callback:
                        #     log_callback(result['message'])
        
        # Final progress update
        if progress_callback:
            progress_callback("progress_bar_map_images", 100, "Map image caching complete")
        
        # Summary logging
        failed_images = total_bboxes - successful_images
        if debug_callback:
            debug_callback(f"Map image caching complete: {successful_images} successful, {failed_images} failed, {total_bboxes} total")
        
        return {
            'total_images_created': successful_images,
            'total_images_failed': failed_images,
            'total_bboxes': total_bboxes,
            'success': True,
            'results': results
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in map image caching: {str(e)}")
        return None


def cache_map_images(json_data=None, combined_route_data=None, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None):
    """
    Cache map images needed for video generation using shared memory exclusively.
    
    Args:
        json_data (dict, optional): Job data containing video parameters
        combined_route_data (dict, optional): Combined route data containing gpx_time_per_video_time
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for storing map images (required)
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting map image caching")
        
        # Validate that shared map cache is provided
        if shared_map_cache is None:
            if log_callback:
                log_callback("Error: Shared map cache is required for map image caching")
            return None
        
        if progress_callback:
            progress_callback("progress_bar_map_images", 0, "Starting map image caching")
        
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
            route_time_per_frame = calculate_route_time_per_frame(json_data, combined_route_data, log_callback, debug_callback)
            
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
        
        # Cache map images for all unique bounding boxes
        cache_result = cache_map_images_for_video(unique_bounding_boxes, json_data, progress_callback, log_callback, debug_callback, max_workers, shared_map_cache)
        
        if cache_result is None:
            if log_callback:
                log_callback("Error: Could not cache map images")
            return None
        
        if progress_callback:
            progress_callback("progress_bar_map_images", 100, "Map image caching complete")
        
        return {
            'route_time_per_frame': route_time_per_frame,
            'unique_bounding_boxes': unique_bounding_boxes,
            'cache_result': cache_result,
            'success': True
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in map image caching: {str(e)}")
        return None
