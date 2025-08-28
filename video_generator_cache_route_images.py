"""
Route image caching system for video generation optimization.
This module pre-renders individual routes as numpy arrays to improve performance in final zoom mode.
"""

import os
import json
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for route image caching

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pickle
import math
from pathlib import Path
from video_generator_calculate_bounding_boxes import load_final_bounding_box
from video_generator_create_single_frame_utils import hex_to_rgba, _get_resolution_scale_factor


def cache_route_images(combined_route_data, json_data, progress_callback=None, log_callback=None, shared_route_cache=None):
    """
    Cache individual route images as numpy arrays for final zoom mode optimization.
    
    This function pre-renders each track of each route as RGB numpy arrays and stores them
    in shared memory for faster composition during video frame generation.
    
    Args:
        combined_route_data (dict): Combined route data containing all routes
        json_data (dict): Job configuration data
        progress_callback (callable, optional): Function to call with progress updates
        log_callback (callable, optional): Function to call for logging messages
        shared_route_cache (dict, optional): Shared memory cache for route images
    
    Returns:
        dict: Metadata about cached route images, or None if caching failed or skipped
    """
    try:
        # Check if we should perform route caching (only for final zoom mode)
        zoom_mode = json_data.get('zoom_mode', 'dynamic')
        
        if zoom_mode != 'final':
            if log_callback:
                log_callback("Skipping route image caching - only applicable to 'final' zoom mode")
            return None
        
        # Check if shared route cache is available
        if shared_route_cache is None:
            if log_callback:
                log_callback("Skipping route image caching - shared route cache not available")
            return None
        
        if log_callback:
            log_callback("Starting route image caching for final zoom mode optimization (shared memory)...")
        
        # Load the final bounding box
        final_bbox = load_final_bounding_box(log_callback=log_callback)
        if final_bbox is None:
            if log_callback:
                log_callback("Error: Could not load final bounding box for route caching")
            return None
        
        if log_callback:
            log_callback(f"Using final bounding box: {final_bbox}")
        
        # Get all routes from combined_route_data
        all_routes = combined_route_data.get('all_routes', [])
        if not all_routes:
            # Fallback for single route mode
            if 'combined_route' in combined_route_data:
                all_routes = [combined_route_data]
            else:
                if log_callback:
                    log_callback("Error: No routes found in combined_route_data")
                return None
        
        # Get video parameters
        width = int(json_data.get('video_resolution_x', 1920))
        height = int(json_data.get('video_resolution_y', 1080))
        line_thickness = float(json_data.get('line_width', 3))
        dpi = 100
        figsize = (width / dpi, height / dpi)
        
        # Calculate effective line width (same logic as in frame generation)
        resolution_scale = _get_resolution_scale_factor(json_data)
        base_line_scale = 1.0  # Base scale for 1080p
        line_scale = base_line_scale * resolution_scale
        desired_pixels = line_thickness * line_scale
        effective_line_width = desired_pixels * 72 / 100  # dpi fixed to 100
        
        # Build filename to RGBA color mapping
        filename_to_rgba = {}
        track_objects = json_data.get('track_objects', [])
        for track_obj in track_objects:
            filename = track_obj.get('filename', '')
            if filename.endswith('.gpx'):
                filename = filename[:-4]  # Remove .gpx extension
            color_hex = track_obj.get('color', '#ff0000')
            filename_to_rgba[filename] = hex_to_rgba(color_hex)
        
        # Cache metadata to track all cached route images
        cache_metadata = {
            'route_images': [],  # List of cached route image info
            'bbox': final_bbox,
            'resolution': (width, height),
            'effective_line_width': effective_line_width
        }
        
        total_routes = len(all_routes)
        route_image_count = 0
        
        # Process each route
        for loop_index, route_data in enumerate(all_routes):
            combined_route = route_data.get('combined_route', [])
            if not combined_route:
                continue
            
            # Get the actual route_index from the route data (not the loop index)
            actual_route_index = route_data.get('route_index', 0)
            route_name = route_data.get('track_name', f'Route {loop_index}')
            
            if log_callback:
                log_callback(f"Caching route {loop_index + 1}/{total_routes}: '{route_name}' ({len(combined_route)} points, route_index={actual_route_index})")
            
            # # Update progress
            # if progress_callback:
            #     progress_percentage = int((loop_index / total_routes) * 90)  # Leave 10% for final processing
            #     progress_text = f"Caching route images: {loop_index + 1}/{total_routes}"
            #     progress_callback("progress_bar_main", progress_percentage, progress_text)
            
            # Split route points into tracks based on new_route flag (same logic as frame generation)
            tracks = []
            current_track = []
            
            for point in combined_route:
                if len(point) >= 7:  # Check for minimum length
                    new_route_flag = point[6]  # New route flag at index 6
                    if new_route_flag and current_track:  # If new_route is True and we have points
                        tracks.append(current_track)
                        current_track = []
                current_track.append(point)
            
            # Add the last track if not empty
            if current_track:
                tracks.append(current_track)
            
            # Cache each track as a separate image
            for track_index, track in enumerate(tracks):
                if not track:  # Skip empty tracks
                    continue
                
                # Cache this track as an individual route image using the actual route_index
                cached_image_info = _cache_single_track_image(
                    track, actual_route_index, track_index, final_bbox, 
                    width, height, dpi, effective_line_width, 
                    filename_to_rgba, shared_route_cache, log_callback
                )
                
                if cached_image_info:
                    cache_metadata['route_images'].append(cached_image_info)
                    route_image_count += 1
        
        # Final progress update
        if progress_callback:
            progress_callback("progress_bar_combined_route", 100, "Route image caching completed")
        
        if log_callback:
            log_callback(f"Route image caching completed: {route_image_count} track images cached from {total_routes} routes in shared memory")
        
        # OPTIMIZATION: Build track-to-cache-key mapping once and store in shared cache
        # This prevents rebuilding the same mapping for every frame during video generation
        track_to_cached_image = {}
        for cache_key in shared_route_cache.keys():
            if cache_key.startswith('route_') and '_track_' in cache_key:
                # Parse route_index and track_index from key like "route_X_track_Y"
                try:
                    parts = cache_key.split('_')
                    if len(parts) >= 4:  # route_X_track_Y
                        route_index = int(parts[1])
                        track_index = int(parts[3])
                        track_to_cached_image[(route_index, track_index)] = cache_key
                except (ValueError, IndexError):
                    continue  # Skip malformed keys
        
        # Store the mapping in shared cache for frame generation to use
        shared_route_cache['_track_to_cached_image_mapping'] = track_to_cached_image
        
        if log_callback:
            log_callback(f"Built track-to-cache mapping with {len(track_to_cached_image)} entries for frame generation optimization")
        
        return cache_metadata
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in route image caching: {str(e)}")
        return None


def _cache_single_track_image(track, route_index, track_index, bbox, width, height, dpi, 
                             effective_line_width, filename_to_rgba, shared_route_cache, log_callback=None):
    """
    Cache a single track as a numpy array image.
    
    Args:
        track (list): List of track points
        route_index (int): Index of the route this track belongs to
        track_index (int): Index of this track within the route
        bbox (tuple): Bounding box (lon_min, lon_max, lat_min, lat_max)
        width (int): Video width in pixels
        height (int): Video height in pixels
        dpi (int): DPI for matplotlib
        effective_line_width (float): Calculated line width
        filename_to_rgba (dict): Filename to RGBA color mapping
        shared_route_cache (dict): Shared memory cache for route images
        log_callback (callable, optional): Function for logging
    
    Returns:
        dict: Information about the cached image, or None if failed
    """
    try:
        figsize = (width / dpi, height / dpi)
        
        # Create matplotlib figure (same setup as frame generation)
        plt.rcParams['figure.constrained_layout.use'] = False
        fig = plt.figure(figsize=figsize, facecolor='none', frameon=False)  # Use 'none' for transparency
        
        # Create axes that fill the entire figure with no padding or margins
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()  # Turn off axis
        fig.add_axes(ax)
        
        # Convert GPS bounding box to Web Mercator coordinates (same as video frame generation)
        def _gps_to_web_mercator(lon, lat):
            x = lon * 20037508.34 / 180
            y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
            y = y * 20037508.34 / 180
            return x, y
        
        lon_min, lon_max, lat_min, lat_max = bbox
        x_min, y_min = _gps_to_web_mercator(lon_min, lat_min)
        x_max, y_max = _gps_to_web_mercator(lon_max, lat_max)
        mercator_bbox = (x_min, x_max, y_min, y_max)
        
        # Set transparent background using Web Mercator coordinates
        ax.set_xlim(mercator_bbox[0], mercator_bbox[1])
        ax.set_ylim(mercator_bbox[2], mercator_bbox[3])
        ax.set_facecolor('none')  # Set axes background to transparent
        
        # Extract coordinates and filename for this track
        track_lats = [point[1] for point in track]  # Latitude at index 1
        track_lons = [point[2] for point in track]  # Longitude at index 2
        filename = track[0][7] if len(track[0]) > 7 else None
        
        # Get color for this track
        if filename and filename_to_rgba and filename in filename_to_rgba:
            rgba_color = filename_to_rgba[filename]
#            rgba_color = (1.0, 1.0, 0.0, 1.0)  # Debug color - remove this to return to file-specific colors
        else:
            rgba_color = (1.0, 0.0, 0.0, 1.0)  # Default red if no color mapping found
        
        # Convert GPS coordinates to Web Mercator for plotting (same as video frame generation)
        mercator_coords = [_gps_to_web_mercator(lon, lat) for lon, lat in zip(track_lons, track_lats)]
        mercator_track_lons = [coord[0] for coord in mercator_coords]
        mercator_track_lats = [coord[1] for coord in mercator_coords]
        
        # OPTIMIZATION: Use LineCollection with single color and width for better performance
        from matplotlib.collections import LineCollection
        
        # Create line segment for this track
        segment = list(zip(mercator_track_lons, mercator_track_lats))
        
        # Create and add LineCollection with single color and width
        line_collection = LineCollection(
            [segment],
            color=rgba_color,  # Single color for this track
            linewidth=effective_line_width  # Single width for this track
        )
        ax.add_collection(line_collection)
        
        # Convert figure to numpy array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        track_array = np.asarray(buf)
        
        # Keep RGBA format for proper transparency handling during compositing
        track_array_rgba = track_array  # Keep all 4 channels (RGBA)
        
        # Clean up matplotlib figure
        plt.close(fig)
        
        # Store the track image in shared memory cache (no disk saving for better performance)
        cache_key = f"route_{route_index}_track_{track_index}"
        shared_route_cache[cache_key] = track_array_rgba
        
        # Return metadata about this cached image
        cached_image_info = {
            'cache_key': cache_key,
            'route_index': route_index,
            'track_index': track_index,
            'track_filename': filename,
            'rgba_color': rgba_color,
            'num_points': len(track)
        }
        
        return cached_image_info
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error caching track image for route {route_index}, track {track_index}: {str(e)}")
        # Clean up matplotlib figure on error
        try:
            plt.close(fig)
        except:
            pass
        return None 


def load_cached_route_metadata(log_callback=None):
    """
    Load cached route metadata for frame generation optimization.
    
    Args:
        log_callback (callable, optional): Function for logging
    
    Returns:
        dict: Cache metadata containing route image info, or None if not available
    """
    try:
        cache_dir = Path("temporary files/route_cache")
        metadata_file = cache_dir / "cache_metadata.pkl"
        
        if not metadata_file.exists():
            return None  # No cached routes available
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        if log_callback:
            route_count = len(metadata.get('route_images', []))
            log_callback(f"Loaded cached route metadata: {route_count} cached route images available")
        
        return metadata
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error loading cached route metadata: {str(e)}")
        return None


def load_cached_route_image(cache_filepath, log_callback=None):
    """
    Load a specific cached route image from file.
    
    Args:
        cache_filepath (str): Full path to the cached route image file
        log_callback (callable, optional): Function for logging
    
    Returns:
        numpy.ndarray: RGB image array, or None if loading failed
    """
    try:
        with open(cache_filepath, 'rb') as f:
            route_image = pickle.load(f)
        
        return route_image
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error loading cached route image from {cache_filepath}: {str(e)}")
        return None


def is_track_fully_completed(track, target_time, tail_length, gpx_time_per_video_time, route_specific_tail_info=None, route_index=None):
    """
    Determine if a track segment should use cached imagery.
    
    For route segments to use cached images, ALL points in the segment must be 
    completely in the past relative to the current target time. This prevents
    showing future parts of the route prematurely.
    
    Args:
        track (list): List of track points
        target_time (float): Target elapsed time for current frame (in video seconds)
        tail_length (int): Tail length in video seconds  
        gpx_time_per_video_time (float): GPX seconds per video second ratio
        route_specific_tail_info (dict, optional): Dictionary containing route-specific tail information for simultaneous mode
        route_index (int, optional): Route index for simultaneous mode
    
    Returns:
        bool: True if ALL points in track segment are in the past, False otherwise
    """
    if not track or not gpx_time_per_video_time or target_time is None:
        return False
    
    # SIMULTANEOUS MODE FIX: Use route-specific timing if available
    if route_specific_tail_info and route_index is not None and route_index in route_specific_tail_info:
        # Use route-specific timing for simultaneous mode
        route_info = route_specific_tail_info[route_index]
        route_end_time = route_info['route_end_time']
        route_delay_seconds = route_info['route_delay_seconds']
        
        # Calculate current route time accounting for staggered start delay
        current_route_time = target_time * gpx_time_per_video_time
        
        # FIX: Route should be cached when it ends AND tail has faded out
        # This prevents cached routes from appearing while tails are still visible
        tail_duration_route = tail_length * gpx_time_per_video_time
        route_completion_time = route_end_time + tail_duration_route
        
        # Route is completed if current time is past its completion time (including tail fade-out)
        is_completed = current_route_time > route_completion_time
        
        return is_completed
    else:
        # Original logic for sequential mode or fallback
        # Convert target_time from video seconds to route seconds
        target_route_time = target_time * gpx_time_per_video_time   
        max_accumulated_time = track[-1][4]  # Latest point in track (last element)
        
        # FIX: For sequential mode, also cache when route ends AND tail fades out
        # This prevents cached routes from appearing while tails are still visible
        tail_duration_route = tail_length * gpx_time_per_video_time
        track_completion_time = max_accumulated_time + tail_duration_route
        
        # Only use cached image if ALL points in track are sufficiently in the past
        is_completed = target_route_time > track_completion_time
        
        return is_completed


def composite_cached_route_image(ax, route_image, bbox):
    """
    Composite a cached route image onto the matplotlib axes.
    
    Args:
        ax (matplotlib.axes.Axes): Regular matplotlib axes for compositing
        route_image (numpy.ndarray): RGBA route image array
        bbox (tuple): Bounding box (lon_min, lon_max, lat_min, lat_max) in GPS coordinates
    """
    try:
        # Convert GPS bounding box to Web Mercator coordinates (same as frame generation)
        
        def _gps_to_web_mercator(lon, lat):
            x = lon * 20037508.34 / 180
            y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
            y = y * 20037508.34 / 180
            return x, y
        
        lon_min, lon_max, lat_min, lat_max = bbox
        x_min, y_min = _gps_to_web_mercator(lon_min, lat_min)
        x_max, y_max = _gps_to_web_mercator(lon_max, lat_max)
        mercator_bbox = (x_min, x_max, y_min, y_max)
        
        # Composite the cached route image onto regular matplotlib axes using Web Mercator coordinates
        ax.imshow(route_image, extent=mercator_bbox, 
                 aspect='auto', interpolation='nearest', zorder=1)  # Bottom layer for cached routes
        
    except Exception as e:
        print(f"Error compositing cached route image: {str(e)}") 