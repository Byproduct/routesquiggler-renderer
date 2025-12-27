"""
Multiprocessing utilities for generating images at different zoom levels.
"""

import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from queue import Empty
import traceback
import time
import cartopy.crs as ccrs
import ftplib
from io import BytesIO
from PIL import Image

# Increase PIL image size limit to 200 megapixels to handle large renders
Image.MAX_IMAGE_PIXELS = 200_000_000

from image_generator_maptileutils import create_map_tiles, set_cache_directory
from image_generator_postprocess import add_stamp_to_plot, add_legend_to_plot, optimize_png_bytes, add_title_text_to_plot
from image_generator_utils import ImageGenerator, calculate_resolution_scale
from html_file_generator import generate_image_gallery_html
from write_log import write_log, write_debug_log
from speed_based_color import speed_based_color

class StatusUpdate:
    """Status update message from a worker process."""
    def __init__(self, zoom_level: int, status: str, error: Optional[str] = None, textfield: bool = True):
        self.zoom_level = zoom_level
        self.status = status
        self.error = error
        self.textfield = textfield  # Whether to update the text field (log widget)
        self.timestamp = time.time()


def split_track_at_longitude_wrap(lats: List[float], lons: List[float]) -> List[Tuple[List[float], List[float]]]:
    """
    Split a track into segments when it crosses the 180°/-180° longitude boundary.
    
    Args:
        lats: List of latitude coordinates
        lons: List of longitude coordinates
        
    Returns:
        List of (lats_segment, lons_segment) tuples, where each segment doesn't cross the boundary
    """
    if len(lats) < 2 or len(lons) < 2:
        return [(lats, lons)]
    
    segments = []
    current_lats = [lats[0]]
    current_lons = [lons[0]]
    wrap_count = 0
    
    for i in range(1, len(lats)):
        prev_lon = lons[i-1]
        curr_lon = lons[i]
        
        # Check if we've crossed the 180°/-180° boundary
        # We need to handle both positive and negative longitude jumps
        lon_diff = curr_lon - prev_lon
        
        # If the difference is greater than 180 degrees, we've crossed the boundary
        # This handles cases like going from 179° to -179° (diff = -358°) or -179° to 179° (diff = 358°)
        if abs(lon_diff) > 180:
            wrap_count += 1
            # Save the current segment
            if len(current_lats) > 1:  # Only add segments with at least 2 points
                segments.append((current_lats.copy(), current_lons.copy()))
            
            # Start a new segment
            current_lats = [lats[i]]
            current_lons = [lons[i]]
        else:
            # Continue the current segment
            current_lats.append(lats[i])
            current_lons.append(lons[i])
    
    # Add the final segment
    if len(current_lats) > 1:
        segments.append((current_lats, current_lons))
    elif len(current_lats) == 1 and segments:  # Single point, add to last segment if it exists
        segments[-1][0].append(current_lats[0])
        segments[-1][1].append(current_lons[0])
    
    # If no segments were created (e.g., single point or no wrapping), return original
    if not segments:
        return [(lats, lons)]
    
    # Log if wrapping was detected
    if wrap_count > 0:
        write_log(f"Longitude wrapping detected: track split into {len(segments)} segments after {wrap_count} wrap(s)")
    
    return segments


def generate_images_parallel(
    zoom_levels: List[int],
    map_style: str,
    map_bounds: Tuple[float, float, float, float],
    track_coords_with_metadata: List[tuple],
    track_lookup: Dict[str, Any],
    map_transparency: int = 30,
    line_width: int = 3,
    resolution_x: int = 1920,
    resolution_y: int = 1080,
    legend: str = "off",
    statistics: str = "off",
    statistics_data: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    status_queue: Optional[mp.Queue] = None,
    storage_box_address: Optional[str] = None,
    storage_box_user: Optional[str] = None,
    storage_box_password: Optional[str] = None,
    job_id: Optional[str] = None,
    folder: Optional[str] = None,
    route_name: Optional[str] = None,
    max_workers: Optional[int] = None,
    route_points_data: Optional[Dict[str, Any]] = None
) -> Tuple[List[Tuple[int, bytes]], mp.Queue]:
    """
    Generate images for multiple zoom levels in parallel.
    Also generates and uploads an HTML gallery page.
    
    Args:
        max_workers: Maximum number of concurrent worker processes. If None, uses all zoom levels.
    """
    if status_queue is None:
        status_queue = mp.Queue()
    
    # Calculate which zoom level should be the thumbnail
    sorted_zoom_levels = sorted(zoom_levels)
    middle_index = (len(sorted_zoom_levels) - 1) // 2  # This gives us the lower middle for even numbers
    thumbnail_zoom_level = sorted_zoom_levels[middle_index]
    
    # Determine number of workers to use
    if max_workers is None:
        max_workers = len(zoom_levels)  # Use all zoom levels (original behavior)
    else:
        max_workers = min(max_workers, len(zoom_levels))  # Don't exceed number of zoom levels
    
    write_debug_log(f"Using {max_workers} worker processes for {len(zoom_levels)} zoom levels")
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Create a list to track which zoom levels are being processed
    zoom_queue = list(zoom_levels)
    active_processes = {}  # Maps process to zoom_level
    
    # Start initial batch of worker processes (up to max_workers)
    while len(active_processes) < max_workers and zoom_queue:
        zoom_level = zoom_queue.pop(0)
        process = mp.Process(
            target=_worker_process,
            args=(
                zoom_level,
                map_style,
                map_bounds,
                track_coords_with_metadata,
                track_lookup,
                map_transparency,
                line_width,
                resolution_x,
                resolution_y,
                legend,
                statistics,
                statistics_data,
                json_data,
                status_queue,
                result_queue,
                storage_box_address,
                storage_box_user,
                storage_box_password,
                job_id,
                folder,
                route_name,
                zoom_level == thumbnail_zoom_level,  # Pass thumbnail=True for middle zoom level
                sorted_zoom_levels,  # Pass all zoom levels
                route_points_data  # Pass RoutePoint data for speed-based coloring
            )
        )
        process.start()
        active_processes[process] = zoom_level
        write_debug_log(f"Started worker process for zoom level {zoom_level}")
    
    # Collect results and start new processes as others complete
    results = []
    completed_processes = 0
    total_processes = len(zoom_levels)
    
    while completed_processes < total_processes:
        try:
            # Get a result (this will block until a process completes)
            result = result_queue.get(timeout=600)  # 10 minute timeout per process
            if result[1] is not None:  # Only add successful results
                results.append(result)
            completed_processes += 1
            write_debug_log(f"Completed zoom level {result[0]} ({completed_processes}/{total_processes})")
            
            # Find and clean up the completed process
            completed_process = None
            for process, zoom_level in active_processes.items():
                if zoom_level == result[0]:
                    completed_process = process
                    break
            
            if completed_process:
                completed_process.join()
                completed_process.close()
                del active_processes[completed_process]
            
            # Start a new process if there are more zoom levels to process
            if zoom_queue:
                next_zoom_level = zoom_queue.pop(0)
                process = mp.Process(
                    target=_worker_process,
                    args=(
                        next_zoom_level,
                        map_style,
                        map_bounds,
                        track_coords_with_metadata,
                        track_lookup,
                        map_transparency,
                        line_width,
                        resolution_x,
                        resolution_y,
                        legend,
                        statistics,
                        statistics_data,
                        json_data,
                        status_queue,
                        result_queue,
                        storage_box_address,
                        storage_box_user,
                        storage_box_password,
                        job_id,
                        folder,
                        route_name,
                        next_zoom_level == thumbnail_zoom_level,
                        sorted_zoom_levels,
                        route_points_data  # Pass RoutePoint data for speed-based coloring
                    )
                )
                process.start()
                active_processes[process] = next_zoom_level
                write_debug_log(f"Started worker process for zoom level {next_zoom_level}")
                
        except Empty:
            write_log("Timeout waiting for worker process to complete")
            break
    
    # Clean up any remaining processes
    for process in active_processes.keys():
        if process.is_alive():
            process.terminate()
        process.join()
        process.close()
    
    return results, status_queue

def _worker_process(
    zoom_level: int,
    map_style: str,
    map_bounds: Tuple[float, float, float, float],
    track_coords_with_metadata: List[tuple],
    track_lookup: Dict[str, Any],
    map_transparency: int,
    line_width: int,
    resolution_x: int,
    resolution_y: int,
    legend: str,
    statistics: str,
    statistics_data: Optional[Dict[str, str]],
    json_data: Optional[Dict[str, Any]],
    status_queue: mp.Queue,
    result_queue: mp.Queue,
    storage_box_address: str,
    storage_box_user: str,
    storage_box_password: str,
    job_id: str,
    folder: str,
    route_name: str,
    thumbnail: bool = False,
    zoom_levels: Optional[List[int]] = None,
    route_points_data: Optional[Dict[str, Any]] = None
):
    """Worker process function that generates a single image."""
    try:
        result = generate_image_for_zoom_level(
            zoom_level=zoom_level,
            map_style=map_style,
            map_bounds=map_bounds,
            track_coords_with_metadata=track_coords_with_metadata,
            track_lookup=track_lookup,
            map_transparency=map_transparency,
            line_width=line_width,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            legend=legend,
            statistics=statistics,
            statistics_data=statistics_data,
            json_data=json_data,
            status_queue=status_queue,
            storage_box_address=storage_box_address,
            storage_box_user=storage_box_user,
            storage_box_password=storage_box_password,
            job_id=job_id,
            folder=folder,
            route_name=route_name,
            thumbnail=thumbnail,
            zoom_levels=zoom_levels,
            route_points_data=route_points_data
        )
        result_queue.put(result)
    except Exception as e:
        write_log(f"Error in worker process for zoom level {zoom_level}: {str(e)}")
        result_queue.put((zoom_level, None))  # Signal error for this zoom level





def generate_image_for_zoom_level(
    zoom_level: int, 
    map_style: str,
    map_bounds: Tuple[float, float, float, float],
    track_coords_with_metadata: List[tuple],
    track_lookup: Dict[str, Any],
    map_transparency: int,
    line_width: int,
    resolution_x: int,
    resolution_y: int,
    legend: str,
    statistics: str,
    statistics_data: Optional[Dict[str, str]],
    json_data: Optional[Dict[str, Any]],
    status_queue: mp.Queue,
    storage_box_address: str,
    storage_box_user: str,
    storage_box_password: str,
    job_id: str,
    folder: str,
    route_name: str,
    thumbnail: bool = False,
    zoom_levels: Optional[List[int]] = None,
    route_points_data: Optional[Dict[str, Any]] = None
) -> Tuple[int, Optional[bytes]]:
    """
    Generate map image for a specific zoom level.
    
    Args:
        zoom_level: The zoom level to generate
        map_style: Map style to use
        map_bounds: Tuple of (lon_min, lon_max, lat_min, lat_max)
        track_coords_with_metadata: List of (lats, lons, color, name, filename) tuples
        track_lookup: Dictionary mapping filenames to track metadata
        map_transparency: Map transparency value
        line_width: Line width for tracks
        resolution_x: Image width in pixels
        resolution_y: Image height in pixels
        legend: Legend type ('off', 'file_name', 'year', 'month', 'day')
        status_queue: Queue to send status updates to main process
        storage_box_address: FTP server address
        storage_box_user: FTP username
        storage_box_password: FTP password
        job_id: Job ID for the folder structure
        folder: Subfolder name
        route_name: Base name for the output file
        thumbnail: Whether this image should be generated as a thumbnail
        zoom_levels: Optional list of zoom levels for gallery generation
        
    Returns:
        Tuple of (zoom_level, image_bytes)
    """
    try:
        def update_status(status: str, textfield: bool = True):
            """Helper to send status updates."""
            try:
                status_queue.put(StatusUpdate(zoom_level, status, error=None, textfield=textfield))
                # Small sleep to ensure the status is processed
                time.sleep(0.01)
            except Exception as e:
                write_log(f"Failed to send status update: {e}")
        
        # Create an ImageGenerator instance for this process
        generator = ImageGenerator()
        
        # Ensure resolution values are numeric to avoid TypeError
        try:
            resolution_x_value = int(resolution_x)
        except (ValueError, TypeError):
            resolution_x_value = 1920  # Default fallback
        
        try:
            resolution_y_value = int(resolution_y)
        except (ValueError, TypeError):
            resolution_y_value = 1080  # Default fallback
        
        # Calculate bounds with proper aspect ratio
        update_status("calculating bounds", textfield=False)
        lon_min_padded, lon_max_padded, lat_min_padded, lat_max_padded = generator.calculate_aspect_ratio_bounds(
            map_bounds, target_aspect_ratio=resolution_x_value/resolution_y_value
        )
        
        # Ensure cache directory is set for this map style
        try:
            set_cache_directory(map_style)
        except Exception as e:
            write_log(f"Failed to set cache directory for style '{map_style}': {e}")
        
        # Load map tiles (download from server or read from cache)
        update_status("loading tiles", textfield=False)
        map_tiles = create_map_tiles(map_style)
        
        # Now set up the figure with the loaded tiles
        update_status("plotting", textfield=False)
        # Ensure map_transparency is numeric to avoid TypeError
        try:
            map_transparency_value = int(map_transparency)
        except (ValueError, TypeError):
            map_transparency_value = 30  # Default fallback
        alpha = (100 - map_transparency_value) / 100.0
        
        # Calculate figure dimensions
        figsize_width, figsize_height = resolution_x_value/100, resolution_y_value/100
        try:
            update_status(f"fig setup: target_px={resolution_x_value}x{resolution_y_value} figsize_in={figsize_width:.2f}x{figsize_height:.2f} dpi=100", textfield=True)
        except Exception:
            pass
        
        fig, ax = generator.setup_map_figure(
            map_tiles=map_tiles,
            extents=(lon_min_padded, lon_max_padded, lat_min_padded, lat_max_padded),
            zoom_level=zoom_level,
            alpha=alpha,
            figsize=(figsize_width, figsize_height)
        )
        try:
            canvas_w, canvas_h = fig.canvas.get_width_height()
            update_status(f"canvas size px={canvas_w}x{canvas_h}", textfield=True)
        except Exception:
            pass
        
        # Ensure numeric line width
        try:
            line_width_value = float(line_width)
        except Exception:
            line_width_value = 1.0
        
        # Determine image scale category once per process
        image_scale = calculate_resolution_scale(resolution_x_value, resolution_y_value)
        
        # Plot all tracks
        update_status("drawing tracks", textfield=False)
        # Matplotlib linewidth is specified in points (1 pt = 1/72 inch).
        # We want the stroke width in *pixels* to scale with image_scale while
        # remaining visually proportional regardless of figure DPI (100).
        # Convert desired pixels → points: points = pixels * 72 / dpi.
        desired_pixels = line_width_value * image_scale
        effective_line_width = desired_pixels * 72 / 100  # dpi fixed to 100  - earlier halving taken out, seems to conform to preview images now
        
        # Check if speed-based or HR-based coloring is enabled and we have RoutePoint data
        use_speed_based_color = json_data and json_data.get('speed_based_color', False) and route_points_data is not None
        use_hr_based_color = json_data and json_data.get('hr_based_color', False) and route_points_data is not None
        use_data_based_color = use_speed_based_color or use_hr_based_color
        
        # Check if HR-based width is enabled
        use_hr_based_width = json_data and json_data.get('hr_based_width', False) and route_points_data is not None
        hr_width_min = float(json_data.get('hr_based_width_min', 50)) if use_hr_based_width else None
        hr_width_max = float(json_data.get('hr_based_width_max', 180)) if use_hr_based_width else None
        
        def _calculate_hr_based_width_image(hr_value, hr_min, hr_max):
            """Calculate line width based on heart rate value (1-10 range)."""
            if hr_value is None:
                return 5.0  # Default middle width if no HR data
            if hr_value <= hr_min:
                return 1.0
            if hr_value >= hr_max:
                return 10.0
            hr_range = hr_max - hr_min
            if hr_range <= 0:
                return 5.0
            normalized_hr = (hr_value - hr_min) / hr_range
            return 1.0 + normalized_hr * 9.0
        
        # Import RoutePoint if needed for data-based coloring or HR-based width
        if use_data_based_color or use_hr_based_width:
                from video_generator_create_combined_route import RoutePoint
                
        if use_data_based_color:
            # DATA-BASED COLORING MODE: Draw segments individually with speed-based or HR-based colors
            if use_speed_based_color:
                value_min = float(json_data.get('speed_based_color_min', 5))
                value_max = float(json_data.get('speed_based_color_max', 35))
                color_type = "speed"
            else:  # use_hr_based_color
                value_min = float(json_data.get('hr_based_color_min', 50))
                value_max = float(json_data.get('hr_based_color_max', 180))
                color_type = "HR"
            
            value_range = value_max - value_min
            
            # Validate range to prevent division by zero
            if value_range <= 0:
                write_log(f"WARNING: Invalid {color_type} range ({value_min} to {value_max}). Falling back to standard mode.")
                use_data_based_color = False
                use_speed_based_color = False
                use_hr_based_color = False
            
            if use_data_based_color:
                # Get all routes from route_points_data
                all_routes = route_points_data.get('all_routes', [])
                if not all_routes:
                    all_routes = [route_points_data]  # Fallback to single route
                
                write_debug_log(f"{color_type.capitalize()}-based coloring: Found {len(all_routes)} route(s) in route_points_data")
                
                track_count = 0
                total_segments = 0
                skipped_segments = 0
                
                for route_idx, route in enumerate(all_routes):
                    combined_route = route.get('combined_route', [])
                    write_debug_log(f"Route {route_idx}: {len(combined_route)} points in combined_route")
                    
                    if not combined_route or len(combined_route) < 2:
                        write_debug_log(f"Route {route_idx}: Skipping (too few points)")
                        continue
                    
                    track_count += 1
                    
                    # Draw each segment between consecutive points with speed-based color
                    for j in range(len(combined_route) - 1):
                        current_point = combined_route[j]
                        next_point = combined_route[j + 1]
                        
                        # Skip segments that cross track boundaries
                        if current_point.new_route_flag or next_point.new_route_flag:
                            skipped_segments += 1
                            continue
                        
                        # Get value from current point (speed or HR)
                        if use_speed_based_color:
                            value = current_point.current_speed_smoothed
                        else:  # use_hr_based_color
                            value = current_point.heart_rate_smoothed
                        
                        # Debug: log first few values to diagnose issues
                        if total_segments < 5:
                            write_debug_log(f"Segment {total_segments}: {color_type}_value={value}, {color_type}_min={value_min}, {color_type}_max={value_max}")
                        
                        if value is None:
                            # If no data, use default color (red)
                            segment_color = (1.0, 0.0, 0.0)
                        else:
                            # Normalize value to 0-1 range
                            normalized_value = (value - value_min) / value_range
                            normalized_value = max(0.0, min(1.0, normalized_value))  # Clamp to 0-1
                            
                            # Get RGB color from speed_based_color function (returns 0-1 range, matplotlib format)
                            rgb = speed_based_color(normalized_value)
                            
                            # Use directly as matplotlib color
                            segment_color = rgb
                        
                        # Calculate line width (HR-based or fixed)
                        if use_hr_based_width:
                            hr_width_value = _calculate_hr_based_width_image(current_point.heart_rate_smoothed, hr_width_min, hr_width_max)
                            # Apply resolution scale multiplier (same as effective_line_width)
                            desired_pixels = hr_width_value * image_scale
                            segment_width = desired_pixels * 72 / 100  # dpi fixed to 100
                        else:
                            segment_width = effective_line_width
                        
                        # Draw this segment
                        ax.plot(
                            [current_point.lon, next_point.lon],
                            [current_point.lat, next_point.lat],
                            transform=ccrs.PlateCarree(),
                            color=segment_color,
                            linewidth=segment_width
                        )
                        total_segments += 1
                
                write_debug_log(f"Plotted {track_count} routes with {color_type}-based coloring: {total_segments} segments drawn, {skipped_segments} segments skipped (zoom level {zoom_level})")
        
        if not use_data_based_color:
            # STANDARD MODE: Use existing coordinate-based plotting
            # If HR-based width is enabled and we have route points data, draw segments individually
            if use_hr_based_width:
                # HR-BASED WIDTH MODE: Draw segments individually with varying widths
                all_routes = route_points_data.get('all_routes', []) if route_points_data else []
                if not all_routes and route_points_data:
                    all_routes = [route_points_data]  # Fallback to single route
                
                track_count = 0
                total_segments = 0
                
                for route_idx, route in enumerate(all_routes):
                    combined_route = route.get('combined_route', [])
                    
                    if not combined_route or len(combined_route) < 2:
                        continue
                    
                    track_count += 1
                    
                    # Get filename from first point and find the color
                    point_filename = combined_route[0].filename
                    
                    # Find the color from track_coords_with_metadata
                    route_color = (1.0, 0.0, 0.0)  # Default red
                    if track_coords_with_metadata:
                        for lats, lons, color, name, filename in track_coords_with_metadata:
                            if filename and point_filename and filename.lower() == point_filename.lower():
                                route_color = color
                                break
                    
                    # Draw each segment with HR-based width
                    for j in range(len(combined_route) - 1):
                        current_point = combined_route[j]
                        next_point = combined_route[j + 1]
                        
                        # Skip segments that cross track boundaries
                        if current_point.new_route_flag or next_point.new_route_flag:
                            continue
                        
                        # Calculate HR-based width
                        hr_width_value = _calculate_hr_based_width_image(current_point.heart_rate_smoothed, hr_width_min, hr_width_max)
                        # Apply resolution scale multiplier (same as effective_line_width)
                        desired_pixels = hr_width_value * image_scale
                        segment_width = desired_pixels * 72 / 100  # dpi fixed to 100
                        
                        ax.plot(
                            [current_point.lon, next_point.lon],
                            [current_point.lat, next_point.lat],
                            transform=ccrs.PlateCarree(),
                            color=route_color,
                            linewidth=segment_width
                        )
                        total_segments += 1
                
                write_debug_log(f"Plotted {track_count} routes with HR-based width: {total_segments} segments (zoom level {zoom_level})")
            else:
                # STANDARD MODE WITHOUT HR-BASED WIDTH
                track_count = 0
                total_segments = 0
                for lats, lons, color, name, filename in track_coords_with_metadata:
                    track_count += 1
                    # Split track at longitude wrap if necessary
                    segments = split_track_at_longitude_wrap(lats, lons)
                    total_segments += len(segments)
                    for segment_lats, segment_lons in segments:
                        ax.plot(segment_lons, segment_lats, transform=ccrs.PlateCarree(), color=color, linewidth=effective_line_width)
                
                write_debug_log(f"Plotted {track_count} tracks as {total_segments} segments (zoom level {zoom_level})")
        
        # Add statistics if requested
        if statistics != "off":
            update_status("adding statistics", textfield=False)
            add_statistics_to_plot(ax, statistics_data, json_data or {}, resolution_x_value, resolution_y_value, image_scale, statistics)
        
        # Add legend if requested
        if legend != "off":
            update_status("adding legend", textfield=False)
            # Determine legend theme from job JSON (default to light)
            legend_theme = (json_data or {}).get('legend_theme', 'light')
            add_legend_to_plot(
                ax=ax,
                track_coords_with_metadata=track_coords_with_metadata,
                track_lookup=track_lookup,
                legend_type=legend,
                image_width=resolution_x_value,
                image_height=resolution_y_value,
                image_scale=image_scale,
                legend_theme=legend_theme
            )
        
        # Add title text if boolean flag is true (text is route_name)
        try:
            if json_data is not None and isinstance(json_data.get('title_text'), bool) and json_data.get('title_text') is True:
                title_text_value = (route_name or json_data.get('route_name') or '').strip()
                if title_text_value:
                    update_status("adding title", textfield=False)
                    add_title_text_to_plot(
                        ax=ax,
                        title_text=title_text_value,
                        image_width=resolution_x_value,
                        image_height=resolution_y_value,
                        image_scale=image_scale,
                    )
        except Exception as e:
            write_log(f"Failed to add title text: {e}")

        # Add speed-based color, HR-based color, or HR-based width label if enabled (before stamp so stamp is on top)
        if json_data and (json_data.get('speed_based_color_label', False) or json_data.get('hr_based_color_label', False) or json_data.get('hr_based_width_label', False)):
            if json_data.get('hr_based_width_label', False):
                label_type = "HR width"
            elif json_data.get('hr_based_color_label', False):
                label_type = "HR color"
            else:
                label_type = "speed"
            update_status(f"adding {label_type} label", textfield=False)
            from image_generator_postprocess import add_speed_based_color_label_to_plot
            add_speed_based_color_label_to_plot(
                ax=ax,
                json_data=json_data,
                image_width=resolution_x_value,
                image_height=resolution_y_value,
                image_scale=image_scale
            )

        # Add stamp
        update_status("adding stamp", textfield=False)
        add_stamp_to_plot(
            ax=ax,
            image_width=resolution_x_value,
            image_height=resolution_y_value,
            image_scale=image_scale
        )
        
        # Save to bytes
        update_status("saving image", textfield=False)
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches=None, pad_inches=0)
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        try:
            with Image.open(BytesIO(image_bytes)) as im_probe:
                update_status(f"saved png size px={im_probe.width}x{im_probe.height}", textfield=True)
        except Exception:
            pass
        
        # Close figure to free memory
        fig.clear()
        
        # Optimize PNG
        update_status("compressing", textfield=False)
        optimized_bytes = optimize_png_bytes(image_bytes)
        
        # Create thumbnail if requested
        thumbnail_bytes = None
        if thumbnail:
            # Open the optimized image
            thumb_img = Image.open(BytesIO(optimized_bytes))
            
            # Calculate thumbnail dimensions
            target_width = 568
            target_height = 320
            aspect_ratio = thumb_img.width / thumb_img.height
            
            if aspect_ratio > (target_width / target_height):
                # Image is wider than target ratio, fit to width
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Image is taller than target ratio, fit to height
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # Resize the image
            thumb_img = thumb_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save thumbnail
            thumb_buffer = BytesIO()
            thumb_img.save(thumb_buffer, format='PNG', optimize=True)
            thumbnail_bytes = thumb_buffer.getvalue()

        # Upload the files
        filename = f"{route_name}_zoom{zoom_level}.png"
        success = upload_to_storage_box(
            image_bytes=optimized_bytes,
            storage_box_address=storage_box_address,
            storage_box_user=storage_box_user,
            storage_box_password=storage_box_password,
            job_id=job_id,
            folder=folder,
            filename=filename,
            thumbnail_bytes=thumbnail_bytes if thumbnail else None,
            update_status=update_status,
            route_name=route_name,
            zoom_levels=zoom_levels if thumbnail else None  # Only pass all zoom levels when uploading thumbnail
        )

        if not success:
            update_status("error")
            return zoom_level, None

        # Add final status update after successful upload and verification
        update_status("complete", textfield=False)
        return zoom_level, optimized_bytes
        
    except Exception as e:
        error_msg = f"Error generating image for zoom level {zoom_level}: {str(e)}\n{traceback.format_exc()}"
        write_log(error_msg)
        try:
            status_queue.put(StatusUpdate(zoom_level, "error", error_msg))
            time.sleep(0.01)  # Small sleep to ensure the error status is processed
        except Exception as send_error:
            write_log(f"Failed to send error status: {send_error}")
        return zoom_level, None


def add_statistics_to_plot(ax, statistics_data: Dict[str, str], json_data: Dict, image_width: int, image_height: int, image_scale: int | None = None, statistics_theme: str = "light"):
    """
    Add statistics text to the plot in the top-right corner.
    
    Args:
        ax: Matplotlib axes
        statistics_data: Dictionary containing calculated statistics
        json_data: JSON data containing statistics configuration
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels  
        image_scale: Optional image scale factor for text sizing; if None, calculated from resolution
        statistics_theme: Theme for statistics display ('light' or 'dark')
    """
    if not statistics_data:
        return
        
    # Build statistics lines to display based on configuration
    stats_lines = []
    
    if json_data.get('statistics_starting_time', False):
        stats_lines.append(f"{statistics_data.get('starting_time', 'N/A')}")
    
    if json_data.get('statistics_ending_time', False):
        stats_lines.append(f"{statistics_data.get('ending_time', 'N/A')}")
    
    if json_data.get('statistics_elapsed_time', False):
        stats_lines.append(f"{statistics_data.get('elapsed_time', 'N/A')}")
    
    # Determine units based on imperial_units setting
    imperial_units = json_data and json_data.get('imperial_units', False) is True
    distance_unit = "miles" if imperial_units else "km"
    speed_unit = "mph" if imperial_units else "km/h"
    
    if json_data.get('statistics_distance', False):
        stats_lines.append(f"{statistics_data.get('distance', 'N/A')} {distance_unit}")
    
    if json_data.get('statistics_average_speed', False):
        stats_lines.append(f"{statistics_data.get('average_speed', 'N/A')} {speed_unit}")
    
    if json_data.get('statistics_average_hr', False):
        avg_hr = statistics_data.get('average_hr', '0')
        if avg_hr and avg_hr != '0':
            stats_lines.append(f"{avg_hr} ❤")
    
    # If no statistics are configured to be shown, return
    if not stats_lines:
        return
    
    # Calculate image scale if not provided
    if image_scale is None:
        image_scale = calculate_resolution_scale(image_width, image_height)
    
    # Set theme colors
    if statistics_theme == 'dark':
        bg_color = '#2d2d2d'      # Dark gray background
        border_color = '#cccccc'  # Light gray border
        text_color = '#ffffff'    # White text
    else:  # light theme (default)
        bg_color = 'white'        # White background
        border_color = '#333333'  # Dark gray border
        text_color = '#333333'    # Dark gray text
    
    # Calculate font size based on image scale
    base_font_size = 10
    font_size = base_font_size * image_scale
    
    # Combine all statistics lines
    stats_text = '\n'.join(stats_lines)
    
    # Position in top right with scaled padding (converted to axes coordinates)
    # Base padding is 10px, scaled by image_scale
    # Use axes coordinates (0-1 range) for consistent positioning
    base_padding_pixels = 10
    padding_pixels = base_padding_pixels * image_scale
    padding_x = padding_pixels / image_width  # Convert to axes coordinates
    padding_y = padding_pixels / image_height # Convert to axes coordinates
    
    text_x = 1.0 - padding_x  # Right edge minus padding
    text_y = 1.0 - padding_y  # Top edge minus padding
    
    # Add the statistics text
    ax.text(
        text_x, text_y, stats_text,
        transform=ax.transAxes,  # Use axes coordinates
        color=text_color,
        fontsize=font_size,
        fontweight='bold',
        ha='right',     # Right align
        va='top',       # Top align
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor=bg_color,
            edgecolor=border_color,
            alpha=0.9,
            linewidth=1
        )
    )


def upload_to_storage_box(
    image_bytes: bytes,
    storage_box_address: str,
    storage_box_user: str,
    storage_box_password: str,
    job_id: str,
    folder: str,
    filename: str,
    thumbnail_bytes: Optional[bytes] = None,
    update_status=None,
    route_name: Optional[str] = None,
    zoom_levels: Optional[List[int]] = None
) -> bool:
    """
    Upload an image and optionally its thumbnail to the storage box and verify they exist.
    
    Args:
        image_bytes: The image data to upload
        storage_box_address: FTP server address
        storage_box_user: FTP username
        storage_box_password: FTP password
        job_id: Job ID for the folder structure
        folder: Subfolder name
        filename: Name of the file to save
        thumbnail_bytes: Optional thumbnail image data to upload
        update_status: Optional callback for status updates
        route_name: Optional route name for gallery generation
        zoom_levels: Optional list of zoom levels for gallery generation
        
    Returns:
        bool: True if upload successful and verified, False otherwise
    """
    ftp = None
    try:
        if update_status:
            update_status("uploading", textfield=False)
            
        # Connect to FTP
        ftp = ftplib.FTP(storage_box_address)
        ftp.login(storage_box_user, storage_box_password)
        
        # Create or enter media directory
        try:
            ftp.mkd('media')
        except ftplib.error_perm:
            pass  # Directory might already exist
        ftp.cwd('media')
        
        # Create job directory if it doesn't exist
        try:
            ftp.mkd(job_id)
        except ftplib.error_perm:
            pass  # Directory might already exist
            
        # Change to job directory
        ftp.cwd(job_id)
        
        # Create folder if it doesn't exist
        try:
            ftp.mkd(folder)
        except ftplib.error_perm:
            pass  # Directory might already exist
            
        # Change to folder
        ftp.cwd(folder)
        
        # Upload the main file
        ftp.storbinary(f'STOR {filename}', BytesIO(image_bytes))
        
        # Verify main file exists and has size
        file_size = 0
        try:
            file_size = ftp.size(filename)
        except:
            write_log(f"Failed to get size for {filename}")
            return False
            
        if file_size <= 0:
            write_log(f"File {filename} has zero or negative size")
            return False

        # If we have a thumbnail, upload it too
        if thumbnail_bytes:
            thumb_filename = "thumbnail.png"
            ftp.storbinary(f'STOR {thumb_filename}', BytesIO(thumbnail_bytes))
            
            # Verify thumbnail exists and has size
            thumb_size = 0
            try:
                thumb_size = ftp.size(thumb_filename)
            except:
                write_log(f"Failed to get size for {thumb_filename}")
                return False
                
            if thumb_size <= 0:
                write_log(f"File {thumb_filename} has zero or negative size")
                return False
            
            # After successful thumbnail upload, generate and upload gallery
            if route_name and zoom_levels:
                try:
                    # Generate HTML content
                    html_content = generate_image_gallery_html(route_name, zoom_levels)
                    
                    # Upload HTML file
                    ftp.storbinary('STOR images.html', BytesIO(html_content.encode('utf-8')))
                    
                    # Verify HTML file exists and has size
                    try:
                        html_size = ftp.size('images.html')
                        if html_size <= 0:
                            write_log("Gallery HTML file has zero or negative size")
                        else:
                            write_debug_log(f"Successfully uploaded gallery HTML ({html_size} bytes)")
                    except:
                        write_log("Failed to verify gallery HTML file size")
                        
                except Exception as e:
                    write_log(f"Error generating/uploading gallery HTML: {str(e)}")
        
        return True
        
    except Exception as e:
        write_log(f"Upload failed: {str(e)}")
        return False
    finally:
        if ftp:
            try:
                ftp.quit()
            except:
                pass