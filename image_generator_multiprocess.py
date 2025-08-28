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

from image_generator_maptileutils import debug_log, create_map_tiles, set_cache_directory
from image_generator_postprocess import add_stamp_to_plot, add_legend_to_plot, optimize_png_bytes, add_title_text_to_plot
from image_generator_utils import ImageGenerator
from html_file_generator import generate_image_gallery_html

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
        debug_log(f"Longitude wrapping detected: track split into {len(segments)} segments after {wrap_count} wrap(s)")
    
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
    max_workers: Optional[int] = None
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
    
    debug_log(f"Using {max_workers} worker processes for {len(zoom_levels)} zoom levels")
    
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
                sorted_zoom_levels  # Pass all zoom levels
            )
        )
        process.start()
        active_processes[process] = zoom_level
        debug_log(f"Started worker process for zoom level {zoom_level}")
    
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
            debug_log(f"Completed zoom level {result[0]} ({completed_processes}/{total_processes})")
            
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
                        sorted_zoom_levels
                    )
                )
                process.start()
                active_processes[process] = next_zoom_level
                debug_log(f"Started worker process for zoom level {next_zoom_level}")
                
        except Empty:
            debug_log("Timeout waiting for worker process to complete")
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
    zoom_levels: Optional[List[int]] = None
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
            zoom_levels=zoom_levels
        )
        result_queue.put(result)
    except Exception as e:
        debug_log(f"Error in worker process for zoom level {zoom_level}: {str(e)}")
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
    zoom_levels: Optional[List[int]] = None
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
                debug_log(f"Failed to send status update: {e}")
        
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
            debug_log(f"Failed to set cache directory for style '{map_style}': {e}")
        
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
        total_pixels = resolution_x_value * resolution_y_value
        if total_pixels < 8_000_000:
            image_scale = 1
        elif total_pixels < 18_000_000:
            image_scale = 2
        elif total_pixels < 33_000_000:
            image_scale = 3
        else:
            image_scale = 4
        
        # Plot all tracks
        update_status("drawing tracks", textfield=False)
        # Matplotlib linewidth is specified in points (1 pt = 1/72 inch).
        # We want the stroke width in *pixels* to scale with image_scale while
        # remaining visually proportional regardless of figure DPI (100).
        # Convert desired pixels → points: points = pixels * 72 / dpi.
        desired_pixels = line_width_value * image_scale
        effective_line_width = desired_pixels * 72 / 100  # dpi fixed to 100  - earlier halving taken out, seems to conform to preview images now
        
        track_count = 0
        total_segments = 0
        for lats, lons, color, name, filename in track_coords_with_metadata:
            track_count += 1
            # Split track at longitude wrap if necessary
            segments = split_track_at_longitude_wrap(lats, lons)
            total_segments += len(segments)
            for segment_lats, segment_lons in segments:
                ax.plot(segment_lons, segment_lats, transform=ccrs.PlateCarree(), color=color, linewidth=effective_line_width)
        
        debug_log(f"Plotted {track_count} tracks as {total_segments} segments (zoom level {zoom_level})")
        
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
            debug_log(f"Failed to add title text: {e}")

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
        debug_log(error_msg)
        try:
            status_queue.put(StatusUpdate(zoom_level, "error", error_msg))
            time.sleep(0.01)  # Small sleep to ensure the error status is processed
        except Exception as send_error:
            debug_log(f"Failed to send error status: {send_error}")
        return zoom_level, None


def add_statistics_to_plot(ax, statistics_data: Dict[str, str], json_data: Dict, image_width: int, image_height: int, image_scale: int, statistics_theme: str = "light"):
    """
    Add statistics text to the plot in the top-right corner.
    
    Args:
        ax: Matplotlib axes
        statistics_data: Dictionary containing calculated statistics
        json_data: JSON data containing statistics configuration
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels  
        image_scale: Image scale factor for text sizing
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
    
    if json_data.get('statistics_distance', False):
        stats_lines.append(f"{statistics_data.get('distance', 'N/A')} km")
    
    if json_data.get('statistics_average_speed', False):
        stats_lines.append(f"{statistics_data.get('average_speed', 'N/A')} km/h")
    
    # If no statistics are configured to be shown, return
    if not stats_lines:
        return
    
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
    
    # Position in top right with 10px padding (converted to axes coordinates)
    # Use axes coordinates (0-1 range) for consistent positioning
    padding_x = 10 / image_width  # Convert 10px to axes coordinates
    padding_y = 10 / image_height # Convert 10px to axes coordinates
    
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
        
        # Create or enter jobs directory
        try:
            ftp.mkd('jobs')
        except ftplib.error_perm:
            pass  # Directory might already exist
        ftp.cwd('jobs')
        
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
            debug_log(f"Failed to get size for {filename}")
            return False
            
        if file_size <= 0:
            debug_log(f"File {filename} has zero or negative size")
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
                debug_log(f"Failed to get size for {thumb_filename}")
                return False
                
            if thumb_size <= 0:
                debug_log(f"File {thumb_filename} has zero or negative size")
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
                            debug_log("Gallery HTML file has zero or negative size")
                        else:
                            debug_log(f"Successfully uploaded gallery HTML ({html_size} bytes)")
                    except:
                        debug_log("Failed to verify gallery HTML file size")
                        
                except Exception as e:
                    debug_log(f"Error generating/uploading gallery HTML: {str(e)}")
        
        return True
        
    except Exception as e:
        debug_log(f"Upload failed: {str(e)}")
        return False
    finally:
        if ftp:
            try:
                ftp.quit()
            except:
                pass