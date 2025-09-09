"""
Video generation single frame creation for the Route Squiggler render client.
This module handles generating individual video frames with route visualization and tail effects.
"""

import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for video frame generation

import matplotlib.pyplot as plt
import matplotlib.patheffects
import math
from video_generator_route_statistics import _calculate_video_statistics, _draw_current_speed_at_point, _draw_current_elevation_at_point, _draw_video_statistics
from video_generator_coordinate_encoder import encode_coords
from video_generator_calculate_bounding_boxes import calculate_bounding_box_for_points, load_final_bounding_box
from video_generator_create_single_frame_legend import get_filename_legend_data, get_year_legend_data, get_month_legend_data, get_day_legend_data, get_people_legend_data, create_legend
from video_generator_create_single_frame_utils import (get_tail_color_for_route, _draw_name_tags_for_routes, _get_resolution_scale_factor)



def _gps_to_web_mercator(lon, lat):
    """
    Convert GPS coordinates (WGS84) to Web Mercator projection (EPSG:3857).
    
    Args:
        lon (float): Longitude in degrees
        lat (float): Latitude in degrees
    
    Returns:
        tuple: (x, y) in Web Mercator meters
    """
    # Web Mercator transformation
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y


def _convert_bbox_to_web_mercator(bbox):
    """
    Convert GPS bounding box to Web Mercator bounding box.
    
    Args:
        bbox (tuple): (lon_min, lon_max, lat_min, lat_max) in GPS coordinates
    
    Returns:
        tuple: (x_min, x_max, y_min, y_max) in Web Mercator coordinates
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    
    x_min, y_min = _gps_to_web_mercator(lon_min, lat_min)
    x_max, y_max = _gps_to_web_mercator(lon_max, lat_max)
    
    return (x_min, x_max, y_min, y_max)


def _find_track_boundaries(route_points):
    """
    Pre-compute track boundary indices using binary search on filename changes.
    
    Since filenames change exactly when new_route_flag is True, and same filenames
    persist in uninterrupted chunks, we can use binary search to find filename
    transitions much faster than linear scanning.
    
    For sparse flags (e.g., 5 in 5000 points), this is ~O(k * log(n/k)) instead of O(n),
    where k is the number of tracks and n is the number of points.
    
    Args:
        route_points (list): List of route points with filenames at index 7
    
    Returns:
        list: List of (start_index, end_index) tuples for each track segment
    """
    if not route_points:
        return []
    
    def _binary_search_filename_end(start_idx, end_idx, target_filename):
        """
        Binary search to find where target_filename ends (changes to different filename).
        Returns the index of the first point with a different filename.
        """
        left, right = start_idx, end_idx - 1
        result = end_idx  # Default: filename persists to the end
        
        while left <= right:
            mid = (left + right) // 2
            
            # Get filename at mid position
            if len(route_points[mid]) >= 8:
                mid_filename = route_points[mid][7]  # filename at index 7
                
                if mid_filename == target_filename:
                    # Same filename, look for transition further right
                    left = mid + 1
                else:
                    # Different filename found, could be our transition point
                    result = mid
                    right = mid - 1
            else:
                # Point doesn't have filename, skip it
                right = mid - 1
        
        return result
    
    boundaries = []
    current_start = 0
    
    while current_start < len(route_points):
        # Get the filename for the current segment
        if len(route_points[current_start]) >= 8:
            current_filename = route_points[current_start][7]
        else:
            # Skip points without proper structure
            current_start += 1
            continue
        
        # Binary search to find where this filename ends
        filename_end = _binary_search_filename_end(current_start, len(route_points), current_filename)
        
        # Add this track boundary
        if filename_end > current_start:
            boundaries.append((current_start, filename_end))
        
        # Move to the next segment
        current_start = filename_end
        
        # Safety check to prevent infinite loops
        if current_start == filename_end and current_start < len(route_points):
            current_start += 1
    
    return boundaries


def _binary_search_tail_start_index(route_points, tail_start_time):
    """
    Binary search to find the first index where accumulated_time >= tail_start_time.
    
    This allows for efficient tail point collection by finding the exact start of the tail window.
    
    Args:
        route_points (list): List of route points, chronologically ordered by accumulated_time
        tail_start_time (float): Start time of the tail window
    
    Returns:
        int: Index of first point that should be included in tail (inclusive), suitable for slicing
    """
    if not route_points:
        return 0
    
    left, right = 0, len(route_points) - 1
    result = len(route_points)  # Default: no points in tail if none meet criteria
    
    while left <= right:
        mid = (left + right) // 2
        
        # Check if this point has the minimum required length and extract accumulated_time
        if len(route_points[mid]) >= 5:
            accumulated_time = route_points[mid][4]  # accumulated_time at index 4
            
            if accumulated_time >= tail_start_time:
                # This point should be included in tail, look for earlier start point
                result = mid
                right = mid - 1
            else:
                # This point is too early for tail, look for later points
                left = mid + 1
        else:
            # Point doesn't have enough elements, skip it
            left = mid + 1
    
    return result


def get_legend_theme_colors(legend_theme):
    """Get legend colors based on theme"""
    if legend_theme == "dark":
        return {
            'facecolor': '#2d2d2d',  # Dark gray background
            'edgecolor': '#cccccc',  # Light gray border
            'textcolor': '#ffffff',  # White text
            'framealpha': 0.9        # Slightly more opaque for dark theme
        }
    elif legend_theme == "light":
        return {
            'facecolor': 'white',    # White background
            'edgecolor': 'black',    # Black border
            'textcolor': 'black',    # Black text
            'framealpha': 0.8        # Standard transparency
        }
    else:
        # Default to light theme
        return {
            'facecolor': 'white',
            'edgecolor': 'black',
            'textcolor': 'black',
            'framealpha': 0.8
        }


def _draw_multi_route_tail(tail_points, tail_color_setting, tail_width, effective_line_width, filename_to_rgba, ax, fade_out_progress=None, fluffy_tail=False):
    """
    Draw tail segments with proper color handling for route transitions.
    Splits the tail into route-specific segments when routes change within the tail window.
    
    Args:
        tail_points (list): List of all points in the tail window (chronologically sorted)
        tail_color_setting (str): Tail color setting ('light', 'dark', or hex color)
        tail_width (float): Tail width multiplier
        effective_line_width (float): Base line width
        filename_to_rgba (dict): Filename to RGBA color mapping
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        fade_out_progress (float, optional): Fade-out progress (0.0 = just ended, 1.0 = fully faded)
        fluffy_tail (bool): If True, use LineCollection for "fluffy" tail effect; if False, use individual plots
    """
    if len(tail_points) <= 1:
        return
    
    # OPTIMIZED: Since tail_points are chronologically ordered, directly access first and last points
    # instead of iterating through the entire list to find min/max
    time_min = tail_points[0][4]   # First point has earliest time
    time_max = tail_points[-1][4]  # Last point has latest time
    time_range = time_max - time_min
    
    if time_range <= 0:
        return
    
    # SIMULTANEOUS MODE TAIL FIX: Apply fade-out effect if fade_out_progress is provided
    if fade_out_progress is not None:
        # Clamp fade_out_progress to valid range
        fade_out_progress = max(0.0, min(1.0, fade_out_progress))
        
        # Calculate fade-out alpha (1.0 = fully visible, 0.0 = fully transparent)
        fade_out_alpha = 1.0 - fade_out_progress
        
        # If fully faded out, don't draw anything
        if fade_out_alpha <= 0.0:
            return
    else:
        fade_out_alpha = 1.0  # No fade-out effect
    
    # Group consecutive points by route (filename) to handle route transitions
    route_segments = []
    current_segment = []
    current_filename = None
    
    # Debug: Track unique filenames in this tail
    unique_filenames = set()
    
    for point in tail_points:
        point_filename = point[7] if len(point) > 7 else None
        if point_filename:
            unique_filenames.add(point_filename)
    
    # Debug: Log all unique filenames found in tail points
    if unique_filenames:
        print("\nDebug - Filenames in tail points:")
        for fn in sorted(unique_filenames):
            color = filename_to_rgba.get(fn, "NOT FOUND")
            print(f"  {fn}: {color}")
    else:
        print("\nDebug: No filenames found in tail points")
    
    # Reset for processing
    current_segment = []
    current_filename = None
    
    for point in tail_points:
        point_filename = point[7] if len(point) > 7 else None
        
        # Start new segment when route changes or on first point
        if point_filename != current_filename:
            if current_segment:
                route_segments.append((current_filename, current_segment))
            current_segment = [point]
            current_filename = point_filename
        else:
            current_segment.append(point)
    
    # Add the last segment
    if current_segment:
        route_segments.append((current_filename, current_segment))
    
    if fluffy_tail:
        # FLUFFY TAIL EFFECT: Use LineCollection for the "fluffy" appearance
        from matplotlib.collections import LineCollection
        
        # Collect all line segments and their properties for LineCollection
        line_segments = []
        line_colors = []
        line_widths = []
        
        # Draw each route segment with its appropriate color
        for segment_filename, segment_points in route_segments:
            if len(segment_points) <= 1:
                continue
            
            # Get route color for this segment
            if segment_filename and filename_to_rgba and segment_filename in filename_to_rgba:
                route_color_rgba = filename_to_rgba[segment_filename]
            else:
                route_color_rgba = (1.0, 0.0, 0.0, 1.0)
            
            # Get tail color for this route
            tail_rgba_color = get_tail_color_for_route(tail_color_setting, route_color_rgba)
            
            # SIMULTANEOUS MODE TAIL FIX: Apply fade-out alpha to both route and tail colors
            route_color_rgba = (route_color_rgba[0], route_color_rgba[1], route_color_rgba[2], route_color_rgba[3] * fade_out_alpha)
            tail_rgba_color = (tail_rgba_color[0], tail_rgba_color[1], tail_rgba_color[2], tail_rgba_color[3] * fade_out_alpha)
            
            # Extract coordinates and times for this segment
            segment_lats = [point[1] for point in segment_points]
            segment_lons = [point[2] for point in segment_points]
            segment_times = [point[4] for point in segment_points]
            
            # Draw tail segments for this route portion
            for j in range(len(segment_points) - 1):
                current_point = segment_points[j]
                next_point = segment_points[j + 1]
                
                # Skip segments that cross track boundaries (new_route_flag)
                if (len(current_point) >= 7 and current_point[6]) or (len(next_point) >= 7 and next_point[6]):
                    continue
                
                # Calculate interpolation factor based on overall tail time range
                segment_time = (segment_times[j] + segment_times[j + 1]) / 2
                interp_factor = (segment_time - time_min) / time_range
                
                # Pre-calculate interpolated values for this segment
                segment_width = effective_line_width + (effective_line_width * tail_width - effective_line_width) * interp_factor
                segment_color = (
                    route_color_rgba[0] + (tail_rgba_color[0] - route_color_rgba[0]) * interp_factor,
                    route_color_rgba[1] + (tail_rgba_color[1] - route_color_rgba[1]) * interp_factor,
                    route_color_rgba[2] + (tail_rgba_color[2] - route_color_rgba[2]) * interp_factor,
                    route_color_rgba[3] + (tail_rgba_color[3] - route_color_rgba[3]) * interp_factor
                )
                
                # Create line segment for this tail portion (convert GPS to Web Mercator coordinates)
                lon1, lat1 = segment_lons[j], segment_lats[j]
                lon2, lat2 = segment_lons[j + 1], segment_lats[j + 1]
                x1, y1 = _gps_to_web_mercator(lon1, lat1)
                x2, y2 = _gps_to_web_mercator(lon2, lat2)
                
                # Add to LineCollection data
                line_segments.append([(x1, y1), (x2, y2)])
                line_colors.append(segment_color)
                line_widths.append(segment_width)
        
        # Create and add LineCollection if we have any tail segments
        if line_segments:
            line_collection = LineCollection(
                line_segments,
                colors=line_colors,
                linewidths=line_widths,
                zorder=30  # Top layer for tails (above active routes)
            )
            ax.add_collection(line_collection)
    
    else:
        # STANDARD TAILS: Use individual ax.plot() calls for crisp, individual segments
        # Draw each route segment with its appropriate color
        for segment_filename, segment_points in route_segments:
            if len(segment_points) <= 1:
                continue
            
            # Get route color for this segment
            if segment_filename and filename_to_rgba and segment_filename in filename_to_rgba:
                route_color_rgba = filename_to_rgba[segment_filename]
            else:
                route_color_rgba = (1.0, 0.0, 0.0, 1.0)
            
            # Get tail color for this route
            tail_rgba_color = get_tail_color_for_route(tail_color_setting, route_color_rgba)
            
            # SIMULTANEOUS MODE TAIL FIX: Apply fade-out alpha to both route and tail colors
            route_color_rgba = (route_color_rgba[0], route_color_rgba[1], route_color_rgba[2], route_color_rgba[3] * fade_out_alpha)
            tail_rgba_color = (tail_rgba_color[0], tail_rgba_color[1], tail_rgba_color[2], tail_rgba_color[3] * fade_out_alpha)
            
            # Extract coordinates and times for this segment
            segment_lats = [point[1] for point in segment_points]
            segment_lons = [point[2] for point in segment_points]
            segment_times = [point[4] for point in segment_points]
            
            # Draw tail segments for this route portion
            for j in range(len(segment_points) - 1):
                current_point = segment_points[j]
                next_point = segment_points[j + 1]
                
                # Skip segments that cross track boundaries (new_route_flag)
                if (len(current_point) >= 7 and current_point[6]) or (len(next_point) >= 7 and next_point[6]):
                    continue
                
                # Calculate interpolation factor based on overall tail time range
                segment_time = (segment_times[j] + segment_times[j + 1]) / 2
                interp_factor = (segment_time - time_min) / time_range
                
                # Pre-calculate interpolated values for this segment
                segment_width = effective_line_width + (effective_line_width * tail_width - effective_line_width) * interp_factor
                segment_color = (
                    route_color_rgba[0] + (tail_rgba_color[0] - route_color_rgba[0]) * interp_factor,
                    route_color_rgba[1] + (tail_rgba_color[1] - route_color_rgba[1]) * interp_factor,
                    route_color_rgba[2] + (tail_rgba_color[2] - route_color_rgba[2]) * interp_factor,
                    route_color_rgba[3] + (tail_rgba_color[3] - route_color_rgba[3]) * interp_factor
                )
                
                # Draw this segment (convert GPS to Web Mercator coordinates)
                lon1, lat1 = segment_lons[j], segment_lats[j]
                lon2, lat2 = segment_lons[j + 1], segment_lats[j + 1]
                x1, y1 = _gps_to_web_mercator(lon1, lat1)
                x2, y2 = _gps_to_web_mercator(lon2, lat2)
                
                ax.plot(
                    [x1, x2], 
                    [y1, y2], 
                    color=segment_color, 
                    linewidth=segment_width,
                    zorder=30  # Top layer for tails (above active routes)
                )


def _draw_route_tail(tail_points, tail_rgba_color, tail_width, effective_line_width, filename_to_rgba, ax, effective_leading_time=None, fade_out_progress=None, fluffy_tail=False):
    """
    Draw tail segments for a route with color and width interpolation.
    
    Args:
        tail_points (list): List of points for tail drawing
        tail_rgba_color (tuple): RGBA color for tail
        tail_width (float): Tail width multiplier
        effective_line_width (float): Base line width
        filename_to_rgba (dict): Filename to RGBA color mapping
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        effective_leading_time (float, optional): Override leading time for tail gradient calculation
        fade_out_progress (float, optional): Fade-out progress (0.0 = just ended, 1.0 = fully faded)
        fluffy_tail (bool): If True, use LineCollection for "fluffy" tail effect; if False, use individual plots
    """
    if len(tail_points) <= 1:
        return
    
    # Pre-extract all coordinates for better performance (keep this for the actual coordinate extraction)
    tail_lats = [point[1] for point in tail_points]
    tail_lons = [point[2] for point in tail_points]
    
    # SIMULTANEOUS MODE TAIL FIX: Apply fade-out effect if fade_out_progress is provided
    if fade_out_progress is not None:
        # Clamp fade_out_progress to valid range
        fade_out_progress = max(0.0, min(1.0, fade_out_progress))
        
        # Calculate fade-out alpha (1.0 = fully visible, 0.0 = fully transparent)
        fade_out_alpha = 1.0 - fade_out_progress
        
        # If fully faded out, don't draw anything
        if fade_out_alpha <= 0.0:
            return
        
        # Apply fade-out alpha to tail color
        tail_rgba_color = (tail_rgba_color[0], tail_rgba_color[1], tail_rgba_color[2], tail_rgba_color[3] * fade_out_alpha)
    else:
        fade_out_alpha = 1.0  # No fade-out effect
    
    # Calculate time range for gradient calculation
    if effective_leading_time is not None:
        # TAIL-ONLY FIX: Use virtual leading time for proper gradient calculation
        time_min = tail_points[0][4]   # First point has earliest time
        time_max = effective_leading_time  # Use virtual leading time
        time_range = time_max - time_min
    else:
        # Normal operation: use actual point times
        # OPTIMIZED: Since tail_points are chronologically ordered, directly access first and last points
        # instead of iterating through the entire list to find min/max
        time_min = tail_points[0][4]   # First point has earliest time  
        time_max = tail_points[-1][4]  # Last point has latest time
    time_range = time_max - time_min
    
    if time_range > 0:
        if fluffy_tail:
            # FLUFFY TAIL EFFECT: Use LineCollection for the "fluffy" appearance
            from matplotlib.collections import LineCollection
            
            # Collect all line segments and their properties for LineCollection
            line_segments = []
            line_colors = []
            line_widths = []
            
            # Draw tail segments individually (no batching to avoid wiggling)
            for j in range(len(tail_points) - 1):
                current_point = tail_points[j]
                next_point = tail_points[j + 1]
                
                # Skip segments that cross track boundaries
                if (len(current_point) >= 7 and current_point[6]) or (len(next_point) >= 7 and next_point[6]):
                    continue
                
                # Calculate interpolation factor using the effective time range
                segment_time = (tail_points[j][4] + tail_points[j + 1][4]) / 2  # Use accumulated_time directly from points
                interp_factor = (segment_time - time_min) / time_range
                
                # Clamp interpolation factor to valid range (important for virtual leading time)
                interp_factor = max(0.0, min(1.0, interp_factor))
                
                # Get base color for this segment
                segment_filename = current_point[7] if len(current_point) > 7 else None
                if segment_filename and filename_to_rgba and segment_filename in filename_to_rgba:
                    base_rgba_color = filename_to_rgba[segment_filename]
                    # SIMULTANEOUS MODE TAIL FIX: Apply fade-out alpha to base color
                    base_rgba_color = (base_rgba_color[0], base_rgba_color[1], base_rgba_color[2], base_rgba_color[3] * fade_out_alpha)
                else:
                    base_rgba_color = (1.0, 0.0, 0.0, 1.0 * fade_out_alpha)
                
                # Pre-calculate interpolated values
                segment_width = effective_line_width + (effective_line_width * tail_width - effective_line_width) * interp_factor
                segment_color = (
                    base_rgba_color[0] + (tail_rgba_color[0] - base_rgba_color[0]) * interp_factor,
                    base_rgba_color[1] + (tail_rgba_color[1] - base_rgba_color[1]) * interp_factor,
                    base_rgba_color[2] + (tail_rgba_color[2] - base_rgba_color[2]) * interp_factor,
                    base_rgba_color[3] + (tail_rgba_color[3] - base_rgba_color[3]) * interp_factor
                )
                
                # Create line segment for this tail portion (convert GPS to Web Mercator coordinates)
                lon1, lat1 = tail_lons[j], tail_lats[j]
                lon2, lat2 = tail_lons[j + 1], tail_lats[j + 1]
                x1, y1 = _gps_to_web_mercator(lon1, lat1)
                x2, y2 = _gps_to_web_mercator(lon2, lat2)
                
                # Add to LineCollection data
                line_segments.append([(x1, y1), (x2, y2)])
                line_colors.append(segment_color)
                line_widths.append(segment_width)
            
            # Create and add LineCollection if we have any tail segments
            if line_segments:
                line_collection = LineCollection(
                    line_segments,
                    colors=line_colors,
                    linewidths=line_widths,
                    zorder=30  # Top layer for tails (above active routes)
                )
                ax.add_collection(line_collection)
        
        else:
            # STANDARD TAILS: Use individual ax.plot() calls for crisp, individual segments
            # Draw tail segments individually (no batching to avoid wiggling)
            for j in range(len(tail_points) - 1):
                current_point = tail_points[j]
                next_point = tail_points[j + 1]
                
                # Skip segments that cross track boundaries
                if (len(current_point) >= 7 and current_point[6]) or (len(next_point) >= 7 and next_point[6]):
                    continue
                
                # Calculate interpolation factor using the effective time range
                segment_time = (tail_points[j][4] + tail_points[j + 1][4]) / 2  # Use accumulated_time directly from points
                interp_factor = (segment_time - time_min) / time_range
                
                # Clamp interpolation factor to valid range (important for virtual leading time)
                interp_factor = max(0.0, min(1.0, interp_factor))
                
                # Get base color for this segment
                segment_filename = current_point[7] if len(current_point) > 7 else None
                if segment_filename and filename_to_rgba and segment_filename in filename_to_rgba:
                    base_rgba_color = filename_to_rgba[segment_filename]
                    # SIMULTANEOUS MODE TAIL FIX: Apply fade-out alpha to base color
                    base_rgba_color = (base_rgba_color[0], base_rgba_color[1], base_rgba_color[2], base_rgba_color[3] * fade_out_alpha)
                else:
                    base_rgba_color = (1.0, 0.0, 0.0, 1.0 * fade_out_alpha)
                
                # Pre-calculate interpolated values
                segment_width = effective_line_width + (effective_line_width * tail_width - effective_line_width) * interp_factor
                segment_color = (
                    base_rgba_color[0] + (tail_rgba_color[0] - base_rgba_color[0]) * interp_factor,
                    base_rgba_color[1] + (tail_rgba_color[1] - base_rgba_color[1]) * interp_factor,
                    base_rgba_color[2] + (tail_rgba_color[2] - base_rgba_color[2]) * interp_factor,
                    base_rgba_color[3] + (tail_rgba_color[3] - base_rgba_color[3]) * interp_factor
                )
                
                # Draw this segment individually (convert GPS to Web Mercator coordinates)
                lon1, lat1 = tail_lons[j], tail_lats[j]
                lon2, lat2 = tail_lons[j + 1], tail_lats[j + 1]
                x1, y1 = _gps_to_web_mercator(lon1, lat1)
                x2, y2 = _gps_to_web_mercator(lon2, lat2)
                
                ax.plot(
                    [x1, x2], 
                    [y1, y2], 
                    color=segment_color, 
                    linewidth=segment_width,
                    zorder=30  # Top layer for tails (above active routes)
                )


def _draw_video_title(ax, title_text, effective_line_width, json_data, resolution_scale=None):
    """
    Draw video title at the top center of the frame.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        title_text (str): Title text to display
        effective_line_width (float): Base line width for scaling (unused, kept for compatibility)
        json_data (dict): Job data containing video parameters for resolution scaling
        resolution_scale (float, optional): Pre-calculated resolution scale factor for optimization
    """
    if not title_text:
        return
    
    # Use hardcoded base font size that scales with resolution
    base_font_size = 16  # Hardcoded base size for 1080p
    font_size = base_font_size * resolution_scale
    
    # Position at top center with 10px padding
    # Use relative coordinates for consistent positioning across different resolutions
    padding_factor = 0.01  # Roughly equivalent to 10px padding
    text_x = 0.5          # Center horizontally (50% of axis)
    text_y = 1.0 - padding_factor  # Top edge minus padding
    
    # Draw title text with light gray color and black outline
    ax.text(
        text_x, text_y, title_text,
        transform=ax.transAxes,  # Use axes coordinates
        color='#cccccc',         # Light gray text
            fontsize=font_size,
            fontweight='bold',
        ha='center',             # Center align horizontally
        va='top',                # Top align vertically
        # Create black outline effect using path effects
        path_effects=[
            matplotlib.patheffects.Stroke(linewidth=2, foreground='black'),
            matplotlib.patheffects.Normal()
        ],
        zorder=110  # Very top layer - above everything else
    )


def generate_video_frame_in_memory(frame_number, points_for_frame, json_data, shared_map_cache=None, filename_to_rgba=None, gpx_time_per_video_time=None, stamp_array=None, target_time=None, shared_route_cache=None, virtual_leading_time=None, route_specific_tail_info=None):
    """
    Generate a single frame for the video in memory, returning numpy array instead of saving to disk.
    Uses shared memory cache for map images exclusively.
    Supports both single route and multiple routes modes.
       
    Args:
        frame_number (int): Frame number being generated
        points_for_frame (list): List of route points for this frame (single route) or list of sub-lists (multiple routes)
        json_data (dict): Job data containing video parameters
        shared_map_cache (dict, optional): Shared memory cache for map images (required)
        filename_to_rgba (dict, optional): Pre-computed filename to RGBA color mapping
        gpx_time_per_video_time (float, optional): GPX time per video time ratio
        stamp_array (numpy.ndarray, optional): Pre-loaded stamp image array
        target_time (float, optional): Target elapsed time for this frame (used for timestamp calculation)
        shared_route_cache (dict, optional): Shared memory cache for route images
        virtual_leading_time (float, optional): Virtual leading time for tail-only phase (in route seconds)
        route_specific_tail_info (dict, optional): Dictionary containing route-specific tail information
    
    Returns:
        numpy array representing the frame image (H, W, 3) or None if failed
    """
    
    if not points_for_frame:
        return None
    
    # NORMALIZE INPUT: Ensure points_for_frame is always a list of lists
    # Check if we have single route (list of points) or multiple routes (list of lists)
    if points_for_frame and not isinstance(points_for_frame[0], list):
        # Single route mode - wrap the single list of points in another list
        points_for_frame = [points_for_frame]
    
    # Check if route has at least 2 points for frame generation
    total_points = sum(len(route_points) for route_points in points_for_frame if route_points)
    if total_points < 2:
        # Create black frame if route has fewer than 2 points
        height = int(json_data.get('video_resolution_y', 1080))
        width = int(json_data.get('video_resolution_x', 1920))
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Validate that shared map cache is provided
    if shared_map_cache is None:
        print(f"Error: Shared map cache is required for frame generation")
        return None
       
    zoom_mode = json_data.get('zoom_mode', 'dynamic')
    
    # Get video parameters from json_data
    width = int(json_data.get('video_resolution_x', 1920))
    height = int(json_data.get('video_resolution_y', 1080))
    map_transparency = float(json_data.get('map_transparency', 0))
    map_opacity = 100 - map_transparency
    line_thickness = float(json_data.get('line_width', 3))
    
    # Get year colors from json_data if available
    year_colors = json_data.get('year_colors', {})
    
    # OPTIMIZATION: Pre-calculate constants for statistics (these never change between frames)
    is_multiple_routes = True  # Function now always handles multiple routes mode
    original_route_end_frame = None
    
   
    video_length = float(json_data.get('video_length', 30))
    video_fps = float(json_data.get('video_fps', 30))
    original_route_end_frame = int(video_length * video_fps)
    
    dpi = 100
    figsize = (width / dpi, height / dpi)
    
    # Multiple routes mode - points_for_frame is a list of sub-lists
    tracks = []
    
    for route_points in points_for_frame:
        if not route_points:  # Skip empty routes
            continue
        
        # OPTIMIZED: Split this route's points into tracks using pre-computed boundaries + slicing
        # Instead of O(n) iteration + individual appends, we find boundaries once and slice efficiently
        track_boundaries = _find_track_boundaries(route_points)
        
        # Use slicing to create tracks (much faster than individual appends)
        for start_idx, end_idx in track_boundaries:
            track = route_points[start_idx:end_idx]  # O(1) slice operation
            if track:  # Only add non-empty tracks
                tracks.append(track)

    # Check zoom mode
    zoom_mode = json_data.get('zoom_mode', 'dynamic')
    
    if zoom_mode == 'final':
        # Use the pre-calculated final bounding box for all frames
        final_bbox = load_final_bounding_box()
        if final_bbox is None:
            print(f"Error: Could not load final bounding box for frame {frame_number}")
            return None
        bbox = final_bbox
    else:
        # Dynamic zoom mode - calculate bounding box for current frame
        # Flatten points_for_frame to get all points for this frame
        all_points = []
        for route_points in points_for_frame:
            if route_points:
                all_points.extend(route_points)
        
        if not all_points:
            print(f"Error: No points available for frame {frame_number}")
            return None
        
        # Use the shared bounding box calculation function for consistency
        bbox = calculate_bounding_box_for_points(all_points, padding_percent=0.1)
        
        if bbox is None:
            print(f"Error: Could not calculate bounding box for frame {frame_number}")
            return None
    
    # Check if we have a precached map image for these bounds
    encoded_bbox = encode_coords(*bbox)
    
    # Try to load from shared memory cache (exclusive approach for better performance)
    img = None
    if shared_map_cache is not None and encoded_bbox in shared_map_cache:
        # Load from shared memory (fastest)
        img = shared_map_cache[encoded_bbox]
    else:
        # Image not found in shared cache
        print(f"ERROR: Map image not found in shared cache for frame {frame_number}: {encoded_bbox}")
        print(f"Frame {frame_number}: Bounding box: {bbox}")
        print(f"Frame {frame_number}: Encoded bbox: {encoded_bbox}")
        return None
    
    try:
        # Convert GPS bounding box to Web Mercator coordinates (same as cached map images)
        mercator_bbox = _convert_bbox_to_web_mercator(bbox)
        
        # Create a figure with the appropriate dimensions and no padding
        plt.rcParams['figure.constrained_layout.use'] = False
        fig = plt.figure(figsize=figsize, facecolor='white', frameon=False)
        
        # Use regular matplotlib axes in Web Mercator coordinate space
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()  # Turn off axis
        fig.add_axes(ax)
        
        # Display the cached map image in Web Mercator coordinate space
        alpha = map_opacity / 100.0
        ax.imshow(img, extent=mercator_bbox, aspect='auto', alpha=alpha, zorder=0)  # Bottom layer for base map
        
        # Set the axis limits to Web Mercator coordinates
        ax.set_xlim(mercator_bbox[0], mercator_bbox[1])
        ax.set_ylim(mercator_bbox[2], mercator_bbox[3])
        
        # OPTIMIZATION: Pre-calculate resolution scale once per frame (constant for entire video)
        resolution_scale = _get_resolution_scale_factor(json_data)
        
        # Calculate effective line width based on resolution scaling (consistent with font scaling)
        base_line_scale = 1.0  # Base scale for 1080p
        line_scale = base_line_scale * resolution_scale
        
        desired_pixels = line_thickness * line_scale
        effective_line_width = desired_pixels * 72 / 100  # dpi fixed to 100
             
        # PHASE 1: Draw completed routes from cache (bottom layer)
        # OPTIMIZATION: Use cached route images for completed tracks in final zoom mode
        tail_length = int(json_data.get('tail_length', 0))
        
        active_tracks = []     # Track which tracks need active drawing
        
        # FIX: Use the exact same track splitting logic as caching phase
        # Group tracks by route_index first, then split each route into tracks
        route_tracks = {}  # {route_index: [tracks]}
        
        for route_points in points_for_frame:
            if not route_points:
                continue
            
            # Get route_index from first point
            route_index = route_points[0][0] if len(route_points) > 0 and len(route_points[0]) > 0 else 0
            
            if route_index not in route_tracks:
                route_tracks[route_index] = []
            
            # Split this route's points into tracks using the same logic as caching phase
            tracks_for_route = []
            current_track = []
            
            for point in route_points:
                if len(point) >= 7:  # Check for minimum length
                    new_route_flag = point[6]  # New route flag at index 6
                    if new_route_flag and current_track:  # If new_route is True and we have points
                        tracks_for_route.append(current_track)
                        current_track = []
                current_track.append(point)
            
            # Add the last track if not empty
            if current_track:
                tracks_for_route.append(current_track)
            
            route_tracks[route_index].extend(tracks_for_route)
        
        # Now process each route's tracks with correct indexing (same as caching phase)
        for route_index, tracks_for_route in route_tracks.items():
            for track_index, track in enumerate(tracks_for_route):
                if not track:  # Skip empty tracks
                    continue

                # Just add all tracks to active tracks (used to be a failed attempt at caching full routes - plotting is still faster than overlaying image)               
                active_tracks.append((route_index, track_index, track))
                      
        # PHASE 2: Draw active routes and their tails (top layer)
        # Optimization: Group active routes by filename for better LineCollection performance
        # Since all active routes have the same width, we can create separate LineCollections per filename
        from matplotlib.collections import LineCollection
        
        # Group active tracks by filename for optimized LineCollection creation
        filename_tracks = {}  # {filename: [track_segments]}
        
        for track_info in active_tracks:
            route_index, track_index, track = track_info
            
            # Draw the main track line
            track_lats = [point[1] for point in track]  # Latitude at index 1
            track_lons = [point[2] for point in track]  # Longitude at index 2

            # Get filename from first point of the track (index 7)
            filename = track[0][7] if len(track[0]) > 7 else None
                      
            # Look up pre-computed RGBA color for this filename
            if filename and filename_to_rgba:
                if filename in filename_to_rgba:
                    rgba_color = filename_to_rgba[filename]
                else:
                    # Try to find a matching filename with different case or whitespace
                    matching_key = next((k for k in filename_to_rgba.keys() 
                                      if k and filename and k.lower() == filename.lower()), None)
                    if matching_key:
                        rgba_color = filename_to_rgba[matching_key]
                    else:
                        print(f"WARNING: No color mapping found for filename: '{filename}'. Using default red.")
                        rgba_color = (1.0, 0.0, 0.0, 1.0)  # Default red if no color mapping found
            else:
                print(f"WARNING: No filename or no filename_to_rgba mapping. Using default red for track.")
                rgba_color = (1.0, 0.0, 0.0, 1.0)  # Default red if no color mapping found
            
            # Convert GPS coordinates to Web Mercator for plotting (same coordinate system as cached images)
            mercator_coords = [_gps_to_web_mercator(lon, lat) for lon, lat in zip(track_lons, track_lats)]
            mercator_track_lons = [coord[0] for coord in mercator_coords]
            mercator_track_lats = [coord[1] for coord in mercator_coords]

            # Create line segment for this track
            # LineCollection expects segments as [(x1, y1), (x2, y2), ...] format
            segment = list(zip(mercator_track_lons, mercator_track_lats))
            
            # Group by filename for optimized LineCollection creation
            if filename not in filename_tracks:
                filename_tracks[filename] = {
                    'segments': [],
                    'color': rgba_color
                }
            filename_tracks[filename]['segments'].append(segment)
        
        # Create separate LineCollection for each filename (better performance)
        for filename, track_data in filename_tracks.items():
            if track_data['segments']:
                line_collection = LineCollection(
                    track_data['segments'],
                    color=track_data['color'],  # Single color for all segments in this filename
                    linewidth=effective_line_width,  # Single width for all active routes
                    zorder=20  # Middle layer for active routes (below tails)
                )
                ax.add_collection(line_collection)
        
        # PHASE 3: Draw tails for all routes (top layer - drawn after all route lines)
        if tail_length > 0 and gpx_time_per_video_time is not None:
            # Calculate tail duration in route time
            tail_duration_route = gpx_time_per_video_time * tail_length
            tail_width = float(json_data.get('tail_width', 2))
            tail_color_setting = json_data.get('tail_color', 'light')
            
            # NEW APPROACH: Group all points by route_index to identify independent routes
            route_groups = {}  # {route_index: [points]}
            
            for route_points in points_for_frame:
                if not route_points:
                    continue
                
                for point in route_points:
                    if len(point) >= 8:  # Ensure we have route_index
                        route_index = point[0]  # route_index at index 0
                        if route_index not in route_groups:
                            route_groups[route_index] = []
                        route_groups[route_index].append(point)
            
            # Draw independent tail for each route_index
            for route_index, route_points in route_groups.items():
                if len(route_points) < 2:  # Need at least 2 points for a tail
                    continue
                              
                # Determine the effective leading time (virtual or actual)
                effective_leading_time = None
                
                # SIMULTANEOUS MODE TAIL FIX: Use route-specific virtual leading time if available
                if route_specific_tail_info and route_index in route_specific_tail_info:
                    route_info = route_specific_tail_info[route_index]
                    # Check if virtual_leading_time exists (only for tail-only frames)
                    if 'virtual_leading_time' in route_info:
                        effective_leading_time = route_info['virtual_leading_time']
                    else:
                        # For normal frames, use the actual leading point time
                        leading_point = route_points[-1]
                        effective_leading_time = leading_point[4]  # accumulated_time
                elif virtual_leading_time is not None:
                    # Fallback to global virtual leading time (for sequential mode)
                    effective_leading_time = virtual_leading_time
                else:
                    # Normal operation: use actual leading point time
                    leading_point = route_points[-1]
                    effective_leading_time = leading_point[4]  # accumulated_time
                
                # Calculate tail start time (backwards from effective leading time)
                tail_start_time = effective_leading_time - tail_duration_route
                
                # OPTIMIZED: Use binary search + slicing instead of reverse iteration + individual appends
                # Since points are chronological, we can find the exact cutoff point and slice efficiently
                cutoff_index = _binary_search_tail_start_index(route_points, tail_start_time)
                tail_points = route_points[cutoff_index:]  # All points from cutoff to end (already in chronological order)
                
                # Draw tail if we have enough points
                if len(tail_points) > 1:
                    # Get route color
                    route_filename = tail_points[0][7] if len(tail_points[0]) > 7 else None
                    if route_filename and filename_to_rgba and route_filename in filename_to_rgba:
                        route_color_rgba = filename_to_rgba[route_filename]
                    else:
                        route_color_rgba = (1.0, 0.0, 0.0, 1.0)
                    
                    # Get tail color
                    tail_rgba_color = get_tail_color_for_route(tail_color_setting, route_color_rgba)
                    
                    # SIMULTANEOUS MODE TAIL FIX: Calculate fade-out progress for this route
                    fade_out_progress = None
                    if route_specific_tail_info and route_index in route_specific_tail_info:
                        route_info = route_specific_tail_info[route_index]
                        route_end_time = route_info['route_end_time']
                        route_delay_seconds = route_info['route_delay_seconds']
                        
                        # Calculate how much time has passed since this route ended
                        current_route_time = target_time * gpx_time_per_video_time if target_time and gpx_time_per_video_time else 0
                        time_since_route_end = current_route_time - route_end_time
                        
                        # Calculate fade-out progress (0.0 = just ended, 1.0 = fully faded)
                        tail_duration_route = gpx_time_per_video_time * tail_length if gpx_time_per_video_time else 0
                        if tail_duration_route > 0:
                            fade_out_progress = time_since_route_end / tail_duration_route
                            fade_out_progress = max(0.0, min(1.0, fade_out_progress))  # Clamp to valid range
                    
                    # Draw the tail with fade-out progress (zorder=30 - above routes but below labels)
                    _draw_multi_route_tail(tail_points, tail_color_setting, tail_width, effective_line_width, filename_to_rgba, ax, fade_out_progress, fluffy_tail=json_data.get('fluffy_tail', False))
        
        name_tags_setting = json_data.get('name_tags')
        if name_tags_setting in ['light', 'dark']:
            _draw_name_tags_for_routes(points_for_frame, json_data, filename_to_rgba, effective_line_width, ax)

        # Add legend if requested (zorder=45 - above tails but below statistics)
        legend_type = json_data.get('legend', '')
        if legend_type in ['file_name', 'year', 'month', 'day', 'people']:
            # Get legend theme colors (same for all legend types)
            legend_theme = json_data.get('legend_theme', 'light')
            theme_colors = get_legend_theme_colors(legend_theme)
            
            if legend_type == 'file_name':
                # Get legend data - flatten points_for_frame for legend generation
                flattened_points = [point for route_points in points_for_frame for point in route_points]
                legend_handles, legend_labels = get_filename_legend_data(flattened_points, json_data, effective_line_width)
                
                if legend_handles and legend_labels:
                    create_legend(ax, legend_handles, legend_labels, theme_colors, effective_line_width)
            
            elif legend_type == 'year':
                # Get legend data - flatten points_for_frame for legend generation
                flattened_points = [point for route_points in points_for_frame for point in route_points]
                legend_handles, legend_labels = get_year_legend_data(flattened_points, json_data, effective_line_width)
                
                if legend_handles and legend_labels:
                    create_legend(ax, legend_handles, legend_labels, theme_colors, effective_line_width)
            
            elif legend_type == 'month':
                # Get legend data - flatten points_for_frame for legend generation
                flattened_points = [point for route_points in points_for_frame for point in route_points]
                legend_handles, legend_labels = get_month_legend_data(flattened_points, json_data, effective_line_width)
                
                if legend_handles and legend_labels:
                    create_legend(ax, legend_handles, legend_labels, theme_colors, effective_line_width)
            
            elif legend_type == 'day':
                # Get legend data - flatten points_for_frame for legend generation
                flattened_points = [point for route_points in points_for_frame for point in route_points]
                legend_handles, legend_labels = get_day_legend_data(flattened_points, json_data, effective_line_width)
                
                if legend_handles and legend_labels:
                    create_legend(ax, legend_handles, legend_labels, theme_colors, effective_line_width)
            
            elif legend_type == 'people':
                # Get legend data - flatten points_for_frame for legend generation
                flattened_points = [point for route_points in points_for_frame for point in route_points]
                legend_handles, legend_labels = get_people_legend_data(flattened_points, json_data, effective_line_width)
                
                if legend_handles and legend_labels:
                    create_legend(ax, legend_handles, legend_labels, theme_colors, effective_line_width)
        
        # Add title text if enabled (boolean flag) (zorder=55 - top layer, above statistics)
        if isinstance(json_data.get('title_text'), bool) and json_data.get('title_text') is True:
            title_text_value = (json_data.get('route_name') or '').strip()
            if title_text_value:
                _draw_video_title(ax, title_text_value, effective_line_width, json_data, resolution_scale=resolution_scale)
        
        # Add statistics if enabled (zorder=50 - top layer, above everything else) - MOVED TO END
        statistics_setting = json_data.get('statistics', 'off')
        if statistics_setting in ['light', 'dark']:
            statistics_data = _calculate_video_statistics(
                points_for_frame, 
                json_data, 
                gpx_time_per_video_time, 
                frame_number,
                is_multiple_routes,  # Pass pre-calculated route type
                original_route_end_frame  # Pass pre-calculated end frame
            )
            if statistics_data:
                # Check if current speed should be displayed separately at the point
                current_speed_enabled = json_data.get('statistics_current_speed', False)
                current_speed_value = statistics_data.get('current_speed')
                
                # Check if current elevation should be displayed separately at the point
                current_elevation_enabled = json_data.get('statistics_current_elevation', False)
                current_elevation_value = statistics_data.get('current_elevation')
                
                # Determine if we're at the end of the route (suppress current speed/elevation display)
                at_end_of_route = False
                if frame_number is not None and original_route_end_frame is not None:
                    # Suppress current speed/elevation display during:
                    # 1. Last frame of original route
                    # 2. Tail-only phase (after original route ends)  
                    # 3. Cloned frames (handled by tail-only logic)
                    at_end_of_route = frame_number >= original_route_end_frame
                
                # Only show current speed if enabled, available, and not at end of route
                if current_speed_enabled and current_speed_value and not at_end_of_route:
                    # Draw current speed near the latest point
                    _draw_current_speed_at_point(ax, points_for_frame, current_speed_value, effective_line_width, statistics_setting, json_data, resolution_scale=resolution_scale)
                
                # Only show current elevation if enabled, available, and not at end of route
                if current_elevation_enabled and current_elevation_value and not at_end_of_route:
                    # Draw current elevation near the latest point (below current speed)
                    _draw_current_elevation_at_point(ax, points_for_frame, current_elevation_value, effective_line_width, statistics_setting, json_data, resolution_scale=resolution_scale)
                
                # Draw other statistics in top-right corner
                # Exclude current speed if we're at end of route, otherwise include it normally
                exclude_speed = at_end_of_route or not current_speed_enabled or not current_speed_value
                # Also exclude current elevation from top-right display since it's shown at the point
                exclude_elevation = current_elevation_enabled and current_elevation_value and not at_end_of_route
                _draw_video_statistics(ax, statistics_data, json_data, effective_line_width, statistics_setting, exclude_current_speed=exclude_speed, exclude_current_elevation=exclude_elevation, resolution_scale=resolution_scale)
        
        # Convert figure to numpy array directly (much faster than PNG buffer)
        # Draw the figure to the canvas
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        buf = fig.canvas.buffer_rgba()
        
        # Convert to numpy array
        frame_array = np.asarray(buf)
        
        # Convert RGBA to RGB (remove alpha channel)
        frame_array = frame_array[:, :, :3]
        
        # Add stamp to the frame if provided
        if stamp_array is not None:
            height = int(json_data.get('video_resolution_y', 1080))
            width = int(json_data.get('video_resolution_x', 1920))
            
            # Get stamp dimensions
            stamp_height, stamp_width = stamp_array.shape[:2]
            
            # Calculate position (bottom left with 20 pixels padding)
            padding = 20
            y_start = height - padding - stamp_height
            y_end = height - padding
            x_start = padding
            x_end = padding + stamp_width
            
            # Ensure stamp fits within frame bounds
            if y_start >= 0 and x_start >= 0 and y_end <= height and x_end <= width:
                # Check if stamp has alpha channel (RGBA)
                if stamp_array.shape[2] == 4:
                    # RGBA stamp - apply alpha blending
                    alpha = stamp_array[:, :, 3:4] / 255.0  # Normalize alpha to 0-1
                    rgb_stamp = stamp_array[:, :, :3] / 255.0  # Normalize RGB to 0-1
                    
                    # Extract the region where stamp will be placed
                    frame_region = frame_array[y_start:y_end, x_start:x_end].astype(np.float32) / 255.0
                    
                    # Alpha blend: result = alpha * stamp + (1 - alpha) * background
                    blended = alpha * rgb_stamp + (1 - alpha) * frame_region
                    
                    # Convert back to uint8 and place in frame
                    frame_array[y_start:y_end, x_start:x_end] = (blended * 255).astype(np.uint8)
                else:
                    # RGB stamp - direct replacement (no transparency)
                    frame_array[y_start:y_end, x_start:x_end] = stamp_array
            else:
                print(f"Warning: Stamp does not fit within frame bounds for frame {frame_number}")
        
        # Clean up matplotlib figure to prevent memory leaks
        plt.close(fig)
        
        return frame_array
        
    except Exception as e:
        print(f"Error generating frame {frame_number}: {e}")
        # Clean up matplotlib figure on error
        try:
            plt.close(fig)
        except:
            pass
        return None 