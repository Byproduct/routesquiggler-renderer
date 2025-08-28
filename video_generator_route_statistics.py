"""
Video generation statistics functions.
This module contains functions for calculating and drawing statistics on video frames.
"""

from datetime import datetime
import math


def _get_resolution_scale_factor(json_data):
    """
    Get font size scale factor based on vertical video resolution.
    Uses 1080p as the baseline (scale factor = 1.0).
    Uses linear scaling to maintain consistent proportional screen space.
    
    Args:
        json_data (dict): Job data containing video parameters
    
    Returns:
        float: Scale factor for font sizes
    """
    resolution_y = int(json_data.get('video_resolution_y', 1080))
    baseline_resolution = 1080
    # Use linear scaling to maintain same proportion of screen space
    return resolution_y / baseline_resolution


def _calculate_video_statistics(points_for_frame, json_data, gpx_time_per_video_time, frame_number=None, is_multiple_routes=None, original_route_end_frame=None):
    """
    Calculate statistics for video frame from the current points_for_frame data.
    OPTIMIZED version to minimize per-frame computational overhead.
    
    Args:
        points_for_frame (list): List of route points (single route) or list of sub-lists (multiple routes)
        json_data (dict): Job data containing statistics configuration
        gpx_time_per_video_time (float): GPX time per video time ratio
        frame_number (int, optional): Current frame number for tail detection
        is_multiple_routes (bool, optional): Pre-calculated route type detection
        original_route_end_frame (int, optional): Pre-calculated end frame for tail detection
    
    Returns:
        dict: Dictionary containing calculated statistics or None if calculation fails
    """
    if not points_for_frame:
        return None
    
    # OPTIMIZATION 1: Use pre-calculated constants instead of recalculating every frame
    if original_route_end_frame is None and frame_number is not None:
        # Fallback calculation if not provided (should be avoided in production)
        video_length = float(json_data.get('video_length', 30))
        video_fps = float(json_data.get('video_fps', 30))
        original_route_end_frame = int(video_length * video_fps)
    
    # Detect if we're in tail-only mode
    is_tail_only = frame_number is not None and original_route_end_frame is not None and frame_number > original_route_end_frame
    
    # OPTIMIZATION 2: Use pre-calculated route type detection
    if is_multiple_routes is None:
        # Fallback detection if not provided (should be avoided in production)
        is_multiple_routes = points_for_frame and isinstance(points_for_frame[0], list)
    
    # OPTIMIZATION 3: Get last point efficiently without expensive list operations
    last_point = None
    
    if is_multiple_routes:
        # Multiple routes mode - find the most recent point by checking last point of each route
        # Since routes are chronologically ordered, we can optimize this
        latest_time = None
        for route_points in reversed(points_for_frame):  # Start from end since more likely to have recent data
            if not route_points:  # Skip empty routes
                continue
            route_last_point = route_points[-1]
            if len(route_last_point) >= 5:
                point_time = route_last_point[4]  # accumulated_time at index 4
                if latest_time is None or point_time > latest_time:
                    latest_time = point_time
                    last_point = route_last_point
            if latest_time is not None:  # Found a valid point, can break early for efficiency
                break
    else:
        # Single route mode - simple and fast
        if points_for_frame:
            last_point = points_for_frame[-1]
    
    if not last_point or len(last_point) < 6:
        return None
    
    statistics_data = {}
    
    # Extract data from last point
    timestamp = last_point[3]  # Element 3: timestamp
    accumulated_time = last_point[4]  # Element 4: accumulated_time (seconds)
    accumulated_distance = last_point[5]  # Element 5: accumulated_distance (meters)
    
    # Statistics: Current time (timestamp of last point)
    if json_data.get('statistics_current_time', False) and timestamp:
        try:
            current_time_str = timestamp.strftime('%Y-%m-%d %H:%M')
            statistics_data['current_time'] = current_time_str
        except:
            statistics_data['current_time'] = str(timestamp)
    
    # Statistics: Elapsed time (format depends on whether we're in tail mode)
    if json_data.get('statistics_elapsed_time', False):
        hours = int(accumulated_time // 3600)
        minutes = int((accumulated_time % 3600) // 60)
        seconds = int(accumulated_time % 60)
        
        if is_tail_only:
            # Tail mode: show seconds for final precise time
            if hours > 0:
                elapsed_time_str = f"{hours}h {minutes:02d}min {seconds:02d}s"
            elif minutes > 0:
                elapsed_time_str = f"{minutes:02d}min {seconds:02d}s"
            else:
                elapsed_time_str = f"{seconds:02d}s"
        else:
            # Normal mode: show only hours and minutes for cleaner display
            if hours > 0:
                elapsed_time_str = f"{hours}h {minutes:02d}min"
            elif minutes > 0:
                elapsed_time_str = f"{minutes:02d}min"
            else:
                elapsed_time_str = f"{seconds:02d}s"  # Still show seconds if less than a minute
        
        statistics_data['elapsed_time'] = elapsed_time_str
    
    # Statistics: Distance (accumulated distance converted to km with 1 decimal)
    if json_data.get('statistics_distance', False):
        distance_km = accumulated_distance / 1000.0
        statistics_data['distance'] = f"{distance_km:.1f}"
    
    # Statistics: Speed handling (different logic for normal vs tail mode)
    current_speed_requested = json_data.get('statistics_current_speed', False)
    average_speed_requested = json_data.get('statistics_average_speed', False)
    current_elevation_requested = json_data.get('statistics_current_elevation', False)
    
    if is_tail_only:
        # Tail mode: hide current speed, show average speed if either speed stat is requested
        if current_speed_requested or average_speed_requested:
            if accumulated_time > 0:
                avg_speed_kmh = (accumulated_distance / 1000.0 * 3600) / accumulated_time
                statistics_data['average_speed'] = f"{avg_speed_kmh:.1f}"
            else:
                statistics_data['average_speed'] = "0.0"
    else:
        # Normal mode: show both speeds as requested
        
        # Average speed calculation
        if average_speed_requested:
            if accumulated_time > 0:
                avg_speed_kmh = (accumulated_distance / 1000.0 * 3600) / accumulated_time
                statistics_data['average_speed'] = f"{avg_speed_kmh:.1f}"
            else:
                statistics_data['average_speed'] = "0.0"
        
        # Current speed calculation - smoothed over 0.5 video seconds.
        if current_speed_requested and gpx_time_per_video_time is not None and gpx_time_per_video_time > 0:
            # Calculate how much route time corresponds to 0.5 video seconds
            video_seconds_to_check = 0.5
            route_time_threshold = video_seconds_to_check * gpx_time_per_video_time
            
            # Find point that is closest to being 0.5 video seconds back from last point
            target_time = accumulated_time - route_time_threshold
            comparison_point = None
            
            # OPTIMIZATION 4: Efficient point search using chronological ordering
            if is_multiple_routes:
                # Search through routes efficiently - start from the route with the last point
                # and work backwards since we know the data is chronologically ordered
                for route_points in reversed(points_for_frame):
                    if not route_points:
                        continue
                    
                    # Check if this route has data in our target time range
                    if len(route_points) > 0 and len(route_points[0]) >= 5:
                        route_start_time = route_points[0][4]
                        route_end_time = route_points[-1][4]
                        
                        # If target time is within this route's range, search it
                        if target_time >= route_start_time and target_time <= route_end_time:
                            # Binary search would be even faster, but reverse iteration is good enough
                            for point in reversed(route_points):
                                if len(point) >= 5 and point[4] <= target_time:
                                    comparison_point = point
                                    break
                            if comparison_point:
                                break
                        elif target_time > route_end_time:
                            # Target time is after this route, use the last point from this route
                            comparison_point = route_points[-1]
                            break
            else:
                # Single route mode - simple reverse search
                for point in reversed(points_for_frame):
                    if len(point) >= 5 and point[4] <= target_time:
                        comparison_point = point
                        break
            
            if comparison_point and len(comparison_point) >= 6:
                time_diff = accumulated_time - comparison_point[4]  # Time difference in seconds
                distance_diff = accumulated_distance - comparison_point[5]  # Distance difference in meters
                
                if time_diff > 0:
                    # Calculate current speed in km/h
                    current_speed_kmh = (distance_diff / 1000.0 * 3600) / time_diff
                    statistics_data['current_speed'] = f"{round(current_speed_kmh)}"  # Rounded to no decimals
                else:
                    statistics_data['current_speed'] = "0"
            else:
                # OPTIMIZATION 5: Efficient fallback using first point from appropriate route structure
                first_point = None
                if is_multiple_routes:
                    # Find the first point across all routes
                    for route_points in points_for_frame:
                        if route_points and len(route_points[0]) >= 6:
                            first_point = route_points[0]
                            break
                else:
                    # Single route mode
                    if points_for_frame and len(points_for_frame[0]) >= 6:
                        first_point = points_for_frame[0]
                
                if first_point:
                    time_diff = accumulated_time - first_point[4]
                    distance_diff = accumulated_distance - first_point[5]
                    
                    if time_diff > 0:
                        current_speed_kmh = (distance_diff / 1000.0 * 3600) / time_diff
                        statistics_data['current_speed'] = f"{round(current_speed_kmh)}"
                    else:
                        statistics_data['current_speed'] = "0"
                else:
                    statistics_data['current_speed'] = "0"
        
        # Current elevation calculation - smoothed over 0.5 video seconds.
        if current_elevation_requested and gpx_time_per_video_time is not None and gpx_time_per_video_time > 0:
            # Calculate how much route time corresponds to 0.5 video seconds
            video_seconds_to_check = 0.5
            route_time_threshold = video_seconds_to_check * gpx_time_per_video_time
            
            # Find point that is closest to being 0.5 video seconds back from last point
            target_time = accumulated_time - route_time_threshold
            comparison_point = None
            
            # Use the same efficient point search logic as current speed
            if is_multiple_routes:
                # Search through routes efficiently - start from the route with the last point
                # and work backwards since we know the data is chronologically ordered
                for route_points in reversed(points_for_frame):
                    if not route_points:
                        continue
                    
                    # Check if this route has data in our target time range
                    if len(route_points) > 0 and len(route_points[0]) >= 5:
                        route_start_time = route_points[0][4]
                        route_end_time = route_points[-1][4]
                        
                        # If target time is within this route's range, search it
                        if target_time >= route_start_time and target_time <= route_end_time:
                            # Binary search would be even faster, but reverse iteration is good enough
                            for point in reversed(route_points):
                                if len(point) >= 5 and point[4] <= target_time:
                                    comparison_point = point
                                    break
                            if comparison_point:
                                break
                        elif target_time > route_end_time:
                            # Target time is after this route, use the last point from this route
                            comparison_point = route_points[-1]
                            break
            else:
                # Single route mode - simple reverse search
                for point in reversed(points_for_frame):
                    if len(point) >= 5 and point[4] <= target_time:
                        comparison_point = point
                        break
            
            # Get current elevation from last point (index 8)
            current_elevation = None
            if last_point and len(last_point) >= 9:
                current_elevation = last_point[8]  # elevation at index 8
            
            # Get comparison elevation from comparison point (index 8)
            comparison_elevation = None
            if comparison_point and len(comparison_point) >= 9:
                comparison_elevation = comparison_point[8]  # elevation at index 8
            
            # Calculate smoothed elevation (average of current and comparison)
            if current_elevation is not None and comparison_elevation is not None:
                smoothed_elevation = (current_elevation + comparison_elevation) / 2
                statistics_data['current_elevation'] = f"{round(smoothed_elevation)}"  # Rounded to nearest integer
            elif current_elevation is not None:
                # Only current elevation available
                statistics_data['current_elevation'] = f"{round(current_elevation)}"  # Rounded to nearest integer
            else:
                # No elevation data available
                statistics_data['current_elevation'] = None
    
    return statistics_data


def _draw_current_speed_at_point(ax, points_for_frame, current_speed, effective_line_width, theme='light', json_data=None, resolution_scale=None):
    """
    Draw current speed near the latest point on the chart.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        points_for_frame (list): List of route points (single route) or list of sub-lists (multiple routes)
        current_speed (str): Current speed value to display
        effective_line_width (float): Base line width for scaling (unused, kept for compatibility)
        theme (str): Theme for speed display ('light' or 'dark')
        json_data (dict): Job data containing video parameters for resolution scaling
        resolution_scale (float, optional): Pre-calculated resolution scale factor for optimization
    """
    if not current_speed or not points_for_frame:
        return
    
    # Find the latest point across all routes to position the speed label
    latest_point = None
    latest_time = None
    
    # Determine if we have multiple routes or single route
    is_multiple_routes = points_for_frame and isinstance(points_for_frame[0], list)
    
    if is_multiple_routes:
        # Multiple routes mode - find the most recent point across all routes
        for route_points in points_for_frame:
            if not route_points:  # Skip empty routes
                continue
            route_last_point = route_points[-1]
            if len(route_last_point) >= 5:
                point_time = route_last_point[4]  # accumulated_time at index 4
                if latest_time is None or point_time > latest_time:
                    latest_time = point_time
                    latest_point = route_last_point
    else:
        # Single route mode
        if points_for_frame:
            latest_point = points_for_frame[-1]
    
    if not latest_point or len(latest_point) < 3:
        return
    
    # Extract coordinates from the latest point
    lat = latest_point[1]  # Latitude at index 1
    lon = latest_point[2]  # Longitude at index 2
    
    # Convert GPS coordinates to Web Mercator for proper positioning
    def _gps_to_web_mercator(lon, lat):
        """Convert GPS coordinates to Web Mercator projection."""
        import math
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34 / 180
        return x, y
    
    x, y = _gps_to_web_mercator(lon, lat)
    
    # Calculate offset for speed label position (scales with resolution)
    # Convert scaled pixels to coordinate space based on current axes limits
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    # Use pre-calculated resolution scale if provided, otherwise calculate it
    if resolution_scale is None:
        resolution_scale = _get_resolution_scale_factor(json_data) if json_data else 1.0
    
    # Calculate resolution-scaled offset (15px baseline for 1080p)
    base_offset_pixels = 15  # Baseline for 1080p
    scaled_offset_pixels = base_offset_pixels * resolution_scale
    
    # Estimate pixel-to-coordinate conversion (rough approximation)
    # This could be more precise, but for positioning it should be adequate
    width = int(ax.figure.get_figwidth() * ax.figure.dpi)
    x_offset = (scaled_offset_pixels / width) * x_range  # Resolution-scaled pixels in coordinate space
    
    speed_x = x + x_offset
    
    # Set theme colors
    if theme == 'dark':
        bg_color = '#2d2d2d'      # Dark gray background
        border_color = '#cccccc'  # Light gray border
        text_color = '#ffffff'    # White text
    else:  # light theme (default)
        bg_color = 'white'        # White background
        border_color = '#333333'  # Dark gray border
        text_color = '#333333'    # Dark gray text
    
    # Use hardcoded base font size that scales with resolution
    base_font_size = 13  # Hardcoded base size for 1080p 
    font_size = base_font_size * resolution_scale
    
    # Add the current speed text
    ax.text(
        speed_x, y, f"{current_speed} km/h",
        color=text_color,
        fontsize=font_size,
        fontweight='bold',
        ha='left',      # Left align (since we're positioning to the right)
        va='center',    # Center vertically
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor=bg_color,
            edgecolor=border_color,
            alpha=0.9,
            linewidth=1
        ),
        zorder=100  # Very top layer - above all other elements
    )


def _draw_current_elevation_at_point(ax, points_for_frame, current_elevation, effective_line_width, theme='light', json_data=None, resolution_scale=None):
    """
    Draw current elevation near the latest point on the chart, below the current speed.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        points_for_frame (list): List of route points (single route) or list of sub-lists (multiple routes)
        current_elevation (str): Current elevation value to display
        effective_line_width (float): Base line width for scaling (unused, kept for compatibility)
        theme (str): Theme for elevation display ('light' or 'dark')
        json_data (dict): Job data containing video parameters for resolution scaling
        resolution_scale (float, optional): Pre-calculated resolution scale factor for optimization
    """
    if not current_elevation or not points_for_frame:
        return
    
    # Find the latest point across all routes to position the elevation label
    latest_point = None
    latest_time = None
    
    # Determine if we have multiple routes or single route
    is_multiple_routes = points_for_frame and isinstance(points_for_frame[0], list)
    
    if is_multiple_routes:
        # Multiple routes mode - find the most recent point across all routes
        for route_points in points_for_frame:
            if not route_points:  # Skip empty routes
                continue
            route_last_point = route_points[-1]
            if len(route_last_point) >= 5:
                point_time = route_last_point[4]  # accumulated_time at index 4
                if latest_time is None or point_time > latest_time:
                    latest_time = point_time
                    latest_point = route_last_point
    else:
        # Single route mode
        if points_for_frame:
            latest_point = points_for_frame[-1]
    
    if not latest_point or len(latest_point) < 3:
        return
    
    # Extract coordinates from the latest point
    lat = latest_point[1]  # Latitude at index 1
    lon = latest_point[2]  # Longitude at index 2
    
    # Convert GPS coordinates to Web Mercator for proper positioning
    def _gps_to_web_mercator(lon, lat):
        """Convert GPS coordinates to Web Mercator projection."""
        import math
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34 / 180
        return x, y
    
    x, y = _gps_to_web_mercator(lon, lat)
    
    # Calculate offset for elevation label position (scales with resolution)
    # Convert scaled pixels to coordinate space based on current axes limits
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    # Use pre-calculated resolution scale if provided, otherwise calculate it
    if resolution_scale is None:
        resolution_scale = _get_resolution_scale_factor(json_data) if json_data else 1.0
    
    # Calculate resolution-scaled offset (15px baseline for 1080p)
    base_offset_pixels = 15  # Baseline for 1080p
    scaled_offset_pixels = base_offset_pixels * resolution_scale
    
    # Estimate pixel-to-coordinate conversion (rough approximation)
    # This could be more precise, but for positioning it should be adequate
    width = int(ax.figure.get_figwidth() * ax.figure.dpi)
    x_offset = (scaled_offset_pixels / width) * x_range  # Resolution-scaled pixels in coordinate space
    
    elevation_x = x + x_offset
    
    # Calculate vertical offset to position below current speed
    # Use a smaller offset for elevation to place it below speed
    base_vertical_offset_pixels = 25  # Baseline for 1080p
    scaled_vertical_offset_pixels = base_vertical_offset_pixels * resolution_scale
    height = int(ax.figure.get_figheight() * ax.figure.dpi)
    y_offset = (scaled_vertical_offset_pixels / height) * y_range  # Resolution-scaled pixels in coordinate space
    
    elevation_y = y - y_offset
    
    # Set theme colors
    if theme == 'dark':
        bg_color = '#2d2d2d'      # Dark gray background
        border_color = '#cccccc'  # Light gray border
        text_color = '#ffffff'    # White text
    else:  # light theme (default)
        bg_color = 'white'        # White background
        border_color = '#333333'  # Dark gray border
        text_color = '#333333'    # Dark gray text
    
    # Use hardcoded base font size that scales with resolution
    base_font_size = 13  # Hardcoded base size for 1080p 
    font_size = base_font_size * resolution_scale
    
    # Add the current elevation text with up arrow symbol
    ax.text(
        elevation_x, elevation_y, f"↑{current_elevation}m",
        color=text_color,
        fontsize=font_size,
        fontweight='bold',
        ha='left',      # Left align (since we're positioning to the right)
        va='center',    # Center vertically
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor=bg_color,
            edgecolor=border_color,
            alpha=0.9,
            linewidth=1
        ),
        zorder=100  # Very top layer - above all other elements
    )


def _draw_video_statistics(ax, statistics_data, json_data, effective_line_width, theme='light', exclude_current_speed=False, exclude_current_elevation=False, resolution_scale=None):
    """
    Draw statistics on the video frame in the top-right corner.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        statistics_data (dict): Dictionary containing calculated statistics
        json_data (dict): Job data containing video parameters and statistics configuration
        effective_line_width (float): Base line width for scaling (unused, kept for compatibility)
        theme (str): Theme for statistics display ('light' or 'dark')
        exclude_current_speed (bool): If True, don't include current speed in this display
        exclude_current_elevation (bool): If True, don't include current elevation in this display
        resolution_scale (float, optional): Pre-calculated resolution scale factor for optimization
    """
    if not statistics_data:
        return
    
    # Build statistics lines to display based on what's available
    stats_lines = []
    
    if 'current_time' in statistics_data:
        stats_lines.append(statistics_data['current_time'])
    
    if 'elapsed_time' in statistics_data:
        stats_lines.append(statistics_data['elapsed_time'])
    
    if 'distance' in statistics_data:
        stats_lines.append(f"{statistics_data['distance']} km")
    
    # Include current speed only if not excluded and not being displayed at point
    current_speed_at_point = json_data.get('statistics_current_speed', False)
    if not exclude_current_speed and 'current_speed' in statistics_data and not current_speed_at_point:
        stats_lines.append(f"{statistics_data['current_speed']} km/h")
    
    if 'average_speed' in statistics_data:
        stats_lines.append(f"{statistics_data['average_speed']} km/h")
    
    # Include current elevation only if not excluded and not being displayed at point
    current_elevation_at_point = json_data.get('statistics_current_elevation', False)
    if not exclude_current_elevation and 'current_elevation' in statistics_data and not current_elevation_at_point:
        stats_lines.append(f"↑{statistics_data['current_elevation']}m")
    
    # If no statistics are available, return
    if not stats_lines:
        return
    
    # Set theme colors
    if theme == 'dark':
        bg_color = '#2d2d2d'      # Dark gray background
        border_color = '#cccccc'  # Light gray border
        text_color = '#ffffff'    # White text
    else:  # light theme (default)
        bg_color = 'white'        # White background
        border_color = '#333333'  # Dark gray border
        text_color = '#333333'    # Dark gray text
    
    # Use pre-calculated resolution scale if provided, otherwise calculate it
    if resolution_scale is None:
        resolution_scale = _get_resolution_scale_factor(json_data)
    
    # Use hardcoded base font size that scales with resolution
    base_font_size = 12  # Hardcoded base size for 1080p
    font_size = base_font_size * resolution_scale
    
    # Combine all statistics lines
    stats_text = '\n'.join(stats_lines)
    
    # Position in top right with 10px padding (converted to relative coordinates)
    # Use relative coordinates (0.99 = 99% of axis) for consistent positioning
    padding_factor = 0.01  # Roughly equivalent to 10px padding
    text_x = 1.0 - padding_factor  # Right edge minus padding
    text_y = 1.0 - padding_factor  # Top edge minus padding
    
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
        ),
        zorder=100  # Very top layer - above all other elements
    ) 