"""
Video generation statistics functions.
This module contains functions for calculating and drawing statistics on video frames.
"""

# Standard library imports
import math
from datetime import datetime

# Local imports
from video_generator_create_combined_route import RoutePoint


def _is_imperial_units(json_data):
    """Returns true if imperial_units is True in json_data."""
    return json_data and json_data.get('imperial_units', False) is True


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
            point_time = route_last_point.accumulated_time
            if latest_time is None or point_time > latest_time:
                latest_time = point_time
                last_point = route_last_point
            if latest_time is not None:  # Found a valid point, can break early for efficiency
                break
    else:
        # Single route mode - simple and fast
        if points_for_frame:
            last_point = points_for_frame[-1]
    
    if not last_point:
        return None
    
    statistics_data = {}
    
    # Extract data from last point using named attributes
    timestamp = last_point.timestamp
    accumulated_time = last_point.accumulated_time
    accumulated_distance = last_point.accumulated_distance
    
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
    
    # Statistics: Distance (accumulated distance converted to km or miles with 1 decimal)
    if json_data.get('statistics_distance', False):
        imperial_units = _is_imperial_units(json_data)
        if imperial_units:
            # accumulated_distance is already in miles
            distance_value = accumulated_distance
        else:
            # accumulated_distance is in meters, convert to km
            distance_value = accumulated_distance / 1000.0
        statistics_data['distance'] = f"{distance_value:.1f}"
    
    # Statistics: Speed handling (different logic for normal vs tail mode)
    current_speed_requested = json_data.get('statistics_current_speed', False)
    average_speed_requested = json_data.get('statistics_average_speed', False)
    current_elevation_requested = json_data.get('statistics_current_elevation', False)
    current_hr_requested = json_data.get('statistics_current_hr', False)
    average_hr_requested = json_data.get('statistics_average_hr', False)
    
    if is_tail_only:
        # Tail mode: hide current speed, show average speed if either speed stat is requested
        if current_speed_requested or average_speed_requested:
            if accumulated_time > 0:
                imperial_units = _is_imperial_units(json_data)
                if imperial_units:
                    # accumulated_distance is in miles, calculate mph
                    avg_speed = (accumulated_distance * 3600) / accumulated_time
                else:
                    # accumulated_distance is in meters, calculate km/h
                    avg_speed = (accumulated_distance / 1000.0 * 3600) / accumulated_time
                statistics_data['average_speed'] = f"{avg_speed:.1f}"
            else:
                statistics_data['average_speed'] = "0.0"
        
        # Tail mode: also show average heart rate if requested
        if average_hr_requested:
            all_hr_values = []
            if is_multiple_routes:
                for route_points in points_for_frame:
                    if not route_points:
                        continue
                    for point in route_points:
                        if point.heart_rate and point.heart_rate > 0:
                            all_hr_values.append(point.heart_rate)
            else:
                for point in points_for_frame:
                    if point.heart_rate and point.heart_rate > 0:
                        all_hr_values.append(point.heart_rate)
            
            if all_hr_values:
                avg_hr = sum(all_hr_values) / len(all_hr_values)
                statistics_data['average_hr'] = f"{round(avg_hr)}"
            else:
                statistics_data['average_hr'] = None
    else:
        # Normal mode: show both speeds as requested
        
        # Average speed calculation
        if average_speed_requested:
            if accumulated_time > 0:
                imperial_units = _is_imperial_units(json_data)
                if imperial_units:
                    # accumulated_distance is in miles, calculate mph
                    avg_speed = (accumulated_distance * 3600) / accumulated_time
                else:
                    # accumulated_distance is in meters, calculate km/h
                    avg_speed = (accumulated_distance / 1000.0 * 3600) / accumulated_time
                statistics_data['average_speed'] = f"{avg_speed:.1f}"
            else:
                statistics_data['average_speed'] = "0.0"
        
        # Current speed - use pre-calculated smoothed value from route creation
        if current_speed_requested:
            # Use pre-calculated smoothed speed (calculated during route creation with 0.3 video-second window)
            if last_point.current_speed_smoothed is not None:
                statistics_data['current_speed'] = f"{last_point.current_speed_smoothed}"
            else:
                # Fallback: calculate if pre-calculated value not available (shouldn't happen normally)
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
                    route_start_time = route_points[0].accumulated_time
                    route_end_time = route_points[-1].accumulated_time
                    
                    # If target time is within this route's range, search it
                    if target_time >= route_start_time and target_time <= route_end_time:
                        # Binary search would be even faster, but reverse iteration is good enough
                        for point in reversed(route_points):
                            if point.accumulated_time <= target_time:
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
                    if point.accumulated_time <= target_time:
                        comparison_point = point
                        break
            
            # Get current elevation from last point
            current_elevation = last_point.elevation if last_point else None
            
            # Get comparison elevation from comparison point
            comparison_elevation = comparison_point.elevation if comparison_point else None
            
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
        
        # Current heart rate - use pre-calculated smoothed value from route creation
        if current_hr_requested:
            # Use pre-calculated smoothed heart rate (calculated during route creation with 0.3 video-second window)
            if last_point.heart_rate_smoothed is not None:
                statistics_data['current_hr'] = f"{last_point.heart_rate_smoothed}"
            else:
                # No HR data available for this point
                statistics_data['current_hr'] = None
        
        # Average heart rate calculation - average of all HR values up to current point
        if average_hr_requested:
            all_hr_values = []
            if is_multiple_routes:
                for route_points in points_for_frame:
                    if not route_points:
                        continue
                    for point in route_points:
                        if point.heart_rate and point.heart_rate > 0:
                            all_hr_values.append(point.heart_rate)
            else:
                for point in points_for_frame:
                    if point.heart_rate and point.heart_rate > 0:
                        all_hr_values.append(point.heart_rate)
            
            if all_hr_values:
                avg_hr = sum(all_hr_values) / len(all_hr_values)
                statistics_data['average_hr'] = f"{round(avg_hr)}"
            else:
                statistics_data['average_hr'] = None
    
    return statistics_data


def _draw_current_speed_at_point(ax, points_for_frame, current_speed, effective_line_width, theme='light', json_data=None, resolution_scale=None, vertical_position=0):
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
        vertical_position (int): Vertical stack position (0=top, 1=second, 2=third, etc.)
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
            point_time = route_last_point.accumulated_time
            if latest_time is None or point_time > latest_time:
                latest_time = point_time
                latest_point = route_last_point
    else:
        # Single route mode
        if points_for_frame:
            latest_point = points_for_frame[-1]
    
    if not latest_point:
        return
    
    # Extract coordinates from the latest point using named attributes
    lat = latest_point.lat
    lon = latest_point.lon
    
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
        if json_data:
            from image_generator_utils import calculate_resolution_scale
            resolution_x = int(json_data.get('video_resolution_x', 1920))
            resolution_y = int(json_data.get('video_resolution_y', 1080))
            resolution_scale = calculate_resolution_scale(resolution_x, resolution_y)
        else:
            resolution_scale = 1.0
    
    # Calculate resolution-scaled offset (15px baseline for 1080p)
    base_offset_pixels = 15  # Baseline for 1080p
    scaled_offset_pixels = base_offset_pixels * resolution_scale
    
    # Estimate pixel-to-coordinate conversion (rough approximation)
    # This could be more precise, but for positioning it should be adequate
    width = int(ax.figure.get_figwidth() * ax.figure.dpi)
    x_offset = (scaled_offset_pixels / width) * x_range  # Resolution-scaled pixels in coordinate space
    
    speed_x = x + x_offset
    
    # Calculate vertical offset based on position in the stack
    height = int(ax.figure.get_figheight() * ax.figure.dpi)
    base_vertical_offset_pixels = 25  # Baseline spacing for 1080p
    scaled_vertical_offset_pixels = base_vertical_offset_pixels * resolution_scale * vertical_position
    y_offset = (scaled_vertical_offset_pixels / height) * y_range
    speed_y = y - y_offset
    
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
    
    # Determine speed unit based on imperial_units setting
    imperial_units = _is_imperial_units(json_data)
    speed_unit = "mph" if imperial_units else "km/h"
    
    # Add the current speed text
    ax.text(
        speed_x, speed_y, f"{current_speed} {speed_unit}",
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


def _draw_current_elevation_at_point(ax, points_for_frame, current_elevation, effective_line_width, theme='light', json_data=None, resolution_scale=None, vertical_position=0):
    """
    Draw current elevation near the latest point on the chart.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        points_for_frame (list): List of route points (single route) or list of sub-lists (multiple routes)
        current_elevation (str): Current elevation value to display
        effective_line_width (float): Base line width for scaling (unused, kept for compatibility)
        theme (str): Theme for elevation display ('light' or 'dark')
        json_data (dict): Job data containing video parameters for resolution scaling
        resolution_scale (float, optional): Pre-calculated resolution scale factor for optimization
        vertical_position (int): Vertical stack position (0=top, 1=second, 2=third, etc.)
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
            point_time = route_last_point.accumulated_time
            if latest_time is None or point_time > latest_time:
                latest_time = point_time
                latest_point = route_last_point
    else:
        # Single route mode
        if points_for_frame:
            latest_point = points_for_frame[-1]
    
    if not latest_point:
        return
    
    # Extract coordinates from the latest point using named attributes
    lat = latest_point.lat
    lon = latest_point.lon
    
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
        if json_data:
            from image_generator_utils import calculate_resolution_scale
            resolution_x = int(json_data.get('video_resolution_x', 1920))
            resolution_y = int(json_data.get('video_resolution_y', 1080))
            resolution_scale = calculate_resolution_scale(resolution_x, resolution_y)
        else:
            resolution_scale = 1.0
    
    # Calculate resolution-scaled offset (15px baseline for 1080p)
    base_offset_pixels = 15  # Baseline for 1080p
    scaled_offset_pixels = base_offset_pixels * resolution_scale
    
    # Estimate pixel-to-coordinate conversion (rough approximation)
    # This could be more precise, but for positioning it should be adequate
    width = int(ax.figure.get_figwidth() * ax.figure.dpi)
    x_offset = (scaled_offset_pixels / width) * x_range  # Resolution-scaled pixels in coordinate space
    
    elevation_x = x + x_offset
    
    # Calculate vertical offset based on position in the stack
    height = int(ax.figure.get_figheight() * ax.figure.dpi)
    base_vertical_offset_pixels = 25  # Baseline spacing for 1080p
    scaled_vertical_offset_pixels = base_vertical_offset_pixels * resolution_scale * vertical_position
    y_offset = (scaled_vertical_offset_pixels / height) * y_range
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
        elevation_x, elevation_y, f"{current_elevation} m ↑",
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


def _draw_current_hr_at_point(ax, points_for_frame, current_hr, effective_line_width, theme='light', json_data=None, resolution_scale=None, vertical_position=0):
    """
    Draw current heart rate near the latest point on the chart.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        points_for_frame (list): List of route points (single route) or list of sub-lists (multiple routes)
        current_hr (str): Current heart rate value to display
        effective_line_width (float): Base line width for scaling (unused, kept for compatibility)
        theme (str): Theme for HR display ('light' or 'dark')
        json_data (dict): Job data containing video parameters for resolution scaling
        resolution_scale (float, optional): Pre-calculated resolution scale factor for optimization
        vertical_position (int): Vertical stack position (0=top, 1=second, 2=third, etc.)
    """
    if not current_hr or not points_for_frame:
        return
    
    # Find the latest point across all routes to position the HR label
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
            point_time = route_last_point.accumulated_time
            if latest_time is None or point_time > latest_time:
                latest_time = point_time
                latest_point = route_last_point
    else:
        # Single route mode
        if points_for_frame:
            latest_point = points_for_frame[-1]
    
    if not latest_point:
        return
    
    # Extract coordinates from the latest point using named attributes
    lat = latest_point.lat
    lon = latest_point.lon
    
    # Convert GPS coordinates to Web Mercator for proper positioning
    def _gps_to_web_mercator(lon, lat):
        """Convert GPS coordinates to Web Mercator projection."""
        import math
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34 / 180
        return x, y
    
    x, y = _gps_to_web_mercator(lon, lat)
    
    # Calculate offset for HR label position (scales with resolution)
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    # Use pre-calculated resolution scale if provided, otherwise calculate it
    if resolution_scale is None:
        if json_data:
            from image_generator_utils import calculate_resolution_scale
            resolution_x = int(json_data.get('video_resolution_x', 1920))
            resolution_y = int(json_data.get('video_resolution_y', 1080))
            resolution_scale = calculate_resolution_scale(resolution_x, resolution_y)
        else:
            resolution_scale = 1.0
    
    # Calculate resolution-scaled offset (15px baseline for 1080p)
    base_offset_pixels = 15  # Baseline for 1080p
    scaled_offset_pixels = base_offset_pixels * resolution_scale
    
    width = int(ax.figure.get_figwidth() * ax.figure.dpi)
    x_offset = (scaled_offset_pixels / width) * x_range
    
    hr_x = x + x_offset
    
    # Calculate vertical offset based on position in the stack
    height = int(ax.figure.get_figheight() * ax.figure.dpi)
    base_vertical_offset_pixels = 25  # Baseline spacing for 1080p
    scaled_vertical_offset_pixels = base_vertical_offset_pixels * resolution_scale * vertical_position
    y_offset = (scaled_vertical_offset_pixels / height) * y_range
    hr_y = y - y_offset
    
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
    
    # Add the current heart rate text with heart symbol
    # First draw the full text (number + heart) with bbox for proper sizing
    number_text = f"{current_hr}"
    heart_symbol = " ♥"
    full_text = number_text + heart_symbol
    text_obj = ax.text(
        hr_x, hr_y, full_text,
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
    
    # Get the bounding box of the number text to position the heart symbol overlay
    # We need to get the renderer to measure text
    fig = ax.figure
    if fig.canvas is not None:
        renderer = fig.canvas.get_renderer()
    else:
        # Fallback: create a dummy renderer for measurement
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        canvas = FigureCanvasAgg(fig)
        renderer = canvas.get_renderer()
    
    # Measure just the number text to find where heart symbol starts
    temp_text = ax.text(hr_x, hr_y, number_text, fontsize=font_size, fontweight='bold', 
                       ha='left', va='center', visible=False)
    bbox_number = temp_text.get_window_extent(renderer=renderer)
    temp_text.remove()
    
    # Convert bbox to data coordinates to get the right edge
    transform = ax.transData.inverted()
    bbox_data = bbox_number.transformed(transform)
    heart_x = bbox_data.x1  # Right edge of the number text
    
    # Draw the heart symbol in red, positioned right after the number (overlay)
    ax.text(
        heart_x, hr_y, heart_symbol,
        color='red',  # Red color for heart symbol
        fontsize=font_size,
        fontweight='bold',
        ha='left',
        va='center',
        zorder=101  # Slightly above the number text to overlay
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
    
    # Determine units based on imperial_units setting
    imperial_units = _is_imperial_units(json_data)
    distance_unit = "miles" if imperial_units else "km"
    speed_unit = "mph" if imperial_units else "km/h"
    
    if 'distance' in statistics_data:
        stats_lines.append(f"{statistics_data['distance']} {distance_unit}")
    
    # Include current speed only if not excluded and not being displayed at point
    current_speed_at_point = json_data.get('statistics_current_speed', False)
    if not exclude_current_speed and 'current_speed' in statistics_data and not current_speed_at_point:
        stats_lines.append(f"{statistics_data['current_speed']} {speed_unit}")
    
    if 'average_speed' in statistics_data:
        stats_lines.append(f"{statistics_data['average_speed']} {speed_unit}")
    
    # Include current elevation only if not excluded and not being displayed at point
    current_elevation_at_point = json_data.get('statistics_current_elevation', False)
    if not exclude_current_elevation and 'current_elevation' in statistics_data and not current_elevation_at_point:
        stats_lines.append(f"{statistics_data['current_elevation']} m ↑")
    
    # Include average heart rate if available (current HR is displayed at point, not in corner)
    # Store heart rate separately so we can draw the heart symbol in red
    avg_hr_value = None
    if 'average_hr' in statistics_data and statistics_data['average_hr']:
        avg_hr_value = statistics_data['average_hr']
        stats_lines.append(f"{avg_hr_value} ♥")
    
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
        if json_data:
            from image_generator_utils import calculate_resolution_scale
            resolution_x = int(json_data.get('video_resolution_x', 1920))
            resolution_y = int(json_data.get('video_resolution_y', 1080))
            resolution_scale = calculate_resolution_scale(resolution_x, resolution_y)
        else:
            resolution_scale = 1.0
    
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
    text_obj = ax.text(
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
    
    # If average HR is displayed, overlay the heart symbol in red
    if avg_hr_value is not None:
        # Find which line contains the heart rate (count from bottom)
        hr_line_index = None
        for i, line in enumerate(stats_lines):
            if '♥' in line or avg_hr_value in line:
                hr_line_index = len(stats_lines) - 1 - i  # Line index from top (0 = top line)
                break
        
        if hr_line_index is not None:
            # Calculate the y position of the heart rate line
            # Each line has a height, we need to account for line spacing
            fig = ax.figure
            if fig.canvas is not None:
                renderer = fig.canvas.get_renderer()
            else:
                # Fallback: create a dummy renderer for measurement
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                canvas = FigureCanvasAgg(fig)
                renderer = canvas.get_renderer()
            
            # Measure single line height
            temp_text = ax.text(0, 0, "Test", fontsize=font_size, fontweight='bold', visible=False)
            line_bbox = temp_text.get_window_extent(renderer=renderer)
            temp_text.remove()
            
            # Convert to axes coordinates
            transform = ax.transAxes.inverted()
            line_height_axes = line_bbox.transformed(transform).height
            
            # Calculate y position of the heart rate line (from top)
            hr_line_y = text_y - (hr_line_index * line_height_axes)
            
            # Measure the number text to find where heart symbol should go
            number_text = f"{avg_hr_value}"
            temp_text = ax.text(text_x, hr_line_y, number_text, transform=ax.transAxes,
                               fontsize=font_size, fontweight='bold', ha='right', va='top', visible=False)
            number_bbox = temp_text.get_window_extent(renderer=renderer)
            temp_text.remove()
            
            # Convert to axes coordinates to get the left edge of number (since ha='right')
            number_bbox_axes = number_bbox.transformed(transform)
            heart_x = number_bbox_axes.x0  # Left edge of number (right-aligned)
            
            # Draw the heart symbol in red, positioned right after the number
            ax.text(
                heart_x, hr_line_y, " ♥",
                transform=ax.transAxes,
                color='red',  # Red color for heart symbol
                fontsize=font_size,
                fontweight='bold',
                ha='left',  # Left align from the number's left edge
                va='top',
                zorder=101  # Slightly above the statistics text
            ) 