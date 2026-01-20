"""
Utility functions for video generation single frame creation.
This module contains helper functions for color manipulation, name tags, and resolution scaling.
"""

# Local imports
from image_generator_utils import draw_tag
from video_generator_create_combined_route import RoutePoint


def hex_to_rgba(hex_color):
    """Convert hex color to RGBA tuple"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)) + (1.0,)
    return (1.0, 0.0, 0.0, 1.0)  # Default red


def rgba_to_hex(rgba_color):
    """Convert RGBA tuple to hex color string"""
    r, g, b, a = rgba_color
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def lighten_color(rgba_color, factor=0.3):
    """
    Create a lighter version of an RGBA color.
    
    Args:
        rgba_color (tuple): RGBA color tuple (0-1 range)
        factor (float): Lightening factor (0-1, higher = lighter)
    
    Returns:
        tuple: Lightened RGBA color
    """
    r, g, b, a = rgba_color
    
    # Convert to HSL for better lightening
    # Simple approach: move towards white
    r = min(1.0, r + (1.0 - r) * factor)
    g = min(1.0, g + (1.0 - g) * factor)
    b = min(1.0, b + (1.0 - b) * factor)
    
    return (r, g, b, a)


def darken_color(rgba_color, factor=0.3):
    """
    Create a darker version of an RGBA color.
    
    Args:
        rgba_color (tuple): RGBA color tuple (0-1 range)
        factor (float): Darkening factor (0-1, higher = darker)
    
    Returns:
        tuple: Darkened RGBA color
    """
    r, g, b, a = rgba_color
    
    # Simple approach: move towards black
    r = max(0.0, r * (1.0 - factor))
    g = max(0.0, g * (1.0 - factor))
    b = max(0.0, b * (1.0 - factor))
    
    return (r, g, b, a)


def get_tail_color_for_route(tail_color_setting, route_color_rgba):
    """
    Get the appropriate tail color for a route based on the tail_color setting.
    
    Args:
        tail_color_setting (str): Tail color setting ('light', 'dark', or hex color)
        route_color_rgba (tuple): RGBA color of the route
    
    Returns:
        tuple: RGBA color for the tail
    """
    if tail_color_setting == 'light':
        return lighten_color(route_color_rgba, factor=0.4)
    elif tail_color_setting == 'dark':
        return darken_color(route_color_rgba, factor=0.4)
    else:
        # Assume it's a hex color
        return hex_to_rgba(tail_color_setting)


def _gps_to_web_mercator(lon, lat):
    """Convert GPS (lon, lat) to Web Mercator (x, y) meters."""
    import math
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y


def _draw_name_tag(ax, point, runner_name, filename_to_rgba, resolution_scale, theme='light', vertical_offset_points=0):
    """
    Draw a name tag to the right of a point.
    
    This function is a wrapper around the reusable draw_tag function from image_generator_utils.
    It extracts the necessary information from the RoutePoint and calls draw_tag.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        point (RoutePoint): RoutePoint named tuple with route data
        runner_name (str): Name of the runner to display
        filename_to_rgba (dict): Filename to RGBA color mapping
        resolution_scale (float): Resolution scale factor (0.7, 1.0, 2.0, 3.0, or 4.0)
        theme (str): Theme for the name tag ('light' or 'dark')
        vertical_offset_points (float): Vertical offset in points (positive = up)
    """
    # Extract coordinates using named attributes
    lat = point.lat
    lon = point.lon
    filename = point.filename
    
    # Get color for this route (RGBA tuple, 0-1 range)
    if filename and filename_to_rgba and filename in filename_to_rgba:
        rgba_color = filename_to_rgba[filename]
    else:
        rgba_color = (1.0, 0.0, 0.0, 1.0)  # Default red
    
    # Extract RGB from RGBA (drop alpha channel)
    text_color_rgb = rgba_color[:3]
    
    # Use the reusable draw_tag function (font size and offsets are scaled internally)
    draw_tag(
        ax=ax,
        lon=lon,
        lat=lat,
        text=runner_name,
        text_color_rgb=text_color_rgb,
        background_theme=theme,
        resolution_scale=resolution_scale,
        vertical_offset_points=vertical_offset_points
    )


def _draw_name_tags_for_routes(points_for_frame, json_data, filename_to_rgba, resolution_scale, ax):
    """
    Draw name tags for the most recent point of each route.
    
    Args:
        points_for_frame (list): List of route points (single route) or list of sub-lists (multiple routes)
        json_data (dict): Job data containing track_objects
        filename_to_rgba (dict): Filename to RGBA color mapping
        resolution_scale (float): Resolution scale factor (0.7, 1.0, 2.0, 3.0, or 4.0)
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
    """   
    name_tags_setting = json_data.get('name_tags', 'off')
    if name_tags_setting == 'off':
        return  # Don't draw name tags when setting is 'off'
    
    if name_tags_setting not in ['light', 'dark']:
        print(f"Warning: Invalid name_tags setting '{name_tags_setting}', skipping")
        return
    
    # Get track_objects for name mapping
    track_objects = json_data.get('track_objects', [])
    filename_to_name = {}
    for track_obj in track_objects:
        filename = track_obj.get('filename', '')
        name = track_obj.get('name', '')
        if filename and name:
            # Remove .gpx extension if present to match the filename format in points
            if filename.endswith('.gpx'):
                filename = filename[:-4]
            filename_to_name[filename] = name
    
    # Determine if we have multiple routes or single route
    is_multiple_routes = False
    if points_for_frame and isinstance(points_for_frame[0], list):
        is_multiple_routes = True
    
    if is_multiple_routes:
        # Multiple routes mode - draw name tag for the most recent point of each route
        for route_points in points_for_frame:
            if not route_points:  # Skip empty routes
                continue
            
            # Get the most recent point (last point in the route)
            most_recent_point = route_points[-1]
            filename = most_recent_point.filename
            
            if filename and filename in filename_to_name:
                runner_name = filename_to_name[filename]
                _draw_name_tag(ax, most_recent_point, runner_name, filename_to_rgba, resolution_scale, name_tags_setting)
    else:
        # Single route mode - draw name tag for the most recent point
        if not points_for_frame:  # Skip if no points
            return
        
        # Get the most recent point (last point in the route)
        most_recent_point = points_for_frame[-1]
        filename = most_recent_point.filename
        
        if filename and filename in filename_to_name:
            runner_name = filename_to_name[filename]
            _draw_name_tag(ax, most_recent_point, runner_name, filename_to_rgba, resolution_scale, name_tags_setting)


def _draw_filename_tags_for_routes(points_for_frame, json_data, filename_to_rgba, resolution_scale, ax):
    """
    Draw filename tags for the most recent point of each route.
    
    Similar to _draw_name_tags_for_routes but uses the filename (without .gpx extension)
    instead of the name from track_objects. If realtime statistics are enabled, the tag
    is drawn higher to avoid overlap.
    
    Args:
        points_for_frame (list): List of route points (single route) or list of sub-lists (multiple routes)
        json_data (dict): Job data containing settings
        filename_to_rgba (dict): Filename to RGBA color mapping
        resolution_scale (float): Resolution scale factor (0.7, 1.0, 2.0, 3.0, or 4.0)
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
    """   
    filename_tags_setting = json_data.get('filename_tags', 'off')
    if filename_tags_setting == 'off' or filename_tags_setting is None:
        return  # Don't draw filename tags when setting is 'off' or not set
    
    if filename_tags_setting not in ['light', 'dark']:
        print(f"Warning: Invalid filename_tags setting '{filename_tags_setting}', skipping")
        return
    
    # Check if realtime statistics are enabled - if so, we need to offset the tag vertically
    # Count how many realtime stats are enabled to determine offset
    realtime_stats_count = 0
    if json_data.get('statistics_current_speed', False):
        realtime_stats_count += 1
    if json_data.get('statistics_current_elevation', False):
        realtime_stats_count += 1
    if json_data.get('statistics_current_hr', False):
        realtime_stats_count += 1
    
    # Calculate vertical offset: move up by ~25 points per realtime stat enabled (scaled by resolution)
    # This matches the base_vertical_offset_pixels used in _draw_current_*_at_point functions
    vertical_offset = realtime_stats_count * 25 * resolution_scale if realtime_stats_count > 0 else 0
    
    # Determine if we have multiple routes or single route
    is_multiple_routes = False
    if points_for_frame and isinstance(points_for_frame[0], list):
        is_multiple_routes = True
    
    if is_multiple_routes:
        # Multiple routes mode - draw filename tag for the most recent point of each route
        for route_points in points_for_frame:
            if not route_points:  # Skip empty routes
                continue
            
            # Get the most recent point (last point in the route)
            most_recent_point = route_points[-1]
            filename = most_recent_point.filename
            
            if filename:
                # Use filename as the display text (already without .gpx extension in RoutePoint)
                _draw_name_tag(ax, most_recent_point, filename, filename_to_rgba, resolution_scale, filename_tags_setting, vertical_offset)
    else:
        # Single route mode - draw filename tag for the most recent point
        if not points_for_frame:  # Skip if no points
            return
        
        # Get the most recent point (last point in the route)
        most_recent_point = points_for_frame[-1]
        filename = most_recent_point.filename
        
        if filename:
            # Use filename as the display text (already without .gpx extension in RoutePoint)
            _draw_name_tag(ax, most_recent_point, filename, filename_to_rgba, resolution_scale, filename_tags_setting, vertical_offset)