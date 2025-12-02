"""
Utility functions for video generation single frame creation.
This module contains helper functions for color manipulation, name tags, and resolution scaling.
"""

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


def _gps_to_web_mercator(lon, lat):
    """Convert GPS (lon, lat) to Web Mercator (x, y) meters."""
    import math
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y


def _draw_name_tag(ax, point, runner_name, filename_to_rgba, effective_line_width, theme='light'):
    """
    Draw a name tag to the right of a point.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes for drawing
        point (RoutePoint): RoutePoint named tuple with route data
        runner_name (str): Name of the runner to display
        filename_to_rgba (dict): Filename to RGBA color mapping
        effective_line_width (float): Base line width for scaling
        theme (str): Theme for the name tag ('light' or 'dark')
    """
    # Extract coordinates using named attributes
    lat = point.lat
    lon = point.lon
    filename = point.filename
    
    # Get color for this route
    if filename and filename_to_rgba and filename in filename_to_rgba:
        rgba_color = filename_to_rgba[filename]
    else:
        rgba_color = (1.0, 0.0, 0.0, 1.0)  # Default red
    
    # Convert RGBA to hex for text color
    hex_color = rgba_to_hex(rgba_color)
    
    # Convert GPS to Web Mercator coordinates for plotting
    x, y = _gps_to_web_mercator(lon, lat)
    
    # Set theme colors
    if theme == 'dark':
        bg_color = '#2d2d2d'  # Dark gray background
        border_color = '#cccccc'  # Light gray border
        text_color = '#ffffff'  # White text
    else:  # light theme (default)
        bg_color = 'white'
        border_color = hex_color  # Use route color for border
        text_color = hex_color  # Use route color for text
    
    # Use annotate with pixel-offset to ensure consistent on-screen spacing regardless of projection
    horizontal_offset_points = max(6, effective_line_width * 2)  # offset to the right
    ax.annotate(
        runner_name,
        xy=(x, y),
        xycoords='data',
        xytext=(horizontal_offset_points, 0),
        textcoords='offset points',
        color=text_color,
        fontsize=max(8, effective_line_width * 2),
        fontweight='bold',
        ha='left',
        va='center',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor=bg_color,
            edgecolor=border_color,
            alpha=0.9,
            linewidth=1
        ),
        zorder=40,
    )


def _draw_name_tags_for_routes(points_for_frame, json_data, filename_to_rgba, effective_line_width, ax):
    """
    Draw name tags for the most recent point of each route.
    
    Args:
        points_for_frame (list): List of route points (single route) or list of sub-lists (multiple routes)
        json_data (dict): Job data containing track_objects
        filename_to_rgba (dict): Filename to RGBA color mapping
        effective_line_width (float): Base line width for scaling
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
                _draw_name_tag(ax, most_recent_point, runner_name, filename_to_rgba, effective_line_width, name_tags_setting)
    else:
        # Single route mode - draw name tag for the most recent point
        if not points_for_frame:  # Skip if no points
            return
        
        # Get the most recent point (last point in the route)
        most_recent_point = points_for_frame[-1]
        filename = most_recent_point.filename
        
        if filename and filename in filename_to_name:
            runner_name = filename_to_name[filename]
            _draw_name_tag(ax, most_recent_point, runner_name, filename_to_rgba, effective_line_width, name_tags_setting) 