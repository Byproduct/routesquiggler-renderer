# Standard library imports
import math
import os

# Third-party imports
from PIL import Image, ImageDraw, ImageFont


def _draw_text_with_outline(draw, position, text, font, fill_color, outline_color=(0, 0, 0, 255), outline_width=1):
    """
    Draw text with a thin outline for better legibility on any background.
    
    Args:
        draw: ImageDraw object
        position: (x, y) tuple for text position
        text: Text string to draw
        font: Font to use
        fill_color: RGBA tuple for the main text color
        outline_color: RGBA tuple for the outline color (default: black)
        outline_width: Width of the outline in pixels (default: 1)
    """
    x, y = position
    # Draw outline by drawing text in outline_color at offset positions
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
    # Draw main text on top
    draw.text((x, y), text, fill=fill_color, font=font)


def _draw_text_with_red_heart(draw, position, text, font, fill_color, outline_color=(0, 0, 0, 255), outline_width=1):
    """
    Draw text with a red heart symbol. The text should end with " ♥".
    The number part is drawn in fill_color, and the heart symbol is drawn in red.
    
    Args:
        draw: ImageDraw object
        position: (x, y) tuple for text position
        text: Text string ending with " ♥" (e.g., "136 ♥")
        font: Font to use
        fill_color: RGBA tuple for the main text color (for the number)
        outline_color: RGBA tuple for the outline color (default: black)
        outline_width: Width of the outline in pixels (default: 1)
    """
    x, y = position
    
    # Split text into number and heart symbol
    if text.endswith(" ♥"):
        number_text = text[:-2]  # Everything except " ♥"
        heart_symbol = " ♥"
    else:
        # If no heart symbol, just draw normally
        _draw_text_with_outline(draw, position, text, font, fill_color, outline_color, outline_width)
        return
    
    # Measure the number text width
    number_bbox = draw.textbbox((0, 0), number_text, font=font)
    number_width = number_bbox[2] - number_bbox[0]
    
    # Red color for heart symbol (RGBA)
    heart_color = (255, 0, 0, 255)  # Red
    
    # Draw number with outline
    _draw_text_with_outline(draw, (x, y), number_text, font, fill_color, outline_color, outline_width)
    
    # Draw heart symbol with outline in red
    heart_x = x + number_width
    # Draw outline for heart symbol
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((heart_x + dx, y + dy), heart_symbol, fill=outline_color, font=font)
    # Draw heart symbol in red
    draw.text((heart_x, y), heart_symbol, fill=heart_color, font=font)


def update_speed_based_color_range(all_routes, json_data, debug_callback=None):
    """
    Update speed_based_color_min and speed_based_color_max in json_data based on actual data.
    
    If either value is -1, it will be calculated from the minimum/maximum current_speed_smoothed
    values across all routes. If values are already set (not -1), they are left unchanged.
    
    Args:
        all_routes (list): List of route data dictionaries
        json_data (dict): Job data dictionary (will be modified in place)
        debug_callback (callable, optional): Function for debug logging
    
    Returns:
        None (modifies json_data in place)
    """
    if not json_data:
        return
    
    speed_min_setting = json_data.get('speed_based_color_min', -1)
    speed_max_setting = json_data.get('speed_based_color_max', -1)
    
    # Check if we need to calculate either value
    need_min = speed_min_setting == -1
    need_max = speed_max_setting == -1
    
    if not need_min and not need_max:
        if debug_callback:
            debug_callback("Speed-based color range already set, skipping auto-calculation")
        return
    
    # Collect all current_speed_smoothed values from all routes
    all_speeds = []
    for route in all_routes:
        route_points = route.get('combined_route', [])
        for point in route_points:
            if point.current_speed_smoothed is not None:
                all_speeds.append(point.current_speed_smoothed)
    
    if not all_speeds:
        if debug_callback:
            debug_callback("No smoothed speed data available for color range calculation")
        return
    
    # Calculate min/max from actual data
    calculated_min = min(all_speeds)
    calculated_max = max(all_speeds)
    
    # Update json_data if needed
    # Note: speed_based_color_min/max are already in the correct units (mph if imperial_units, km/h otherwise)
    # and should NOT be converted
    imperial_units = json_data and json_data.get('imperial_units', False) is True
    speed_unit = "mph" if imperial_units else "km/h"
    
    if need_min:
        json_data['speed_based_color_min'] = calculated_min
        if debug_callback:
            debug_callback(f"Auto-calculated speed_based_color_min: {calculated_min} {speed_unit}")
    
    if need_max:
        json_data['speed_based_color_max'] = calculated_max
        if debug_callback:
            debug_callback(f"Auto-calculated speed_based_color_max: {calculated_max} {speed_unit}")
    
    if debug_callback and (need_min or need_max):
        debug_callback(f"Speed-based color range: {json_data.get('speed_based_color_min')} - {json_data.get('speed_based_color_max')} {speed_unit}")


def _collect_hr_values_with_255_fix(all_routes, debug_callback=None):
    """
    Collect heart_rate_smoothed values from all routes, fixing Garmin 255 error values.
    
    If max HR is 255 (a Garmin error value), sweep through and replace each 255 with
    the previous non-255 value (or 100 as fallback).
    
    Args:
        all_routes (list): List of route data dictionaries
        debug_callback (callable, optional): Function for debug logging
    
    Returns:
        tuple: (calculated_min, calculated_max) or (None, None) if no valid data
    """
    # First pass: collect all HR values
    all_hrs = []
    for route in all_routes:
        route_points = route.get('combined_route', [])
        for point in route_points:
            if point.heart_rate_smoothed is not None:
                all_hrs.append(point.heart_rate_smoothed)
    
    if not all_hrs:
        return None, None
    
    # Calculate initial min/max
    calculated_min = min(all_hrs)
    calculated_max = max(all_hrs)
    
    # Check if max is 255 (Garmin error value)
    if calculated_max == 255:
        if debug_callback:
            debug_callback("Detected HR value 255 (Garmin error value), cleaning data")
        
        # Second pass: collect HR values with 255 replacement
        cleaned_hrs = []
        last_valid_hr = 100  # Fallback value if no previous valid HR found
        
        for route in all_routes:
            route_points = route.get('combined_route', [])
            for point in route_points:
                if point.heart_rate_smoothed is not None:
                    hr_value = point.heart_rate_smoothed
                    if hr_value == 255:
                        # Replace 255 with the last valid HR value
                        cleaned_hrs.append(last_valid_hr)
                    else:
                        cleaned_hrs.append(hr_value)
                        last_valid_hr = hr_value  # Update last valid for future 255s
        
        if cleaned_hrs:
            calculated_min = min(cleaned_hrs)
            calculated_max = max(cleaned_hrs)
            if debug_callback:
                debug_callback(f"Cleaned HR range: {calculated_min} - {calculated_max} ♥ (replaced 255 error values)")
    
    return calculated_min, calculated_max


def update_hr_based_color_range(all_routes, json_data, debug_callback=None):
    """
    Update hr_based_color_min and hr_based_color_max in json_data based on actual data.
    
    If either value is -1, it will be calculated from the minimum/maximum heart_rate_smoothed
    values across all routes. If values are already set (not -1), they are left unchanged.
    Also handles Garmin 255 error values by replacing them with the previous valid HR.
    
    Args:
        all_routes (list): List of route data dictionaries
        json_data (dict): Job data dictionary (will be modified in place)
        debug_callback (callable, optional): Function for debug logging
    
    Returns:
        None (modifies json_data in place)
    """
    if not json_data:
        return
    
    hr_min_setting = json_data.get('hr_based_color_min', -1)
    hr_max_setting = json_data.get('hr_based_color_max', -1)
    
    # Check if we need to calculate either value
    need_min = hr_min_setting == -1
    need_max = hr_max_setting == -1
    
    if not need_min and not need_max:
        if debug_callback:
            debug_callback("HR-based color range already set, skipping auto-calculation")
        return
    
    # Collect HR values with 255 error value fix
    calculated_min, calculated_max = _collect_hr_values_with_255_fix(all_routes, debug_callback)
    
    if calculated_min is None or calculated_max is None:
        if debug_callback:
            debug_callback("No smoothed heart rate data available for color range calculation")
        return
    
    # Update json_data if needed
    if need_min:
        json_data['hr_based_color_min'] = calculated_min
        if debug_callback:
            debug_callback(f"Auto-calculated hr_based_color_min: {calculated_min} ♥")
    
    if need_max:
        json_data['hr_based_color_max'] = calculated_max
        if debug_callback:
            debug_callback(f"Auto-calculated hr_based_color_max: {calculated_max} ♥")
    
    if debug_callback and (need_min or need_max):
        debug_callback(f"HR-based color range: {json_data.get('hr_based_color_min')} - {json_data.get('hr_based_color_max')} ♥")


def update_hr_based_width_range(all_routes, json_data, debug_callback=None):
    """
    Update hr_based_width_min and hr_based_width_max in json_data based on actual data.
    
    If either value is -1, it will be calculated from the minimum/maximum heart_rate_smoothed
    values across all routes. If values are already set (not -1), they are left unchanged.
    Also handles Garmin 255 error values by replacing them with the previous valid HR.
    
    Args:
        all_routes (list): List of route data dictionaries
        json_data (dict): Job data dictionary (will be modified in place)
        debug_callback (callable, optional): Function for debug logging
    
    Returns:
        None (modifies json_data in place)
    """
    if not json_data:
        return
    
    hr_min_setting = json_data.get('hr_based_width_min', -1)
    hr_max_setting = json_data.get('hr_based_width_max', -1)
    
    # Check if we need to calculate either value
    need_min = hr_min_setting == -1
    need_max = hr_max_setting == -1
    
    if not need_min and not need_max:
        if debug_callback:
            debug_callback("HR-based width range already set, skipping auto-calculation")
        return
    
    # Collect HR values with 255 error value fix
    calculated_min, calculated_max = _collect_hr_values_with_255_fix(all_routes, debug_callback)
    
    if calculated_min is None or calculated_max is None:
        if debug_callback:
            debug_callback("No smoothed heart rate data available for width range calculation")
        return
    
    # Update json_data if needed
    if need_min:
        json_data['hr_based_width_min'] = calculated_min
        if debug_callback:
            debug_callback(f"Auto-calculated hr_based_width_min: {calculated_min} ♥")
    
    if need_max:
        json_data['hr_based_width_max'] = calculated_max
        if debug_callback:
            debug_callback(f"Auto-calculated hr_based_width_max: {calculated_max} ♥")
    
    if debug_callback and (need_min or need_max):
        debug_callback(f"HR-based width range: {json_data.get('hr_based_width_min')} - {json_data.get('hr_based_width_max')} ♥")


def speed_based_color(value: float):
    """
    Return an RGB tuple (0-1 range, matplotlib format) for a value between 0 and 1
    using a Garmin-style speed-based gradient.
    """
    v = max(0.0, min(1.0, float(value)))

    # Key points for interpolation (0-255 range for reference)
    scale_255 = [
        (0.0, (0, 75, 155)),     # dark blue
        (0.25, (60, 150, 245)),  # lighter blue
        (0.5, (125, 225, 25)),   # green
        (0.75, (255, 170, 76)),  # yellow/orange
        (1.0, (175, 0, 0)),      # dark red
    ]
    
    # Convert to 0-1 range for matplotlib
    scale = [(v0, (c0[0] / 255.0, c0[1] / 255.0, c0[2] / 255.0)) for v0, c0 in scale_255]

    for i in range(len(scale) - 1):
        v0, c0 = scale[i]
        v1, c1 = scale[i + 1]

        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0)
            r = c0[0] + (c1[0] - c0[0]) * t
            g = c0[1] + (c1[1] - c0[1]) * t
            b = c0[2] + (c1[2] - c0[2]) * t
            return (r, g, b)

    return scale[-1][1]


def create_speed_based_color_label(speed_based_color_min, speed_based_color_max, imperial_units=False, hr_mode=False, image_scale=1):
    """
    Create a labeled color scale image with min, average, and max labels.
    
    Args:
        speed_based_color_min (float): Minimum value
        speed_based_color_max (float): Maximum value
        imperial_units (bool): If True, use "mph" as unit, otherwise "km/h" (ignored if hr_mode=True)
        hr_mode (bool): If True, use heart rate mode with ♥ unit, otherwise use speed mode with km/h or mph
        image_scale (int): Scale factor (1, 2, 3, or 4) to determine which base image to use
    
    Returns:
        PIL.Image: Image with transparency (RGBA mode) containing the color scale and labels
    """
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    # Format scale for filename (special case for 0.7 -> "07x")
    from image_generator_utils import format_scale_for_label_filename
    scale_str = format_scale_for_label_filename(image_scale)
    image_path = os.path.join(module_dir, "img", f"speed_based_color_{scale_str}.png")
    
    # Debug logging - check file path and existence
    try:
        from image_generator_maptileutils import debug_log
        debug_log(f"create_speed_based_color_label: image_scale={image_scale}, looking for: {image_path}")
        debug_log(f"create_speed_based_color_label: file exists: {os.path.exists(image_path)}")
    except:
        pass
    
    # Check if base image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Base image not found: {image_path}")
    
    # Load the base color scale image
    base_img = Image.open(image_path)
    
    # Debug logging - check loaded image size
    try:
        from image_generator_maptileutils import debug_log
        debug_log(f"create_speed_based_color_label: image_scale={image_scale}, base_img.size={base_img.size} (width x height)")
    except:
        pass
    
    # Convert to RGBA if not already (to support transparency)
    if base_img.mode != "RGBA":
        base_img = base_img.convert("RGBA")
    
    # Calculate average value
    average = (speed_based_color_min + speed_based_color_max) / 2.0
    
    # Format values (round to integers for display, round up average)
    if hr_mode:
        # HR mode: use ♥ as unit
        min_text = f"{int(round(speed_based_color_min))} ♥"
        avg_text = f"{int(math.ceil(average))} ♥"
        max_text = f"{int(round(speed_based_color_max))} ♥"
    else:
        # Speed mode: use km/h or mph
        unit = "mph" if imperial_units else "km/h"
        min_text = f"{int(round(speed_based_color_min))} {unit}"
        avg_text = f"{int(math.ceil(average))} {unit}"
        max_text = f"{int(round(speed_based_color_max))} {unit}"
    
    # Try to load a default font, fall back to default if not available
    # Scale font size with image_scale (convert to int for PIL)
    base_font_size = 12
    font_size = int(round(base_font_size * image_scale))
    try:
        # Try to use a default font (scaled size)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            # Try alternative common font paths on Windows
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
    
    # Create ImageDraw object to measure text
    draw_temp = ImageDraw.Draw(base_img)
    
    # Measure text dimensions to determine required height
    # textbbox returns (left, top, right, bottom) where coordinates are relative to the anchor point
    # When drawing text, the y coordinate is the baseline by default
    # bbox[1] is typically negative (ascenders above baseline), bbox[3] is positive (descenders below baseline)
    text_bboxes = [draw_temp.textbbox((0, 0), text, font=font) for text in [min_text, avg_text, max_text]]
    
    # Calculate the maximum extent below baseline (descenders) and above baseline (ascenders)
    max_descender = max(bbox[3] for bbox in text_bboxes)  # Maximum extent below baseline
    max_ascender = abs(min(bbox[1] for bbox in text_bboxes))  # Maximum extent above baseline (make positive)
    
    # Spacing between base image and text (scale with image_scale, convert to int)
    base_spacing_below = 5
    spacing_below = int(round(base_spacing_below * image_scale))
    # Padding at bottom to ensure descenders are fully visible (scale with image_scale, convert to int)
    base_padding_bottom = 3
    padding_bottom = int(round(base_padding_bottom * image_scale))
    
    # Total height needed: spacing + ascender space + descender space + padding
    total_text_area_height = spacing_below + max_ascender + max_descender + padding_bottom
    
    # Create new image with space for text below
    new_width = base_img.width
    new_height = base_img.height + total_text_area_height
    new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))  # Transparent background
    
    # Debug logging
    try:
        from image_generator_maptileutils import debug_log
        debug_log(f"create_speed_based_color_label: new_img.size=({new_width}, {new_height})")
    except:
        pass
    
    # Paste the base image at the top
    new_img.paste(base_img, (0, 0), base_img if base_img.mode == "RGBA" else None)
    
    # Create ImageDraw object for the new image
    draw = ImageDraw.Draw(new_img)
    
    # Calculate text baseline position: base image height + spacing + ascender height
    # This positions the baseline so ascenders don't overlap the image and descenders have room below
    text_y = int(base_img.height + spacing_below + max_ascender)
    
    # Choose drawing function based on whether we're in HR mode (heart symbol should be red)
    if hr_mode:
        draw_text_func = _draw_text_with_red_heart
    else:
        draw_text_func = _draw_text_with_outline
    
    # Draw left text (min) with outline (and red heart if HR mode)
    # Add padding to prevent cutoff: 5 pixels
    padding_horizontal = 5
    min_x = padding_horizontal
    draw_text_func(draw, (min_x, text_y), min_text, font, fill_color=(255, 255, 255, 255))
    
    # Draw center text (average) with outline (and red heart if HR mode)
    avg_bbox = draw.textbbox((0, 0), avg_text, font=font)
    avg_text_width = avg_bbox[2] - avg_bbox[0]
    avg_x = (new_width - avg_text_width) // 2
    draw_text_func(draw, (avg_x, text_y), avg_text, font, fill_color=(255, 255, 255, 255))
    
    # Draw right text (max) with outline (and red heart if HR mode)
    # Add padding to prevent cutoff: 5 pixels
    max_bbox = draw.textbbox((0, 0), max_text, font=font)
    max_text_width = max_bbox[2] - max_bbox[0]
    max_x = new_width - max_text_width - padding_horizontal
    draw_text_func(draw, (max_x, text_y), max_text, font, fill_color=(255, 255, 255, 255))
    
    # Debug logging before return
    try:
        from image_generator_maptileutils import debug_log
        debug_log(f"create_speed_based_color_label: returning PIL Image with size {new_img.size}")
    except:
        pass
    
    return new_img


def create_hr_based_color_label(hr_based_color_min, hr_based_color_max, image_scale=1):
    """
    Just a redirect to create_speed_based_color_label with hr_mode=True.
    
    Args:
        hr_based_color_min (float): Minimum HR value
        hr_based_color_max (float): Maximum HR value
        image_scale (int): Scale factor (1, 2, 3, or 4) to determine which base image to use
    """
    return create_speed_based_color_label(hr_based_color_min, hr_based_color_max, imperial_units=False, hr_mode=True, image_scale=image_scale)


def create_hr_based_width_label(hr_based_width_min, hr_based_width_max, image_scale=1, route_color=None, skip_recolor=False):
    """
    Create a labeled width scale image with min, average, and max labels.
    Uses hr_based_width_{scale}x.png as the base image and ♥ as the unit.
    
    Args:
        hr_based_width_min (float): Minimum HR value (corresponds to thin line)
        hr_based_width_max (float): Maximum HR value (corresponds to thick line)
        image_scale (int): Scale factor (1, 2, 3, or 4) to determine which base image to use
        route_color (str or tuple, optional): Route color as hex string (e.g., '#2E8B57') or RGB tuple (0-255 range).
            If provided and skip_recolor is False, non-transparent pixels will be recolored from blue to this color.
        skip_recolor (bool): If True, skip recoloring and use original blue color. Default False.
    
    Returns:
        PIL.Image: Image with transparency (RGBA mode) containing the width scale and labels
    """
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    # Format scale for filename (special case for 0.7 -> "07x")
    from image_generator_utils import format_scale_for_label_filename
    scale_str = format_scale_for_label_filename(image_scale)
    image_path = os.path.join(module_dir, "img", f"hr_based_width_{scale_str}.png")
    
    # Check if base image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Base image not found: {image_path}")
    
    # Load the base width scale image
    base_img = Image.open(image_path)
    
    # Convert to RGBA if not already (to support transparency)
    if base_img.mode != "RGBA":
        base_img = base_img.convert("RGBA")
    
    # Recolor non-transparent pixels from blue to route color if route_color is provided and skip_recolor is False
    if route_color and not skip_recolor:
        # Convert route_color to RGB tuple (0-255 range) if it's a hex string
        if isinstance(route_color, str):
            # Remove '#' if present
            hex_color = route_color.lstrip('#')
            if len(hex_color) == 6:
                route_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            else:
                route_rgb = (0, 0, 255)  # Default to blue if invalid hex
        elif isinstance(route_color, (tuple, list)) and len(route_color) >= 3:
            # Assume it's already RGB (0-255 range) or RGBA
            route_rgb = tuple(int(route_color[i]) for i in range(3))
        else:
            route_rgb = (0, 0, 255)  # Default to blue if invalid format
        
        # Convert base image to numpy array for pixel manipulation
        import numpy as np
        img_array = np.array(base_img)
        
        # Create a mask for non-transparent pixels (alpha > 0)
        non_transparent_mask = img_array[:, :, 3] > 0
        
        # Recolor non-transparent pixels: change blue pixels to route color
        # We'll change any non-transparent pixel (assuming they're all blue in the original)
        img_array[non_transparent_mask, 0] = route_rgb[0]  # Red channel
        img_array[non_transparent_mask, 1] = route_rgb[1]  # Green channel
        img_array[non_transparent_mask, 2] = route_rgb[2]  # Blue channel
        # Keep alpha channel unchanged
        
        # Convert back to PIL Image
        base_img = Image.fromarray(img_array, 'RGBA')
    
    # Calculate average value
    average = (hr_based_width_min + hr_based_width_max) / 2.0
    
    # Format values with ♥ unit
    min_text = f"{int(round(hr_based_width_min))} ♥"
    avg_text = f"{int(math.ceil(average))} ♥"
    max_text = f"{int(round(hr_based_width_max))} ♥"
    
    # Try to load a default font, fall back to default if not available
    # Scale font size with image_scale (convert to int for PIL)
    base_font_size = 12
    font_size = int(round(base_font_size * image_scale))
    try:
        # Try to use a default font (scaled size)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            # Try alternative common font paths on Windows
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
    
    # Create ImageDraw object to measure text
    draw_temp = ImageDraw.Draw(base_img)
    
    # Measure text dimensions to determine required height
    text_bboxes = [draw_temp.textbbox((0, 0), text, font=font) for text in [min_text, avg_text, max_text]]
    
    # Calculate the maximum extent below baseline (descenders) and above baseline (ascenders)
    max_descender = max(bbox[3] for bbox in text_bboxes)
    max_ascender = abs(min(bbox[1] for bbox in text_bboxes))
    
    # Spacing between base image and text (scale with image_scale, convert to int)
    base_spacing_below = 5
    spacing_below = int(round(base_spacing_below * image_scale))
    # Padding at bottom to ensure descenders are fully visible (scale with image_scale, convert to int)
    base_padding_bottom = 3
    padding_bottom = int(round(base_padding_bottom * image_scale))
    
    # Total height needed: spacing + ascender space + descender space + padding
    total_text_area_height = spacing_below + max_ascender + max_descender + padding_bottom
    
    # Create new image with space for text below
    new_width = base_img.width
    new_height = base_img.height + total_text_area_height
    new_img = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))  # Transparent background
    
    # Paste the base image at the top
    new_img.paste(base_img, (0, 0), base_img if base_img.mode == "RGBA" else None)
    
    # Create ImageDraw object for the new image
    draw = ImageDraw.Draw(new_img)
    
    # Calculate text baseline position
    text_y = int(base_img.height + spacing_below + max_ascender)
    
    # Draw left text (min) with outline and red heart symbol
    # Add padding to prevent cutoff: 5 pixels
    padding_horizontal = 5
    min_x = padding_horizontal
    _draw_text_with_red_heart(draw, (min_x, text_y), min_text, font, fill_color=(255, 255, 255, 255))
    
    # Draw center text (average) with outline and red heart symbol
    avg_bbox = draw.textbbox((0, 0), avg_text, font=font)
    avg_text_width = avg_bbox[2] - avg_bbox[0]
    avg_x = (new_width - avg_text_width) // 2
    _draw_text_with_red_heart(draw, (avg_x, text_y), avg_text, font, fill_color=(255, 255, 255, 255))
    
    # Draw right text (max) with outline and red heart symbol
    # Add padding to prevent cutoff: 5 pixels
    max_bbox = draw.textbbox((0, 0), max_text, font=font)
    max_text_width = max_bbox[2] - max_bbox[0]
    max_x = new_width - max_text_width - padding_horizontal
    _draw_text_with_red_heart(draw, (max_x, text_y), max_text, font, fill_color=(255, 255, 255, 255))
    
    return new_img


# Create scale as output.png if executed independently
if __name__ == "__main__":
    width = 200
    height = 20

    img = Image.new("RGB", (width, height))
    pixels = img.load()

    for x in range(width):
        v = x / (width - 1)
        color_01 = speed_based_color(v)  # Returns 0-1 range
        # Convert to 0-255 range for PIL Image
        color_255 = (int(color_01[0] * 255), int(color_01[1] * 255), int(color_01[2] * 255))
        for y in range(height):
            pixels[x, y] = color_255

    img.save("output.png")
    print("Saved output.png")