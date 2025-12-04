from PIL import Image, ImageDraw, ImageFont
import os
import math


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


def create_speed_based_color_label(speed_based_color_min, speed_based_color_max, imperial_units=False):
    """
    Create a labeled speed-based color scale image with min, average, and max speed labels.
    
    Args:
        speed_based_color_min (float): Minimum speed value
        speed_based_color_max (float): Maximum speed value
        imperial_units (bool): If True, use "mph" as unit, otherwise "km/h"
    
    Returns:
        PIL.Image: Image with transparency (RGBA mode) containing the color scale and labels
    """
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(module_dir, "img", "speed_based_color.png")
    
    # Load the base color scale image
    base_img = Image.open(image_path)
    
    # Convert to RGBA if not already (to support transparency)
    if base_img.mode != "RGBA":
        base_img = base_img.convert("RGBA")
    
    # Calculate average speed
    speed_average = (speed_based_color_min + speed_based_color_max) / 2.0
    
    # Determine unit
    unit = "mph" if imperial_units else "km/h"
    
    # Format speed values (round to integers for display, round up average)
    min_text = f"{int(round(speed_based_color_min))} {unit}"
    avg_text = f"{int(math.ceil(speed_average))} {unit}"
    max_text = f"{int(round(speed_based_color_max))} {unit}"
    
    # Try to load a default font, fall back to default if not available
    try:
        # Try to use a default font (size 12)
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            # Try alternative common font paths on Windows
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
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
    
    # Spacing between base image and text
    spacing_below = 5
    # Padding at bottom to ensure descenders are fully visible
    padding_bottom = 3
    
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
    
    # Calculate text baseline position: base image height + spacing + ascender height
    # This positions the baseline so ascenders don't overlap the image and descenders have room below
    text_y = base_img.height + spacing_below + max_ascender
    
    # Draw left text (min)
    draw.text((0, text_y), min_text, fill=(255, 255, 255, 255), font=font)
    
    # Draw center text (average)
    avg_bbox = draw.textbbox((0, 0), avg_text, font=font)
    avg_text_width = avg_bbox[2] - avg_bbox[0]
    avg_x = (new_width - avg_text_width) // 2
    draw.text((avg_x, text_y), avg_text, fill=(255, 255, 255, 255), font=font)
    
    # Draw right text (max)
    max_bbox = draw.textbbox((0, 0), max_text, font=font)
    max_text_width = max_bbox[2] - max_bbox[0]
    max_x = new_width - max_text_width
    draw.text((max_x, text_y), max_text, fill=(255, 255, 255, 255), font=font)
    
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