"""
Post-processing to the images after they have been plotted.
"""

# Standard library imports
import os
import subprocess
import tempfile
from collections import OrderedDict

# Third-party imports (matplotlib backend must be set before pyplot import)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for image post-processing

import cartopy.crs as ccrs
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import patheffects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image as PILImage

# Local imports
from image_generator_utils import calculate_resolution_scale
from video_generator_create_single_frame import get_legend_theme_colors
from write_log import write_debug_log, write_log


def optimize_png_bytes(png_data: bytes) -> bytes:
    """
    Optimize PNG data using oxipng CLI with balanced settings.  
    "Png compression test" (separate project) compares time/file size results. 

    Args:
        png_data: Raw PNG data as bytes        
    Returns:
        Optimized PNG data as bytes
    """
    try:
        write_debug_log("Starting PNG optimization with oxipng CLI")
        
        # Create a temporary file to write the PNG data
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(png_data)
        
        try:
            # Run oxipng CLI with equivalent settings
            # --opt 3: Optimization level 3 (1-6)
            # --strip safe: Strip unnecessary chunks that won't affect rendering
            # Note: --deflate and --no-interlace removed as they're no longer supported in newer oxipng versions
            # Compression is now handled automatically by the optimization level
            result = subprocess.run(
                ['oxipng', '--opt', '4', '--strip', 'safe', temp_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Read the optimized PNG data
            with open(temp_path, 'rb') as optimized_file:
                optimized = optimized_file.read()
            
            original_kb = len(png_data) / 1024
            optimized_kb = len(optimized) / 1024
            reduction = (1 - len(optimized) / len(png_data)) * 100
            
            write_debug_log(f"PNG Optimized. Original size: {original_kb:.1f}KB - Optimized size: {optimized_kb:.1f}KB - Reduction: {reduction:.1f}%")
            
            return optimized
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                pass  # Ignore cleanup errors
                
    except subprocess.CalledProcessError as e:
        write_debug_log(f"PNG optimization failed (oxipng error): {e.stderr if e.stderr else str(e)}")
        return png_data  # Return original data if optimization fails
    except Exception as e:
        write_debug_log(f"PNG optimization failed: {e}")
        return png_data  # Return original data if optimization fails

def add_title_text_to_plot(ax, title_text: str, image_width: int, image_height: int, image_scale: int | None = None, theme: str = 'light'):
    """
    Draw a title at the top center of the image, visually matching video title styling.

    Args:
        ax: Matplotlib axis object
        title_text: Text to render as the title
        image_width: Width in pixels (for scaling and padding conversion)
        image_height: Height in pixels
        image_scale: Optional precomputed image scale; if None it's derived from pixels
        theme: Theme for title text ('light' or 'dark')
            - 'light': light-colored text with dark outline
            - 'dark': dark-colored text with light outline
    """
    if not title_text:
        return
    # Determine image scale if not provided
    if image_scale is None:
        total_pixels = image_width * image_height
        if total_pixels < 8_000_000:
            image_scale = 1
        elif total_pixels < 18_000_000:
            image_scale = 2
        elif total_pixels < 33_000_000:
            image_scale = 3
        else:
            image_scale = 4

    base_font_size = 22
    font_size = base_font_size * image_scale

    # Scale padding with resolution_scale, then convert to axes coordinates
    base_padding_pixels = 10
    padding_pixels = base_padding_pixels * image_scale
    padding_y = padding_pixels / image_height

    text_x = 0.5
    text_y = 1.0 - padding_y

    # Calculate outline width: default 0.7, scaled by image_scale
    outline_width = 0.7 * image_scale

    if theme == 'dark':
        text_color = '#1a1a1a'  
        outline_color = '#e8e8e8' 
    else:  # 'light' theme (default)
        text_color = '#e8e8e8'  
        outline_color = '#1a1a1a' 

    ax.text(
        text_x,
        text_y,
        title_text,
        transform=ax.transAxes,
        color=text_color,
        fontsize=font_size,
        fontweight='bold',
        ha='center',
        va='top',
        path_effects=[
            patheffects.Stroke(linewidth=outline_width, foreground=outline_color),
            patheffects.Normal()
        ],
        zorder=110,
    )

def add_stamp_to_plot(ax, image_width: int, image_height: int, image_scale: int | None = None):
    """
    Add a resolution-appropriate stamp image to the bottom-left corner of the
    plot.

    Args:
        ax:           Matplotlib axis object.
        image_width:  Output image width in pixels.
        image_height: Output image height in pixels.
    """
    # Compute total pixels once for logging/reference
    total_pixels = image_width * image_height

    # Determine stamp by scale if provided, otherwise derive scale from pixel count
    if image_scale is None:
        image_scale = calculate_resolution_scale(image_width, image_height)

    stamp_scale_int = int(image_scale)
    stamp_filename = f"stamp_{stamp_scale_int}x.png"

    # Resolve path to the stamp inside the "img" folder next to this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    stamp_file = os.path.join(module_dir, "img", stamp_filename)

    padding_pixels = 10
    
    write_debug_log(f"Selected stamp '{stamp_filename}' based on {total_pixels:,} pixels")
    write_debug_log(f"Full stamp path: {stamp_file}")
    
    # Check if stamp file exists
    if not os.path.exists(stamp_file):
        write_debug_log(f"Stamp file not found: {stamp_file}")
        return
    
    try:
        # Load the stamp image
        stamp_image = mpimg.imread(stamp_file)
        write_debug_log(f"Loaded stamp image with shape: {stamp_image.shape}")
        
        # Get stamp dimensions
        stamp_height_pixels, stamp_width_pixels = stamp_image.shape[:2]
        write_debug_log(f"Stamp dimensions: {stamp_width_pixels}x{stamp_height_pixels} pixels")
        
        # Convert pixel measurements to figure coordinates (0-1 range)
        padding_left_fig = padding_pixels / image_width
        padding_bottom_fig = padding_pixels / image_height
        stamp_width_fig = stamp_width_pixels / image_width
        stamp_height_fig = stamp_height_pixels / image_height
        
        # Position stamp in bottom-left corner
        stamp_x = padding_left_fig
        stamp_y = padding_bottom_fig
        
        write_debug_log(f"Stamp position in figure coords: x={stamp_x:.4f}, y={stamp_y:.4f}")
        write_debug_log(f"Stamp size in figure coords: width={stamp_width_fig:.4f}, height={stamp_height_fig:.4f}")
        
        # Create inset axes positioned at bottom-left
        stamp_ax = ax.inset_axes([stamp_x, stamp_y, stamp_width_fig, stamp_height_fig], transform=ax.transAxes)
        
        # Display the stamp image
        stamp_ax.imshow(stamp_image)
        stamp_ax.axis('off')  # Hide axes for the stamp
        
        write_debug_log(f"Stamp added successfully at position ({stamp_x:.4f}, {stamp_y:.4f})")
        
    except Exception as e:
        write_debug_log(f"Error loading or displaying stamp: {e}")
        raise e

def add_legend_to_plot(ax, track_coords_with_metadata, track_lookup, legend_type: str, image_width: int, image_height: int, image_scale: int | None = None, legend_theme: str = "light"):
    """
    Add a legend to the plot based on the specified type.
    
    Args:
        ax: Matplotlib axis object
        track_coords_with_metadata: List of (lats, lons, color, name, filename) tuples
        track_lookup: Dictionary mapping filenames to track metadata
        legend_type: Type of legend ('file_name', 'year', 'month', 'day', 'people')
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        legend_theme: Legend theme ('light' or 'dark')
    """
    write_debug_log(f"Creating {legend_type} legend for {len(track_coords_with_metadata)} tracks")
    
    # Collect legend data based on type
    legend_items = OrderedDict()  # Use OrderedDict to maintain order
    
    for _, _, color, name, filename in track_coords_with_metadata:
        track_metadata = track_lookup.get(filename, {})
        date_str = track_metadata.get('date', '')
        
        if legend_type == "file_name":
            # Use filename as legend label
            label = filename
            legend_items[label] = color
            
        elif legend_type == "year":
            # Extract year from date
            if date_str and len(date_str) >= 4:
                year = date_str[:4]
                if year not in legend_items:  # Only add first occurrence (first route of the year)
                    legend_items[year] = color
                    
        elif legend_type == "month":
            # Extract year-month from date
            if date_str and len(date_str) >= 7:
                year_month = date_str[:7]  # YYYY-MM format
                if year_month not in legend_items:  # Only add first occurrence
                    legend_items[year_month] = color
                    
        elif legend_type == "day":
            # Use full date as label
            if date_str:
                if date_str not in legend_items:  # Only add first occurrence
                    legend_items[date_str] = color
        
        elif legend_type == "people":
            # Use provided track name when available; fallback to filename if name empty
            person_label = (name or "").strip() or filename
            if person_label not in legend_items:
                legend_items[person_label] = color
    
    write_debug_log(f"Legend items collected: {list(legend_items.keys())}")
    
    if not legend_items:
        write_debug_log("No legend items to display")
        return
    
    # Sort legend items alphabetically by label for consistent legend order
    sorted_legend_items = OrderedDict(sorted(legend_items.items()))
    write_debug_log(f"Legend items sorted alphabetically: {list(sorted_legend_items.keys())}")
    
    # Create legend patches
    legend_patches = []
    for label, color in sorted_legend_items.items():
        patch = mpatches.Patch(color=color, label=label)
        legend_patches.append(patch)
    
    # Determine theme colors for legend box and text
    theme_colors = get_legend_theme_colors(legend_theme)
    facecolor = theme_colors.get('facecolor', 'white')
    edgecolor = theme_colors.get('edgecolor', 'black')
    text_color = theme_colors.get('textcolor', '#333333')
    frame_alpha = theme_colors.get('framealpha', 0.8)

    # Calculate legend position and size
    legend_height_pixels = 150
    padding_pixels = 10
    
    # Calculate position in figure coordinates (0-1 range)
    legend_height_fig = legend_height_pixels / image_height
    padding_right_fig = padding_pixels / image_width
    padding_bottom_fig = padding_pixels / image_height
    
    # Position legend in bottom-right corner
    legend_x = 1.0 - padding_right_fig
    legend_y = padding_bottom_fig
    
    write_debug_log(f"Legend position: x={legend_x:.3f}, y={legend_y:.3f}, height={legend_height_fig:.3f}")
    
    # Determine font size based on provided image_scale or derive from pixels
    base_font_size = 10
    if image_scale is None:
        total_pixels = image_width * image_height
        if total_pixels < 8_000_000:
            image_scale = 1
        elif total_pixels < 18_000_000:
            image_scale = 2
        elif total_pixels < 33_000_000:
            image_scale = 3
        else:
            image_scale = 4

    font_size = base_font_size * image_scale
    write_debug_log(f"Legend font size set to {font_size} using image_scale {image_scale}")

    # Create the legend
    legend = ax.legend(
        handles=legend_patches,
        loc='lower right',
        bbox_to_anchor=(legend_x, legend_y, 0, legend_height_fig),
        bbox_transform=ax.transAxes,
        fontsize=font_size,
        frameon=True,
        fancybox=True,
        framealpha=frame_alpha,
        facecolor=facecolor,
        edgecolor=edgecolor,
        ncol=1
    )
    
    # Set legend properties
    legend.get_frame().set_linewidth(0.5)
    # Apply text color to legend labels
    try:
        for text in legend.get_texts():
            text.set_color(text_color)
    except Exception:
        pass
    
        write_debug_log(f"Legend created with {len(legend_patches)} items")


def add_speed_based_color_label_to_plot(ax, json_data: dict, image_width: int, image_height: int, image_scale: int | None = None, track_coords_with_metadata=None):
    """
    Add color and/or width labels to the plot if enabled.
    Supports displaying both a color label (speed or HR based) and a width label (HR based) side by side.
    
    Args:
        ax: Matplotlib axis object
        json_data: Job data dictionary containing label images
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        image_scale: Optional image scale factor; if None, calculated from resolution
        track_coords_with_metadata: Optional list of (lats, lons, color, name, filename) tuples.
            Used to extract the first route color for hr_based_width_label recoloring.
    """
    # Calculate image scale if not provided
    if image_scale is None:
        image_scale = calculate_resolution_scale(image_width, image_height)
    
    write_debug_log(f"add_speed_based_color_label_to_plot: image_width={image_width}, image_height={image_height}, image_scale={image_scale}")
    
    # Collect all labels to display (color label is either speed or HR based, width label is HR based)
    labels_to_draw = []  # List of (label_array, label_name) tuples
    
    if json_data:
        # Get color label (speed-based or HR-based, mutually exclusive)
        if json_data.get('speed_based_color_label', False):
            # Try to load the appropriate scaled version from file
            color_label = None
            module_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Determine filename based on scale (special case for 0.7 -> "07x")
            from image_generator_utils import format_scale_for_label_filename
            scale_str = format_scale_for_label_filename(image_scale)
            label_filename = f"speed_based_color_label_{scale_str}.png"
            
            label_file = os.path.join(module_dir, "img", label_filename)
            
            # Try to load the scaled version
            if os.path.exists(label_file):
                try:
                    color_label = mpimg.imread(label_file)
                    # Verify the loaded label has the expected size (approximately)
                    # Expected width for 1x is ~200px, so for scale N it should be ~200*N px
                    expected_min_width = 150 * image_scale  # Allow some tolerance
                    actual_width = color_label.shape[1]
                    if actual_width < expected_min_width:
                        write_debug_log(f"WARNING: Loaded {label_filename} has width {actual_width}px, expected at least {expected_min_width}px for scale {image_scale}. Recreating on-the-fly.")
                        color_label = None  # Force recreation
                    else:
                        write_debug_log(f"Loaded speed-based color label from {label_filename} (shape: {color_label.shape})")
                except Exception as e:
                    write_debug_log(f"Failed to load {label_filename}: {e}")
                    color_label = None
            
            # Create label on-the-fly using appropriately scaled base image
            if color_label is None:
                write_debug_log(f"Creating speed-based color label on-the-fly with image_scale={image_scale}")
                try:
                    from speed_based_color import create_speed_based_color_label
                    speed_min = json_data.get('speed_based_color_min', 5)
                    speed_max = json_data.get('speed_based_color_max', 35)
                    imperial_units = json_data and json_data.get('imperial_units', False) is True
                    label_image = create_speed_based_color_label(speed_min, speed_max, imperial_units, hr_mode=False, image_scale=image_scale)
                    write_debug_log(f"PIL Image size after creation: {label_image.size} (width x height)")
                    color_label = np.array(label_image)
                    write_debug_log(f"Successfully created speed-based color label on-the-fly using scale {image_scale}x (numpy shape: {color_label.shape}, expected width ~{200 * image_scale}px)")
                except Exception as e:
                    write_debug_log(f"Failed to create speed-based color label on-the-fly: {e}")
                    import traceback
                    write_debug_log(f"Traceback: {traceback.format_exc()}")
                    # Don't use fallback - if creation fails, we can't proceed with wrong-sized label
                    color_label = None
            
            if color_label is not None and isinstance(color_label, np.ndarray):
                labels_to_draw.append((color_label, "Speed-based color"))
        elif json_data.get('hr_based_color_label', False):
            # Try to load the appropriate scaled version from file
            color_label = None
            module_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Determine filename based on scale (special case for 0.7 -> "07x")
            # HR-based color uses the same base image as speed-based color
            from image_generator_utils import format_scale_for_label_filename
            scale_str = format_scale_for_label_filename(image_scale)
            label_filename = f"speed_based_color_label_{scale_str}.png"
            
            label_file = os.path.join(module_dir, "img", label_filename)
            
            # Try to load the scaled version
            if os.path.exists(label_file):
                try:
                    color_label = mpimg.imread(label_file)
                    # Verify the loaded label has the expected size (approximately)
                    # Expected width for 1x is ~200px, so for scale N it should be ~200*N px
                    expected_min_width = 150 * image_scale  # Allow some tolerance
                    actual_width = color_label.shape[1]
                    if actual_width < expected_min_width:
                        write_debug_log(f"WARNING: Loaded {label_filename} has width {actual_width}px, expected at least {expected_min_width}px for scale {image_scale}. Recreating on-the-fly.")
                        color_label = None  # Force recreation
                    else:
                        write_debug_log(f"Loaded HR-based color label from {label_filename} (shape: {color_label.shape})")
                except Exception as e:
                    write_debug_log(f"Failed to load {label_filename}: {e}")
                    color_label = None
            
            # Create label on-the-fly using appropriately scaled base image
            if color_label is None:
                write_debug_log(f"Creating HR-based color label on-the-fly with image_scale={image_scale}")
                try:
                    from speed_based_color import create_hr_based_color_label
                    hr_min = json_data.get('hr_based_color_min', 50)
                    hr_max = json_data.get('hr_based_color_max', 180)
                    label_image = create_hr_based_color_label(hr_min, hr_max, image_scale=image_scale)
                    write_debug_log(f"PIL Image size after creation: {label_image.size} (width x height)")
                    color_label = np.array(label_image)
                    write_debug_log(f"Successfully created HR-based color label on-the-fly using scale {image_scale}x (numpy shape: {color_label.shape}, expected width ~{200 * image_scale}px)")
                except Exception as e:
                    write_debug_log(f"Failed to create HR-based color label on-the-fly: {e}")
                    import traceback
                    write_debug_log(f"Traceback: {traceback.format_exc()}")
                    # Don't use fallback - if creation fails, we can't proceed with wrong-sized label
                    color_label = None
            
            if color_label is not None and isinstance(color_label, np.ndarray):
                labels_to_draw.append((color_label, "HR-based color"))
        
        # Get width label (can coexist with color label)
        if json_data.get('hr_based_width_label', False):
            # Always create the label on-the-fly to ensure text labels are included
            # The create_hr_based_width_label function loads the base image and adds text labels
            width_label = None
            try:
                from speed_based_color import create_hr_based_width_label
                hr_min = json_data.get('hr_based_width_min', 50)
                hr_max = json_data.get('hr_based_width_max', 180)
                
                # Extract first route color for recoloring (if speed_based_color or hr_based_color is not enabled)
                route_color = None
                skip_recolor = json_data.get('speed_based_color', False) or json_data.get('hr_based_color', False)  # Skip recoloring if speed_based_color or hr_based_color is enabled
                
                if not skip_recolor and track_coords_with_metadata and len(track_coords_with_metadata) > 0:
                    # Get color from first route (third element in tuple: lats, lons, color, name, filename)
                    first_route_color = track_coords_with_metadata[0][2]
                    if first_route_color:
                        route_color = first_route_color
                        write_debug_log(f"Using first route color for hr_based_width_label: {route_color}")
                
                label_image = create_hr_based_width_label(hr_min, hr_max, image_scale=image_scale, route_color=route_color, skip_recolor=skip_recolor)
                width_label = np.array(label_image)
                write_debug_log(f"Created HR-based width label on-the-fly using scale {image_scale}x (shape: {width_label.shape})")
            except Exception as e:
                write_debug_log(f"Failed to create HR-based width label on-the-fly: {e}")
                import traceback
                write_debug_log(f"Traceback: {traceback.format_exc()}")
                # Don't use fallback - if creation fails, we can't proceed with wrong-sized label
                width_label = None
            
            if width_label is not None and isinstance(width_label, np.ndarray):
                labels_to_draw.append((width_label, "HR-based width"))
                write_debug_log(f"Added HR-based width label to labels_to_draw. Total labels now: {len(labels_to_draw)}")
            else:
                write_debug_log(f"HR-based width label is None or not a numpy array. width_label={width_label}, type={type(width_label)}")
    
    write_debug_log(f"Total labels to draw: {len(labels_to_draw)}")
    for i, (label_array, label_name) in enumerate(labels_to_draw):
        write_debug_log(f"  Label {i}: {label_name}, shape={label_array.shape}")
    if not labels_to_draw:
        write_debug_log("No labels to draw, returning early")
        return
    
    try:
        # Scale padding with image_scale
        base_padding_bottom = 20
        padding_bottom = base_padding_bottom * image_scale
        padding_bottom_fig = padding_bottom / image_height
        # Scale gap between labels with image_scale
        base_gap_between_labels = 100  # pixels
        gap_between_labels = base_gap_between_labels * image_scale
        gap_fig = gap_between_labels / image_width
        
        def draw_single_label(label_array, label_name, x_position):
            """Helper to draw a single label at given x position (in figure coordinates)."""
            label_height, label_width = label_array.shape[:2]
            label_height_fig = label_height / image_height
            label_width_fig = label_width / image_width
            
            write_debug_log(f"{label_name} label - actual pixels: width={label_width}px, height={label_height}px")
            write_debug_log(f"{label_name} label position: x={x_position:.4f}, y={padding_bottom_fig:.4f}")
            write_debug_log(f"{label_name} label size in figure coords: width={label_width_fig:.4f}, height={label_height_fig:.4f}")
            
            # Create inset axes for the label (match stamp code approach)
            label_ax = ax.inset_axes([x_position, padding_bottom_fig, label_width_fig, label_height_fig], transform=ax.transAxes)
            
            # Debug: verify label array properties
            write_debug_log(f"{label_name} label array: dtype={label_array.dtype}, shape={label_array.shape}, min={label_array.min()}, max={label_array.max()}")
            
            # Display the label image (match stamp code - no normalization, no zorder on imshow)
            im = label_ax.imshow(label_array)
            label_ax.axis('off')  # Hide axes for the label
            
            # Debug: verify the image was created
            write_debug_log(f"{label_name} label imshow object created: {im}, axes visible: {label_ax.get_visible()}, axes position: {label_ax.get_position()}")
            
            write_debug_log(f"{label_name} label added successfully")
        
        if len(labels_to_draw) == 1:
            # Single label: center it horizontally
            label_array, label_name = labels_to_draw[0]
            write_debug_log(f"Drawing single label: {label_name}")
            label_width_fig = label_array.shape[1] / image_width
            x_position = (1.0 - label_width_fig) / 2.0
            draw_single_label(label_array, label_name, x_position)
        else:
            # Two labels: draw side by side with gap
            label1_array, label1_name = labels_to_draw[0]
            label2_array, label2_name = labels_to_draw[1]
            write_debug_log(f"Drawing two labels side by side: {label1_name} and {label2_name}")
            
            label1_width_fig = label1_array.shape[1] / image_width
            label2_width_fig = label2_array.shape[1] / image_width
            
            # Calculate total width and starting positions
            total_width_fig = label1_width_fig + gap_fig + label2_width_fig
            start_x = (1.0 - total_width_fig) / 2.0
            
            write_debug_log(f"Two-label layout: start_x={start_x:.4f}, label1_width_fig={label1_width_fig:.4f}, gap_fig={gap_fig:.4f}, label2_width_fig={label2_width_fig:.4f}")
            
            # Draw first label (color)
            draw_single_label(label1_array, label1_name, start_x)
            
            # Draw second label (width)
            label2_x = start_x + label1_width_fig + gap_fig
            write_debug_log(f"Drawing second label at x={label2_x:.4f}")
            draw_single_label(label2_array, label2_name, label2_x)
        
    except Exception as e:
        write_debug_log(f"Error adding labels: {e}")


def add_markers_to_plot(ax, track_coords_with_metadata, image_width: int, image_height: int, lon_min: float, lon_max: float, lat_min: float, lat_max: float, image_scale: int | None = None):
    """
    Add start and finish markers to the plot at the first and last points of the route.
    
    Args:
        ax: Matplotlib axis object with cartopy projection
        track_coords_with_metadata: List of (lats, lons, color, name, filename) tuples
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        lon_min: Minimum longitude in degrees (PlateCarree)
        lon_max: Maximum longitude in degrees (PlateCarree)
        lat_min: Minimum latitude in degrees (PlateCarree)
        lat_max: Maximum latitude in degrees (PlateCarree)
        image_scale: Optional image scale factor; if None, calculated from resolution
    """
    if not track_coords_with_metadata:
        write_debug_log("No track coordinates available for markers")
        return
    
    # Calculate image scale if not provided
    if image_scale is None:
        image_scale = calculate_resolution_scale(image_width, image_height)
    
    write_debug_log(f"add_markers_to_plot: image_width={image_width}, image_height={image_height}, image_scale={image_scale}")
    
    # Get first point from first track and last point from last track
    first_track = track_coords_with_metadata[0]
    last_track = track_coords_with_metadata[-1]
    
    first_lats, first_lons, _, _, _ = first_track
    last_lats, last_lons, _, _, _ = last_track
    
    if not first_lats or not first_lons:
        write_debug_log("First track has no coordinates")
        return
    
    if not last_lats or not last_lons:
        write_debug_log("Last track has no coordinates")
        return
    
    # Get first point coordinates
    start_lat = first_lats[0]
    start_lon = first_lons[0]
    
    # Get last point coordinates
    end_lat = last_lats[-1]
    end_lon = last_lons[-1]
    
    write_debug_log(f"Start marker at: lat={start_lat}, lon={start_lon}")
    write_debug_log(f"Finish marker at: lat={end_lat}, lon={end_lon}")
    
    # Resolve path to the img folder
    module_dir = os.path.dirname(os.path.abspath(__file__))
    start_file = os.path.join(module_dir, "img", "start.png")
    finish_file = os.path.join(module_dir, "img", "finish.png")
    
    # Base size is 35 pixels, multiply by image_scale
    target_size_pixels = int(35 * image_scale)
    
    def load_and_resize_marker(file_path, marker_name):
        """Load and resize a marker image to target size while maintaining aspect ratio."""
        if not os.path.exists(file_path):
            write_debug_log(f"{marker_name} file not found: {file_path}")
            return None
        
        try:
            # Load image using PIL for better resizing control
            img = PILImage.open(file_path)
            original_width, original_height = img.size
            
            # Calculate new size maintaining aspect ratio
            aspect_ratio = original_width / original_height
            if original_width >= original_height:
                new_width = target_size_pixels
                new_height = int(target_size_pixels / aspect_ratio)
            else:
                new_height = target_size_pixels
                new_width = int(target_size_pixels * aspect_ratio)
            
            # Resize image
            img_resized = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
            
            # Convert to numpy array for matplotlib
            marker_array = np.array(img_resized)
            
            write_debug_log(f"{marker_name} loaded and resized: original={original_width}x{original_height}, resized={new_width}x{new_height}")
            
            return marker_array, new_width, new_height
            
        except Exception as e:
            write_debug_log(f"Error loading {marker_name}: {e}")
            return None
    
    # Load and resize markers
    start_result = load_and_resize_marker(start_file, "Start")
    finish_result = load_and_resize_marker(finish_file, "Finish")
    
    # Calculate spans in degrees (bounds are already in PlateCarree/degrees)
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    write_debug_log(f"Map extent (degrees): lon=[{lon_min:.6f}, {lon_max:.6f}], lat=[{lat_min:.6f}, {lat_max:.6f}]")
    write_debug_log(f"Map spans: lon_span={lon_span:.6f}, lat_span={lat_span:.6f}")
    
    if not start_result:
        write_debug_log("Failed to load start marker, skipping")
    else:
        start_array, start_width, start_height = start_result
        try:
            # Convert geographic coordinates to figure coordinates (0-1 range)
            # Normalize coordinates to 0-1 range
            start_x_fig = (start_lon - lon_min) / lon_span
            start_y_fig = (start_lat - lat_min) / lat_span
            
            # Calculate marker size in figure coordinates
            marker_width_fig = start_width / image_width
            marker_height_fig = start_height / image_height
            
            # Position marker so bottom-left corner is at the point
            marker_x_fig = start_x_fig
            marker_y_fig = start_y_fig
            
            write_debug_log(f"Start marker image size: {start_width}x{start_height} pixels")
            write_debug_log(f"Start point in figure coords: ({start_x_fig:.4f}, {start_y_fig:.4f})")
            write_debug_log(f"Marker size in figure coords: {marker_width_fig:.4f}x{marker_height_fig:.4f}")
            write_debug_log(f"Marker position in figure coords: ({marker_x_fig:.4f}, {marker_y_fig:.4f})")
            
            # Use inset_axes like the stamp does, but position at geographic coordinates
            # Convert to figure coordinates first (same approach as preview_image.py)
            marker_ax = ax.inset_axes(
                [marker_x_fig, marker_y_fig, marker_width_fig, marker_height_fig],
                transform=ax.transAxes
            )
            marker_ax.imshow(start_array)
            marker_ax.axis('off')
            write_debug_log(f"Added start marker image at: ({start_lon:.6f}, {start_lat:.6f})")
            
        except Exception as e:
            write_debug_log(f"Error placing start marker: {e}")
            import traceback
            write_debug_log(f"Traceback: {traceback.format_exc()}")
    
    if not finish_result:
        write_debug_log("Failed to load finish marker, skipping")
    else:
        finish_array, finish_width, finish_height = finish_result
        try:
            # Convert geographic coordinates to figure coordinates (0-1 range)
            # Normalize coordinates to 0-1 range
            end_x_fig = (end_lon - lon_min) / lon_span
            end_y_fig = (end_lat - lat_min) / lat_span
            
            # Calculate marker size in figure coordinates
            marker_width_fig = finish_width / image_width
            marker_height_fig = finish_height / image_height
            
            # Position marker so bottom-left corner is at the point
            marker_x_fig = end_x_fig
            marker_y_fig = end_y_fig
            
            write_debug_log(f"Finish marker image size: {finish_width}x{finish_height} pixels")
            write_debug_log(f"End point in figure coords: ({end_x_fig:.4f}, {end_y_fig:.4f})")
            write_debug_log(f"Marker size in figure coords: {marker_width_fig:.4f}x{marker_height_fig:.4f}")
            write_debug_log(f"Marker position in figure coords: ({marker_x_fig:.4f}, {marker_y_fig:.4f})")
            
            # Use inset_axes like the stamp does, but position at geographic coordinates
            # Convert to figure coordinates first (same approach as preview_image.py)
            marker_ax = ax.inset_axes(
                [marker_x_fig, marker_y_fig, marker_width_fig, marker_height_fig],
                transform=ax.transAxes
            )
            marker_ax.imshow(finish_array)
            marker_ax.axis('off')
            write_debug_log(f"Added finish marker image at: ({end_lon:.6f}, {end_lat:.6f})")
            
        except Exception as e:
            write_debug_log(f"Error placing finish marker: {e}")
            import traceback
            write_debug_log(f"Traceback: {traceback.format_exc()}")