"""
Post-processing to the images after they have been plotted.
"""

import os

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for image post-processing

import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from matplotlib import patheffects
from collections import OrderedDict
import oxipng
from image_generator_maptileutils import debug_log
from video_generator_create_single_frame import get_legend_theme_colors

def optimize_png_bytes(png_data: bytes) -> bytes:
    """
    Optimize PNG data using pyoxipng with balanced settings.  
    "Png compression test" (separate project) compares time/file size results. 

    Args:
        png_data: Raw PNG data as bytes        
    Returns:
        Optimized PNG data as bytes
    """
    try:
        debug_log("Starting PNG optimization with pyoxipng")
        optimized = oxipng.optimize_from_memory(
            png_data,
            level=3,  # Optimization level 1-6
            interlace=oxipng.Interlacing.Off,  # Disable interlacing for web
            strip=oxipng.StripChunks.safe(),  # Strip unnecessary chunks that won't affect rendering
            deflate=oxipng.Deflaters.libdeflater(10)  # Deflate compression 0-12
        )
        
        original_kb = len(png_data) / 1024
        optimized_kb = len(optimized) / 1024
        reduction = (1 - len(optimized) / len(png_data)) * 100
        
        debug_log(f"PNG Optimized. Original size: {original_kb:.1f}KB - Optimized size: {optimized_kb:.1f}KB - Reduction: {reduction:.1f}%")
        
        return optimized
    except Exception as e:
        debug_log(f"PNG optimization failed: {e}")
        return png_data  # Return original data if optimization fails

def add_title_text_to_plot(ax, title_text: str, image_width: int, image_height: int, image_scale: int | None = None):
    """
    Draw a title at the top center of the image, visually matching video title styling.

    Args:
        ax: Matplotlib axis object
        title_text: Text to render as the title
        image_width: Width in pixels (for scaling and padding conversion)
        image_height: Height in pixels
        image_scale: Optional precomputed image scale; if None it's derived from pixels
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

    base_font_size = 16
    font_size = base_font_size * image_scale

    # Convert ~10px padding into axes coordinates
    padding_pixels = 10
    padding_y = padding_pixels / image_height

    text_x = 0.5
    text_y = 1.0 - padding_y

    ax.text(
        text_x,
        text_y,
        title_text,
        transform=ax.transAxes,
        color='#cccccc',
        fontsize=font_size,
        fontweight='bold',
        ha='center',
        va='top',
        path_effects=[
            patheffects.Stroke(linewidth=2, foreground='black'),
            patheffects.Normal()
        ],
        zorder=110,
    )

def add_stamp_to_plot(ax, image_width: int, image_height: int, image_scale: int | None = None):
    """
    Add a resolution-appropriate stamp image to the bottom-left corner of the
    plot.

    The stamp resolution is chosen according to the total number of output
    pixels (image_width × image_height):

    < 8 MP   → stamp_1x.png
    8–18 MP  → stamp_2x.png
    18–33 MP → stamp_3x.png
    ≥ 33 MP  → stamp_4x.png

    Args:
        ax:           Matplotlib axis object.
        image_width:  Output image width in pixels.
        image_height: Output image height in pixels.
    """
    # Compute total pixels once for logging/reference
    total_pixels = image_width * image_height

    # Determine stamp by scale if provided, otherwise derive scale from pixel count
    if image_scale is None:
        if total_pixels < 8_000_000:
            image_scale = 1
        elif total_pixels < 18_000_000:
            image_scale = 2
        elif total_pixels < 33_000_000:
            image_scale = 3
        else:
            image_scale = 4

    stamp_filename = f"stamp_{image_scale}x.png"

    # Resolve path to the stamp inside the "img" folder next to this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    stamp_file = os.path.join(module_dir, "img", stamp_filename)

    padding_pixels = 10
    
    debug_log(f"Selected stamp '{stamp_filename}' based on {total_pixels:,} pixels")
    debug_log(f"Full stamp path: {stamp_file}")
    
    # Check if stamp file exists
    if not os.path.exists(stamp_file):
        debug_log(f"Stamp file not found: {stamp_file}")
        return
    
    try:
        # Load the stamp image
        stamp_image = mpimg.imread(stamp_file)
        debug_log(f"Loaded stamp image with shape: {stamp_image.shape}")
        
        # Get stamp dimensions
        stamp_height_pixels, stamp_width_pixels = stamp_image.shape[:2]
        debug_log(f"Stamp dimensions: {stamp_width_pixels}x{stamp_height_pixels} pixels")
        
        # Convert pixel measurements to figure coordinates (0-1 range)
        padding_left_fig = padding_pixels / image_width
        padding_bottom_fig = padding_pixels / image_height
        stamp_width_fig = stamp_width_pixels / image_width
        stamp_height_fig = stamp_height_pixels / image_height
        
        # Position stamp in bottom-left corner
        stamp_x = padding_left_fig
        stamp_y = padding_bottom_fig
        
        debug_log(f"Stamp position in figure coords: x={stamp_x:.4f}, y={stamp_y:.4f}")
        debug_log(f"Stamp size in figure coords: width={stamp_width_fig:.4f}, height={stamp_height_fig:.4f}")
        
        # Create inset axes positioned at bottom-left
        stamp_ax = ax.inset_axes([stamp_x, stamp_y, stamp_width_fig, stamp_height_fig], transform=ax.transAxes)
        
        # Display the stamp image
        stamp_ax.imshow(stamp_image)
        stamp_ax.axis('off')  # Hide axes for the stamp
        
        debug_log(f"Stamp added successfully at position ({stamp_x:.4f}, {stamp_y:.4f})")
        
    except Exception as e:
        debug_log(f"Error loading or displaying stamp: {e}")
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
    debug_log(f"Creating {legend_type} legend for {len(track_coords_with_metadata)} tracks")
    
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
    
    debug_log(f"Legend items collected: {list(legend_items.keys())}")
    
    if not legend_items:
        debug_log("No legend items to display")
        return
    
    # Sort legend items alphabetically by label for consistent legend order
    sorted_legend_items = OrderedDict(sorted(legend_items.items()))
    debug_log(f"Legend items sorted alphabetically: {list(sorted_legend_items.keys())}")
    
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
    
    debug_log(f"Legend position: x={legend_x:.3f}, y={legend_y:.3f}, height={legend_height_fig:.3f}")
    
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
    debug_log(f"Legend font size set to {font_size} using image_scale {image_scale}")

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
    
    debug_log(f"Legend created with {len(legend_patches)} items")