"""
Image generation map tiles caching for the Route Squiggler render client.
This module identifies the map tiles needed for image generation, then delegates
caching to the unified map_tile_caching module.
"""

# Standard library imports
import math
from typing import List, Tuple

# Local imports
from image_generator_maptileutils import set_cache_directory
from map_tile_caching import cache_required_tiles


def pre_cache_map_tiles_for_images(
    zoom_levels: List[int],
    map_bounds: Tuple[float, float, float, float],
    map_style: str,
    resolution_x: int,
    resolution_y: int,
    storage_box_credentials: dict = None,
    log_callback=None,
    debug_callback=None,
    progress_callback=None,
    provider_queue_callback=None,
    provider_start_callback=None,
    provider_progress_callback=None,
) -> dict | None:
    """
    Identify and cache map tiles for all zoom levels required for image generation.

    Args:
        zoom_levels: List of zoom levels to cache tiles for
        map_bounds: Tuple (lon_min, lon_max, lat_min, lat_max)
        map_style: Map style to use
        resolution_x: Image width in pixels (for calculating padded bounds)
        resolution_y: Image height in pixels (for calculating padded bounds)
        storage_box_credentials: Storage box credentials dict or None
        log_callback: Optional callback for logging messages
        debug_callback: Optional callback for debug messages
        progress_callback: Optional callback for progress updates

    Returns:
        dict: Tile caching result including counts and success flag
    """
    try:
        if debug_callback:
            debug_callback("Starting tile pre-caching for image generation")

        set_cache_directory(map_style)

        # Calculate padded bounds (same as image generation does)
        from image_generator_utils import ImageGenerator
        generator = ImageGenerator()
        target_aspect_ratio = resolution_x / resolution_y

        # Gather all required tiles across all zoom levels
        all_required_tiles = set()

        for zoom_level in zoom_levels:
            lon_min_padded, lon_max_padded, lat_min_padded, lat_max_padded = (
                generator.calculate_aspect_ratio_bounds(
                    map_bounds, target_aspect_ratio=target_aspect_ratio
                )
            )

            def deg2num(lat_deg, lon_deg, zoom):
                lat_rad = math.radians(lat_deg)
                n = 2.0 ** zoom
                x = int((lon_deg + 180.0) / 360.0 * n)
                y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
                return x, y

            northwest = deg2num(lat_max_padded, lon_min_padded, zoom_level)
            southeast = deg2num(lat_min_padded, lon_max_padded, zoom_level)

            x_min, y_min = northwest[0], northwest[1]
            x_max, y_max = southeast[0], southeast[1]
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    if zoom_level < 0 or zoom_level > 20:
                        continue
                    max_coord = 2 ** zoom_level
                    if not (0 <= x < max_coord and 0 <= y < max_coord):
                        continue
                    all_required_tiles.add((x, y, zoom_level))

        # Hand off to the unified caching system
        return cache_required_tiles(
            required_tiles=all_required_tiles,
            map_style=map_style,
            storage_box_credentials=storage_box_credentials,
            log_callback=log_callback,
            debug_callback=debug_callback,
            progress_callback=progress_callback,
            provider_queue_callback=provider_queue_callback,
            provider_start_callback=provider_start_callback,
            provider_progress_callback=provider_progress_callback,
        )

    except Exception as e:
        if log_callback:
            log_callback(f"Error in tile pre-caching: {str(e)}")
        import traceback
        if debug_callback:
            debug_callback(f"Traceback: {traceback.format_exc()}")
        return None
