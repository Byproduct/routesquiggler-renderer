"""
Video generation map image caching for the Route Squiggler render client.
This module handles caching map images for each unique bounding box needed for video frame generation.
"""

# Standard library imports
import json
import os
import time
from multiprocessing import Manager, Pool
from pathlib import Path

# Third-party imports (matplotlib backend must be set before pyplot import)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for map image caching

import cartopy.crs as ccrs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from image_generator_maptileutils import create_map_tiles, detect_zoom_level, set_cache_directory
from image_generator_utils import calculate_resolution_scale, apply_tile_threshold_multiplier
from job_request import set_attribution_from_theme
from video_generator_calculate_bounding_boxes import calculate_route_time_per_frame, calculate_unique_bounding_boxes
from video_generator_coordinate_encoder import encode_coords
from video_generator_cache_map_tiles import is_tile_cached
from video_generator_create_single_frame import _convert_bbox_to_web_mercator, _gps_to_web_mercator

# Debug feature to check that map image creation (step 4) has all required tiles available in local cache.
USE_CACHE_ONLY_TILES_WRAPPER = False


class _CacheOnlyTilesWrapper:
    """
    Wraps a CartoPy tile source so get_image() only serves from disk cache.
    Raises if a tile is requested that is not cached (guarantees no network in step 4).
    Debug feature disabled by default.
    """
    def __init__(self, tiles, map_style):
        self._tiles = tiles
        self._map_style = map_style
        self.crs = tiles.crs
        import cartopy
        self._cache_dir = cartopy.config.get('cache_dir', '')

    def get_image(self, tile):
        x, y, z = tile
        if not is_tile_cached(self._cache_dir, x, y, z, self._map_style):
            raise RuntimeError(
                f"Map tile ({x},{y},{z}) not in cache; step 4 must not download from network. "
                "Ensure step 3 (cache map tiles) completed successfully."
            )
        return self._tiles.get_image(tile)

    def __getattr__(self, name):
        return getattr(self._tiles, name)

def _get_from_cache_safe(shared_map_cache, encoded_bbox):
    """
    Get an image from the cache. Returns a copy of the numpy array.
    
    Args:
        shared_map_cache: The shared map cache dict
        encoded_bbox: The encoded bounding box key
        
    Returns:
        numpy array or None if not found
    """
    if shared_map_cache is None:
        return None
    if encoded_bbox in shared_map_cache:
        return shared_map_cache[encoded_bbox].copy()
    return None


def _render_map_image(bbox, json_data, canvas_size_override=None):
    """
    Render a map image for *bbox* and return it as an (H, W, 3) uint8 numpy
    array.  Does not read from or write to any cache.

    Used both by :func:`create_map_image_worker` (pre-caching phase) and by
    the per-frame on-the-fly fallback when a cache miss occurs during frame
    generation.

    Args:
        bbox: (lon_min, lon_max, lat_min, lat_max) in GPS degrees.
        json_data (dict): Job parameters (map_style, resolution, opacity, …).
        canvas_size_override (tuple | None): (width_px, height_px) or None.
            Pass the oversized square canvas size for follow_3d_rotate bg images.

    Returns:
        numpy.ndarray (H, W, 3) uint8, or None on failure.
    """
    try:
        lon_min, lon_max, lat_min, lat_max = bbox

        map_style = json_data.get('map_style', 'osm')
        map_transparency = float(json_data.get('map_transparency', 0))
        map_opacity = 100 - map_transparency
        map_alpha = map_opacity / 100.0
        video_resolution_x = int(json_data.get('video_resolution_x', 1920))
        video_resolution_y = int(json_data.get('video_resolution_y', 1080))

        max_tiles_config = {
            'osm': 100, 'otm': 100, 'cyclosm': 100,
            'stadia_light': 100, 'stadia_dark': 100, 'stadia_outdoors': 100,
            'stadia_toner': 100, 'stadia_watercolor': 150,
        }
        base_max_map_tiles = max_tiles_config.get(map_style, 100)
        resolution_scale = calculate_resolution_scale(video_resolution_x, video_resolution_y)
        map_detail = json_data.get('map_detail')
        video_tilt = json_data.get('video_tilt')
        max_map_tiles = apply_tile_threshold_multiplier(
            base_max_map_tiles, resolution_scale, min_value=1, map_detail=map_detail, video_tilt=video_tilt,
        )
        if canvas_size_override is not None:
            canvas_w, canvas_h = canvas_size_override
            area_ratio = (canvas_w * canvas_h) / max(1, video_resolution_x * video_resolution_y)
            max_map_tiles = max(1, int(max_map_tiles * area_ratio))

        if canvas_size_override is not None:
            width, height = canvas_size_override
        else:
            width, height = video_resolution_x, video_resolution_y
        dpi = 100
        figsize = (width / dpi, height / dpi)

        set_cache_directory(map_style)
        map_tiles = create_map_tiles(map_style, log_cache_miss=True)
        zoom_level = detect_zoom_level(
            (lon_min, lon_max, lat_min, lat_max),
            max_tiles=max_map_tiles,
            map_style=map_style,
        )

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=map_tiles.crs)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_aspect('auto')
        ax.add_image(map_tiles, zoom_level, alpha=map_alpha)
        plt.tight_layout(pad=0)
        fig.set_size_inches(figsize)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        # .copy() detaches the array from the figure buffer before we close the figure
        img_array = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)
        plt.clf()
        plt.close('all')
        return img_array

    except Exception as e:
        plt.close('all')
        print(f"_render_map_image error for bbox {bbox}: {e}")
        return None


def create_map_image_worker(args):
    """
    Worker function to create a map image for a single bounding box.

    Args:
        args (tuple): Contains
            (bbox_index, bbox, json_data, shared_progress_dict,
             shared_map_cache, canvas_size_override)
            canvas_size_override: (width_px, height_px) or None.
            When provided the image is rendered at that canvas size instead
            of the default video resolution.  Used for follow_3d_rotate
            background (bg) images which need an oversized square canvas.

    Returns:
        dict: Result information for this bounding box
    """
    bbox_index, bbox, json_data, shared_progress_dict, shared_map_cache, canvas_size_override = args

    def _bump_progress():
        with shared_progress_dict['lock']:
            shared_progress_dict['completed'] += 1
            return shared_progress_dict['completed'], shared_progress_dict['total']

    try:
        lon_min, lon_max, lat_min, lat_max = bbox
        encoded_bbox = encode_coords(lon_min, lon_max, lat_min, lat_max)

        # Skip immediately if the cache memory limit was already reached by a
        # previous worker.  The frame generator will produce this image on-the-fly.
        if shared_progress_dict.get('cache_limit_reached', False):
            current_progress, total_work = _bump_progress()
            return {
                'bbox_index': bbox_index,
                'bbox': bbox,
                'success': True,
                'was_cached': False,
                'was_skipped': True,
                'current_progress': current_progress,
                'total_work': total_work,
                'message': f"Skipped bbox {bbox_index + 1} (cache memory limit reached)",
            }

        # Fast path: image already in shared cache from an earlier worker.
        if shared_map_cache is not None and encoded_bbox in shared_map_cache:
            current_progress, total_work = _bump_progress()
            return {
                'bbox_index': bbox_index,
                'bbox': bbox,
                'success': True,
                'was_cached': True,
                'was_skipped': False,
                'current_progress': current_progress,
                'total_work': total_work,
                'message': f"Map image already in cache for bbox {bbox_index + 1}",
            }

        # Render the map image (no cache interaction).
        img_array = _render_map_image(bbox, json_data, canvas_size_override)

        if img_array is None:
            current_progress, total_work = _bump_progress()
            return {
                'bbox_index': bbox_index,
                'bbox': bbox,
                'success': False,
                'error': 'render returned None',
                'was_cached': False,
                'was_skipped': False,
                'current_progress': current_progress,
                'total_work': total_work,
                'message': f"Error creating map image for bbox {bbox_index + 1}: render returned None",
            }

        # Store in shared cache and track memory usage.
        if shared_map_cache is not None:
            shared_map_cache[encoded_bbox] = img_array

            limit_bytes = shared_progress_dict.get('cache_limit_bytes', 0)
            if limit_bytes > 0:
                with shared_progress_dict['lock']:
                    new_total = shared_progress_dict.get('cache_bytes', 0) + img_array.nbytes
                    shared_progress_dict['cache_bytes'] = new_total
                    if new_total >= limit_bytes and not shared_progress_dict.get('cache_limit_reached', False):
                        shared_progress_dict['cache_limit_reached'] = True

        current_progress, total_work = _bump_progress()
        return {
            'bbox_index': bbox_index,
            'bbox': bbox,
            'success': True,
            'was_cached': False,
            'was_skipped': False,
            'current_progress': current_progress,
            'total_work': total_work,
            'message': f"Created map image for bbox {bbox_index + 1}",
        }

    except Exception as e:
        plt.close('all')
        current_progress, total_work = _bump_progress()
        return {
            'bbox_index': bbox_index,
            'bbox': bbox,
            'success': False,
            'error': str(e),
            'was_cached': False,
            'was_skipped': False,
            'current_progress': current_progress,
            'total_work': total_work,
            'message': f"Error creating map image for bbox {bbox_index + 1}: {str(e)}",
        }


def cache_map_images_for_video(unique_bounding_boxes, json_data, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, canvas_size_override=None):
    """
    Cache map images for all unique bounding boxes required for video generation.
    Uses shared memory cache exclusively for maximum performance.

    Args:
        unique_bounding_boxes (set): Set of unique bounding boxes as tuples (lon_min, lon_max, lat_min, lat_max)
        json_data (dict): Job data containing video parameters
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for storing map images (required)
        canvas_size_override (tuple, optional): (width_px, height_px) canvas size to use instead
            of the video resolution.  Pass this for follow_3d_rotate background images.

    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting map image caching for video generation")
        
        # Validate that shared map cache is provided
        if shared_map_cache is None:
            if log_callback:
                log_callback("Error: Shared map cache is required for map image caching")
            return None
        
        if progress_callback:
            progress_callback("progress_bar_map_images", 0, "Preparing map image caching")
        
        # Convert bounding boxes to list and sort for locality (lat_min, lon_min)
        # so that chunks processed by the same worker tend to share tile cache
        bbox_list = sorted(unique_bounding_boxes, key=lambda b: (b[2], b[0]))  # (lat_min, lon_min)
        total_bboxes = len(bbox_list)
        
        if debug_callback:
            debug_callback(f"Processing {total_bboxes} unique bounding boxes for map images")
        
        # Determine number of workers to use
        if max_workers is None:
            max_workers = min(4, total_bboxes)  # Default to 4 workers or number of bboxes, whichever is smaller
        
        # Limit workers to available work
        workers_to_use = min(max_workers, total_bboxes)
        
        if debug_callback:
            debug_callback(f"Using {workers_to_use} worker processes for map image generation")
        
        # Handle case where there are no bounding boxes
        if total_bboxes == 0:
            if debug_callback:
                debug_callback("No bounding boxes to process")
            if progress_callback:
                progress_callback("progress_bar_map_images", 100, "No map images needed")
            return {
                'total_images_created': 0,
                'total_bboxes': 0,
                'success': True,
                'results': []
            }
        
        # Create shared progress tracking
        with Manager() as manager:
            shared_progress_dict = manager.dict()
            shared_progress_dict['completed'] = 0
            shared_progress_dict['total'] = total_bboxes
            shared_progress_dict['lock'] = manager.Lock()
            # Cache-limit tracking.  cache_limit_bytes == 0 means unlimited.
            from config import config as _config
            _limit_gb = _config.map_image_cache_size
            shared_progress_dict['cache_limit_bytes'] = int(_limit_gb * 1024 ** 3) if _limit_gb is not None else 0
            shared_progress_dict['cache_bytes'] = 0
            shared_progress_dict['cache_limit_reached'] = False
            
            # Create shared memory cache for map images (unlimited cache for dynamic zoom)
            if shared_map_cache is None:
                shared_map_cache = manager.dict()
            
            # Prepare work arguments for each worker
            work_args = []
            for bbox_index, bbox in enumerate(bbox_list):
                work_args.append((bbox_index, bbox, json_data, shared_progress_dict, shared_map_cache, canvas_size_override))
            
            # Start multiprocessing
            results = []
            rendered_images = 0
            skipped_images = 0
            failed_images = 0

            def _tally(result):
                nonlocal rendered_images, skipped_images, failed_images
                if result.get('was_skipped', False):
                    skipped_images += 1
                elif result['success']:
                    rendered_images += 1
                else:
                    failed_images += 1

            if workers_to_use == 1:
                # Single-threaded execution for debugging or when only one worker
                if debug_callback:
                    debug_callback("Using single-threaded execution")

                for work_arg in work_args:
                    result = create_map_image_worker(work_arg)
                    results.append(result)
                    _tally(result)

                    # Update progress
                    progress_percent = int((result['current_progress'] / result['total_work']) * 100)
                    progress_text = f"Creating map image {result['current_progress']}/{result['total_work']}"

                    if progress_callback:
                        progress_callback("progress_bar_map_images", progress_percent, progress_text)

                    # if log_callback:
                    #    log_callback(result['message'])
            else:
                # Multi-threaded execution
                if debug_callback:
                    debug_callback(f"Using multi-threaded execution with {workers_to_use} workers")

                with Pool(processes=workers_to_use) as pool:
                    # Submit all work
                    if progress_callback:
                        progress_callback("progress_bar_map_images", 0, "Starting map image creation")

                    # Process results as they complete (one bbox per task)
                    for result in pool.imap_unordered(create_map_image_worker, work_args):
                        results.append(result)
                        _tally(result)

                        # Update progress
                        progress_percent = int((result['current_progress'] / result['total_work']) * 100)
                        progress_text = f"Creating map image {result['current_progress']}/{result['total_work']}"

                        if progress_callback:
                            progress_callback("progress_bar_map_images", progress_percent, progress_text)

                        # if log_callback:
                        #     log_callback(result['message'])

        # Final progress update
        if progress_callback:
            progress_callback("progress_bar_map_images", 100, "Map image caching complete")

        return {
            'total_images_created': rendered_images,
            'total_images_skipped': skipped_images,
            'total_images_failed': failed_images,
            'total_bboxes': total_bboxes,
            'success': True,
            'results': results
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in map image caching: {str(e)}")
        return None


def cache_map_images(json_data=None, combined_route_data=None, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, unique_bounding_boxes=None):
    """
    Cache map images needed for video generation using shared memory exclusively.
    
    Args:
        json_data (dict, optional): Job data containing video parameters
        combined_route_data (dict, optional): Combined route data containing gpx_time_per_video_time
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for storing map images (required)
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting map image caching")
        
        # Validate that shared map cache is provided
        if shared_map_cache is None:
            if log_callback:
                log_callback("Error: Shared map cache is required for map image caching")
            return None
        
        if progress_callback:
            progress_callback("progress_bar_map_images", 0, "Starting map image caching")
        
        # Load job data if not provided
        if json_data is None:
            data_file = Path("data.json")
            if not data_file.exists():
                if log_callback:
                    log_callback("Error: data.json not found")
                return None
            
            with open(data_file, 'r') as f:
                json_data = json.load(f)
            set_attribution_from_theme(json_data)
        
        # Use gpx_time_per_video_time from combined_route_data if available, otherwise calculate it
        if combined_route_data and 'gpx_time_per_video_time' in combined_route_data:
            gpx_time_per_video_time = combined_route_data['gpx_time_per_video_time']
            video_fps = float(json_data.get('video_fps', 30))
            route_time_per_frame = gpx_time_per_video_time / video_fps
            
            if debug_callback:
                # debug_callback(f"Using gpx_time_per_video_time from combined_route_data: {gpx_time_per_video_time}")
                # debug_callback(f"Calculated route_time_per_frame: {route_time_per_frame:.6f} seconds")
                pass
        else:
            # Fallback to calculating route time per frame
            route_time_per_frame = calculate_route_time_per_frame(json_data, combined_route_data, log_callback, debug_callback)
            
            if route_time_per_frame is None:
                if log_callback:
                    log_callback("Error: Could not calculate route time per frame")
                return None
            
            if debug_callback:
                debug_callback(f"Route time per frame: {route_time_per_frame:.4f} seconds")
        
        # Fallback to calculate unique bounding boxes again if providing them from earlier step fails.
        if unique_bounding_boxes is None:
            debug_callback(f"Warning: Unique bounding boxes should have been calculated earlier and provided as parameter, but weren't. Calculating again.")
            unique_bounding_boxes = calculate_unique_bounding_boxes(
                json_data,
                route_time_per_frame,
                log_callback,
                max_workers,
                combined_route_data,
                debug_callback,
            )
            
            if unique_bounding_boxes is None:
                if log_callback:
                    log_callback("Error: Could not calculate unique bounding boxes")
                return None
            
            if debug_callback:
                debug_callback(f"Found {len(unique_bounding_boxes)} unique bounding boxes")
        else:
            if debug_callback:
                debug_callback(f"Reusing {len(unique_bounding_boxes)} precomputed unique bounding boxes")
        
        # For follow_3d_rotate all bboxes in unique_bounding_boxes are already the
        # expanded background bboxes (calculate_unique_bounding_boxes only returns bg
        # bboxes for this mode).  They must be rendered on the oversized square canvas
        # so that heading rotation never exposes black corners.
        video_mode = json_data.get('video_mode', 'dynamic')
        if video_mode == 'follow_3d_rotate':
            from video_generator_follow_3d import compute_bg_canvas_size
            video_w = int(json_data.get('video_resolution_x', 1920))
            video_h = int(json_data.get('video_resolution_y', 1080))
            canvas_size = compute_bg_canvas_size(video_w, video_h)
            if debug_callback:
                debug_callback(
                    f"follow_3d_rotate map images: {len(unique_bounding_boxes)} bg bboxes "
                    f"at {canvas_size}×{canvas_size}px"
                )
            canvas_override = (canvas_size, canvas_size)
        else:
            canvas_override = None

        cache_result = cache_map_images_for_video(
            unique_bounding_boxes, json_data, progress_callback, log_callback,
            debug_callback, max_workers, shared_map_cache,
            canvas_size_override=canvas_override,
        )
        if cache_result is None:
            if log_callback:
                log_callback("Error: Could not cache map images")
            return None
        
        if progress_callback:
            progress_callback("progress_bar_map_images", 100, "Map image caching complete")
        
        return {
            'route_time_per_frame': route_time_per_frame,
            'unique_bounding_boxes': unique_bounding_boxes,
            'cache_result': cache_result,
            'success': True
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in map image caching: {str(e)}")
        return None
