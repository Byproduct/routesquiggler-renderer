"""
Video generation map tiles caching for the Route Squiggler render client.
This module identifies the map tiles needed for video frame generation, then
delegates caching to the unified map_tile_caching module.
"""

# Standard library imports
import json
import math
import os
from pathlib import Path

# Local imports
from image_generator_maptileutils import set_cache_directory
from job_request import set_attribution_from_theme
from map_tile_caching import cache_required_tiles
from video_generator_calculate_bounding_boxes import calculate_route_time_per_frame, calculate_unique_bounding_boxes

# Debug flag to control console output
MAP_TILE_CACHING_DEBUG = False


def is_tile_cached(cache_dir, x, y, zoom, map_style):
    """
    Check if a tile is already cached in the file system.
    
    Args:
        cache_dir (str): Base cache directory
        x (int): Tile x coordinate
        y (int): Tile y coordinate
        zoom (int): Zoom level
        map_style (str): Map style (used to determine file extension)
    
    Returns:
        bool: True if tile is cached, False otherwise
    """
    try:
        if map_style.startswith('stadia'):
            subdir = 'StadiaTiles'
        elif map_style.startswith('geoapify'):
            subdir = 'GeoapifyTiles'
        elif map_style.startswith('thunderforest'):
            subdir = 'ThunderforestTiles'
        elif map_style.startswith('esri256'):
            subdir = 'Esri256Tiles'
        elif map_style.startswith('esri512'):
            subdir = 'Esri512Tiles'
        elif map_style.startswith('mapbox'):
            subdir = 'MapboxTiles'
        else:
            style_subdir_mapping = {
                'osm': 'OSM',
                'otm': 'OpenTopoMapTiles',
                'cyclosm': 'CyclOSMTiles',
                'geoapify_carto': 'GeoapifyCarto',
                'geoapify_bright': 'GeoapifyBright',
                'geoapify_bright_grey': 'GeoapifyBrightGrey',
                'geoapify_bright_smooth': 'GeoapifyBrightSmooth',
                'geoapify_klokantech': 'GeoapifyKlokantech',
                'geoapify_liberty': 'GeoapifyLiberty',
                'geoapify_maptiler': 'GeoapifyMaptiler',
                'geoapify_toner': 'GeoapifyToner',
                'geoapify_toner_grey': 'GeoapifyTonerGrey',
                'geoapify_positron': 'GeoapifyPositron',
                'geoapify_positron_blue': 'GeoapifyPositronBlue',
                'geoapify_positron_red': 'GeoapifyPositronRed',
                'geoapify_dark': 'GeoapifyDark',
                'geoapify_dark_brown': 'GeoapifyDarkBrown',
                'geoapify_grey': 'GeoapifyGrey',
                'geoapify_purple': 'GeoapifyPurple',
                'geoapify_purple_roads': 'GeoapifyPurpleRoads',
                'geoapify_yellow_roads': 'GeoapifyYellowRoads',
                'thunderforest_atlas': 'ThunderforestAtlas',
                'thunderforest_mobile_atlas': 'ThunderforestMobileAtlas',
                'thunderforest_cycle': 'ThunderforestCycle',
                'thunderforest_landscape': 'ThunderforestLandscape',
                'thunderforest_neighbourhood': 'ThunderforestNeighbourhood',
                'thunderforest_outdoors': 'ThunderforestOutdoors',
                'thunderforest_pioneer': 'ThunderforestPioneer',
                'thunderforest_spinal': 'ThunderforestSpinal',
                'thunderforest_transport': 'ThunderforestTransport',
                'thunderforest_transport_dark': 'ThunderforestTransportDark',
            }
            subdir = style_subdir_mapping.get(map_style, map_style.upper())

        tile_file = os.path.join(cache_dir, subdir, f"{x}_{y}_{zoom}.npy")

        if MAP_TILE_CACHING_DEBUG:
            print(f"Looking for tile file: {tile_file}")
            print(f"  - File exists: {os.path.exists(tile_file)}")
            if os.path.exists(tile_file):
                print(f"  - File size: {os.path.getsize(tile_file)} bytes")
        
        if os.path.exists(tile_file) and os.path.getsize(tile_file) > 0:
            return True
        
        return False
    except Exception as e:
        if MAP_TILE_CACHING_DEBUG:
            print(f"Error checking tile cache: {e}")
        return False


def pre_cache_map_tiles_for_video(
    unique_bounding_boxes,
    json_data,
    storage_box_credentials=None,
    progress_callback=None,
    log_callback=None,
    debug_callback=None,
    provider_queue_callback=None,
    provider_start_callback=None,
    provider_progress_callback=None,
    canvas_size_override=None,
    max_tiles_preflight=None,
):
    """
    Identify and cache map tiles for all unique (bbox, zoom) pairs required
    for video generation.

    Args:
        unique_bounding_boxes (list): Unique ``(bbox, zoom)`` pairs where
            bbox is ``(lon_min, lon_max, lat_min, lat_max)`` and zoom is an
            int. The zoom level was assigned by step 2.5's zoom stabilization
            and is authoritative across the entire pipeline.
        json_data (dict): Job data containing video parameters.
        storage_box_credentials (dict | None): Storage box credentials.
        progress_callback: ``(bar_name, pct, text)`` progress function.
        log_callback: Regular log function.
        debug_callback: Debug log function.
        canvas_size_override: (width_px, height_px) or None. Still accepted
            for forward/backward compatibility of the call site (used by
            follow_3d_rotate) but no longer affects zoom selection here
            because zoom levels arrive already-decided from step 2.5.
        max_tiles_preflight (int | None): If set and the computed total
            number of unique required tiles is strictly greater than this
            limit, return early without downloading anything. The returned
            dict has ``preflight_exceeded=True`` and ``total_required_tiles``
            so the caller can decide to retry with reduced map detail.

    Returns:
        dict: Cache results, or ``None`` on error.
    """
    try:
        if debug_callback:
            debug_callback("Starting tile pre-caching for video generation")

        map_style = json_data.get('map_style', 'osm')
        set_cache_directory(map_style)

        pairs = list(unique_bounding_boxes)
        total_pairs = len(pairs)

        if debug_callback:
            debug_callback(
                f"Processing {total_pairs} unique (bbox, zoom) pairs "
                f"(canvas_size_override={canvas_size_override})"
            )

        # Phase 1: Gather all required tiles across all (bbox, zoom) pairs.
        # Zoom levels arrive already-decided from step 2.5's stabilization
        # pass, so we do not call detect_zoom_level here.
        all_required_tiles = set()

        for bbox, zoom_level in pairs:
            lon_min, lon_max, lat_min, lat_max = bbox

            def deg2num(lat_deg, lon_deg, zoom):
                lat_rad = math.radians(lat_deg)
                n = 2.0 ** zoom
                x = int((lon_deg + 180.0) / 360.0 * n)
                y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
                return x, y

            northwest = deg2num(lat_max, lon_min, zoom_level)
            southeast = deg2num(lat_min, lon_max, zoom_level)

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

        total_required = len(all_required_tiles)
        if debug_callback:
            debug_callback(f"Found {total_required} unique tiles required across all bounding boxes")

        # Preflight: if a tile-budget limit was supplied and we exceed it,
        # bail out before downloading so the caller can lower map_detail and
        # retry. We mark the call as successful (no error occurred) and let
        # the caller dispatch on `preflight_exceeded`.
        if max_tiles_preflight is not None and total_required > max_tiles_preflight:
            if log_callback:
                log_callback(
                    f"Tile preflight: {total_required} tiles required "
                    f"(limit {max_tiles_preflight}); skipping download to allow retry."
                )
            return {
                'total_tiles_cached': 0,
                'total_tiles_skipped': 0,
                'total_tiles_to_cache': total_required,
                'total_required_tiles': total_required,
                'total_bboxes': total_pairs,
                'error_types': {},
                'success': True,
                'preflight_exceeded': True,
                'tiles_local': 0,
                'tiles_remote': 0,
                'tiles_service': 0,
            }

        # Hand off to the unified caching system
        tile_cache_info = cache_required_tiles(
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
        return {
            'total_tiles_cached': total_required,
            'total_tiles_skipped': 0,
            'total_tiles_to_cache': total_required,
            'total_required_tiles': total_required,
            'total_bboxes': total_pairs,
            'error_types': {},
            'success': bool(tile_cache_info.get('success')) if tile_cache_info else False,
            'preflight_exceeded': False,
            'tiles_local': int(tile_cache_info.get('count_local', 0)) if tile_cache_info else 0,
            'tiles_remote': int(tile_cache_info.get('count_remote', 0)) if tile_cache_info else 0,
            'tiles_service': int(tile_cache_info.get('count_provider', 0)) if tile_cache_info else 0,
        }

    except Exception as e:
        if log_callback:
            log_callback(f"Error in map tile pre-caching: {str(e)}")
        return None


def cache_map_tiles(
    json_data=None,
    combined_route_data=None,
    storage_box_credentials=None,
    progress_callback=None,
    log_callback=None,
    debug_callback=None,
    max_workers=None,
    provider_queue_callback=None,
    provider_start_callback=None,
    provider_progress_callback=None,
    max_tiles_preflight=None,
):
    """
    Cache map tiles needed for video generation.
    
    Args:
        json_data (dict, optional): Job data containing video parameters
        combined_route_data (dict, optional): Combined route data containing gpx_time_per_video_time
        storage_box_credentials (dict | None): Storage box credentials.
        progress_callback (callable, optional): Function to call with progress updates
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        max_tiles_preflight (int, optional): If set and the total number of
            required tiles exceeds this limit, return early without
            downloading. The returned dict will have ``preflight_exceeded=True``
            and ``total_required_tiles``; ``success`` is still True because
            no error occurred. Used to enable map_detail auto-downgrade for
            free jobs that would otherwise consume excessive tile bandwidth.
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting map tile caching")
        
        if progress_callback:
            progress_callback("progress_bar_tiles", 0, "Starting map tile caching")
        
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
        
        # Use gpx_time_per_video_time from combined_route_data if available
        if combined_route_data and 'gpx_time_per_video_time' in combined_route_data:
            gpx_time_per_video_time = combined_route_data['gpx_time_per_video_time']
            video_fps = float(json_data.get('video_fps', 30))
            route_time_per_frame = gpx_time_per_video_time / video_fps
        else:
            route_time_per_frame = calculate_route_time_per_frame(json_data, log_callback)
            if route_time_per_frame is None:
                if log_callback:
                    log_callback("Error: Could not calculate route time per frame")
                return None
            if debug_callback:
                debug_callback(f"Route time per frame: {route_time_per_frame:.4f} seconds")
        
        # Calculate unique bounding boxes across all frames
        unique_bounding_boxes = calculate_unique_bounding_boxes(
            json_data, route_time_per_frame, log_callback,
            max_workers, combined_route_data, debug_callback,
        )
        
        if unique_bounding_boxes is None:
            if log_callback:
                log_callback("Error: Could not calculate unique bounding boxes")
            return None
        
        if debug_callback:
            debug_callback(f"Found {len(unique_bounding_boxes)} unique bounding boxes")

        # For follow_3d_rotate the bboxes passed in are the oversized bg
        # bboxes (expanded to cover a square canvas of diag(W,H) px).  Step 4
        # renders them on that canvas with an area-ratio-inflated max-tiles
        # budget, so we must use the same budget here or the two steps will
        # pick different zoom levels and step 4 will download missing tiles.
        video_mode = json_data.get('video_mode', 'dynamic')
        if video_mode == 'follow_3d_rotate':
            from video_generator_follow_3d import compute_bg_canvas_size
            video_w = int(json_data.get('video_resolution_x', 1920))
            video_h = int(json_data.get('video_resolution_y', 1080))
            canvas_size = compute_bg_canvas_size(video_w, video_h)
            canvas_override = (canvas_size, canvas_size)
            if debug_callback:
                debug_callback(
                    f"follow_3d_rotate tile caching: canvas {canvas_size}x{canvas_size}px "
                    f"for {video_w}x{video_h} output"
                )
        else:
            canvas_override = None

        # Pre-cache map tiles for all unique bounding boxes
        cache_result = pre_cache_map_tiles_for_video(
            unique_bounding_boxes, json_data,
            storage_box_credentials=storage_box_credentials,
            progress_callback=progress_callback,
            log_callback=log_callback,
            debug_callback=debug_callback,
            provider_queue_callback=provider_queue_callback,
            provider_start_callback=provider_start_callback,
            provider_progress_callback=provider_progress_callback,
            canvas_size_override=canvas_override,
            max_tiles_preflight=max_tiles_preflight,
        )
        
        if cache_result is None:
            if log_callback:
                log_callback("Error: Could not cache map tiles")
            return None

        inner_ok = bool(cache_result.get('success'))
        preflight_exceeded = bool(cache_result.get('preflight_exceeded'))

        if progress_callback and not preflight_exceeded:
            progress_callback("progress_bar_tiles", 100, "Map tile caching complete")

        return {
            'route_time_per_frame': route_time_per_frame,
            'unique_bounding_boxes': unique_bounding_boxes,
            'cache_result': cache_result,
            'success': inner_ok,
            'preflight_exceeded': preflight_exceeded,
            'total_required_tiles': cache_result.get('total_required_tiles'),
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in map tiles caching: {str(e)}")
        return None
