"""
Follow-2D camera mode for Route Squiggler.

In follow_2d mode the camera follows the most recently drawn route point,
zoomed to a fixed spatial extent set by the video_zoom parameter (km diameter,
vertical span). Camera movement is smoothed with an Exponential Moving Average
so it continuously 'catches up' to the target instead of snapping, producing
smooth arcs through direction changes and graceful glides when the route jumps
between geographically separate tracks.

All per-frame bounding boxes are pre-computed once, sequentially (EMA requires
sequential state), and stored in combined_route_data['follow_2d_bboxes_per_frame']
before both map-tile caching (step 3) and frame rendering (step 5).  Workers in
step 5 simply look up their frame's pre-computed bbox — this guarantees the two
steps use bit-for-bit identical values.
"""

import math

from video_generator_utils import binary_search_cutoff_index, compute_sequential_ending_lengths

# Approximate km per degree of latitude (constant).
_KM_PER_DEG_LAT = 111.0


def bbox_from_follow_2d_center(lat, lon, video_zoom_km, video_resolution_x, video_resolution_y):
    """
    Compute a bounding box centred at (lat, lon) for follow_2d mode.

    The vertical span equals video_zoom_km.  The horizontal span is scaled
    to match the video's aspect ratio, accounting for latitude distortion so
    the on-screen appearance is consistent regardless of latitude.

    Args:
        lat (float): Centre latitude in degrees.
        lon (float): Centre longitude in degrees.
        video_zoom_km (float): Vertical diameter of the view in km.
        video_resolution_x (int): Video width in pixels.
        video_resolution_y (int): Video height in pixels.

    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) rounded to 6 decimal places.
    """
    aspect_ratio = video_resolution_x / video_resolution_y
    half_height_km = video_zoom_km / 2.0

    # Latitude radius is straightforward (degrees latitude are uniform).
    lat_radius = half_height_km / _KM_PER_DEG_LAT

    # Longitude degrees per km shrink toward the poles.
    cos_lat = max(0.001, abs(math.cos(math.radians(lat))))
    lon_radius = (half_height_km * aspect_ratio) / (_KM_PER_DEG_LAT * cos_lat)

    return (
        round(max(-180.0, lon - lon_radius), 6),
        round(min(180.0,  lon + lon_radius), 6),
        round(max(-90.0,  lat - lat_radius), 6),
        round(min(90.0,   lat + lat_radius), 6),
    )


def precompute_follow_2d_bboxes(json_data, combined_route_data, log_callback=None, debug_callback=None):
    """
    Pre-compute per-frame bounding boxes for follow_2d mode.

    Uses an Exponential Moving Average (EMA) so the camera always drifts
    toward the most recent route point rather than snapping to it.  With time
    constant tau the camera closes roughly 63 % of the remaining gap every tau
    seconds.  The default of 1.5 s gives smooth motion while remaining visually
    responsive; it can be overridden with the camera_smoothing job parameter.

    The result is a list indexed by [frame_number - 1] because the streaming
    generator uses 1-based frame numbers.  The list covers all frames including
    the tail / cloned-ending phase; the camera holds the final route position
    during those extra frames.

    Args:
        json_data (dict): Job data.
        combined_route_data (dict): Combined route data (must contain 'combined_route').
        log_callback (callable, optional): For error/warning messages.
        debug_callback (callable, optional): For verbose debug messages.

    Returns:
        list[tuple]: Per-frame bbox list, or None on error.
    """
    try:
        video_zoom_km       = float(json_data.get('video_zoom', 5.0))
        video_fps           = float(json_data.get('video_fps', 30))
        video_resolution_x  = int(json_data.get('video_resolution_x', 1920))
        video_resolution_y  = int(json_data.get('video_resolution_y', 1080))
        video_length        = float(json_data.get('video_length', 30))
        tail_length         = int(json_data.get('tail_length', 0))

        gpx_time_per_video_time = float(combined_route_data.get('gpx_time_per_video_time', 1.0))

        # EMA time constant (seconds).  Smaller = snappier camera.
        # Exposed as camera_smoothing so jobs can tune it if needed.
        tau = float(json_data.get('camera_smoothing', 1.5))
        dt  = 1.0 / video_fps
        # Per-frame smoothing coefficient derived from the continuous-time EMA formula.
        alpha = 1.0 - math.exp(-dt / tau)

        # Total frame count including tail and cloned-ending frames.
        extended_video_length, _, _ = compute_sequential_ending_lengths(video_length, tail_length)
        total_frames = int(extended_video_length * video_fps)

        # follow_2d is sequential-only — always uses combined_route.
        combined_route = combined_route_data.get('combined_route', [])
        if not combined_route:
            if log_callback:
                log_callback("Error: follow_2d mode requires a combined_route in combined_route_data")
            return None

        if debug_callback:
            debug_callback(
                f"Precomputing follow_2d bboxes: {total_frames} frames, "
                f"zoom={video_zoom_km} km, tau={tau} s, alpha/frame={alpha:.5f}"
            )

        first_point     = combined_route[0]
        last_route_time = combined_route[-1].accumulated_time

        # Initialise the EMA camera at the very first route point.
        camera_lat = first_point.lat
        camera_lon = first_point.lon

        bboxes = []

        for frame_number in range(1, total_frames + 1):
            target_time_video = (frame_number - 1) / video_fps
            target_time_route = target_time_video * gpx_time_per_video_time

            # During tail / cloned-ending phase the route is fully drawn;
            # hold the camera at the last point.
            target_time_route = min(target_time_route, last_route_time)

            # Most recent route point visible at this moment.
            cutoff = binary_search_cutoff_index(combined_route, target_time_route)
            if cutoff > 0:
                target_point = combined_route[cutoff - 1]
                target_lat   = target_point.lat
                target_lon   = target_point.lon
            else:
                target_lat = first_point.lat
                target_lon = first_point.lon

            # EMA update: camera drifts toward target.
            camera_lat = alpha * target_lat + (1.0 - alpha) * camera_lat
            camera_lon = alpha * target_lon + (1.0 - alpha) * camera_lon

            bbox = bbox_from_follow_2d_center(
                camera_lat, camera_lon,
                video_zoom_km,
                video_resolution_x, video_resolution_y,
            )
            bboxes.append(bbox)

        if debug_callback:
            unique_count = len(set(bboxes))
            debug_callback(
                f"follow_2d: precomputed {total_frames} frames → {unique_count} unique bboxes"
            )

        return bboxes

    except Exception as e:
        if log_callback:
            log_callback(f"Error in precompute_follow_2d_bboxes: {str(e)}")
        return None
