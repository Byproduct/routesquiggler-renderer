"""
Shared utility functions used by video generator modules.
"""

import math
from typing import NamedTuple


# Final pan-out (follow modes only): over the last N seconds before the goal
# the camera smoothly pans to the full-route view and any tilt / rotation fades
# to zero. The duration matches the existing 5-second tail + cloned-ending
# window so the zoomed-out view is already in place when fade-out begins.
FINAL_PAN_OUT_DURATION_SECONDS = 5.0

# Very short videos skip the pan-out entirely — the pan would otherwise eat
# most of the route animation. Matches the 10s threshold documented to users.
FINAL_PAN_OUT_MIN_VIDEO_LENGTH_SECONDS = 10.0


def smoothstep(t):
    """Smoothstep easing: 3t² - 2t³, with the input clamped to [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def lerp_bbox(bbox_a, bbox_b, t):
    """
    Linearly interpolate between two bboxes corner-by-corner.

    Both inputs are (lon_min, lon_max, lat_min, lat_max). The result is rounded
    to 6 decimals to match the precision used elsewhere in the pipeline, so
    frame-to-frame bbox values used as cache keys stay stable.
    """
    return tuple(round(a + (b - a) * float(t), 6) for a, b in zip(bbox_a, bbox_b))


def parse_final_pan_out_flag(json_data):
    """Parse json_data['final_pan_out'] into a bool (default True)."""
    raw = json_data.get('final_pan_out', True) if json_data else True
    if isinstance(raw, str):
        return raw.lower() in ('true', '1', 'yes')
    return bool(raw)


def should_apply_final_pan_out(json_data):
    """
    Return True when the final pan-out feature should be active for this job.

    Gated on both the ``final_pan_out`` job flag (default on) and a minimum
    ``video_length`` so very short videos don't spend most of their runtime
    in the pan. Called from every precompute path (bbox, rotation, tilt) so
    all three agree on when the effect kicks in.
    """
    if not parse_final_pan_out_flag(json_data):
        return False
    try:
        video_length = float(json_data.get('video_length', 0))
    except (TypeError, ValueError):
        return False
    return video_length >= FINAL_PAN_OUT_MIN_VIDEO_LENGTH_SECONDS


def final_pan_out_frame_window(video_length, video_fps):
    """
    Return ``(anchor_frame, goal_frame, pan_frames)`` for the final pan-out.

    Frame numbers are 1-based:
      * ``goal_frame``   — last frame of the main route phase (= ``video_length * fps``).
                           Its bbox / tilt / rotation equals the fully zoomed-out state.
      * ``anchor_frame`` — last frame before the pan begins. Its current
                           (follow) bbox / tilt / rotation is the 'from' value.
      * ``pan_frames``   — number of frames in the pan window
                           (anchor_frame+1 .. goal_frame, inclusive).
    """
    goal_frame = int(float(video_length) * float(video_fps))
    pan_frames = int(FINAL_PAN_OUT_DURATION_SECONDS * float(video_fps))
    anchor_frame = goal_frame - pan_frames
    return anchor_frame, goal_frame, pan_frames


def binary_search_cutoff_index(route_points, target_time):
    """
    Binary search to find the cutoff index for points up to target_time.

    Returns the index of the first point that should be EXCLUDED (i.e., where accumulated_time > target_time).
    This allows for efficient list slicing: route_points[:cutoff_index] gives all valid points.

    Args:
        route_points (list): List of RoutePoint objects, chronologically ordered by accumulated_time
        target_time (float): Target accumulated_time threshold

    Returns:
        int: Index where to cut off the points (exclusive), suitable for list slicing
    """
    if not route_points:
        return 0

    left, right = 0, len(route_points) - 1
    result = len(route_points)  # Default: include all points if none exceed target_time

    while left <= right:
        mid = (left + right) // 2

        # Use named attribute for accumulated_time
        accumulated_time = route_points[mid].accumulated_time

        if accumulated_time <= target_time:
            # This point should be included, look for later cutoff point
            left = mid + 1
        else:
            # This point should be excluded, it might be our cutoff point
            result = mid
            right = mid - 1

    return result


def is_simultaneous_mode(combined_route_data):
    """
    Detect if we're in simultaneous mode (multiple distinct routes) vs single route or split tracks.

    Simultaneous mode means multiple routes with different route_index values (truly different
    routes). A single route with multiple tracks (split) shares the same route_index.

    Args:
        combined_route_data (dict): Combined route data containing optional 'all_routes' list.

    Returns:
        bool: True if multiple distinct routes (simultaneous mode), False otherwise.
    """
    all_routes = combined_route_data.get('all_routes', None) if combined_route_data else None
    if not all_routes or len(all_routes) <= 1:
        return False
    route_indices = set()
    for route_data in all_routes:
        route_points = route_data.get('combined_route', [])
        if route_points:
            first_point = route_points[0]
            route_indices.add(first_point.route_index)
    return len(route_indices) > 1


def get_route_start_times(all_routes):
    """
    Build a map of route_data id -> start timestamp and find the earliest start time.
    Used for staggered/simultaneous route timing.

    Args:
        all_routes (list): List of route dicts, each with 'combined_route' (list of points with .timestamp).

    Returns:
        tuple: (route_start_times dict, earliest_start_time or None)
    """
    earliest_start_time = None
    route_start_times = {}
    for route_data in all_routes:
        route_points = route_data.get('combined_route', [])
        if route_points:
            route_start_timestamp = route_points[0].timestamp
            if route_start_timestamp:
                route_start_times[id(route_data)] = route_start_timestamp
                if earliest_start_time is None or route_start_timestamp < earliest_start_time:
                    earliest_start_time = route_start_timestamp
    return route_start_times, earliest_start_time


def get_route_delay_seconds(route_data, route_start_times, earliest_start_time):
    """
    Return this route's start delay in seconds relative to the earliest route.
    Used for staggered/simultaneous route timing.

    Args:
        route_data: Route dict (used as key via id(route_data)).
        route_start_times (dict): Map from id(route_data) to start timestamp.
        earliest_start_time: Earliest start time across all routes.

    Returns:
        float: Delay in seconds (0.0 if not found or no earliest).
    """
    route_id = id(route_data)
    if route_id not in route_start_times or not earliest_start_time:
        return 0.0
    route_start_timestamp = route_start_times[route_id]
    return (route_start_timestamp - earliest_start_time).total_seconds()


def compute_sequential_ending_lengths(video_length, tail_length):
    """
    Compute extended_video_length, final_video_length, and cloned_ending_duration for sequential mode.
    Ending is always 5 seconds total: tail_length seconds of tail frames plus cloned frames for the rest.

    Args:
        video_length (float): Duration of the main route video in seconds.
        tail_length (int): Tail fade-out duration in seconds (from json_data tail_length).

    Returns:
        tuple: (extended_video_length, final_video_length, cloned_ending_duration)
    """
    ending_duration_required = 5.0
    cloned_ending_duration = max(0.0, ending_duration_required - tail_length)
    extended_video_length = video_length + tail_length if tail_length > 0 else video_length
    final_video_length = extended_video_length + cloned_ending_duration
    return extended_video_length, final_video_length, cloned_ending_duration


def compute_simultaneous_ending_lengths(video_length, tail_length=0):
    """
    Compute extended_video_length, final_video_length, and cloned_ending_duration for simultaneous mode.
    Each route handles its own tail; we extend by tail_length for the last route's fade-out (if any),
    then add 5 seconds of cloned ending.

    Args:
        video_length (float): Duration of the main route video in seconds.
        tail_length (int): Tail fade-out duration in seconds for the last route (0 if no tail phase).

    Returns:
        tuple: (extended_video_length, final_video_length, cloned_ending_duration)
    """
    cloned_ending_duration = 5.0
    extended_video_length = video_length + tail_length
    final_video_length = extended_video_length + cloned_ending_duration
    return extended_video_length, final_video_length, cloned_ending_duration


def compute_sequential_frames_to_skip(combined_route, route_time_per_frame, frames_to_generate):
    """
    Compute how many leading frames to skip in sequential mode when the route has
    insufficient geographically distinct points (e.g. interpolated points with same lat/lon).

    In sequential mode, early frames may not contain enough distinct points to draw
    a meaningful line; skipping them avoids "vertical slice" artifacts and saves CPU.

    Args:
        combined_route (list): List of route points with .lat, .lon, .accumulated_time.
        route_time_per_frame (float): Route seconds per frame (must be > 0 for non-zero skip).
        frames_to_generate (int): Total number of frames that would be generated.

    Returns:
        int: Number of leading frames to skip (0 if simultaneous or no skip needed).
    """
    if not combined_route or len(combined_route) < 2:
        return max(0, frames_to_generate - 1)
    if route_time_per_frame <= 0:
        return 0
    first_lat = combined_route[0].lat
    first_lon = combined_route[0].lon
    first_diverse_time = None
    for pt in combined_route[1:]:
        if abs(pt.lat - first_lat) > 1e-9 or abs(pt.lon - first_lon) > 1e-9:
            first_diverse_time = pt.accumulated_time
            break
    if first_diverse_time is not None and first_diverse_time > 0:
        first_drawable_frame = math.ceil(first_diverse_time / route_time_per_frame) + 1
        return max(0, min(first_drawable_frame - 1, frames_to_generate - 1))
    if first_diverse_time is None:
        # All points at the same location - skip all but the last frame
        return max(0, frames_to_generate - 1)
    return 0


class FramePointsContext(NamedTuple):
    """
    Per-frame values derived from the frame number, shared between the
    streaming frame worker (live frame generation, Step 5) and the dynamic
    bounding-box precompute (Step 2.5).

    Using a single source of truth for these values guarantees that Step 3
    (map-tile caching), Step 4 (map-image caching) and Step 5 (frame render)
    all agree on which points belong to which frame, so the per-frame bbox
    used to key the cache is bit-for-bit identical across phases.

    Attributes:
        points_for_frame: Animated route points for this frame.  In
            simultaneous multi-route mode this is a list of sub-lists (one
            per route); otherwise it is a flat list of RoutePoint.
            ``generate_video_frame_in_memory`` normalises both shapes to a
            list-of-lists before use.
        target_time_route: Route-time (in seconds) at this frame, already
            clamped for simultaneous-mode effective video length.
        is_tail_only_frame: Whether this frame is past the original route
            end (sequential tail / cloned-ending phase).  Forced to False in
            simultaneous mode where each route handles its own tail.
        effective_video_length: Video duration used for frame-index math,
            possibly shorter than json_data['video_length'] in simultaneous
            mode when the latest-starting route can finish earlier.
        original_route_end_frame: Last frame number belonging to the main
            route (before tail / cloned-ending frames).
        hide_complete_routes: Parsed boolean form of
            json_data['hide_complete_routes'].
    """

    points_for_frame: list
    target_time_route: float
    is_tail_only_frame: bool
    effective_video_length: float
    original_route_end_frame: int
    hide_complete_routes: bool


def _parse_hide_complete_routes(json_data):
    """Parse json_data['hide_complete_routes'] into a bool (accepts str or bool)."""
    raw = json_data.get('hide_complete_routes', False) if json_data else False
    if isinstance(raw, str):
        return raw.lower() in ('true', '1', 'yes')
    return bool(raw)


def compute_frame_points_context(
    frame_number,
    combined_route_data,
    json_data,
    is_simultaneous_mode_flag,
    gpx_time_per_video_time,
    debug_log_callback=None,
):
    """
    Compute the animated points for a single frame and the timing context
    around it, using the same logic the streaming frame worker uses live.

    Intended callers:
      * ``_streaming_frame_worker`` (Step 5) — passes a debug callback so
        the historical ``frame_number <= 3`` debug log lines keep firing.
      * ``precompute_dynamic_bboxes`` (Step 2.5) — passes no callback so
        the precompute is silent and produces no duplicate log output.

    Args:
        frame_number (int): 1-based frame index.
        combined_route_data (dict): Route data containing 'combined_route'
            and optionally 'all_routes'.
        json_data (dict): Job data with video_length, video_fps, tail_length,
            hide_complete_routes.
        is_simultaneous_mode_flag (bool): True when multiple distinct routes
            are animated concurrently.
        gpx_time_per_video_time (float): GPX-seconds per video-second.
        debug_log_callback (callable | None): If provided, invoked with
            ``(message: str)`` for the three historical frame-scoped debug
            messages (all gated on ``frame_number <= 3``).  Pass ``None``
            to silence.

    Returns:
        FramePointsContext
    """
    video_length = float(json_data.get('video_length', 30))
    video_fps = float(json_data.get('video_fps', 30))
    tail_length = int(json_data.get('tail_length', 0))

    target_time_video = (frame_number - 1) / video_fps
    target_time_route = (
        target_time_video * gpx_time_per_video_time
        if gpx_time_per_video_time and gpx_time_per_video_time > 0
        else 0
    )

    all_routes = combined_route_data.get('all_routes', None) if combined_route_data else None

    # SIMULTANEOUS MODE: shorten the effective video length so the latest-starting
    # route can finish.  Mirrors the logic in _streaming_frame_worker.
    effective_video_length = video_length
    if is_simultaneous_mode_flag and all_routes and len(all_routes) > 1:
        earliest = None
        latest = None
        max_route_duration = 0.0
        for route_data in all_routes:
            route_points = route_data.get('combined_route', [])
            if route_points:
                first_point = route_points[0]
                last_point = route_points[-1]
                start_ts = first_point.timestamp
                if start_ts:
                    if earliest is None or start_ts < earliest:
                        earliest = start_ts
                    if latest is None or start_ts > latest:
                        latest = start_ts
                        max_route_duration = last_point.accumulated_time
        if earliest and latest:
            delay_from_earliest = (latest - earliest).total_seconds()
            min_needed = delay_from_earliest + max_route_duration
            effective_video_length = min(min_needed, video_length)
            effective_target_time_video = min(target_time_video, effective_video_length)
            target_time_route = (
                effective_target_time_video * gpx_time_per_video_time
                if gpx_time_per_video_time and gpx_time_per_video_time > 0
                else 0
            )

    original_route_end_frame = int(effective_video_length * video_fps)
    is_tail_only_frame = frame_number > original_route_end_frame
    hide_complete_routes = _parse_hide_complete_routes(json_data)

    # Simultaneous mode: each route fades its own tail, no global tail phase.
    if is_simultaneous_mode_flag and is_tail_only_frame:
        is_tail_only_frame = False

    debug_enabled = debug_log_callback is not None and frame_number <= 3

    if is_simultaneous_mode_flag and all_routes and len(all_routes) > 1:
        # Simultaneous multi-route: list of sub-lists, one per active route.
        points_for_frame = []
        route_start_times, earliest_start_time = get_route_start_times(all_routes)

        for route_data in all_routes:
            route_points = route_data.get('combined_route', [])
            route_delay_seconds = get_route_delay_seconds(
                route_data, route_start_times, earliest_start_time
            )
            route_target_time = target_time_route - route_delay_seconds

            route_end_time = route_points[-1].accumulated_time if route_points else 0
            route_is_complete = route_points and route_target_time >= route_end_time

            if hide_complete_routes and route_is_complete:
                if debug_enabled:
                    debug_log_callback(
                        f"Frame {frame_number}: Skipping complete route (simultaneous mode)"
                    )
                continue

            if is_tail_only_frame:
                # Per-route tail fade-out window.  (Not reached because the
                # simultaneous-mode override above forces is_tail_only_frame=False,
                # but kept for parity with the streaming worker.)
                route_end_time_with_delay = route_end_time + route_delay_seconds
                time_since_route_end = target_time_route - route_end_time_with_delay
                tail_duration_route = (
                    gpx_time_per_video_time * tail_length if gpx_time_per_video_time else 0
                )
                if 0 <= time_since_route_end <= tail_duration_route:
                    cutoff_index = binary_search_cutoff_index(route_points, route_end_time)
                    route_points_for_frame = route_points[:cutoff_index]
                else:
                    route_points_for_frame = []
            else:
                if route_target_time >= 0:
                    cutoff_index = binary_search_cutoff_index(route_points, route_target_time)
                    route_points_for_frame = route_points[:cutoff_index]
                else:
                    route_points_for_frame = []

            if route_points_for_frame:
                points_for_frame.append(route_points_for_frame)

        return FramePointsContext(
            points_for_frame=points_for_frame,
            target_time_route=target_time_route,
            is_tail_only_frame=is_tail_only_frame,
            effective_video_length=effective_video_length,
            original_route_end_frame=original_route_end_frame,
            hide_complete_routes=hide_complete_routes,
        )

    # Single-route mode (or single route group with multiple tracks).
    combined_route = combined_route_data.get('combined_route', []) if combined_route_data else []

    # `filter_time` is the reference point that hide_complete_routes uses to
    # decide whether a file is "already behind us".  For normal frames this is
    # the live target time.  For tail-only frames the cutoff is frozen at the
    # last-normal-frame time, and the filter uses that same frozen value so
    # the set of visible files doesn't change during the tail phase.
    if is_tail_only_frame:
        final_route_target_time = ((original_route_end_frame - 1) / video_fps) * gpx_time_per_video_time
        if final_route_target_time >= 0:
            cutoff_index = binary_search_cutoff_index(combined_route, final_route_target_time)
            points_for_frame = combined_route[:cutoff_index]
        else:
            points_for_frame = []
        filter_time = final_route_target_time
    else:
        cutoff_index = binary_search_cutoff_index(combined_route, target_time_route)
        points_for_frame = combined_route[:cutoff_index]
        filter_time = target_time_route

    # hide_complete_routes: drop points whose file's final point is already
    # behind us AND whose file has been superseded by a later one.
    #
    # Naively using `last_time <= filter_time` alone would hide the file that
    # carries the route's overall final point as soon as we reach (or float
    # past) the route end -- because route_time_per_frame is computed so that
    # target_time_route on the last normal frame lands exactly on
    # total_accumulated_time.  That leaves the tail-only phase and the cloned
    # ending with empty (black) frames because nothing is drawn.  We protect
    # the finale file(s) by excluding any file whose last_time equals the
    # latest last_time across all files from the "completed" set -- they have
    # no successor to hand off to, so there is nothing to hide them in favour
    # of.  Files with gaps between them still get hidden exactly as before
    # (intermediate files' last_time is strictly less than the finale's).
    if hide_complete_routes and points_for_frame:
        filename_to_last_time = {}
        for point in combined_route:
            filename = getattr(point, 'filename', None)
            if filename:
                existing = filename_to_last_time.get(filename)
                if existing is None or point.accumulated_time > existing:
                    filename_to_last_time[filename] = point.accumulated_time

        route_final_time = (
            max(filename_to_last_time.values()) if filename_to_last_time else 0.0
        )

        completed_filenames = set()
        for filename, last_time in filename_to_last_time.items():
            # Skip the file(s) that carry the route's final point -- they
            # have no successor and must stay visible so the tail can fade.
            if last_time >= route_final_time:
                continue
            if last_time <= filter_time:
                completed_filenames.add(filename)
                if debug_enabled:
                    debug_log_callback(
                        f"Frame {frame_number}: File '{filename}' is complete "
                        f"(last point time: {last_time:.2f}, filter time: {filter_time:.2f})"
                    )

        if completed_filenames:
            original_count = len(points_for_frame)
            points_for_frame = [
                p for p in points_for_frame
                if not (getattr(p, 'filename', None) in completed_filenames)
            ]
            if debug_enabled:
                filtered_count = len(points_for_frame)
                debug_log_callback(
                    f"Frame {frame_number}: Filtered out {original_count - filtered_count} "
                    f"points from {len(completed_filenames)} completed file(s). "
                    f"Remaining: {filtered_count} points"
                )

    return FramePointsContext(
        points_for_frame=points_for_frame,
        target_time_route=target_time_route,
        is_tail_only_frame=is_tail_only_frame,
        effective_video_length=effective_video_length,
        original_route_end_frame=original_route_end_frame,
        hide_complete_routes=hide_complete_routes,
    )
