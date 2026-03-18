"""
Shared utility functions used by video generator modules.
"""

import math


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
