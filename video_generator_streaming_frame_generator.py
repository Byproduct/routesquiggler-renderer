"""
Streaming frame generation for MoviePy.

This module contains the multiprocessing-based StreamingFrameGenerator used by
video creation/caching code.
"""

# Standard library imports
import gc
import multiprocessing
from multiprocessing import Pool
import os
import pickle
import tempfile
import time

# Third-party imports (matplotlib backend must be set before pyplot import)
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for frame generation

import matplotlib.pyplot as plt
import numpy as np

# Local imports
from config import config
from image_generator_utils import calculate_resolution_scale
from video_generator_create_single_frame import generate_video_frame_in_memory
from video_generator_create_single_frame_utils import hex_to_rgba
from video_generator_utils import (
    binary_search_cutoff_index,
    compute_sequential_ending_lengths,
    compute_sequential_frames_to_skip,
    compute_simultaneous_ending_lengths,
    get_route_delay_seconds,
    get_route_start_times,
    is_simultaneous_mode,
)
from write_log import write_debug_log, write_log


def _timestamp_to_accumulated_time(poi_timestamp, combined_route):
    """
    Convert a POI timestamp to an equivalent accumulated_time by finding where it falls
    in the route's timeline and interpolating.

    This is necessary because accumulated_time in the route excludes gaps between GPX files
    (where recording was paused), while POI timestamps are absolute. By interpolating,
    we find the "route time" that corresponds to the POI's wall-clock time.

    Args:
        poi_timestamp: datetime object representing the POI's timestamp
        combined_route: List of RoutePoint objects with timestamps and accumulated_time

    Returns:
        float: Equivalent accumulated_time, or None if timestamp can't be mapped
    """
    if not combined_route or poi_timestamp is None:
        return None

    # Find the first point with a valid timestamp to use as reference
    first_valid_point = None
    for point in combined_route:
        if point.timestamp is not None:
            first_valid_point = point
            break

    if first_valid_point is None:
        return None

    # If POI timestamp is before or at the first point, return 0
    if poi_timestamp <= first_valid_point.timestamp:
        return 0.0

    # Find the last point with a valid timestamp
    last_valid_point = None
    for point in reversed(combined_route):
        if point.timestamp is not None:
            last_valid_point = point
            break

    # If POI timestamp is at or after the last point, return the last accumulated_time
    if last_valid_point and poi_timestamp >= last_valid_point.timestamp:
        return last_valid_point.accumulated_time

    # Binary search to find the two points that bracket the POI timestamp
    # Then interpolate between them
    prev_point = None
    for point in combined_route:
        if point.timestamp is None:
            continue

        if point.timestamp >= poi_timestamp:
            if prev_point is None:
                # POI is at or before the first point
                return point.accumulated_time

            # Interpolate between prev_point and point
            time_span = (point.timestamp - prev_point.timestamp).total_seconds()
            if time_span <= 0:
                return prev_point.accumulated_time

            poi_offset = (poi_timestamp - prev_point.timestamp).total_seconds()
            fraction = poi_offset / time_span

            accumulated_time_span = point.accumulated_time - prev_point.accumulated_time
            return prev_point.accumulated_time + (fraction * accumulated_time_span)

        prev_point = point

    # Fallback: return the last point's accumulated_time
    return last_valid_point.accumulated_time if last_valid_point else None


def _save_shared_worker_data(data_dict, job_id, temp_dir=None):
    """
    Save shared worker data to a pickle file to avoid redundant pickling for each worker.

    Args:
        data_dict: Dictionary containing shared data (combined_route_data, json_data, etc.)
        job_id: Job ID for creating unique filename
        temp_dir: Optional temporary directory path (defaults to system temp)

    Returns:
        str: Path to the pickle file
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Create job-specific filename
    filename = f"route_squiggler_worker_data_{job_id}.pkl"
    filepath = os.path.join(temp_dir, filename)

    # Save data to pickle file
    with open(filepath, "wb") as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return filepath


def _load_shared_worker_data(filepath):
    """
    Load shared worker data from a pickle file.

    Args:
        filepath: Path to the pickle file

    Returns:
        dict: Dictionary containing shared data
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def _cleanup_shared_worker_data(filepath):
    """
    Clean up the shared worker data pickle file.

    Args:
        filepath: Path to the pickle file to delete
    """
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        # Log but don't fail if cleanup fails
        write_debug_log(f"Warning: Failed to cleanup worker data file {filepath}: {e}")


def _streaming_frame_worker(args):
    """
    Worker function for generating a single frame and returning it as numpy array
    instead of saving to disk. Supports both single route and multiple routes modes.
    OPTIMIZED: Better memory management and performance for computationally intensive tasks.
    OPTIMIZED: Uses binary search + list slicing instead of linear search + individual appends.
    OPTIMIZED: Loads shared data from pickle file to avoid redundant pickling.
    """
    # Args format: (frame_number, target_time_video, shared_data_filepath)
    frame_number, target_time_video, shared_data_filepath = args

    # Load shared data from pickle file
    shared_data = _load_shared_worker_data(shared_data_filepath)
    json_data = shared_data["json_data"]
    route_time_per_frame = shared_data["route_time_per_frame"]
    combined_route_data = shared_data["combined_route_data"]
    shared_map_cache = shared_data["shared_map_cache"]
    filename_to_rgba = shared_data["filename_to_rgba"]
    gpx_time_per_video_time = shared_data["gpx_time_per_video_time"]
    stamp_array = shared_data["stamp_array"]
    shared_route_cache = shared_data["shared_route_cache"]
    is_simultaneous_mode_flag = shared_data.get("is_simultaneous_mode", False)

    try:
        # Ensure frame_number is an integer
        frame_number = int(frame_number)

        # The caching logic needs video seconds, but point collection needs route seconds
        target_time_route = (
            target_time_video * gpx_time_per_video_time if gpx_time_per_video_time > 0 else 0
        )

        # Check if we have multiple routes
        all_routes = combined_route_data.get("all_routes", None)

        # Calculate the frame number where the original route ends
        video_length = float(json_data.get("video_length", 30))
        video_fps = float(json_data.get("video_fps", 30))
        tail_length = int(json_data.get("tail_length", 0))  # Ensure tail_length is int

        # SIMULTANEOUS MODE FIX: Calculate effective video duration for staggered routes (flag from shared_data)
        effective_video_length = video_length
        if is_simultaneous_mode_flag and all_routes and len(all_routes) > 1:
            # Find the earliest start time and latest start time across all routes
            earliest_start_time = None
            latest_start_time = None
            max_route_duration = 0.0

            for route_data in all_routes:
                route_points = route_data.get("combined_route", [])
                if route_points and len(route_points) > 0:
                    first_point = route_points[0]
                    last_point = route_points[-1]

                    route_start_timestamp = first_point.timestamp
                    route_duration = last_point.accumulated_time

                    if route_start_timestamp:
                        if (
                            earliest_start_time is None
                            or route_start_timestamp < earliest_start_time
                        ):
                            earliest_start_time = route_start_timestamp

                        if latest_start_time is None or route_start_timestamp > latest_start_time:
                            latest_start_time = route_start_timestamp
                            max_route_duration = route_duration

            # Calculate the minimum video duration needed for all routes to complete
            if earliest_start_time and latest_start_time:
                delay_from_earliest = (latest_start_time - earliest_start_time).total_seconds()
                min_video_duration_needed = delay_from_earliest + max_route_duration

                # Use the minimum video duration needed, but don't exceed the actual video duration
                effective_video_length = min(min_video_duration_needed, video_length)

                # Adjust target_time_route to use effective video duration
                effective_target_time_video = min(target_time_video, effective_video_length)
                target_time_route = (
                    effective_target_time_video * gpx_time_per_video_time
                    if gpx_time_per_video_time > 0
                    else 0
                )

        # Calculate the frame number where the original route ends (using effective video length)
        original_route_end_frame = int(effective_video_length * video_fps)

        # Check if we're in the tail-only phase (extra frames after original route ends)
        is_tail_only_frame = frame_number > original_route_end_frame

        # Get hide_complete_routes parameter (defaults to False if not specified)
        hide_complete_routes_raw = json_data.get("hide_complete_routes", False)
        if isinstance(hide_complete_routes_raw, str):
            hide_complete_routes = hide_complete_routes_raw.lower() in ("true", "1", "yes")
        else:
            hide_complete_routes = bool(hide_complete_routes_raw)

        # SIMULTANEOUS MODE ENDING FIX: Use different logic for simultaneous mode ending
        if is_simultaneous_mode_flag and is_tail_only_frame:
            is_tail_only_frame = False  # Treat as normal frame for simultaneous mode

        if is_simultaneous_mode_flag and all_routes and len(all_routes) > 1:
            # Multiple routes mode - create a list of sub-lists for each route
            points_for_frame = []

            # STAGGERED ROUTES FIX: Calculate route start delays
            route_start_times, earliest_start_time = get_route_start_times(all_routes)

            for route_data in all_routes:
                route_points = route_data.get("combined_route", [])

                route_delay_seconds = get_route_delay_seconds(
                    route_data, route_start_times, earliest_start_time
                )

                # Calculate route-specific target time accounting for start delay
                route_target_time = target_time_route - route_delay_seconds

                # Check if route is complete (all points included)
                route_end_time = route_points[-1].accumulated_time if route_points else 0
                route_is_complete = route_points and route_target_time >= route_end_time

                # Skip complete routes if hide_complete_routes is enabled
                if hide_complete_routes and route_is_complete:
                    if frame_number <= 3 and config.debug_logging:
                        write_debug_log(
                            f"Frame {frame_number}: Skipping complete route (simultaneous mode)"
                        )
                    continue

                if is_tail_only_frame:
                    # SIMULTANEOUS MODE TAIL FIX: Each route should have its own individual tail fade-out
                    route_end_time_with_delay = route_end_time + route_delay_seconds

                    current_route_time = target_time_route
                    time_since_route_end = current_route_time - route_end_time_with_delay

                    tail_duration_route = (
                        gpx_time_per_video_time * tail_length if gpx_time_per_video_time else 0
                    )

                    if time_since_route_end >= 0 and time_since_route_end <= tail_duration_route:
                        # Route has ended and we're within tail duration - show fading tail
                        cutoff_index = binary_search_cutoff_index(route_points, route_end_time)
                        route_points_for_frame = route_points[:cutoff_index]
                    else:
                        route_points_for_frame = []
                else:
                    # Normal frame - include points only if this route should have started
                    if route_target_time >= 0:
                        cutoff_index = binary_search_cutoff_index(route_points, route_target_time)
                        route_points_for_frame = route_points[:cutoff_index]
                    else:
                        route_points_for_frame = []

                if route_points_for_frame:
                    points_for_frame.append(route_points_for_frame)
        else:
            # Single route mode (backward compatibility)
            combined_route = combined_route_data.get("combined_route", [])

        # Handle tail-only frames and normal frames for single route mode
        if not all_routes or len(all_routes) <= 1:
            if is_tail_only_frame:
                final_route_target_time = ((original_route_end_frame - 1) / video_fps) * gpx_time_per_video_time

                if final_route_target_time >= 0:
                    cutoff_index = binary_search_cutoff_index(combined_route, final_route_target_time)
                    points_for_frame = combined_route[:cutoff_index]
                else:
                    points_for_frame = []
            else:
                cutoff_index = binary_search_cutoff_index(combined_route, target_time_route)
                points_for_frame = combined_route[:cutoff_index]

            # Apply hide_complete_routes filtering if enabled
            if hide_complete_routes and points_for_frame:
                filename_to_last_time = {}

                for point in combined_route:
                    if hasattr(point, "filename") and point.filename:
                        filename = point.filename
                        if (
                            filename not in filename_to_last_time
                            or point.accumulated_time > filename_to_last_time[filename]
                        ):
                            filename_to_last_time[filename] = point.accumulated_time

                completed_filenames = set()
                for filename, last_time in filename_to_last_time.items():
                    if last_time <= target_time_route:
                        completed_filenames.add(filename)
                        if frame_number <= 3 and config.debug_logging:
                            write_debug_log(
                                f"Frame {frame_number}: File '{filename}' is complete (last point time: {last_time:.2f}, target: {target_time_route:.2f})"
                            )

                if completed_filenames:
                    original_count = len(points_for_frame)
                    points_for_frame = [
                        p
                        for p in points_for_frame
                        if not (
                            hasattr(p, "filename")
                            and p.filename
                            and p.filename in completed_filenames
                        )
                    ]
                    filtered_count = len(points_for_frame)

                    if frame_number <= 3 and config.debug_logging:
                        write_debug_log(
                            f"Frame {frame_number}: Filtered out {original_count - filtered_count} points from {len(completed_filenames)} completed file(s). Remaining: {filtered_count} points"
                        )

        # Check if we have enough geographically distinct points to draw lines.
        has_drawable_content = False
        if all_routes and len(all_routes) > 1:
            has_drawable_content = any(len(route_points) > 0 for route_points in points_for_frame)
            if hide_complete_routes and frame_number <= 10 and config.debug_logging:
                route_count = len(points_for_frame)
                total_points = sum(len(route_points) for route_points in points_for_frame)
                write_debug_log(
                    f"Frame {frame_number} FINAL: {route_count} routes in points_for_frame, {total_points} total points, has_drawable_content={has_drawable_content}"
                )
        else:
            if len(points_for_frame) >= 2:
                ref_lat = points_for_frame[0].lat
                ref_lon = points_for_frame[0].lon
                has_drawable_content = any(
                    abs(p.lat - ref_lat) > 1e-9 or abs(p.lon - ref_lon) > 1e-9
                    for p in points_for_frame[1:]
                )

        if not has_drawable_content:
            if frame_number <= 10 and config.debug_logging:
                point_count = (
                    sum(len(rp) for rp in points_for_frame)
                    if (all_routes and len(all_routes) > 1)
                    else len(points_for_frame)
                )
                write_debug_log(
                    f"Frame {frame_number}: No geographically distinct points ({point_count} total), skipping frame"
                )
            return frame_number, None

        # Validate that shared map cache is provided
        if shared_map_cache is None:
            print("Error: Shared map cache is required for frame generation")
            return None

        # Pre-clear matplotlib state to reduce memory pressure
        plt.close("all")

        # Calculate virtual leading time for tail-only frames (progressive forward movement)
        virtual_leading_time = None
        route_specific_tail_info = {}

        # Calculate route-specific tail info for ALL frames, not just tail-only frames
        if is_simultaneous_mode_flag and all_routes and len(all_routes) > 1:
            route_start_times, earliest_start_time = get_route_start_times(all_routes)

            for route_data in all_routes:
                route_points = route_data.get("combined_route", [])
                if not route_points:
                    continue

                route_delay_seconds = get_route_delay_seconds(
                    route_data, route_start_times, earliest_start_time
                )

                route_end_time = route_points[-1].accumulated_time if route_points else 0
                route_end_time_with_delay = route_end_time + route_delay_seconds

                route_index = route_points[0].route_index if route_points else 0
                route_specific_tail_info[route_index] = {
                    "route_end_time": route_end_time_with_delay,
                    "route_delay_seconds": route_delay_seconds,
                }

                if is_tail_only_frame:
                    tail_only_frame_offset = frame_number - original_route_end_frame
                    route_virtual_leading_time = route_end_time_with_delay + (
                        tail_only_frame_offset * route_time_per_frame
                    )
                    route_specific_tail_info[route_index][
                        "virtual_leading_time"
                    ] = route_virtual_leading_time

        if is_tail_only_frame:
            tail_only_frame_offset = frame_number - original_route_end_frame
            route_end_time = (original_route_end_frame - 1) * route_time_per_frame
            virtual_leading_time = route_end_time + (tail_only_frame_offset * route_time_per_frame)

        # Filter points of interest for this frame
        points_of_interest_for_frame = []
        poi_setting = json_data.get("points_of_interest", "off")
        if poi_setting in ["light", "dark"]:
            all_pois = json_data.get("_points_of_interest_data", [])

            combined_route_for_poi = combined_route_data.get("combined_route", [])
            if not combined_route_for_poi and all_routes:
                combined_route_for_poi = (
                    all_routes[0].get("combined_route", []) if all_routes else []
                )

            for poi in all_pois:
                if poi.timestamp is None:
                    points_of_interest_for_frame.append(poi)
                else:
                    poi_accumulated_time = _timestamp_to_accumulated_time(
                        poi.timestamp, combined_route_for_poi
                    )
                    if poi_accumulated_time is not None and poi_accumulated_time <= target_time_route:
                        points_of_interest_for_frame.append(poi)

        frame_array = generate_video_frame_in_memory(
            frame_number,
            points_for_frame,
            json_data,
            shared_map_cache,
            filename_to_rgba,
            gpx_time_per_video_time,
            stamp_array,
            target_time_video,
            shared_route_cache,
            virtual_leading_time,
            route_specific_tail_info,
            points_of_interest_for_frame,
        )

        # MEMORY MANAGEMENT: Aggressive cleanup to prevent memory accumulation
        if frame_number % 3 == 0:
            plt.close("all")
            if hasattr(plt, "_pylab_helpers"):
                plt._pylab_helpers.Gcf.destroy_all()
            gc.collect(2)

        return frame_number, frame_array

    except Exception as e:
        print(f"Error generating frame {frame_number}: {e}")
        try:
            plt.close("all")
            gc.collect()
        except Exception:
            pass
        return frame_number, None


class StreamingFrameGenerator:
    """
    Generator class that yields frames for moviepy using multiprocessing.
    Uses an improved producer-consumer pattern with adaptive buffering and continuous frame generation.
    """

    def __init__(
        self,
        json_data,
        route_time_per_frame,
        combined_route_data,
        max_workers=None,
        shared_map_cache=None,
        shared_route_cache=None,
        progress_callback=None,
        gpx_time_per_video_time=None,
        debug_callback=None,
        user=None,
        simultaneous_mode=None,
    ):
        self.json_data = json_data
        self.route_time_per_frame = route_time_per_frame
        self.combined_route_data = combined_route_data
        if simultaneous_mode is None:
            simultaneous_mode = is_simultaneous_mode(combined_route_data)
        self.is_simultaneous_mode = simultaneous_mode
        self.shared_map_cache = shared_map_cache
        self.shared_route_cache = shared_route_cache
        self.progress_callback = progress_callback
        self.gpx_time_per_video_time = gpx_time_per_video_time
        self.debug_callback = debug_callback
        self.user = user
        self.job_id = str(json_data.get("job_id", "")) if json_data else ""

        # Pre-compute filename-to-RGBA color mapping once for all frames
        self.filename_to_rgba = {}
        if "track_objects" in json_data:
            for track_obj in json_data["track_objects"]:
                filename = track_obj["filename"]
                filename = os.path.basename(filename)
                if filename.endswith(".gpx"):
                    filename = filename[:-4]

                try:
                    rgba_color = hex_to_rgba(track_obj["color"])
                except Exception:
                    rgba_color = (1.0, 0.0, 0.0, 1.0)  # Default red

                self.filename_to_rgba[filename] = rgba_color

        # Pre-load stamp array once for all frames (only when stamp is enabled via json_data)
        self.stamp_array = None
        if json_data.get("stamp", False):
            try:
                width = int(json_data.get("video_resolution_x", 1920))
                height = int(json_data.get("video_resolution_y", 1080))

                image_scale = calculate_resolution_scale(width, height)

                if image_scale < 1:
                    stamp_file = "img/stamp_small.npy"
                elif image_scale == 1:
                    stamp_file = "img/stamp_1x.npy"
                elif image_scale == 2:
                    stamp_file = "img/stamp_2x.npy"
                elif image_scale == 3:
                    stamp_file = "img/stamp_3x.npy"
                elif image_scale >= 4:
                    stamp_file = "img/stamp_4x.npy"
                else:
                    stamp_file = "img/stamp_1x.npy"

                self.stamp_array = np.load(stamp_file)
                write_debug_log(f"Loaded stamp: {stamp_file} (scale: {image_scale})")
            except Exception as e:
                write_log(f"Warning: Could not load stamp: {e}")

        # Calculate video parameters
        video_length = float(json_data.get("video_length", 30))
        video_fps = float(json_data.get("video_fps", 30))

        # Calculate ending parameters based on mode
        if self.is_simultaneous_mode:
            tail_length = int(json_data.get("tail_length", 0))
            extended_video_length, final_video_length, cloned_ending_duration = (
                compute_simultaneous_ending_lengths(video_length, tail_length)
            )

            if progress_callback:
                progress_callback(
                    "progress_bar_frames",
                    0,
                    f"SIMULTANEOUS MODE: Creating video: {video_length}s route + {tail_length}s tail + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total",
                )
                progress_callback(
                    "progress_bar_frames",
                    0,
                    f"Each route handles its own completion and tail fade-out with {tail_length}s fade-out for the last route",
                )
        else:
            tail_length = int(json_data.get("tail_length", 0))
            extended_video_length, final_video_length, cloned_ending_duration = (
                compute_sequential_ending_lengths(video_length, tail_length)
            )

            if progress_callback:
                if tail_length > 0 and cloned_ending_duration > 0:
                    progress_callback(
                        "progress_bar_frames",
                        0,
                        f"SEQUENTIAL MODE: Creating video: {video_length}s route + {tail_length}s tail + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total",
                    )
                elif tail_length > 0:
                    progress_callback(
                        "progress_bar_frames",
                        0,
                        f"SEQUENTIAL MODE: Creating video: {video_length}s route + {tail_length}s tail = {final_video_length}s total (no additional frames needed)",
                    )
                else:
                    progress_callback(
                        "progress_bar_frames",
                        0,
                        f"SEQUENTIAL MODE: Creating video: {video_length}s route + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total",
                    )

        # Calculate frame counts
        original_frames = int(video_length * video_fps)
        tail_frames = int(tail_length * video_fps) if tail_length > 0 else 0
        cloned_frames = int(cloned_ending_duration * video_fps)

        self.total_frames = int(final_video_length * video_fps)
        self.original_frames = original_frames
        self.tail_frames = tail_frames
        self.cloned_frames = cloned_frames
        self.extended_video_length = extended_video_length
        self.final_video_length = final_video_length

        # Determine number of workers
        if max_workers is None:
            max_workers = min(4, multiprocessing.cpu_count())
        self.num_workers = min(max_workers, self.total_frames)

        write_debug_log(f"Using {self.num_workers} worker processes for frame generation")

        # Log the ending structure for user feedback
        if progress_callback:
            if tail_length > 0 and cloned_ending_duration > 0:
                progress_callback(
                    "progress_bar_frames",
                    0,
                    f"5-second ending: {tail_length}s tail + {cloned_ending_duration:.1f}s cloned frames",
                )
            elif tail_length > 0:
                progress_callback(
                    "progress_bar_frames", 0, f"5-second ending: {tail_length}s tail (no additional frames needed)"
                )
            else:
                progress_callback(
                    "progress_bar_frames",
                    0,
                    f"5-second ending: {cloned_ending_duration:.1f}s cloned frames (no tail)",
                )

        # Pickle shared worker data once to avoid redundant pickling for each worker
        shared_data_dict = {
            "json_data": self.json_data,
            "route_time_per_frame": self.route_time_per_frame,
            "combined_route_data": self.combined_route_data,
            "shared_map_cache": self.shared_map_cache,
            "filename_to_rgba": self.filename_to_rgba,
            "gpx_time_per_video_time": self.gpx_time_per_video_time,
            "stamp_array": self.stamp_array,
            "shared_route_cache": self.shared_route_cache,
            "is_simultaneous_mode": self.is_simultaneous_mode,
        }
        self.shared_data_filepath = _save_shared_worker_data(shared_data_dict, self.job_id)
        write_debug_log(f"Pickled shared worker data to {self.shared_data_filepath}")

        # Prepare work items (only frames up to the extended_video_length)
        self.work_items = []
        frames_to_generate = int(self.extended_video_length * video_fps)

        for frame_number in range(1, frames_to_generate + 1):
            target_time_video = (frame_number - 1) / video_fps
            self.work_items.append((frame_number, target_time_video, self.shared_data_filepath))

        self.last_rendered_frame = None
        self.last_successfully_returned_frame = None
        self.frames_to_generate = frames_to_generate

        # Pre-calculate leading frames to skip (sequential mode only).
        self.frames_to_skip = 0
        if not self.is_simultaneous_mode:
            combined_route = combined_route_data.get("combined_route", [])
            self.frames_to_skip = compute_sequential_frames_to_skip(
                combined_route, self.route_time_per_frame, frames_to_generate
            )

        self.renderable_frames = self.frames_to_generate - self.frames_to_skip

        if self.frames_to_skip > 0:
            skipped_duration = self.frames_to_skip / video_fps
            self.final_video_length -= skipped_duration
            self.total_frames = int(self.final_video_length * video_fps)
            write_debug_log(
                f"Sequential mode: skipping {self.frames_to_skip} leading frames ({skipped_duration:.2f}s) with < 2 points"
            )
            if progress_callback:
                progress_callback(
                    "progress_bar_frames",
                    0,
                    f"Skipping {self.frames_to_skip} leading frames ({skipped_duration:.2f}s) with insufficient points for lines",
                )

        # 10% status update tracking (based on renderable frames, not total)
        self.ten_percent_frame_count = (
            self.renderable_frames // 10 if self.renderable_frames >= 10 else 0
        )
        self.last_status_update_milestone = 0
        self.last_status_update_time = time.time()

        # Initialize multiprocessing components with improved buffering
        self.manager = multiprocessing.Manager()
        self.frame_buffer = self.manager.dict()

        # Buffer sizing strategy
        if self.total_frames <= 1000:
            self.buffer_size = min(5 * self.num_workers, self.total_frames)
        elif self.total_frames <= 5000:
            self.buffer_size = min(10 * self.num_workers, self.total_frames)
        else:
            self.buffer_size = min(15 * self.num_workers, 200, self.total_frames)

        write_debug_log(
            f"Buffer size: {self.buffer_size} frames (total frames: {self.total_frames}, skipped: {self.frames_to_skip})"
        )

        self.current_frame = 0
        self.pool = None
        self.pending_results = {}
        self.next_frame_to_request = self.frames_to_skip + 1
        self.frames_requested = 0
        self.frames_generated = 0
        self.last_reported_progress_percent = -1
        self.last_consumed_frame = 0
        self.frames_cleanup_threshold = 50

        # Buffer health metrics
        self.buffer_health_check_interval = 50
        self.last_buffer_health_check = 0
        self.consecutive_on_demand_requests = 0
        self.max_consecutive_on_demand = 3

        # Frame prioritization system
        self.priority_frames = set()
        self.last_requested_frame = 0

        # Start the worker pool
        self._start_workers()

    def _start_workers(self):
        """Start the multiprocessing pool and begin frame generation"""
        self.pool = Pool(processes=self.num_workers, maxtasksperchild=50)

        if self.progress_callback:
            self.progress_callback("progress_bar_frames", 0, "Starting frame generation workers")

        self._pre_warm_buffer()
        self._request_more_frames()

    def _pre_warm_buffer(self):
        """Pre-warm the buffer with initial frames to prevent startup bottleneck"""
        write_debug_log("Pre-warming buffer with initial frames")

        pre_warm_count = min(self.buffer_size // 2, self.renderable_frames)

        for _ in range(pre_warm_count):
            if self.next_frame_to_request <= self.frames_to_generate:
                frame_number = self.next_frame_to_request
                work_item = self.work_items[frame_number - 1]

                async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
                self.pending_results[frame_number] = async_result

                self.next_frame_to_request += 1
                self.frames_requested += 1

        write_debug_log(f"Pre-warmed {pre_warm_count} frames")

        wait_time = 0
        max_wait = 10

        while (
            len(self.frame_buffer) < min(pre_warm_count // 4, 10)
            and wait_time < max_wait
        ):
            time.sleep(0.5)
            self._collect_ready_frames()
            wait_time += 0.5

        write_debug_log(f"Buffer pre-warmed: {len(self.frame_buffer)} frames ready")

    def _request_more_frames(self):
        """Request more frames to be generated up to buffer limit"""
        available_slots = self.buffer_size - len(self.frame_buffer) - len(self.pending_results)

        frames_to_request = 0
        if available_slots > 0:
            batch_size = min(available_slots, self.num_workers * 2)
            frames_to_request = min(
                batch_size, self.frames_to_generate - self.next_frame_to_request + 1
            )

        priority_frames_to_request = []
        regular_frames_to_request = []

        for frame_num in sorted(self.priority_frames):
            if (
                frame_num not in self.frame_buffer
                and frame_num not in self.pending_results
                and frame_num <= self.frames_to_generate
            ):
                priority_frames_to_request.append(frame_num)
                if len(priority_frames_to_request) >= frames_to_request:
                    break

        remaining_slots = frames_to_request - len(priority_frames_to_request)
        for _ in range(remaining_slots):
            if self.next_frame_to_request <= self.frames_to_generate:
                regular_frames_to_request.append(self.next_frame_to_request)
                self.next_frame_to_request += 1

        for frame_number in priority_frames_to_request:
            work_item = self.work_items[frame_number - 1]
            async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
            self.pending_results[frame_number] = async_result
            self.frames_requested += 1
            self.priority_frames.discard(frame_number)

        for frame_number in regular_frames_to_request:
            work_item = self.work_items[frame_number - 1]
            async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
            self.pending_results[frame_number] = async_result
            self.frames_requested += 1

    def _check_ten_percent_status_update(self):
        """Check if we've reached a 10% milestone and update status if so."""
        if not self.user or not self.job_id or self.ten_percent_frame_count == 0:
            return

        current_milestone = (self.frames_generated // self.ten_percent_frame_count) * 10

        if (
            current_milestone > self.last_status_update_milestone
            and 10 <= current_milestone <= 90
        ):
            if self.frames_generated % self.ten_percent_frame_count == 0:
                now = time.time()
                if now - self.last_status_update_time >= 5:
                    from update_status import update_status

                    update_status(
                        f"Rendering frames ({self.job_id}) {current_milestone}%",
                        api_key=self.user,
                    )
                    self.last_status_update_time = now
                self.last_status_update_milestone = current_milestone

    def _collect_ready_frames(self):
        """Collect any frames that are ready and add them to buffer"""
        completed_frames = []

        for frame_number, async_result in list(self.pending_results.items()):
            if async_result.ready():
                try:
                    _, frame_array = async_result.get(timeout=0.1)
                    if frame_array is not None:
                        self.frame_buffer[frame_number] = frame_array
                    else:
                        height = int(self.json_data.get("video_resolution_y", 1080))
                        width = int(self.json_data.get("video_resolution_x", 1920))
                        self.frame_buffer[frame_number] = np.zeros((height, width, 3), dtype=np.uint8)

                    self.frames_generated += 1

                    if self.progress_callback:
                        progress_percent = int((self.frames_generated / self.renderable_frames) * 100)
                        if (
                            progress_percent >= self.last_reported_progress_percent + 5
                            or progress_percent >= 100
                        ):
                            progress_text = f"Generated frame {self.frames_generated}/{self.renderable_frames}"
                            self.progress_callback("progress_bar_frames", progress_percent, progress_text)
                            self.last_reported_progress_percent = progress_percent

                    completed_frames.append(frame_number)

                except Exception as e:
                    print(f"Error collecting frame {frame_number}: {e}")
                    height = int(self.json_data.get("video_resolution_y", 1080))
                    width = int(self.json_data.get("video_resolution_x", 1920))
                    self.frame_buffer[frame_number] = np.zeros((height, width, 3), dtype=np.uint8)
                    self.frames_generated += 1

                    if self.progress_callback:
                        progress_percent = int((self.frames_generated / self.renderable_frames) * 100)
                        if (
                            progress_percent >= self.last_reported_progress_percent + 5
                            or progress_percent >= 100
                        ):
                            progress_text = f"Generated frame {self.frames_generated}/{self.renderable_frames}"
                            self.progress_callback("progress_bar_frames", progress_percent, progress_text)
                            self.last_reported_progress_percent = progress_percent

                    completed_frames.append(frame_number)

        for frame_number in completed_frames:
            del self.pending_results[frame_number]

        if completed_frames:
            self._request_more_frames()
            self._check_ten_percent_status_update()

        if self.frames_generated - self.last_buffer_health_check >= self.buffer_health_check_interval:
            self._monitor_buffer_health()
            self.last_buffer_health_check = self.frames_generated

    def _monitor_buffer_health(self):
        """Monitor buffer health and adjust strategy if needed"""
        buffer_ratio = len(self.frame_buffer) / max(1, self.buffer_size)
        pending_count = len(self.pending_results)

        if buffer_ratio < 0.3 and pending_count < self.num_workers:
            if self.debug_callback:
                self.debug_callback("Buffer getting low - increasing request rate")
            self._request_more_frames()

        if self.consecutive_on_demand_requests > self.max_consecutive_on_demand:
            if self.debug_callback:
                self.debug_callback("Too many on-demand requests - aggressive buffer refill")
            self._aggressive_buffer_refill()
            self.consecutive_on_demand_requests = 0

    def _cleanup_old_frames(self, current_frame_number):
        """Clean up frames that are significantly behind the current position."""
        if current_frame_number <= self.frames_cleanup_threshold:
            return

        cleanup_threshold = current_frame_number - self.frames_cleanup_threshold

        frames_to_remove = []
        for frame_num in list(self.frame_buffer.keys()):
            if frame_num < cleanup_threshold and frame_num != self.frames_to_generate:
                frames_to_remove.append(frame_num)

        for frame_num in frames_to_remove:
            del self.frame_buffer[frame_num]

        if frames_to_remove and self.debug_callback:
            self.debug_callback(f"Cleaned up {len(frames_to_remove)} old frames (threshold: {cleanup_threshold})")

    def _aggressive_buffer_refill(self):
        """Aggressively refill the buffer when falling behind"""
        available_slots = self.buffer_size - len(self.frame_buffer) - len(self.pending_results)
        if available_slots <= 0:
            return

        batch_size = min(available_slots, self.num_workers * 3)
        frames_to_request = min(batch_size, self.frames_to_generate - self.next_frame_to_request + 1)

        priority_frames_to_request = []
        regular_frames_to_request = []

        for frame_num in sorted(self.priority_frames):
            if (
                frame_num not in self.frame_buffer
                and frame_num not in self.pending_results
                and frame_num <= self.frames_to_generate
            ):
                priority_frames_to_request.append(frame_num)
                if len(priority_frames_to_request) >= frames_to_request:
                    break

        remaining_slots = frames_to_request - len(priority_frames_to_request)
        for _ in range(remaining_slots):
            if self.next_frame_to_request <= self.frames_to_generate:
                regular_frames_to_request.append(self.next_frame_to_request)
                self.next_frame_to_request += 1

        for frame_number in priority_frames_to_request:
            work_item = self.work_items[frame_number - 1]
            async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
            self.pending_results[frame_number] = async_result
            self.frames_requested += 1
            self.priority_frames.discard(frame_number)

        for frame_number in regular_frames_to_request:
            work_item = self.work_items[frame_number - 1]
            async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
            self.pending_results[frame_number] = async_result
            self.frames_requested += 1

    def make_frame(self, t):
        """Function called by moviepy to get frame at time t."""
        video_fps = float(self.json_data.get("video_fps", 30))
        frame_number = int(t * video_fps) + 1 + self.frames_to_skip

        frame_number = max(
            self.frames_to_skip + 1,
            min(frame_number, self.frames_to_skip + self.total_frames),
        )

        # Handle cloned frames (frames beyond the extended video length)
        if frame_number > self.frames_to_generate:
            if self.last_rendered_frame is not None:
                return self.last_rendered_frame.copy()

            if self.frames_to_generate in self.frame_buffer:
                self.last_rendered_frame = self.frame_buffer[self.frames_to_generate].copy()
                return self.last_rendered_frame.copy()

            self._collect_ready_frames()

            if self.frames_to_generate in self.frame_buffer:
                self.last_rendered_frame = self.frame_buffer[self.frames_to_generate].copy()
                return self.last_rendered_frame.copy()

            if self.last_successfully_returned_frame is not None:
                print(
                    f"Warning: Cloned frame {frame_number} requested but last frame {self.frames_to_generate} not ready yet, using last successful frame"
                )
                return self.last_successfully_returned_frame.copy()

            print(
                f"Warning: Cloned frame {frame_number} requested but last frame {self.frames_to_generate} not ready yet, using black frame"
            )
            height = int(self.json_data.get("video_resolution_y", 1080))
            width = int(self.json_data.get("video_resolution_x", 1920))
            return np.zeros((height, width, 3), dtype=np.uint8)

        # Update frame prioritization based on access pattern
        if frame_number > self.last_requested_frame:
            upcoming_frames = range(
                frame_number + 1, min(frame_number + 10, self.frames_to_generate + 1)
            )
            self.priority_frames.update(upcoming_frames)
        elif frame_number < self.last_requested_frame:
            nearby_frames = range(
                max(self.frames_to_skip + 1, frame_number - 5),
                min(frame_number + 15, self.frames_to_generate + 1),
            )
            self.priority_frames.update(nearby_frames)

        self.last_requested_frame = frame_number

        self._collect_ready_frames()

        buffer_ratio = len(self.frame_buffer) / max(1, self.buffer_size)
        if buffer_ratio < 0.4:
            self._request_more_frames()

        if frame_number in self.priority_frames:
            self.priority_frames.discard(frame_number)

        if frame_number not in self.frame_buffer and frame_number not in self.pending_results:
            self.consecutive_on_demand_requests += 1

            if frame_number <= self.frames_to_generate:
                if self.consecutive_on_demand_requests > 1:
                    write_debug_log(
                        f"Requesting missing frame {frame_number} on-demand (consecutive: {self.consecutive_on_demand_requests})"
                    )
                work_item = self.work_items[frame_number - 1]
                async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
                self.pending_results[frame_number] = async_result

                if frame_number > self.next_frame_to_request:
                    self.next_frame_to_request = frame_number + 1
        else:
            self.consecutive_on_demand_requests = 0

        max_wait_time = 60
        wait_start = time.time()

        while frame_number not in self.frame_buffer:
            if frame_number in self.pending_results:
                time.sleep(0.1)
                self._collect_ready_frames()

                if time.time() - wait_start > max_wait_time:
                    print(
                        f"Timeout waiting for frame {frame_number} (timeout: {max_wait_time}s), using last successful frame instead of black frame"
                    )
                    if self.last_successfully_returned_frame is not None:
                        return self.last_successfully_returned_frame.copy()

                    for fallback_frame_num in range(frame_number - 1, max(1, frame_number - 10), -1):
                        if fallback_frame_num in self.frame_buffer:
                            fallback_frame = self.frame_buffer[fallback_frame_num].copy()
                            print(
                                f"Using frame {fallback_frame_num} as fallback for frame {frame_number}"
                            )
                            return fallback_frame

                    print("No fallback frame available, using black frame")
                    height = int(self.json_data.get("video_resolution_y", 1080))
                    width = int(self.json_data.get("video_resolution_x", 1920))
                    return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                print(
                    f"Frame {frame_number} disappeared from pending, using last successful frame instead of black frame"
                )
                if self.last_successfully_returned_frame is not None:
                    return self.last_successfully_returned_frame.copy()

                for fallback_frame_num in range(frame_number - 1, max(1, frame_number - 10), -1):
                    if fallback_frame_num in self.frame_buffer:
                        fallback_frame = self.frame_buffer[fallback_frame_num].copy()
                        print(
                            f"Using frame {fallback_frame_num} as fallback for frame {frame_number}"
                        )
                        return fallback_frame

                print("No fallback frame available, using black frame")
                height = int(self.json_data.get("video_resolution_y", 1080))
                width = int(self.json_data.get("video_resolution_x", 1920))
                return np.zeros((height, width, 3), dtype=np.uint8)

        frame_array = self.frame_buffer.pop(frame_number)

        if frame_number == self.frames_to_generate:
            self.last_rendered_frame = frame_array.copy()

        self.last_successfully_returned_frame = frame_array.copy()
        self.last_consumed_frame = max(self.last_consumed_frame, frame_number)
        self._cleanup_old_frames(frame_number)

        if frame_number % 100 == 0:
            gc.collect()

        return frame_array

    def cleanup(self):
        """Clean up multiprocessing resources"""
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

        if hasattr(self, "frame_buffer"):
            try:
                self.frame_buffer.clear()
            except Exception:
                pass

        self.last_rendered_frame = None
        self.last_successfully_returned_frame = None
        self.pending_results.clear()

        if hasattr(self, "shared_data_filepath") and self.shared_data_filepath:
            _cleanup_shared_worker_data(self.shared_data_filepath)
            self.shared_data_filepath = None

    def __del__(self):
        """Destructor to ensure cleanup happens even if cleanup() isn't called explicitly"""
        try:
            self.cleanup()
        except Exception:
            pass

