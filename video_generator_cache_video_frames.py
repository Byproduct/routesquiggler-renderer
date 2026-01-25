"""
Functions related to generating video frames.
This file contains things related to preparation and multiprocessing.
The actual map plotting is in video_generator_create_single_frame.py.
"""

# Standard library imports
import gc
import json
import multiprocessing
import os
import re
import sys
import time
from contextlib import contextmanager
from io import StringIO
from multiprocessing import Manager, Pool
from pathlib import Path

# Third-party imports (matplotlib backend must be set before pyplot import)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for video frame caching

import imageio_ffmpeg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy import VideoClip
from PIL import Image

# Increase PIL image size limit to 200 megapixels to handle large renders
Image.MAX_IMAGE_PIXELS = 200_000_000

# Local imports
from config import config


def get_ffmpeg_executable():
    """
    Get the correct ffmpeg executable path for the current platform.
    Uses the same ffmpeg binary that MoviePy uses via imageio_ffmpeg for consistency.
    
    Returns:
        str: Path to ffmpeg executable
    """
    # First, try to use the same ffmpeg that MoviePy/imageio uses
    # This ensures consistency with video generation
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
    except Exception:
        # If imageio_ffmpeg fails for any reason, fall back to platform defaults
        pass
    
    # Fallback: platform-specific defaults
    if os.name == 'nt':  # Windows
        # Check if ffmpeg.exe exists in the project root (same directory as this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_ffmpeg = os.path.join(script_dir, 'ffmpeg.exe')
        if os.path.exists(local_ffmpeg):
            return local_ffmpeg
        # Fallback to system ffmpeg.exe (should be in PATH)
        return 'ffmpeg.exe'
    else:  # Linux/Mac
        # On Linux/Mac, ffmpeg should be in PATH
        return 'ffmpeg'
from image_generator_utils import calculate_resolution_scale
from video_generator_calculate_bounding_boxes import calculate_route_time_per_frame
from video_generator_create_combined_route import RoutePoint
from video_generator_create_single_frame import generate_video_frame_in_memory
from video_generator_create_single_frame_utils import hex_to_rgba
from write_log import write_debug_log, write_log


class MoviePyDebugLogger:
    """Custom logger for MoviePy that only outputs when debug logging is enabled."""
    def __init__(self, debug_callback=None):
        self.debug_callback = debug_callback
    
    def message(self, message):
        """Log MoviePy messages only when debug is enabled."""
        if config.debug_logging and self.debug_callback:
            self.debug_callback(f"MoviePy: {message}")
    
    def error(self, message):
        """Log MoviePy errors only when debug is enabled."""
        if config.debug_logging and self.debug_callback:
            self.debug_callback(f"MoviePy error: {message}")
    
    def warning(self, message):
        """Log MoviePy warnings only when debug is enabled."""
        if config.debug_logging and self.debug_callback:
            self.debug_callback(f"MoviePy warning: {message}")


@contextmanager
def suppress_moviepy_output():
    """Context manager to suppress MoviePy stdout/stderr output when debug is off."""
    if not config.debug_logging:
        # Redirect stdout and stderr to devnull when debug is off
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    else:
        # When debug is on, don't suppress anything
        yield


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


def _binary_search_cutoff_index(route_points, target_time):
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


def _streaming_frame_worker(args):
    """
    Worker function for generating a single frame and returning it as numpy array
    instead of saving to disk. Supports both single route and multiple routes modes.
    OPTIMIZED: Better memory management and performance for computationally intensive tasks.
    OPTIMIZED: Uses binary search + list slicing instead of linear search + individual appends.
    """
    frame_number, json_data, route_time_per_frame, combined_route_data, shared_map_cache, filename_to_rgba, gpx_time_per_video_time, stamp_array, target_time_video, shared_route_cache = args
    

    try:
        # Ensure frame_number is an integer
        frame_number = int(frame_number)
        
        # The caching logic needs video seconds, but point collection needs route seconds
        target_time_route = target_time_video * gpx_time_per_video_time if gpx_time_per_video_time > 0 else 0
        
        # Check if we have multiple routes
        all_routes = combined_route_data.get('all_routes', None)
        
        # Calculate the frame number where the original route ends
        video_length = float(json_data.get('video_length', 30))
        video_fps = float(json_data.get('video_fps', 30))
        tail_length = int(json_data.get('tail_length', 0))  # Ensure tail_length is int
        
        # SIMULTANEOUS MODE FIX: Calculate effective video duration for staggered routes
        effective_video_length = video_length
        if all_routes and len(all_routes) > 1:
            # Check if routes have different route_index values (truly different routes vs split tracks)
            route_indices = set()
            for route_data in all_routes:
                route_points = route_data.get('combined_route', [])
                if route_points:
                    first_point = route_points[0]
                    route_indices.add(first_point.route_index)
            
            has_different_route_indices = len(route_indices) > 1
            
            if has_different_route_indices:
                # SIMULTANEOUS MODE FIX: Calculate effective video duration
                # Find the earliest start time and latest start time across all routes
                earliest_start_time = None
                latest_start_time = None
                max_route_duration = 0.0
                
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    if route_points and len(route_points) > 0:
                        first_point = route_points[0]
                        last_point = route_points[-1]
                        
                        route_start_timestamp = first_point.timestamp
                        route_duration = last_point.accumulated_time
                        
                        if route_start_timestamp:
                            if earliest_start_time is None or route_start_timestamp < earliest_start_time:
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
                    # This ensures the route timing calculation matches the bounding box calculation
                    effective_target_time_video = min(target_time_video, effective_video_length)
                    target_time_route = effective_target_time_video * gpx_time_per_video_time if gpx_time_per_video_time > 0 else 0
        
        # Calculate the frame number where the original route ends (using effective video length)
        original_route_end_frame = int(effective_video_length * video_fps)
        
        # Check if we're in the tail-only phase (extra frames after original route ends)
        is_tail_only_frame = frame_number > original_route_end_frame
        
        # Get hide_complete_routes parameter (defaults to False if not specified)
        # Handle both boolean and string values (in case JSON parsing returns string)
        hide_complete_routes_raw = json_data.get('hide_complete_routes', False)
        if isinstance(hide_complete_routes_raw, str):
            # Convert string to boolean
            hide_complete_routes = hide_complete_routes_raw.lower() in ('true', '1', 'yes')
        else:
            hide_complete_routes = bool(hide_complete_routes_raw)
        
        # Debug: Log the parameter value (only for first few frames to avoid spam)
        if frame_number <= 3:
            print(f"DEBUG Frame {frame_number}: hide_complete_routes = {hide_complete_routes} (raw: {hide_complete_routes_raw}, type: {type(hide_complete_routes_raw)})")
        
        # SIMULTANEOUS MODE ENDING FIX: Detect if we're in simultaneous mode
        is_simultaneous_mode = False
        if all_routes and len(all_routes) > 1:
            # Check if routes have different route_index values (truly different routes vs split tracks)
            route_indices = set()
            for route_data in all_routes:
                route_points = route_data.get('combined_route', [])
                if route_points:
                    first_point = route_points[0]
                    route_indices.add(first_point.route_index)
            
            is_simultaneous_mode = len(route_indices) > 1
        
        # SIMULTANEOUS MODE ENDING FIX: Use different logic for simultaneous mode ending
        if is_simultaneous_mode and is_tail_only_frame:
            # For simultaneous mode, don't use the sequential tail-only phase
            # Instead, let each route handle its own completion and tail fade-out naturally
            # This preserves the individual route timing and cached route appearance
            is_tail_only_frame = False  # Treat as normal frame for simultaneous mode
        
        if all_routes and len(all_routes) > 1:
            # Check if routes have different route_index values (truly different routes vs split tracks)
            route_indices = set()
            for route_data in all_routes:
                route_points = route_data.get('combined_route', [])
                if route_points:
                    first_point = route_points[0]
                    route_indices.add(first_point.route_index)
            
            has_different_route_indices = len(route_indices) > 1
            
            if has_different_route_indices:
                # Multiple routes mode - create a list of sub-lists for each route
                points_for_frame = []
                
                # STAGGERED ROUTES FIX: Calculate route start delays
                # Find the earliest start time across all routes to establish the video timeline baseline
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
                
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    
                    # Calculate this route's delay relative to the earliest route
                    route_delay_seconds = 0.0
                    route_id = id(route_data)
                    if route_id in route_start_times and earliest_start_time:
                        route_start_timestamp = route_start_times[route_id]
                        route_delay_seconds = (route_start_timestamp - earliest_start_time).total_seconds()
                    
                    # Calculate route-specific target time accounting for start delay
                    route_target_time = target_time_route - route_delay_seconds
                    
                    # Check if route is complete (all points included)
                    route_end_time = route_points[-1].accumulated_time if route_points else 0
                    route_is_complete = route_points and route_target_time >= route_end_time
                    
                    # Debug: Log completion status for first few frames
                    if frame_number <= 3 and hide_complete_routes:
                        print(f"DEBUG Frame {frame_number} (simultaneous): route_target_time={route_target_time:.2f}, route_end_time={route_end_time:.2f}, complete={route_is_complete}")
                    
                    # Skip complete routes if hide_complete_routes is enabled
                    if hide_complete_routes and route_is_complete:
                        if frame_number <= 3:
                            print(f"DEBUG Frame {frame_number}: Skipping complete route (simultaneous mode)")
                        continue
                    
                    if is_tail_only_frame:
                        # SIMULTANEOUS MODE TAIL FIX: Each route should have its own individual tail fade-out
                        # Calculate when this specific route ends (accounting for its start delay)
                        route_end_time_with_delay = route_end_time + route_delay_seconds
                        
                        # Calculate how much time has passed since this route ended
                        current_route_time = target_time_route
                        time_since_route_end = current_route_time - route_end_time_with_delay
                        
                        # Only show tail if this route has ended and we're within the tail duration
                        tail_duration_route = gpx_time_per_video_time * tail_length if gpx_time_per_video_time else 0
                        
                        if time_since_route_end >= 0 and time_since_route_end <= tail_duration_route:
                            # Route has ended and we're within tail duration - show fading tail
                            # Calculate the fade-out progress (0.0 = just ended, 1.0 = fully faded)
                            fade_progress = time_since_route_end / tail_duration_route if tail_duration_route > 0 else 1.0
                            
                            # Use the route's actual end time for tail calculation
                            # This ensures the tail fades out from the route's actual end point
                            cutoff_index = _binary_search_cutoff_index(route_points, route_end_time)
                            route_points_for_frame = route_points[:cutoff_index]
                        else:
                            # Route hasn't ended yet or tail has fully faded - don't show this route
                            route_points_for_frame = []
                    else:
                        # Normal frame - find the points up to the route-specific target time
                        # Only include points if this route should have started by now
                        if route_target_time >= 0:
                            # OPTIMIZED: Use binary search + list slicing instead of linear search + appends
                            cutoff_index = _binary_search_cutoff_index(route_points, route_target_time)
                            route_points_for_frame = route_points[:cutoff_index]
                        else:
                            route_points_for_frame = []
                    
                    # Only add this route's points if it has any points
                    if route_points_for_frame:
                        points_for_frame.append(route_points_for_frame)
            # else:
            #     # SEQUENTIAL MODE RENDERING PATH - DISABLED
            #     # Single route group with multiple tracks - process all tracks sequentially
            #     # In sequential mode, each route starts after the previous one ends
            #     # COMMENTED OUT: This path was causing rendering issues. Using single-route path instead.
            #     pass
        else:
            # Single route mode (backward compatibility)
            combined_route = combined_route_data.get('combined_route', [])
        
        # Handle tail-only frames and normal frames for single route mode
        if not all_routes or len(all_routes) <= 1:
            if is_tail_only_frame:
                # TAIL-ONLY FIX: Create progressive virtual leading edge that moves forwards
                # This simulates the route continuing forward as tail fades out naturally
                
                # Include all points up to the route end (don't reduce points)
                final_route_target_time = (original_route_end_frame - 1) * route_time_per_frame
                
                # Only include points if this route should have started by the final frame time
                if final_route_target_time >= 0:
                    # OPTIMIZED: Use binary search + list slicing instead of linear search + appends
                    cutoff_index = _binary_search_cutoff_index(combined_route, final_route_target_time)
                    points_for_frame = combined_route[:cutoff_index]
                else:
                    points_for_frame = []
            else:
                # Normal frame - find the points up to the target time
                # OPTIMIZED: Use binary search + list slicing instead of linear search + appends
                cutoff_index = _binary_search_cutoff_index(combined_route, target_time_route)
                points_for_frame = combined_route[:cutoff_index]
            
            # Apply hide_complete_routes filtering if enabled
            if hide_complete_routes and points_for_frame:
                # Find which files have completed (all their points are included)
                # A file is complete if its last point's accumulated_time <= target_time_route
                filename_to_last_time = {}
                
                # First pass: find the last accumulated_time for each filename in the full combined_route
                for point in combined_route:
                    if hasattr(point, 'filename') and point.filename:
                        filename = point.filename
                        if filename not in filename_to_last_time or point.accumulated_time > filename_to_last_time[filename]:
                            filename_to_last_time[filename] = point.accumulated_time
                
                # Determine which files are complete (their last point's time <= target_time_route)
                completed_filenames = set()
                for filename, last_time in filename_to_last_time.items():
                    if last_time <= target_time_route:
                        # This file's last point has been reached, so the file is complete
                        completed_filenames.add(filename)
                        if frame_number <= 3:
                            print(f"DEBUG Frame {frame_number}: File '{filename}' is complete (last point time: {last_time:.2f}, target: {target_time_route:.2f})")
                
                # Filter out points from completed files
                if completed_filenames:
                    original_count = len(points_for_frame)
                    points_for_frame = [p for p in points_for_frame if not (hasattr(p, 'filename') and p.filename and p.filename in completed_filenames)]
                    filtered_count = len(points_for_frame)
                    
                    if frame_number <= 3:
                        print(f"DEBUG Frame {frame_number}: Filtered out {original_count - filtered_count} points from {len(completed_filenames)} completed file(s). Remaining: {filtered_count} points")
        
        # Check if we have any points (either as a list of lists or a single list)
        has_points = False
        if all_routes and len(all_routes) > 1:
            # Multiple routes mode - check if any route has points
            has_points = any(len(route_points) > 0 for route_points in points_for_frame)
            # Enhanced debug for sequential mode
            if hide_complete_routes and frame_number <= 10:
                route_count = len(points_for_frame)
                total_points = sum(len(route_points) for route_points in points_for_frame)
                print(f"DEBUG Frame {frame_number} FINAL: {route_count} routes in points_for_frame, {total_points} total points, has_points={has_points}")
        else:
            # Single route mode - check if the list has points
            has_points = len(points_for_frame) > 0
        
        if not has_points:
            if hide_complete_routes and frame_number <= 10:
                print(f"DEBUG Frame {frame_number}: No points found, returning None")
            return frame_number, None
        
        # Validate that shared map cache is provided
        if shared_map_cache is None:
            print(f"Error: Shared map cache is required for frame generation")
            return None
        
        # OPTIMIZED: Pre-clear matplotlib state to reduce memory pressure
        plt.close('all')
        
        # Calculate virtual leading time for tail-only frames (progressive forward movement)
        virtual_leading_time = None
        route_specific_tail_info = {}  # Store route-specific tail timing information
        
        # FIX: Calculate route-specific tail info for ALL frames, not just tail-only frames
        # This enables proper completion detection in simultaneous mode
        if all_routes and len(all_routes) > 1:
            # Check if routes have different route_index values
            route_indices = set()
            for route_data in all_routes:
                route_points = route_data.get('combined_route', [])
                if route_points:
                    first_point = route_points[0]
                    route_indices.add(first_point.route_index)
            
            has_different_route_indices = len(route_indices) > 1
            
            if has_different_route_indices:
                # Calculate route start delays
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
                
                # Calculate route-specific tail timing for all routes
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    if not route_points:
                        continue
                    
                    # Calculate this route's delay relative to the earliest route
                    route_delay_seconds = 0.0
                    route_id = id(route_data)
                    if route_id in route_start_times and earliest_start_time:
                        route_start_timestamp = route_start_times[route_id]
                        route_delay_seconds = (route_start_timestamp - earliest_start_time).total_seconds()
                    
                    # Calculate when this specific route ends
                    route_end_time = route_points[-1].accumulated_time if route_points else 0
                    route_end_time_with_delay = route_end_time + route_delay_seconds
                    
                    # Store route-specific tail information
                    route_index = route_points[0].route_index if route_points else 0
                    route_specific_tail_info[route_index] = {
                        'route_end_time': route_end_time_with_delay,
                        'route_delay_seconds': route_delay_seconds
                    }
                    
                    # For tail-only frames, also calculate virtual leading time
                    if is_tail_only_frame:
                        # Calculate how many frames into the tail-only phase we are
                        tail_only_frame_offset = frame_number - original_route_end_frame
                        
                        # Calculate route-specific virtual leading time
                        route_virtual_leading_time = route_end_time_with_delay + (tail_only_frame_offset * route_time_per_frame)
                        route_specific_tail_info[route_index]['virtual_leading_time'] = route_virtual_leading_time
        
        if is_tail_only_frame:
            # Calculate how many frames into the tail-only phase we are
            tail_only_frame_offset = frame_number - original_route_end_frame
            
            # Calculate virtual leading time that moves forward with each tail-only frame
            # Start from the route end and continue forward by route_time_per_frame for each tail frame  
            route_end_time = (original_route_end_frame - 1) * route_time_per_frame
            virtual_leading_time = route_end_time + (tail_only_frame_offset * route_time_per_frame)
        
        # Filter points of interest for this frame
        # POIs without timestamp are always shown; POIs with timestamp appear when their time is reached
        points_of_interest_for_frame = []
        poi_setting = json_data.get('points_of_interest', 'off')
        if poi_setting in ['light', 'dark']:
            all_pois = json_data.get('_points_of_interest_data', [])
            
            # Get the combined route for timestamp-to-accumulated_time conversion
            # This handles gaps between GPX files correctly
            combined_route_for_poi = combined_route_data.get('combined_route', [])
            if not combined_route_for_poi and all_routes:
                # If no single combined_route, use the first route's data
                combined_route_for_poi = all_routes[0].get('combined_route', []) if all_routes else []
            
            for poi in all_pois:
                if poi.timestamp is None:
                    # POIs without timestamp are always included
                    points_of_interest_for_frame.append(poi)
                else:
                    # Convert POI timestamp to equivalent accumulated_time
                    # This correctly handles gaps between GPX files
                    poi_accumulated_time = _timestamp_to_accumulated_time(poi.timestamp, combined_route_for_poi)
                    if poi_accumulated_time is not None and poi_accumulated_time <= target_time_route:
                        points_of_interest_for_frame.append(poi)
        
        # Generate frame in memory instead of saving to disk
        frame_array = generate_video_frame_in_memory(
            frame_number, points_for_frame, json_data, shared_map_cache, filename_to_rgba, gpx_time_per_video_time, stamp_array, target_time_video, shared_route_cache, virtual_leading_time, route_specific_tail_info, points_of_interest_for_frame
        )
        
        # MEMORY MANAGEMENT: Aggressive cleanup to prevent memory accumulation
        # This is especially important on Linux where memory fragmentation can occur.
        # Clean up every 3 frames to prevent memory buildup in workers.
        if frame_number % 3 == 0:
            # Force matplotlib to clean up any lingering figures
            plt.close('all')
            
            # Clear matplotlib's internal caches periodically
            if hasattr(plt, '_pylab_helpers'):
                plt._pylab_helpers.Gcf.destroy_all()
            
            # Force full garbage collection to free up memory
            # Generation 2 collection cleans up all unreachable objects
            gc.collect(2)
        
        return frame_number, frame_array
        
    except Exception as e:
        print(f"Error generating frame {frame_number}: {e}")
        # Clean up on error
        try:
            plt.close('all')
            gc.collect()
        except:
            pass
        return frame_number, None


class StreamingFrameGenerator:
    """
    Generator class that yields frames for moviepy using multiprocessing.
    Uses an improved producer-consumer pattern with adaptive buffering and continuous frame generation.
    """
    def __init__(self, json_data, route_time_per_frame, combined_route_data, max_workers=None, shared_map_cache=None, shared_route_cache=None, progress_callback=None, gpx_time_per_video_time=None, debug_callback=None, user=None):
        self.json_data = json_data
        self.route_time_per_frame = route_time_per_frame
        self.combined_route_data = combined_route_data
        self.shared_map_cache = shared_map_cache
        self.shared_route_cache = shared_route_cache
        self.progress_callback = progress_callback
        self.gpx_time_per_video_time = gpx_time_per_video_time
        self.debug_callback = debug_callback
        self.user = user
        self.job_id = str(json_data.get('job_id', '')) if json_data else ''
        
        # Pre-compute filename-to-RGBA color mapping once for all frames
        self.filename_to_rgba = {}
        if 'track_objects' in json_data:
            for track_obj in json_data['track_objects']:
                filename = track_obj['filename']
                # Remove path and extension to match the format in points_for_frame
                filename = os.path.basename(filename)  # Remove path
                if filename.endswith('.gpx'):
                    filename = filename[:-4]  # Remove .gpx extension
                
                # Convert hex color to RGBA once
                try:
                    rgba_color = hex_to_rgba(track_obj['color'])
                except:
                    rgba_color = (1.0, 0.0, 0.0, 1.0)  # Default red
                
                self.filename_to_rgba[filename] = rgba_color
                                  
        # Pre-load stamp array once for all frames
        self.stamp_array = None
        try:
            width = int(json_data.get('video_resolution_x', 1920))
            height = int(json_data.get('video_resolution_y', 1080))
            
            # Calculate resolution scale to determine appropriate stamp size
            image_scale = calculate_resolution_scale(width, height)
            
            # Choose stamp file based on resolution scale
            if image_scale < 1:
                # Use stamp_small for resolutions with scale < 1 (less than 1 MP)
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
                # Fallback to 1x for any other scale value
                stamp_file = "img/stamp_1x.npy"
            
            self.stamp_array = np.load(stamp_file)
            write_debug_log(f"Loaded stamp: {stamp_file} (scale: {image_scale})")
        except Exception as e:
            write_log(f"Warning: Could not load stamp: {e}")
        
        # Calculate video parameters
        video_length = float(json_data.get('video_length', 30))
        video_fps = float(json_data.get('video_fps', 30))
        
        # SIMULTANEOUS MODE ENDING FIX: Detect if we're in simultaneous mode
        is_simultaneous_mode = False
        all_routes = combined_route_data.get('all_routes', None)
        if all_routes and len(all_routes) > 1:
            # Check if routes have different route_index values (truly different routes vs split tracks)
            route_indices = set()
            for route_data in all_routes:
                route_points = route_data.get('combined_route', [])
                if route_points:
                    first_point = route_points[0]
                    route_indices.add(first_point.route_index)
            
            is_simultaneous_mode = len(route_indices) > 1
        
        # Calculate ending parameters based on mode
        if is_simultaneous_mode:
            # SIMULTANEOUS MODE: Each route handles its own completion and tail fade-out
            # We need to extend the video to account for the tail fade-out of the last route
            tail_length = int(json_data.get('tail_length', 0))  # Get the tail length for fade-out
            
            # Total video length: original video + tail fade-out of the last route + 5 seconds of cloned ending
            extended_video_length = video_length + tail_length  # Extend by tail length for fade-out
            ending_duration_required = 5.0  # Always 5 seconds of ending total
            cloned_ending_duration = ending_duration_required  # All 5 seconds via cloned frames
            final_video_length = extended_video_length + cloned_ending_duration
            
            if progress_callback:
                progress_callback("progress_bar_frames", 0, f"SIMULTANEOUS MODE: Creating video: {video_length}s route + {tail_length}s tail + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
                progress_callback("progress_bar_frames", 0, f"Each route handles its own completion and tail fade-out with {tail_length}s fade-out for the last route")
        else:
            # SEQUENTIAL MODE: Use the original tail-only phase logic
            tail_length = int(json_data.get('tail_length', 0))  # Ensure tail_length is int
            ending_duration_required = 5.0  # Always 5 seconds of ending total
            cloned_ending_duration = max(0.0, ending_duration_required - tail_length)  # Additional time needed via cloned frames
            
            # Total video length includes: original video + tail frames + cloned ending
            extended_video_length = video_length + tail_length if tail_length > 0 else video_length
            final_video_length = extended_video_length + cloned_ending_duration
            
            if progress_callback:
                if tail_length > 0 and cloned_ending_duration > 0:
                    progress_callback("progress_bar_frames", 0, f"SEQUENTIAL MODE: Creating video: {video_length}s route + {tail_length}s tail + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
                elif tail_length > 0:
                    progress_callback("progress_bar_frames", 0, f"SEQUENTIAL MODE: Creating video: {video_length}s route + {tail_length}s tail = {final_video_length}s total (no additional frames needed)")
                else:
                    progress_callback("progress_bar_frames", 0, f"SEQUENTIAL MODE: Creating video: {video_length}s route + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
        
        # Calculate frame counts
        original_frames = int(video_length * video_fps)
        tail_frames = int(tail_length * video_fps) if tail_length > 0 else 0  
        cloned_frames = int(cloned_ending_duration * video_fps)
        total_frames = int(final_video_length * video_fps)

        self.total_frames = int(final_video_length * video_fps)
        self.original_frames = original_frames
        self.tail_frames = tail_frames  
        self.cloned_frames = cloned_frames
        self.extended_video_length = extended_video_length  # Video length including tails but excluding clones
        self.final_video_length = final_video_length  # Final video length including everything
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(4, multiprocessing.cpu_count())
        self.num_workers = min(max_workers, self.total_frames)
        
        write_debug_log(f"Using {self.num_workers} worker processes for frame generation")
        
        # Log the ending structure for user feedback
        if progress_callback: # Changed from log_callback to progress_callback
            if tail_length > 0 and cloned_ending_duration > 0:
                progress_callback("progress_bar_frames", 0, f"5-second ending: {tail_length}s tail + {cloned_ending_duration:.1f}s cloned frames")
            elif tail_length > 0:
                progress_callback("progress_bar_frames", 0, f"5-second ending: {tail_length}s tail (no additional frames needed)")
            else:
                progress_callback("progress_bar_frames", 0, f"5-second ending: {cloned_ending_duration:.1f}s cloned frames (no tail)")
        
        # Prepare work items with pre-computed color mapping and stamp
        # Only generate frames up to the extended_video_length (excludes cloned frames)
        self.work_items = []
        frames_to_generate = int(self.extended_video_length * video_fps)  # Only frames that need rendering
        
        for frame_number in range(1, frames_to_generate + 1):
            # Calculate target_time for this frame in route seconds
            target_time_route = (frame_number - 1) * self.route_time_per_frame
            
            # Convert target_time from route seconds to video seconds for generate_video_frame_in_memory
            # Since gpx_time_per_video_time = route_seconds / video_seconds
            # Therefore: video_seconds = route_seconds / gpx_time_per_video_time
            target_time_video = target_time_route / self.gpx_time_per_video_time if self.gpx_time_per_video_time > 0 else 0
            
            self.work_items.append((
                frame_number, self.json_data, self.route_time_per_frame, self.combined_route_data, self.shared_map_cache, self.filename_to_rgba, self.gpx_time_per_video_time, self.stamp_array, target_time_video, self.shared_route_cache
            ))
        
        # Store the last rendered frame for cloning (will be set during frame generation)
        self.last_rendered_frame = None
        self.last_successfully_returned_frame = None  # Track last successfully returned frame for fallback
        self.frames_to_generate = frames_to_generate
        
        # 10% status update tracking
        # Calculate frame count for each 10% milestone (total frames are always divisible by 10)
        self.ten_percent_frame_count = frames_to_generate // 10 if frames_to_generate >= 10 else 0
        self.last_status_update_milestone = 0  # Track last reported milestone (0, 10, 20, ..., 90)
        
        # Initialize multiprocessing components with improved buffering
        self.manager = multiprocessing.Manager()
        self.frame_buffer = self.manager.dict()  # {frame_number: frame_array}
        
        # IMPROVED BUFFERING STRATEGY:
        # For videos with thousands of frames, use a larger buffer to prevent on-demand requests
        # Buffer size should be proportional to the total frames but capped for memory efficiency
        if self.total_frames <= 1000:
            # Small videos: use 5x workers buffer
            self.buffer_size = min(5 * self.num_workers, self.total_frames)
        elif self.total_frames <= 5000:
            # Medium videos: use 10x workers buffer
            self.buffer_size = min(10 * self.num_workers, self.total_frames)
        else:
            # Large videos: use 15x workers buffer, but cap at 200 frames for memory efficiency
            self.buffer_size = min(15 * self.num_workers, 200, self.total_frames)
        
        write_debug_log(f"Buffer size: {self.buffer_size} frames (total frames: {self.total_frames})")
        
        self.current_frame = 0
        self.pool = None
        self.pending_results = {}  # {frame_number: AsyncResult}
        self.next_frame_to_request = 1
        self.frames_requested = 0
        self.frames_generated = 0  # Track total frames generated
        
        # NEW: Track buffer health and performance metrics
        self.buffer_health_check_interval = 50  # Check buffer health every 50 frames
        self.last_buffer_health_check = 0
        self.consecutive_on_demand_requests = 0
        self.max_consecutive_on_demand = 3  # Threshold to trigger buffer refill
        
        # NEW: Frame prioritization system
        self.priority_frames = set()  # Frames that should be prioritized
        self.last_requested_frame = 0  # Track the last frame requested by MoviePy
        
        # Start the worker pool
        self._start_workers()
        
    def _start_workers(self):
        """Start the multiprocessing pool and begin frame generation"""
        # Worker pool configuration with memory management in mind.
        # Each worker will handle at most 50 frames before being recycled.
        # This balances:
        # - Performance: Avoiding too-frequent worker restarts
        # - Memory: Preventing memory accumulation in long-running workers
        # On Linux with 'spawn' start method (set in main.py), workers start fresh,
        # but matplotlib/numpy can still accumulate memory over many frames.
        self.pool = Pool(processes=self.num_workers, maxtasksperchild=50)
        
        # Initial progress update
        if self.progress_callback:
            self.progress_callback("progress_bar_frames", 0, "Starting frame generation workers")
        
        # NEW: Pre-warm the buffer before starting consumption
        self._pre_warm_buffer()
        
        # Start generating initial batch of frames
        self._request_more_frames()
    
    def _pre_warm_buffer(self):
        """Pre-warm the buffer with initial frames to prevent startup bottleneck"""
        write_debug_log(f"Pre-warming buffer with initial frames")
        
        # Calculate how many frames to pre-warm (aim for 50% of buffer size)
        pre_warm_count = min(self.buffer_size // 2, self.total_frames)
        
        # Request initial frames to get the pipeline started
        for _ in range(pre_warm_count):
            if self.next_frame_to_request <= self.total_frames:
                frame_number = self.next_frame_to_request
                work_item = self.work_items[frame_number - 1]
                
                # Submit frame generation job
                async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
                self.pending_results[frame_number] = async_result
                
                self.next_frame_to_request += 1
                self.frames_requested += 1
        
        write_debug_log(f"Pre-warmed {pre_warm_count} frames")
        
        # Wait for some frames to complete to establish the pipeline
        wait_time = 0
        max_wait = 10  # Maximum 10 seconds to wait for initial frames
        
        while len(self.frame_buffer) < min(pre_warm_count // 4, 10) and wait_time < max_wait:
            time.sleep(0.5)
            self._collect_ready_frames()
            wait_time += 0.5
        
        write_debug_log(f"Buffer pre-warmed: {len(self.frame_buffer)} frames ready")
        
    def _request_more_frames(self):
        """Request more frames to be generated up to buffer limit"""
        # IMPROVED: More aggressive frame requesting with better buffer management
        frames_to_request = 0
        
        # Calculate how many frames we can request
        available_slots = self.buffer_size - len(self.frame_buffer) - len(self.pending_results)
        
        if available_slots > 0:
            # Request frames in batches to maintain continuous pipeline
            batch_size = min(available_slots, self.num_workers * 2)  # Request 2x workers worth of frames
            frames_to_request = min(batch_size, self.total_frames - self.next_frame_to_request + 1)
        
        # NEW: Prioritize frames that are likely to be requested soon
        priority_frames_to_request = []
        regular_frames_to_request = []
        
        # First, check if we have priority frames to request
        for frame_num in sorted(self.priority_frames):
            if (frame_num not in self.frame_buffer and 
                frame_num not in self.pending_results and 
                frame_num <= self.frames_to_generate):  # Only process frames that can be generated
                priority_frames_to_request.append(frame_num)
                if len(priority_frames_to_request) >= frames_to_request:
                    break
        
        # Then fill remaining slots with sequential frames
        remaining_slots = frames_to_request - len(priority_frames_to_request)
        for _ in range(remaining_slots):
            if self.next_frame_to_request <= self.frames_to_generate:  # Only generate frames within the limit
                regular_frames_to_request.append(self.next_frame_to_request)
                self.next_frame_to_request += 1
        
        # Request priority frames first
        for frame_number in priority_frames_to_request:
            work_item = self.work_items[frame_number - 1]
            async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
            self.pending_results[frame_number] = async_result
            self.frames_requested += 1
            self.priority_frames.discard(frame_number)  # Remove from priority set
        
        # Then request regular frames
        for frame_number in regular_frames_to_request:
            work_item = self.work_items[frame_number - 1]
            async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
            self.pending_results[frame_number] = async_result
            self.frames_requested += 1
    
    def _check_ten_percent_status_update(self):
        """Check if we've reached a 10% milestone and update status if so."""
        if not self.user or not self.job_id or self.ten_percent_frame_count == 0:
            return
        
        # Calculate current milestone (10, 20, 30, ..., 90)
        current_milestone = (self.frames_generated // self.ten_percent_frame_count) * 10
        
        # Only report milestones 10-90 (not 0% or 100%)
        if current_milestone > self.last_status_update_milestone and 10 <= current_milestone <= 90:
            # Check if we're exactly at a 10% boundary
            if self.frames_generated % self.ten_percent_frame_count == 0:
                from update_status import update_status
                update_status(f"rendering ({self.job_id}) {current_milestone}%", api_key=self.user)
                self.last_status_update_milestone = current_milestone
    
    def _collect_ready_frames(self):
        """Collect any frames that are ready and add them to buffer"""
        completed_frames = []
        
        for frame_number, async_result in list(self.pending_results.items()):
            if async_result.ready():
                try:
                    result_frame_number, frame_array = async_result.get(timeout=0.1)
                    if frame_array is not None:
                        self.frame_buffer[frame_number] = frame_array
                        # Increment frames generated counter
                        self.frames_generated += 1
                        
                        # Update progress if callback is provided
                        if self.progress_callback:
                            progress_percent = int((self.frames_generated / self.frames_to_generate) * 100)  # Use frames_to_generate
                            progress_text = f"Generated frame {self.frames_generated}/{self.frames_to_generate}"
                            self.progress_callback("progress_bar_frames", progress_percent, progress_text)
                    else:
                        # Create black frame if generation failed
                        height = int(self.json_data.get('video_resolution_y', 1080))
                        width = int(self.json_data.get('video_resolution_x', 1920))
                        self.frame_buffer[frame_number] = np.zeros((height, width, 3), dtype=np.uint8)
                        # Still count as generated (even if it's a black frame)
                        self.frames_generated += 1
                        
                        # Update progress if callback is provided
                        if self.progress_callback:
                            progress_percent = int((self.frames_generated / self.frames_to_generate) * 100)  # Use frames_to_generate
                            progress_text = f"Generated frame {self.frames_generated}/{self.frames_to_generate}"
                            self.progress_callback("progress_bar_frames", progress_percent, progress_text)
                    
                    completed_frames.append(frame_number)
                    
                except Exception as e:
                    print(f"Error collecting frame {frame_number}: {e}")
                    # Create black frame on error
                    height = int(self.json_data.get('video_resolution_y', 1080))
                    width = int(self.json_data.get('video_resolution_x', 1920))
                    self.frame_buffer[frame_number] = np.zeros((height, width, 3), dtype=np.uint8)
                    # Still count as generated (even if it's a black frame)
                    self.frames_generated += 1
                    
                    # Update progress if callback is provided
                    if self.progress_callback:
                        progress_percent = int((self.frames_generated / self.frames_to_generate) * 100)  # Use frames_to_generate
                        progress_text = f"Generated frame {self.frames_generated}/{self.frames_to_generate}"
                        self.progress_callback("progress_bar_frames", progress_percent, progress_text)
                    
                    completed_frames.append(frame_number)
        
        # Remove completed frames from pending
        for frame_number in completed_frames:
            del self.pending_results[frame_number]
        
        # IMPROVED: Always request more frames after collecting to maintain pipeline
        if completed_frames:
            self._request_more_frames()
            # Check if we've hit a 10% milestone and update server status
            self._check_ten_percent_status_update()
        
        # NEW: Periodic buffer health monitoring
        if self.frames_generated - self.last_buffer_health_check >= self.buffer_health_check_interval:
            self._monitor_buffer_health()
            self.last_buffer_health_check = self.frames_generated
    
    def _monitor_buffer_health(self):
        """Monitor buffer health and adjust strategy if needed"""
        buffer_utilization = len(self.frame_buffer) / self.buffer_size * 100
        pending_count = len(self.pending_results)
        buffer_ratio = len(self.frame_buffer) / max(1, self.buffer_size)
        
        # Log buffer health for debugging
        # print(f"Buffer health: {buffer_utilization:.1f}% full ({len(self.frame_buffer)}/{self.buffer_size}), {pending_count} pending")
        
        # If buffer is getting low and we have many pending, increase request rate
        if buffer_ratio < 0.3 and pending_count < self.num_workers:
            if self.debug_callback:
                self.debug_callback("Buffer getting low - increasing request rate")
            self._request_more_frames()
        
        # If we're falling behind, request more aggressively
        if self.consecutive_on_demand_requests > self.max_consecutive_on_demand:
            if self.debug_callback:
                self.debug_callback("Too many on-demand requests - aggressive buffer refill")
            self._aggressive_buffer_refill()
            self.consecutive_on_demand_requests = 0
    
    def _aggressive_buffer_refill(self):
        """Aggressively refill the buffer when falling behind"""
        # Request frames more aggressively
        available_slots = self.buffer_size - len(self.frame_buffer) - len(self.pending_results)
        if available_slots > 0:
            # Request up to 3x workers worth of frames
            batch_size = min(available_slots, self.num_workers * 3)
            frames_to_request = min(batch_size, self.total_frames - self.next_frame_to_request + 1)
            
            # NEW: Prioritize frames during aggressive refill
            priority_frames_to_request = []
            regular_frames_to_request = []
            
            # First, check priority frames
            for frame_num in sorted(self.priority_frames):
                if (frame_num not in self.frame_buffer and 
                    frame_num not in self.pending_results and 
                    frame_num <= self.frames_to_generate):  # Only process frames that can be generated
                    priority_frames_to_request.append(frame_num)
                    if len(priority_frames_to_request) >= frames_to_request:
                        break
            
            # Then fill with sequential frames
            remaining_slots = frames_to_request - len(priority_frames_to_request)
            for _ in range(remaining_slots):
                if self.next_frame_to_request <= self.frames_to_generate:  # Only generate frames within the limit
                    regular_frames_to_request.append(self.next_frame_to_request)
                    self.next_frame_to_request += 1
            
            # Request priority frames first
            for frame_number in priority_frames_to_request:
                work_item = self.work_items[frame_number - 1]
                async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
                self.pending_results[frame_number] = async_result
                self.frames_requested += 1
                self.priority_frames.discard(frame_number)
            
            # Then request regular frames
            for frame_number in regular_frames_to_request:
                work_item = self.work_items[frame_number - 1]
                async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
                self.pending_results[frame_number] = async_result
                self.frames_requested += 1
    
    def make_frame(self, t):
        """
        Function called by moviepy to get frame at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            numpy array representing the frame
        """
        # Calculate frame number from time
        video_fps = float(self.json_data.get('video_fps', 30))
        frame_number = int(t * video_fps) + 1
        
        # Clamp frame number to valid range
        frame_number = max(1, min(frame_number, self.total_frames))
        
        # NEW: Handle cloned frames (frames beyond the extended video length)
        if frame_number > self.frames_to_generate:
            # This is a cloned frame - return the last rendered frame
            if self.last_rendered_frame is not None:
                return self.last_rendered_frame.copy()  # Return copy to avoid modifying original
            else:
                # Try to get the last generated frame from the buffer
                if self.frames_to_generate in self.frame_buffer:
                    # Store it for future use and return a copy
                    self.last_rendered_frame = self.frame_buffer[self.frames_to_generate].copy()
                    return self.last_rendered_frame.copy()
                
                # Try to collect any ready frames that might include the last frame
                self._collect_ready_frames()
                
                if self.frames_to_generate in self.frame_buffer:
                    # Store it for future use and return a copy
                    self.last_rendered_frame = self.frame_buffer[self.frames_to_generate].copy()
                    return self.last_rendered_frame.copy()
                
                # Last resort: try to use last successfully returned frame, or black frame if not available
                if self.last_successfully_returned_frame is not None:
                    print(f"Warning: Cloned frame {frame_number} requested but last frame {self.frames_to_generate} not ready yet, using last successful frame")
                    return self.last_successfully_returned_frame.copy()
                else:
                    print(f"Warning: Cloned frame {frame_number} requested but last frame {self.frames_to_generate} not ready yet, using black frame")
                    height = int(self.json_data.get('video_resolution_y', 1080))
                    width = int(self.json_data.get('video_resolution_x', 1920))
                    return np.zeros((height, width, 3), dtype=np.uint8)
        
        # NEW: Update frame prioritization based on sequential access pattern
        if frame_number > self.last_requested_frame:
            # Sequential forward access - prioritize upcoming frames
            upcoming_frames = range(frame_number + 1, min(frame_number + 10, min(self.frames_to_generate, self.total_frames) + 1))
            self.priority_frames.update(upcoming_frames)
        elif frame_number < self.last_requested_frame:
            # Backward access - might be seeking, prioritize nearby frames
            nearby_frames = range(max(1, frame_number - 5), min(frame_number + 15, min(self.frames_to_generate, self.total_frames) + 1))
            self.priority_frames.update(nearby_frames)
        
        self.last_requested_frame = frame_number
        
        # Collect any ready frames first
        self._collect_ready_frames()
        
        # IMPROVED: Proactive buffer maintenance with better thresholds
        buffer_ratio = len(self.frame_buffer) / max(1, self.buffer_size)
        
        # If buffer is getting low, request more frames proactively
        if buffer_ratio < 0.4:  # More aggressive threshold (was 0.5)
            self._request_more_frames()
        
        # NEW: Prioritize frames that are likely to be requested soon
        if frame_number in self.priority_frames:
            self.priority_frames.discard(frame_number)  # Remove from priority set
        
        # If frame is not in buffer or pending, request it specifically
        if frame_number not in self.frame_buffer and frame_number not in self.pending_results:
            # Track on-demand requests
            self.consecutive_on_demand_requests += 1
            
            # Request this specific frame if it hasn't been requested yet AND it's within the generateable range
            if frame_number <= self.frames_to_generate:  # Only request frames that can be generated
                if self.consecutive_on_demand_requests > 1:
                    write_debug_log(f"Requesting missing frame {frame_number} on-demand (consecutive: {self.consecutive_on_demand_requests})")
                work_item = self.work_items[frame_number - 1]
                async_result = self.pool.apply_async(_streaming_frame_worker, (work_item,))
                self.pending_results[frame_number] = async_result
                
                # Also update our tracking
                if frame_number > self.next_frame_to_request:
                    self.next_frame_to_request = frame_number + 1
            # Note: If frame_number > self.frames_to_generate, it's a cloned frame that will be handled above
        else:
            # Reset consecutive on-demand counter if frame was found
            self.consecutive_on_demand_requests = 0
        
        # Wait for the specific frame we need
        max_wait_time = 60  # Maximum wait time in seconds (hardcoded to 60 seconds)
        wait_start = time.time()
        
        while frame_number not in self.frame_buffer:
            # Check if we're still waiting for this frame to be generated
            if frame_number in self.pending_results:
                # Wait a bit and check again
                time.sleep(0.1)
                self._collect_ready_frames()
                
                # IMPROVED: More aggressive buffer maintenance during waiting
                if buffer_ratio < 0.3:  # Even more aggressive during waiting
                    self._request_more_frames()
                
                # Check timeout
                if time.time() - wait_start > max_wait_time:
                    print(f"Timeout waiting for frame {frame_number} (timeout: {max_wait_time}s), using last successful frame instead of black frame")
                    # Try to return the last successfully returned frame instead of black frame
                    if self.last_successfully_returned_frame is not None:
                        return self.last_successfully_returned_frame.copy()
                    
                    # Try to get the previous frame from buffer (frame_number - 1, frame_number - 2, etc.)
                    for fallback_frame_num in range(frame_number - 1, max(1, frame_number - 10), -1):
                        if fallback_frame_num in self.frame_buffer:
                            fallback_frame = self.frame_buffer[fallback_frame_num].copy()
                            print(f"Using frame {fallback_frame_num} as fallback for frame {frame_number}")
                            return fallback_frame
                    
                    # Last resort: return black frame if no fallback available
                    print(f"No fallback frame available, using black frame")
                    height = int(self.json_data.get('video_resolution_y', 1080))
                    width = int(self.json_data.get('video_resolution_x', 1920))
                    return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                # This really shouldn't happen now, but just in case
                print(f"Frame {frame_number} disappeared from pending, using last successful frame instead of black frame")
                # Try to return the last successfully returned frame instead of black frame
                if self.last_successfully_returned_frame is not None:
                    return self.last_successfully_returned_frame.copy()
                
                # Try to get the previous frame from buffer
                for fallback_frame_num in range(frame_number - 1, max(1, frame_number - 10), -1):
                    if fallback_frame_num in self.frame_buffer:
                        fallback_frame = self.frame_buffer[fallback_frame_num].copy()
                        print(f"Using frame {fallback_frame_num} as fallback for frame {frame_number}")
                        return fallback_frame
                
                # Last resort: return black frame if no fallback available
                print(f"No fallback frame available, using black frame")
                height = int(self.json_data.get('video_resolution_y', 1080))
                width = int(self.json_data.get('video_resolution_x', 1920))
                return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get the frame and remove it from buffer to save memory
        frame_array = self.frame_buffer.pop(frame_number)
        
        # NEW: Store the last rendered frame for cloning if this is the final generated frame
        if frame_number == self.frames_to_generate:
            self.last_rendered_frame = frame_array.copy()  # Store copy for cloning
        
        # Track the last successfully returned frame for fallback on timeout
        self.last_successfully_returned_frame = frame_array.copy()
        
        # MEMORY MANAGEMENT: Periodic garbage collection in parent process
        # This helps clean up frame arrays that have been consumed from the Manager.dict()
        # and prevents memory fragmentation, especially important on Linux.
        if frame_number % 100 == 0:
            gc.collect()
        
        return frame_array
    
    def cleanup(self):
        """Clean up multiprocessing resources"""
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None


def create_video_streaming(json_data, route_time_per_frame, combined_route_data, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True, user=None):
    """
    Create video by streaming frames directly to ffmpeg without saving to disk.
    
    Args:
        json_data (dict): Job data containing video parameters
        route_time_per_frame (float): Time per frame in seconds
        combined_route_data (dict): Combined route data
        progress_callback (callable, optional): Function to call with progress updates
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for map images
        shared_route_cache (dict, optional): Shared memory cache for route images
        gpx_time_per_video_time (float, optional): GPX time per video time ratio
        gpu_rendering (bool, optional): Whether to use GPU rendering (default: True)
        user (str, optional): API key for status updates at 10% intervals (username misleading here, it's the "username" of the renderer used as an API key - not the user to which the job belongs)
    
    Returns:
        str: Path to the created video file, or None if failed
    """
    try:
        if debug_callback:
            debug_callback("Starting streaming video creation")
        
        # Calculate video parameters
        video_length = float(json_data.get('video_length', 30))
        video_fps = float(json_data.get('video_fps', 30))
        
        # SIMULTANEOUS MODE ENDING FIX: Detect if we're in simultaneous mode
        is_simultaneous_mode = False
        all_routes = combined_route_data.get('all_routes', None)
        if all_routes and len(all_routes) > 1:
            # Check if routes have different route_index values (truly different routes vs split tracks)
            route_indices = set()
            for route_data in all_routes:
                route_points = route_data.get('combined_route', [])
                if route_points:
                    first_point = route_points[0]
                    route_indices.add(first_point.route_index)
            
            is_simultaneous_mode = len(route_indices) > 1
        
        # Calculate ending parameters based on mode
        if is_simultaneous_mode:
            # SIMULTANEOUS MODE: Each route handles its own completion and tail fade-out
            # We don't need the sequential tail-only phase, just add 5 seconds of cloned ending
            tail_length = 0  # No sequential tail phase needed
            ending_duration_required = 5.0  # Always 5 seconds of ending total
            cloned_ending_duration = ending_duration_required  # All 5 seconds via cloned frames
            
            # Total video length: original video + cloned ending
            extended_video_length = video_length  # No tail phase
            final_video_length = extended_video_length + cloned_ending_duration
            
            if debug_callback:
                debug_callback(f"SIMULTANEOUS MODE: Creating video: {video_length}s route + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
                debug_callback(f"Each route handles its own completion and tail fade-out individually")
        else:
            # SEQUENTIAL MODE: Use the original tail-only phase logic
            tail_length = int(json_data.get('tail_length', 0))  # Ensure tail_length is int
            ending_duration_required = 5.0  # Always 5 seconds of ending total
            cloned_ending_duration = max(0.0, ending_duration_required - tail_length)  # Additional time needed via cloned frames
            
            # Total video length includes: original video + tail frames + cloned ending
            extended_video_length = video_length + tail_length if tail_length > 0 else video_length
            final_video_length = extended_video_length + cloned_ending_duration
            
            if debug_callback:
                if tail_length > 0 and cloned_ending_duration > 0:
                    debug_callback(f"SEQUENTIAL MODE: Creating video: {video_length}s route + {tail_length}s tail + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
                elif tail_length > 0:
                    debug_callback(f"SEQUENTIAL MODE: Creating video: {video_length}s route + {tail_length}s tail = {final_video_length}s total (no additional frames needed)")
                else:
                    debug_callback(f"SEQUENTIAL MODE: Creating video: {video_length}s route + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
        
        # Calculate frame counts
        original_frames = int(video_length * video_fps)
        tail_frames = int(tail_length * video_fps) if tail_length > 0 else 0  
        cloned_frames = int(cloned_ending_duration * video_fps)
        total_frames = int(final_video_length * video_fps)

        if debug_callback:
            if tail_length > 0 and cloned_ending_duration > 0:
                debug_callback(f"Frame breakdown: {original_frames} route + {tail_frames} tail + {cloned_frames} cloned = {total_frames} total frames")
            elif tail_length > 0:
                debug_callback(f"Frame breakdown: {original_frames} route + {tail_frames} tail = {total_frames} total frames")
            else:
                debug_callback(f"Frame breakdown: {original_frames} route + {cloned_frames} cloned = {total_frames} total frames")
        
        # Generate output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Determine output directory and filename based on test job flag
        is_test_job = json_data.get('test_job', False)
        job_id = json_data.get('job_id', '')
        
        if is_test_job:
            # Test job: save in temporary_files/test jobs with timestamp filename
            output_dir = os.path.join('temporary files', 'test jobs')
            output_filename = f"route_video_{timestamp}.mp4"
        else:
            # Regular job: save in temporary_files/job_id with route_name filename
            output_dir = os.path.join('temporary files', str(job_id))
            route_name = json_data.get('route_name', 'route_video')
            
            # Clean the route name for use as filename (remove invalid characters)
            clean_route_name = re.sub(r'[<>:"/\\|?*]', '_', route_name)
            clean_route_name = clean_route_name.strip()
            
            # If route name is empty or only contains invalid characters, fall back to timestamp
            if not clean_route_name or clean_route_name == '_':
                output_filename = f"route_video_{timestamp}.mp4"
            else:
                output_filename = f"{clean_route_name}.mp4"
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        if debug_callback:
            debug_callback(f"Output video will be saved to: {output_path}")
            if is_test_job:
                debug_callback("Test job mode: Video will be saved in test jobs directory")
            else:
                debug_callback(f"Regular job mode: Video will be saved in job directory: {job_id}")
                debug_callback(f"Video filename: {output_filename}")
        
        # Create the frame generator
        frame_generator = StreamingFrameGenerator(json_data, route_time_per_frame, combined_route_data, max_workers, shared_map_cache, shared_route_cache, progress_callback, gpx_time_per_video_time, debug_callback, user)
        
        # Initial progress update
        if progress_callback:
            progress_callback("progress_bar_frames", 0, "Starting frame generation")
        
        # Create the video clip
        clip = VideoClip(frame_generator.make_frame, duration=frame_generator.final_video_length)
        
        # Write the video file
        if debug_callback:
            debug_callback("Starting video encoding")
        
        # Use GPU rendering setting from config
        if debug_callback:
            if gpu_rendering:
                debug_callback("Using GPU rendering (NVIDIA NVENC)")
            else:
                debug_callback("Using CPU rendering (libx264)")
        
        try:
            # Use moviepy to write the video
            if gpu_rendering:
                # GPU rendering with NVIDIA NVENC
                # Quality/File Size Options:
                # - Lower bitrate = smaller file, lower quality
                # - Higher bitrate = larger file, higher quality
                # - Presets p1-p7: p1 = fastest, p7 = best quality (slowest)
                # - p4 is a good balance between speed and quality
                try:
                    # Suppress MoviePy output when debug is off
                    with suppress_moviepy_output():
                        # Create custom logger that respects debug settings
                        moviepy_logger = MoviePyDebugLogger(debug_callback) if debug_callback else None
                        clip.write_videofile(
                            output_path,
                            fps=video_fps,
                            codec='h264_nvenc',  # NVIDIA GPU encoder
                            bitrate='15000k',    # Can be reduced for smaller files (e.g., '8000k', '5000k')
                            audio=False,
                            threads=max_workers,
                            logger=moviepy_logger if config.debug_logging else None,  # Only use logger when debug is enabled
                            ffmpeg_params=[
                                '-pix_fmt', 'yuv420p',
                                '-preset', 'p5',          # Options: p1 (fastest) to p7 (best quality)
                                '-rc', 'vbr',             # Rate control: vbr (variable), cbr (constant), cqp (constant quality)
                                '-cq', '23',              # Constant quality (18-51, lower = better quality, larger file)
                                '-b:v', '15000k',         # Target bitrate
                                '-maxrate', '20000k',     # Maximum bitrate
                                '-bufsize', '30000k'      # Buffer size for rate control
                            ]
                        )
                except Exception as nvenc_error:
                    # Fallback to CPU encoding if NVENC fails
                    if debug_callback:
                        debug_callback(f"NVENC encoding failed ({nvenc_error}), falling back to CPU encoding")
                    gpu_rendering = False  # Will trigger CPU encoding below
            
            # CPU rendering with libx264 (either by choice or as fallback from failed NVENC)
            if not gpu_rendering:
                # Suppress MoviePy output when debug is off
                with suppress_moviepy_output():
                    # Create custom logger that respects debug settings
                    moviepy_logger = MoviePyDebugLogger(debug_callback) if debug_callback else None
                    clip.write_videofile(
                        output_path,
                        fps=video_fps,
                        codec='libx264',
                        bitrate='15000k',
                        audio=False,
                        threads=max_workers,
                        logger=moviepy_logger if config.debug_logging else None,  # Only use logger when debug is enabled
                        ffmpeg_params=[
                            '-pix_fmt', 'yuv420p',
                            '-preset', 'medium',        # Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
#                        '-tune', 'animation',       # Options: film, animation, grain, stillimage, fastdecode, zerolatency
                            '-crf', '25',               # Constant Rate Factor (0-51, lower = better quality, larger file)
                            '-profile:v', 'high',       # Profile: baseline, main, high, high10, high422, high444
                            '-level', '4.1'             # H.264 level for compatibility
                        ]
                    )
            
            # Final frame generation progress update
            if progress_callback:
                progress_callback("progress_bar_frames", 100, "Frame generation complete")
            
            if debug_callback:
                debug_callback(f"Streaming video creation complete: {output_path}")
            
            # Generate thumbnail for non-test jobs
            if not is_test_job:
                try:
                    if debug_callback:
                        debug_callback("Generating thumbnail from last frame...")
                    
                    # Extract the last frame using ffmpeg
                    import subprocess
                    
                    # Create temporary frame file
                    temp_frame_path = os.path.join(output_dir, 'temp_last_frame.png')
                    
                    # Get the correct ffmpeg executable for the current platform
                    ffmpeg_executable = get_ffmpeg_executable()
                    
                    # Use ffmpeg to extract the last frame
                    cmd = [
                        ffmpeg_executable,
                        '-sseof', '-1',  # Seek to 1 second before end
                        '-i', output_path,
                        '-vframes', '1',
                        '-y',  # Overwrite output file
                        temp_frame_path
                    ]
                    
                    if debug_callback:
                        debug_callback(f"Running FFmpeg command: {' '.join(cmd)}")
                    
                    # Run ffmpeg command
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    if debug_callback:
                        debug_callback(f"FFmpeg command executed successfully")
                    
                    # Load the extracted frame and resize it
                    from PIL import Image
                    pil_image = Image.open(temp_frame_path)
                    
                    # Calculate new width to maintain aspect ratio with 320px height
                    original_width, original_height = pil_image.size
                    new_height = 320
                    new_width = int((original_width / original_height) * new_height)
                    
                    # Resize the image
                    resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save as thumbnail.png in the same directory
                    thumbnail_path = os.path.join(output_dir, 'thumbnail.png')
                    resized_image.save(thumbnail_path, 'PNG')
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_frame_path)
                    except:
                        pass  # Ignore cleanup errors
                    
                    if debug_callback:
                        debug_callback(f"Thumbnail generated: {thumbnail_path} ({new_width}x{new_height})")
                        
                except subprocess.CalledProcessError as e:
                    if log_callback:
                        log_callback(f"Warning: FFmpeg failed to extract last frame: {e}")
                        log_callback(f"FFmpeg stderr: {e.stderr}")
                except Exception as e:
                    if log_callback:
                        log_callback(f"Warning: Failed to generate thumbnail: {str(e)}")
            
            return output_path
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error during video encoding: {str(e)}")
            raise e
        finally:
            # Always clean up resources
            clip.close()
            frame_generator.cleanup()
            
            # Additional cleanup for MoviePy and FFmpeg processes
            try:
                import psutil
                
                # Find and terminate any remaining FFmpeg processes that might be hanging
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # Check if this is an FFmpeg process related to our video generation
                        if proc.info['name'] and 'ffmpeg' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline']
                            if cmdline and any('moviepy' in str(arg).lower() or 'temp' in str(arg).lower() for arg in cmdline):
                                if debug_callback:
                                    debug_callback(f"Terminating hanging FFmpeg process: PID {proc.info['pid']}")
                                proc.terminate()
                                proc.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                        # Process already terminated or we don't have permission
                        pass
                    except Exception as e:
                        if log_callback:
                            log_callback(f"Warning: Error checking FFmpeg process: {str(e)}")
                
                if debug_callback:
                    debug_callback("MoviePy and FFmpeg processes cleaned up")
                    
            except Exception as e:
                if log_callback:
                    log_callback(f"Warning: Error during MoviePy cleanup: {str(e)}")
            
    except Exception as e:
        if log_callback:
            log_callback(f"Error in streaming video creation: {str(e)}")
        return None


def cache_video_frames_for_video(json_data, route_time_per_frame, combined_route_data, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True, user=None):
    """
    Cache video frames for video generation using streaming approach.
    
    Args:
        json_data (dict): Job data containing video parameters
        route_time_per_frame (float): Time per frame in seconds
        combined_route_data (dict): Combined route data
        progress_callback (callable, optional): Function to call with progress updates
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for map images
        shared_route_cache (dict, optional): Shared memory cache for route images
        gpx_time_per_video_time (float, optional): GPX time per video time ratio
        gpu_rendering (bool, optional): Whether to use GPU rendering (default: True)
        user (str, optional): API key for status updates at 10% intervals
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting video frame generation with streaming approach...")
        
        # Use provided shared map cache or create new one if not provided
        if shared_map_cache is None:
            with Manager() as manager:
                shared_map_cache = manager.dict()
        
        # Create video using streaming approach
        video_path = create_video_streaming(
            json_data, route_time_per_frame, combined_route_data, 
            progress_callback, log_callback, debug_callback, max_workers, shared_map_cache, shared_route_cache, gpx_time_per_video_time, gpu_rendering, user
        )
        
        if video_path:
            # Calculate extended video length including tail and cloned ending
            video_length = float(json_data.get('video_length', 30))
            video_fps = float(json_data.get('video_fps', 30))
            
            # SIMULTANEOUS MODE ENDING FIX: Detect if we're in simultaneous mode
            is_simultaneous_mode = False
            all_routes = combined_route_data.get('all_routes', None)
            if all_routes and len(all_routes) > 1:
                # Check if routes have different route_index values (truly different routes vs split tracks)
                route_indices = set()
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    if route_points:
                        first_point = route_points[0]
                        route_indices.add(first_point.route_index)
                
                is_simultaneous_mode = len(route_indices) > 1
            
            # Calculate ending parameters based on mode
            if is_simultaneous_mode:
                # SIMULTANEOUS MODE: Each route handles its own completion and tail fade-out
                # We don't need the sequential tail-only phase, just add 5 seconds of cloned ending
                tail_length = 0  # No sequential tail phase needed
                ending_duration_required = 5.0  # Always 5 seconds of ending total
                cloned_ending_duration = ending_duration_required  # All 5 seconds via cloned frames
                
                # Total video length: original video + cloned ending
                extended_video_length = video_length  # No tail phase
                final_video_length = extended_video_length + cloned_ending_duration
            else:
                # SEQUENTIAL MODE: Use the original tail-only phase logic
                tail_length = int(json_data.get('tail_length', 0))  # Ensure tail_length is int
                ending_duration_required = 5.0  # Always 5 seconds of ending total
                cloned_ending_duration = max(0.0, ending_duration_required - tail_length)  # Additional time needed via cloned frames
                
                # Total video length includes: original video + tail frames + cloned ending
                extended_video_length = video_length + tail_length if tail_length > 0 else video_length
                final_video_length = extended_video_length + cloned_ending_duration
            
            total_frames = int(final_video_length * video_fps)
            frames_generated = int(extended_video_length * video_fps)  # Only frames that were actually generated
            cloned_frames = int(cloned_ending_duration * video_fps)
            
            if debug_callback:
                if is_simultaneous_mode:
                    debug_callback(f"SIMULTANEOUS MODE: Video creation completed: {frames_generated} frames generated + {cloned_frames} frames cloned = {total_frames} total frames")
                else:
                    debug_callback(f"SEQUENTIAL MODE: Video creation completed: {frames_generated} frames generated + {cloned_frames} frames cloned = {total_frames} total frames")
            
            return {
                'total_frames_created': frames_generated,  # Frames actually generated (not including cloned)
                'total_frames_failed': 0,
                'total_frames': total_frames,  # Total frames in final video (including cloned)
                'cloned_frames': cloned_frames,  # New: number of cloned frames
                'success': True,
                'video_path': video_path,
                'shared_map_cache': shared_map_cache,
                'results': []
            }
        else:
            return None
                
    except Exception as e:
        if log_callback:
            log_callback(f"Error in cache_video_frames_for_video: {str(e)}")
        return None


def cache_video_frames(json_data=None, combined_route_data=None, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True, user=None):
    """
    Cache video frames for video generation.
    
    Args:
        json_data (dict, optional): Job data containing video parameters
        combined_route_data (dict, optional): Combined route data containing gpx_time_per_video_time
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for map images
        shared_route_cache (dict, optional): Shared memory cache for route images
        gpx_time_per_video_time (float, optional): GPX time per video time ratio
        gpu_rendering (bool, optional): Whether to use GPU rendering (default: True)
        user (str, optional): API key for status updates at 10% intervals
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting video generation")
        
        if progress_callback:
            progress_callback("progress_bar_frames", 0, "Starting video generation")
        
        # Load job data if not provided
        if json_data is None:
            data_file = Path("data.json")
            if not data_file.exists():
                if log_callback:
                    log_callback("Error: data.json not found")
                return None
            
            with open(data_file, 'r') as f:
                json_data = json.load(f)
        
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
        
        if combined_route_data is None:
            if log_callback:
                log_callback("Error: combined_route_data parameter is required")
            return None
        
        # Create video using streaming approach
        cache_result = cache_video_frames_for_video(
            json_data, route_time_per_frame, combined_route_data, 
            progress_callback, log_callback, debug_callback, max_workers, shared_map_cache, shared_route_cache, gpx_time_per_video_time, gpu_rendering, user
        )
        
        if cache_result is None:
            if log_callback:
                log_callback("Error: Could not create video")
            return None
        
        # Return the result directly since video creation is already handled
        return {
            'route_time_per_frame': route_time_per_frame,
            'combined_route_data': combined_route_data,
            'cache_result': cache_result,
            'success': True
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in video generation: {str(e)}")
        return None 