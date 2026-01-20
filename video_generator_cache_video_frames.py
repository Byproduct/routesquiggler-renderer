"""
Functions related to generating video frames.
This file contains things related to preparation and multiprocessing.
The actual map plotting is in video_generator_create_single_frame.py.
"""

import json
import os
import sys
import time
import gc
import multiprocessing
import numpy as np
from contextlib import contextmanager
from io import StringIO

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for video frame caching

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
from PIL import Image
from pathlib import Path

# Increase PIL image size limit to 200 megapixels to handle large renders
Image.MAX_IMAGE_PIXELS = 200_000_000
from multiprocessing import Pool, Manager
from moviepy import VideoClip
from video_generator_calculate_bounding_boxes import calculate_route_time_per_frame
from video_generator_create_single_frame import generate_video_frame_in_memory
from video_generator_create_single_frame_utils import hex_to_rgba
from video_generator_create_combined_route import RoutePoint
from image_generator_utils import calculate_resolution_scale
from write_log import write_log, write_debug_log
from config import config


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
                    
                    if is_tail_only_frame:
                        # SIMULTANEOUS MODE TAIL FIX: Each route should have its own individual tail fade-out
                        # Calculate when this specific route ends (accounting for its start delay)
                        route_end_time = route_points[-1].accumulated_time if route_points else 0
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
            else:
                # Single route group with multiple tracks - process all tracks concurrently
                points_for_frame = []
                
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    
                    if is_tail_only_frame:
                        # For tail-only frames, use the final route coordinates
                        final_route_target_time = (original_route_end_frame - 1) * route_time_per_frame
                        
                        # OPTIMIZED: Use binary search + list slicing instead of linear search + appends
                        cutoff_index = _binary_search_cutoff_index(route_points, final_route_target_time)
                        route_points_for_frame = route_points[:cutoff_index]
                    else:
                        # Normal frame - find the points up to the target time
                        # OPTIMIZED: Use binary search + list slicing instead of linear search + appends
                        cutoff_index = _binary_search_cutoff_index(route_points, target_time_route)
                        route_points_for_frame = route_points[:cutoff_index]
                    
                    # Only add this route's points if it has any points
                    if route_points_for_frame:
                        points_for_frame.append(route_points_for_frame)
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
        
        # Check if we have any points (either as a list of lists or a single list)
        has_points = False
        if all_routes and len(all_routes) > 1:
            # Multiple routes mode - check if any route has points
            has_points = any(len(route_points) > 0 for route_points in points_for_frame)
        else:
            # Single route mode - check if the list has points
            has_points = len(points_for_frame) > 0
        
        if not has_points:
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
        
        # Generate frame in memory instead of saving to disk
        frame_array = generate_video_frame_in_memory(
            frame_number, points_for_frame, json_data, shared_map_cache, filename_to_rgba, gpx_time_per_video_time, stamp_array, target_time_video, shared_route_cache, virtual_leading_time, route_specific_tail_info
        )
        
        # OPTIMIZED: More aggressive memory cleanup for computationally intensive tasks
        # Clean up every 5 frames instead of 10 to prevent memory accumulation
        if frame_number % 5 == 0:
            # Force matplotlib to clean up any lingering figures
            plt.close('all')
            
            # Clear matplotlib's internal caches periodically
            if hasattr(plt, '_pylab_helpers'):
                plt._pylab_helpers.Gcf.destroy_all()
            
            # Force garbage collection to free up memory
            gc.collect()
        
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
    def __init__(self, json_data, route_time_per_frame, combined_route_data, max_workers=None, shared_map_cache=None, shared_route_cache=None, progress_callback=None, gpx_time_per_video_time=None, debug_callback=None):
        self.json_data = json_data
        self.route_time_per_frame = route_time_per_frame
        self.combined_route_data = combined_route_data
        self.shared_map_cache = shared_map_cache
        self.shared_route_cache = shared_route_cache
        self.progress_callback = progress_callback
        self.gpx_time_per_video_time = gpx_time_per_video_time
        self.debug_callback = debug_callback
        
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
        # OPTIMIZED: Better worker pool configuration for high computational load
        # Increase maxtasksperchild to reduce worker recycling overhead
        # Each worker will handle at most 100 frames before being recycled (was 50)
        # This reduces the overhead of worker recycling for computationally intensive tasks
        self.pool = Pool(processes=self.num_workers, maxtasksperchild=100)
        
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
        
        return frame_array
    
    def cleanup(self):
        """Clean up multiprocessing resources"""
        if self.pool:
            self.pool.terminate()
            self.pool.join()
            self.pool = None


def create_video_streaming(json_data, route_time_per_frame, combined_route_data, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True):
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
        frame_generator = StreamingFrameGenerator(json_data, route_time_per_frame, combined_route_data, max_workers, shared_map_cache, shared_route_cache, progress_callback, gpx_time_per_video_time, debug_callback)
        
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
                    
                    # Use ffmpeg to extract the last frame
                    cmd = [
                        'ffmpeg',
                        '-sseof', '-1',  # Seek to 1 second before end
                        '-i', output_path,
                        '-vframes', '1',
                        '-y',  # Overwrite output file
                        temp_frame_path
                    ]
                    
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


def cache_video_frames_for_video(json_data, route_time_per_frame, combined_route_data, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True):
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
            progress_callback, log_callback, debug_callback, max_workers, shared_map_cache, shared_route_cache, gpx_time_per_video_time, gpu_rendering
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


def cache_video_frames(json_data=None, combined_route_data=None, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True):
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
            progress_callback, log_callback, debug_callback, max_workers, shared_map_cache, shared_route_cache, gpx_time_per_video_time, gpu_rendering
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