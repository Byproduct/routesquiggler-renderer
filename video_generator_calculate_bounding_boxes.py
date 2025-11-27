"""
Video generation bounding box calculation for the Route Squiggler render client.
This module handles calculating unique bounding boxes needed for video frame generation.
"""

import multiprocessing
import math
import pickle
from pathlib import Path
from datetime import timedelta
from write_log import write_log, write_debug_log


def split_coordinates_at_longitude_wrap(lats, lons):
    """
    Split coordinates into segments when they cross the 180°/-180° longitude boundary.
    
    Args:
        lats: List of latitude coordinates
        lons: List of longitude coordinates
        
    Returns:
        List of (lats_segment, lons_segment) tuples, where each segment doesn't cross the boundary
    """
    if len(lats) < 2 or len(lons) < 2:
        return [(lats, lons)]
    
    segments = []
    current_lats = [lats[0]]
    current_lons = [lons[0]]
    wrap_count = 0
    
    for i in range(1, len(lats)):
        prev_lon = lons[i-1]
        curr_lon = lons[i]
        
        # Check if we've crossed the 180°/-180° boundary
        lon_diff = curr_lon - prev_lon
        
        # If the difference is greater than 180 degrees, we've crossed the boundary
        if abs(lon_diff) > 180:
            wrap_count += 1
            # Save the current segment
            if len(current_lats) > 1:  # Only add segments with at least 2 points
                segments.append((current_lats.copy(), current_lons.copy()))
            
            # Start a new segment
            current_lats = [lats[i]]
            current_lons = [lons[i]]
        else:
            # Continue the current segment
            current_lats.append(lats[i])
            current_lons.append(lons[i])
    
    # Add the final segment
    if len(current_lats) > 1:
        segments.append((current_lats, current_lons))
    elif len(current_lats) == 1 and segments:  # Single point, add to last segment if it exists
        segments[-1][0].append(current_lats[0])
        segments[-1][1].append(current_lons[0])
    
    # If no segments were created (e.g., single point or no wrapping), return original
    if not segments:
        return [(lats, lons)]
    
    # Log if wrapping was detected
    if wrap_count > 0:
        print(f"Video mode: Longitude wrapping detected: coordinates split into {len(segments)} segments after {wrap_count} wrap(s)")
    
    return segments


def calculate_bounding_box_for_wrapped_coordinates(lats, lons, padding_percent=0.1):
    """
    Calculate bounding box for coordinates that may wrap around the globe.
    
    Args:
        lats: List of latitude coordinates
        lons: List of longitude coordinates
        padding_percent: Padding percentage
        
    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) in GPS coordinates
    """
    if not lats or not lons:
        return None
    
    # Split coordinates at longitude wrap points
    segments = split_coordinates_at_longitude_wrap(lats, lons)
    
    if len(segments) == 1:
        # No wrapping, use normal calculation
        segment_lats, segment_lons = segments[0]
        lon_min, lon_max = min(segment_lons), max(segment_lons)
        lat_min, lat_max = min(segment_lats), max(segment_lats)
        
        # Ensure minimum latitude distance of 0.01 degrees - this gives a sensible minimum map view and avoids the need for maximally zoomed tiles for a tiny portion of the route.
        lat_distance = lat_max - lat_min
        if lat_distance < 0.01:
            adjustment = (0.01 - lat_distance) / 2
            lat_min -= adjustment
            lat_max += adjustment
    else:
        # Wrapping detected - calculate bounding box for each segment and find the union
        all_lon_mins = []
        all_lon_maxs = []
        all_lat_mins = []
        all_lat_maxs = []
        
        for segment_lats, segment_lons in segments:
            all_lon_mins.append(min(segment_lons))
            all_lon_maxs.append(max(segment_lons))
            all_lat_mins.append(min(segment_lats))
            all_lat_maxs.append(max(segment_lats))
        
        # For longitude, we need to handle the wrap
        # Find the segment with the largest span
        lon_spans = [max_lon - min_lon for min_lon, max_lon in zip(all_lon_mins, all_lon_maxs)]
        largest_span_idx = lon_spans.index(max(lon_spans))
        
        # Use the largest segment for longitude bounds
        lon_min = all_lon_mins[largest_span_idx]
        lon_max = all_lon_maxs[largest_span_idx]
        
        # For latitude, use the global min/max across all segments
        lat_min = min(all_lat_mins)
        lat_max = max(all_lat_maxs)
    
    # Calculate center point
    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2
    
    # Calculate spans with basic padding
    lon_span_raw = lon_max - lon_min
    lat_span_raw = lat_max - lat_min
    
    # Handle case where all points are at the same location (zero span)
    if lon_span_raw == 0 and lat_span_raw == 0:
        # Single point - create a small default bounding box around it
        default_span = 0.001  # About 100 meters at the equator
        lon_min_padded = lon_center - default_span
        lon_max_padded = lon_center + default_span
        lat_min_padded = lat_center - default_span
        lat_max_padded = lat_center + default_span
    elif lon_span_raw == 0:
        # All points have same longitude - create narrow horizontal box
        default_lon_span = lat_span_raw * 0.1  # Make it 10% of lat span
        lon_min_padded = lon_center - default_lon_span
        lon_max_padded = lon_center + default_lon_span
        # Use normal lat span calculation
        lat_padding = lat_span_raw * padding_percent
        lat_min_padded = lat_min - lat_padding
        lat_max_padded = lat_max + lat_padding
    elif lat_span_raw == 0:
        # All points have same latitude - create narrow vertical box
        default_lat_span = lon_span_raw * 0.1  # Make it 10% of lon span
        lat_min_padded = lat_center - default_lat_span
        lat_max_padded = lat_center + default_lat_span
        # Use normal lon span calculation
        lon_padding = lon_span_raw * padding_percent
        lon_min_padded = lon_min - lon_padding
        lon_max_padded = lon_max + lon_padding
    else:
        # Normal case - both spans are non-zero
        # Apply base padding to both dimensions
        lon_padding = lon_span_raw * padding_percent
        lat_padding = lat_span_raw * padding_percent
        
        # Initial padded bounds
        lon_min_padded = lon_min - lon_padding
        lon_max_padded = lon_max + lon_padding
        lat_min_padded = lat_min - lat_padding
        lat_max_padded = lat_max + lat_padding
        
        # Calculate aspect ratio of the route after initial padding
        lon_span_padded = lon_max_padded - lon_min_padded
        lat_span_padded = lat_max_padded - lat_min_padded
        
        # Adjust for latitude distortion - at the equator, 1 degree longitude ≈ 1 degree latitude
        # But at higher latitudes, 1 degree longitude < 1 degree latitude in terms of distance
        cos_factor = max(0.01, abs(math.cos(math.radians(lat_center))))  # Prevent division by zero
        
        # Calculate the effective aspect ratio of the padded route (accounting for latitude distortion)
        route_aspect_ratio = (lon_span_padded * cos_factor) / lat_span_padded
        
        # Target aspect ratio for 1920x1080 frame
        target_aspect_ratio = 16 / 9  # Standard video aspect ratio (1920/1080)
        
        # If route is more vertical than our target aspect ratio, expand horizontally
        if route_aspect_ratio < target_aspect_ratio:
            # Calculate how much horizontal space we need to add to match target aspect ratio
            desired_lon_span = lat_span_padded * target_aspect_ratio / cos_factor
            # How much to add on each side
            lon_expansion = (desired_lon_span - lon_span_padded) / 2
            # Apply expansion
            lon_min_padded -= lon_expansion
            lon_max_padded += lon_expansion
        
        # If route is more horizontal than our target aspect ratio, expand vertically
        elif route_aspect_ratio > target_aspect_ratio:
            # Calculate how much vertical space we need to add to match target aspect ratio
            desired_lat_span = (lon_span_padded * cos_factor) / target_aspect_ratio
            # How much to add on each side
            lat_expansion = (desired_lat_span - lat_span_padded) / 2
            # Apply expansion
            lat_min_padded -= lat_expansion
            lat_max_padded += lat_expansion
    
    # BOUNDS CHECKING: Ensure longitude stays within valid range
    lon_min_padded = max(-180.0, lon_min_padded)
    lon_max_padded = min(180.0, lon_max_padded)
    
    # BOUNDS CHECKING: Ensure latitude stays within valid range
    lat_min_padded = max(-90.0, lat_min_padded)
    lat_max_padded = min(90.0, lat_max_padded)
    
    # Create the final bounding box (rounded to 6 decimal places for consistency)
    final_bbox = (
        round(lon_min_padded, 6),   # lon_min_padded
        round(lon_max_padded, 6),   # lon_max_padded
        round(lat_min_padded, 6),   # lat_min_padded
        round(lat_max_padded, 6)    # lat_max_padded
    )
    
    return final_bbox


def _binary_search_cutoff_index(route_points, target_time):
    """
    Binary search to find the cutoff index for points up to target_time.
    
    Returns the index of the first point that should be EXCLUDED (i.e., where accumulated_time > target_time).
    This allows for efficient list slicing: route_points[:cutoff_index] gives all valid points.
    
    Args:
        route_points (list): List of route points, chronologically ordered by accumulated_time
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
        
        # Check if this point has the minimum required length and extract accumulated_time
        if len(route_points[mid]) >= 5:
            accumulated_time = route_points[mid][4]  # accumulated_time at index 4
            
            if accumulated_time <= target_time:
                # This point should be included, look for later cutoff point
                left = mid + 1
            else:
                # This point should be excluded, it might be our cutoff point
                result = mid
                right = mid - 1
        else:
            # Point doesn't have enough elements, exclude it
            result = mid
            right = mid - 1
    
    return result


def save_final_bounding_box(final_bbox, log_callback=None, debug_callback=None):
    """
    Save the final bounding box to a file for later use.
    
    Args:
        final_bbox (tuple): Final bounding box as (lon_min, lon_max, lat_min, lat_max)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
    """
    try:
        # Create output directory
        output_dir = Path("temporary files/route")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        bbox_file = output_dir / "final_bounding_box.pkl"
        with open(bbox_file, 'wb') as f:
            pickle.dump(final_bbox, f)
        
        if debug_callback:
            debug_callback(f"Saved final bounding box to {bbox_file}")
            
    except Exception as e:
        if log_callback:
            log_callback(f"Error saving final bounding box: {str(e)}")


def load_final_bounding_box(log_callback=None, debug_callback=None):
    """
    Load the final bounding box from file.
    
    Args:
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
    
    Returns:
        tuple: Final bounding box as (lon_min, lon_max, lat_min, lat_max), or None if not found
    """
    try:
        bbox_file = Path("temporary files/route/final_bounding_box.pkl")
        
        if not bbox_file.exists():
            if log_callback:
                log_callback(f"Warning: Final bounding box file not found: {bbox_file}")
            return None
        
        with open(bbox_file, 'rb') as f:
            final_bbox = pickle.load(f)
        
        if debug_callback:
            debug_callback(f"Loaded final bounding box: {final_bbox}")
        
        return final_bbox
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error loading final bounding box: {str(e)}")
        return None


def calculate_route_time_per_frame(json_data, combined_route_data, log_callback=None, debug_callback=None):
    """
    Calculate the time per frame for route progression based on video parameters and route data.
    
    Args:
        json_data (dict): Job data containing video parameters
        combined_route_data (dict): Combined route data
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
    
    Returns:
        float: Time per frame in seconds, or None if calculation fails
    """
    try:
        # Extract video parameters
        video_length = json_data.get('video_length', 0)
        video_fps = json_data.get('video_fps', 30)
        
        # Ensure numeric types
        try:
            video_length = float(video_length)
            video_fps = float(video_fps)
        except (ValueError, TypeError):
            if log_callback:
                log_callback(f"Error: Could not convert video parameters to numbers - video_length: {video_length}, video_fps: {video_fps}")
            return None
        
        # Calculate total number of frames
        total_frames = video_length * video_fps
        
        if debug_callback:
            debug_callback(f"Video parameters: {video_length} seconds × {video_fps} fps = {total_frames} total frames")
        
        if not combined_route_data:
            if log_callback:
                log_callback("Error: combined_route_data is required")
            return None
        
        # Check if we have multiple routes
        all_routes = combined_route_data.get('all_routes', None)
        
        # Check if routes have different route_index values (truly different routes vs split tracks)
        route_indices = set()
        for route_data in all_routes:
            route_points = route_data.get('combined_route', [])
            if route_points:
                first_point = route_points[0]
                if len(first_point) >= 1:
                    route_index = first_point[0]  # route_index at index 0
                    route_indices.add(route_index)
        
        has_multiple_route_groups = len(route_indices) > 1
        
        if has_multiple_route_groups:
            # STAGGERED ROUTES FIX: Calculate total route time accounting for staggered start times
            # Find the earliest start time and latest end time across all routes
            earliest_start_time = None
            latest_end_time = None
            route_count = len(all_routes)
            
            if debug_callback:
                debug_callback(f"Multiple routes mode: Calculating total time from {route_count} staggered routes")
            
            for i, route_data in enumerate(all_routes):
                route_points = route_data.get('combined_route', [])
                if route_points and len(route_points) > 0:
                    # Get start and end times for this route
                    first_point = route_points[0]
                    last_point = route_points[-1]
                    
                    if len(first_point) >= 4 and len(last_point) >= 5:
                        # Route start time (timestamp)
                        route_start_timestamp = first_point[3]  # timestamp at index 3
                        # Route duration (accumulated_time from last point)
                        route_duration = last_point[4]  # accumulated_time at index 4
                        
                        if route_start_timestamp:
                            # Calculate when this route ends (start + duration)
                            route_end_timestamp = route_start_timestamp + timedelta(seconds=route_duration)
                            
                            # Track earliest start and latest end
                            if earliest_start_time is None or route_start_timestamp < earliest_start_time:
                                earliest_start_time = route_start_timestamp
                            
                            if latest_end_time is None or route_end_timestamp > latest_end_time:
                                latest_end_time = route_end_timestamp
                            
                            if debug_callback:
                                track_name = route_data.get('track_name', f'Route {i}')
                                debug_callback(f"  Route '{track_name}': starts at {route_start_timestamp.strftime('%H:%M:%S')}, duration {route_duration:.1f}s, ends at {route_end_timestamp.strftime('%H:%M:%S')}")
            
            # SIMULTANEOUS MODE FIX: Ensure all routes can complete within the specified video duration
            # The issue is that we need to ensure the latest-starting route has enough time to complete
            # within the video duration, not just that the total span fits
            
            if earliest_start_time and latest_end_time:
                total_span_seconds = (latest_end_time - earliest_start_time).total_seconds()
                
                # Find the route that starts latest and needs the most time to complete
                latest_start_time = None
                max_route_duration = 0.0
                
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    if route_points and len(route_points) > 0:
                        first_point = route_points[0]
                        last_point = route_points[-1]
                        
                        if len(first_point) >= 4 and len(last_point) >= 5:
                            route_start_timestamp = first_point[3]
                            route_duration = last_point[4]
                            
                            if route_start_timestamp:
                                if latest_start_time is None or route_start_timestamp > latest_start_time:
                                    latest_start_time = route_start_timestamp
                                    max_route_duration = route_duration
                
                # Calculate the minimum video duration needed for all routes to complete
                if latest_start_time:
                    # The latest-starting route needs: (video_duration - delay) >= route_duration
                    # So: video_duration >= delay + route_duration
                    delay_from_earliest = (latest_start_time - earliest_start_time).total_seconds()
                    min_video_duration_needed = delay_from_earliest + max_route_duration
                    
                    if debug_callback:
                        debug_callback(f"SIMULTANEOUS MODE ANALYSIS:")
                        debug_callback(f"  Total span of all routes: {total_span_seconds:.1f}s")
                        debug_callback(f"  Latest route starts at: {latest_start_time.strftime('%H:%M:%S')} (delay: {delay_from_earliest:.1f}s)")
                        debug_callback(f"  Latest route duration: {max_route_duration:.1f}s")
                        debug_callback(f"  Minimum video duration needed: {min_video_duration_needed:.1f}s")
                        debug_callback(f"  Actual video duration: {video_length:.1f}s")
                    
                    # Use the minimum video duration needed, but don't exceed the actual video duration
                    # This ensures all routes can complete within the available time
                    effective_video_duration = min(min_video_duration_needed, video_length)
                    
                    # Calculate total_route_time based on the effective video duration
                    # This ensures the route_time_per_frame calculation gives enough time for all routes
                    total_route_time = effective_video_duration
                    
                    if debug_callback:
                        debug_callback(f"SIMULTANEOUS MODE FIX: Using effective video duration: {effective_video_duration:.1f}s")
                        debug_callback(f"  This ensures all routes can complete within the video time")
                else:
                    # Fallback to original calculation
                    total_route_time = total_span_seconds
                    if debug_callback:
                        debug_callback(f"SIMULTANEOUS MODE: Using fallback total span: {total_route_time:.1f}s")
            else:
                # Fallback to old method if timestamps are missing
                total_route_time = 0.0
                for i, route_data in enumerate(all_routes):
                    route_points = route_data.get('combined_route', [])
                    if route_points:
                        last_point = route_points[-1]
                    if len(last_point) >= 5:
                        route_time = last_point[4]  # accumulated_time at index 4
                        total_route_time = max(total_route_time, route_time)
                        
                        if log_callback:
                            log_callback(f"Warning: Using fallback calculation (timestamps missing)")
                            log_callback(f"Total route time (max individual duration): {total_route_time:.1f} seconds ({total_route_time/60:.1f} minutes)")
            
            # For multiple routes mode, we already have total_route_time, so we can skip the combined_route check
            # But we need to ensure combined_route is defined for the later code that might reference it
            combined_route = []  # Empty list for multiple routes mode since we don't need it
        else:
            # Single route mode OR single route group with multiple tracks (backward compatibility)
            # For single route groups, treat all tracks as one continuous route
            if all_routes and len(all_routes) > 0:
                # Single route group mode: combine all tracks into total time calculation
                total_route_time = 0.0
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    if route_points:
                        last_point = route_points[-1]
                        if len(last_point) >= 5:
                            route_time = last_point[4]  # accumulated_time at index 4
                            total_route_time = max(total_route_time, route_time)  # Take the longest track
                
                # Set combined_route for later code compatibility
                combined_route = []
            else:
                # Original single route mode
                combined_route = combined_route_data.get('combined_route', [])
        
        # Only check combined_route for original single route mode
        if not all_routes and not combined_route:
                if log_callback:
                    log_callback("Error: No route data found")
                return None
            
        # Calculate total_route_time for original single route mode only
        if not all_routes and combined_route:
            # Get total route time from the last point (only for original single route mode)
            # Route structure: (route_index, lat, lon, timestamp, accumulated_time, accumulated_distance, new_route_flag, filename)
            last_point = combined_route[-1]
            total_route_time = last_point[4]  # accumulated_time is at index 4
            
            if debug_callback:
                debug_callback(f"Original single route mode: Total route time: {total_route_time:.1f} seconds ({total_route_time/60:.1f} minutes)")
        
        # Calculate route time per frame
        if total_frames == 0:
            if log_callback:
                log_callback("Error: Total frames is zero")
            return None
        
        # ROUTE COMPLETION FIX: Ensure ALL route points are processed within video_length
        # Calculate route_time_per_frame to guarantee all points are shown by the final frame
        # This prevents premature transition to tail-only phase
        
        # The formula must use (total_frames - 1) as divisor because:
        # - Frame numbers are 1-indexed: 1, 2, 3, ..., total_frames
        # - target_time_route = (frame_number - 1) * route_time_per_frame
        # - So frame 1 has target_time = 0, and we want the last frame to reach total_route_time
        # - Last frame target_time = (total_frames - 1) * route_time_per_frame = total_route_time
        # - Therefore: route_time_per_frame = total_route_time / (total_frames - 1)
        if total_frames > 1:
            route_time_per_frame = total_route_time / (total_frames - 1)
        else:
            route_time_per_frame = total_route_time
        
        return route_time_per_frame
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error calculating route time per frame: {str(e)}")
        return None


def calculate_bounding_box_for_points(points, padding_percent=0.1):
    """
    Calculate a bounding box for a set of points with consistent padding and aspect ratio adjustment.
    This function ensures that both the caching phase and video frame generation use exactly the same logic.
    
    Args:
        points (list): List of route points with (route_index, lat, lon, timestamp, ...) structure
        padding_percent (float): Padding percentage (default 0.1 = 10%)
    
    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max) in GPS coordinates, rounded to 6 decimal places
    """
    if not points:
        return None
    
    # Extract coordinates from points
    all_lats = [point[1] for point in points if len(point) >= 3]  # lat is at index 1
    all_lons = [point[2] for point in points if len(point) >= 3]  # lon is at index 2
    
    if not all_lats or not all_lons:
        return None
    
    # Use the new wrapped coordinate handling function
    return calculate_bounding_box_for_wrapped_coordinates(all_lats, all_lons, padding_percent)


def calculate_unique_bounding_boxes(json_data, route_time_per_frame, log_callback=None, max_workers=None, combined_route_data=None, debug_callback=None):
    """
    Calculate unique bounding boxes for all frames in the video.
    
    Args:
        json_data (dict): Job data containing video parameters
        route_time_per_frame (float): Route time per frame in seconds
        log_callback (callable, optional): Function to call for logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        combined_route_data (dict, optional): Combined route data containing gpx_time_per_video_time
        debug_callback (callable, optional): Function to call for debug logging messages
    
    Returns:
        list: List of unique bounding boxes as tuples (lon_min, lon_max, lat_min, lat_max), or None if error
    """
    try:
        # Get video parameters and ensure they are numeric
        video_length = json_data.get('video_length')
        video_fps = json_data.get('video_fps')
        
        if video_length is None or video_fps is None:
            if log_callback:
                log_callback("Error: video_length or video_fps missing from job data")
            return None
        
        # Convert to float/int to ensure numeric types
        try:
            video_length = float(video_length)
            video_fps = float(video_fps)
        except (ValueError, TypeError):
            if log_callback:
                log_callback(f"Error: Could not convert video parameters to numbers - video_length: {video_length}, video_fps: {video_fps}")
            return None
        
        # Calculate total number of frames
        total_frames = int(video_length * video_fps)  # Convert to int for range operations
        
        if debug_callback:
            debug_callback(f"Calculating unique bounding boxes for {total_frames} frames")
        
        if not combined_route_data:
            if log_callback:
                log_callback("Error: combined_route_data parameter is required")
            return None
        
        # Check zoom mode
        zoom_mode = json_data.get('zoom_mode', 'dynamic')
        
        if zoom_mode == 'final':
            # For final zoom mode, calculate only the bounding box of the complete route
            if debug_callback:
                debug_callback("Using 'final' zoom mode - calculating single bounding box for entire route")
            
            # Check if we have multiple routes
            all_routes = combined_route_data.get('all_routes', None)
            
            if all_routes and len(all_routes) > 1:
                # Multiple routes mode - collect points from all routes
                combined_route = []
                route_count = len(all_routes)
                
                if debug_callback:
                    debug_callback(f"Final zoom mode: Collecting points from {route_count} routes")
                
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    combined_route.extend(route_points)
                
                # Sort all points by accumulated_time to ensure chronological order
                combined_route.sort(key=lambda point: point[4] if len(point) > 4 else 0)  # accumulated_time at index 4
                
                if debug_callback:
                    debug_callback(f"Final zoom mode: Total points from all routes: {len(combined_route)}")
            else:
                # Single route mode (backward compatibility)
                combined_route = combined_route_data.get('combined_route', [])
                           
            if not combined_route:
                if log_callback:
                    log_callback("Error: No route data found")
                return None
            
            # Calculate bounding box for the entire route
            all_lats = [point[1] for point in combined_route]  # lat is at index 1
            all_lons = [point[2] for point in combined_route]  # lon is at index 2
            
            # Calculate the data extent
            lon_min, lon_max = min(all_lons), max(all_lons)
            lat_min, lat_max = min(all_lats), max(all_lats)
            
            # Calculate center point
            lon_center = (lon_min + lon_max) / 2
            lat_center = (lat_min + lat_max) / 2
            
            # Calculate spans with basic padding (10%)
            lon_span_raw = lon_max - lon_min
            lat_span_raw = lat_max - lat_min
            
            # Handle case where all points are at the same location (zero span)
            if lon_span_raw == 0 and lat_span_raw == 0:
                # Single point - create a small default bounding box around it
                default_span = 0.001  # About 100 meters at the equator
                lon_min_padded = lon_center - default_span
                lon_max_padded = lon_center + default_span
                lat_min_padded = lat_center - default_span
                lat_max_padded = lat_center + default_span
            elif lon_span_raw == 0:
                # All points have same longitude - create narrow horizontal box
                default_lon_span = lat_span_raw * 0.1  # Make it 10% of lat span
                lon_min_padded = lon_center - default_lon_span
                lon_max_padded = lon_center + default_lon_span
                # Use normal lat span calculation
                base_padding = 0.1
                lat_padding = lat_span_raw * base_padding
                lat_min_padded = lat_min - lat_padding
                lat_max_padded = lat_max + lat_padding
            elif lat_span_raw == 0:
                # All points have same latitude - create narrow vertical box
                default_lat_span = lon_span_raw * 0.1  # Make it 10% of lon span
                lat_min_padded = lat_center - default_lat_span
                lat_max_padded = lat_center + default_lat_span
                # Use normal lon span calculation
                base_padding = 0.1
                lon_padding = lon_span_raw * base_padding
                lon_min_padded = lon_min - lon_padding
                lon_max_padded = lon_max + lon_padding
            else:
                # Normal case - both spans are non-zero
                # Base padding (percentage of the span)
                base_padding = 0.1  # 10% padding
                
                # Apply base padding to both dimensions
                lon_padding = lon_span_raw * base_padding
                lat_padding = lat_span_raw * base_padding
                
                # Initial padded bounds
                lon_min_padded = lon_min - lon_padding
                lon_max_padded = lon_max + lon_padding
                lat_min_padded = lat_min - lat_padding
                lat_max_padded = lat_max + lat_padding
                
                # Calculate aspect ratio of the route after initial padding
                lon_span_padded = lon_max_padded - lon_min_padded
                lat_span_padded = lat_max_padded - lat_min_padded
                
                # Adjust for latitude distortion - at the equator, 1 degree longitude ≈ 1 degree latitude
                # But at higher latitudes, 1 degree longitude < 1 degree latitude in terms of distance
                cos_factor = max(0.01, abs(math.cos(math.radians(lat_center))))  # Prevent division by zero
                
                # Calculate the effective aspect ratio of the padded route (accounting for latitude distortion)
                route_aspect_ratio = (lon_span_padded * cos_factor) / lat_span_padded
                
                # Target aspect ratio for 1920x1080 frame
                target_aspect_ratio = 16 / 9  # Standard video aspect ratio (1920/1080)
                
                # If route is more vertical than our target aspect ratio, expand horizontally
                if route_aspect_ratio < target_aspect_ratio:
                    # Calculate how much horizontal space we need to add to match target aspect ratio
                    desired_lon_span = lat_span_padded * target_aspect_ratio / cos_factor
                    # How much to add on each side
                    lon_expansion = (desired_lon_span - lon_span_padded) / 2
                    # Apply expansion
                    lon_min_padded -= lon_expansion
                    lon_max_padded += lon_expansion
                
                # If route is more horizontal than our target aspect ratio, expand vertically
                elif route_aspect_ratio > target_aspect_ratio:
                    # Calculate how much vertical space we need to add to match target aspect ratio
                    desired_lat_span = (lon_span_padded * cos_factor) / target_aspect_ratio
                    # How much to add on each side
                    lat_expansion = (desired_lat_span - lat_span_padded) / 2
                    # Apply expansion
                    lat_min_padded -= lat_expansion
                    lat_max_padded += lat_expansion
            
            # Create the final bounding box (rounded to 6 decimal places)
            final_bbox = (
                round(lon_min_padded, 6),   # lon_min_padded
                round(lon_max_padded, 6),   # lon_max_padded
                round(lat_min_padded, 6),   # lat_min_padded
                round(lat_max_padded, 6)    # lat_max_padded
            )
            
            # Save the final bounding box for later use
            save_final_bounding_box(final_bbox, log_callback, debug_callback)
            
            route_count = len(all_routes) if all_routes else 1
            if debug_callback:
                debug_callback(f"Final zoom mode: calculated single bounding box {final_bbox} from {route_count} route{'s' if route_count > 1 else ''}")
            
            # Return list with single bounding box
            return [final_bbox]
        
        else:
            # Dynamic zoom mode - calculate bounding boxes for each frame as before
            if debug_callback:
                debug_callback("Using 'dynamic' zoom mode - calculating bounding boxes for each frame")
            
            # Divide frames into chunks of 500
            chunk_size = 500
            frame_chunks = []
            
            for start_frame in range(0, total_frames, chunk_size):
                end_frame = min(start_frame + chunk_size, total_frames)
                frame_chunks.append((start_frame, end_frame))
            
            if debug_callback:
                debug_callback(f"Created {len(frame_chunks)} chunks of up to {chunk_size} frames each")
            
            # Determine number of worker processes
            constraints = [multiprocessing.cpu_count(), len(frame_chunks)]
            if max_workers is not None:
                constraints.append(max_workers)
            num_workers = min(constraints)
            
            if debug_callback:
                constraint_names = [f"CPU cores: {multiprocessing.cpu_count()}", f"chunks: {len(frame_chunks)}"]
                if max_workers is not None:
                    constraint_names.append(f"user setting: {max_workers}")
                debug_callback(f"Using {num_workers} worker processes (limited by {', '.join(constraint_names)})")
            
            # Create multiprocessing pool and process chunks
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Prepare arguments for each worker
                worker_args = []
                for start_frame, end_frame in frame_chunks:
                    worker_args.append((
                        combined_route_data,
                        json_data,
                        route_time_per_frame,
                        start_frame,
                        end_frame
                    ))
                
                # Execute workers in parallel
                if debug_callback:
                    debug_callback("Starting parallel processing of frame chunks")
                
                results = pool.starmap(process_frame_chunk, worker_args)
            
            # Collect all bounding boxes from workers
            all_bounding_boxes = []
            for result in results:
                if result:  # Only add non-empty results
                    all_bounding_boxes.extend(result)
            
            # Create set of unique bounding boxes
            unique_bounding_boxes = list(set(all_bounding_boxes))
            
            if debug_callback:
                debug_callback(f"Collected {len(all_bounding_boxes)} total bounding boxes")
            
            return unique_bounding_boxes
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error calculating unique bounding boxes: {str(e)}")
        return None


def process_frame_chunk(combined_route_data, json_data, route_time_per_frame, start_frame, end_frame):
    """
    Worker function to process a chunk of frames and return bounding boxes.
    Supports both single route and multiple routes modes.
    
    Args:
        combined_route_data (dict): Combined route data (may contain multiple routes)
        json_data (dict): Job data containing video parameters
        route_time_per_frame (float): Route time per frame in seconds
        start_frame (int): Starting frame number (inclusive)
        end_frame (int): Ending frame number (exclusive)
    
    Returns:
        list: List of bounding boxes as tuples (lon_min, lon_max, lat_min, lat_max)
    """
    try:
        # Extract video parameters from json_data for consistent time calculation
        video_fps = float(json_data.get('video_fps', 30))
        gpx_time_per_video_time = combined_route_data.get('gpx_time_per_video_time')
        
        # Use gpx_time_per_video_time from combined_route_data if available
        if combined_route_data and 'gpx_time_per_video_time' in combined_route_data:
            gpx_time_per_video_time = combined_route_data['gpx_time_per_video_time']
            route_time_per_frame = gpx_time_per_video_time / video_fps
        else:
            # Fallback to calculating route time per frame
            route_time_per_frame = calculate_route_time_per_frame(json_data, log_callback)
        
        # Check if we have multiple routes
        all_routes = combined_route_data.get('all_routes', None)
        
        if all_routes and len(all_routes) > 1:
            # STAGGERED ROUTES FIX: Apply the same staggered timing logic as in frame generation
            # Calculate route start delays for proper timing
            earliest_start_time = None
            route_start_times = {}
            
            for route_data in all_routes:
                route_points = route_data.get('combined_route', [])
                if route_points and len(route_points[0]) >= 4:  # Check for timestamp at index 3
                    route_start_timestamp = route_points[0][3]  # timestamp at index 3
                    if route_start_timestamp:
                        route_start_times[id(route_data)] = route_start_timestamp
                        if earliest_start_time is None or route_start_timestamp < earliest_start_time:
                            earliest_start_time = route_start_timestamp
            
            bounding_boxes = []
            frames_without_points = 0
            
            # Process each frame in the chunk
            for frame_num in range(start_frame, end_frame):
                # Calculate target time for this frame - use same method as video generation
                # Convert video time to route time using gpx_time_per_video_time
                target_time_video = frame_num / video_fps  # Video time in seconds
                target_time = target_time_video * gpx_time_per_video_time  # Route time in seconds
                
                # Collect points from each route with proper staggered timing
                points_for_frame = []
                
                for route_data in all_routes:
                    route_points = route_data.get('combined_route', [])
                    
                    # Calculate this route's delay relative to the earliest route
                    route_delay_seconds = 0.0
                    route_id = id(route_data)
                    if route_id in route_start_times and earliest_start_time:
                        route_start_timestamp = route_start_times[route_id]
                        route_delay_seconds = (route_start_timestamp - earliest_start_time).total_seconds()
                    
                    # Calculate route-specific target time accounting for start delay
                    route_target_time = target_time - route_delay_seconds
                    
                    # Only include points if this route should have started by now
                    if route_target_time >= 0:
                        # OPTIMIZED: Use binary search + list slicing instead of linear search + appends
                        cutoff_index = _binary_search_cutoff_index(route_points, route_target_time)
                        route_points_for_frame = route_points[:cutoff_index]
                        if route_points_for_frame:  # Only add non-empty route points
                            points_for_frame.append(route_points_for_frame)  # Add as sub-list to match video generation
                
                if points_for_frame:
                    # Flatten points_for_frame to match the shared function's expected input
                    all_points = []
                    for route_points in points_for_frame:
                        if route_points:
                            all_points.extend(route_points)
                    
                    # Use the shared bounding box calculation function for consistency
                    bbox = calculate_bounding_box_for_points(all_points, padding_percent=0.1)
                    
                    if bbox is not None:
                        bounding_boxes.append(bbox)
                    else:
                        frames_without_points += 1
                else:
                    # Debug: Track frames that don't produce bounding boxes
                    frames_without_points += 1
            
            # Debug logging for this chunk
            expected_frames = end_frame - start_frame
            actual_bboxes = len(bounding_boxes)
            route_count = len(all_routes)
            write_debug_log(f"Chunk {start_frame}-{end_frame}: Expected {expected_frames} frames, produced {actual_bboxes} bounding boxes, {frames_without_points} frames had no points (using {route_count} staggered routes)")
            
            return bounding_boxes
        else:
            # Single route mode (backward compatibility) - use original logic
            combined_route = combined_route_data.get('combined_route', [])
        
        if not combined_route:
            return []
        
        bounding_boxes = []
        frames_without_points = 0
        
        # Process each frame in the chunk
        for frame_num in range(start_frame, end_frame):
            # Calculate target time for this frame - use same method as video generation
            # Convert video time to route time using gpx_time_per_video_time
            target_time_video = frame_num / video_fps  # Video time in seconds
            target_time = target_time_video * gpx_time_per_video_time  # Route time in seconds
            
            # Collect points up to the target time for this frame
            # Route structure: (route_index, lat, lon, timestamp, accumulated_time, accumulated_distance, new_route_flag, filename)
            # OPTIMIZED: Use binary search + list slicing instead of linear search + appends
            cutoff_index = _binary_search_cutoff_index(combined_route, target_time)
            points_for_frame = combined_route[:cutoff_index]
            
            if points_for_frame:
                # Use the shared bounding box calculation function for consistency
                bbox = calculate_bounding_box_for_points(points_for_frame, padding_percent=0.1)
                
                if bbox is not None:
                    bounding_boxes.append(bbox)
                else:
                    frames_without_points += 1
            else:
                # Debug: Track frames that don't produce bounding boxes
                frames_without_points += 1
        
        # Debug logging for this chunk
        expected_frames = end_frame - start_frame
        actual_bboxes = len(bounding_boxes)
        write_debug_log(f"Chunk {start_frame}-{end_frame}: Expected {expected_frames} frames, produced {actual_bboxes} bounding boxes, {frames_without_points} frames had no points (single route mode)")
        
        # Additional debug for problematic chunks
        if frames_without_points > 0:
            first_frame_target_time = start_frame * route_time_per_frame
            last_frame_target_time = (end_frame - 1) * route_time_per_frame
            first_route_time = combined_route[0][4] if combined_route else "N/A"  # accumulated_time at index 4
            last_route_time = combined_route[-1][4] if combined_route else "N/A"  # accumulated_time at index 4
            write_debug_log(f"  Debug: Chunk target times {first_frame_target_time:.2f}-{last_frame_target_time:.2f}s, route times {first_route_time}-{last_route_time}s")
        
        return bounding_boxes
        
    except Exception as e:
        # Return empty list on error (will be logged by parent process)
        write_log(f"Error in process_frame_chunk({start_frame}-{end_frame}): {str(e)}")
        return [] 