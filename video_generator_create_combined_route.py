"""
Video generation combined route creation for the Route Squiggler render client.
This module handles creating combined routes from multiple chronologically sorted GPX files.
Supports both single route mode and multiple simultaneous routes
based on track names from track_objects.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple, Optional
import gpxpy
from image_generator_utils import calculate_haversine_distance

# Conversion constants for imperial units
METERS_TO_MILES = 0.000621371
KMH_TO_MPH = 0.621371


class RoutePoint(NamedTuple):
    """
    Named tuple representing a single point in a route.
    
    This structure is used throughout the video generator pipeline for storing
    GPS coordinates along with associated metadata like timing and statistics.
    
    The smoothed fields (heart_rate_smoothed, current_speed_smoothed) are calculated
    after the route is fully created, using a 0.5 video-second smoothing window.
    These pre-calculated values avoid per-frame recalculation during video rendering.
    
    Accumulated_distance and current_speed_smoothed are stored in metric units
    (meters/km/h) or imperial units (miles/mph) based on the imperial_units setting.
    """
    route_index: int                    # Unique index for each track/route
    lat: float                          # Latitude
    lon: float                          # Longitude
    timestamp: Optional[datetime]       # Timestamp (rounded to nearest second)
    accumulated_time: float             # Accumulated time in seconds from route start
    accumulated_distance: float         # Accumulated distance (meters or miles based on imperial_units)
    new_route_flag: bool                # True if this is the first point in a new route segment
    filename: str                       # Filename without extension (for color mapping)
    elevation: Optional[float]          # Elevation in meters (if available)
    heart_rate: int                     # Heart rate in BPM (0 if not available)
    # Pre-calculated smoothed values (populated after route creation if statistics are enabled)
    heart_rate_smoothed: Optional[int] = None      # Smoothed HR over 0.5 video-second window
    current_speed_smoothed: Optional[int] = None   # Smoothed speed (km/h or mph based on imperial_units)


def _calculate_smoothed_statistics_for_route(route_points, gpx_time_per_video_time, video_fps, calculate_hr=False, calculate_speed=False, imperial_units=False):
    """
    Calculate smoothed heart rate and speed for all points in a route.
    
    Uses a 0.5 video-second smoothing window, converted to route time using gpx_time_per_video_time.
    This pre-calculation avoids per-frame recalculation during video rendering.
    
    Args:
        route_points (list): List of RoutePoint objects for a single route
        gpx_time_per_video_time (float): Ratio of route time to video time
        video_fps (float): Video frames per second
        calculate_hr (bool): Whether to calculate smoothed heart rate
        calculate_speed (bool): Whether to calculate smoothed speed
        imperial_units (bool): If True, convert speed to mph instead of km/h
    
    Returns:
        list: New list of RoutePoint objects with smoothed values populated
    """
    if not route_points or (not calculate_hr and not calculate_speed):
        return route_points
    
    # Calculate the smoothing window in route time (0.5 video seconds)
    video_seconds_to_check = 0.5
    route_time_threshold = video_seconds_to_check * gpx_time_per_video_time
    
    updated_points = []
    
    for i, point in enumerate(route_points):
        smoothed_hr = None
        smoothed_speed = None
        
        # Calculate target time for comparison (0.5 video seconds back from this point)
        target_time = point.accumulated_time - route_time_threshold
        
        if calculate_hr and point.heart_rate > 0:
            # Collect all HR values within the smoothing window
            hr_values = []
            for j in range(i, -1, -1):  # Search backwards from current point
                prev_point = route_points[j]
                if prev_point.accumulated_time < target_time:
                    break  # We've gone past the window
                if prev_point.heart_rate > 0:
                    hr_values.append(prev_point.heart_rate)
            
            if hr_values:
                smoothed_hr = round(sum(hr_values) / len(hr_values))
        
        if calculate_speed:
            # Find comparison point that is approximately 0.5 video seconds back
            comparison_point = None
            for j in range(i - 1, -1, -1):  # Search backwards
                prev_point = route_points[j]
                if prev_point.accumulated_time <= target_time:
                    comparison_point = prev_point
                    break
            
            if comparison_point:
                time_diff = point.accumulated_time - comparison_point.accumulated_time
                distance_diff = point.accumulated_distance - comparison_point.accumulated_distance
                
                if time_diff > 0:
                    if imperial_units:
                        # Distance is in miles, convert to mph
                        speed_mph = (distance_diff * 3600) / time_diff
                        smoothed_speed = round(speed_mph)
                    else:
                        # Distance is in meters, convert to km/h
                        speed_kmh = (distance_diff / 1000.0 * 3600) / time_diff
                        smoothed_speed = round(speed_kmh)
            else:
                # Fallback: use first point if no comparison point found
                if i > 0:
                    first_point = route_points[0]
                    time_diff = point.accumulated_time - first_point.accumulated_time
                    distance_diff = point.accumulated_distance - first_point.accumulated_distance
                    
                    if time_diff > 0:
                        if imperial_units:
                            # Distance is in miles, convert to mph
                            speed_mph = (distance_diff * 3600) / time_diff
                            smoothed_speed = round(speed_mph)
                        else:
                            # Distance is in meters, convert to km/h
                            speed_kmh = (distance_diff / 1000.0 * 3600) / time_diff
                            smoothed_speed = round(speed_kmh)
                    else:
                        smoothed_speed = 0
                else:
                    smoothed_speed = 0
        
        # Create new point with smoothed values
        updated_point = point._replace(
            heart_rate_smoothed=smoothed_hr,
            current_speed_smoothed=smoothed_speed
        )
        updated_points.append(updated_point)
    
    return updated_points


def _update_speed_based_color_range(all_routes, json_data, debug_callback=None):
    """
    Update speed_based_color_min and speed_based_color_max in json_data based on actual data.
    
    If either value is -1, it will be calculated from the minimum/maximum current_speed_smoothed
    values across all routes. If values are already set (not -1), they are left unchanged.
    
    Args:
        all_routes (list): List of route data dictionaries
        json_data (dict): Job data dictionary (will be modified in place)
        debug_callback (callable, optional): Function for debug logging
    
    Returns:
        None (modifies json_data in place)
    """
    if not json_data:
        return
    
    speed_min_setting = json_data.get('speed_based_color_min', -1)
    speed_max_setting = json_data.get('speed_based_color_max', -1)
    
    # Check if we need to calculate either value
    need_min = speed_min_setting == -1
    need_max = speed_max_setting == -1
    
    if not need_min and not need_max:
        if debug_callback:
            debug_callback("Speed-based color range already set, skipping auto-calculation")
        return
    
    # Collect all current_speed_smoothed values from all routes
    all_speeds = []
    for route in all_routes:
        route_points = route.get('combined_route', [])
        for point in route_points:
            if point.current_speed_smoothed is not None:
                all_speeds.append(point.current_speed_smoothed)
    
    if not all_speeds:
        if debug_callback:
            debug_callback("No smoothed speed data available for color range calculation")
        return
    
    # Calculate min/max from actual data
    calculated_min = min(all_speeds)
    calculated_max = max(all_speeds)
    
    # Update json_data if needed
    # Note: speed_based_color_min/max are already in the correct units (mph if imperial_units, km/h otherwise)
    # and should NOT be converted
    imperial_units = _is_imperial_units(json_data)
    speed_unit = "mph" if imperial_units else "km/h"
    
    if need_min:
        json_data['speed_based_color_min'] = calculated_min
        if debug_callback:
            debug_callback(f"Auto-calculated speed_based_color_min: {calculated_min} {speed_unit}")
    
    if need_max:
        json_data['speed_based_color_max'] = calculated_max
        if debug_callback:
            debug_callback(f"Auto-calculated speed_based_color_max: {calculated_max} {speed_unit}")
    
    if debug_callback and (need_min or need_max):
        debug_callback(f"Speed-based color range: {json_data.get('speed_based_color_min')} - {json_data.get('speed_based_color_max')} {speed_unit}")


def _apply_smoothed_statistics_to_routes(all_routes, gpx_time_per_video_time, json_data, debug_callback=None):
    """
    Apply smoothed statistics calculation to all routes.
    
    Args:
        all_routes (list): List of route data dictionaries
        gpx_time_per_video_time (float): Ratio of route time to video time
        json_data (dict): Job data containing statistics configuration
        debug_callback (callable, optional): Function for debug logging
    
    Returns:
        list: Updated routes with smoothed statistics calculated
    """
    # Check if we need to calculate any smoothed values
    calculate_hr = json_data.get('statistics_current_hr', False) if json_data else False
    calculate_speed = json_data.get('statistics_current_speed', False) if json_data else False
    
    if not calculate_hr and not calculate_speed:
        if debug_callback:
            debug_callback("Smoothed statistics calculation skipped (not enabled in job settings)")
        return all_routes
    
    video_fps = float(json_data.get('video_fps', 30)) if json_data else 30.0
    
    if debug_callback:
        enabled_stats = []
        if calculate_hr:
            enabled_stats.append("heart_rate")
        if calculate_speed:
            enabled_stats.append("current_speed")
        debug_callback(f"Calculating smoothed statistics for: {', '.join(enabled_stats)}")
    
    imperial_units = _is_imperial_units(json_data)
    
    for route in all_routes:
        route_points = route.get('combined_route', [])
        if route_points:
            updated_points = _calculate_smoothed_statistics_for_route(
                route_points, 
                gpx_time_per_video_time, 
                video_fps,
                calculate_hr=calculate_hr,
                calculate_speed=calculate_speed,
                imperial_units=imperial_units
            )
            route['combined_route'] = updated_points
            route['total_points'] = len(updated_points)
    
    if debug_callback:
        total_points = sum(len(route.get('combined_route', [])) for route in all_routes)
        debug_callback(f"Smoothed statistics applied to {total_points} points across {len(all_routes)} routes")
    
    return all_routes


def _extract_heart_rate_from_gpxpy_point(point) -> int:
    """
    Extract heart rate value from a gpxpy trackpoint's extensions.
    
    Supports the common Garmin TrackPointExtension format:
    <extensions>
        <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>76</gpxtpx:hr>
        </gpxtpx:TrackPointExtension>
    </extensions>
    
    Args:
        point: gpxpy GPXTrackPoint object
    
    Returns:
        Heart rate as integer, or 0 if not found
    """
    if not hasattr(point, 'extensions') or not point.extensions:
        return 0
    
    # gpxpy stores extensions as a list of XML elements
    for extension in point.extensions:
        # Search recursively for hr element
        for elem in extension.iter():
            if elem.tag.endswith('}hr') or elem.tag == 'hr':
                if elem.text:
                    try:
                        return int(elem.text)
                    except ValueError:
                        pass
    
    return 0


def _is_imperial_units(json_data):
    """
    Returns true if imperial_units is True in json_data.
    """
    return json_data and json_data.get('imperial_units', False) is True


def create_combined_route(sorted_gpx_files, json_data=None, progress_callback=None, log_callback=None, debug_callback=None):
    """
    Create combined routes from chronologically sorted GPX files.
    Supports multiple simultaneous routes based on track names.
    
    Args:
        sorted_gpx_files (list): List of GPX file info dictionaries sorted chronologically
        json_data (dict, optional): Job data containing configuration parameters and track_objects
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
    
    Returns:
        dict: Combined route data with 'combined_route' and 'total_distance' keys, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Creating combined routes from sorted GPX files.")
        
        # Create output directory
        output_dir = Path("temporary files/route")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have track_objects with names for multiple routes
        track_objects = json_data.get('track_objects', []) if json_data else []
        use_multiple_routes = False
        track_name_map = {}  # filename -> name mapping
        
        if track_objects:
            # Check if first track has a non-empty name
            first_name = track_objects[0].get('name', '') if track_objects else ''
            if first_name:
                use_multiple_routes = True
                # Create mapping from filename to track name
                for track_obj in track_objects:
                    filename = track_obj.get('filename', '')
                    name = track_obj.get('name', '')
                    if filename and name:
                        track_name_map[filename] = name
                
                if debug_callback:
                    debug_callback(f"Multiple routes mode: Found {len(track_name_map)} named tracks")
            else:
                if debug_callback:
                    debug_callback("Single route mode: All track names are empty")
        
        # Used to be split into separate functions for single and multiple runners, but the function is now unified
        if not use_multiple_routes:
            # For single route mode, create an empty track_name_map so all files become "unnamed"
            track_name_map = {}
        
        return _create_multiple_routes(sorted_gpx_files, track_name_map, json_data, progress_callback, log_callback, debug_callback)
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in combined route creation: {str(e)}")
        return None



def _create_multiple_routes(sorted_gpx_files, track_name_map, json_data=None, progress_callback=None, log_callback=None, debug_callback=None):
    """
    Create combined routes from chronologically sorted GPX files.
    Handles both single runner (empty track_name_map) and multiple runners (populated track_name_map).
    """
    try:
        # Determine if this is single route mode (empty track_name_map means all files are unnamed)
        is_single_route_mode = len(track_name_map) == 0
        
        if debug_callback:
            if is_single_route_mode:
                debug_callback("Creating single combined route.")
            else:
                debug_callback("Creating multiple combined routes.")
        
        # Group files by track name
        files_by_track = {}
        unnamed_files = []
        
        for gpx_info in sorted_gpx_files:
            filename = gpx_info.get('filename', '')
            if filename in track_name_map:
                track_name = track_name_map[filename]
                if track_name not in files_by_track:
                    files_by_track[track_name] = []
                files_by_track[track_name].append(gpx_info)
            else:
                # File not found in track_name_map, add to unnamed files
                unnamed_files.append(gpx_info)
        
        if debug_callback:
            debug_callback(f"Grouped files by track: {len(files_by_track)} named tracks, {len(unnamed_files)} unnamed files")
            for track_name, files in files_by_track.items():
                debug_callback(f"  Track '{track_name}': {len(files)} files")
        
        # Create routes for each track
        all_routes = []
        total_tracks = len(files_by_track)
        
        for track_index, (track_name, track_files) in enumerate(files_by_track.items()):
            if debug_callback:
                debug_callback(f"Processing track '{track_name}' ({track_index + 1}/{total_tracks})")
            
            # Update progress
            if progress_callback:
                progress_percentage = int((track_index / total_tracks) * 90)  # Leave 10% for final processing
                progress_text = f"Processing track {track_index + 1}/{total_tracks}: {track_name}"
                progress_callback("progress_bar_combined_route", progress_percentage, progress_text)
            
            # Create route for this track
            track_route_data = _create_route_for_track(track_files, track_index, track_name, json_data, log_callback, debug_callback)
            if track_route_data:
                all_routes.append(track_route_data)
        
        # Handle unnamed files if any (create a separate route)
        if unnamed_files:
            if debug_callback:
                debug_callback(f"Processing {len(unnamed_files)} unnamed files")
            
            unnamed_route_data = _create_route_for_track(unnamed_files, len(all_routes), "Unnamed", json_data, log_callback, debug_callback)
            if unnamed_route_data:
                all_routes.append(unnamed_route_data)
        
        if not all_routes:
            if log_callback:
                log_callback("Warning: No routes were created successfully")
            return None
        
        # Calculate overall statistics
        total_distance = sum(route.get('total_distance', 0) for route in all_routes)
        total_points = sum(route.get('total_points', 0) for route in all_routes)
        total_files = sum(route.get('total_files', 0) for route in all_routes)
        
        # Calculate GPX time per video time ratio (use the first route's time as reference)
        video_length = json_data.get('video_length', 0) if json_data else 0
        video_fps = json_data.get('video_fps', 30) if json_data else 30
        
        # Ensure numeric types
        try:
            video_length = float(video_length)
            video_fps = float(video_fps)
        except (ValueError, TypeError):
            if log_callback:
                log_callback(f"Error: Could not convert video parameters to numbers - video_length: {video_length}, video_fps: {video_fps}")
            return None
        
        # Calculate total frames
        total_frames = video_length * video_fps
        
        # SIMULTANEOUS MODE FIX: Calculate total span for staggered routes
        # The issue is that we need to use the total span from earliest start to latest end
        # This allows all routes to be drawn at their proper staggered times
        
        # Find the earliest start time and latest end time across all routes
        earliest_start_time = None
        latest_end_time = None
        
        for route in all_routes:
            combined_route = route.get('combined_route', [])
            if combined_route and len(combined_route) > 0:
                first_point = combined_route[0]
                last_point = combined_route[-1]
                
                route_start_timestamp = first_point.timestamp
                route_duration = last_point.accumulated_time
                
                if route_start_timestamp:
                    # Calculate when this route ends (start + duration)
                    route_end_timestamp = route_start_timestamp + timedelta(seconds=route_duration)
                    
                    if earliest_start_time is None or route_start_timestamp < earliest_start_time:
                        earliest_start_time = route_start_timestamp
                    
                    if latest_end_time is None or route_end_timestamp > latest_end_time:
                        latest_end_time = route_end_timestamp
        
        # Calculate total accumulated time based on mode
        if is_single_route_mode:
            # For single route mode, use the accumulated time from the last point
            if all_routes and all_routes[0].get('combined_route'):
                last_point = all_routes[0]['combined_route'][-1]
                total_accumulated_time = last_point.accumulated_time
                if debug_callback:
                    debug_callback(f"Single route mode: Using sequential time accumulation: {total_accumulated_time:.1f}s")
            else:
                total_accumulated_time = 0
                if log_callback:
                    log_callback("Warning: No route data found for time calculation")
        else:
            # For multiple routes mode, use simultaneous mode logic
            if earliest_start_time and latest_end_time:
                total_span_seconds = (latest_end_time - earliest_start_time).total_seconds()
                total_accumulated_time = total_span_seconds
                
                if debug_callback:
                    debug_callback(f"Simultaneous mode: Using total span: {total_span_seconds:.1f}s")
                    debug_callback(f"  Earliest route starts at: {earliest_start_time.strftime('%H:%M:%S')}")
                    debug_callback(f"  Latest route ends at: {latest_end_time.strftime('%H:%M:%S')}")
            else:
                # Fallback to first route's time if no timestamps found
                total_accumulated_time = all_routes[0].get('total_accumulated_time', 0) if all_routes else 0
                if log_callback:
                    log_callback(f"Warning: Using fallback total_accumulated_time: {total_accumulated_time:.1f}s")
        
        # Calculate route_time_per_frame using the same method as caching phase
        # Use adjusted calculation to ensure complete route coverage
        # 
        # ROUTE COMPLETION FIX: The formula must use (total_frames - 1) as divisor because:
        # - Frame numbers are 1-indexed: 1, 2, 3, ..., total_frames
        # - target_time_route = (frame_number - 1) * route_time_per_frame
        # - So frame 1 has target_time = 0, and we want the last frame to reach total_accumulated_time
        # - Last frame target_time = (total_frames - 1) * route_time_per_frame = total_accumulated_time
        # - Therefore: route_time_per_frame = total_accumulated_time / (total_frames - 1)
        if total_frames > 1:
            route_time_per_frame = total_accumulated_time / (total_frames - 1)
        else:
            route_time_per_frame = total_accumulated_time
        
        # Calculate gpx_time_per_video_time using the same method as caching phase
        gpx_time_per_video_time = route_time_per_frame * video_fps
        
        imperial_units = _is_imperial_units(json_data)
        distance_unit = "miles" if imperial_units else "meters"
        distance_unit_short = "mi" if imperial_units else "m"
        
        if debug_callback:
            debug_callback(f"Created {len(all_routes)} routes:")
            for i, route in enumerate(all_routes):
                route_name = route.get('track_name', f'Route {i}')
                points = route.get('total_points', 0)
                distance = route.get('total_distance', 0)
                debug_callback(f"  Route {i}: '{route_name}' - {points} points, {distance:.2f} {distance_unit_short}")
            debug_callback(f"Total across all routes: {total_points} points, {total_distance:.2f} {distance_unit_short}")
        
        # Write debug file for all routes
        debug_file_path = Path("temporary files/route") / "debug_tuple.txt"
        try:
            imperial_units = _is_imperial_units(json_data)
            distance_unit_label = "miles" if imperial_units else "meters"
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Route_Index, Latitude, Longitude, Timestamp, Accumulated time (s), Accumulated Distance ({distance_unit_label}), New Route, Filename, Elevation, Heart Rate\n")
                for route in all_routes:
                    combined_route = route.get('combined_route', [])
                    for point in combined_route:
                        timestamp_str = point.timestamp.strftime('%Y-%m-%d %H:%M:%S') if point.timestamp else "None"
                        elevation_str = str(point.elevation) if point.elevation is not None else ""
                        heart_rate_str = str(point.heart_rate) if point.heart_rate else ""
                        f.write(f"{point.route_index}, {point.lat}, {point.lon}, {timestamp_str}, {point.accumulated_time:.0f}, {point.accumulated_distance:.2f}, {point.new_route_flag}, {point.filename}, {elevation_str}, {heart_rate_str}\n")
            
            if debug_callback:
                debug_callback(f"Wrote debug_tuple.txt successfully to {debug_file_path}")
        except Exception as e:
            if log_callback:
                log_callback(f"Error writing debug file: {str(e)}")
        
        # Progress update before saving
        if progress_callback:
            progress_callback("progress_bar_combined_route", 95, "Saving combined routes")
        
        # Add gpx_time_per_video_time to all routes BEFORE saving
        for route in all_routes:
            route['gpx_time_per_video_time'] = gpx_time_per_video_time
        
        # Calculate smoothed statistics for all routes (if enabled in job settings)
        # This pre-calculates smoothed HR and speed values to avoid per-frame recalculation
        all_routes = _apply_smoothed_statistics_to_routes(all_routes, gpx_time_per_video_time, json_data, debug_callback)
        
        # Auto-calculate speed-based color range if needed (if speed_based_color_min/max are -1)
        # This must be done after smoothed statistics are calculated
        _update_speed_based_color_range(all_routes, json_data, debug_callback)
              
        # Final progress update
        if progress_callback:
            progress_callback("progress_bar_combined_route", 100, "Multiple routes creation completed")
        
        # Return the first route for backward compatibility, but include all routes info
        first_route = all_routes[0] if all_routes else None
        if first_route:
            first_route['all_routes'] = all_routes
            first_route['total_routes'] = len(all_routes)
        
        return first_route
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in multiple routes creation: {str(e)}")
        return None


def _create_route_for_track(track_files, route_index, track_name, json_data=None, log_callback=None, debug_callback=None):
    """
    Create a single route for a specific track from its files.
    """
    try:
        if debug_callback:
            debug_callback(f"Creating route {route_index} for track '{track_name}' with {len(track_files)} files")
        
        # List to store all points in chronological order
        combined_route = []
        
        # Accumulated distance and time
        accumulated_distance = 0.0
        accumulated_time = 0.0
        
        # Previous point for distance calculation
        prev_lat = None
        prev_lon = None
        
        # Previous timestamp for time calculation within each file
        prev_timestamp = None
        
        # Check if heart rate extraction is needed (optimization: skip if not requested)
        extract_heart_rate = False
        if json_data:
            extract_heart_rate = json_data.get('statistics_average_hr', False) or json_data.get('statistics_current_hr', False)
        
        # Check if imperial units should be used
        imperial_units = _is_imperial_units(json_data)
        distance_unit = "miles" if imperial_units else "meters"
        distance_unit_short = "mi" if imperial_units else "m"
        
        # Process each file in chronological order
        for file_index, gpx_info in enumerate(track_files):
            filename = gpx_info.get('filename', f'file_{file_index}')
            content = gpx_info.get('content', '')
            
            # Convert accumulated_distance for display
            display_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
            
            if debug_callback:
                debug_callback(f"  Processing {filename}, accumulated distance: {display_distance:.2f} {distance_unit_short}, accumulated time: {accumulated_time:.1f} seconds")
            
            # Always extract filename without extension for color mapping
            # This is needed for route colors even if name_tags and legend are disabled
            filename_without_ext = Path(filename).stem
            
            try:
                # Parse the GPX content
                gpx = gpxpy.parse(content)
                
                # Flag to mark the first point in this file
                first_point_in_file = True
                
                # Process all track points
                for track in gpx.tracks:
                    for segment in track.segments:
                        for point in segment.points:
                            # Get latitude, longitude and timestamp
                            lat = point.latitude
                            lon = point.longitude
                            timestamp = point.time
                            
                            # Extract elevation if enabled and available
                            elevation = None
                            if json_data and json_data.get('statistics_current_elevation', False):
                                if point.elevation is not None:
                                    try:
                                        elevation = float(point.elevation)
                                    except (ValueError, TypeError):
                                        elevation = None
                            
                            # Extract heart rate if enabled
                            heart_rate = 0
                            if extract_heart_rate:
                                heart_rate = _extract_heart_rate_from_gpxpy_point(point)
                            
                            # Calculate distance from previous point
                            point_distance = 0.0
                            if prev_lat is not None and prev_lon is not None and not first_point_in_file:
                                point_distance = calculate_haversine_distance(prev_lat, prev_lon, lat, lon)
                                
                                accumulated_distance += point_distance
                            
                            # Calculate time increment from previous point (skip for first point in file)
                            if timestamp is not None and prev_timestamp is not None and not first_point_in_file:
                                time_increment = (timestamp - prev_timestamp).total_seconds()
                                if time_increment >= 0:  # Only add positive time increments
                                    accumulated_time += time_increment
                            
                            # Convert accumulated_distance to miles if imperial_units is True
                            stored_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
                            
                            # Add to combined route using RoutePoint named tuple
                            combined_route.append(RoutePoint(
                                route_index=route_index,
                                lat=lat,
                                lon=lon,
                                timestamp=timestamp.replace(microsecond=0) if timestamp else None,
                                accumulated_time=accumulated_time,
                                accumulated_distance=stored_distance,
                                new_route_flag=first_point_in_file,
                                filename=filename_without_ext,
                                elevation=elevation,
                                heart_rate=heart_rate
                            ))
                            
                            # Update previous point and timestamp for next iteration
                            prev_lat = lat
                            prev_lon = lon
                            prev_timestamp = timestamp
                            
                            # Clear first point flag after first point
                            if first_point_in_file:
                                first_point_in_file = False
                            
            except Exception as e:
                if log_callback:
                    log_callback(f"  Error processing file {filename}: {str(e)}")
                continue
        
        # Convert accumulated_distance for display
        display_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
        
        if debug_callback:
            debug_callback(f"  Completed track '{track_name}'. Total points: {len(combined_route)}")
            if imperial_units:
                debug_callback(f"  Total distance: {display_distance:.2f} miles")
            else:
                debug_callback(f"  Total distance: {display_distance:.2f} meters ({display_distance/1000:.2f} km)")
            debug_callback(f"  Total time: {accumulated_time:.1f} seconds ({accumulated_time/60:.1f} minutes)")
        
        # Apply pruning if json_data is provided
        if json_data:
            # Check if pruning should be applied based on job parameters
            route_accuracy = json_data.get('route_accuracy', 'normal')  # Default to 'normal' if not specified
            prune_route = route_accuracy != 'maximum'  # Prune unless route_accuracy is 'maximum'
            total_points = len(combined_route)
            
            if not prune_route:
                # Pruning disabled by job parameters (route_accuracy = 'maximum')
                if debug_callback:
                    debug_callback(f"  Route pruning skipped for track '{track_name}' as per job parameters (route_accuracy = 'maximum').")
                # Convert total_distance to miles if imperial_units is True
                stored_total_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
                final_route_data = {
                    'combined_route': combined_route,
                    'total_distance': stored_total_distance,
                    'total_points': len(combined_route),
                    'total_files': len(track_files),
                    'track_name': track_name,
                    'route_index': route_index
                }
            else:
                # Read video parameters and ensure numeric types
                video_length = float(json_data.get('video_length', 0))
                video_fps = float(json_data.get('video_fps', 30))
                total_frames = video_length * video_fps
                # Avoid division by zero
                if total_frames <= 1:
                    total_frames = max(1.0, total_frames)
                # Compute route_time_per_frame for this track
                # Use (total_frames - 1) as divisor to ensure complete route coverage
                # (see explanation in _create_multiple_routes)
                if total_frames > 1:
                    route_time_per_frame = accumulated_time / (total_frames - 1)
                else:
                    route_time_per_frame = accumulated_time
                gpx_time_per_video_time = route_time_per_frame * video_fps
                raw_interval_seconds = gpx_time_per_video_time / video_fps
                
                reduced_interval_seconds = raw_interval_seconds * 0.8
                pruning_interval = max(1, int(round(reduced_interval_seconds)))
                interpolation_interval_seconds = raw_interval_seconds * 2.0
                
                if debug_callback:
                    debug_callback(
                        f"  Calculated dynamic pruning interval for track '{track_name}': ~1 point/frame → "
                        f"raw={raw_interval_seconds:.2f}s (gpx_time_per_video_time/video_fps), reduced by 20% → "
                        f"{reduced_interval_seconds:.2f}s, interval={pruning_interval}s"
                    )
                    debug_callback(
                        f"  Starting route pruning for track '{track_name}': {total_points:,} points, interval: {pruning_interval} seconds"
                    )
                    debug_callback(
                        f"  Interpolation interval for track '{track_name}' (for later use): {interpolation_interval_seconds:.2f}s"
                    )
                
                # Create temporary route data for pruning
                # Note: combined_route already has distances in miles if imperial_units is True
                stored_total_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
                temp_route_data = {
                    'combined_route': combined_route,
                    'total_distance': stored_total_distance,
                    'total_points': len(combined_route),
                    'total_files': len(track_files),
                    'track_name': track_name,
                    'route_index': route_index
                }
                
                # Apply pruning (pass imperial_units flag)
                final_route_data = prune_route_by_interval(temp_route_data, pruning_interval, json_data, log_callback, debug_callback)
                
                if not final_route_data:
                    if log_callback:
                        log_callback(f"  Error: Pruning failed for track '{track_name}', using original route")
                    final_route_data = temp_route_data
                else:
                    # Update track_name and route_index in the pruned data
                    final_route_data['track_name'] = track_name
                    final_route_data['route_index'] = route_index
                # Store interpolation interval for later use
                final_route_data['interpolation_interval_seconds'] = interpolation_interval_seconds
                # Interpolate additional points to cover large gaps immediately after pruning (per track)
                try:
                    interpolated = interpolate_route_by_interval(
                        final_route_data.get('combined_route', []),
                        interpolation_interval_seconds,
                        log_callback,
                        debug_callback,
                    )
                    final_route_data['combined_route'] = interpolated
                    final_route_data['total_points'] = len(interpolated)
                except Exception as e:
                    if log_callback:
                        log_callback(f"  Warning: Interpolation step failed for track '{track_name}': {str(e)}")

        else:
            # No json_data provided, create route data without pruning
            # Convert total_distance to miles if imperial_units is True (but json_data is None, so use metric)
            stored_total_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
            final_route_data = {
                'combined_route': combined_route,
                'total_distance': stored_total_distance,
                'total_points': len(combined_route),
                'total_files': len(track_files),
                'track_name': track_name,
                'route_index': route_index
            }
        
        return final_route_data
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error creating route for track '{track_name}': {str(e)}")
        return None


def prune_route_by_interval(combined_route_data, interval_seconds, json_data=None, log_callback=None, debug_callback=None):
    """
    Prune route points by time interval, keeping points that are at least 
    interval_seconds apart in time.
    
    Args:
        combined_route_data (dict): Combined route data with 'combined_route' key
        interval_seconds (int): Minimum time interval in seconds between kept points
        json_data (dict, optional): Job data for checking imperial_units setting
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
    
    Returns:
        dict: Pruned route data with updated 'combined_route' and stats, or None if error
    """
    try:
        if not combined_route_data or 'combined_route' not in combined_route_data:
            if log_callback:
                log_callback("Error: Invalid combined route data for pruning")
            return None
        
        combined_route = combined_route_data['combined_route']
        
        if debug_callback:
            debug_callback(f"Pruning route by {interval_seconds} second intervals")
            debug_callback(f"Original route has {len(combined_route)} points")
        
        if len(combined_route) <= 2:
            if debug_callback:
                debug_callback("Route has 2 or fewer points, no pruning needed")
            return combined_route_data
        
        # Collect points with time information
        # Using RoutePoint named tuple for clarity
        points_with_time = []
        for i, point in enumerate(combined_route):
            points_with_time.append((i, point, point.timestamp))
        
        # Determine which points to keep
        indices_to_keep = []
        
        if len(points_with_time) == 0:
            if log_callback:
                log_callback("No points found in route")
            return None
        elif len(points_with_time) <= 2:
            # Keep all points if 2 or fewer
            indices_to_keep = [0, len(points_with_time) - 1] if len(points_with_time) == 2 else [0]
        else:
            # Always keep first point
            indices_to_keep.append(0)
            last_kept_time = points_with_time[0][2]  # timestamp from first point
            
            # Process middle points
            for i in range(1, len(points_with_time) - 1):
                _, point, current_time = points_with_time[i]
                
                # If either point doesn't have time data, keep it
                if current_time is None or last_kept_time is None:
                    indices_to_keep.append(i)
                    last_kept_time = current_time if current_time else last_kept_time
                    continue
                
                # Calculate time difference in seconds
                time_diff = abs((current_time - last_kept_time).total_seconds())
                
                # If points are far enough apart in time, keep this point
                if time_diff >= interval_seconds:
                    indices_to_keep.append(i)
                    last_kept_time = current_time
            
            # Always keep last point
            indices_to_keep.append(len(points_with_time) - 1)
        
        # Remove duplicate indices and sort
        indices_to_keep = sorted(list(set(indices_to_keep)))
        
        # Create pruned route with only kept points
        pruned_route = []
        for i in indices_to_keep:
            pruned_route.append(combined_route[i])
        
        # Check if imperial units should be used
        imperial_units = _is_imperial_units(json_data)
        
        # Recalculate accumulated distances and times for pruned route
        recalculated_route = []
        accumulated_distance = 0.0
        accumulated_time = 0.0
        prev_lat = None
        prev_lon = None
        prev_timestamp = None
        
        for i, point in enumerate(pruned_route):
            # Calculate distance from previous point (skip for first point or new route start)
            if prev_lat is not None and prev_lon is not None and not point.new_route_flag:
                point_distance = calculate_haversine_distance(prev_lat, prev_lon, point.lat, point.lon)
                accumulated_distance += point_distance
            
            # Calculate time increment from previous point (skip for first point or new route start)
            if point.timestamp is not None and prev_timestamp is not None and not point.new_route_flag:
                time_increment = (point.timestamp - prev_timestamp).total_seconds()
                if time_increment >= 0:  # Only add positive time increments
                    accumulated_time += time_increment
            
            # Convert accumulated_distance to miles if imperial_units is True
            stored_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
            
            # Add point with recalculated accumulated time and distance
            recalculated_route.append(RoutePoint(
                route_index=point.route_index,
                lat=point.lat,
                lon=point.lon,
                timestamp=point.timestamp,
                accumulated_time=accumulated_time,
                accumulated_distance=stored_distance,
                new_route_flag=point.new_route_flag,
                filename=point.filename,
                elevation=point.elevation,
                heart_rate=point.heart_rate
            ))
            
            # Update previous point and timestamp for next iteration
            prev_lat = point.lat
            prev_lon = point.lon
            prev_timestamp = point.timestamp
        
        # Convert total_distance to miles if imperial_units is True
        stored_total_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
        
        # Create pruned route data
        pruned_route_data = {
            'combined_route': recalculated_route,
            'total_distance': stored_total_distance,
            'total_points': len(recalculated_route),
            'total_files': combined_route_data.get('total_files', 0),
            'pruning_stats': {
                'original_count': len(combined_route),
                'pruned_count': len(recalculated_route),
                'deleted_count': len(combined_route) - len(recalculated_route),
                'interval_seconds': interval_seconds
            }
        }
        
        if debug_callback:
            debug_callback("Pruning complete")
        
        return pruned_route_data
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in route pruning: {str(e)}")
        return None


def interpolate_route_by_interval(route_points, interpolation_interval_seconds, log_callback=None, debug_callback=None):
    """Interpolate points so that gaps larger than the threshold in accumulated_time are filled.
    New points copy all attributes of the previous point except accumulated_time, which is advanced.
    Interpolation is skipped if the next point has new_route_flag=True.
    """
    try:
        if not route_points or len(route_points) < 2:
            return route_points
        
        original_count = len(route_points)
        new_points = []
        i = 0
        while i < len(route_points) - 1:
            current_point = route_points[i]
            next_point = route_points[i + 1]
            new_points.append(current_point)
            
            # Skip interpolation across new route boundaries
            if next_point.new_route_flag:
                i += 1
                continue
            
            # Use named attributes for accumulated_time
            current_time = current_point.accumulated_time
            next_time = next_point.accumulated_time
            
            # If times invalid or not increasing, move on safely
            try:
                time_gap = float(next_time) - float(current_time)
            except Exception:
                time_gap = 0.0
            
            # Add interpolated points until within threshold
            while time_gap > float(interpolation_interval_seconds):
                # Build a new point copying everything from current_point but with advanced accumulated_time
                new_accumulated_time = float(current_time) + float(interpolation_interval_seconds)
                
                # Create interpolated point - new_route_flag should be False for interpolated points
                new_point = RoutePoint(
                    route_index=current_point.route_index,
                    lat=current_point.lat,
                    lon=current_point.lon,
                    timestamp=current_point.timestamp,
                    accumulated_time=new_accumulated_time,
                    accumulated_distance=current_point.accumulated_distance,
                    new_route_flag=False,
                    filename=current_point.filename,
                    elevation=current_point.elevation,
                    heart_rate=current_point.heart_rate,
                )
                new_points.append(new_point)
                # Advance current for next iteration relative to the same next_point
                current_point = new_point
                current_time = new_accumulated_time
                time_gap = float(next_point.accumulated_time) - float(current_time)
            
            i += 1
        
        # Append the last original point
        new_points.append(route_points[-1])
        
        if debug_callback:
            debug_callback(f"Interpolation complete: {original_count} → {len(new_points)} points")
        
        return new_points
    except Exception as e:
        if log_callback:
            log_callback(f"Error during interpolation: {str(e)}")
        return route_points


def _reset_track_accumulated_values(track_points, json_data=None):
    """
    Resets accumulated_time and accumulated_distance for a single track's points.
    This is necessary because when splitting tracks, we want each track's accumulated
    values to start from 0, but we need to preserve the original accumulated values
    for the final combined_route.
    
    Args:
        track_points (list): List of RoutePoint objects
        json_data (dict, optional): Job data for checking imperial_units setting
    """
    if not track_points:
        return []
    
    # Check if imperial units should be used
    imperial_units = _is_imperial_units(json_data)
    
    recalculated_track = []
    accumulated_distance = 0.0
    accumulated_time = 0.0
    prev_lat = None
    prev_lon = None
    prev_timestamp = None
    
    for point in track_points:
        # Calculate distance from previous point (skip for first point or new route start)
        if prev_lat is not None and prev_lon is not None and not point.new_route_flag:
            point_distance = calculate_haversine_distance(prev_lat, prev_lon, point.lat, point.lon)
            accumulated_distance += point_distance
        
        # Calculate time increment from previous point (skip for first point or new route start)
        if point.timestamp is not None and prev_timestamp is not None and not point.new_route_flag:
            time_increment = (point.timestamp - prev_timestamp).total_seconds()
            if time_increment >= 0:  # Only add positive time increments
                accumulated_time += time_increment
        
        # Convert accumulated_distance to miles if imperial_units is True
        stored_distance = accumulated_distance * METERS_TO_MILES if imperial_units else accumulated_distance
        
        # Add point with recalculated accumulated time and distance
        recalculated_track.append(RoutePoint(
            route_index=point.route_index,
            lat=point.lat,
            lon=point.lon,
            timestamp=point.timestamp,
            accumulated_time=accumulated_time,
            accumulated_distance=stored_distance,
            new_route_flag=point.new_route_flag,
            filename=point.filename,
            elevation=point.elevation,
            heart_rate=point.heart_rate
        ))
        
        # Update previous point and timestamp for next iteration
        prev_lat = point.lat
        prev_lon = point.lon
        prev_timestamp = point.timestamp
    
    return recalculated_track
