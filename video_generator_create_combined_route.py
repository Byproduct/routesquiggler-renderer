"""
Video generation combined route creation for the Route Squiggler render client.
This module handles creating combined routes from multiple chronologically sorted GPX files.
Supports both single route mode and multiple simultaneous routes
based on track names from track_objects.
"""

from datetime import datetime, timedelta
from pathlib import Path
import gpxpy
from image_generator_utils import calculate_haversine_distance


def create_combined_route(sorted_gpx_files, json_data=None, progress_callback=None, log_callback=None):
    """
    Create combined routes from chronologically sorted GPX files.
    Supports multiple simultaneous routes based on track names.
    
    Args:
        sorted_gpx_files (list): List of GPX file info dictionaries sorted chronologically
        json_data (dict, optional): Job data containing configuration parameters and track_objects
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
    
    Returns:
        dict: Combined route data with 'combined_route' and 'total_distance' keys, or None if error
    """
    try:
        if log_callback:
            log_callback("Creating combined routes from sorted GPX files...")
        
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
                
                if log_callback:
                    log_callback(f"Multiple routes mode: Found {len(track_name_map)} named tracks")
            else:
                if log_callback:
                    log_callback("Single route mode: All track names are empty")
        
        # Used to be split into separate functions for single and multiple runners, but the function is now unified
        if not use_multiple_routes:
            # For single route mode, create an empty track_name_map so all files become "unnamed"
            track_name_map = {}
        
        return _create_multiple_routes(sorted_gpx_files, track_name_map, json_data, progress_callback, log_callback)
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in combined route creation: {str(e)}")
        return None



def _create_multiple_routes(sorted_gpx_files, track_name_map, json_data=None, progress_callback=None, log_callback=None):
    """
    Create combined routes from chronologically sorted GPX files.
    Handles both single runner (empty track_name_map) and multiple runners (populated track_name_map).
    """
    try:
        # Determine if this is single route mode (empty track_name_map means all files are unnamed)
        is_single_route_mode = len(track_name_map) == 0
        
        if log_callback:
            if is_single_route_mode:
                log_callback("Creating single combined route...")
            else:
                log_callback("Creating multiple combined routes...")
        
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
        
        if log_callback:
            log_callback(f"Grouped files by track: {len(files_by_track)} named tracks, {len(unnamed_files)} unnamed files")
            for track_name, files in files_by_track.items():
                log_callback(f"  Track '{track_name}': {len(files)} files")
        
        # Create routes for each track
        all_routes = []
        total_tracks = len(files_by_track)
        
        for track_index, (track_name, track_files) in enumerate(files_by_track.items()):
            if log_callback:
                log_callback(f"Processing track '{track_name}' ({track_index + 1}/{total_tracks})")
            
            # Update progress
            if progress_callback:
                progress_percentage = int((track_index / total_tracks) * 90)  # Leave 10% for final processing
                progress_text = f"Processing track {track_index + 1}/{total_tracks}: {track_name}"
                progress_callback("progress_bar_combined_route", progress_percentage, progress_text)
            
            # Create route for this track
            track_route_data = _create_route_for_track(track_files, track_index, track_name, json_data, log_callback)
            if track_route_data:
                all_routes.append(track_route_data)
        
        # Handle unnamed files if any (create a separate route)
        if unnamed_files:
            if log_callback:
                log_callback(f"Processing {len(unnamed_files)} unnamed files")
            
            unnamed_route_data = _create_route_for_track(unnamed_files, len(all_routes), "Unnamed", json_data, log_callback)
            if unnamed_route_data:
                all_routes.append(unnamed_route_data)
        
        if not all_routes:
            if log_callback:
                log_callback("No routes were created successfully")
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
                
                if len(first_point) >= 4 and len(last_point) >= 5:
                    route_start_timestamp = first_point[3]  # timestamp at index 3
                    route_duration = last_point[4]  # accumulated_time at index 4
                    
                    if route_start_timestamp:
                        # Calculate when this route ends (start + duration)
                        route_end_timestamp = route_start_timestamp + timedelta(seconds=route_duration)
                        
                        if earliest_start_time is None or route_start_timestamp < earliest_start_time:
                            earliest_start_time = route_start_timestamp
                        
                        if latest_end_time is None or route_end_timestamp > latest_end_time:
                            latest_end_time = route_end_timestamp
        
        # Calculate total accumulated time based on mode
        if is_single_route_mode:
            # For single route mode, use simple sequential time accumulation (like old _create_single_route)
            # Get the accumulated time from the last point of the single route
            if all_routes and all_routes[0].get('combined_route'):
                last_point = all_routes[0]['combined_route'][-1]
                total_accumulated_time = last_point[4] if len(last_point) > 4 else 0  # accumulated_time at index 4
            else:
                total_accumulated_time = 0
            if log_callback:
                log_callback(f"Single route mode: Using sequential time accumulation: {total_accumulated_time:.1f}s")
        else:
            # For multiple routes mode, use simultaneous mode logic
            if earliest_start_time and latest_end_time:
                total_span_seconds = (latest_end_time - earliest_start_time).total_seconds()
                
                if log_callback:
                    log_callback(f"SIMULTANEOUS MODE ANALYSIS:")
                    log_callback(f"  Earliest route starts at: {earliest_start_time.strftime('%H:%M:%S')}")
                    log_callback(f"  Latest route ends at: {latest_end_time.strftime('%H:%M:%S')}")
                    log_callback(f"  Total span of all routes: {total_span_seconds:.1f}s ({total_span_seconds/60:.1f} minutes)")
                    log_callback(f"  Video duration: {video_length:.1f}s")
                
                # Use the total span as total_accumulated_time
                # This allows all routes to be drawn at their proper staggered times
                total_accumulated_time = total_span_seconds
                
                if log_callback:
                    log_callback(f"SIMULTANEOUS MODE FIX: Using total span: {total_span_seconds:.1f}s")
                    log_callback(f"  This allows all routes to be drawn at their proper staggered times")
                    log_callback(f"  Routes that start after {video_length:.1f}s won't appear in the video")
            else:
                # Fallback to first route's time if no timestamps found
                total_accumulated_time = all_routes[0].get('total_accumulated_time', 0) if all_routes else 0
        
        # Calculate route_time_per_frame using the same method as caching phase
        # Use adjusted calculation to ensure complete route coverage
        route_time_per_frame = total_accumulated_time / total_frames
        
        # Verify that the last frame will capture all points
        final_frame_target_time = (total_frames - 1) * route_time_per_frame
        
        if final_frame_target_time < total_accumulated_time * 0.99:  # Allow 1% tolerance
            # Adjust route_time_per_frame to ensure complete coverage
            route_time_per_frame = total_accumulated_time / (total_frames - 1)  # Adjust for 0-based indexing
        
        # Calculate gpx_time_per_video_time using the same method as caching phase
        gpx_time_per_video_time = route_time_per_frame * video_fps
        
        if log_callback:
            log_callback(f"Created {len(all_routes)} routes:")
            for i, route in enumerate(all_routes):
                route_name = route.get('track_name', f'Route {i}')
                points = route.get('total_points', 0)
                distance = route.get('total_distance', 0)
                log_callback(f"  Route {i}: '{route_name}' - {points} points, {distance:.2f}m")
            log_callback(f"Total across all routes: {total_points} points, {total_distance:.2f}m")
        
        # Write debug file for all routes
        debug_file_path = Path("temporary files/route") / "debug_tuple.txt"
        try:
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                f.write("Route_Index, Latitude, Longitude, Timestamp, Accumulated time (s), Accumulated Distance (m), New Route, Filename, Elevation\n")
                for route in all_routes:
                    combined_route = route.get('combined_route', [])
                    for route_index, lat, lon, timestamp, accumulated_time, distance, new_route, filename, elevation in combined_route:
                        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "None"
                        elevation_str = str(elevation) if elevation is not None else ""
                        f.write(f"{route_index}, {lat}, {lon}, {timestamp_str}, {accumulated_time:.0f}, {distance:.2f}, {new_route}, {filename}, {elevation_str}\n")
            
            if log_callback:
                log_callback(f"Wrote debug_tuple.txt successfully to {debug_file_path}")
        except Exception as e:
            if log_callback:
                log_callback(f"Error writing debug file: {str(e)}")
        
        # Progress update before saving
        if progress_callback:
            progress_callback("progress_bar_combined_route", 95, "Saving combined routes")
        
        # Add gpx_time_per_video_time to all routes BEFORE saving
        for route in all_routes:
            route['gpx_time_per_video_time'] = gpx_time_per_video_time
              
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


def _create_route_for_track(track_files, route_index, track_name, json_data=None, log_callback=None):
    """
    Create a single route for a specific track from its files.
    """
    try:
        if log_callback:
            log_callback(f"Creating route {route_index} for track '{track_name}' with {len(track_files)} files")
        
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
        
        # Process each file in chronological order
        for file_index, gpx_info in enumerate(track_files):
            filename = gpx_info.get('filename', f'file_{file_index}')
            content = gpx_info.get('content', '')
            
            if log_callback:
                log_callback(f"  Processing {filename}, accumulated distance: {accumulated_distance:.2f} meters, accumulated time: {accumulated_time:.1f} seconds")
            
            # Extract filename without extension for the legend or name tags if the user has either enabled
            if (json_data.get('name_tags', False)) or (json_data.get('legend', 'off') != 'off'):
                filename_without_ext = Path(filename).stem
            else:
                filename_without_ext = None
            
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
                            
                            # Add to combined route with structure: route_index, lat, lon, timestamp, accumulated_time, accumulated_distance, new_route_flag, filename, elevation
                            combined_route.append((
                                route_index,             # Route index (unique for each track)
                                lat,                    # Latitude
                                lon,                    # Longitude
                                timestamp,              # Timestamp
                                accumulated_time,       # Accumulated time in seconds
                                accumulated_distance,   # Accumulated distance in meters
                                first_point_in_file,    # New route flag
                                filename_without_ext,   # Filename without extension
                                elevation               # Elevation
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
        
        if log_callback:
            log_callback(f"  Completed track '{track_name}'. Total points: {len(combined_route)}")
            log_callback(f"  Total distance: {accumulated_distance:.2f} meters ({accumulated_distance/1000:.2f} km)")
            log_callback(f"  Total time: {accumulated_time:.1f} seconds ({accumulated_time/60:.1f} minutes)")
        
        # Apply pruning if json_data is provided
        if json_data:
            # Check if pruning should be applied based on job parameters
            route_accuracy = json_data.get('route_accuracy', 'normal')  # Default to 'normal' if not specified
            prune_route = route_accuracy != 'maximum'  # Prune unless route_accuracy is 'maximum'
            total_points = len(combined_route)
            
            if not prune_route:
                # Pruning disabled by job parameters (route_accuracy = 'maximum')
                if log_callback:
                    log_callback(f"  Route pruning skipped for track '{track_name}' as per job parameters (route_accuracy = 'maximum').")
                final_route_data = {
                    'combined_route': combined_route,
                    'total_distance': accumulated_distance,
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
                # Compute route_time_per_frame for this track and adjust for full coverage
                route_time_per_frame = accumulated_time / total_frames
                final_frame_target_time = (total_frames - 1) * route_time_per_frame
                if final_frame_target_time < accumulated_time * 0.99:
                    route_time_per_frame = accumulated_time / max(1.0, (total_frames - 1))
                gpx_time_per_video_time = route_time_per_frame * video_fps
                raw_interval_seconds = gpx_time_per_video_time / video_fps
                
                reduced_interval_seconds = raw_interval_seconds * 0.8
                pruning_interval = max(1, int(round(reduced_interval_seconds)))
                interpolation_interval_seconds = raw_interval_seconds * 2.0
                
                if log_callback:
                    log_callback(
                        f"  Calculated dynamic pruning interval for track '{track_name}': ~1 point/frame → "
                        f"raw={raw_interval_seconds:.2f}s (gpx_time_per_video_time/video_fps), reduced by 20% → "
                        f"{reduced_interval_seconds:.2f}s, interval={pruning_interval}s"
                    )
                    log_callback(
                        f"  Starting route pruning for track '{track_name}': {total_points:,} points, interval: {pruning_interval} seconds"
                    )
                    log_callback(
                        f"  Interpolation interval for track '{track_name}' (for later use): {interpolation_interval_seconds:.2f}s"
                    )
                
                # Create temporary route data for pruning
                temp_route_data = {
                    'combined_route': combined_route,
                    'total_distance': accumulated_distance,
                    'total_points': len(combined_route),
                    'total_files': len(track_files),
                    'track_name': track_name,
                    'route_index': route_index
                }
                
                # Apply pruning
                final_route_data = prune_route_by_interval(temp_route_data, pruning_interval, log_callback)
                
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
                    )
                    final_route_data['combined_route'] = interpolated
                    final_route_data['total_points'] = len(interpolated)
                except Exception as e:
                    if log_callback:
                        log_callback(f"  Warning: Interpolation step failed for track '{track_name}': {str(e)}")

        else:
            # No json_data provided, create route data without pruning
            final_route_data = {
                'combined_route': combined_route,
                'total_distance': accumulated_distance,
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


def prune_route_by_interval(combined_route_data, interval_seconds, log_callback=None):
    """
    Prune route points by time interval, keeping points that are at least 
    interval_seconds apart in time.
    
    Args:
        combined_route_data (dict): Combined route data with 'combined_route' key
        interval_seconds (int): Minimum time interval in seconds between kept points
        log_callback (callable, optional): Function to call for logging messages
    
    Returns:
        dict: Pruned route data with updated 'combined_route' and stats, or None if error
    """
    try:
        if not combined_route_data or 'combined_route' not in combined_route_data:
            if log_callback:
                log_callback("Error: Invalid combined route data for pruning")
            return None
        
        combined_route = combined_route_data['combined_route']
        
        if log_callback:
            log_callback(f"Pruning route by {interval_seconds} second intervals...")
            log_callback(f"Original route has {len(combined_route)} points")
        
        if len(combined_route) <= 2:
            if log_callback:
                log_callback("Route has 2 or fewer points, no pruning needed")
            return combined_route_data
        
        # Collect points with time information
        # Combined route structure: (route_index, lat, lon, timestamp, accumulated_time, accumulated_distance, new_route_flag, filename, elevation)
        points_with_time = []
        for i, point in enumerate(combined_route):
            route_index, lat, lon, timestamp, accumulated_time, accumulated_distance, new_route_flag, filename, elevation = point
            points_with_time.append((i, point, timestamp))
        
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
        
        # Recalculate accumulated distances and times for pruned route
        recalculated_route = []
        accumulated_distance = 0.0
        accumulated_time = 0.0
        prev_lat = None
        prev_lon = None
        prev_timestamp = None
        
        for i, point in enumerate(pruned_route):
            route_index, lat, lon, timestamp, old_accumulated_time, old_distance, new_route_flag, filename, elevation = point
            
            # Calculate distance from previous point (skip for first point or new route start)
            if prev_lat is not None and prev_lon is not None and not new_route_flag:
                point_distance = calculate_haversine_distance(prev_lat, prev_lon, lat, lon)
                accumulated_distance += point_distance
            
            # Calculate time increment from previous point (skip for first point or new route start)
            if timestamp is not None and prev_timestamp is not None and not new_route_flag:
                time_increment = (timestamp - prev_timestamp).total_seconds()
                if time_increment >= 0:  # Only add positive time increments
                    accumulated_time += time_increment
            
            # Add point with recalculated accumulated time and distance
            recalculated_route.append((
                route_index,
                lat,
                lon,
                timestamp,
                accumulated_time,
                accumulated_distance,
                new_route_flag,
                filename,
                elevation
            ))
            
            # Update previous point and timestamp for next iteration
            prev_lat = lat
            prev_lon = lon
            prev_timestamp = timestamp
        
        # Create pruned route data
        pruned_route_data = {
            'combined_route': recalculated_route,
            'total_distance': accumulated_distance,
            'total_points': len(recalculated_route),
            'total_files': combined_route_data.get('total_files', 0),
            'pruning_stats': {
                'original_count': len(combined_route),
                'pruned_count': len(recalculated_route),
                'deleted_count': len(combined_route) - len(recalculated_route),
                'interval_seconds': interval_seconds
            }
        }
        
        if log_callback:
            log_callback("Pruning complete")
        
        return pruned_route_data
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in route pruning: {str(e)}")
        return None


def interpolate_route_by_interval(route_points, interpolation_interval_seconds, log_callback=None):
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
            next_is_new_route = False
            if len(next_point) >= 7:
                next_is_new_route = bool(next_point[6])
            
            if next_is_new_route:
                i += 1
                continue
            
            # Accumulated time indices (index 4)
            current_time = current_point[4]
            next_time = next_point[4]
            
            # If times invalid or not increasing, move on safely
            try:
                time_gap = float(next_time) - float(current_time)
            except Exception:
                time_gap = 0.0
            
            # Add interpolated points until within threshold
            while time_gap > float(interpolation_interval_seconds):
                # Build a new point copying everything from current_point but with advanced accumulated_time
                new_accumulated_time = float(current_time) + float(interpolation_interval_seconds)
                route_index, lat, lon, timestamp, _, accumulated_distance, _, filename, elevation = (
                    current_point[0], current_point[1], current_point[2], current_point[3], current_point[4],
                    current_point[5], current_point[6] if len(current_point) > 6 else False, current_point[7] if len(current_point) > 7 else None,
                    current_point[8] if len(current_point) > 8 else None
                )
                # new_route_flag should be False for interpolated points
                new_point = (
                    route_index,
                    lat,
                    lon,
                    timestamp,
                    new_accumulated_time,
                    accumulated_distance,
                    False,
                    filename,
                    elevation,
                )
                new_points.append(new_point)
                # Advance current for next iteration relative to the same next_point
                current_point = new_point
                current_time = new_accumulated_time
                time_gap = float(next_point[4]) - float(current_time)
            
            i += 1
        
        # Append the last original point
        new_points.append(route_points[-1])
        
        if log_callback:
            log_callback(f"Interpolation complete: {original_count} → {len(new_points)} points")
        
        return new_points
    except Exception as e:
        if log_callback:
            log_callback(f"Error during interpolation: {str(e)}")
        return route_points


def _reset_track_accumulated_values(track_points):
    """
    Resets accumulated_time and accumulated_distance for a single track's points.
    This is necessary because when splitting tracks, we want each track's accumulated
    values to start from 0, but we need to preserve the original accumulated values
    for the final combined_route.
    """
    if not track_points:
        return []
    
    recalculated_track = []
    accumulated_distance = 0.0
    accumulated_time = 0.0
    prev_lat = None
    prev_lon = None
    prev_timestamp = None
    
    for point in track_points:
        route_index, lat, lon, timestamp, old_accumulated_time, old_distance, new_route_flag, filename, elevation = point
        
        # Calculate distance from previous point (skip for first point or new route start)
        if prev_lat is not None and prev_lon is not None and not new_route_flag:
            point_distance = calculate_haversine_distance(prev_lat, prev_lon, lat, lon)
            accumulated_distance += point_distance
        
        # Calculate time increment from previous point (skip for first point or new route start)
        if timestamp is not None and prev_timestamp is not None and not new_route_flag:
            time_increment = (timestamp - prev_timestamp).total_seconds()
            if time_increment >= 0:  # Only add positive time increments
                accumulated_time += time_increment
        
        # Add point with recalculated accumulated time and distance
        recalculated_track.append((
            route_index,
            lat,
            lon,
            timestamp,
            accumulated_time,
            accumulated_distance,
            new_route_flag,
            filename,
            elevation
        ))
        
        # Update previous point and timestamp for next iteration
        prev_lat = lat
        prev_lon = lon
        prev_timestamp = timestamp
    
    return recalculated_track

