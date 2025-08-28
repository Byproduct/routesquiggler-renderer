"""
Video generation combined route creation for the Route Squiggler render client.
This module handles creating combined routes from multiple chronologically sorted GPX files.
Supports both single route mode (backward compatibility) and multiple simultaneous routes
based on track names from track_objects.
"""

import os
import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import gpxpy


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
        
        if use_multiple_routes:
            return _create_multiple_routes(sorted_gpx_files, track_name_map, json_data, progress_callback, log_callback)
        else:
            return _create_single_route(sorted_gpx_files, json_data, progress_callback, log_callback)
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in combined route creation: {str(e)}")
        return None


def _create_single_route(sorted_gpx_files, json_data=None, progress_callback=None, log_callback=None):
    """
    Create a single combined route from chronologically sorted GPX files.
    This is the original implementation for backward compatibility.
    """
    try:
        if log_callback:
            log_callback("Creating single combined route...")
        
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
        
        total_files = len(sorted_gpx_files)
        
        if progress_callback:
            progress_callback("progress_bar_combined_route", 0, "Starting single route creation")
        
        # Process each file in chronological order
        for file_index, gpx_info in enumerate(sorted_gpx_files):
            filename = gpx_info.get('filename', f'file_{file_index}')
            content = gpx_info.get('content', '')
            
            if log_callback:
                log_callback(f"Processing {filename}, accumulated distance: {accumulated_distance:.2f} meters, accumulated time: {accumulated_time:.1f} seconds")
            
            # Update progress
            if progress_callback:
                progress_percentage = int((file_index / total_files) * 90)  # Leave 10% for final processing
                progress_text = f"Processing file {file_index + 1}/{total_files}: {Path(filename).stem}"
                progress_callback("progress_bar_combined_route", progress_percentage, progress_text)
            
            # Extract filename without extension for labels
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
                            
                            # Calculate distance from previous point
                            point_distance = 0.0
                            if prev_lat is not None and prev_lon is not None and not first_point_in_file:
                                # Convert decimal degrees to radians
                                lat1, lon1, lat2, lon2 = map(math.radians, [prev_lat, prev_lon, lat, lon])
                                
                                # Haversine formula
                                dlat = lat2 - lat1
                                dlon = lon2 - lon1
                                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                                c = 2 * math.asin(math.sqrt(a))
                                # Radius of earth in meters
                                r = 6371000
                                point_distance = c * r
                                
                                accumulated_distance += point_distance
                            
                            # Calculate time increment from previous point (skip for first point in file)
                            if timestamp is not None and prev_timestamp is not None and not first_point_in_file:
                                time_increment = (timestamp - prev_timestamp).total_seconds()
                                if time_increment >= 0:  # Only add positive time increments
                                    accumulated_time += time_increment
                            elif prev_timestamp is None and timestamp is not None:
                                # First timestamp encountered - log it
                                if log_callback:
                                    log_callback(f"First timestamp encountered: {timestamp}")
                            
                            # Add to combined route with structure: route_index, lat, lon, timestamp, accumulated_time, accumulated_distance, new_route_flag, filename, elevation
                            combined_route.append((
                                0,                      # Route index (0 for single route)
                                lat,                    # Latitude
                                lon,                    # Longitude
                                timestamp,              # Timestamp
                                accumulated_time,       # Accumulated time in seconds
                                accumulated_distance,   # Accumulated distance in meters
                                first_point_in_file,    # New route flag
                                filename_without_ext,   # Filename without extension
                                elevation                # Elevation
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
                    log_callback(f"Error processing file {filename}: {str(e)}")
                continue
        
        if log_callback:
            log_callback(f"Completed processing all files. Total points: {len(combined_route)}")
            log_callback(f"Total distance: {accumulated_distance:.2f} meters ({accumulated_distance/1000:.2f} km)")
            log_callback(f"Total time: {accumulated_time:.1f} seconds ({accumulated_time/60:.1f} minutes)")
        
        # Calculate GPX time per video time ratio
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
        
        # Calculate route_time_per_frame using the same method as caching phase
        # Use adjusted calculation to ensure complete route coverage
        route_time_per_frame = accumulated_time / total_frames
        
        # Verify that the last frame will capture all points
        final_frame_target_time = (total_frames - 1) * route_time_per_frame
        
        if final_frame_target_time < accumulated_time * 0.99:  # Allow 1% tolerance
            # Adjust route_time_per_frame to ensure complete coverage
            route_time_per_frame = accumulated_time / (total_frames - 1)  # Adjust for 0-based indexing
        
        # Calculate gpx_time_per_video_time using the same method as caching phase
        gpx_time_per_video_time = route_time_per_frame * video_fps
        
        if log_callback:
            log_callback(f"Video length: {video_length} seconds")
            log_callback(f"Video fps: {video_fps}")
            log_callback(f"Total frames: {total_frames}")
            log_callback(f"Route time per frame: {route_time_per_frame:.6f} seconds")
            log_callback(f"Final frame target time: {(total_frames - 1) * route_time_per_frame:.1f} seconds")
            log_callback(f"Route coverage: {((total_frames - 1) * route_time_per_frame / accumulated_time * 100):.1f}%")
            log_callback(f"GPX time per video time: {gpx_time_per_video_time:.6f} seconds of GPX time per second of video")
        
        # Check if pruning should be applied based on job parameters
        route_accuracy = json_data.get('route_accuracy', 'normal') if json_data else 'normal'  # Default to 'normal' if not specified
        prune_route = route_accuracy != 'maximum'  # Prune unless route_accuracy is 'maximum'
        total_points = len(combined_route)
        
        if not prune_route:
            # Pruning disabled by job parameters (route_accuracy = 'maximum')
            if log_callback:
                log_callback("Route pruning skipped as per job parameters (route_accuracy = 'maximum').")
            final_route_data = {
                'combined_route': combined_route,
                'total_distance': accumulated_distance,
                'total_points': len(combined_route),
                'total_files': total_files
            }
        else:
            # Determine pruning interval dynamically to target ~1 point per video frame
            try:
                # Use gpx_time_per_video_time and video_fps to calculate raw per-frame GPX seconds
                raw_interval_seconds = gpx_time_per_video_time / float(video_fps)
            except Exception:
                # Fallback to route_time_per_frame if division fails
                raw_interval_seconds = float(route_time_per_frame)
            # Reduce threshold by 20% to keep a few more points
            reduced_interval_seconds = raw_interval_seconds * 0.8
            # Round to integer seconds, minimum 1 second
            pruning_interval = max(1, int(round(reduced_interval_seconds)))
            # Also calculate interpolation interval for later use
            interpolation_interval_seconds = raw_interval_seconds * 2.0
            
            if log_callback:
                log_callback(
                    f"Calculated dynamic pruning interval: ~1 point/frame → raw={raw_interval_seconds:.2f}s (gpx_time_per_video_time/video_fps), "
                    f"reduced by 20% → {reduced_interval_seconds:.2f}s, interval={pruning_interval}s"
                )
                log_callback(f"Starting route pruning: {total_points:,} points, interval: {pruning_interval} seconds")
                log_callback(f"Interpolation interval (for later use): {interpolation_interval_seconds:.2f}s")
            
            # Create temporary route data for pruning
            temp_route_data = {
                'combined_route': combined_route,
                'total_distance': accumulated_distance,
                'total_points': len(combined_route),
                'total_files': total_files
            }
            
            # Apply pruning
            final_route_data = prune_route_by_interval(temp_route_data, pruning_interval, log_callback)
            
            if not final_route_data:
                if log_callback:
                    log_callback("Error: Pruning failed, using original route")
                final_route_data = temp_route_data
            # Store interpolation interval for later use
            final_route_data['interpolation_interval_seconds'] = interpolation_interval_seconds
            # Interpolate additional points to cover large gaps immediately after pruning
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
                    log_callback(f"Warning: Interpolation step failed: {str(e)}")
        
        # Update variables for file writing
        combined_route = final_route_data['combined_route']
        accumulated_distance = final_route_data['total_distance']
        
        # Write debug file
        debug_file_path = Path("temporary files/route") / "debug_tuple.txt"
        try:
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                f.write("Route_Index, Latitude, Longitude, Timestamp, Accumulated time (s), Accumulated Distance (m), New Route, Filename, Elevation\n")
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
            progress_callback("progress_bar_combined_route", 95, "Saving combined route")
        
        # Add gpx_time_per_video_time to the route data BEFORE saving
        final_route_data['gpx_time_per_video_time'] = gpx_time_per_video_time
        
        # Write result to the combined route file as a list for future multiple routes support
        # combined_route_file = Path("temporary files/route") / "combined_route.pkl"
        
        # Wrap the route data in a list to support multiple routes in the future
        # routes_list = [final_route_data]
        
        # try:
        #     with open(combined_route_file, 'wb') as f:
        #         pickle.dump(routes_list, f)
            
        #     if log_callback:
        #         log_callback(f"Saved combined route to {combined_route_file}")
        # except Exception as e:
        #     if log_callback:
        #         log_callback(f"Error saving combined route file: {str(e)}")
        #     return None
        
        # Final progress update
        if progress_callback:
            progress_callback("progress_bar_combined_route", 100, "Route creation completed")
        
        return final_route_data
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in single route creation: {str(e)}")
        return None


def _create_multiple_routes(sorted_gpx_files, track_name_map, json_data=None, progress_callback=None, log_callback=None):
    """
    Create multiple combined routes from chronologically sorted GPX files,
    one route per unique track name.
    """
    try:
        if log_callback:
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
        
        # Calculate the total span of all routes
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
            # Fallback to original calculation if timestamps are missing
            total_accumulated_time = 0.0
            for route in all_routes:
                combined_route = route.get('combined_route', [])
                if combined_route:
                    # Get the last point's accumulated time
                    last_point = combined_route[-1]
                    route_accumulated_time = last_point[4]  # accumulated_time at index 4
                    total_accumulated_time = max(total_accumulated_time, route_accumulated_time)
            
            if log_callback:
                log_callback(f"SIMULTANEOUS MODE: Using fallback calculation (timestamps missing)")
                log_callback(f"  Total accumulated time (max individual duration): {total_accumulated_time:.1f}s")
        
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
        
        # Write result to the combined route file
        # combined_route_file = Path("temporary files/route") / "combined_route.pkl"
        
        # try:
        #     with open(combined_route_file, 'wb') as f:
        #         pickle.dump(all_routes, f)
            
        #     if log_callback:
        #         log_callback(f"Saved {len(all_routes)} combined routes to {combined_route_file}")
        # except Exception as e:
        #     if log_callback:
        #         log_callback(f"Error saving combined route file: {str(e)}")
        #     return None
        
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
            
            # Extract filename without extension for labels
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
                            
                            # Calculate distance from previous point
                            point_distance = 0.0
                            if prev_lat is not None and prev_lon is not None and not first_point_in_file:
                                # Convert decimal degrees to radians
                                lat1, lon1, lat2, lon2 = map(math.radians, [prev_lat, prev_lon, lat, lon])
                                
                                # Haversine formula
                                dlat = lat2 - lat1
                                dlon = lon2 - lon1
                                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                                c = 2 * math.asin(math.sqrt(a))
                                # Radius of earth in meters
                                r = 6371000
                                point_distance = c * r
                                
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
                                elevation                # Elevation
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
                # Determine pruning interval dynamically to target ~1 point per video frame (per track)
                try:
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
                except Exception:
                    # Conservative fallback: aim to keep more points
                    raw_interval_seconds = max(1.0, accumulated_time / 10000.0)
                    video_fps = 30.0
                
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


def load_all_routes(log_callback=None):
    """
    Load all combined routes from the saved file.
    
    Args:
        log_callback (callable, optional): Function to call for logging messages
    
    Returns:
        list: List of all route data dictionaries, or None if error
    """
    try:
        combined_route_file = Path("temporary files/route/combined_route.pkl")
        
        if not combined_route_file.exists():
            if log_callback:
                log_callback(f"Combined route file not found: {combined_route_file}")
            return None
        
        with open(combined_route_file, 'rb') as f:
            routes_data = pickle.load(f)
        
        # Handle both old single route format and new list format
        if isinstance(routes_data, list):
            if len(routes_data) > 0:
                if log_callback:
                    log_callback(f"Loaded {len(routes_data)} combined routes")
                return routes_data
            else:
                if log_callback:
                    log_callback("No routes found in file")
                return None
        else:
            # Legacy single route format - wrap in list
            if log_callback:
                log_callback("Loaded legacy single route format")
            return [routes_data]
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error loading all routes: {str(e)}")
        return None


def load_combined_route(log_callback=None):
    """
    Load combined route data from the saved file.
    Returns the first route for backward compatibility, but includes all routes info if multiple routes exist.
    For single route groups with multiple tracks, splits tracks into separate all_routes entries.
    """
    try:
        combined_route_file = Path("temporary files/route/combined_route.pkl")
        
        if not combined_route_file.exists():
            if log_callback:
                log_callback(f"Combined route file not found: {combined_route_file}")
            return None
        
        with open(combined_route_file, 'rb') as f:
            routes_data = pickle.load(f)
        
        # Handle both old single route format and new list format
        if isinstance(routes_data, list):
            if len(routes_data) > 0:
                # Check if we have multiple route groups or a single route group with multiple tracks
                if len(routes_data) > 1:
                    # Multiple route groups - use as-is
                    result = routes_data[0]  # Return first route for backward compatibility
                    result['all_routes'] = routes_data
                    result['total_routes'] = len(routes_data)
                    
                    if log_callback:
                        log_callback(f"Loaded {len(routes_data)} combined routes:")
                        for i, route in enumerate(routes_data):
                            track_name = route.get('track_name', f'Route {i}')
                            points = route.get('total_points', 0)
                            distance = route.get('total_distance', 0)
                            log_callback(f"  Route {i}: '{track_name}' - {points} points, {distance:.2f}m")
                else:
                    # Single entry - check if it contains multiple tracks within it
                    single_route = routes_data[0]
                    combined_route = single_route.get('combined_route', [])
                    
                    # Count the number of tracks by counting new_route_flag occurrences
                    track_count = 0
                    for point in combined_route:
                        if len(point) >= 7 and point[6]:  # new_route_flag at index 6
                            track_count += 1
                    
                    if track_count > 1:
                        # Single route group with multiple tracks - split into separate all_routes entries
                        if log_callback:
                            log_callback(f"Detected single route group with {track_count} tracks - splitting for proper timing calculation")
                        
                        # Check if we should use sequential or concurrent timing
                        # For single route groups (same route_index), use sequential timing
                        # For multiple route groups (different route_index), use concurrent timing with reset
                        use_sequential_timing = True  # Single route group = sequential
                        
                        # Split tracks based on new_route_flag
                        all_routes = []
                        current_track = []
                        current_track_index = 0
                        
                        for point in combined_route:
                            if len(point) >= 7:  # Check for minimum length
                                new_route_flag = point[6]  # New route flag at index 6
                                if new_route_flag and current_track:  # If new_route is True and we have points
                                    # Process and save previous track
                                    if use_sequential_timing:
                                        # Sequential timing: keep original accumulated values
                                        processed_track = current_track
                                    else:
                                        # Concurrent timing: reset accumulated values
                                        processed_track = _reset_track_accumulated_values(current_track)
                                    
                                    track_data = {
                                        'combined_route': processed_track,
                                        'total_points': len(processed_track),
                                        'total_distance': processed_track[-1][5] if processed_track else 0,  # accumulated_distance at index 5
                                        'track_name': f'Track {current_track_index}',
                                        'route_index': processed_track[0][0] if processed_track else 0  # route_index at index 0
                                    }
                                    all_routes.append(track_data)
                                    current_track = []
                                    current_track_index += 1
                            current_track.append(point)
                        
                        # Add the last track if not empty
                        if current_track:
                            if use_sequential_timing:
                                # Sequential timing: keep original accumulated values
                                processed_track = current_track
                            else:
                                # Concurrent timing: reset accumulated values
                                processed_track = _reset_track_accumulated_values(current_track)
                            
                            track_data = {
                                'combined_route': processed_track,
                                'total_points': len(processed_track),
                                'total_distance': processed_track[-1][5] if processed_track else 0,  # accumulated_distance at index 5
                                'track_name': f'Track {current_track_index}',
                                'route_index': processed_track[0][0] if processed_track else 0  # route_index at index 0
                            }
                            all_routes.append(track_data)
                        
                        # Create result with all_routes info
                        result = all_routes[0]  # Return first track for backward compatibility
                        result['all_routes'] = all_routes
                        result['total_routes'] = len(all_routes)
                        
                        if log_callback:
                            timing_type = "sequential" if use_sequential_timing else "concurrent"
                            log_callback(f"Split single route group into {len(all_routes)} tracks ({timing_type} timing):")
                            for i, track in enumerate(all_routes):
                                track_name = track.get('track_name', f'Track {i}')
                                points = track.get('total_points', 0)
                                distance = track.get('total_distance', 0)
                                last_point = track.get('combined_route', [])[-1] if track.get('combined_route') else None
                                duration = last_point[4] if last_point and len(last_point) > 4 else 0  # accumulated_time at index 4
                                log_callback(f"  Track {i}: '{track_name}' - {points} points, {distance:.2f}m, {duration:.1f}s")
                    else:
                        # True single route (no multiple tracks)
                        result = single_route
                    if log_callback:
                        log_callback(f"Loaded combined route with {result.get('total_points', 0)} points (single route)")
            else:
                if log_callback:
                    log_callback("No routes found in file")
                return None
        else:
            # Legacy single route format
            result = routes_data
            if log_callback:
                log_callback(f"Loaded combined route with {result.get('total_points', 0)} points (legacy format)")
        
        return result
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error loading combined route: {str(e)}")
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
                # Convert decimal degrees to radians
                lat1, lon1, lat2, lon2 = map(math.radians, [prev_lat, prev_lon, lat, lon])
                
                # Haversine formula
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                # Radius of earth in meters
                r = 6371000
                point_distance = c * r
                
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
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [prev_lat, prev_lon, lat, lon])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            # Radius of earth in meters
            r = 6371000
            point_distance = c * r
            
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


def test_altitude_structure():
    """
    Test function to verify the new elevation field structure works correctly.
    This function can be called to test the elevation support.
    """
    print("Testing elevation field structure...")
    
    # Create a test route point with elevation
    test_point = (
        0,                      # route_index
        68.385591,             # lat
        23.643681,             # lon
        datetime.now(),        # timestamp
        0.0,                   # accumulated_time
        0.0,                   # accumulated_distance
        True,                  # new_route_flag
        "test_file",           # filename
        100.5                  # elevation
    )
    
    # Verify the structure
    assert len(test_point) == 9, f"Expected 9 elements, got {len(test_point)}"
    assert test_point[8] == 100.5, f"Expected elevation 100.5, got {test_point[8]}"
    
    # Test with None elevation (when not extracted from GPX)
    test_point_none = (
        0,                      # route_index
        68.385591,             # lat
        23.643681,             # lon
        datetime.now(),        # timestamp
        0.0,                   # accumulated_time
        0.0,                   # accumulated_distance
        True,                  # new_route_flag
        "test_file",           # filename
        None                   # elevation (None when not extracted)
    )
    
    assert len(test_point_none) == 9, f"Expected 9 elements, got {len(test_point_none)}"
    assert test_point_none[8] is None, f"Expected elevation None, got {test_point_none[8]}"
    
    print("✓ Elevation structure test passed!")
    return True


def test_elevation_statistics():
    """
    Test function to verify the elevation statistics calculation works correctly.
    """
    print("Testing elevation statistics calculation...")
    
    # Create test data with elevation
    test_points = [
        (0, 68.385591, 23.643681, datetime.now(), 0.0, 0.0, True, "test_file", 100.0),   # First point
        (0, 68.385592, 23.643682, datetime.now(), 1.0, 10.0, False, "test_file", 105.0), # Second point
        (0, 68.385593, 23.643683, datetime.now(), 2.0, 20.0, False, "test_file", 110.0), # Third point
    ]
    
    # Test JSON data with elevation enabled
    test_json_data = {
        'statistics_current_elevation': True,
        'video_length': 10,
        'video_fps': 30
    }
    
    # Import the statistics function
    from video_generator_route_statistics import _calculate_video_statistics
    
    # Test single route mode
    statistics_data = _calculate_video_statistics(
        test_points, 
        test_json_data, 
        gpx_time_per_video_time=1.0
    )
    
    if statistics_data and 'current_elevation' in statistics_data:
        print(f"✓ Current elevation calculated: {statistics_data['current_elevation']}")
    else:
        print("✗ Current elevation not calculated")
    
    print("✓ Elevation statistics test completed!")
    return True


def test_elevation_display():
    """
    Test function to verify the elevation display functionality works correctly.
    """
    print("Testing elevation display functionality...")
    
    # Create test data with elevation
    test_points = [
        (0, 68.385591, 23.643681, datetime.now(), 0.0, 0.0, True, "test_file", 100.0),   # First point
        (0, 68.385592, 23.643682, datetime.now(), 1.0, 10.0, False, "test_file", 105.0), # Second point
        (0, 68.385593, 23.643683, datetime.now(), 2.0, 20.0, False, "test_file", 110.0), # Third point
    ]
    
    # Test JSON data with elevation enabled
    test_json_data = {
        'statistics_current_elevation': True,
        'statistics_current_speed': True,
        'video_length': 10,
        'video_fps': 30
    }
    
    # Import the statistics function
    from video_generator_route_statistics import _calculate_video_statistics
    
    # Test with gpx_time_per_video_time = 1.0
    statistics_data = _calculate_video_statistics(
        test_points, 
        test_json_data, 
        gpx_time_per_video_time=1.0
    )
    
    print(f"Statistics data: {statistics_data}")
    
    if statistics_data:
        print(f"Current speed: {statistics_data.get('current_speed', 'Not found')}")
        print(f"Current elevation: {statistics_data.get('current_elevation', 'Not found')}")
    
    print("✓ Elevation display test completed!")
    return True


def test_video_generation_conditions():
    """
    Test function to simulate the actual video generation conditions.
    """
    print("Testing video generation conditions...")
    
    # Create test data with elevation - more realistic values
    from datetime import datetime, timedelta
    base_time = datetime.now()
    
    test_points = [
        (0, 68.385591, 23.643681, base_time, 0.0, 0.0, True, "test_file", 100.0),                    # First point
        (0, 68.385592, 23.643682, base_time + timedelta(seconds=1), 1.0, 100.0, False, "test_file", 105.0),  # Second point
        (0, 68.385593, 23.643683, base_time + timedelta(seconds=2), 2.0, 200.0, False, "test_file", 110.0),  # Third point
        (0, 68.385594, 23.643684, base_time + timedelta(seconds=3), 3.0, 300.0, False, "test_file", 115.0),  # Fourth point
    ]
    
    # Test JSON data with elevation enabled
    test_json_data = {
        'statistics_current_elevation': True,
        'statistics_current_speed': True,
        'video_length': 30,
        'video_fps': 30
    }
    
    # Simulate frame generation conditions
    frame_number = 1
    video_length = float(test_json_data.get('video_length', 30))
    video_fps = float(test_json_data.get('video_fps', 30))
    original_route_end_frame = int(video_length * video_fps)
    gpx_time_per_video_time = 1.0
    
    print(f"Frame {frame_number}: video_length={video_length}, video_fps={video_fps}, original_route_end_frame={original_route_end_frame}")
    print(f"Frame {frame_number}: gpx_time_per_video_time={gpx_time_per_video_time}")
    
    # Check if we're in tail-only mode
    is_tail_only = frame_number is not None and original_route_end_frame is not None and frame_number > original_route_end_frame
    print(f"Frame {frame_number}: is_tail_only={is_tail_only}")
    
    # Import the statistics function
    from video_generator_route_statistics import _calculate_video_statistics
    
    # Test with actual video generation parameters
    statistics_data = _calculate_video_statistics(
        test_points, 
        test_json_data, 
        gpx_time_per_video_time,
        frame_number=frame_number,
        original_route_end_frame=original_route_end_frame
    )
    
    print(f"Statistics data: {statistics_data}")
    
    if statistics_data:
        print(f"Current speed: {statistics_data.get('current_speed', 'Not found')}")
        print(f"Current elevation: {statistics_data.get('current_elevation', 'Not found')}")
    
    print("✓ Video generation conditions test completed!")
    return True
