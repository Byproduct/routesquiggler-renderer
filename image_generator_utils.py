"""
Utility classes and functions for GPX processing and image generation.
This module provides core utilities used by the multiprocess image generation system,
including GPX coordinate extraction, image dimensions calculation, and map figure setup.
"""

from image_generator_maptileutils import debug_log
import numpy as np
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import math

import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server


current_directory = os.path.dirname(os.path.abspath(__file__))

class GPXProcessor:
    """Handles loading and processing of GPX files."""

    @staticmethod
    def extract_coordinates_from_gpx_content(gpx_content: str) -> Tuple[List[float], List[float]]:
        """
        Extract latitude and longitude coordinates from GPX file content.

        Args:
            gpx_content: String content of a GPX file

        Returns:
            Tuple of (latitudes, longitudes) lists
        """
        lats = []
        lons = []

        try:
            # Parse the GPX XML content
            root = ET.fromstring(gpx_content)

            # Handle namespaces in GPX files
            namespaces = {
                'gpx': 'http://www.topografix.com/GPX/1/1',
                'gpx10': 'http://www.topografix.com/GPX/1/0'
            }

            # Try both GPX 1.1 and 1.0 namespaces
            for ns_prefix, ns_uri in namespaces.items():
                # Look for track points
                for trkpt in root.findall(f'.//{{{ns_uri}}}trkpt'):
                    lat = float(trkpt.get('lat'))
                    lon = float(trkpt.get('lon'))
                    lats.append(lat)
                    lons.append(lon)

                # Also check for waypoints if no track points found
                if not lats:
                    for wpt in root.findall(f'.//{{{ns_uri}}}wpt'):
                        lat = float(wpt.get('lat'))
                        lon = float(wpt.get('lon'))
                        lats.append(lat)
                        lons.append(lon)

                # Break if we found coordinates
                if lats:
                    break

            # If no namespace worked, try without namespace
            if not lats:
                for trkpt in root.findall('.//trkpt'):
                    lat = float(trkpt.get('lat'))
                    lon = float(trkpt.get('lon'))
                    lats.append(lat)
                    lons.append(lon)

        except (ET.ParseError, ValueError, AttributeError) as e:
            print(f"Error parsing GPX content: {e}")
            return [], []

        return lats, lons

    @staticmethod
    def calculate_statistics_from_gpx_files(gpx_files_info: List[Dict], track_lookup: Dict) -> Optional[Dict]:
        """
        Calculate statistics from multiple GPX files including timestamps, distance, and speed.
        
        Args:
            gpx_files_info: List of GPX file information dictionaries
            track_lookup: Dictionary mapping filenames to track metadata
        
        Returns:
            Dictionary containing calculated statistics or None if calculation fails
        """
        all_timestamps = []
        all_coordinates = []
        total_distance = 0.0
        
        # Process each GPX file
        for gpx_info in gpx_files_info:
            filename = gpx_info.get('filename', '')
            content = gpx_info.get('content', '')
            
            if not content:
                continue
                
            try:
                # Parse the GPX XML content
                root = ET.fromstring(content)
                
                # Handle namespaces in GPX files
                namespaces = {
                    'gpx': 'http://www.topografix.com/GPX/1/1',
                    'gpx10': 'http://www.topografix.com/GPX/1/0'
                }
                
                file_points = []
                
                # Try both GPX 1.1 and 1.0 namespaces
                for ns_prefix, ns_uri in namespaces.items():
                    # Look for track points with timestamps
                    for trkpt in root.findall(f'.//{{{ns_uri}}}trkpt'):
                        try:
                            lat = float(trkpt.get('lat'))
                            lon = float(trkpt.get('lon'))
                            
                            # Try to find timestamp
                            time_elem = trkpt.find(f'{{{ns_uri}}}time')
                            if time_elem is not None and time_elem.text:
                                # Parse timestamp (ISO 8601 format)
                                timestamp_str = time_elem.text
                                # Handle different timestamp formats
                                try:
                                    if timestamp_str.endswith('Z'):
                                        # Handle both with and without milliseconds
                                        if '.' in timestamp_str:
                                            # Has milliseconds, use fromisoformat
                                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                        else:
                                            # No milliseconds, use strptime
                                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
                                    elif '+' in timestamp_str:
                                        # Handle timezone offset
                                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    else:
                                        # Use fromisoformat for other formats
                                        timestamp = datetime.fromisoformat(timestamp_str)
                                    
                                    file_points.append((lat, lon, timestamp))
                                    all_timestamps.append(timestamp)
                                except ValueError as parse_error:
                                    print(f"Warning: Could not parse timestamp '{timestamp_str}' in {filename}: {parse_error}")
                                    continue
                                
                        except (ValueError, TypeError):
                            continue
                    
                    # Break if we found points with timestamps
                    if file_points:
                        break
                
                # If no namespace worked, try without namespace
                if not file_points:
                    for trkpt in root.findall('.//trkpt'):
                        try:
                            lat = float(trkpt.get('lat'))
                            lon = float(trkpt.get('lon'))
                            
                            # Try to find timestamp
                            time_elem = trkpt.find('time')
                            if time_elem is not None and time_elem.text:
                                timestamp_str = time_elem.text
                                try:
                                    if timestamp_str.endswith('Z'):
                                        # Handle both with and without milliseconds
                                        if '.' in timestamp_str:
                                            # Has milliseconds, use fromisoformat
                                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                        else:
                                            # No milliseconds, use strptime
                                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
                                    elif '+' in timestamp_str:
                                        # Handle timezone offset
                                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    else:
                                        # Use fromisoformat for other formats
                                        timestamp = datetime.fromisoformat(timestamp_str)
                                    
                                    file_points.append((lat, lon, timestamp))
                                    all_timestamps.append(timestamp)
                                except ValueError as parse_error:
                                    print(f"Warning: Could not parse timestamp '{timestamp_str}' in {filename}: {parse_error}")
                                    continue
                                
                        except (ValueError, TypeError):
                            continue
                
                # Calculate distance for this file
                if len(file_points) > 1:
                    file_distance = 0.0
                    for i in range(1, len(file_points)):
                        prev_lat, prev_lon, _ = file_points[i-1]
                        curr_lat, curr_lon, _ = file_points[i]
                        
                        # Haversine formula for distance calculation
                        lat1, lon1, lat2, lon2 = map(math.radians, [prev_lat, prev_lon, curr_lat, curr_lon])
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                        c = 2 * math.asin(math.sqrt(a))
                        # Radius of earth in meters
                        r = 6371000
                        distance = c * r
                        file_distance += distance
                    
                    total_distance += file_distance
                    all_coordinates.extend(file_points)
                    
            except (ET.ParseError, ValueError, AttributeError) as e:
                print(f"Error parsing GPX file {filename} for statistics: {e}")
                continue
        
        # If no timestamps found, return None
        if not all_timestamps:
            return None
        
        # Calculate statistics - FIXED: Sum individual route durations instead of total span
        # Group timestamps by file to calculate individual route durations
        file_timestamps = {}
        for gpx_info in gpx_files_info:
            filename = gpx_info.get('filename', '')
            content = gpx_info.get('content', '')
            
            if not content:
                continue
                
            try:
                # Parse the GPX XML content
                root = ET.fromstring(content)
                
                # Handle namespaces in GPX files
                namespaces = {
                    'gpx': 'http://www.topografix.com/GPX/1/1',
                    'gpx10': 'http://www.topografix.com/GPX/1/0'
                }
                
                file_timestamps_list = []
                
                # Try both GPX 1.1 and 1.0 namespaces
                for ns_prefix, ns_uri in namespaces.items():
                    # Look for track points with timestamps
                    for trkpt in root.findall(f'.//{{{ns_uri}}}trkpt'):
                        try:
                            # Try to find timestamp
                            time_elem = trkpt.find(f'{{{ns_uri}}}time')
                            if time_elem is not None and time_elem.text:
                                # Parse timestamp (ISO 8601 format)
                                timestamp_str = time_elem.text
                                # Handle different timestamp formats
                                try:
                                    if timestamp_str.endswith('Z'):
                                        # Handle both with and without milliseconds
                                        if '.' in timestamp_str:
                                            # Has milliseconds, use fromisoformat
                                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                        else:
                                            # No milliseconds, use strptime
                                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
                                    elif '+' in timestamp_str:
                                        # Handle timezone offset
                                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    else:
                                        # Use fromisoformat for other formats
                                        timestamp = datetime.fromisoformat(timestamp_str)
                                    
                                    file_timestamps_list.append(timestamp)
                                except ValueError as parse_error:
                                    continue
                                
                        except (ValueError, TypeError):
                            continue
                    
                    # Break if we found points with timestamps
                    if file_timestamps_list:
                        break
                
                # If no namespace worked, try without namespace
                if not file_timestamps_list:
                    for trkpt in root.findall('.//trkpt'):
                        try:
                            # Try to find timestamp
                            time_elem = trkpt.find('time')
                            if time_elem is not None and time_elem.text:
                                timestamp_str = time_elem.text
                                try:
                                    if timestamp_str.endswith('Z'):
                                        # Handle both with and without milliseconds
                                        if '.' in timestamp_str:
                                            # Has milliseconds, use fromisoformat
                                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                        else:
                                            # No milliseconds, use strptime
                                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
                                    elif '+' in timestamp_str:
                                        # Handle timezone offset
                                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                    else:
                                        # Use fromisoformat for other formats
                                        timestamp = datetime.fromisoformat(timestamp_str)
                                    
                                    file_timestamps_list.append(timestamp)
                                except ValueError as parse_error:
                                    continue
                                
                        except (ValueError, TypeError):
                            continue
                
                if file_timestamps_list:
                    file_timestamps[filename] = file_timestamps_list
                    
            except (ET.ParseError, ValueError, AttributeError) as e:
                print(f"Error parsing GPX file {filename} for statistics: {e}")
                continue
        
        # Calculate total elapsed time by summing individual route durations
        total_elapsed_seconds = 0.0
        for filename, timestamps in file_timestamps.items():
            if len(timestamps) >= 2:
                route_start = min(timestamps)
                route_end = max(timestamps)
                route_duration = (route_end - route_start).total_seconds()
                total_elapsed_seconds += route_duration
        
        # Get overall start and end times for display purposes
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        
        # Format timestamps
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M')
        
        # Format elapsed time using the corrected total
        hours = int(total_elapsed_seconds // 3600)
        minutes = int((total_elapsed_seconds % 3600) // 60)
        seconds = int(total_elapsed_seconds % 60)
        
        if hours > 0:
            elapsed_time_str = f"{hours}h {minutes}min {seconds}s"
        elif minutes > 0:
            elapsed_time_str = f"{minutes}min {seconds}s"
        else:
            elapsed_time_str = f"{seconds}s"
        
        # Convert distance to kilometers
        distance_km = total_distance / 1000.0
        
        # Calculate average speed (km/h) using the corrected elapsed time
        if total_elapsed_seconds > 0:
            avg_speed_kmh = (distance_km * 3600) / total_elapsed_seconds
        else:
            avg_speed_kmh = 0.0
        
        return {
            'starting_time': start_time_str,
            'ending_time': end_time_str,
            'elapsed_time': elapsed_time_str,
            'distance': f"{distance_km:.1f}",
            'average_speed': f"{avg_speed_kmh:.1f}"
        }


class ImageGenerator:
    """Main class for generating images from GPX data."""

    # Class-level constants
    PROJECTION = ccrs.PlateCarree()

    def calculate_aspect_ratio_bounds(self, map_bounds: Tuple[float, float, float, float],
                                      target_aspect_ratio: float = 16/9) -> Tuple[float, float, float, float]:
        """
        Calculate map bounds that maintain the target aspect ratio.

        Args:
            map_bounds: Tuple (lon_min, lon_max, lat_min, lat_max) in decimal degrees
            target_aspect_ratio: Desired width/height ratio (default 16:9)

        Returns:
            Adjusted bounds (lon_min, lon_max, lat_min, lat_max) with proper aspect ratio
        """
        lon_min, lon_max, lat_min, lat_max = map_bounds

        # Calculate center points
        lon_center = (lon_min + lon_max) / 2
        lat_center = (lat_min + lat_max) / 2

        # Calculate original spans
        lon_span_raw = lon_max - lon_min
        lat_span_raw = lat_max - lat_min

        debug_log(
            f"Original route spans - Longitude: {lon_span_raw:.6f}°, Latitude: {lat_span_raw:.6f}°")
        debug_log(
            f"Route aspect ratio (lon/lat): {lon_span_raw/lat_span_raw:.3f}, Target: {target_aspect_ratio:.3f}")

        # Add base padding to both dimensions
        base_padding = 0.07  # 7% padding
        lon_padding = lon_span_raw * base_padding
        lat_padding = lat_span_raw * base_padding

        # Apply padding to get minimum required spans
        lon_span_padded = lon_span_raw + 2 * lon_padding
        lat_span_padded = lat_span_raw + 2 * lat_padding

        debug_log(
            f"Padded spans - Longitude: {lon_span_padded:.6f}°, Latitude: {lat_span_padded:.6f}°")

        # Account for Mercator projection distortion
        cos_factor = np.cos(np.radians(lat_center))
        debug_log(
            f"Mercator cos factor at {lat_center:.3f}°: {cos_factor:.6f}")

        # Calculate what the spans would be if we use each dimension as the constraining factor
        # Option 1: Use latitude as height constraint
        if_lat_constrains_lon_span = lat_span_padded * target_aspect_ratio / cos_factor

        # Option 2: Use longitude as width constraint
        if_lon_constrains_lat_span = lon_span_padded * cos_factor / target_aspect_ratio

        debug_log(
            f"If latitude constrains: longitude span = {if_lat_constrains_lon_span:.6f}°")
        debug_log(
            f"If longitude constrains: latitude span = {if_lon_constrains_lat_span:.6f}°")

        # Choose the option that results in the smaller total area (more efficient use of space)
        # while still containing all the padded route data
        if if_lat_constrains_lon_span >= lon_span_padded and if_lon_constrains_lat_span >= lat_span_padded:
            # Both options work, choose the one with smaller total area
            area_if_lat_constrains = if_lat_constrains_lon_span * lat_span_padded
            area_if_lon_constrains = lon_span_padded * if_lon_constrains_lat_span

            debug_log(
                f"Both constraints work - Lat constraint area: {area_if_lat_constrains:.6f}, Lon constraint area: {area_if_lon_constrains:.6f}")

            if area_if_lat_constrains <= area_if_lon_constrains:
                # Use latitude as constraint
                final_lat_span = lat_span_padded
                final_lon_span = if_lat_constrains_lon_span
                debug_log("Using latitude as constraining dimension")
            else:
                # Use longitude as constraint
                final_lat_span = if_lon_constrains_lat_span
                final_lon_span = lon_span_padded
                debug_log("Using longitude as constraining dimension")
        elif if_lat_constrains_lon_span >= lon_span_padded:
            # Only latitude constraint works
            final_lat_span = lat_span_padded
            final_lon_span = if_lat_constrains_lon_span
            debug_log("Only latitude constraint works")
        elif if_lon_constrains_lat_span >= lat_span_padded:
            # Only longitude constraint works
            final_lat_span = if_lon_constrains_lat_span
            final_lon_span = lon_span_padded
            debug_log("Only longitude constraint works")
        else:
            # Neither constraint works perfectly (shouldn't happen with reasonable data)
            # Fall back to latitude constraint
            final_lat_span = lat_span_padded
            final_lon_span = if_lat_constrains_lon_span
            debug_log(
                "Neither constraint works perfectly, falling back to latitude constraint")

        debug_log(
            f"Final spans - Longitude: {final_lon_span:.6f}°, Latitude: {final_lat_span:.6f}°")
        debug_log(
            f"Final aspect ratio: {(final_lon_span * cos_factor) / final_lat_span:.3f}")

        # Calculate final bounds centered around the original data
        lat_min_padded = lat_center - final_lat_span / 2
        lat_max_padded = lat_center + final_lat_span / 2
        lon_min_padded = lon_center - final_lon_span / 2
        lon_max_padded = lon_center + final_lon_span / 2

        return (lon_min_padded, lon_max_padded, lat_min_padded, lat_max_padded)

    def setup_map_figure(self, map_tiles, extents: Tuple[float, float, float, float],
                         zoom_level: int, alpha: float = 0.5, figsize: Tuple[float, float] = (9.6, 5.4)):
        """
        Set up a matplotlib figure with map tiles.

        Args:
            map_tiles: CartoPy image tiles object
            extents: Tuple (lon_min, lon_max, lat_min, lat_max)
            zoom_level: Integer zoom level for map tiles
            alpha: Opacity of map tiles (0.0-1.0)
            figsize: Tuple (width, height) in inches for the figure size

        Returns:
            tuple: (fig, ax) matplotlib figure and axis objects
        """
        # Create the figure with the specified dimensions at 100 DPI
        fig = plt.figure(figsize=figsize, dpi=100)

        # Create subplot that fills the entire figure (no margins)
        ax = fig.add_subplot(1, 1, 1, projection=map_tiles.crs)

        # Remove all margins and padding to ensure exact pixel dimensions
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_position([0, 0, 1, 1])  # Fill entire figure area

        # Unpack extents
        lon_min, lon_max, lat_min, lat_max = extents

        # Set the map extent explicitly
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())

        # Force the aspect ratio and turn off axis
        ax.set_aspect('auto')
        ax.axis('off')  # Remove axis labels, ticks, and spines

        # Set background color to white
        ax.set_facecolor('white')

        # Add the map tiles with the specified opacity
        ax.add_image(map_tiles, zoom_level, alpha=alpha)

        return fig, ax