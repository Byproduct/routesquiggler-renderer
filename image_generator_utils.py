"""
Utility classes and functions for GPX processing and image and also video generation (image-only name is now misleading).
"""

# Standard library imports
import math
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, NamedTuple, Optional, Tuple

# Third-party imports (matplotlib backend must be set before pyplot import)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from image_generator_maptileutils import debug_log


current_directory = os.path.dirname(os.path.abspath(__file__))


# --- GPX time harmonization ---
TIME_NO_MS_RE = re.compile(r'<time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z</time>')
TIME_MS_RE = re.compile(r'(<time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d{3}(Z</time>)')

def harmonize_gpx_times(gpx_text: str) -> str:
    """
    Harmonize GPX time formats by removing milliseconds if present.
    
    If any <time> tag without milliseconds exists, return as-is (fast path).
    Otherwise, if <time> tags with milliseconds exist, strip the .ddd part everywhere.
    
    Args:
        gpx_text: The GPX file content as a string
        
    Returns:
        The GPX text with harmonized time formats
    """
    # Fast path: if the non-ms format exists, keep file untouched
    if TIME_NO_MS_RE.search(gpx_text):
        return gpx_text

    # Otherwise, if ms tags exist, drop the .ddd across the file
    if TIME_MS_RE.search(gpx_text):
        return TIME_MS_RE.sub(r'\1\2', gpx_text)

    # Neither format found: return unchanged
    return gpx_text


# --- Points of Interest ---

def normalize_timestamp(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Normalize a datetime to be timezone-aware (UTC).
    
    This ensures consistent datetime comparisons by converting all timestamps
    to UTC timezone-aware datetimes. Handles:
    - None values (returns None)
    - Timezone-naive datetimes (assumes UTC)
    - Timezone-aware datetimes (converts to UTC)
    
    Args:
        dt: A datetime object (may be naive or aware) or None
    
    Returns:
        A timezone-aware datetime in UTC, or None if input is None
    """
    if dt is None:
        return None
    
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=timezone.utc)
    else:
        # Already timezone-aware - convert to UTC for consistency
        return dt.astimezone(timezone.utc)


class PointOfInterest(NamedTuple):
    """
    Named tuple representing a point of interest (waypoint) from GPX data.
    
    Used for rendering waypoints/POIs on images with optional name labels.
    """
    lat: float                           # Latitude
    lon: float                           # Longitude
    name: str                            # Name of the POI (can be empty string)
    timestamp: Optional[datetime]        # Timestamp (can be None)


def extract_points_of_interest(
    gpx_files_info: List[Dict],
    max_points: Optional[int] = None,
    log_callback=None
) -> List[PointOfInterest]:
    """
    Extract points of interest (waypoints) from GPX files.
    
    Args:
        gpx_files_info: List of dictionaries with 'content' key containing GPX text
        max_points: Maximum number of POIs to extract (None for unlimited)
        log_callback: Optional function to call for logging messages
    
    Returns:
        List of PointOfInterest named tuples
    """
    import gpxpy
    
    points_of_interest = []
    
    for gpx_info in gpx_files_info:
        content = gpx_info.get('content', '')
        if not content:
            continue
        
        try:
            gpx = gpxpy.parse(content)
            
            for waypoint in gpx.waypoints:
                # Extract name (can be None or empty)
                name = waypoint.name if waypoint.name else ''
                
                # Extract and normalize timestamp (can be None)
                # Normalization ensures timezone-aware UTC for consistent comparisons
                timestamp = normalize_timestamp(waypoint.time)
                
                poi = PointOfInterest(
                    lat=waypoint.latitude,
                    lon=waypoint.longitude,
                    name=name,
                    timestamp=timestamp
                )
                points_of_interest.append(poi)
                
                # Check if we've reached the maximum
                if max_points is not None and len(points_of_interest) >= max_points:
                    if log_callback:
                        log_callback(f"Reached maximum of {max_points} points of interest, stopping extraction")
                    return points_of_interest
                    
        except Exception as e:
            if log_callback:
                log_callback(f"Warning: Error extracting waypoints from GPX file: {str(e)}")
            continue
    
    if log_callback and points_of_interest:
        log_callback(f"Extracted {len(points_of_interest)} points of interest from GPX files")
    
    return points_of_interest


def extract_and_store_points_of_interest(
    json_data: Dict,
    gpx_files_info: List[Dict],
    log_callback=None
) -> None:
    """
    Extract points of interest from GPX files and store them in json_data.
    
    This is a convenience function that handles the common pattern of checking
    the points_of_interest setting, extracting POIs, and storing them in json_data.
    
    Args:
        json_data: Job data dictionary (will be modified in place)
        gpx_files_info: List of dictionaries with 'content' key containing GPX text
        log_callback: Optional function to call for logging messages
    """
    points_of_interest_setting = json_data.get('points_of_interest', 'off')
    
    if points_of_interest_setting in ['light', 'dark']:
        max_poi = json_data.get('points_of_interest_max', None)
        # Ensure max_poi is an integer if provided
        if max_poi is not None:
            try:
                max_poi = int(max_poi)
            except (ValueError, TypeError):
                max_poi = None
        
        pois = extract_points_of_interest(
            gpx_files_info,
            max_points=max_poi,
            log_callback=log_callback
        )
        json_data['_points_of_interest_data'] = pois
    else:
        json_data['_points_of_interest_data'] = []


def load_gpx_files_from_zip(zip_path: str, log_callback=None) -> List[Dict]:
    """
    Load and decode GPX files from a zip archive.
    
    Args:
        zip_path: Path to the zip file containing GPX files
        log_callback: Optional callback function for logging messages
        
    Returns:
        List of dictionaries with 'filename', 'name', and 'content' keys
    """
    gpx_files_info = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                with zip_ref.open(file_name) as gpx_file:
                    gpx_content = gpx_file.read()
                    # Try different encodings
                    for encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            gpx_text = gpx_content.decode(encoding)
                            # Basic validation that this is a GPX file
                            if '<gpx' in gpx_text:
                                # Harmonize time formats (remove milliseconds if needed)
                                gpx_text = harmonize_gpx_times(gpx_text)
                                
                                gpx_files_info.append({
                                    'filename': file_name,
                                    'name': file_name,
                                    'content': gpx_text
                                })
                                break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If no encoding worked, log an error
                        if log_callback:
                            log_callback(f"Failed to decode {file_name} with any supported encoding")
    except Exception as e:
        if log_callback:
            log_callback(f"Error loading GPX files from zip: {str(e)}")
        raise
    
    return gpx_files_info


def _is_imperial_units(json_data):
    """Returns true if imperial_units is True in json_data."""
    return json_data and json_data.get('imperial_units', False) is True


def calculate_resolution_scale(resolution_x: int, resolution_y: int) -> float:
    """
    Calculate the resolution scale based on total pixel count.
    
    The scale is determined by the total number of pixels (resolution_x × resolution_y):
    
    < 1 MP   → scale 0.7
    < 8 MP   → scale 1
    8–18 MP  → scale 2
    18–33 MP → scale 3
    ≥ 33 MP  → scale 4
    
    Args:
        resolution_x: Image width in pixels
        resolution_y: Image height in pixels
    
    Returns:
        Resolution scale (0.7, 1, 2, 3, or 4)
    """
    total_pixels = resolution_x * resolution_y
    
    if total_pixels < 1_000_000:
        scale = 0.7
    elif total_pixels < 8_000_000:
        scale = 1.0
    elif total_pixels < 18_000_000:
        scale = 2.0
    elif total_pixels < 33_000_000:
        scale = 3.0
    else:
        scale = 4.0
    
    debug_log(f"calculate_resolution_scale: resolution_x={resolution_x}, resolution_y={resolution_y}, total_pixels={total_pixels:,}, scale={scale}")
    
    return scale


def format_scale_for_label_filename(image_scale: float) -> str:
    """
    Format the image scale for use in label filenames.
    
    Special case: 0.7 becomes "07x" (for hr_based_width_07x.png and speed_based_color_07x.png)
    Other scales: 1.0 -> "1x", 2.0 -> "2x", 3.0 -> "3x", 4.0 -> "4x"
    
    Args:
        image_scale: The resolution scale (0.7, 1.0, 2.0, 3.0, or 4.0)
    
    Returns:
        String representation for filename (e.g., "07x", "1x", "2x", etc.)
    """
    if image_scale == 0.7:
        return "07x"
    else:
        # Convert to int and then to string (1.0 -> "1", 2.0 -> "2", etc.)
        return f"{int(image_scale)}x"


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees
    
    Returns:
        Distance between the two points in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    distance = c * r
    
    return distance


def _extract_heart_rate_from_trkpt(trkpt) -> Optional[int]:
    """
    Extract heart rate value from a GPX trackpoint's extensions.
    
    Supports the common Garmin TrackPointExtension format:
    <extensions>
        <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>76</gpxtpx:hr>
        </gpxtpx:TrackPointExtension>
    </extensions>
    
    Args:
        trkpt: XML element representing a trackpoint
    
    Returns:
        Heart rate as integer, or None if not found
    """
    # Common namespaces for heart rate extensions
    hr_namespaces = [
        'http://www.garmin.com/xmlschemas/TrackPointExtension/v1',
        'http://www.garmin.com/xmlschemas/TrackPointExtension/v2',
    ]
    
    # Try to find extensions element
    extensions = trkpt.find('extensions') or trkpt.find('{http://www.topografix.com/GPX/1/1}extensions')
    if extensions is None:
        return None
    
    # Try each known namespace for heart rate
    for ns in hr_namespaces:
        # Try to find TrackPointExtension
        tpe = extensions.find(f'{{{ns}}}TrackPointExtension')
        if tpe is not None:
            hr_elem = tpe.find(f'{{{ns}}}hr')
            if hr_elem is not None and hr_elem.text:
                try:
                    return int(hr_elem.text)
                except ValueError:
                    pass
    
    # Fallback: search for any element named 'hr' in extensions (handles various formats)
    for elem in extensions.iter():
        if elem.tag.endswith('}hr') or elem.tag == 'hr':
            if elem.text:
                try:
                    return int(elem.text)
                except ValueError:
                    pass
    
    return None


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
    def calculate_statistics_from_gpx_files(gpx_files_info: List[Dict], track_lookup: Dict, json_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Calculate statistics from multiple GPX files including timestamps, distance, speed, and heart rate.
        
        Args:
            gpx_files_info: List of GPX file information dictionaries
            track_lookup: Dictionary mapping filenames to track metadata
            json_data: Optional job configuration to check which statistics are enabled
        
        Returns:
            Dictionary containing calculated statistics or None if calculation fails
        """
        all_timestamps = []
        all_coordinates = []
        total_distance = 0.0
        
        # Check if heart rate extraction is needed (optimization: skip if not requested)
        extract_heart_rate = json_data.get('statistics_average_hr', False) if json_data else False
        all_heart_rates = []
        
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
                                    
                                    # Extract heart rate if enabled
                                    if extract_heart_rate:
                                        hr_value = _extract_heart_rate_from_trkpt(trkpt)
                                        if hr_value is not None:
                                            all_heart_rates.append(hr_value)
                                    
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
                                    
                                    # Extract heart rate if enabled
                                    if extract_heart_rate:
                                        hr_value = _extract_heart_rate_from_trkpt(trkpt)
                                        if hr_value is not None:
                                            all_heart_rates.append(hr_value)
                                    
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
                        
                        # Calculate distance using haversine formula
                        distance = calculate_haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)
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
        
        # Convert distance based on imperial_units setting
        imperial_units = _is_imperial_units(json_data)
        if imperial_units:
            # Convert meters to miles
            distance_value = total_distance * 0.000621371
        else:
            # Convert meters to kilometers
            distance_value = total_distance / 1000.0
        
        # Calculate average speed using the corrected elapsed time
        if total_elapsed_seconds > 0:
            if imperial_units:
                # Distance is in miles, calculate mph
                avg_speed = (distance_value * 3600) / total_elapsed_seconds
            else:
                # Distance is in km, calculate km/h
                avg_speed = (distance_value * 3600) / total_elapsed_seconds
        else:
            avg_speed = 0.0
        
        # Calculate average heart rate if we collected HR data
        if all_heart_rates:
            avg_hr = sum(all_heart_rates) / len(all_heart_rates)
        else:
            avg_hr = 0
        
        return {
            'starting_time': start_time_str,
            'ending_time': end_time_str,
            'elapsed_time': elapsed_time_str,
            'distance': f"{distance_value:.1f}",
            'average_speed': f"{avg_speed:.1f}",
            'average_hr': f"{round(avg_hr)}"
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


def draw_tag(ax, lon: float, lat: float, text: str, text_color_rgb: Tuple[float, float, float], 
             background_theme: str = 'light', resolution_scale: float = 1.0,
             horizontal_offset_points: Optional[float] = None, vertical_offset_points: Optional[float] = None,
             horizontal_offset_coords: Optional[float] = None):
    """
    Draw a tag/label on a matplotlib axes at the specified coordinates.
    
    This function is reusable for both image and video generation modes. It automatically
    handles coordinate projection:
    - For cartopy axes (image generation): Uses lat/lon directly
    - For regular axes in Web Mercator (video generation): Converts lat/lon to Web Mercator
    
    Args:
        ax: Matplotlib axes object (cartopy projection for images, Web Mercator for videos)
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        text: Text to display in the tag
        text_color_rgb: RGB color tuple for text (values 0.0-1.0)
        background_theme: Background theme - 'light' (white bg, colored border/text) 
                         or 'dark' (dark bg, white text, light border)
        resolution_scale: Scale factor for resolution (0.7, 1.0, 2.0, 3.0, or 4.0). Default 1.0
        horizontal_offset_points: Optional horizontal offset in points. If None, uses 6 * resolution_scale
        vertical_offset_points: Optional vertical offset in points (positive = up). If None, uses 0
        horizontal_offset_coords: Optional horizontal offset in coordinate space (for video mode alignment).
                                  If provided in video mode, takes precedence over horizontal_offset_points
    
    Example:
        # Light theme with custom color
        draw_tag(ax, -122.4, 37.8, "Runner Name", (1.0, 0.0, 0.0), 'light')
        
        # Dark theme with 2x resolution
        draw_tag(ax, -122.4, 37.8, "Runner Name", (1.0, 0.0, 0.0), 'dark', resolution_scale=2.0)
    """
    # Convert RGB tuple (0-1 range) to hex for matplotlib
    r, g, b = text_color_rgb
    hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    
    # Set theme colors
    if background_theme == 'dark':
        bg_color = '#2d2d2d'  # Dark gray background
        border_color = '#cccccc'  # Light gray border
        text_color = hex_color  # Use route color for text
    else:  # light theme (default)
        bg_color = 'white'
        border_color = hex_color  # Use text color for border
        text_color = hex_color  # Use text color for text
    
    # Calculate resolution-scaled values
    font_size = 13 * resolution_scale
    border_width = max(1, resolution_scale)  # At least 1px border
    
    # Set defaults if not provided (scale with resolution)
    if horizontal_offset_points is None:
        horizontal_offset_points = 6 * resolution_scale
    if vertical_offset_points is None:
        vertical_offset_points = 0
    
    # Check if axes has a cartopy projection (image generation) or is in Web Mercator (video generation)
    # Cartopy axes have a 'projection' attribute
    has_cartopy_projection = hasattr(ax, 'projection') and ax.projection is not None
    
    if has_cartopy_projection:
        # Image generation: Use lat/lon with PlateCarree transform for Cartopy
        # Cartopy axes need coordinates transformed from PlateCarree (lon/lat) to the axes' native projection
        x, y = lon, lat
        xycoords = ccrs.PlateCarree()._as_mpl_transform(ax)
        # For image mode, always use points-based offset
        use_coord_offset = False
    else:
        # Video generation: Convert to Web Mercator coordinates
        # Web Mercator transformation
        x = lon * 20037508.34 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.34 / 180
        xycoords = 'data'
        # For video mode, use coordinate offset if provided (for alignment with statistics)
        use_coord_offset = horizontal_offset_coords is not None
    
    # Use annotate with appropriate offset system
    if use_coord_offset:
        # Use coordinate-based offset for horizontal alignment with statistics
        tag_x = x + horizontal_offset_coords
        ax.annotate(
            text,
            xy=(tag_x, y),
            xycoords='data',
            xytext=(0, vertical_offset_points),
            textcoords='offset points',
            color=text_color,
            fontsize=font_size,
            fontweight='bold',
            ha='left',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor=bg_color,
                edgecolor=border_color,
                alpha=0.9,
                linewidth=border_width
            ),
            zorder=40,
        )
    else:
        # Use pixel-offset for consistent on-screen spacing
        ax.annotate(
            text,
            xy=(x, y),
            xycoords=xycoords,
            xytext=(horizontal_offset_points, vertical_offset_points),
            textcoords='offset points',
            color=text_color,
            fontsize=font_size,
            fontweight='bold',
            ha='left',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor=bg_color,
                edgecolor=border_color,
                alpha=0.9,
                linewidth=border_width
            ),
            zorder=40,
        )


def add_filename_tags_to_image(ax, track_coords_with_metadata: List[tuple], json_data: Dict, 
                                resolution_scale: float, image_width: int, image_height: int,
                                lon_min: float, lon_max: float, lat_min: float, lat_max: float):
    """
    Add filename tags to an image at the starting point of each route.
    
    Tags are positioned to avoid overlap - if a tag would overlap with an existing tag,
    it is moved downward until it doesn't overlap.
    
    Args:
        ax: Matplotlib axes object (with cartopy projection)
        track_coords_with_metadata: List of (lats, lons, color, name, filename) tuples
        json_data: Job configuration containing filename_tags setting and track_objects
        resolution_scale: Scale factor for resolution (0.7, 1.0, 2.0, 3.0, or 4.0)
        image_width: Image width in pixels
        image_height: Image height in pixels
        lon_min: Minimum longitude of map bounds
        lon_max: Maximum longitude of map bounds
        lat_min: Minimum latitude of map bounds
        lat_max: Maximum latitude of map bounds
    """
    if not track_coords_with_metadata:
        return
    
    # Check filename_tags setting
    filename_tags_setting = json_data.get('filename_tags', 'off') if json_data else 'off'
    if filename_tags_setting not in ['light', 'dark']:
        return
    
    # Calculate coordinate spans for converting offsets to coordinate space
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    
    # Base tag height in pixels (approximate, for overlap detection)
    # Font size 10 at scale 1.0, plus padding
    base_tag_height_pixels = 20 * resolution_scale
    # Convert to latitude units
    tag_height_lat = (base_tag_height_pixels / image_height) * lat_span
    
    # Base horizontal offset in pixels, converted to longitude
    base_offset_pixels = 6 * resolution_scale
    horizontal_offset_lon = (base_offset_pixels / image_width) * lon_span
    
    # Track placed tag positions for overlap detection: list of (lon, lat_top, lat_bottom)
    placed_tags = []
    
    for lats, lons, color, name, filename in track_coords_with_metadata:
        if not lats or not lons or not filename:
            continue
        
        # Get starting point
        start_lat = lats[0]
        start_lon = lons[0]
        
        # Remove .gpx extension if present for display
        display_filename = filename
        if display_filename.endswith('.gpx'):
            display_filename = display_filename[:-4]
        
        # Parse color to RGB tuple (0-1 range)
        if isinstance(color, str):
            # Hex color string
            color_hex = color.lstrip('#')
            if len(color_hex) == 6:
                text_color_rgb = tuple(int(color_hex[i:i+2], 16)/255.0 for i in (0, 2, 4))
            else:
                text_color_rgb = (1.0, 0.0, 0.0)  # Default red
        elif isinstance(color, tuple) and len(color) >= 3:
            text_color_rgb = color[:3]
        else:
            text_color_rgb = (1.0, 0.0, 0.0)  # Default red
        
        # Calculate initial tag position (to the right of start point)
        tag_lon = start_lon + horizontal_offset_lon
        tag_lat = start_lat
        
        # Tag bounds (approximate)
        tag_lat_top = tag_lat + tag_height_lat / 2
        tag_lat_bottom = tag_lat - tag_height_lat / 2
        
        # Check for overlap with existing tags and move down if needed
        # Only check tags that are horizontally close (within a reasonable range)
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            overlap_found = False
            for placed_lon, placed_lat_top, placed_lat_bottom in placed_tags:
                # Check if horizontally close (within ~3x tag width in longitude)
                lon_distance = abs(tag_lon - placed_lon)
                if lon_distance < horizontal_offset_lon * 10:
                    # Check vertical overlap
                    if not (tag_lat_bottom > placed_lat_top or tag_lat_top < placed_lat_bottom):
                        # Overlap detected - move this tag down
                        overlap_found = True
                        # Move down: 0.5 = just touching (half tag height), + 0.05 = small gap
                        # (Original was 0.6 = 0.1 gap, now 0.55 = 0.05 gap, halved)
                        tag_lat = placed_lat_bottom - tag_height_lat * 0.51
                        tag_lat_top = tag_lat + tag_height_lat / 2
                        tag_lat_bottom = tag_lat - tag_height_lat / 2
                        break
            
            if not overlap_found:
                break
            iteration += 1
        
        # Record this tag's position
        placed_tags.append((tag_lon, tag_lat_top, tag_lat_bottom))
        
        # Calculate vertical offset from original position
        vertical_offset_lat = tag_lat - start_lat
        # Convert to points (approximate - using image dimensions)
        vertical_offset_points = (vertical_offset_lat / lat_span) * image_height
        
        # Draw the tag using the reusable function
        draw_tag(
            ax=ax,
            lon=start_lon,
            lat=start_lat,
            text=display_filename,
            text_color_rgb=text_color_rgb,
            background_theme=filename_tags_setting,
            resolution_scale=resolution_scale,
            vertical_offset_points=vertical_offset_points
        )