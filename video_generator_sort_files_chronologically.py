"""
Video generation chronological file sorting for the Route Squiggler render client.
This module handles sorting GPX files chronologically based on their timestamps.
"""

# Standard library imports
import re
from datetime import datetime, timezone
from pathlib import Path


def sort_gpx_files_chronologically(gpx_files_info, log_callback=None, debug_callback=None):
    """
    Sort GPX files chronologically based on their time tags.
    
    Args:
        gpx_files_info (list): List of GPX file info dictionaries with 'filename', 'name', 'content' keys
        log_callback (callable, optional): Function to call for logging messages
    
    Returns:
        list: List of tuples (datetime, gpx_file_info) sorted chronologically, or None if error
    """
    try:
        # List to store files in chronological order
        chronological_files = []
        
        # Regular expression for time tags - updated to handle milliseconds
        # Matches both formats: YYYY-MM-DDTHH:MM:SSZ and YYYY-MM-DDTHH:MM:SS.000Z
        time_pattern = r'<time>(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z)</time>'
        
        if debug_callback:
            debug_callback(f"Sorting {len(gpx_files_info)} GPX files chronologically...")
        
        for gpx_info in gpx_files_info:
            try:
                filename = gpx_info.get('filename', 'unknown')
                content = gpx_info.get('content', '')
                
                if not content:
                    if log_callback:
                        log_callback(f"Warning: Empty content for {filename}")
                    continue
                
                # Find the first time tag
                match = re.search(time_pattern, content)
                if match:
                    time_str = match.group(1)
                    
                    # Parse the datetime - handle different timestamp formats
                    # All datetimes are normalized to timezone-aware UTC to avoid comparison errors
                    try:
                        if time_str.endswith('Z'):
                            # Handle both with and without milliseconds
                            if '.' in time_str:
                                # Has milliseconds: 2026-01-21T13:45:00.000Z
                                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                            else:
                                # No milliseconds: 2026-01-21T12:15:13Z
                                dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
                                # Make timezone-aware (UTC) to match millisecond format
                                dt = dt.replace(tzinfo=timezone.utc)
                        elif '+' in time_str or '-' in time_str[10:]:  # Check for timezone offset after date
                            # Handle explicit timezone offset
                            dt = datetime.fromisoformat(time_str)
                            # If somehow still naive, assume UTC
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            # No timezone info, assume UTC
                            dt = datetime.fromisoformat(time_str)
                            dt = dt.replace(tzinfo=timezone.utc)
                        
                        # Add to chronological list
                        chronological_files.append((dt, gpx_info))
                        
                        if debug_callback:
                            debug_callback(f"Found timestamp in {filename}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except ValueError as parse_error:
                        if log_callback:
                            log_callback(f"Warning: Could not parse timestamp '{time_str}' in {filename}: {parse_error}")
                        continue
                else:
                    if log_callback:
                        log_callback(f"Warning: No time tag found in {filename}")
                        
            except Exception as e:
                if log_callback:
                    log_callback(f"Error processing file {gpx_info.get('filename', 'unknown')}: {str(e)}")
                continue
        
        if not chronological_files:
            if log_callback:
                log_callback("No valid GPX files with timestamps found.")
            return None
        
        # Sort chronological files by datetime
        chronological_files.sort(key=lambda x: x[0])
        
        # Report the result
        if chronological_files:
            oldest_file = chronological_files[0][1].get('filename', 'unknown')
            newest_file = chronological_files[-1][1].get('filename', 'unknown')
            oldest_time = chronological_files[0][0].strftime('%Y-%m-%d %H:%M:%S')
            newest_time = chronological_files[-1][0].strftime('%Y-%m-%d %H:%M:%S')
            
            if debug_callback:
                debug_callback(f"Chronological sorting completed:")
                debug_callback(f"Oldest route: {oldest_file} ({oldest_time})")
                debug_callback(f"Newest route: {newest_file} ({newest_time})")
                debug_callback(f"Total files sorted: {len(chronological_files)}")
        
        return chronological_files
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in chronological sorting: {str(e)}")
        return None


def get_sorted_gpx_list(gpx_files_info, log_callback=None, debug_callback=None):
    """
    Get a list of GPX file info dictionaries sorted chronologically.
    
    Args:
        gpx_files_info (list): List of GPX file info dictionaries
        log_callback (callable, optional): Function to call for logging messages
    
    Returns:
        list: List of GPX file info dictionaries sorted chronologically, or original list if sorting fails
    """
    chronological_files = sort_gpx_files_chronologically(gpx_files_info, log_callback, debug_callback)
    
    if chronological_files:
        # Extract just the GPX file info from the tuples
        sorted_gpx_files = [gpx_info for datetime_obj, gpx_info in chronological_files]
        return sorted_gpx_files
    else:
        if debug_callback:
            debug_callback("Chronological sorting failed, using original order")
        return gpx_files_info


def get_chronological_summary(gpx_files_info, log_callback=None):
    """
    Get a summary of the chronological arrangement without sorting.
    
    Args:
        gpx_files_info (list): List of GPX file info dictionaries
        log_callback (callable, optional): Function to call for logging messages
    
    Returns:
        dict: Summary with 'oldest_file', 'newest_file', 'oldest_time', 'newest_time', 'total_files'
    """
    chronological_files = sort_gpx_files_chronologically(gpx_files_info, log_callback)
    
    if chronological_files and len(chronological_files) > 0:
        oldest_info = chronological_files[0]
        newest_info = chronological_files[-1]
        
        return {
            'oldest_file': oldest_info[1].get('filename', 'unknown'),
            'newest_file': newest_info[1].get('filename', 'unknown'),
            'oldest_time': oldest_info[0],
            'newest_time': newest_info[0],
            'total_files': len(chronological_files),
            'success': True
        }
    else:
        return {
            'success': False,
            'total_files': len(gpx_files_info) if gpx_files_info else 0
        } 