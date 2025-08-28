"""
Video generation chronological file sorting for the Route Squiggler render client.
This module handles sorting GPX files chronologically based on their timestamps.
"""

import re
from datetime import datetime
from pathlib import Path


def sort_gpx_files_chronologically(gpx_files_info, log_callback=None):
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
        
        if log_callback:
            log_callback(f"Sorting {len(gpx_files_info)} GPX files chronologically...")
        
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
                    try:
                        if time_str.endswith('Z'):
                            # Handle both with and without milliseconds
                            if '.' in time_str:
                                # Has milliseconds, use fromisoformat
                                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                            else:
                                # No milliseconds, use strptime
                                dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ')
                        elif '+' in time_str:
                            # Handle timezone offset
                            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        else:
                            # Use fromisoformat for other formats
                            dt = datetime.fromisoformat(time_str)
                        
                        # Add to chronological list
                        chronological_files.append((dt, gpx_info))
                        
                        if log_callback:
                            log_callback(f"Found timestamp in {filename}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
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
            
            if log_callback:
                log_callback(f"Chronological sorting completed:")
                log_callback(f"Oldest route: {oldest_file} ({oldest_time})")
                log_callback(f"Newest route: {newest_file} ({newest_time})")
                log_callback(f"Total files sorted: {len(chronological_files)}")
        
        return chronological_files
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in chronological sorting: {str(e)}")
        return None


def get_sorted_gpx_list(gpx_files_info, log_callback=None):
    """
    Get a list of GPX file info dictionaries sorted chronologically.
    
    Args:
        gpx_files_info (list): List of GPX file info dictionaries
        log_callback (callable, optional): Function to call for logging messages
    
    Returns:
        list: List of GPX file info dictionaries sorted chronologically, or original list if sorting fails
    """
    chronological_files = sort_gpx_files_chronologically(gpx_files_info, log_callback)
    
    if chronological_files:
        # Extract just the GPX file info from the tuples
        sorted_gpx_files = [gpx_info for datetime_obj, gpx_info in chronological_files]
        return sorted_gpx_files
    else:
        if log_callback:
            log_callback("Chronological sorting failed, using original order")
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