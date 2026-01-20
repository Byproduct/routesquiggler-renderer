"""
Hard drive space checking utilities for the Route Squiggler render client.
This module provides functions to check available free space on drives.
"""

import os
import shutil
import sys
from pathlib import Path

# Configurable minimum free space threshold (in GB)
MINIMUM_FREE_SPACE_GB = 20

def _is_windows():
    """Check if running on Windows."""
    return sys.platform == 'win32' or os.name == 'nt'

def _get_system_drive_path():
    """
    Get the system drive path based on the platform.
    
    Returns:
        str: System drive path ('C:\\' on Windows, '/' on Linux/Unix)
    """
    if _is_windows():
        # On Windows, use SystemDrive environment variable or default to C:
        system_drive = os.environ.get('SystemDrive', 'C:')
        return system_drive + "\\"
    else:
        # On Linux/Unix, use root filesystem
        return "/"

def check_current_drive():
    """
    Check the current working directory's drive/filesystem for available free space.
    On Windows, checks the drive where the script is running.
    On Linux, checks the filesystem where the script is running.
    
    Returns:
        bool: True if free space is >= MINIMUM_FREE_SPACE_GB, False otherwise
    """
    try:
        # Get the directory where the script is running (not just cwd, but actual script location)
        # This ensures we check the drive where the script is located, not just where it was launched from
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if _is_windows():
            # On Windows, get the drive letter
            drive_path = Path(script_dir).drive
            if drive_path:
                drive_path = drive_path + "\\"
            else:
                # Fallback to current directory drive
                current_dir = os.getcwd()
                drive_path = Path(current_dir).drive
                if drive_path:
                    drive_path = drive_path + "\\"
                else:
                    return False
        else:
            # On Linux/Unix, get the root of the filesystem containing the script
            drive_path = Path(script_dir).root
        
        return check_drive_space(drive_path)
        
    except Exception as e:
        print(f"Error checking current drive space: {e}")
        return False

def check_system_drive():
    """
    Check the system drive (where the OS is installed) for available free space.
    On Windows, checks the system drive (typically C:\).
    On Linux, checks the root filesystem (/).
    
    Returns:
        bool: True if free space is >= MINIMUM_FREE_SPACE_GB, False otherwise
    """
    try:
        drive_path = _get_system_drive_path()
        return check_drive_space(drive_path)
        
    except Exception as e:
        print(f"Error checking system drive space: {e}")
        return False

def check_drive_space(drive_path):
    """
    Check the specified drive/filesystem for available free space.
    
    Args:
        drive_path (str): Path to the drive/filesystem to check (e.g., "C:\\" on Windows, "/" on Linux)
    
    Returns:
        bool: True if free space is >= MINIMUM_FREE_SPACE_GB, False otherwise
    """
    try:
        # Get disk usage statistics
        total, used, free = shutil.disk_usage(drive_path)
        
        # Convert free space from bytes to GB
        free_gb = free / (1024**3)
        
        # Check if free space meets the minimum threshold
        if free_gb >= MINIMUM_FREE_SPACE_GB:
            return True
        else:
            print(f"Warning: Drive {drive_path} has only {free_gb:.2f} GB free space, "
                  f"which is below the minimum threshold of {MINIMUM_FREE_SPACE_GB} GB")
            return False
            
    except Exception as e:
        print(f"Error checking drive space for {drive_path}: {e}")
        return False

def get_free_space_gb(drive_path=None):
    """
    Get the available free space in GB for a specified drive/filesystem or current drive.
    
    Args:
        drive_path (str, optional): Path to the drive/filesystem to check. If None, uses script's drive.
    
    Returns:
        float: Free space in GB, or -1 if error
    """
    try:
        if drive_path is None:
            # Use script's drive/filesystem
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if _is_windows():
                drive_path = Path(script_dir).drive
                if drive_path:
                    drive_path = drive_path + "\\"
                else:
                    # Fallback to current directory
                    current_dir = os.getcwd()
                    drive_path = Path(current_dir).drive
                    if drive_path:
                        drive_path = drive_path + "\\"
                    else:
                        return -1
            else:
                drive_path = Path(script_dir).root
        
        # Get disk usage statistics
        total, used, free = shutil.disk_usage(drive_path)
        
        # Convert free space from bytes to GB
        free_gb = free / (1024**3)
        
        return free_gb
        
    except Exception as e:
        print(f"Error getting free space for {drive_path}: {e}")
        return -1

def are_current_and_system_drives_same():
    """
    Check if the script's drive/filesystem and system drive/filesystem are the same.
    
    Returns:
        bool: True if both are the same, False otherwise
    """
    try:
        # Get script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if _is_windows():
            # On Windows, compare drive letters
            script_drive = Path(script_dir).drive
            system_drive = os.environ.get('SystemDrive', 'C:')
            
            if script_drive and system_drive:
                return script_drive.upper() == system_drive.upper()
            return False
        else:
            # On Linux, check if script is on root filesystem
            # Get the root of the script's filesystem
            script_root = Path(script_dir).root
            system_root = "/"
            
            # If script is on root filesystem, they're the same
            return script_root == system_root
            
    except Exception as e:
        print(f"Error checking if drives are the same: {e}")
        return False

def get_current_drive():
    """
    Get the drive/filesystem identifier where the script is located.
    
    Returns:
        str: Drive letter (e.g., "C:") on Windows, or filesystem path on Linux
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if _is_windows():
            drive = Path(script_dir).drive
            return drive if drive else ""
        else:
            # On Linux, return the root of the filesystem
            return Path(script_dir).root
            
    except Exception as e:
        print(f"Error getting current drive: {e}")
        return ""

def get_system_drive():
    """
    Get the system drive/filesystem identifier.
    
    Returns:
        str: System drive letter (e.g., "C:") on Windows, or "/" on Linux
    """
    try:
        if _is_windows():
            # On Windows, return just the drive letter without backslash
            system_drive = os.environ.get('SystemDrive', 'C:')
            return system_drive
        else:
            # On Linux, return root filesystem
            return "/"
    except Exception as e:
        print(f"Error getting system drive: {e}")
        return ""