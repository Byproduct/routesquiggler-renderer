"""
Hard drive space checking utilities for the Route Squiggler render client.
This module provides functions to check available free space on drives.
"""

import os
import shutil
from pathlib import Path

# Configurable minimum free space threshold (in GB)
MINIMUM_FREE_SPACE_GB = 20

def check_current_drive():
    """
    Check the current working directory's drive for available free space.
    
    Returns:
        bool: True if free space is >= MINIMUM_FREE_SPACE_GB, False otherwise
    """
    try:
        # Get the current working directory
        current_dir = os.getcwd()
        drive_path = Path(current_dir).drive
        
        if not drive_path:
            # If no drive letter (e.g., on Linux), use the root of current directory
            drive_path = Path(current_dir).root
        else:
            # Add backslash for Windows drive paths
            drive_path = drive_path + "\\"
        
        return check_drive_space(drive_path)
        
    except Exception as e:
        print(f"Error checking current drive space: {e}")
        return False

def check_system_drive():
    """
    Check the system drive (where the OS is installed) for available free space.
    
    Returns:
        bool: True if free space is >= MINIMUM_FREE_SPACE_GB, False otherwise
    """
    try:
        # Get the system drive (where the OS is installed)
        system_drive = os.environ.get('SystemDrive', 'C:')
        drive_path = system_drive + "\\"
        
        return check_drive_space(drive_path)
        
    except Exception as e:
        print(f"Error checking system drive space: {e}")
        return False

def check_drive_space(drive_path):
    """
    Check the specified drive for available free space.
    
    Args:
        drive_path (str): Path to the drive to check (e.g., "C:\\")
    
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
    Get the available free space in GB for a specified drive or current drive.
    
    Args:
        drive_path (str, optional): Path to the drive to check. If None, uses current drive.
    
    Returns:
        float: Free space in GB, or -1 if error
    """
    try:
        if drive_path is None:
            # Use current drive
            current_dir = os.getcwd()
            drive_path = Path(current_dir).drive
            if not drive_path:
                drive_path = Path(current_dir).root
            else:
                drive_path = drive_path + "\\"
        
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
    Check if the current working directory and system drive are the same.
    
    Returns:
        bool: True if both drives are the same, False otherwise
    """
    try:
        # Get current working directory drive
        current_dir = os.getcwd()
        current_drive = Path(current_dir).drive
        
        # Get system drive
        system_drive = os.environ.get('SystemDrive', 'C:')
        
        # Compare drives (case-insensitive for Windows)
        if current_drive and system_drive:
            return current_drive.upper() == system_drive.upper()
        else:
            # If we can't determine drives (e.g., on Linux), assume they're different
            return False
            
    except Exception as e:
        print(f"Error checking if drives are the same: {e}")
        return False

def get_current_drive():
    """
    Get the drive letter of the current working directory.
    
    Returns:
        str: Drive letter (e.g., "C:") or empty string if not available
    """
    try:
        current_dir = os.getcwd()
        drive = Path(current_dir).drive
        return drive if drive else ""
    except Exception as e:
        print(f"Error getting current drive: {e}")
        return ""

def get_system_drive():
    """
    Get the system drive letter.
    
    Returns:
        str: System drive letter (e.g., "C:") or empty string if not available
    """
    try:
        system_drive = os.environ.get('SystemDrive', 'C:')
        return system_drive if system_drive else ""
    except Exception as e:
        print(f"Error getting system drive: {e}")
        return ""