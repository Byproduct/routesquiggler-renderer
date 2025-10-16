#!/usr/bin/env python3
"""
Appends program messages to daily log files.
"""

import os
from datetime import datetime


def write_log(message):
    """
    Write a message to both console and daily log file.
    
    Args:
        message (str): The message to log
    """
    print(message)
    
    # Write to daily log file
    try:
        log_dir = "log"       
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{today}.txt")
        
        # Append message to log file with timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
    except Exception as e:
        print(f"Error writing to log file: {e}")


def write_debug_log(message):
    """
    Write a debug message if debug logging is enabled.
    """
    from config import config
    if config.debug_logging:
        write_log(f"Debug: {message}")