#!/usr/bin/env python3
"""
Status update module for the Route Squiggler render client.

This module provides functionality to update the rendering client's status
on the server without blocking the main program execution.

HOW THE SYSTEM WORKS:
-------------------
The update_status() function launches a completely separate process using
Python's multiprocessing module. This ensures that:

1. The main program continues executing without waiting for the network request
2. Network timeouts or connection issues don't block the main rendering workflow
3. Status updates are fire-and-forget - if they fail, it doesn't affect the client

The separate process:
- Loads the configuration to get the API key (from config.user)
- Sends a POST request to the status API endpoint
- Includes a 30-second timeout to prevent hanging
- Receives the response (to avoid server-side hanging) but doesn't process it
- Terminates automatically when done

Since status updates are non-critical, failures are silently ignored.
"""

# Standard library imports
import json
import multiprocessing
import os
import sys
from datetime import datetime

# Third-party imports
import requests

# Configurable API endpoint URL for status updates
# Change this value if the status API endpoint changes
# Note: Django URLs typically require a trailing slash
STATUS_API_URL = "https://routesquiggler.com/rendering_client_status_api/"


def _update_status_worker(status, api_key, api_url):
    """
    Worker function that runs in a separate process to update the status.
    
    This function is called by the multiprocessing module and should not
    be called directly. Use update_status() instead.
    
    Args:
        status (str): The status string to send to the server
        api_key (str): The API key (from config.user)
        api_url (str): The full URL to the status API endpoint
    """
    log_file = "status_update_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    try:
        # Log the request
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] Starting status update\n")
            f.write(f"  URL: {api_url}\n")
            f.write(f"  Status: {status}\n")
            f.write(f"  API Key: {api_key[:8]} (truncated)\n")
        
        # Prepare the request payload
        payload = {
            'api_key': api_key,
            'status': str(status)
        }
        
        # Send POST request with 30-second timeout
        # We receive the response to avoid server-side hanging, but don't process it
        response = requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        # Log the response
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"  Response Status Code: {response.status_code}\n")
            f.write(f"  Response Headers: {dict(response.headers)}\n")
            try:
                response_text = response.text[:500]  # Limit response text length
                f.write(f"  Response Body: {response_text}\n")
            except Exception:
                f.write(f"  Response Body: (could not read)\n")
            f.write(f"[{timestamp}] Status update completed\n\n")
        
        # Response received - no need to process it further
        # Status updates are non-critical, so we silently ignore any errors
        
    except requests.exceptions.Timeout:
        # Timeout after 30 seconds
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ERROR: Request timeout after 30 seconds\n\n")
    except requests.exceptions.RequestException as e:
        # Any other network error
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ERROR: RequestException - {type(e).__name__}: {str(e)}\n\n")
    except Exception as e:
        # Any other error
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ERROR: Exception - {type(e).__name__}: {str(e)}\n")
            import traceback
            f.write(f"  Traceback: {traceback.format_exc()}\n\n")


def update_status(status, api_key=None):
    """
    Update the rendering client's status on the server.
    
    This function launches a separate process to send the status update,
    so the main program doesn't have to wait for the network connection.
    
    Args:
        status (str): The status string to send to the server
        api_key (str, optional): The API key. If not provided, loads from config.txt
                                 (uses the 'user' field from config)
    
    Example:
        update_status("Rendering job #123")
        update_status("Idle")
    """
    log_file = "status_update_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # # Log the function call
    # try:
    #     with open(log_file, 'a', encoding='utf-8') as f:
    #         f.write(f"[{timestamp}] update_status() called\n")
    #         f.write(f"  Status parameter: {status}\n")
    #         f.write(f"  API key provided: {api_key is not None}\n")
    # except Exception:
    #     pass  # Don't fail if logging fails
    
    # Load config if api_key not provided
    if api_key is None:
        try:
            from config import config
            api_key = config.user
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"  Loaded API key from config: {api_key[:8] if api_key else 'None'}\n")
            except Exception:
                pass
            if not api_key:
                # No API key available - can't update status
                try:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"  ERROR: No API key available - aborting\n\n")
                except Exception:
                    pass
                return
        except Exception as e:
            # Config loading failed - can't update status
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"  ERROR: Config loading failed - {str(e)}\n\n")
            except Exception:
                pass
            return
    
    # Validate that we have required parameters
    if not api_key or not status:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"  ERROR: Missing required parameters (api_key={bool(api_key)}, status={bool(status)})\n\n")
        except Exception:
            pass
        return
    
    # Launch a separate process to send the status update
    # This ensures the main program doesn't block on network I/O
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"  Launching worker process\n")
    except Exception:
        pass
    
    process = multiprocessing.Process(
        target=_update_status_worker,
        args=(status, api_key, STATUS_API_URL)
    )
    process.daemon = True  # Process will terminate when main process exits
    process.start()
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"  Worker process started (PID: {process.pid})\n\n")
    except Exception:
        pass
    
    # Don't wait for the process - let it run independently
    # The process will clean up automatically when done

