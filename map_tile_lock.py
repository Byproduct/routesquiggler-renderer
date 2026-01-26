"""
Map tile lock management for the Route Squiggler render client.
This module handles acquiring and releasing locks for map tile services
to prevent multiple users from downloading from the same service simultaneously.
"""

# Standard library imports
import time

# Third-party imports
import requests

# API endpoint for map tile locking
MAP_TILE_LOCK_API_URL = "https://routesquiggler.com/map_tile_lock_api/"

# Timeout settings
LOCK_REQUEST_TIMEOUT = 30  # seconds to wait for lock response
UNLOCK_REQUEST_TIMEOUT = 5  # seconds to wait for unlock response
LOCK_RETRY_INTERVAL = 10  # seconds to wait between retries when locked
MAX_LOCK_WAIT_TIME = 60 * 60  # 60 minutes maximum wait time for lock


def get_service_from_map_style(map_style):
    """
    Get the service name from the map style.
    
    Args:
        map_style (str): The map style from json_data
        
    Returns:
        str: The service name ("osm", "otm", or "stadia")
    """
    if map_style == "osm":
        return "osm"
    elif map_style == "otm":
        return "otm"
    elif map_style in ("stadia_light", "stadia_dark", "stadia_outdoors", "stadia_toner", "stadia_watercolor"):
        return "stadia"
    else:
        # Default to osm for unknown styles
        return "osm"


def acquire_map_tile_lock(json_data, log_callback=None, debug_callback=None):
    """
    Acquire a lock for the map tile service before downloading tiles.
    
    Sends a lock request to the API and waits for a response.
    - If 200 OK or no response after 30 seconds: proceed
    - If 423 Locked: retry every 10 seconds for up to 60 minutes
    
    Args:
        json_data (dict): Job data containing map_style
        log_callback (callable, optional): Function to call for important logging messages
        debug_callback (callable, optional): Function to call for debug logging messages (only shown in debug mode)
        
    Returns:
        tuple: (success: bool, error_message: str or None)
            - (True, None) if lock acquired successfully or timeout (proceed)
            - (False, error_message) if lock could not be acquired after 60 minutes
    """
    map_style = json_data.get('map_style', 'osm')
    service = get_service_from_map_style(map_style)
    lock_message = f"{service} lock"
    
    if debug_callback:
        debug_callback(f"Requesting map tile lock for service: {service}")
    
    start_time = time.time()
    attempt = 0
    
    while True:
        attempt += 1
        elapsed_time = time.time() - start_time
        
        # Check if we've exceeded the maximum wait time
        if elapsed_time > MAX_LOCK_WAIT_TIME:
            error_msg = f"Failed to acquire map tile lock after 60 minutes. Service: {service}"
            if log_callback:
                log_callback(error_msg)
            return (False, error_msg)
        
        try:
            if debug_callback:
                debug_callback(f"Sending lock request (attempt {attempt}): {lock_message}")
            
            response = requests.post(
                MAP_TILE_LOCK_API_URL,
                data=lock_message,
                headers={'Content-Type': 'text/plain'},
                timeout=LOCK_REQUEST_TIMEOUT
            )
            
            if debug_callback:
                debug_callback(f"Lock API response: {response.status_code}")
            
            if response.status_code == 200:
                # Lock acquired successfully
                if debug_callback:
                    debug_callback(f"Map tile lock acquired for service: {service}")
                return (True, None)
            
            elif response.status_code == 423:
                # Locked - another user is downloading
                remaining_time = MAX_LOCK_WAIT_TIME - elapsed_time
                remaining_minutes = int(remaining_time / 60)
                if debug_callback:
                    debug_callback(f"Service {service} is locked by another user. Retrying in {LOCK_RETRY_INTERVAL} seconds (max wait: {remaining_minutes} minutes remaining)")
                time.sleep(LOCK_RETRY_INTERVAL)
                continue
            
            else:
                # Unexpected response - treat as success (proceed)
                if debug_callback:
                    debug_callback(f"Unexpected lock response code: {response.status_code}. Proceeding with map tile download.")
                return (True, None)
                
        except requests.Timeout:
            # No response after 30 seconds - proceed
            if debug_callback:
                debug_callback(f"Lock request timed out after {LOCK_REQUEST_TIMEOUT} seconds. Proceeding with map tile download.")
            return (True, None)
            
        except requests.RequestException as e:
            # Network error - proceed (don't block on API issues)
            if debug_callback:
                debug_callback(f"Lock request failed with error: {str(e)}. Proceeding with map tile download.")
            return (True, None)


def release_map_tile_lock(json_data, log_callback=None, debug_callback=None):
    """
    Release the lock for the map tile service after downloading tiles.
    
    Sends an unlock request to the API. Does not wait more than 5 seconds.
    The response will only be 204 No Content (or no message at all).
    
    Args:
        json_data (dict): Job data containing map_style
        log_callback (callable, optional): Function to call for important logging messages
        debug_callback (callable, optional): Function to call for debug logging messages (only shown in debug mode)
    """
    map_style = json_data.get('map_style', 'osm')
    service = get_service_from_map_style(map_style)
    unlock_message = f"{service} unlock"
    
    if debug_callback:
        debug_callback(f"Releasing map tile lock for service: {service}")
    
    try:
        if debug_callback:
            debug_callback(f"Sending unlock request: {unlock_message}")
        
        response = requests.post(
            MAP_TILE_LOCK_API_URL,
            data=unlock_message,
            headers={'Content-Type': 'text/plain'},
            timeout=UNLOCK_REQUEST_TIMEOUT
        )
        
        if debug_callback:
            debug_callback(f"Unlock API response: {response.status_code}")
            
    except requests.Timeout:
        # Timeout is acceptable - we don't wait for response
        if debug_callback:
            debug_callback(f"Unlock request timed out after {UNLOCK_REQUEST_TIMEOUT} seconds (this is acceptable)")
            
    except requests.RequestException as e:
        # Network error - log but don't fail (lock will expire on server)
        if debug_callback:
            debug_callback(f"Unlock request failed with error: {str(e)} (lock will expire on server)")

