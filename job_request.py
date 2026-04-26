"""
Job request handling for the Route Squiggler render client.
HTTP helpers shared by the terminal client and generators (status, confirmation, jobs.db upload).
"""

# Standard library imports
import time

# Wall-clock time of last successful jobs.db upload to the storage box.
_JOBS_DB_LAST_UPLOAD_TIME: float | None = None
_JOBS_DB_UPLOAD_INTERVAL_SEC = 24 * 60 * 60

# Third-party imports
import requests

# Local imports
from network_retry import retry_operation
from write_log import write_debug_log


def upload_jobs_db(log_callback=None):
    """
    Refresh jobs.db on the storage box at most once per 24 hours (in-process timer).
    """
    global _JOBS_DB_LAST_UPLOAD_TIME

    now = time.time()
    if _JOBS_DB_LAST_UPLOAD_TIME is not None:
        if now - _JOBS_DB_LAST_UPLOAD_TIME < _JOBS_DB_UPLOAD_INTERVAL_SEC:
            return

    from jobs_db_upload import upload_jobs_db_to_storage

    if upload_jobs_db_to_storage(log_callback=log_callback):
        _JOBS_DB_LAST_UPLOAD_TIME = time.time()


def set_attribution_from_theme(json_data):
    """
    Set json_data['attribution'] to 'light', 'dark', or keep 'off'. Mutates json_data in place.
    If attribution is already 'off', leave it as is. Otherwise set from theme-like variables.
    Priority: title_text, statistics, legend, name_tags, filename_tags; default 'light'.
    """
    if json_data.get('attribution') == 'off':
        return
    for key in ('title_text', 'statistics', 'legend', 'name_tags', 'filename_tags'):
        val = json_data.get(key)
        if val in ('light', 'dark'):
            json_data['attribution'] = val
            return
    json_data['attribution'] = 'light'


def normalize_video_mode_aliases(json_data, log_callback=None):
    """
    Normalize legacy/alias video_mode values in-place.

    Currently maps deprecated "static" mode to "final" for backward compatibility with older job producers. 
    The deprecated value "static" should no longer be generated anywhere, so this can eventually be removed. 
    """
    raw_video_mode = json_data.get('video_mode')
    if not isinstance(raw_video_mode, str):
        return json_data

    normalized_mode = raw_video_mode.strip().lower()
    if normalized_mode == 'static':
        json_data['video_mode'] = 'final'
        if log_callback:
            log_callback("video_mode 'static' detected, using fallback mode 'final'")

    return json_data


def apply_vertical_video_swap(json_data, log_callback=None):
    """
    Swap video resolution dimensions if vertical_video is True.
    
    Args:
        json_data (dict): The job data dictionary
        log_callback (callable, optional): Function to call for logging messages
        
    Returns:
        dict: The modified json_data with swapped video resolution if applicable
    """
    vertical_video = json_data.get('vertical_video', False)
    
    if not vertical_video:
        return json_data
    
    # Swap video resolution if present
    if 'video_resolution_x' in json_data and 'video_resolution_y' in json_data:
        original_x = json_data['video_resolution_x']
        original_y = json_data['video_resolution_y']
        json_data['video_resolution_x'] = original_y
        json_data['video_resolution_y'] = original_x
        if log_callback:
            log_callback(
                f"vertical_video enabled: Swapped video resolution from {original_x}x{original_y} to {original_y}x{original_x}"
            )
    
    return json_data


def confirm_job_receipt(api_url, user, hardware_id, app_version, job_id, log_callback=None):
    """Confirm job receipt via API call.
    Automatically retries up to 15 times with 60 second delays on network failures.
    
    Args:
        api_url (str): The API base URL
        user (str): The API user/key
        hardware_id (str): The hardware ID
        app_version (str): The application version
        job_id (str): The job ID to confirm
        log_callback (callable, optional): Function to call for logging messages
        
    Returns:
        bool: True if successful, False otherwise
    """

    def _do_confirm():
        """Internal function that performs the actual confirmation."""
        try:
            if not api_url:
                if log_callback:
                    log_callback("Cannot confirm job receipt: missing api_url")
                return False

            # Construct URL from config api_url + confirm_job endpoint
            # The api_url might be just the base domain or might already include /jobs_api/v1/
            # We'll use the full path as specified by the API
            base_url = api_url.rstrip('/')
            # Check if base_url already contains /jobs_api/v1
            if '/jobs_api/v1' in base_url:
                url = f"{base_url}/confirm_job/"
            else:
                url = f"{base_url}/jobs_api/v1/confirm_job/"
            
            # Log URL to debug only (not regular output)
            write_debug_log(f"Confirming job receipt at URL: {url}")
            
            headers = {
                'X-API-Key': user,
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
            }
            # Ensure job_id is an integer (API expects integer, not string)
            try:
                job_id_int = int(job_id) if not isinstance(job_id, int) else job_id
            except (ValueError, TypeError):
                if log_callback:
                    log_callback(f"Warning: job_id '{job_id}' is not a valid integer, sending as-is")
                job_id_int = job_id
            
            body = {'hardware_id': hardware_id, 'app_version': app_version, 'job_id': job_id_int}

            response = requests.post(url, headers=headers, json=body, timeout=10)
            if response.status_code == 200:
                # Log success to debug only (not regular output)
                write_debug_log(f"Job #{job_id} confirmed successfully")
                return True
            else:
                if log_callback:
                    log_callback(f"Failed to confirm job receipt: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            if log_callback:
                log_callback(f"Error confirming job receipt: {str(e)}")
            return False
    
    # Wrap the confirmation operation with retry logic
    return retry_operation(
        operation=_do_confirm,
        max_attempts=5,
        retry_delay=5,
        log_callback=log_callback,
        operation_name=f"Job confirmation for job {job_id}",
    )


def update_job_status(api_url, user, hardware_id, app_version, job_id, status, log_callback=None):
    """Update job status via API call.
    Automatically retries up to 15 times with 60 second delays on network failures.
    
    Args:
        api_url (str): The API base URL
        user (str): The API user/key
        hardware_id (str): The hardware ID
        app_version (str): The application version
        job_id (str): The job ID to update status for
        status (str): The status to set
        log_callback (callable, optional): Function to call for logging messages
        
    Returns:
        bool: True if successful, False otherwise
    """

    def _do_update():
        """Internal function that performs the actual status update."""
        try:
            if not api_url:
                if log_callback:
                    log_callback("Cannot update job status: missing api_url")
                return False

            # Construct URL from config api_url + status endpoint
            url = f"{api_url.rstrip('/')}/status/"
            headers = {
                'X-API-Key': user,
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
            }
            body = {
                'hardware_id': hardware_id,
                'app_version': app_version,
                'job_id': str(job_id),
                'status': status,
            }

            response = requests.post(url, headers=headers, json=body, timeout=10)
            if response.status_code != 200:
                if log_callback:
                    log_callback(f"Failed to update job status: {response.status_code} - {response.text}")
                return False
            return True

        except Exception as e:
            if log_callback:
                log_callback(f"Error updating job status: {str(e)}")
            return False
    
    # Wrap the status update operation with retry logic
    return retry_operation(
        operation=_do_update,
        max_attempts=50,
        retry_delay=20,
        log_callback=log_callback,
        operation_name=f"Status update for job {job_id}",
    )
