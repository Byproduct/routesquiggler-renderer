#!/usr/bin/env python3
"""
Terminal-specific functions for Route Squiggler render client
"""

import os
import signal
import sys
import threading
import time
import json
import zipfile
import requests
import traceback
from io import BytesIO
from queue import Empty

# Import configuration
from config import config

# Local imports
from write_log import write_log, write_debug_log
from image_generator_multiprocess import StatusUpdate
from image_generator_utils import harmonize_gpx_times
from sync_map_tiles import sync_map_tiles

# Global flags for graceful shutdown
shutdown_requested = False
shutdown_force_count = 0  # Count how many times Ctrl+C was pressed


def sync_map_tiles_terminal():
    """Sync map tiles to storage box."""
    try:
        write_log("Cleaning bad map tiles from local cache.")
        import map_tile_cache_sweep
        map_tile_cache_sweep.main()           
    except Exception as sweep_error:
        write_log(f"Warning: Cache cleaning failed: {sweep_error}")
            
    write_log("Syncing map tiles to storage box.")
    success, uploaded_count, downloaded_count = sync_map_tiles(
        storage_box_address=config.storage_box_address,
        storage_box_user=config.storage_box_user,
        storage_box_password=config.storage_box_password,
        local_cache_dir="map tile cache",
        log_callback=lambda msg: write_debug_log(f"Sync: {msg}"),  
        progress_callback=lambda msg: write_debug_log(f"Progress: {msg}"), 
        sync_state_callback=lambda state: None,  
        max_workers=1,
        dry_run=False,
        upload_only=True
    )
            
    if success:
        write_log(f"Map tile sync completed successfully. Uploaded: {uploaded_count}, Downloaded: {downloaded_count}")
        return success, uploaded_count, downloaded_count
    else:
        write_log(f"Map tile sync failed. Uploaded: {uploaded_count}, Downloaded: {downloaded_count}")
        return False, 0, 0


def monitor_status_queue_terminal(status_queue):
    """Monitor the status queue and print updates to console for terminal mode.
    
    Args:
        status_queue: Multiprocessing queue with StatusUpdate objects
    """
    def monitor_loop():
        """Background loop to monitor status queue."""
        try:
            while True:
                try:
                    # Check for status updates
                    update = status_queue.get(timeout=0.1)
                    if not isinstance(update, StatusUpdate):
                        continue
                    
                    # Format and print the status update
                    if update.error:
                        write_debug_log(f"Zoom {update.zoom_level}: ERROR - {update.error}")
                    else:
                        write_debug_log(f"Zoom {update.zoom_level}: {update.status}")
                        
                except Empty:
                    # No updates available, continue monitoring
                    continue
                except Exception as e:
                    # Queue might be closed, exit loop
                    if "handle is closed" in str(e).lower():
                        break
                    write_debug_log(f"Status monitor error: {str(e)}")
                    break
        except Exception as e:
            write_debug_log(f"Status monitor thread error: {str(e)}")
    
    # Start monitoring in a background thread
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    return monitor_thread


def run_test_image_terminal(bootup_manager, folder_name=None, app=None):
    """Run a test image job in terminal mode.
    
    Args:
        bootup_manager: BootupManager instance with config and credentials
        folder_name: Name of folder to run, or None to use first folder alphabetically
        app: QCoreApplication instance for processing Qt events (optional for backward compatibility)
    """
    import glob
    import json
    import traceback
    from image_generator_main import ImageGeneratorWorker
    from image_generator_utils import load_gpx_files_from_zip
    
    try:
        base_dir = 'test images'
        
        # Check if the base directory exists
        if not os.path.exists(base_dir):
            write_log(f"Directory '{base_dir}' not found.")
            return False
        
        # Find all immediate subdirectories containing data.json
        subfolders = [f for f in glob.glob(os.path.join(base_dir, '*')) 
                     if os.path.isdir(f) and os.path.exists(os.path.join(f, 'data.json'))]

        if not subfolders:
            write_log(f"No test job folders found in '{base_dir}'.")
            return False

        # Sort for deterministic order
        subfolders = sorted(subfolders)
        
        # Select the folder
        selected_folder = None
        if folder_name:
            # Look for the specified folder
            for folder in subfolders:
                if os.path.basename(folder) == folder_name:
                    selected_folder = folder
                    break
            if not selected_folder:
                write_log(f"Test folder '{folder_name}' not found in '{base_dir}'.")
                write_log(f"Available folders: {', '.join([os.path.basename(f) for f in subfolders])}")
                return False
        else:
            # Use first folder alphabetically
            selected_folder = subfolders[0]
        
        write_log(f"Running test image job from folder: {os.path.basename(selected_folder)}")
        
        # Load JSON
        with open(os.path.join(selected_folder, 'data.json'), 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        # Ensure this is treated as an image job
        json_data['job_type'] = json_data.get('job_type', 'image')

        # Load GPX files using shared utility function
        zip_path = os.path.join(selected_folder, 'gpx_files.zip')
        gpx_files_info = load_gpx_files_from_zip(zip_path, log_callback=write_log)

        if not gpx_files_info:
            write_log("No valid GPX files found in the ZIP")
            return False
        
        write_debug_log(f"Loaded {len(gpx_files_info)} GPX file(s)")
        
        # Start the image generation worker with same parameters as GUI version
        worker = ImageGeneratorWorker(
            json_data,
            gpx_files_info,
            bootup_manager.storage_box_credentials,
            bootup_manager.config.api_url,
            bootup_manager.config.user,
            bootup_manager.hardware_id,
            bootup_manager.config.app_version,
            max_workers=config.thread_count  # Use thread count from config/command line
        )
        
        # Connect log signal to write_log for terminal output
        worker.log_message.connect(lambda msg: write_log(msg))
        
        # Connect debug message signal to write_debug_log for terminal output
        if hasattr(worker, 'debug_message'):
            worker.debug_message.connect(lambda msg: write_debug_log(msg))
        
        # Connect status queue monitoring for terminal mode
        def on_status_queue_ready(status_queue):
            """Start monitoring the status queue when it's ready."""
            monitor_status_queue_terminal(status_queue)
        
        worker.status_queue_ready.connect(on_status_queue_ready)
        
        # Connect zoom levels signal (just for info, we don't need to create labels)
        def on_zoom_levels_ready(zoom_levels):
            """Log zoom levels when ready."""
            write_debug_log(f"Processing zoom levels: {zoom_levels}")
        
        worker.zoom_levels_ready.connect(on_zoom_levels_ready)
        
        # Process events to ensure connections are established
        if app:
            app.processEvents()
        
        # Run the worker synchronously (no threading in terminal mode)
        worker.image_generator_process()
        
        # Process any remaining events
        if app:
            app.processEvents()
        
        write_log("Test image job completed successfully.")
        return True
        
    except Exception as e:
        write_log(f"Failed to run test image job: {str(e)}")
        write_log(traceback.format_exc())
        return False


def run_test_video_terminal(bootup_manager, folder_name=None, app=None):
    """Run a test video job in terminal mode.
    
    Args:
        bootup_manager: BootupManager instance with config and credentials
        folder_name: Name of folder to run, or None to use first folder alphabetically
        app: QCoreApplication instance for processing Qt events (optional for backward compatibility)
    """
    import glob
    import json
    import traceback
    from video_generator_main import VideoGeneratorWorker
    from image_generator_utils import load_gpx_files_from_zip
    
    try:
        base_dir = 'test videos'
        
        # Check if the base directory exists
        if not os.path.exists(base_dir):
            write_log(f"Directory '{base_dir}' not found.")
            return False
        
        # Find all immediate subdirectories containing data.json
        subfolders = [f for f in glob.glob(os.path.join(base_dir, '*')) 
                     if os.path.isdir(f) and os.path.exists(os.path.join(f, 'data.json'))]

        if not subfolders:
            write_log(f"No test job folders found in '{base_dir}'.")
            return False

        # Sort for deterministic order
        subfolders = sorted(subfolders)
        
        # Select the folder
        selected_folder = None
        if folder_name:
            # Look for the specified folder
            for folder in subfolders:
                if os.path.basename(folder) == folder_name:
                    selected_folder = folder
                    break
            if not selected_folder:
                write_log(f"Test folder '{folder_name}' not found in '{base_dir}'.")
                write_log(f"Available folders: {', '.join([os.path.basename(f) for f in subfolders])}")
                return False
        else:
            # Use first folder alphabetically
            selected_folder = subfolders[0]
        
        write_log(f"Running test video job from folder: {os.path.basename(selected_folder)}")
        
        # Load JSON
        with open(os.path.join(selected_folder, 'data.json'), 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

        # Ensure this is treated as a video job
        json_data['job_type'] = 'video'

        # Load GPX files using shared utility function
        zip_path = os.path.join(selected_folder, 'gpx_files.zip')
        gpx_files_info = load_gpx_files_from_zip(zip_path, log_callback=write_log)

        if not gpx_files_info:
            write_log("No valid GPX files found in the ZIP")
            return False
        
        write_debug_log(f"Loaded {len(gpx_files_info)} GPX file(s)")
        
        # Start the video generation worker with same parameters as GUI version
        worker = VideoGeneratorWorker(
            json_data,
            gpx_files_info,
            bootup_manager.storage_box_credentials,
            bootup_manager.config.api_url,
            bootup_manager.config.user,
            bootup_manager.hardware_id,
            bootup_manager.config.app_version,
            max_workers=config.thread_count,  # Use thread count from config/command line
            is_test=True,
            gpu_rendering=bootup_manager.config.gpu_rendering
        )
        
        # Connect log signal to write_log for terminal output
        worker.log_message.connect(lambda msg: write_log(msg))
        
        # Connect debug message signal to write_debug_log for terminal output (if available)
        if hasattr(worker, 'debug_message'):
            worker.debug_message.connect(lambda msg: write_debug_log(msg))
        
        # Connect progress updates for terminal mode
        def on_progress_update(progress_bar_name, percentage, progress_text):
            """Log progress updates to console."""
            write_debug_log(f"{progress_bar_name}: {percentage}% - {progress_text}")
        
        worker.progress_update.connect(on_progress_update)
        
        # Process events to ensure connections are established
        if app:
            app.processEvents()
        
        # Run the worker synchronously (no threading in terminal mode)
        worker.video_generator_process()
        
        # Process any remaining events
        if app:
            app.processEvents()
        
        write_log("Test video job completed successfully.")
        return True
        
    except Exception as e:
        write_log(f"Failed to run test video job: {str(e)}")
        write_log(traceback.format_exc())
        return False


def signal_handler(signum, frame):
    """Handle shutdown signals (Ctrl+C)."""
    global shutdown_requested, shutdown_force_count
    
    shutdown_force_count += 1
    
    if shutdown_force_count == 1:
        write_log("\nShutdown requested. Will exit after current operation completes...")
        write_log("(Press Ctrl+C again to force immediate exit)")
        shutdown_requested = True
    elif shutdown_force_count == 2:
        write_log("\nForcing immediate exit...")
        sys.exit(0)
    else:
        # Third time - really force it
        write_log("\nForcing exit NOW!")
        os._exit(1)


def request_job_terminal(api_url, user, hardware_id, app_version):
    """Request a new job from the server (terminal version using requests library).
    
    This function uses threading to make the long-running request interruptible.
    
    Returns:
        tuple: (json_data, gpx_files_info) if job available, or (None, None) if no jobs
    """
    global shutdown_requested
    
    # Container for the result from the thread
    result_container = {'response': None, 'error': None}
    
    def make_request():
        """Function to run in background thread."""
        try:
            # Construct URL
            url = f"{api_url.rstrip('/')}/request_job/"
            headers = {
                'X-API-Key': user,
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            body = {
                'hardware_id': hardware_id,
                'app_version': app_version
            }
            
            write_debug_log(f"Requesting job from server: {url}")
            # Don't log API key
            safe_headers = {k: ('***' if k == 'X-API-Key' else v) for k, v in headers.items()}
            write_debug_log(f"Request headers: {safe_headers}")
            write_debug_log(f"Request body: {body}")
            
            # Make the request with a long timeout (95 seconds like the GUI version)
            write_debug_log("Sending POST request (may take up to 95 seconds)...")
            response = requests.post(url, headers=headers, json=body, timeout=95)
            result_container['response'] = response
            
        except Exception as e:
            result_container['error'] = e
    
    # Start request in background thread
    request_thread = threading.Thread(target=make_request, daemon=True)
    request_thread.start()
    
    # Wait for request to complete, checking shutdown flag periodically
    write_debug_log("Waiting for job request to complete (press Ctrl+C to cancel)...")
    while request_thread.is_alive():
        request_thread.join(timeout=0.5)  # Check every 0.5 seconds
        if shutdown_requested:
            write_log("Shutdown requested while waiting for job request. Request will be abandoned.")
            return None, None
    
    # Check if we got an error
    if result_container['error']:
        e = result_container['error']
        if isinstance(e, requests.Timeout):
            write_log("Job request timed out (no jobs available)")
            return None, None
        elif isinstance(e, requests.RequestException):
            write_log(f"Network error requesting job: {str(e)}")
            write_debug_log(traceback.format_exc())
            return None, None
        else:
            write_log(f"Error requesting job: {str(e)}")
            write_debug_log(traceback.format_exc())
            return None, None
    
    # Get the response
    response = result_container['response']
    if not response:
        write_log("No response received from server")
        return None, None
    
    try:
        write_debug_log(f"Received response with status code: {response.status_code}")
        # Log response headers (just the important ones for debugging)
        write_debug_log(f"Response Content-Type: {response.headers.get('Content-Type', 'not set')}")
        write_debug_log(f"Response Content-Length: {response.headers.get('Content-Length', 'not set')}")
        
        if response.status_code != 200:
            write_log(f"Job request failed with status {response.status_code}: {response.text[:200]}")
            return None, None
        
        # Check content type
        content_type = response.headers.get('Content-Type', '')
        write_debug_log(f"Response content type: {content_type}")
        
        if 'application/json' in content_type.lower():
            # No jobs available
            write_debug_log("Response is JSON, checking status...")
            json_response = response.json()
            write_debug_log(f"JSON response: {json_response}")
            if json_response.get('status') == 'no_job':
                message = json_response.get('message', 'No jobs currently available.')
                write_debug_log(message)
                return None, None
        
        # Process as ZIP data
        write_debug_log(f"Processing job ZIP data (size: {len(response.content)} bytes)...")
        result = process_job_zip_terminal(response.content)
        write_debug_log(f"ZIP processing result: {'Success' if result[0] else 'Failed'}")
        return result
        
    except Exception as e:
        write_log(f"Error processing response: {str(e)}")
        write_debug_log(traceback.format_exc())
        return None, None


def process_job_zip_terminal(zip_data):
    """Process the ZIP file containing job data and GPX files.
    
    Returns:
        tuple: (json_data, gpx_files_info) if successful, or (None, None) on error
    """
    try:
        write_debug_log("Creating ZipFile object from response data...")
        # Process the outer ZIP file which contains data.json and gpx_files.zip
        with zipfile.ZipFile(BytesIO(zip_data), 'r') as outer_zip:
            write_debug_log(f"ZIP file opened successfully. Contents: {outer_zip.namelist()}")
            
            # Read job parameters from data.json
            try:
                write_debug_log("Reading data.json from ZIP...")
                with outer_zip.open('data.json') as data_file:
                    json_data = json.loads(data_file.read().decode('utf-8'))
                write_debug_log(f"data.json parsed successfully. Job ID: {json_data.get('job_id', '?')}, Type: {json_data.get('job_type', 'image')}")
            except Exception as e:
                write_log(f"Error reading data.json: {str(e)}")
                write_debug_log(traceback.format_exc())
                return None, None
            
            # Read the inner gpx_files.zip
            try:
                write_debug_log("Reading gpx_files.zip from outer ZIP...")
                with outer_zip.open('gpx_files.zip') as gpx_zip_file:
                    gpx_zip_data = gpx_zip_file.read()
                write_debug_log(f"gpx_files.zip read successfully (size: {len(gpx_zip_data)} bytes)")
            except Exception as e:
                write_log(f"Error reading gpx_files.zip: {str(e)}")
                write_debug_log(traceback.format_exc())
                return None, None
            
            # Process GPX files from the inner ZIP
            gpx_files_info = []
            try:
                write_debug_log("Opening inner gpx_files.zip...")
                with zipfile.ZipFile(BytesIO(gpx_zip_data), 'r') as gpx_zip:
                    gpx_files_list = gpx_zip.namelist()
                    write_debug_log(f"Processing {len(gpx_files_list)} GPX files: {gpx_files_list}")
                    
                    for file_name in gpx_files_list:
                        write_debug_log(f"Processing GPX file: {file_name}")
                        with gpx_zip.open(file_name) as gpx_file:
                            gpx_content = gpx_file.read()
                            # Try different encodings
                            for encoding in ['utf-8', 'latin1', 'cp1252']:
                                try:
                                    gpx_text = gpx_content.decode(encoding)
                                    if '<gpx' in gpx_text:  # Basic validation
                                        gpx_text = harmonize_gpx_times(gpx_text)
                                        gpx_files_info.append({
                                            'filename': file_name,
                                            'name': file_name,
                                            'content': gpx_text
                                        })
                                        write_debug_log(f"Successfully decoded {file_name} with {encoding} encoding")
                                        break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                write_log(f"Failed to decode {file_name} with any supported encoding")
                                continue
            except Exception as e:
                write_log(f"Error processing inner gpx_files.zip: {str(e)}")
                write_debug_log(traceback.format_exc())
                return None, None
            
            if not gpx_files_info:
                write_log("No valid GPX files found in the ZIP")
                return None, None
            
            write_debug_log(f"Successfully processed {len(gpx_files_info)} GPX files")
            return json_data, gpx_files_info
            
    except zipfile.BadZipFile as e:
        write_log(f"Invalid ZIP file received: {str(e)}")
        write_debug_log(traceback.format_exc())
        return None, None
    except Exception as e:
        write_log(f"Error processing ZIP response: {str(e)}")
        write_debug_log(traceback.format_exc())
        return None, None


def process_job_terminal(json_data, gpx_files_info, bootup_manager, app):
    """Process a job in terminal mode (image or video).
    
    Args:
        json_data: Job data from server
        gpx_files_info: List of GPX file information
        bootup_manager: BootupManager instance
        app: QCoreApplication instance for processing Qt events
    
    Returns:
        bool: True if successful, False otherwise
    """
    global shutdown_requested
    
    try:
        # Check if shutdown was requested before starting
        if shutdown_requested:
            write_log("Shutdown requested before job processing started. Skipping job.")
            return False
        
        job_type = json_data.get('job_type', 'image')
        job_id = json_data.get('job_id', '?')
        
        write_log(f"Processing job #{job_id} (type: {job_type})")
        
        if job_type == 'video':
            # Process video job
            from video_generator_main import VideoGeneratorWorker
            
            worker = VideoGeneratorWorker(
                json_data,
                gpx_files_info,
                bootup_manager.storage_box_credentials,
                bootup_manager.config.api_url,
                bootup_manager.config.user,
                bootup_manager.hardware_id,
                bootup_manager.config.app_version,
                max_workers=config.thread_count,
                is_test=False,
                gpu_rendering=bootup_manager.config.gpu_rendering
            )
            
            # Connect log signals
            worker.log_message.connect(lambda msg: write_log(msg))
            if hasattr(worker, 'debug_message'):
                worker.debug_message.connect(lambda msg: write_debug_log(msg))
            
            # Connect progress updates
            def on_progress_update(progress_bar_name, percentage, progress_text):
                write_debug_log(f"{progress_bar_name}: {percentage}% - {progress_text}")
            worker.progress_update.connect(on_progress_update)
            
            # Process events to ensure connections are established
            app.processEvents()
            
            # Run synchronously
            worker.video_generator_process()
            
        else:
            # Process image job
            from image_generator_main import ImageGeneratorWorker
            
            worker = ImageGeneratorWorker(
                json_data,
                gpx_files_info,
                bootup_manager.storage_box_credentials,
                bootup_manager.config.api_url,
                bootup_manager.config.user,
                bootup_manager.hardware_id,
                bootup_manager.config.app_version,
                max_workers=config.thread_count
            )
            
            # Connect log signals
            worker.log_message.connect(lambda msg: write_log(msg))
            if hasattr(worker, 'debug_message'):
                worker.debug_message.connect(lambda msg: write_debug_log(msg))
            
            # Connect status queue monitoring
            def on_status_queue_ready(status_queue):
                monitor_status_queue_terminal(status_queue)
            worker.status_queue_ready.connect(on_status_queue_ready)
            
            # Connect zoom levels signal
            def on_zoom_levels_ready(zoom_levels):
                write_debug_log(f"Processing zoom levels: {zoom_levels}")
            worker.zoom_levels_ready.connect(on_zoom_levels_ready)
            
            # Process events to ensure connections are established
            app.processEvents()
            
            # Run synchronously
            worker.image_generator_process()
        
        # Process any remaining events
        app.processEvents()
        
        write_log(f"Job #{job_id} completed successfully")
        return True
        
    except Exception as e:
        write_log(f"Error processing job: {str(e)}")
        write_log(traceback.format_exc())
        return False


def run_job_processing_loop_terminal(bootup_manager, app):
    """Main loop for processing jobs from the server in terminal mode.
    
    This function continuously requests and processes jobs until shutdown is requested.
    Press Ctrl+C to gracefully shutdown.
    
    Args:
        bootup_manager: BootupManager instance
        app: QCoreApplication instance for processing Qt events
    """
    global shutdown_requested
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    write_log("Starting job processing loop. Press Ctrl+C to stop.")
    
    retry_delay = 10  # Delay between retries when no jobs are available
    
    while not shutdown_requested:
        try:
            # Process Qt events to keep things responsive
            write_debug_log("Processing Qt events...")
            app.processEvents()
            
            # Request a new job
            write_debug_log("Calling request_job_terminal()...")
            json_data, gpx_files_info = request_job_terminal(
                bootup_manager.config.api_url,
                bootup_manager.config.user,
                bootup_manager.hardware_id,
                bootup_manager.config.app_version
            )
            
            write_debug_log(f"request_job_terminal() returned. json_data={'present' if json_data else 'None'}, gpx_files_info={'present' if gpx_files_info else 'None'}")
            
            if json_data and gpx_files_info:
                # Process the job
                write_debug_log("Job data received, starting processing...")
                success = process_job_terminal(json_data, gpx_files_info, bootup_manager, app)
                write_debug_log(f"Job processing completed with success={success}")
                
                if success:
                    if not shutdown_requested:
                        write_log("Job completed. Requesting next job...")
                        # Small delay before requesting next job
                        time.sleep(0.5)
                        app.processEvents()
                    else:
                        write_log("Job completed. Exiting as shutdown was requested.")
                        break
                else:
                    write_log(f"Job processing failed. Retrying in {retry_delay} seconds...")
                    if not shutdown_requested:
                        time.sleep(retry_delay)
                        app.processEvents()
                    else:
                        write_debug_log("Shutdown requested, skipping retry delay.")
                        break
            else:
                # No jobs available
                if not shutdown_requested:
                    write_debug_log(f"No jobs available. Checking again in {retry_delay} seconds...")
                    # Sleep in small increments to allow quick response to shutdown
                    for i in range(retry_delay * 10):
                        if shutdown_requested:
                            write_debug_log("Shutdown requested during wait period.")
                            break
                        time.sleep(0.1)
                        app.processEvents()
                else:
                    # Shutdown was requested during job request
                    write_debug_log("Shutdown flag detected, exiting loop...")
                    break
        
        except KeyboardInterrupt:
            # Handle Ctrl+C directly for immediate response
            write_log("\nShutdown requested (Ctrl+C). Exiting...")
            break
        
        except Exception as e:
            write_log(f"Error in job processing loop: {str(e)}")
            write_log(traceback.format_exc())
            if not shutdown_requested:
                write_log(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                app.processEvents()
    
    write_log("Job processing loop stopped. Exiting.")
    return True