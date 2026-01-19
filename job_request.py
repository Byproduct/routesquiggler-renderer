"""
Job request handling for the Route Squiggler render client.
This module handles requesting and processing new jobs from the server.
"""

import json
import traceback
from io import BytesIO
import zipfile
import requests
from PySide6.QtCore import QObject, Signal, QThread, QTimer, QMetaObject, Qt
from image_generator_utils import harmonize_gpx_times
from network_retry import retry_operation


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
            log_callback(f"vertical_video enabled: Swapped video resolution from {original_x}x{original_y} to {original_y}x{original_x}")
    
    return json_data


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
                'Pragma': 'no-cache'
            }
            body = {
                'hardware_id': hardware_id,
                'app_version': app_version,
                'job_id': str(job_id),
                'status': status
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
        max_attempts=15,
        retry_delay=60,
        log_callback=log_callback,
        operation_name=f"Status update for job {job_id}"
    )

class JobRequestWorker(QObject):
    """Worker class to handle job requests in a separate thread."""
    finished = Signal()
    error = Signal(str)
    job_received = Signal(object, list)  # Emits json_data and gpx_files_info
    log_message = Signal(str)

    def __init__(self, api_url, user, hardware_id, app_version):
        super().__init__()
        self.api_url = api_url
        self.user = user
        self.hardware_id = hardware_id
        self.app_version = app_version

class JobRequestThread(QThread):
    """Thread class to run the job request worker."""
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.worker.moveToThread(self)
        
    def run(self):
        """Execute the worker's request_job method."""
        self.worker.request_job()

class JobRequestManager:
    """Manager class for handling job requests and their lifecycle."""
    
    def __init__(self, main_window):
        """Initialize with reference to the main window."""
        self.main_window = main_window
        self.job_request_worker = None
        self.job_request_thread = None
        self.job_retry_timer = None
        self.current_status = None  # Track current status to avoid redundant updates
    
    def request_new_job(self):
        """Request a new job from the server using simple synchronous approach (was working)."""
        # Show the no_jobs_label when starting a job request
        self.main_window.no_jobs_label.show()
        
        # Use the simple approach that was working - no JobRequestWorker to avoid painter conflicts
        self.request_new_job_simple()
    
    def on_job_request_error(self, error_msg):
        """Handle job request errors."""
        self.main_window.log_widget.add_log(f"=== on_job_request_error called with: {error_msg} ===")
        
        # Check if this is specifically a "no jobs available" situation
        if error_msg == "No jobs available":
            # Keep no_jobs_label visible for "no jobs available"
            self.main_window.log_widget.add_log("No jobs available - will retry in 10 seconds")
            pass
        else:
            # Hide no_jobs_label for other errors
            self.main_window.no_jobs_label.hide()
            self.main_window.log_widget.add_log(f"Other error: {error_msg} - will retry in 10 seconds")
        
        # If we're still in working mode, try requesting another job after a delay
        if self.main_window.pause_button.isEnabled():
            self.main_window.log_widget.add_log("Pause button is enabled - creating retry timer")
            
            # Use QMetaObject.invokeMethod to create timer in main thread
            QMetaObject.invokeMethod(
                self.main_window, 
                "create_job_retry_timer", 
                Qt.ConnectionType.QueuedConnection
            )
        else:
            self.main_window.log_widget.add_log("Pause button is NOT enabled - not creating retry timer")
    
    def on_job_request_finished(self):
        """Handle job request completion (worker finished without error or job received)."""
        self.main_window.log_widget.add_log("=== on_job_request_finished called ===")
        
        # Immediately clean up the worker and thread since we're done with this request
        if self.job_request_thread:
            try:
                # First, disconnect all signals from the worker to prevent lingering connections
                if self.job_request_worker:
                    try:
                        self.job_request_worker.finished.disconnect()
                    except:
                        pass  # Signal might not be connected
                    try:
                        self.job_request_worker.error.disconnect()
                    except:
                        pass  # Signal might not be connected
                    try:
                        self.job_request_worker.job_received.disconnect()
                    except:
                        pass  # Signal might not be connected
                    try:
                        self.job_request_worker.log_message.disconnect()
                    except:
                        pass  # Signal might not be connected
                
                # Clean up the thread and worker
                self.job_request_thread.quit()
                self.job_request_thread.wait(5000)  # Wait up to 5 seconds
                self.job_request_thread.deleteLater()
                self.job_request_thread = None
                self.job_request_worker = None
                self.main_window.log_widget.add_log("Cleaned up job request worker and thread after completion")
            except Exception as e:
                self.main_window.log_widget.add_log(f"Warning: Error cleaning up job request worker: {str(e)}")
                self.job_request_thread = None
                self.job_request_worker = None
    
    def delayed_job_retry(self):
        """Retry requesting a new job after a delay."""
        self.main_window.log_widget.add_log("=== delayed_job_retry called ===")
        self.main_window.log_widget.add_log("Retrying job request.")
        self.request_new_job_qt_network()
    
    def on_job_received(self, json_data, gpx_files_info):
        """Handle received job data."""
        self.main_window.log_widget.add_log("=== on_job_received called ===")
        try:
            # Hide no_jobs_label when actual job data is received
            self.main_window.no_jobs_label.hide()
            
            # Get job type and job ID
            job_type = json_data.get('job_type', 'image')  # Default to 'image' for backward compatibility
            job_id = json_data.get('job_id', '?')
            
            self.main_window.log_widget.add_log(f"Processing job #{job_id} (type: {job_type})")
            
            # CRITICAL: Validate that required attributes are set before processing
            # This prevents race conditions where job is requested before bootup completes
            if not self.main_window.hardware_id:
                error_msg = "Cannot process job: hardware_id not yet available (bootup may not be complete)"
                self.main_window.log_widget.add_log(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
            if not self.main_window.api_url:
                error_msg = "Cannot process job: api_url not yet available (bootup may not be complete)"
                self.main_window.log_widget.add_log(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
            if not self.main_window.user:
                error_msg = "Cannot process job: user/api_key not yet available (bootup may not be complete)"
                self.main_window.log_widget.add_log(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
            if not hasattr(self.main_window, 'bootup_manager') or not self.main_window.bootup_manager:
                error_msg = "Cannot process job: bootup_manager not available"
                self.main_window.log_widget.add_log(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Validate bootup_manager has required credentials
            if not hasattr(self.main_window.bootup_manager, 'storage_box_credentials'):
                error_msg = "Cannot process job: storage_box_credentials not available"
                self.main_window.log_widget.add_log(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Update header label based on job type
            if job_type == 'video':
                self.main_window.header_label.setText(f"Creating video #{job_id}")
                # Show video progress bars for video jobs
                self.main_window.show_video_progress_bars()
            else:
                self.main_window.header_label.setText(f"Creating image #{job_id}")
                # Hide video progress bars for image jobs (they will be hidden in clear_status_labels too)
                self.main_window.hide_video_progress_bars()
            
            # Clear previous status labels
            self.main_window.clear_status_labels()
            
            # Route to appropriate generator based on job type
            if job_type == 'video':
                self.main_window.log_widget.add_log("Creating video generator worker.")
                # Import here to avoid circular import
                from video_generator_main import VideoGeneratorWorker, VideoWorkerThread
                
                # Check if this is a test job (test_job_folders list is populated during test mode)
                is_test_job = bool(self.main_window.test_video_manager.test_job_folders)
                
                # Create video worker
                self.main_window.worker = VideoGeneratorWorker(
                    json_data, 
                    gpx_files_info, 
                    self.main_window.bootup_manager.storage_box_credentials, 
                    self.main_window.api_url, 
                    self.main_window.user, 
                    self.main_window.hardware_id, 
                    self.main_window.app_version, 
                    self.main_window.available_threads,
                    is_test=is_test_job,
                    gpu_rendering=self.main_window.bootup_manager.config.gpu_rendering
                )
                self.main_window.worker_thread = VideoWorkerThread(self.main_window.worker)
            else:
                self.main_window.log_widget.add_log("Creating image generator worker.")
                # Import here to avoid circular import
                from image_generator_main import ImageGeneratorWorker, ImageWorkerThread
                
                # Create image worker
                self.main_window.worker = ImageGeneratorWorker(
                    json_data, 
                    gpx_files_info, 
                    self.main_window.bootup_manager.storage_box_credentials, 
                    self.main_window.api_url, 
                    self.main_window.user, 
                    self.main_window.hardware_id, 
                    self.main_window.app_version, 
                    self.main_window.available_threads
                )
                self.main_window.worker_thread = ImageWorkerThread(self.main_window.worker)
            
            self.main_window.log_widget.add_log("Connecting worker signals.")
            # Connect signals (same for both worker types)
            self.main_window.worker.finished.connect(self.main_window.on_worker_finished)
            self.main_window.worker.error.connect(self.main_window.on_worker_error)
            self.main_window.worker.log_message.connect(self.main_window.log_widget.add_log)
            self.main_window.worker.job_completed.connect(self.main_window.on_job_completed)
            
            # Connect debug message signal for both image and video workers
            if hasattr(self.main_window.worker, 'debug_message'):
                self.main_window.worker.debug_message.connect(self.main_window.log_widget.add_debug_log)
            
            # Connect video-specific signals only for video workers
            if job_type == 'video':
                self.main_window.worker.progress_update.connect(self.main_window.on_video_progress_update)
            
            # Connect image-specific signals only for image workers
            if job_type != 'video':
                self.main_window.worker.status_queue_ready.connect(self.main_window.setup_status_monitoring)
                self.main_window.worker.zoom_levels_ready.connect(self.main_window.create_status_labels)
            
            self.main_window.log_widget.add_log("Starting worker thread...")
            # Start thread
            self.main_window.worker_thread.start()
            
            # CRITICAL: Only update status to "working" AFTER worker thread successfully starts
            # This prevents jobs from being stuck in "working" state if thread fails to start
            # Verify thread is actually running before updating status
            # Note: isRunning() returns True immediately after start() if the thread started successfully
            if not self.main_window.worker_thread.isRunning():
                error_msg = "Worker thread failed to start"
                self.main_window.log_widget.add_log(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
            
            # Now that we've verified the thread is running, update status to "working"
            from update_status import update_status
            update_status(f"working ({job_id})", api_key=self.main_window.user)
            self.current_status = f"working ({job_id})"
            
            self.main_window.log_widget.add_log("Worker thread started successfully")
            
            # Immediately clean up the job request worker and thread since we've successfully started processing
            if self.job_request_thread:
                try:
                    # First, disconnect all signals from the worker to prevent lingering connections
                    if self.job_request_worker:
                        try:
                            self.job_request_worker.finished.disconnect()
                        except:
                            pass  # Signal might not be connected
                        try:
                            self.job_request_worker.error.disconnect()
                        except:
                            pass  # Signal might not be connected
                        try:
                            self.job_request_worker.job_received.disconnect()
                        except:
                            pass  # Signal might not be connected
                        try:
                            self.job_request_worker.log_message.disconnect()
                        except:
                            pass  # Signal might not be connected
                    
                    # Clean up the thread and worker
                    self.job_request_thread.quit()
                    self.job_request_thread.wait(5000)  # Wait up to 5 seconds
                    self.job_request_thread.deleteLater()
                    self.job_request_thread = None
                    self.job_request_worker = None
                    self.main_window.log_widget.add_log("Cleaned up job request worker and thread after successful job start")
                except Exception as e:
                    self.main_window.log_widget.add_log(f"Warning: Error cleaning up job request worker: {str(e)}")
                    self.job_request_thread = None
                    self.job_request_worker = None
            
            self.main_window.log_widget.add_log("=== on_job_received completed successfully ===")
            
        except Exception as e:
            self.main_window.log_widget.add_log(f"Error in on_job_received: {str(e)}")
            import traceback
            self.main_window.log_widget.add_log(traceback.format_exc())
            
            # Extract job_id to report error to server (job was requested but worker never started)
            job_id = None
            try:
                if json_data:
                    job_id = json_data.get('job_id', '?')
            except:
                pass
            
            # Report error to server if we have job_id (job was successfully requested but failed to start)
            if job_id and job_id != '?':
                try:
                    update_job_status(
                        self.main_window.api_url,
                        self.main_window.user,
                        self.main_window.hardware_id,
                        self.main_window.app_version,
                        job_id,
                        'error',
                        self.main_window.log_widget.add_log
                    )
                    self.main_window.log_widget.add_log(f"Reported error to server for job #{job_id} (worker failed to start)")
                except Exception as status_error:
                    self.main_window.log_widget.add_log(f"Warning: Failed to report error to server: {str(status_error)}")
            
            # Clean up the job request worker and thread on error
            if self.job_request_thread:
                try:
                    # First, disconnect all signals from the worker to prevent lingering connections
                    if self.job_request_worker:
                        try:
                            self.job_request_worker.finished.disconnect()
                        except:
                            pass  # Signal might not be connected
                        try:
                            self.job_request_worker.error.disconnect()
                        except:
                            pass  # Signal might not be connected
                        try:
                            self.job_request_worker.job_received.disconnect()
                        except:
                            pass  # Signal might not be connected
                        try:
                            self.job_request_worker.log_message.disconnect()
                        except:
                            pass  # Signal might not be connected
                    
                    # Clean up the thread and worker
                    self.job_request_thread.quit()
                    self.job_request_thread.wait(5000)  # Wait up to 5 seconds
                    self.job_request_thread.deleteLater()
                    self.job_request_thread = None
                    self.job_request_worker = None
                    self.main_window.log_widget.add_log("Cleaned up job request worker and thread after error")
                except Exception as cleanup_error:
                    self.main_window.log_widget.add_log(f"Warning: Error cleaning up job request worker: {str(cleanup_error)}")
                    self.job_request_thread = None
                    self.job_request_worker = None
            
            # Don't re-raise the exception - handle it gracefully
            self.main_window.log_widget.add_log("Job processing failed - will retry later")
            # Trigger error handling to retry the job request
            self.main_window.on_worker_error(f"Job processing failed: {str(e)}")
    
    def cancel_retry_timer(self):
        """Cancel any pending job retry timer."""
        if self.job_retry_timer:
            try:
                # Stop the timer safely
                self.job_retry_timer.stop()
                # Use deleteLater() to ensure deletion happens in the main thread
                self.job_retry_timer.deleteLater()
                self.job_retry_timer = None
            except Exception as e:
                # If there's an error stopping the timer, just log it and continue
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.log_widget.add_log(f"Warning: Error stopping retry timer: {str(e)}")
                self.job_retry_timer = None
    
    def cleanup_job_request_worker(self):
        """Clean up the job request thread (keep the worker for reuse)."""
        if self.job_request_thread:
            try:
                # First, disconnect all signals from the worker to prevent lingering connections
                if self.job_request_worker:
                    try:
                        self.job_request_worker.finished.disconnect()
                    except:
                        pass  # Signal might not be connected
                    try:
                        self.job_request_worker.error.disconnect()
                    except:
                        pass  # Signal might not be connected
                    try:
                        self.job_request_worker.job_received.disconnect()
                    except:
                        pass  # Signal might not be connected
                    try:
                        self.job_request_worker.log_message.disconnect()
                    except:
                        pass  # Signal might not be connected
                
                # Clean up the thread
                self.job_request_thread.quit()
                self.job_request_thread.wait(5000)  # Wait up to 5 seconds
                self.job_request_thread.deleteLater()
                self.job_request_thread = None
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.log_widget.add_log("Cleaned up job request thread (worker kept for reuse)")
            except Exception as e:
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.log_widget.add_log(f"Warning: Error cleaning up job request thread: {str(e)}")
                self.job_request_thread = None
    
    def reset_job_request_worker(self):
        """Completely reset the job request worker (for persistent issues)."""
        if hasattr(self, 'main_window') and self.main_window:
            self.main_window.log_widget.add_log("=== Resetting job request worker completely ===")
        
        # Clean up thread first
        if self.job_request_thread:
            try:
                # Disconnect all signals
                if self.job_request_worker:
                    try:
                        self.job_request_worker.finished.disconnect()
                    except:
                        pass
                    try:
                        self.job_request_worker.error.disconnect()
                    except:
                        pass
                    try:
                        self.job_request_worker.job_received.disconnect()
                    except:
                        pass
                    try:
                        self.job_request_worker.log_message.disconnect()
                    except:
                        pass
                
                # Clean up thread
                self.job_request_thread.quit()
                self.job_request_thread.wait(5000)
                self.job_request_thread.deleteLater()
                self.job_request_thread = None
            except Exception as e:
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.log_widget.add_log(f"Warning: Error cleaning up thread during reset: {str(e)}")
                self.job_request_thread = None
        
        # Clean up worker
        if self.job_request_worker:
            try:
                self.job_request_worker.deleteLater()
                self.job_request_worker = None
            except Exception as e:
                if hasattr(self, 'main_window') and self.main_window:
                    self.main_window.log_widget.add_log(f"Warning: Error cleaning up worker during reset: {str(e)}")
                self.job_request_worker = None
        
        if hasattr(self, 'main_window') and self.main_window:
            self.main_window.log_widget.add_log("Job request worker completely reset") 

    def request_new_job_qt_network(self):
        """Request a new job using QNetworkAccessManager (Qt native networking, non-blocking)."""
        # Show the no_jobs_label when starting a job request
        self.main_window.no_jobs_label.show()
        
        # Show a progress indicator for the long API call
        self.main_window.log_widget.add_log("Starting long API call (may take up to 95 seconds)...")
        
        # Update the play label to show we're working
        self.main_window.play_label.setText("Requesting new job from server...")
        
        # Update status to "idle" if not already idle
        if self.current_status != "idle":
            from update_status import update_status
            update_status("idle", api_key=self.main_window.user)
            self.current_status = "idle"
        
        try:
            from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
            from PySide6.QtCore import QUrl, QByteArray
            import json
            
            # Create network manager (this is safe to reuse)
            if not hasattr(self, 'network_manager'):
                self.network_manager = QNetworkAccessManager()
            
            # Construct URL and headers
            url = f"{self.main_window.api_url.rstrip('/')}/request_job/"
            request = QNetworkRequest(QUrl(url))
            
            # Set headers
            request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
            request.setRawHeader(b"X-API-Key", self.main_window.user.encode())
            request.setRawHeader(b"Cache-Control", b"no-cache")
            request.setRawHeader(b"Pragma", b"no-cache")
            
            # Prepare request body
            body = {
                'hardware_id': self.main_window.hardware_id,
                'app_version': self.main_window.app_version
            }
            body_data = QByteArray(json.dumps(body).encode())
            
            self.main_window.log_widget.add_log(f"Requesting job from server: {url}")
            
            # Make the request (non-blocking)
            reply = self.network_manager.post(request, body_data)
            
            # Connect signals to handle the response
            reply.finished.connect(lambda: self.on_qt_network_response(reply))
            reply.errorOccurred.connect(lambda error: self.on_qt_network_error(reply, error))
            
            # Store the reply to prevent garbage collection
            self.current_network_reply = reply
            
        except Exception as e:
            self.main_window.log_widget.add_log(f"Error starting Qt network request: {str(e)}")
            import traceback
            self.main_window.log_widget.add_log(traceback.format_exc())
            # Reset the play label
            self.main_window.play_label.setText("Working. Press pause to stop.")
    
    def on_qt_network_response(self, reply):
        """Handle the response from Qt network request."""
        try:
            from PySide6.QtNetwork import QNetworkReply
            from PySide6.QtNetwork import QNetworkRequest
            
            if reply.error() == QNetworkReply.NetworkError.NoError:
                status_code = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
                self.main_window.log_widget.add_log(f"Response status: {status_code}")
                
                if status_code == 200:
                    self.main_window.log_widget.add_log("Processing 200 response...")
                    
                    # Get response data
                    response_data = reply.readAll()
                    
                    # Check content type
                    content_type = reply.header(QNetworkRequest.ContentTypeHeader)
                    
                    if content_type and 'application/json' in content_type.lower():
                        self.main_window.log_widget.add_log("Detected JSON response, checking for no_job status...")
                        try:
                            import json
                            json_response = json.loads(response_data.data().decode())
                            if json_response.get('status') == 'no_job':
                                message = json_response.get('message', 'No jobs currently available.')
                                self.main_window.log_widget.add_log(message)
                                self.main_window.play_label.setText("Working. Press pause to stop.")
                                self.on_job_request_error("No jobs available")
                                return
                        except Exception as e:
                            self.main_window.log_widget.add_log(f"Error parsing JSON response: {str(e)}")
                    
                    # Process as ZIP data
                    self.main_window.log_widget.add_log("Processing job ZIP data...")
                    self.process_qt_network_zip_response(response_data.data())
                else:
                    self.main_window.log_widget.add_log(f"Job request failed: {status_code}")
                    self.on_job_request_error(f"Job request failed: {status_code}")
            else:
                error_msg = f"Network error: {reply.errorString()}"
                self.main_window.log_widget.add_log(error_msg)
                self.on_job_request_error(error_msg)
                
        except Exception as e:
            self.main_window.log_widget.add_log(f"Error processing Qt network response: {str(e)}")
            import traceback
            self.main_window.log_widget.add_log(traceback.format_exc())
            self.on_job_request_error(f"Error processing response: {str(e)}")
        finally:
            # Clean up the reply
            reply.deleteLater()
            self.current_network_reply = None
            # Reset the play label
            self.main_window.play_label.setText("Working. Press pause to stop.")
    
    def on_qt_network_error(self, reply, error):
        """Handle network errors from Qt network request."""
        try:
            error_msg = f"Network error: {reply.errorString()}"
            self.main_window.log_widget.add_log(error_msg)
            self.on_job_request_error(error_msg)
        finally:
            # Clean up the reply
            reply.deleteLater()
            self.current_network_reply = None
            # Reset the play label
            self.main_window.play_label.setText("Working. Press pause to stop.")
    
    def process_qt_network_zip_response(self, response_data):
        """Process ZIP response data from Qt network request."""
        job_id = None  # Track job_id so we can report errors if processing fails
        
        try:
            # Process the outer ZIP file which contains data.json and gpx_files.zip
            with zipfile.ZipFile(BytesIO(response_data), 'r') as outer_zip:
                self.main_window.log_widget.add_log("ZIP file opened successfully")
                
                # Read job parameters from data.json
                try:
                    with outer_zip.open('data.json') as data_file:
                        json_data = json.loads(data_file.read().decode('utf-8'))
                    job_id = json_data.get('job_id', '?')
                    self.main_window.log_widget.add_log(f"data.json parsed successfully. Job ID: {job_id}")
                    
                    # Apply vertical_video resolution swap if enabled
                    json_data = apply_vertical_video_swap(json_data, self.main_window.log_widget.add_log)
                except Exception as e:
                    self.main_window.log_widget.add_log(f"Error reading data.json: {str(e)}")
                    # Can't report error to server since we don't have job_id yet
                    self.on_job_request_error("Failed to read data.json")
                    return

                # Read the inner gpx_files.zip
                try:
                    with outer_zip.open('gpx_files.zip') as gpx_zip_file:
                        gpx_zip_data = gpx_zip_file.read()
                    self.main_window.log_widget.add_log("gpx_files.zip read successfully")
                except Exception as e:
                    self.main_window.log_widget.add_log(f"Error reading gpx_files.zip: {str(e)}")
                    # Report error to server since we have job_id
                    if job_id:
                        update_job_status(
                            self.main_window.api_url,
                            self.main_window.user,
                            self.main_window.hardware_id,
                            self.main_window.app_version,
                            job_id,
                            'error',
                            self.main_window.log_widget.add_log
                        )
                        self.main_window.log_widget.add_log(f"Reported error to server for job #{job_id}")
                    self.on_job_request_error("Failed to read gpx_files.zip")
                    return

                # Process GPX files from the inner ZIP
                gpx_files_info = []
                try:
                    with zipfile.ZipFile(BytesIO(gpx_zip_data), 'r') as gpx_zip:
                        self.main_window.log_widget.add_log("Processing GPX files...")
                        for file_name in gpx_zip.namelist():
                            with gpx_zip.open(file_name) as gpx_file:
                                gpx_content = gpx_file.read()
                                # Try different encodings
                                for encoding in ['utf-8', 'latin1', 'cp1252']:
                                    try:
                                        gpx_text = gpx_content.decode(encoding)
                                        if '<gpx' in gpx_text:  # Basic validation that this is a GPX file

                                            gpx_text = harmonize_gpx_times(gpx_text) # Convert format with milliseconds into format without milliseconds

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
                                    self.main_window.log_widget.add_log(f"Failed to decode {file_name} with any supported encoding")
                                    continue
                except Exception as e:
                    self.main_window.log_widget.add_log(f"Error processing GPX files: {str(e)}")
                    # Report error to server since we have job_id
                    if job_id:
                        update_job_status(
                            self.main_window.api_url,
                            self.main_window.user,
                            self.main_window.hardware_id,
                            self.main_window.app_version,
                            job_id,
                            'error',
                            self.main_window.log_widget.add_log
                        )
                        self.main_window.log_widget.add_log(f"Reported error to server for job #{job_id}")
                    self.on_job_request_error("Error processing GPX files")
                    return
                
                if not gpx_files_info:
                    self.main_window.log_widget.add_log("No valid GPX files found in the ZIP")
                    # Report error to server since we have job_id
                    if job_id:
                        update_job_status(
                            self.main_window.api_url,
                            self.main_window.user,
                            self.main_window.hardware_id,
                            self.main_window.app_version,
                            job_id,
                            'error',
                            self.main_window.log_widget.add_log
                        )
                        self.main_window.log_widget.add_log(f"Reported error to server for job #{job_id}")
                    self.on_job_request_error("No valid GPX files found")
                    return
                
                self.main_window.log_widget.add_log(f"Found {len(gpx_files_info)} GPX files, processing job data...")
                # Process the job data
                self.on_job_received(json_data, gpx_files_info)
                
        except zipfile.BadZipFile as e:
            self.main_window.log_widget.add_log(f"Invalid ZIP file received: {str(e)}")
            import traceback
            self.main_window.log_widget.add_log(traceback.format_exc())
            # Can't report error if ZIP is completely invalid (no job_id available)
            self.on_job_request_error(f"Invalid ZIP file received: {str(e)}")
        except Exception as e:
            self.main_window.log_widget.add_log(f"Error processing ZIP response: {str(e)}")
            import traceback
            self.main_window.log_widget.add_log(traceback.format_exc())
            # Report error to server if we have job_id
            if job_id:
                update_job_status(
                    self.main_window.api_url,
                    self.main_window.user,
                    self.main_window.hardware_id,
                    self.main_window.app_version,
                    job_id,
                    'error',
                    self.main_window.log_widget.add_log
                )
                self.main_window.log_widget.add_log(f"Reported error to server for job #{job_id}")
            self.on_job_request_error(f"Error processing ZIP response: {str(e)}") 