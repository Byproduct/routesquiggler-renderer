"""
Bootup functionality for the Route Squiggler Render Client.
This module handles initialization tasks like loading config, getting hardware ID, and setting up system tray.
"""

# Standard library imports
import ftplib
import hashlib
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

# Third-party imports
import requests
from PySide6.QtCore import QObject, QThread, Signal

# Local imports
import map_tile_cache_sweep
from debug_logger import setup_debug_logging
from get_hardware_id import get_hardware_id
from sync_map_tiles import sync_map_tiles
from write_log import write_debug_log, write_log

# Conditionally import SystemTray - it handles its own platform detection
try:
    from tray import SystemTray
    SYSTEM_TRAY_AVAILABLE = True
except ImportError:
    SYSTEM_TRAY_AVAILABLE = False
    SystemTray = None

class BootupManager:
    def __init__(self, main_window, log_callback=None, progress_callback=None):
        self.main_window = main_window
        
        if log_callback:
            # Use provided callback in GUI mode
            self.log_callback = log_callback
        else:
            from write_log import write_log
            self.log_callback = write_log
        
        self.progress_callback = progress_callback or (lambda msg: None) 
        self.collapse_log_callback = None  
        self.config_callback = None
        self.hardware_id_callback = None
        self.system_tray_callback = None
        self.drive_space_warning_callback = None
        
        # Import global config object
        from config import config
        self.config = config
        
        # Setup debug logging if enabled
        from debug_logger import setup_debug_logging
        setup_debug_logging(config.debug_logging)
        
        self.log_callback("Configuration loaded")
        if config.gui:
            self.log_callback("GUI mode enabled")
        else:
            self.log_callback("Terminal-only mode enabled")
        if config.debug_logging:
            self.log_callback("Debug logging enabled")
        if config.gpu_rendering:
            self.log_callback("GPU rendering enabled")
        else:
            self.log_callback("GPU rendering disabled")
        
        # Note: config callback will be called later when BootupWorker sets it up

    @property
    def storage_box_credentials(self):
        """Return storage box credentials as a dictionary."""
        return {
            'address': self.config.storage_box_address,
            'user': self.config.storage_box_user,
            'password': self.config.storage_box_password
        }

    
    def get_and_log_hardware_id(self):
        """Get the hardware ID and store it in a variable, then log it"""
        try:
            self.hardware_id = get_hardware_id()
            self.log_callback(f"Hardware ID: {self.hardware_id}")
            if self.hardware_id_callback:
                self.hardware_id_callback(self.hardware_id)
            return True
        except Exception as e:
            self.hardware_id = None
            self.log_callback(f"Failed to get hardware ID: {str(e)}")
            return False
    
    def display_system_info(self):
        """Display all system and configuration information"""
        self.log_callback(f"App version: {self.config.app_version or 'Not set'}")
        return True
    
    def setup_system_tray(self):
        """Setup system tray functionality using separate SystemTray class"""
        try:
            if not SYSTEM_TRAY_AVAILABLE or SystemTray is None:
                self.log_callback("System tray functionality not available on this platform")
                self.system_tray = None
                return True  # Return True to not break the bootup chain
                
            # Don't create SystemTray in worker thread - signal main thread to create it
            self.log_callback("Setting up system tray")
            if self.system_tray_callback:
                self.system_tray_callback(True)  # Signal that system tray should be created
            return True
        except Exception as e:
            self.log_callback(f"Failed to setup system tray: {str(e)}")
            self.system_tray = None
            return True  # Return True to not break the bootup chain
    
    def make_heartbeat_call(self):
        """Make a heartbeat API call to register with the server"""
        if not self.config.user or not self.hardware_id or not self.config.api_url:
            self.log_callback("Cannot make heartbeat call: missing user, hardware_id, or api_url")
            return False
        
        try:
            # Construct URL from config api_url + heartbeat endpoint
            url = f"{self.config.api_url.rstrip('/')}/heartbeat/"
            headers = {
                'X-API-Key': self.config.user,
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            body = {
                'hardware_id': self.hardware_id,
                'app_version': self.config.app_version
            }
            
            self.log_callback("Testing connection to web server")
            
            response = requests.post(url, headers=headers, json=body, timeout=30)
            
            # Handle different response codes with appropriate logging
            if response.status_code == 200:
                # Success - show the response and collapse the log
                self.log_callback(f"Web server connection successful: {response.text}")
                # Collapse the log since everything is working properly
                if self.collapse_log_callback:
                    self.collapse_log_callback()
                return True
            elif response.status_code == 401:
                # Unauthorized - concise error message, keep log expanded
                self.log_callback(f"Response 401: {response.text}")
            elif response.status_code == 404:
                # Not found - concise error message, keep log expanded
                self.log_callback(f"Response 404: {response.text}")
            else:
                # Unexpected status code - show all debug info, keep log expanded
                self.log_callback(f"Unexpected response status: {response.status_code}")
                self.log_callback(f"URL: {url}")
                self.log_callback(f"Headers: {headers}")
                self.log_callback(f"Body: {body}")
                self.log_callback(f"Response headers: {dict(response.headers)}")
                self.log_callback(f"Response body: {response.text}")
            return False
                
        except requests.exceptions.Timeout:
            self.log_callback("Heartbeat call timed out after 30 seconds")
            return False
        except requests.exceptions.ConnectionError as e:
            self.log_callback(f"Heartbeat call failed: Connection error - {str(e)}")
            return False
        except Exception as e:
            self.log_callback(f"Heartbeat call failed: {str(e)}")
            return False

    def test_storage_box(self):
        """Test connection to storage box by creating a test folder and file"""
        if not all([self.config.storage_box_address, self.config.storage_box_user, self.config.storage_box_password]):
            self.log_callback("Cannot test storage box: missing connection details")
            return False

        ftp = None
        test_folder = None
        try:
            # Connect to FTP
            ftp = ftplib.FTP(self.config.storage_box_address)
            ftp.login(self.config.storage_box_user, self.config.storage_box_password)

            # Create random folder name
            test_folder = str(random.randint(0, 99))
            ftp.mkd(test_folder)
            ftp.cwd(test_folder)

            # Create and upload test file
            test_content = b"This is a test upload file"
            ftp.storbinary('STOR bootup_test.txt', BytesIO(test_content))

            # Read back the file
            received_content = BytesIO()
            ftp.retrbinary('RETR bootup_test.txt', received_content.write)
            
            # Verify content
            if received_content.getvalue() != test_content:
                raise Exception("File content verification failed")

            # Clean up: delete file first, then go back and remove folder
            ftp.delete('bootup_test.txt')  # Delete the test file first
            ftp.cwd('..')  # Go back to parent directory
            ftp.rmd(test_folder)  # Now remove the empty directory
            
            self.log_callback("Storage box connection successful.")
            return True

        except ftplib.error_perm as e:
            self.log_callback(f"Storage box permission error: {str(e)}")
            return False
        except ftplib.error_temp as e:
            self.log_callback(f"Storage box temporary error: {str(e)}")
            return False
        except Exception as e:
            self.log_callback(f"Storage box test failed: {str(e)}")
            return False
        finally:
            if ftp:
                try:
                    # If test folder still exists, try to clean it up
                    if test_folder:
                        try:
                            ftp.delete('logitestupload.txt')  # Try to delete the test file first
                            ftp.cwd('..')  # Then go back to parent directory
                            ftp.rmd(test_folder)  # Finally try to remove the now-empty directory
                        except:
                            pass  # Ignore cleanup errors
                    ftp.quit()
                except:
                    pass  # Ignore disconnection errors

    def do_sync_map_tile_cache(self):
        """Sync map tile cache files between local machine and storage box using the modular syncer."""
        # Check if map tile cache syncing is disabled in config
        if not self.config.sync_map_tile_cache:
            self.log_callback("Map tile cache syncing is disabled in config. Skipping sync.")
            return True  # Return True to not block bootup
        
        if not all([self.config.storage_box_address, self.config.storage_box_user, self.config.storage_box_password]):
            self.log_callback("Cannot sync map tile cache: missing storage box credentials.")
            return False

        try:
            try:
                # Run cache sweep in production mode (actually delete files)
                import sys
                original_argv = sys.argv
                sys.argv = ['map_tile_cache_sweep.py']  # Production mode (no 'test' parameter)
                
                # Capture the main function from our sweep script
                import importlib.util
                spec = importlib.util.spec_from_file_location("map_tile_cache_sweep", "map_tile_cache_sweep.py")
                sweep_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sweep_module)
                
                # Run the sweep
                sweep_module.main()
                
                # Restore original argv
                sys.argv = original_argv
                
                self.log_callback("Cache cleaning completed")
            except Exception as sweep_error:
                self.log_callback(f"Warning: Cache cleaning failed: {str(sweep_error)}")
                # Continue with sync even if sweep fails
            
            # Use the modular sync_map_tiles function
            success, uploaded_count, downloaded_count = sync_map_tiles(
                storage_box_address=self.config.storage_box_address,
                storage_box_user=self.config.storage_box_user,
                storage_box_password=self.config.storage_box_password,
                local_cache_dir="map tile cache",
                log_callback=self.log_callback,
                progress_callback=self.progress_callback,
                sync_state_callback=self.sync_state_callback,
                max_workers=10
            )
            
            return success

        except Exception as e:
            self.log_callback(f"Map tile cache sync failed: {str(e)}")
            return False

    def check_drive_space(self):
        """Check both current and system drives for sufficient free space."""
        try:
            import hard_drive_space_check
                      
            # Check current drive
            current_drive_ok = hard_drive_space_check.check_current_drive()
            if not current_drive_ok:
                self.log_callback("Warning: Current drive has less than 20GB free space. Map tile cache may grow over time and fill the drive completely.")
            
            # Check system drive
            system_drive_ok = hard_drive_space_check.check_system_drive()
            if not system_drive_ok:
                self.log_callback("Warning: System drive has less than 20GB free space.")
            
            # Check if current and system drives are the same
            drives_same = hard_drive_space_check.are_current_and_system_drives_same()
            if drives_same:
                current_drive = hard_drive_space_check.get_current_drive()
                self.log_callback(f"Warning: Application is running on the system drive ({current_drive}).\nIt's recommended to move this program to a storage drive if you have one available.\nThis program doesn't require the system drive, and the map tile cache may become large.")
            
            # If any check fails, emit a warning signal
            if not current_drive_ok or not system_drive_ok or drives_same:
                warning_message = "⚠️"
                warnings = []
                
                if not current_drive_ok:
                    warnings.append("Warning: Current drive has less than 20GB free space. Map tile cache may grow over time and fill the drive completely.")
                if not system_drive_ok:
                    warnings.append("Warning: System drive has less than 20GB free space.")
                if drives_same:
                    warnings.append("Warning: Application is running on the system drive.\nIt's recommended to move this program to a storage drive if you have one available.\nThis program doesn't require the system drive, and the map tile cache may become large.")
                
                if len(warnings) == 1:
                    warning_message += f"{warnings[0]}"
                else:
                    warning_message += f"{'  '.join(warnings)}"
                
                # Emit drive space warning signal
                if self.drive_space_warning_callback:
                    self.drive_space_warning_callback(warning_message)
            
            return True  # Always return True as this is a warning, not a critical failure
            
        except Exception as e:
            self.log_callback(f"Error checking drive space: {str(e)}")
            return True  # Return True to not block bootup on drive space check errors

    def sync_state_callback(self, state):
        """Handle sync state changes (start/complete)."""
        if not self.main_window:
            # Terminal mode - no GUI elements to update
            return
            
        if state == 'start':
            # Trigger the same behavior as pause button - stop requesting jobs
            if hasattr(self.main_window, 'pause_processing'):
                self.main_window.pause_processing()
            # Hide play/pause buttons during sync
            if hasattr(self.main_window, 'play_button'):
                self.main_window.play_button.hide()
            if hasattr(self.main_window, 'pause_button'):
                self.main_window.pause_button.hide()
            if hasattr(self.main_window, 'play_label'):
                self.main_window.play_label.setText("Syncing map tile cache...")
                self.main_window.play_label.show()
        elif state == 'complete':
            # Show play/pause buttons again and restore normal state
            if hasattr(self.main_window, 'play_button'):
                self.main_window.play_button.show()
            if hasattr(self.main_window, 'pause_button'):
                self.main_window.pause_button.show()
            if hasattr(self.main_window, 'play_label'):
                self.main_window.play_label.setText("Processing paused. Press play to continue.")
                self.main_window.play_label.hide()

    def run_bootup_terminal(self):
        """Run bootup sequence directly for terminal mode (no threading)."""
        try:
            write_log("------------------------------------------")
            write_log("Starting up Route Squiggler render client.")
                      
            write_debug_log("Getting hardware ID.")
            if not self.get_and_log_hardware_id():
                write_log("Hardware ID retrieval failed.")
                return False
            
            write_debug_log("Testing web server connection.")
            if not self.make_heartbeat_call():
                write_log("Web server connection failed.")
                return False
            
            # Test storage box connection
            write_debug_log("Testing storage box connection.")
            if not self.test_storage_box():
                write_log("Storage box connection failed.")
                return False
            
            # Sync map tile cache
            write_log("Syncing map tile cache")
            if not self.do_sync_map_tile_cache():
                write_log("Warning: map tile cache sync failed. Continuing regardless.")
            
            write_debug_log("Checking drive space.")
            self.check_drive_space()
            return True
            
        except Exception as e:
            write_log(f"Bootup failed with error: {str(e)}")
            return False


class BootupWorker(QObject):
    """Worker class to handle bootup sequence in a separate thread."""
    step_completed = Signal(str, bool)  # step_name, success
    finished = Signal(bool)  # overall_success
    progress = Signal(str)  # progress message
    log_message = Signal(str)  # Thread-safe logging signal
    collapse_log = Signal()  # Signal to collapse the log
    config_loaded = Signal(str, str, str)  # app_version, api_url, user
    hardware_id_ready = Signal(str)  # hardware_id
    system_tray_ready = Signal(bool)  # Flag to create system tray in main thread
    drive_space_warning = Signal(str)  # Drive space warning message
    
    def __init__(self, bootup_manager):
        super().__init__()
        self.bootup_manager = bootup_manager
        # Set up thread-safe logging callback
        self.bootup_manager.log_callback = self.log_message.emit
        # Set up progress callback
        self.bootup_manager.progress_callback = self.progress.emit
        # Set up collapse log callback
        self.bootup_manager.collapse_log_callback = self.collapse_log.emit
        # Set up config callback
        self.bootup_manager.config_callback = self.config_loaded.emit
        # Set up hardware ID callback
        self.bootup_manager.hardware_id_callback = self.hardware_id_ready.emit
        # Set up system tray callback
        self.bootup_manager.system_tray_callback = self.system_tray_ready.emit
        # Set up drive space warning callback
        self.bootup_manager.drive_space_warning_callback = self.drive_space_warning.emit
        
    def run_bootup(self):
        """Run the complete bootup sequence."""
        # Emit config loaded signal first, now that callbacks are connected
        self.config_loaded.emit(
            self.bootup_manager.config.app_version, 
            self.bootup_manager.config.api_url, 
            self.bootup_manager.config.user
        )
        
        steps = [
            ("Getting hardware ID", self.bootup_manager.get_and_log_hardware_id),
            ("Displaying system info", self.bootup_manager.display_system_info),
            ("Setting up system tray", self.bootup_manager.setup_system_tray),
            ("Testing web server connection", self.bootup_manager.make_heartbeat_call),
            ("Testing storage box connection", self.bootup_manager.test_storage_box),
            ("Syncing map tile cache", self.bootup_manager.do_sync_map_tile_cache),
            ("Checking drive space", self.bootup_manager.check_drive_space)
        ]
        
        overall_success = True
        
        for step_name, step_func in steps:
            self.progress.emit(f"Running: {step_name}...")
            success = step_func()
            self.step_completed.emit(step_name, success)
            
            if not success:
                overall_success = False
                break
        
        self.finished.emit(overall_success)


class BootupThread(QThread):
    """Thread to run the bootup worker."""
    
    def __init__(self, bootup_worker):
        super().__init__()
        self.bootup_worker = bootup_worker
        self.bootup_worker.moveToThread(self)
        
    def run(self):
        """Start the bootup process."""
        self.bootup_worker.run_bootup() 