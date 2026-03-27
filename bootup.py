"""
Bootup functionality for the Route Squiggler Render Client.
Handles initialization tasks like loading config, hardware ID, server and storage checks, and clock cache.
"""

# Standard library imports
import ftplib
import os
import random
from io import BytesIO

# Third-party imports
import requests

# Local imports
from debug_logger import setup_debug_logging
from get_hardware_id import get_hardware_id
from utils.clock_generator import EXPECTED_CLOCK_FILE_COUNT, generate_all_clocks
from write_log import write_debug_log, write_log


class BootupManager:
    def __init__(self, main_window, log_callback=None):
        self.main_window = main_window

        if log_callback:
            self.log_callback = log_callback
        else:
            from write_log import write_log
            self.log_callback = write_log

        self.hardware_id_callback = None
        self.drive_space_warning_callback = None
        
        # Import global config object
        from config import config
        self.config = config
        
        # Setup debug logging if enabled
        from debug_logger import setup_debug_logging
        setup_debug_logging(config.debug_logging)
        
        self.log_callback("Configuration loaded")
        self.log_callback("Terminal mode")
        if config.debug_logging:
            self.log_callback("Debug logging enabled")
        if config.gpu_rendering:
            self.log_callback("GPU rendering enabled")
        else:
            self.log_callback("GPU rendering disabled")

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

    def _log_update_required_if_version_mismatch(self, response_text: str) -> None:
        if "App version mismatch" not in response_text:
            return
        self.log_callback("")
        self.log_callback("================")
        self.log_callback("Update required!")
        self.log_callback("================")
        self.log_callback("")
    
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
                
                # Update status to "starting up" immediately after successful server connection
                from update_status import update_status
                update_status("starting up", api_key=self.config.user)
                return True
            elif response.status_code == 401:
                # Unauthorized - concise error message, keep log expanded
                self.log_callback(f"Response 401: {response.text}")
                self._log_update_required_if_version_mismatch(response.text)
            elif response.status_code == 404:
                # Not found - concise error message, keep log expanded
                self.log_callback(f"Response 404: {response.text}")
                self._log_update_required_if_version_mismatch(response.text)
            elif response.status_code == 502:
                self.log_callback("Heartbeat failed with status 502 - service down")
                write_debug_log(
                    f"502 response body (first 500 chars): {response.text[:500]}"
                )
                self._log_update_required_if_version_mismatch(response.text)
            else:
                # Unexpected status code - show all debug info, keep log expanded
                self.log_callback(f"Unexpected response status: {response.status_code}")
                self.log_callback(f"URL: {url}")
                self.log_callback(f"Headers: {headers}")
                self.log_callback(f"Body: {body}")
                self.log_callback(f"Response headers: {dict(response.headers)}")
                self.log_callback(f"Response body: {response.text}")
                self._log_update_required_if_version_mismatch(response.text)
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

    def do_generate_clock_cache(self):
        """Generate clock .npy files into img/clock if not already complete. Does not block bootup on failure."""
        try:
            self.log_callback("Checking clock images")
            ran = generate_all_clocks()
            if ran:
                self.log_callback("Clock cache generated.")
            else:
                write_debug_log(f"Clock cache already complete ({EXPECTED_CLOCK_FILE_COUNT} files); skipped.")
            return True
        except Exception as e:
            self.log_callback(f"Clock generation failed: {str(e)}. Continuing bootup.")
            return True  # Do not block bootup

    def do_enforce_cache_size_limit(self):
        """Delete oldest map-tile cache files if the cache exceeds the configured size limit."""
        try:
            from utils.cache_size_limit import enforce_cache_size_limit
            return enforce_cache_size_limit(self.config, self.log_callback)
        except Exception as e:
            self.log_callback(f"Cache size limit check failed: {str(e)}. Continuing bootup.")
            return True

    def do_sync_map_tile_cache(self):
        """Legacy stub — map tile syncing is now handled per-job by map_tile_caching.py."""
        return True

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
            
            # Generate clock cache (skips if img/clock already has 8640 files)
            write_debug_log("Generating clock images.")
            self.do_generate_clock_cache()
            
            write_debug_log("Checking drive space.")
            self.check_drive_space()
            
            write_debug_log("Enforcing cache size limit.")
            self.do_enforce_cache_size_limit()
            return True
            
        except Exception as e:
            write_log(f"Bootup failed with error: {str(e)}")
            return False
