#!/usr/bin/env python3
"""
Route Squiggler render client - main entry point
"""

import os
import sys
import argparse

# Import configuration
from config import config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Route Squiggler Render Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use GUI setting from config.txt
  python main.py help               # Show this help message
  python main.py gui                # Force GUI mode
  python main.py nogui              # Force terminal-only mode
  python main.py threads 7          # Set thread count to 7
  python main.py debuglog           # Enable debug logging
  python main.py nodebuglog         # Disable debug logging
  python main.py gui threads 4      # Force GUI mode with 4 threads
  python main.py nogui threads 8    # Force terminal mode with 8 threads
  python main.py testimage          # Run test image on first folder (nogui only)
  python main.py testimage 353      # Run test image on folder "353" (nogui only)
  python main.py nogui debuglog testimage  # Multiple arguments combined
        """
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Command arguments: gui, nogui, threads <number>, debuglog, nodebuglog, testimage [folder], help'
    )
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Standard library imports
import multiprocessing as mp
from queue import Empty  

# Calculate default thread count (max-2, minimum 1)
default_threads = max(1, mp.cpu_count() - 2)

def handle_arguments():
    """Handle command line arguments and set configuration."""
    global config
    
    # Initialize with defaults
    gui_override = None
    thread_override = None
    debuglog_override = None
    testimage_folder = None
    
    # Parse arguments
    i = 0
    while i < len(args.args):
        arg = args.args[i]
        
        if arg == 'help':
            # Show help and exit
            import argparse
            parser = argparse.ArgumentParser(
                description='Route Squiggler Render Client',
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Examples:
  python main.py                    # Use GUI setting from config.txt
  python main.py help               # Show this help message
  python main.py gui                # Force GUI mode
  python main.py nogui              # Force terminal-only mode
  python main.py threads 7          # Set thread count to 7
  python main.py debuglog           # Enable debug logging
  python main.py nodebuglog         # Disable debug logging
  python main.py gui threads 4      # Force GUI mode with 4 threads
  python main.py nogui threads 8    # Force terminal mode with 8 threads
  python main.py testimage          # Run test image on first folder (nogui only)
  python main.py testimage 353      # Run test image on folder "353" (nogui only)
  python main.py nogui debuglog testimage  # Multiple arguments combined
                """
            )
            parser.print_help()
            sys.exit(0)
            
        elif arg == 'gui':
            gui_override = True
            
        elif arg == 'nogui':
            gui_override = False
        
        elif arg == 'debuglog':
            debuglog_override = True
        
        elif arg == 'nodebuglog':
            debuglog_override = False
            
        elif arg == 'threads':
            # Next argument should be the thread count
            if i + 1 >= len(args.args):
                print("Error: Thread count value required when using 'threads'")
                print("Usage: python main.py threads <number>")
                sys.exit(1)
            
            try:
                thread_count = int(args.args[i + 1])
                if thread_count < 1:
                    print(f"Error: Thread count must be at least 1, got {thread_count}")
                    sys.exit(1)
                if thread_count > mp.cpu_count():
                    print(f"Warning: Thread count ({thread_count}) exceeds CPU cores ({mp.cpu_count()})")
                thread_override = thread_count
                i += 1  # Skip the next argument since we consumed it
            except ValueError:
                print(f"Error: Invalid thread count '{args.args[i + 1]}'. Must be a number.")
                sys.exit(1)
        
        elif arg == 'testimage':
            # testimage forces nogui mode
            if gui_override is True:
                print("Error: 'testimage' cannot be used with 'gui' mode")
                print("The 'testimage' command is only available in terminal (nogui) mode")
                sys.exit(1)
            gui_override = False
            # Check if next argument is a folder name (not another command)
            if i + 1 < len(args.args) and args.args[i + 1] not in ['gui', 'nogui', 'threads', 'debuglog', 'nodebuglog', 'help', 'testimage']:
                testimage_folder = args.args[i + 1]
                i += 1  # Skip the next argument since we consumed it
            else:
                # Use empty string to indicate "use first folder"
                testimage_folder = ""
                
        else:
            print(f"Error: Unknown argument '{arg}'")
            print("Valid arguments: gui, nogui, threads <number>, debuglog, nodebuglog, testimage [folder], help")
            sys.exit(1)
            
        i += 1
    
    # Apply overrides
    if gui_override is not None:
        config.gui = gui_override
        
    if thread_override is not None:
        config.thread_count = thread_override
    else:
        config.thread_count = default_threads
    
    if debuglog_override is not None:
        config.debug_logging = debuglog_override
    
    return gui_override, thread_override, debuglog_override, testimage_folder

# Handle command line arguments
gui_override, thread_override, debuglog_override, testimage_folder = handle_arguments()  

# Local imports
from write_log import write_log, write_debug_log

# Third-party imports
import matplotlib
matplotlib.use('Agg')

# Local imports
from bootup import BootupManager, BootupThread, BootupWorker
from image_generator_main import ImageGeneratorWorker, ImageWorkerThread
from image_generator_multiprocess import StatusUpdate
from image_generator_test import TestImageManager
from job_request import JobRequestManager
import map_tile_cache_sweep
from sync_map_tiles import sync_map_tiles
from video_generator_test import TestVideoManager
import video_generator_cache_map_tiles

# GUI-specific imports
if config.gui:
    from PySide6.QtWidgets import QApplication
    from main_gui import MainWindow

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
    import threading
    from queue import Empty
    from image_generator_multiprocess import StatusUpdate
    
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

def run_test_image_terminal(bootup_manager, folder_name=None):
    """Run a test image job in terminal mode.
    
    Args:
        bootup_manager: BootupManager instance with config and credentials
        folder_name: Name of folder to run, or None to use first folder alphabetically
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
        
        # Run the worker synchronously (no threading in terminal mode)
        worker.image_generator_process()
        
        write_log("Test image job completed successfully.")
        return True
        
    except Exception as e:
        write_log(f"Failed to run test image job: {str(e)}")
        write_log(traceback.format_exc())
        return False
           
def main():
    # Show which mode is being used
    if gui_override is not None:
        write_log(f"Starting Route Squiggler Render Client in {'GUI' if config.gui else 'terminal-only'} mode (overridden by command line)")
    else:
        write_log(f"Starting Route Squiggler Render Client in {'GUI' if config.gui else 'terminal-only'} mode (from config.txt)")
    
    # Show thread count being used
    if thread_override is not None:
        write_log(f"Using {config.thread_count} threads (overridden by command line)")
    else:
        write_log(f"Using {config.thread_count} threads (default: max-2, min 1)")
    
    # Show debug logging status
    if debuglog_override is not None:
        write_log(f"Debug logging: {'enabled' if config.debug_logging else 'disabled'} (overridden by command line)")
    else:
        write_log(f"Debug logging: {'enabled' if config.debug_logging else 'disabled'} (from config.txt)")
    
    if config.gui:
        # GUI mode
        app = QApplication(sys.argv)       
        app.setApplicationName("Route Squiggler - Render Client")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("Route Squiggler")
        app.setQuitOnLastWindowClosed(True)       
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
    else:
        # Terminal-only mode
        
        # Run bootup sequence
        from bootup import BootupManager
        bootup_manager = BootupManager(None)  # No main_window needed for terminal mode
        bootup_success = bootup_manager.run_bootup_terminal()
        
        if bootup_success:
            write_log("Bootup completed successfully")           
        else:
            write_log("Bootup failed - application may not function properly")
        
        # Check if we should run a test image job
        if testimage_folder is not None:
            write_log("Test image mode requested")
            # Convert empty string to None for "first folder" logic
            folder_arg = testimage_folder if testimage_folder else None
            success = run_test_image_terminal(bootup_manager, folder_arg)
            if success:
                write_log("Test image job completed. Exiting.")
            else:
                write_log("Test image job failed. Exiting.")
        else:
            write_log("Terminal mode completed. Exiting.")


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    main() 