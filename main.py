#!/usr/bin/env python3
"""
Route Squiggler render client - main entry point
"""

import os
import sys
import argparse

# Import configuration
from config import config

# Shared help text to avoid duplication
HELP_TEXT = """
Argument examples:
  python main.py                            # Use settings in config.txt
  python main.py gui / nogui                # Override GUI or terminal-only mode
  python main.py debuglog / nodebuglog      # Override verbose logging on/off
  python main.py threads 7                  # Override thread count (default max-2) 
  python main.py testimage                  # Run test image on first folder (terminal only)
  python main.py testimage 353              # Run test image on folder "353" (nogui only)
  python main.py testvideo                  # Run test video on first folder (nogui only)
  python main.py testvideo 351              # Run test video on folder "351" (nogui only)
  python main.py nogui debuglog testimage   # Multiple arguments can be combined
"""

# Standard library imports
import multiprocessing as mp
from queue import Empty  

# Calculate default thread count (max-2, minimum 1)
default_threads = max(1, mp.cpu_count() - 2)

def parse_and_handle_arguments():
    """Parse command line arguments and apply configuration changes."""
    global config
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Route Squiggler Render Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog=HELP_TEXT
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Command arguments: gui, nogui, threads <number>, debuglog, nodebuglog, testimage [folder], testvideo [folder], help'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize with defaults
    gui_override = None
    thread_override = None
    debuglog_override = None
    testimage_folder = None
    testvideo_folder = None
    
    # Process arguments
    i = 0
    while i < len(args.args):
        arg = args.args[i]
        
        if arg == 'help':
            # Show help and exit
            print(HELP_TEXT.strip())
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
            if i + 1 < len(args.args) and args.args[i + 1] not in ['gui', 'nogui', 'threads', 'debuglog', 'nodebuglog', 'help', 'testimage', 'testvideo']:
                testimage_folder = args.args[i + 1]
                i += 1  # Skip the next argument since we consumed it
            else:
                # Use empty string to indicate "use first folder"
                testimage_folder = ""
        
        elif arg == 'testvideo':
            # testvideo forces nogui mode
            if gui_override is True:
                print("Error: 'testvideo' cannot be used with 'gui' mode")
                print("The 'testvideo' command is only available in terminal (nogui) mode")
                sys.exit(1)
            gui_override = False
            # Check if next argument is a folder name (not another command)
            if i + 1 < len(args.args) and args.args[i + 1] not in ['gui', 'nogui', 'threads', 'debuglog', 'nodebuglog', 'help', 'testimage', 'testvideo']:
                testvideo_folder = args.args[i + 1]
                i += 1  # Skip the next argument since we consumed it
            else:
                # Use empty string to indicate "use first folder"
                testvideo_folder = ""
                
        else:
            print(f"Error: Unknown argument '{arg}'")
            print("Valid arguments: gui, nogui, threads <number>, debuglog, nodebuglog, testimage [folder], testvideo [folder], help")
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
    
    return gui_override, thread_override, debuglog_override, testimage_folder, testvideo_folder

# Parse and handle command line arguments
gui_override, thread_override, debuglog_override, testimage_folder, testvideo_folder = parse_and_handle_arguments()  

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
from main_terminal import sync_map_tiles_terminal, run_test_image_terminal, run_test_video_terminal, run_job_processing_loop_terminal

# GUI-specific imports
if config.gui:
    from PySide6.QtWidgets import QApplication
    from main_gui import MainWindow
else:
    # Terminal mode needs QCoreApplication for Qt event loop (workers use Qt signals)
    from PySide6.QtCore import QCoreApplication


           
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
        # Create QCoreApplication to provide Qt event loop for worker signals
        app = QCoreApplication(sys.argv)
        
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
            success = run_test_image_terminal(bootup_manager, folder_arg, app)
            if success:
                write_log("Test image job completed. Exiting.")
            else:
                write_log("Test image job failed. Exiting.")
        # Check if we should run a test video job
        elif testvideo_folder is not None:
            write_log("Test video mode requested")
            # Convert empty string to None for "first folder" logic
            folder_arg = testvideo_folder if testvideo_folder else None
            success = run_test_video_terminal(bootup_manager, folder_arg, app)
            if success:
                write_log("Test video job completed. Exiting.")
            else:
                write_log("Test video job failed. Exiting.")
        else:
            # Start job processing loop (like pressing play button in GUI mode)
            write_log("Starting job processing mode...")
            run_job_processing_loop_terminal(bootup_manager, app)


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    main() 