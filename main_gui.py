#!/usr/bin/env python3
"""
GUI-specific functionality for the Route Squiggler Render Client.
This module contains classes and functions that are only used in GUI mode.
"""

# Standard library imports
import multiprocessing as mp
import os
from queue import Empty

# Third-party imports
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Local imports
import map_tile_cache_sweep
import video_generator_cache_map_tiles
from bootup import BootupManager, BootupThread, BootupWorker
from image_generator_main import ImageGeneratorWorker, ImageWorkerThread
from image_generator_multiprocess import StatusUpdate
from image_generator_test import TestImageManager
from job_request import JobRequestManager
from log_widget import LogWidget
from sync_map_tiles import sync_map_tiles
from video_generator_test import TestVideoManager

# System tray (optional)
try:
    from tray import SystemTray
    SYSTEM_TRAY_AVAILABLE = True
except ImportError:
    SYSTEM_TRAY_AVAILABLE = False
    SystemTray = None


class FileCheckWorker(QObject):
    """Worker to check for files to upload in background thread."""
    finished = Signal(int)  # Emits number of files to upload
    
    def __init__(self, storage_box_address, storage_box_user, storage_box_password):
        super().__init__()
        self.storage_box_address = storage_box_address
        self.storage_box_user = storage_box_user
        self.storage_box_password = storage_box_password
    
    def check_files(self):
        """Check for files to upload using dry run."""
        try:
            # Perform dry run to check for files to upload
            success, files_to_upload, files_to_download = sync_map_tiles(
                storage_box_address=self.storage_box_address,
                storage_box_user=self.storage_box_user,
                storage_box_password=self.storage_box_password,
                local_cache_dir="map tile cache",
                log_callback=lambda msg: None,  # Don't log during exit check
                progress_callback=lambda msg: None,  # Don't show progress during exit check
                sync_state_callback=lambda state: None,  # Don't change UI state during exit check
                dry_run=True,
                upload_only=True  # Only check for uploads
            )
            
            self.finished.emit(files_to_upload if success else 0)
            
        except Exception as e:
            # If there's any error, assume no files to upload
            self.finished.emit(0)


class SyncWorker(QObject):
    """Worker to perform file syncing in background thread."""
    finished = Signal(bool, int, int)  # Emits success, uploaded_count, downloaded_count
    
    def __init__(self, storage_box_address, storage_box_user, storage_box_password):
        super().__init__()
        self.storage_box_address = storage_box_address
        self.storage_box_user = storage_box_user
        self.storage_box_password = storage_box_password
    
    def sync_files(self):
        """Perform the actual file sync."""
        try:
            # Clean bad tiles from cache before syncing
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
                
            except Exception as sweep_error:
                # Continue with sync even if sweep fails (silent for exit sync)
                pass
            
            # Perform the actual sync
            success, uploaded_count, downloaded_count = sync_map_tiles(
                storage_box_address=self.storage_box_address,
                storage_box_user=self.storage_box_user,
                storage_box_password=self.storage_box_password,
                local_cache_dir="map tile cache",
                log_callback=lambda msg: None,  # Don't log during exit sync
                progress_callback=lambda msg: None,  # Don't show progress during exit sync
                sync_state_callback=lambda state: None,  # Don't change UI state during exit sync
                max_workers=5,  # Use fewer workers for exit sync
                dry_run=False,
                upload_only=True
            )
            
            self.finished.emit(success, uploaded_count, downloaded_count)
            
        except Exception as e:
            # If sync fails, emit failure
            self.finished.emit(False, 0, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Route Squiggler - Render Client")
        self.setGeometry(100, 100, 1024, 768)
        
        # Set dark mode styling for the main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
        """)
        
        # Initialize configuration variables
        self.app_version = None
        self.api_url = None
        self.user = None
        self.hardware_id = None
        self.worker_thread = None
        self.worker = None

        # Initialize status tracking
        self.status_labels = {}  # Dict to store status labels for each zoom level
        self.status_timer = None  # Timer for checking status updates
        self.status_queue = None  # Queue for receiving status updates
        
        # Initialize bootup thread attributes
        self.bootup_worker = None
        self.bootup_thread = None
        
        # Exit state tracking
        self.intentionally_quitting = False
        
        # File check worker for background file checking
        self.file_check_worker = None
        self.file_check_thread = None
        
        # Sync worker for background file syncing
        self.sync_worker = None
        self.sync_thread = None
        
        # CPU threads configuration
        self.cpu_cores = mp.cpu_count()
        # Use thread count from config (set by command line or default)
        from config import config
        self.available_threads = config.thread_count
        
        # Video progress bars configuration
        self.hide_video_progress_on_completion = False

        # Create test image manager
        self.test_image_manager = TestImageManager(self)

        # Create test video manager
        self.test_video_manager = TestVideoManager(self)

        # Create job request manager
        self.job_request_manager = JobRequestManager(self)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ["ROUTE_SQUIGGLER_RENDERER_ROOT_DIRECTORY"] = str(base_dir)
        # Get root dir like this: 	root_dir = os.environ.get("ROUTE_SQUIGGLER_RENDERER_ROOT_DIRECTORY", os.path.dirname(os.path.abspath(__file__)))
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create top section that will contain status, play controls and buttons
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create status container (top left corner)
        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(5)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        # Add header label to status container
        self.header_label = QLabel("")
        self.header_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                background-color: transparent;
                min-width: 200px;
            }
        """)
        
        # Create video progress bars container
        self.video_progress_container = QWidget()
        self.video_progress_layout = QVBoxLayout(self.video_progress_container)
        self.video_progress_layout.setContentsMargins(0, 0, 0, 0)
        self.video_progress_layout.setSpacing(3)
        
        # Create the six video progress bars
        self.progress_bar_combined_route = QProgressBar()
        self.progress_bar_combined_route.setTextVisible(True)
        self.progress_bar_combined_route.setFormat("Creating combined route")
        self.progress_bar_combined_route.setMinimum(0)
        self.progress_bar_combined_route.setMaximum(100)
        self.progress_bar_combined_route.setValue(0)
        
        self.progress_bar_tiles = QProgressBar()
        self.progress_bar_tiles.setTextVisible(True)
        self.progress_bar_tiles.setFormat("Verifying / downloading map tiles")
        self.progress_bar_tiles.setMinimum(0)
        self.progress_bar_tiles.setMaximum(100)
        self.progress_bar_tiles.setValue(0)
        
        self.progress_bar_map_images = QProgressBar()
        self.progress_bar_map_images.setTextVisible(True)
        self.progress_bar_map_images.setFormat("Creating map images")
        self.progress_bar_map_images.setMinimum(0)
        self.progress_bar_map_images.setMaximum(100)
        self.progress_bar_map_images.setValue(0)
        
        self.progress_bar_frames = QProgressBar()
        self.progress_bar_frames.setTextVisible(True)
        self.progress_bar_frames.setFormat("Creating video frames")
        self.progress_bar_frames.setMinimum(0)
        self.progress_bar_frames.setMaximum(100)
        self.progress_bar_frames.setValue(0)
        
        self.progress_bar_upload = QProgressBar()
        self.progress_bar_upload.setTextVisible(True)
        self.progress_bar_upload.setFormat("Uploading files")
        self.progress_bar_upload.setMinimum(0)
        self.progress_bar_upload.setMaximum(100)
        self.progress_bar_upload.setValue(0)
        
        # Style all progress bars
        progress_bar_style = """
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #3c3c3c;
                text-align: center;
                color: #ffffff;
                font-size: 11px;
                height: 20px;
                min-width: 200px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """
        
        for progress_bar in [self.progress_bar_combined_route, self.progress_bar_tiles, self.progress_bar_map_images, 
                           self.progress_bar_frames, self.progress_bar_upload]:
            progress_bar.setStyleSheet(progress_bar_style)
        
        # Add progress bars to the container
        self.video_progress_layout.addWidget(self.progress_bar_combined_route)
        self.video_progress_layout.addWidget(self.progress_bar_tiles)
        self.video_progress_layout.addWidget(self.progress_bar_map_images)
        self.video_progress_layout.addWidget(self.progress_bar_frames)
        self.video_progress_layout.addWidget(self.progress_bar_upload)
        
        # Hide video progress bars initially
        self.video_progress_container.hide()
        
        # Create status grid for zoom level processing
        self.status_grid = QWidget()
        self.status_grid_layout = QGridLayout(self.status_grid)
        self.status_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.status_grid_layout.setSpacing(5)
        self.status_grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        # Add status elements to status container
        status_layout.addWidget(self.header_label)
        status_layout.addWidget(self.video_progress_container)
        status_layout.addWidget(self.status_grid)
        
        # Create play container (center of top section)
        play_container = QWidget()
        play_layout = QVBoxLayout(play_container)
        play_layout.setContentsMargins(0, 0, 0, 0)
        play_layout.setSpacing(2)  # Minimal spacing between button and label
        play_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        
        # Create play and pause buttons
        buttons_row = QWidget()
        buttons_row_layout = QHBoxLayout(buttons_row)
        buttons_row_layout.setContentsMargins(0, 0, 0, 0)
        buttons_row_layout.setSpacing(5)

        self.play_button = QPushButton("▶️")
        self.pause_button = QPushButton("⏸️")
        for button in [self.play_button, self.pause_button]:
            button.setFixedSize(70, 70)
            button.setStyleSheet("""
                QPushButton {
                    font-size: 40px;
                    background: transparent;
                    border: none;
                    padding: 5px;
                }
                QPushButton:hover {
                    opacity: 0.8;
                }
                QPushButton:disabled {
                    opacity: 0.5;
                }
            """)
        
        self.play_button.clicked.connect(self.start_processing)
        self.pause_button.clicked.connect(self.pause_processing)
        
        # Initially, pause is disabled (since we start paused)
        self.pause_button.setEnabled(False)
        
        buttons_row_layout.addWidget(self.play_button)
        buttons_row_layout.addWidget(self.pause_button)
        
        self.play_label = QLabel("Bootup successful! Press play to start working.")
        self.play_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #cccccc;
                background-color: transparent;
            }
        """)
        self.play_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add bootup status label (will replace play_label initially)
        self.bootup_status_label = QLabel("Starting bootup sequence...")
        self.bootup_status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #ffaa00;
                background-color: #444444;
                border: 1px solid #666666;
                padding: 8px;
                border-radius: 4px;
                margin: 5px;
            }
        """)
        self.bootup_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add no_jobs_label below play_label
        self.no_jobs_label = QLabel("Idle - waiting for jobs from the server.")
        self.no_jobs_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #999999;
                background-color: transparent;
                font-style: italic;
            }
        """)
        self.no_jobs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_jobs_label.hide()  # Hidden by default
        
        # Add drive space warning label
        self.drive_space_warning_label = QLabel("")
        self.drive_space_warning_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #ffaa00;
                background-color: #444444;
                border: 1px solid #ffaa00;
                padding: 6px;
                border-radius: 4px;
                margin: 3px;
                font-weight: bold;
            }
        """)
        self.drive_space_warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drive_space_warning_label.hide()  # Hidden by default
        
        # Hide play controls initially - they'll be shown after successful bootup
        self.play_button.hide()
        self.pause_button.hide()
        self.play_label.hide()
        
        play_layout.addWidget(buttons_row)
        play_layout.addWidget(self.bootup_status_label)  # Show bootup status initially
        play_layout.addWidget(self.play_label)  # Hidden initially
        play_layout.addWidget(self.no_jobs_label)
        play_layout.addWidget(self.drive_space_warning_label)
        
        # Create buttons container (right side of top section)
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(5)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        
        # Add hide to tray button
        self.hide_button = QPushButton("Hide to tray")
        self.hide_button.clicked.connect(self.hide_to_tray)
        self.hide_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 1px solid #45a049;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                max-width: 120px;
                max-height: 35px;
            }
            QPushButton:hover {
                background-color: #45a049;
                border: 1px solid #3d8b40;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        # Add exit program button
        self.exit_button = QPushButton("Exit program")
        self.exit_button.clicked.connect(self.exit_program)
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 1px solid #45a049;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                max-width: 120px;
                max-height: 35px;
            }
            QPushButton:hover {
                background-color: #45a049;
                border: 1px solid #3d8b40;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        
        # Add CPU threads dropdown
        self.cpu_threads_combo = QComboBox()
        for i in range(1, self.cpu_cores + 1):
            if i == self.cpu_cores:
                # Add "(max)" to the last option
                self.cpu_threads_combo.addItem(f"{i} thread{'s' if i > 1 else ''} (max)", i)
            else:
                self.cpu_threads_combo.addItem(f"{i} thread{'s' if i > 1 else ''}", i)
        # Set to the thread count from config (set by command line or default)
        # Find the index that matches our config thread count
        target_threads = config.thread_count
        target_index = -1
        for i in range(self.cpu_threads_combo.count()):
            if self.cpu_threads_combo.itemData(i) == target_threads:
                target_index = i
                break
        
        # If we found a match, use it; otherwise use the default (max-2, min 1)
        if target_index >= 0:
            self.cpu_threads_combo.setCurrentIndex(target_index)
        else:
            # Fallback to default calculation
            default_index = max(0, self.cpu_cores - 2 - 1)  # -1 because index is 0-based
            self.cpu_threads_combo.setCurrentIndex(default_index)
        self.cpu_threads_combo.currentIndexChanged.connect(self.on_threads_changed)
        
        # Style the combo box to match buttons
        self.cpu_threads_combo.setStyleSheet("""
            QComboBox {
                background-color: #4CAF50;
                color: white;
                border: 1px solid #45a049;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                max-width: 120px;
                max-height: 35px;
                min-width: 120px;
            }
            QComboBox:hover {
                background-color: #45a049;
                border: 1px solid #3d8b40;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #3d8b40;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background-color: #45a049;
            }
            QComboBox::down-arrow {
                image: none;
                border: 1px solid white;
                width: 0px;
                height: 0px;
                border-top: 4px solid white;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                margin: 4px;
            }
            QComboBox QAbstractItemView {
                background-color: #4CAF50;
                color: white;
                border: 1px solid #45a049;
                selection-background-color: #45a049;
            }
        """)
        
        # Add test image button
        self.test_image_button = QPushButton("Test image")
        self.test_image_button.setStyleSheet(self.hide_button.styleSheet())
        
        # Add test video button
        self.test_video_button = QPushButton("Test video")
        self.test_video_button.setStyleSheet(self.hide_button.styleSheet())
        
        # Add buttons to their container in desired order
        buttons_layout.addWidget(self.exit_button)  # Exit button first
        buttons_layout.addWidget(self.hide_button)  # Hide button second
        buttons_layout.addWidget(self.cpu_threads_combo)  # CPU threads dropdown third
        buttons_layout.addStretch()  # Empty padding
        buttons_layout.addWidget(self.test_image_button)
        buttons_layout.addWidget(self.test_video_button)
        
        # Hide exit and hide buttons initially - they'll be shown after successful bootup
        self.exit_button.hide()
        self.hide_button.hide()
        
        # Add status container, play container and buttons container to top layout
        top_layout.addWidget(status_container, 0)  # No stretch to keep status at natural size
        top_layout.addWidget(play_container, 1)  # Stretch factor 1 to center play button
        top_layout.addWidget(buttons_container, 0)  # No stretch to keep buttons at natural size
        
        # Add widgets to main layout
        main_layout.addWidget(top_section)
        
        # Create log widget
        self.log_widget = LogWidget(self)
        
        # Create splitter and add widgets
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(central_widget)
        self.splitter.addWidget(self.log_widget)
        
        # Set splitter as the main window's central widget
        self.setCentralWidget(self.splitter)
        
        # Set initial splitter sizes - start with log expanded
        self.splitter.setSizes([0, self.height() - 50])  # Give most space to log initially
        self.log_widget.set_normal_sizes([600, 100])  # But remember normal sizes for collapsing - changed from 100 to 200 for 10 lines instead of 5
        
        # Hide main content initially since log is expanded
        central_widget.hide()
        
        # Add some initial log entries
        self.log_widget.add_log("UI initialized")
        self.log_widget.add_log(f"Detected {self.cpu_cores} CPU cores")
        self.log_widget.add_log(f"Default thread count: {self.available_threads} (max-2, min 1)")
        
        # Show UI immediately so it can update during bootup
        central_widget.show()
        
        # Initialize bootup manager and start async bootup
        self.bootup_manager = BootupManager(self)
        self.start_async_bootup()
        
        # Connect test image button
        self.test_image_button.clicked.connect(self.test_image_manager.test_image)
        
        # Connect test video button
        self.test_video_button.clicked.connect(self.test_video_manager.test_video)
    
    def hide_to_tray(self):
        """Hide the application to system tray"""
        if hasattr(self, 'system_tray') and self.system_tray and self.system_tray.is_visible():
            self.hide()
            self.log_widget.add_log("Application hidden to system tray")
        else:
            self.log_widget.add_log("System tray not available")
        
    def show_normal(self):
        """Show and raise the window"""
        self.show()
        self.raise_()
        self.activateWindow()
    
    def quit_application(self):
        """Quit the application immediately."""
        self.intentionally_quitting = True  # Set flag to bypass closeEvent sync check
        self.log_widget.add_log("Application closing...")
        if hasattr(self, 'system_tray') and self.system_tray:
            self.system_tray.hide_tray_icon()
        self.close()  # Close the window directly instead of QApplication.quit()

    def exit_program(self):
        """Exit the application with map tile sync check."""
        # Get storage box credentials
        if not hasattr(self, 'bootup_manager') or not self.bootup_manager:
            self.intentionally_quitting = True
            self.close()
            return
        
        credentials = self.bootup_manager.storage_box_credentials
        if not all([credentials['address'], credentials['user'], credentials['password']]):
            self.intentionally_quitting = True
            self.close()
            return
        
        # Show a "checking files" dialog
        self.checking_dialog = QMessageBox(self)
        self.checking_dialog.setWindowTitle("Checking Files")
        self.checking_dialog.setText("Checking for files to sync...")
        self.checking_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
        self.checking_dialog.show()
        
        # Force the dialog to display
        QApplication.processEvents()
        
        # Start background file check
        self.start_background_file_check(credentials)

    def check_for_upload_files(self):
        """Check if there are files to upload using dry run."""
        try:
            # Get storage box credentials from bootup manager directly
            if not hasattr(self, 'bootup_manager') or not self.bootup_manager:
                return 0
            
            credentials = self.bootup_manager.storage_box_credentials
            if not all([credentials['address'], credentials['user'], credentials['password']]):
                return 0
            
            # Perform dry run to check for files to upload
            success, files_to_upload, files_to_download = sync_map_tiles(
                storage_box_address=credentials['address'],
                storage_box_user=credentials['user'],
                storage_box_password=credentials['password'],
                local_cache_dir="map tile cache",
                log_callback=lambda msg: None,  # Don't log during exit check
                progress_callback=lambda msg: None,  # Don't show progress during exit check
                sync_state_callback=lambda state: None,  # Don't change UI state during exit check
                dry_run=True,
                upload_only=True  # Only check for uploads
            )
            
            return files_to_upload if success else 0
            
        except Exception as e:
            # If there's any error, assume no files to upload
            return 0

    def show_exit_sync_dialog(self, files_to_upload):
        """Show dialog asking user if they want to sync before exiting."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Map Tile Sync")
        msg_box.setText(f"There are {files_to_upload} map tiles to sync.")
        
        # Create custom buttons
        sync_button = msg_box.addButton("Sync then quit (recommended)", QMessageBox.ButtonRole.AcceptRole)
        quit_button = msg_box.addButton("Quit now (I'm in a hurry)", QMessageBox.ButtonRole.RejectRole)
        
        # Set default button
        msg_box.setDefaultButton(sync_button)
        
        # Show the dialog
        result = msg_box.exec()
        
        # Check which button was clicked based on the result
        if result == 2:  # AcceptRole (Sync then quit)
            # User chose to sync then quit
            self.sync_and_quit()
        else:  # RejectRole or any other value (Quit now)
            # User chose to quit immediately
            self.quit_application()

    def sync_and_quit(self):
        """Perform upload-only sync then quit."""
        # Get storage box credentials from bootup manager
        if not hasattr(self, 'bootup_manager') or not self.bootup_manager:
            self.intentionally_quitting = True
            self.close()
            return
        
        credentials = self.bootup_manager.storage_box_credentials
        if not all([credentials['address'], credentials['user'], credentials['password']]):
            self.intentionally_quitting = True
            self.close()
            return
        
        # Show a simple progress dialog
        self.sync_progress_dialog = QMessageBox(self)
        self.sync_progress_dialog.setWindowTitle("Syncing Map Tiles")
        self.sync_progress_dialog.setText("Syncing map tiles before exit...")
        self.sync_progress_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
        self.sync_progress_dialog.show()
        
        # Force the dialog to display
        QApplication.processEvents()
        
        # Start background sync
        self.start_background_sync(credentials)

    def closeEvent(self, event):
        """Handle application close with map tile sync check."""
        # COMMENTED OUT: Map tile sync on exit
        # This was disabled to allow immediate exit without checking/syncing map tiles
        # To restore this functionality, uncomment the code below
        
        # # If we're intentionally quitting, just close immediately
        # if self.intentionally_quitting:
        #     event.accept()
        #     return
        #     
        # # Get storage box credentials
        # if not hasattr(self, 'bootup_manager') or not self.bootup_manager:
        #     event.accept()
        #     return
        # 
        # credentials = self.bootup_manager.storage_box_credentials
        # if not all([credentials['address'], credentials['user'], credentials['password']]):
        #     event.accept()
        #     return
        # 
        # # Show a "checking files" dialog
        # self.checking_dialog = QMessageBox(self)
        # self.checking_dialog.setWindowTitle("Checking Files")
        # self.checking_dialog.setText("Checking for files to sync...")
        # self.checking_dialog.setStandardButtons(QMessageBox.StandardButton.NoButton)
        # self.checking_dialog.show()
        # 
        # # Force the dialog to display
        # QApplication.processEvents()
        # 
        # # Start background file check
        # self.start_background_file_check(credentials)
        # 
        # # Don't close yet - wait for background check to complete
        # event.ignore()
        
        # Allow immediate exit
        event.accept()

    def create_status_labels(self, zoom_levels):
        """Create status labels for each zoom level."""
        # Clear existing labels
        self.clear_status_labels()
        
        # Create new labels
        for i, zoom_level in enumerate(zoom_levels):
            label = QLabel(f"Zoom {zoom_level}: waiting")
            label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #cccccc;
                    background-color: #3c3c3c;
                    border: 1px solid #555555;
                    padding: 4px 8px;
                    border-radius: 3px;
                    min-width: 160px;
                }
            """)
            row = i  # stack vertically
            col = 0
            self.status_grid_layout.addWidget(label, row, col)
            self.status_labels[zoom_level] = label
        
        # Show the central widget if it was hidden
        central_widget = self.centralWidget().widget(0)  # Get the central widget from splitter
        central_widget.show()
    
    def clear_status_labels(self):
        """Clear all status labels."""
        try:
            # Stop the status update timer first to prevent conflicts
            if self.status_timer:
                try:
                    self.status_timer.stop()
                    self.status_timer = None
                except Exception as e:
                    # If there's an error stopping the timer, just log it and continue
                    self.log_widget.add_log(f"Warning: Error stopping status timer: {str(e)}")
                    self.status_timer = None
            
            # Clear the status queue to prevent stale updates
            self.status_queue = None
            
            # Remove and delete all existing labels with proper error handling
            while self.status_grid_layout.count():
                try:
                    item = self.status_grid_layout.takeAt(0)
                    if item and item.widget():
                        widget = item.widget()
                        widget.hide()  # Hide first to prevent painting issues
                        widget.deleteLater()
                except Exception as e:
                    self.log_widget.add_log(f"Warning: Error clearing status label: {str(e)}")
                    continue
            
            self.status_labels.clear()
            
            # Hide video progress bars when clearing status (for image jobs)
            # But only if we're not in development mode
            if self.hide_video_progress_on_completion:
                self.hide_video_progress_bars()
            
            # Force a repaint to ensure UI is updated
            QApplication.processEvents()
            
        except Exception as e:
            self.log_widget.add_log(f"Error in clear_status_labels: {str(e)}")
            import traceback
            self.log_widget.add_log(traceback.format_exc())
    
    def setup_status_monitoring(self, status_queue):
        """Set up status monitoring with the provided queue."""
        self.status_queue = status_queue
        
        # Start the status update timer if not already running
        if not self.status_timer:
            self.status_timer = QTimer(self)  # Make sure timer has a parent
            self.status_timer.timeout.connect(self.check_status_updates)
            self.status_timer.start(100) 
#            self.log_widget.add_log("Status monitoring started")

    def check_status_updates(self):
        """Check for status updates from worker processes."""
        if not self.status_queue:
            return
        
        try:
            # Process all available updates
            updates_processed = 0
            max_updates_per_check = 100  # Prevent processing too many at once
            
            while updates_processed < max_updates_per_check:
                try:
                    # Use get_nowait() to avoid blocking
                    update = self.status_queue.get_nowait()
                    if not isinstance(update, StatusUpdate):
                        continue
                        
                    label = self.status_labels.get(update.zoom_level)
                    if not label:
                        continue
                        
                    # Update the label text and style
                    if update.error:
                        label.setText(f"Zoom {update.zoom_level}: error")
                        label.setStyleSheet("""
                            QLabel {
                                font-size: 14px;
                                color: white;
                                background-color: #ff4444;
                                border: 1px solid #cc0000;
                                padding: 10px;
                                border-radius: 5px;
                            }
                        """)
                        # Only update log widget if textfield is True
                        if getattr(update, 'textfield', True):
                            self.log_widget.add_log(f"Error in zoom level {update.zoom_level}: {update.error}")
                    else:
                        # Force an immediate update
                        label.setText(f"Zoom {update.zoom_level}: {update.status}")
                        
                        if update.status == "complete":
                            label.setStyleSheet("""
                                QLabel {
                                    font-size: 14px;
                                    color: white;
                                    background-color: #4CAF50;
                                    border: 1px solid #45a049;
                                    padding: 10px;
                                    border-radius: 5px;
                                }
                            """)
                        elif update.status == "error":
                            label.setStyleSheet("""
                                QLabel {
                                    font-size: 14px;
                                    color: white;
                                    background-color: #ff4444;
                                    border: 1px solid #cc0000;
                                    padding: 10px;
                                    border-radius: 5px;
                                }
                            """)
                        else:
                            # In progress style - different colors for different stages
                            color = {
                                "calculating bounds": "#8E24AA",  # Purple
                                "loading tiles": "#7E57C2",        # Light purple  
                                "plotting": "#2196F3",           # Blue
                                "drawing tracks": "#4CAF50",     # Green
                                "adding legend": "#FF9800",      # Orange
                                "adding stamp": "#E91E63",       # Pink
                                "saving image": "#00BCD4",       # Cyan
                                "compressing": "#795548",        # Brown
                                "uploading": "#607D8B",          # Blue grey
                            }.get(update.status, "#444444")
                            
                            label.setStyleSheet(f"""
                                QLabel {{
                                    font-size: 14px;
                                    color: #ffffff;
                                    background-color: {color};
                                    border: 1px solid #666666;
                                    padding: 10px;
                                    border-radius: 5px;
                                }}
                            """)
                        
                        # Only update log widget if textfield is True
                        if getattr(update, 'textfield', True):
                            self.log_widget.add_log(f"Zoom level {update.zoom_level}: {update.status}")
                    
                    # Force the label to repaint immediately
                    label.repaint()
                    
                    # Increment the counter
                    updates_processed += 1
                    
                except Empty:
                    # No more updates in queue
                    break
                    
            # Process any pending events to ensure UI updates
            if updates_processed > 0:
                QApplication.processEvents()
                
        except Exception as e:
            self.log_widget.add_log(f"Error processing status updates: {str(e)}")
            import traceback
            self.log_widget.add_log(traceback.format_exc())
    
   
    def on_worker_finished(self):
        """Handle completion and move to next test job if available."""
        try:
            # Determine job type based on worker type
            if hasattr(self.worker, 'video_generator_process'):
                self.log_widget.add_log("Video generation completed successfully")
                # Hide video progress bars after completion if flag is set
                if self.hide_video_progress_on_completion:
                    self.hide_video_progress_bars()
            else:
                self.log_widget.add_log("Image generation completed successfully")

            # Clean up worker thread first
            if self.worker_thread:
                try:
                    self.worker_thread.quit()
                    self.worker_thread.wait(5000)  # Wait up to 5 seconds
                    self.worker_thread = None
                    self.worker = None
                except Exception as e:
                    self.log_widget.add_log(f"Warning: Error cleaning up worker thread: {str(e)}")
                    self.worker_thread = None
                    self.worker = None

            # Additional cleanup for matplotlib and Qt resources
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                
                # Close all matplotlib figures to release Qt painters
                plt.close('all')
                
                # Clear matplotlib cache
                matplotlib.pyplot.clf()
                matplotlib.pyplot.cla()
                
                # Force garbage collection to clean up any remaining Qt objects
                import gc
                gc.collect()
                
                self.log_widget.add_log("Matplotlib and Qt resources cleaned up")
                
            except Exception as e:
                self.log_widget.add_log(f"Warning: Error during matplotlib cleanup: {str(e)}")

            # Additional cleanup for MoviePy and FFmpeg processes
            try:
                import psutil
                
                # Find and terminate any remaining FFmpeg processes that might be hanging
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # Check if this is an FFmpeg process related to our video generation
                        if proc.info['name'] and 'ffmpeg' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline']
                            if cmdline and any('moviepy' in str(arg).lower() or 'temp' in str(arg).lower() for arg in cmdline):
                                self.log_widget.add_log(f"Terminating hanging FFmpeg process: PID {proc.info['pid']}")
                                proc.terminate()
                                proc.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                        # Process already terminated or we don't have permission
                        pass
                    except Exception as e:
                        self.log_widget.add_log(f"Warning: Error checking FFmpeg process: {str(e)}")
                
                # Force kill any remaining FFmpeg processes as a last resort
                try:
                    import subprocess
                    import os
                    if os.name == 'nt':  # Windows
                        subprocess.run(['taskkill', '/f', '/im', 'ffmpeg.exe'], 
                                     capture_output=True, timeout=10)
                    else:  # Linux/Mac
                        subprocess.run(['pkill', '-f', 'ffmpeg'], 
                                     capture_output=True, timeout=10)
                    self.log_widget.add_log("Force-killed any remaining FFmpeg processes")
                except Exception as e:
                    self.log_widget.add_log(f"Warning: Error force-killing FFmpeg: {str(e)}")
                
                self.log_widget.add_log("MoviePy and FFmpeg processes cleaned up")
                
            except Exception as e:
                self.log_widget.add_log(f"Warning: Error during MoviePy cleanup: {str(e)}")

            # Force Qt resource cleanup to prevent painter conflicts
            self.force_qt_cleanup()

            # Check if we're in test mode and handle accordingly
            if self.test_image_manager.test_job_folders:
                self.test_image_manager.on_worker_finished()
            elif self.test_video_manager.test_job_folders:
                self.test_video_manager.on_worker_finished()
            # If pause button is enabled, we're in working mode
            elif self.pause_button.isEnabled():
                # Add a longer delay to prevent UI conflicts and allow cleanup to complete
                QTimer.singleShot(500, self.job_request_manager.request_new_job_qt_network)
            else:
                self.log_widget.add_log("Processing paused. Press play to continue working.")
                
        except Exception as e:
            self.log_widget.add_log(f"Error in on_worker_finished: {str(e)}")
            import traceback
            self.log_widget.add_log(traceback.format_exc())

    def on_worker_error(self, error_msg):
        try:
            # Determine job type based on worker type
            if hasattr(self.worker, 'video_generator_process'):
                self.log_widget.add_log(f"Error in video generation: {error_msg}")
                # Hide video progress bars after error if flag is set
                if self.hide_video_progress_on_completion:
                    self.hide_video_progress_bars()
            else:
                self.log_widget.add_log(f"Error in image generation: {error_msg}")

            # Clean up worker thread first
            if self.worker_thread:
                try:
                    self.worker_thread.quit()
                    self.worker_thread.wait(5000)  # Wait up to 5 seconds
                    self.worker_thread = None
                    self.worker = None
                except Exception as e:
                    self.log_widget.add_log(f"Warning: Error cleaning up worker thread: {str(e)}")
                    self.worker_thread = None
                    self.worker = None

            # Additional cleanup for matplotlib and Qt resources
            try:
                import matplotlib.pyplot as plt
                import matplotlib
                
                # Close all matplotlib figures to release Qt painters
                plt.close('all')
                
                # Clear matplotlib cache
                matplotlib.pyplot.clf()
                matplotlib.pyplot.cla()
                
                # Force garbage collection to clean up any remaining Qt objects
                import gc
                gc.collect()
                
                self.log_widget.add_log("Matplotlib and Qt resources cleaned up")
                
            except Exception as e:
                self.log_widget.add_log(f"Warning: Error during matplotlib cleanup: {str(e)}")

            # Additional cleanup for MoviePy and FFmpeg processes
            try:
                import psutil
                
                # Find and terminate any remaining FFmpeg processes that might be hanging
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # Check if this is an FFmpeg process related to our video generation
                        if proc.info['name'] and 'ffmpeg' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline']
                            if cmdline and any('moviepy' in str(arg).lower() or 'temp' in str(arg).lower() for arg in cmdline):
                                self.log_widget.add_log(f"Terminating hanging FFmpeg process: PID {proc.info['pid']}")
                                proc.terminate()
                                proc.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                        # Process already terminated or we don't have permission
                        pass
                    except Exception as e:
                        self.log_widget.add_log(f"Warning: Error checking FFmpeg process: {str(e)}")
                
                # Force kill any remaining FFmpeg processes as a last resort
                try:
                    import subprocess
                    import os
                    if os.name == 'nt':  # Windows
                        subprocess.run(['taskkill', '/f', '/im', 'ffmpeg.exe'], 
                                     capture_output=True, timeout=10)
                    else:  # Linux/Mac
                        subprocess.run(['pkill', '-f', 'ffmpeg'], 
                                     capture_output=True, timeout=10)
                    self.log_widget.add_log("Force-killed any remaining FFmpeg processes")
                except Exception as e:
                    self.log_widget.add_log(f"Warning: Error force-killing FFmpeg: {str(e)}")
                
                self.log_widget.add_log("MoviePy and FFmpeg processes cleaned up")
                
            except Exception as e:
                self.log_widget.add_log(f"Warning: Error during MoviePy cleanup: {str(e)}")

            # Force Qt resource cleanup to prevent painter conflicts
            self.force_qt_cleanup()

            # Check if we're in test mode and handle accordingly
            if self.test_image_manager.test_job_folders:
                self.test_image_manager.on_worker_error(error_msg)
            elif self.test_video_manager.test_job_folders:
                self.test_video_manager.on_worker_error(error_msg)
                
        except Exception as e:
            self.log_widget.add_log(f"Error in on_worker_error: {str(e)}")
            import traceback
            self.log_widget.add_log(traceback.format_exc())

    def on_job_completed(self, job_id):
        """Handle successful job completion."""
        # Determine job type based on worker type
        if hasattr(self.worker, 'video_generator_process'):
            self.header_label.setText(f"Video #{job_id} created successfully")
        else:
            self.header_label.setText(f"Image #{job_id} created successfully")

    def start_processing(self):
        """Start or resume processing jobs."""
        self.play_button.hide()  # Hide play button
        self.pause_button.show()  # Show pause button
        self.pause_button.setEnabled(True)
        self.play_label.setText("Working. Press pause to stop.")
        
        # Hide drive space warning when user starts processing
        self.drive_space_warning_label.hide()
        
        # Request first/next job using Qt networking (non-blocking UI)
        self.job_request_manager.request_new_job_qt_network()
    
    def pause_processing(self):
        """Pause job processing."""
        self.play_button.show()  # Show play button
        self.pause_button.hide()  # Hide pause button
        self.play_button.setEnabled(True)
        self.play_label.setText("Processing paused. Press play to continue.")
        
        # Hide no_jobs_label when paused
        self.no_jobs_label.hide()
        
        # Cancel any pending job retry
        self.job_request_manager.cancel_retry_timer()
        
        # Clean up any active job request worker
        self.job_request_manager.cleanup_job_request_worker()
        
        self.log_widget.add_log("Cancelled pending job retry and cleaned up job request worker")

    @Slot()
    def create_job_retry_timer(self):
        """Create a retry timer for job requests (called from main thread)."""
        try:
            # Clean up any existing retry timer
            if hasattr(self.job_request_manager, 'job_retry_timer') and self.job_request_manager.job_retry_timer:
                try:
                    self.job_request_manager.job_retry_timer.stop()
                    self.job_request_manager.job_retry_timer.deleteLater()
                    self.log_widget.add_log("Cleaned up existing retry timer")
                except Exception as e:
                    self.log_widget.add_log(f"Warning: Error stopping existing retry timer: {str(e)}")
                self.job_request_manager.job_retry_timer = None
                
            # Create a single-shot timer for delayed retry
            from PySide6.QtCore import QTimer
            self.job_request_manager.job_retry_timer = QTimer(self)
            self.job_request_manager.job_retry_timer.setSingleShot(True)
            self.job_request_manager.job_retry_timer.timeout.connect(self.job_request_manager.delayed_job_retry)
            self.job_request_manager.job_retry_timer.start(10000)  # 10 second delay
            self.log_widget.add_log("Retry timer created and started (10 seconds)")
        except Exception as e:
            self.log_widget.add_log(f"Warning: Error creating retry timer: {str(e)}")
            self.job_request_manager.job_retry_timer = None

    def on_threads_changed(self):
        """Handle changes in CPU threads selection."""
        selected_data = self.cpu_threads_combo.currentData()
        if selected_data:
            self.available_threads = selected_data
            self.log_widget.add_log(f"CPU threads set to: {self.available_threads}")
    
    def get_available_threads(self):
        """Get the currently selected number of threads."""
        return self.available_threads

    def show_video_progress_bars(self):
        """Show the video progress bars and reset their values."""
        # Reset all progress bars to 0
        self.progress_bar_combined_route.setValue(0)
        self.progress_bar_tiles.setValue(0)
        self.progress_bar_map_images.setValue(0)
        self.progress_bar_frames.setValue(0)
        self.progress_bar_upload.setValue(0)
        
        # Show the container
        self.video_progress_container.show()
    
    def hide_video_progress_bars(self):
        """Hide the video progress bars."""
        self.video_progress_container.hide()

    def on_video_progress_update(self, progress_bar_name, percentage, progress_text=""):
        """Handle video progress updates."""
        try:
            # Get the appropriate progress bar by name
            progress_bar = getattr(self, progress_bar_name, None)
            if progress_bar:
                progress_bar.setValue(percentage)
                
                # Set the progress text if provided
                if progress_text:
                    progress_bar.setFormat(f"{progress_text} ({percentage}%)")
                else:
                    progress_bar.setFormat(f"{percentage}%")
                
                # Force immediate update
                progress_bar.repaint()
            else:
                self.log_widget.add_log(f"Warning: Unknown progress bar '{progress_bar_name}'")
        except Exception as e:
            self.log_widget.add_log(f"Error updating progress bar {progress_bar_name}: {str(e)}")

    def start_async_bootup(self):
        """Start the asynchronous bootup process."""
        self.bootup_worker = BootupWorker(self.bootup_manager)
        self.bootup_thread = BootupThread(self.bootup_worker)
        
        # Connect signals
        self.bootup_worker.progress.connect(self.on_bootup_progress)
        self.bootup_worker.step_completed.connect(self.on_bootup_step_completed)
        self.bootup_worker.finished.connect(self.on_bootup_finished)
        self.bootup_worker.log_message.connect(self.log_widget.add_log)
        self.bootup_worker.collapse_log.connect(self.on_collapse_log)
        self.bootup_worker.config_loaded.connect(self.on_config_loaded)
        self.bootup_worker.hardware_id_ready.connect(self.on_hardware_id_ready)
        self.bootup_worker.system_tray_ready.connect(self.on_system_tray_ready)
        self.bootup_worker.drive_space_warning.connect(self.on_drive_space_warning)
        
        # Start bootup thread
        self.bootup_thread.start()
    
    def on_bootup_progress(self, message):
        """Handle bootup progress updates."""
        self.bootup_status_label.setText(message)
    
    def on_bootup_step_completed(self, step_name, success):
        """Handle individual bootup step completion."""
        if success:
            self.bootup_status_label.setText(f"✓ {step_name} completed")
        else:
            self.bootup_status_label.setText(f"✗ {step_name} failed")
            self.bootup_status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    color: #ff4444;
                    background-color: #444444;
                    border: 1px solid #cc0000;
                    padding: 8px;
                    border-radius: 4px;
                    margin: 5px;
                }
            """)
    
    def on_bootup_finished(self, success):
        """Handle bootup completion."""
        if success:
            # Hide bootup status and show normal controls
            self.bootup_status_label.hide()
            self.play_button.show()  # Show play button (system starts in pause mode)
            self.pause_button.hide()  # Hide pause button initially
            self.play_label.show()
            
            # Show exit and hide buttons
            self.exit_button.show()
            self.hide_button.show()
            
            # Set storage box credentials for background tile syncing
            if hasattr(self, 'bootup_manager') and self.bootup_manager:
                credentials = self.bootup_manager.storage_box_credentials
                if all([credentials['address'], credentials['user'], credentials['password']]):
                    video_generator_cache_map_tiles.set_storage_box_credentials(credentials)
                    self.log_widget.add_log("Storage box credentials set for background tile syncing")
            
            self.log_widget.add_log("Bootup completed successfully.")
        else:
            # Show failure message
            self.bootup_status_label.setText("Bootup failed - check log for details")
            self.bootup_status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    color: #ff4444;
                    background-color: #444444;
                    border: 1px solid #cc0000;
                    padding: 8px;
                    border-radius: 4px;
                    margin: 5px;
                }
            """)
            self.log_widget.add_log("Bootup failed - application may not function properly")
        
        # Clean up bootup thread
        if self.bootup_thread:
            self.bootup_thread.quit()
            self.bootup_thread.wait()
            self.bootup_thread = None
            self.bootup_worker = None

    def on_collapse_log(self):
        """Handle collapse log signal from bootup worker."""
        if self.log_widget.log_state == "expanded":
            self.log_widget.toggle_log_size()  # This will go from expanded to normal (10 lines)
    
    def on_config_loaded(self, app_version, api_url, user):
        """Handle config loaded signal from bootup worker."""
        self.app_version = app_version
        self.api_url = api_url
        self.user = user
    
    def on_hardware_id_ready(self, hardware_id):
        """Handle hardware ID ready signal from bootup worker."""
        self.hardware_id = hardware_id
    
    def on_system_tray_ready(self, should_create):
        """Handle system tray ready signal from bootup worker."""
        if should_create:
            try:
                if SYSTEM_TRAY_AVAILABLE and SystemTray is not None:
                    self.system_tray = SystemTray(self)
                    # Connect tray signals to window methods
                    self.system_tray.show_window.connect(self.show_normal)
                    self.system_tray.hide_window.connect(self.hide_to_tray)
                    self.system_tray.quit_application.connect(self.quit_application)
                    # Setup the tray with logging callback
                    self.system_tray.setup(self.log_widget.add_log)
                else:
                    self.system_tray = None
            except Exception as e:
                self.log_widget.add_log(f"Failed to create system tray: {str(e)}")
                self.system_tray = None
        else:
            self.system_tray = None

    def on_drive_space_warning(self, message):
        """Handle drive space warning signal from bootup worker."""
        self.drive_space_warning_label.setText(message)
        self.drive_space_warning_label.show()

    def start_background_file_check(self, credentials):
        """Start a background worker to check for files to upload."""
        self.file_check_worker = FileCheckWorker(credentials['address'], credentials['user'], credentials['password'])
        self.file_check_thread = QThread()
        self.file_check_worker.moveToThread(self.file_check_thread)
        
        self.file_check_thread.started.connect(self.file_check_worker.check_files)
        self.file_check_worker.finished.connect(self.on_file_check_finished)
        
        self.file_check_thread.start()

    def on_file_check_finished(self, files_to_upload):
        """Handle the completion of the background file check."""
        self.checking_dialog.close()
        self.checking_dialog = None

        if files_to_upload > 0:
            # Show dialog asking user if they want to sync
            self.show_exit_sync_dialog(files_to_upload)
        else:
            # No files to upload, close immediately
            self.log_widget.add_log("Application closing...")
            if hasattr(self, 'system_tray') and self.system_tray:
                self.system_tray.hide_tray_icon()
            self.intentionally_quitting = True  # Set flag to bypass closeEvent sync check
            self.close()  # Close the window directly

    def start_background_sync(self, credentials):
        """Start a background worker to perform the actual sync."""
        self.sync_worker = SyncWorker(credentials['address'], credentials['user'], credentials['password'])
        self.sync_thread = QThread()
        self.sync_worker.moveToThread(self.sync_thread)
        
        self.sync_thread.started.connect(self.sync_worker.sync_files)
        self.sync_worker.finished.connect(self.on_sync_finished)
        
        self.sync_thread.start()

    def on_sync_finished(self, success, uploaded_count, downloaded_count):
        """Handle the completion of the background sync."""
        self.sync_progress_dialog.close()
        self.sync_progress_dialog = None

        if success:
            self.log_widget.add_log(f"Map tile sync completed. Uploaded: {uploaded_count}, Downloaded: {downloaded_count}")
            self.quit_application()
        else:
            self.log_widget.add_log(f"Map tile sync failed. Uploaded: {uploaded_count}, Downloaded: {downloaded_count}")
            self.quit_application()

    def cleanup_job_request_worker(self):
        """Clean up the job request worker and thread."""
        if self.job_request_thread:
            try:
                self.job_request_thread.quit()
                self.job_request_thread.wait(5000)  # Wait up to 5 seconds
                self.job_request_thread.deleteLater()
                self.job_request_thread = None
                self.job_request_worker = None
                self.log_widget.add_log("Cleaned up job request worker")
            except Exception as e:
                self.log_widget.add_log(f"Warning: Error cleaning up job request worker: {str(e)}")
                self.job_request_thread = None
                self.job_request_worker = None
    
    def force_qt_cleanup(self):
        """Force cleanup of all Qt resources to prevent painter conflicts."""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Process any pending Qt events
            QApplication.processEvents()
            
            # Force cleanup of any remaining Qt objects (more carefully)
            try:
                for obj in gc.get_objects():
                    if hasattr(obj, 'deleteLater'):
                        try:
                            obj.deleteLater()
                        except:
                            pass
            except Exception as cleanup_error:
                # Ignore cleanup errors (like missing _gdbm module)
                pass
            
            self.log_widget.add_log("Forced Qt resource cleanup completed")
            
        except Exception as e:
            # Don't log warnings about missing modules like _gdbm
            if "_gdbm" not in str(e):
                self.log_widget.add_log(f"Warning: Error during forced Qt cleanup: {str(e)}")
