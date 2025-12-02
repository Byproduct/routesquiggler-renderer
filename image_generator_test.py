"""
Test image generation functionality for the Route Squiggler render client.
This module handles running test images from local test folders.
"""

import os
import glob
import json
import traceback
from PySide6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QLabel
from PySide6.QtCore import Qt
from image_generator_utils import load_gpx_files_from_zip
from job_request import apply_vertical_video_swap

class TestFolderSelectionDialog(QDialog):
    """Custom dialog for selecting test folders."""
    
    def __init__(self, parent, folders, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 300)
        
        # Store the folders
        self.folders = folders
        self.selected_folder = None
        
        # Create layout
        layout = QVBoxLayout()
        
        # Add description label
        description = QLabel("Select a test folder to process:")
        description.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(description)
        
        # Create list widget
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #666666;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444444;
            }
            QListWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #3d8b40;
                color: white;
            }
        """)
        
        # Add folder names to the list
        for folder in folders:
            folder_name = os.path.basename(folder)
            item = QListWidgetItem(folder_name)
            item.setData(Qt.ItemDataRole.UserRole, folder)  # Store full path
            self.list_widget.addItem(item)
        
        # Connect double-click to selection
        self.list_widget.itemDoubleClicked.connect(self.accept_selection)
        
        layout.addWidget(self.list_widget)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.select_button = QPushButton("Select Folder")
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 1px solid #45a049;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
                border: 1px solid #3d8b40;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #666666;
                border: 1px solid #555555;
            }
        """)
        self.select_button.clicked.connect(self.accept_selection)
        self.select_button.setEnabled(False)  # Disabled until selection
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: 1px solid #d32f2f;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
                border: 1px solid #c62828;
            }
            QPushButton:pressed {
                background-color: #c62828;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        # Connect selection change
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        
        self.setLayout(layout)
    
    def on_selection_changed(self):
        """Enable/disable select button based on selection."""
        has_selection = len(self.list_widget.selectedItems()) > 0
        self.select_button.setEnabled(has_selection)
    
    def accept_selection(self):
        """Accept the current selection."""
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            self.selected_folder = selected_items[0].data(Qt.ItemDataRole.UserRole)
            self.accept()

class TestImageManager:
    """Manager class for handling test image generation."""
    
    def __init__(self, main_window):
        """Initialize with reference to the main window."""
        self.main_window = main_window
        self.test_job_folders = []
        self.current_test_index = 0
    
    def test_image(self):
        """Show folder selection dialog for 'test images' directory and process selected folder."""
        try:
            base_dir = 'test images'
            
            # Check if the base directory exists
            if not os.path.exists(base_dir):
                self.main_window.log_widget.add_log(f"Directory '{base_dir}' not found.")
                QMessageBox.warning(
                    self.main_window, 
                    "Directory Not Found", 
                    f"The '{base_dir}' directory was not found.\n\nPlease create this directory and add test image folders to it."
                )
                return
            
            # Find all immediate subdirectories containing data.json
            subfolders = [f for f in glob.glob(os.path.join(base_dir, '*')) 
                         if os.path.isdir(f) and os.path.exists(os.path.join(f, 'data.json'))]

            if not subfolders:
                self.main_window.log_widget.add_log(f"No test job folders found in '{base_dir}'.")
                QMessageBox.information(
                    self.main_window, 
                    "No Test Folders", 
                    f"No test job folders found in '{base_dir}'.\n\nPlease add folders containing 'data.json' files to this directory."
                )
                return

            # Sort for deterministic order
            subfolders = sorted(subfolders)
            
            # Show custom folder selection dialog
            dialog = TestFolderSelectionDialog(
                self.main_window,
                subfolders,
                "Select Test Image Folder"
            )
            
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_folder:
                selected_folder = dialog.selected_folder
            else:
                # User cancelled the dialog
                return
            
            # Set up the test job with only the selected folder
            self.test_job_folders = [selected_folder]
            self.current_test_index = 0

            # Disable button
            self.main_window.test_image_button.setEnabled(False)

            self.start_next_test_image_job()

        except Exception as e:
            self.main_window.log_widget.add_log(f"Failed to initiate test jobs: {str(e)}")
            self.main_window.log_widget.add_log(traceback.format_exc())

    def start_next_test_image_job(self):
        """Load data for the current test index and use existing workflow."""
        if self.current_test_index >= len(self.test_job_folders):
            # All jobs done
            self.main_window.test_image_button.setEnabled(True)
            self.main_window.test_image_button.setText("Test image")
            self.main_window.log_widget.add_log("All test jobs completed.")
            self.test_job_folders = []  # Clear the list to exit test mode
            return

        folder = self.test_job_folders[self.current_test_index]
        self.main_window.log_widget.add_log(f"Starting test job {self.current_test_index + 1}/{len(self.test_job_folders)}: {folder}")

        try:
            # Load JSON
            with open(os.path.join(folder, 'data.json'), 'r', encoding='utf-8') as json_file:
                json_data = json.load(json_file)

            # Ensure this is treated as an image job (default behavior)
            json_data['job_type'] = json_data.get('job_type', 'image')
            
            # Apply vertical_video resolution swap if enabled
            json_data = apply_vertical_video_swap(json_data, self.main_window.log_widget.add_log)

            # Load GPX files using shared utility function
            zip_path = os.path.join(folder, 'gpx_files.zip')
            gpx_files_info = load_gpx_files_from_zip(zip_path, log_callback=self.main_window.log_widget.add_log)

            if not gpx_files_info:
                self.main_window.log_widget.add_log("No valid GPX files found in the ZIP")
                # Move to next job on failure
                self.current_test_index += 1
                self.start_next_test_image_job()
                return
            
            # Update button text before starting job
            self.main_window.test_image_button.setText(f"Processing {self.current_test_index + 1}/{len(self.test_job_folders)}")

            # Use the existing workflow by calling on_job_received
            self.main_window.job_request_manager.on_job_received(json_data, gpx_files_info)

        except Exception as e:
            self.main_window.log_widget.add_log(f"Failed to start job in {folder}: {str(e)}")
            self.main_window.log_widget.add_log(traceback.format_exc())
            # Move to next job on failure
            self.current_test_index += 1
            self.start_next_test_image_job()
    
    def on_worker_finished(self):
        """Handle completion and move to next test job if available."""
        # Move to next test job
        self.current_test_index += 1
        self.start_next_test_image_job()

    def on_worker_error(self, error_msg):
        """Handle worker errors and proceed to next test job."""
        # Move to next test job despite error
        self.current_test_index += 1
        self.start_next_test_image_job() 