from datetime import datetime
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal
from debug_logger import log_ui_message


class SelectableTextEdit(QTextEdit):
    """Custom QTextEdit that allows text selection and copying"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(80)  # Height for 3 lines
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Enable text selection and interaction
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | 
                                   Qt.TextInteractionFlag.TextSelectableByKeyboard)


class LogWidget(QWidget):
    """Widget that handles log display with expandable functionality"""
    
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.log_state = "expanded"  # Can be "collapsed", "normal", or "expanded"
        self.normal_sizes = None
        self.max_lines = 1000  # Maximum number of log lines to keep
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px;
                font-family: Consolas, monospace;
                font-size: 12px;
                color: #cccccc;
            }
        """)
        
        # Create header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Add title
        title = QLabel("Log")
        title.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #ffffff;
                background-color: transparent;
            }
        """)
        header_layout.addWidget(title)
        
        # Add spacer
        header_layout.addStretch()
        
        # Add toggle button
        self.toggle_button = QPushButton("▼")  # Down arrow
        self.toggle_button.setFixedSize(24, 24)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #cccccc;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #ffffff;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_log_size)
        header_layout.addWidget(self.toggle_button)
        
        # Add widgets to layout
        layout.addWidget(header)
        layout.addWidget(self.log_text)
        
        # Initialize timestamp for log entries
        self.last_timestamp = None
    
    def set_normal_sizes(self, sizes):
        """Store the normal sizes for the splitter"""
        self.normal_sizes = sizes
    
    def toggle_log_size(self):
        """Toggle between collapsed and expanded log view"""
        if not self.parent_window.splitter:
            return
        
        central_widget = self.parent_window.centralWidget().widget(0)  # Get the central widget from splitter
        
        if self.log_state == "expanded":
            # Go to normal state (10 lines, approximately 200px)
            self.toggle_button.setText("▲")  # Up arrow
            self.parent_window.splitter.setSizes([self.normal_sizes[0], 200])  # Show 10 lines (200px)
            central_widget.show()
            self.log_state = "collapsed"
        else:  # collapsed
            # Go to expanded state (full height)
            self.toggle_button.setText("▼")  # Down arrow
            self.parent_window.splitter.setSizes([0, self.height()])
            central_widget.hide()
            self.log_state = "expanded"
    
    def add_log(self, message):
        """Add a timestamped message to the log"""
        # Get current timestamp
        now = datetime.now()
        timestamp = now.strftime("[%H:%M:%S]")
        
        # Add message with timestamp
        self.log_text.append(f"{timestamp} {message}")
        
        # Also log to debug file if enabled
        log_ui_message(message)
        
        # Check if we need to limit the number of lines
        self._enforce_line_limit()
        
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _enforce_line_limit(self):
        """Remove oldest lines if we exceed the maximum line limit"""
        document = self.log_text.document()
        line_count = document.blockCount()
        
        if line_count > self.max_lines:
            # Calculate how many lines to remove
            lines_to_remove = line_count - self.max_lines
            
            # Get the cursor and select the lines to remove from the beginning
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            
            # Select lines to remove
            for _ in range(lines_to_remove):
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                cursor.movePosition(cursor.MoveOperation.NextBlock, cursor.MoveMode.KeepAnchor)
            
            # Remove the selected text
            cursor.removeSelectedText()
    
    def set_max_lines(self, max_lines):
        """Set the maximum number of lines to keep in the log"""
        self.max_lines = max_lines
        self._enforce_line_limit()  # Apply the new limit immediately 