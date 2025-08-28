import os
import sys
import io
from datetime import datetime
from contextlib import contextmanager

class DebugLogger:
    """Debug logging system that captures console and UI logs to files."""
    
    def __init__(self):
        self.debug_enabled = False
        self.console_log_file = None
        self.ui_log_file = None
        self.original_stdout = None
        self.original_stderr = None
        self.console_buffer = io.StringIO()
        
    def setup_logging(self, debug_enabled):
        """Setup debug logging if enabled."""
        self.debug_enabled = debug_enabled
        
        if not self.debug_enabled:
            return
            
        # Create log directory if it doesn't exist
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Setup console logging
        self.console_log_file = open(os.path.join(log_dir, "debug_console.log"), "w", encoding="utf-8")
        self.ui_log_file = open(os.path.join(log_dir, "debug_UI.log"), "w", encoding="utf-8")
        
        # Write initial log entry
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self.console_log_file.write(f"{timestamp} Debug logging started\n")
        self.ui_log_file.write(f"{timestamp} Debug logging started\n")
        
        # Note: We're not redirecting stdout/stderr anymore to avoid multiprocessing issues
        # Console output will be captured through explicit logging calls instead
        
    def log_console_message(self, message):
        """Log a console message to the debug console log file."""
        if self.debug_enabled and self.console_log_file:
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            self.console_log_file.write(f"{timestamp} {message}\n")
            self.console_log_file.flush()
        
    def log_ui_message(self, message):
        """Log a UI message to the debug UI log file."""
        if self.debug_enabled and self.ui_log_file:
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            self.ui_log_file.write(f"{timestamp} {message}\n")
            self.ui_log_file.flush()
            
    def cleanup(self):
        """Clean up logging resources."""
        if self.debug_enabled:
            if self.console_log_file:
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                self.console_log_file.write(f"{timestamp} Debug logging stopped\n")
                self.console_log_file.close()
                
            if self.ui_log_file:
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                self.ui_log_file.write(f"{timestamp} Debug logging stopped\n")
                self.ui_log_file.close()

# Global instance
debug_logger = DebugLogger()

def setup_debug_logging(debug_enabled):
    """Setup debug logging globally."""
    debug_logger.setup_logging(debug_enabled)
    
def log_console_message(message):
    """Log a console message to debug log."""
    debug_logger.log_console_message(message)
    
def log_ui_message(message):
    """Log a UI message to debug log."""
    debug_logger.log_ui_message(message)
    
def cleanup_debug_logging():
    """Clean up debug logging."""
    debug_logger.cleanup()

# Monkey patch print function to also log to debug console
original_print = print

def debug_print(*args, **kwargs):
    """Enhanced print function that also logs to debug console."""
    # Call original print
    original_print(*args, **kwargs)
    
    # Also log to debug console if enabled
    if debug_logger.debug_enabled:
        message = " ".join(str(arg) for arg in args)
        log_console_message(message)

# Replace print function globally
import builtins
builtins.print = debug_print 