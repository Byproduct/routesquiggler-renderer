# Reworked existing UI logging to use write_log.py.

import os
import sys

class DebugLogger:
    """Debug logging system that routes console and UI logs to centralized logging."""
    
    def __init__(self):
        self.debug_enabled = False
        
    def setup_logging(self, debug_enabled):
        """Setup debug logging if enabled."""
        self.debug_enabled = debug_enabled
        
    def log_console_message(self, message):
        if self.debug_enabled:
            try:
                from write_log import write_debug_log
                write_debug_log(f"console message: {message}")
            except ImportError:
                pass
        
    def log_ui_message(self, message):
        if self.debug_enabled:
            try:
                from write_log import write_debug_log
                write_debug_log(f"UI message: {message}")
            except ImportError:
                pass
            
    def cleanup(self):
        """Clean up logging resources."""
        if self.debug_enabled:
            try:
                from write_log import write_debug_log
                write_debug_log("Debug logging stopped")
            except ImportError:
                pass

# Global instance
debug_logger = DebugLogger()

def setup_debug_logging(debug_enabled):
    debug_logger.setup_logging(debug_enabled)
    
def log_console_message(message):
    debug_logger.log_console_message(message)
    
def log_ui_message(message):
    debug_logger.log_ui_message(message)
    
def cleanup_debug_logging():
    debug_logger.cleanup()