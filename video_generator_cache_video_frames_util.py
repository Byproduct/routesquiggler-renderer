"""
Utility functions and classes for video frame caching.
Used by video_generator_cache_video_frames.py.
"""

from contextlib import contextmanager
import os
import sys
from io import StringIO

import imageio_ffmpeg

from config import config


def get_ffmpeg_executable():
    """
    Get the correct ffmpeg executable path for the current platform.
    Uses the same ffmpeg binary that MoviePy uses via imageio_ffmpeg for consistency.

    Returns:
        str: Path to ffmpeg executable
    """
    # First, try to use the same ffmpeg that MoviePy/imageio uses
    # This ensures consistency with video generation
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
    except Exception:
        # If imageio_ffmpeg fails for any reason, fall back to platform defaults
        pass

    # Fallback: platform-specific defaults
    if os.name == 'nt':  # Windows
        # Check if ffmpeg.exe exists in the project root (same directory as this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_ffmpeg = os.path.join(script_dir, 'ffmpeg.exe')
        if os.path.exists(local_ffmpeg):
            return local_ffmpeg
        # Fallback to system ffmpeg.exe (should be in PATH)
        return 'ffmpeg.exe'
    else:  # Linux/Mac
        # On Linux/Mac, ffmpeg should be in PATH
        return 'ffmpeg'


class MoviePyDebugLogger:
    """Custom logger for MoviePy that only outputs when debug logging is enabled."""

    def __init__(self, debug_callback=None):
        self.debug_callback = debug_callback

    def message(self, message):
        """Log MoviePy messages only when debug is enabled."""
        if config.debug_logging and self.debug_callback:
            self.debug_callback(f"MoviePy: {message}")

    def error(self, message):
        """Log MoviePy errors only when debug is enabled."""
        if config.debug_logging and self.debug_callback:
            self.debug_callback(f"MoviePy error: {message}")

    def warning(self, message):
        """Log MoviePy warnings only when debug is enabled."""
        if config.debug_logging and self.debug_callback:
            self.debug_callback(f"MoviePy warning: {message}")


@contextmanager
def suppress_moviepy_output():
    """Context manager to suppress MoviePy stdout/stderr output when debug is off."""
    if not config.debug_logging:
        # Redirect stdout and stderr to devnull when debug is off
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    else:
        # When debug is on, don't suppress anything
        yield
