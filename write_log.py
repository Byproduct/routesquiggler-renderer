#!/usr/bin/env python3
"""
Appends program messages to daily log files.
"""

# Standard library imports
import os
import queue
import sys
import threading
from datetime import datetime

# ---------------------------------------------------------------------------
# Background TTE renderer
# Messages are queued and played one at a time by a daemon thread, so the
# calling code never blocks waiting for an animation to finish.
# ---------------------------------------------------------------------------
_tte_queue: queue.Queue = queue.Queue()


def _tte_worker() -> None:
    """Daemon thread: drain the queue and play each TTE animation in order."""
    while True:
        message = _tte_queue.get()
        try:
            from terminaltexteffects.effects.effect_print import Print
            effect = Print(message)
            effect.effect_config.print_speed = 8
            effect.effect_config.print_head_return_speed = 16
            with effect.terminal_output() as terminal:
                for frame in effect:
                    terminal.print(frame)
        except Exception:
            print(message)
        finally:
            _tte_queue.task_done()


_tte_thread = threading.Thread(target=_tte_worker, daemon=True)
_tte_thread.start()


def _print_with_tte(message: str) -> None:
    """
    Enqueue a message for the TTE Print effect (typewriter animation).
    Returns immediately; the animation plays in a background thread.
    Falls back to a plain print if text_effect is disabled or output is not a TTY.
    """
    from config import config
    if not config.text_effect or not sys.stdout.isatty():
        print(message)
        return
    _tte_queue.put(message)


def _write_to_log_file(message: str) -> None:
    """Append a timestamped message to today's daily log file."""
    try:
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{today}.txt")
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")


def write_log(message):
    """
    Write a message to both console and daily log file.

    Args:
        message (str): The message to log
    """
    print(message)
    _write_to_log_file(message)


def write_debug_log(message):
    """
    Write a debug message if debug logging is enabled.
    Console output uses the TTE Print effect (typewriter animation) when running
    in a real terminal; falls back to plain print otherwise.
    """
    from config import config
    if config.debug_logging:
        full_message = f"Debug: {message}"
        _write_to_log_file(full_message)
        _print_with_tte(full_message)