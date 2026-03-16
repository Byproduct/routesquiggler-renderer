#!/usr/bin/env python3
"""
Configuration variables set by the user in config.txt.
"""

import os


class Config:
    
    def __init__(self):
        # Set default values
        self.gui = False
        self.app_version = None
        self.api_url = None
        self.user = None
        self.storage_box_address = None
        self.storage_box_user = None
        self.storage_box_password = None
        self.debug_logging = False
        self.text_effect = False
        self.sync_map_tile_cache = True
        self.gpu_rendering = True
        self.low_spec = False
        self.map_tile_cache_folder = "default"
        # None means unlimited (no pruning); otherwise a number in gigabytes
        self.map_tile_cache_size = None
        # Thread count for rendering: None = "auto" (max(1, cpu_count - 2)), or int for fixed value
        self.thread_count = None
        self._thread_count_from_file = False
        self._base_dir = os.path.dirname(os.path.abspath(__file__))

    @property
    def map_tile_cache_path(self):
        """Resolved absolute path to the map tile cache folder."""
        if self.map_tile_cache_folder.lower() == "default" or not self.map_tile_cache_folder:
            return os.path.join(self._base_dir, "map tile cache")
        return self.map_tile_cache_folder
    
    def load_from_file(self, config_path):
        """Load configuration from a file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Set attributes based on config
                        if key == 'gui':
                            self.gui = value.lower() == 'true'
                        elif key == 'api_url':
                            self.api_url = value
                        elif key == 'user':
                            self.user = value
                        elif key == 'storage_box_address':
                            self.storage_box_address = value
                        elif key == 'storage_box_user':
                            self.storage_box_user = value
                        elif key == 'storage_box_password':
                            self.storage_box_password = value
                        elif key == 'debug_logging':
                            self.debug_logging = value.lower() == 'true'
                        elif key == 'text_effect':
                            self.text_effect = value.lower() == 'true'
                        elif key == 'sync_map_tile_cache':
                            self.sync_map_tile_cache = value.lower() == 'true'
                        elif key == 'gpu_rendering':
                            self.gpu_rendering = value.lower() == 'true'
                        elif key == 'low_spec':
                            self.low_spec = value.lower() == 'true'
                        elif key == 'map_tile_cache_folder':
                            self.map_tile_cache_folder = value
                        elif key == 'map_tile_cache_size':
                            val = value.strip().lower()
                            if val in ('', 'unlimited'):
                                self.map_tile_cache_size = None
                            else:
                                try:
                                    self.map_tile_cache_size = float(val)
                                except ValueError:
                                    print(f"Warning: invalid map_tile_cache_size '{value}' in config.txt; treating as unlimited.")
                                    self.map_tile_cache_size = None
                        elif key == 'threads':
                            val = value.strip().lower()
                            if val in ('', 'auto'):
                                self.thread_count = None
                            else:
                                try:
                                    n = int(val)
                                    if n < 1:
                                        print("Warning: threads in config.txt must be >= 1; using 'auto'.")
                                        self.thread_count = None
                                    else:
                                        self.thread_count = n
                                except ValueError:
                                    print(f"Warning: invalid threads value '{value}' in config.txt; using 'auto'.")
                                    self.thread_count = None
                        elif key == 'leave_temporary_files':
                            self.leave_temporary_files = value.lower() == 'true'
        except FileNotFoundError:
            print("Warning: config.txt not found. Using default configuration.")
        except Exception as e:
            print(f"Warning: Error loading config.txt: {e}. Using default configuration.")


def load_config():
    """Load configuration from config.txt file and return Config object."""
    config = Config()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config.load_from_file(os.path.join(base_dir, 'config.txt'))
    try:
        with open(os.path.join(base_dir, 'app_version.txt'), 'r', encoding='utf-8') as f:
            config.app_version = f.read().strip()
    except FileNotFoundError:
        print("Warning: app_version.txt not found.")
    return config


# Global config instance
config = load_config()
