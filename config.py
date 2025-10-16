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
        self.sync_map_tile_cache = True
        self.gpu_rendering = True
    
    def load_from_file(self, config_path):
        """Load configuration from a file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line and ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Set attributes based on config
                        if key == 'gui':
                            self.gui = value.lower() == 'true'
                        elif key == 'app_version':
                            self.app_version = value
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
                        elif key == 'sync_map_tile_cache':
                            self.sync_map_tile_cache = value.lower() == 'true'
                        elif key == 'gpu_rendering':
                            self.gpu_rendering = value.lower() == 'true'
        except FileNotFoundError:
            print("Warning: config.txt not found. Using default configuration.")
        except Exception as e:
            print(f"Warning: Error loading config.txt: {e}. Using default configuration.")


def load_config():
    """Load configuration from config.txt file and return Config object."""
    config = Config()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.txt')
    config.load_from_file(config_path)
    return config


# Global config instance
config = load_config()
