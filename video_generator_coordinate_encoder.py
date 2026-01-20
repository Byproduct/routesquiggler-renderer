# Encode map bounds (lon_min, lon_max, lat_min, lat_max) into base64. Used to create filenames for saved map images.

# Standard library imports
import base64
import struct

def encode_coords(lon_min, lon_max, lat_min, lat_max):
    binary = struct.pack('ffff', lon_min, lon_max, lat_min, lat_max)
    return base64.urlsafe_b64encode(binary).decode('ascii').rstrip('=')

def decode_coords(encoded):
    padding = '=' * (-len(encoded) % 4)
    binary = base64.urlsafe_b64decode(encoded + padding)
    return struct.unpack('ffff', binary)