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


def make_map_cache_key(bbox, zoom_level):
    """
    Composite in-memory map-image cache key.

    Video rendering keys cached map images by both bounding box and zoom
    level. Including zoom is necessary because follow-mode zoom stabilization
    can assign two different zoom levels to the same bbox at different frames
    (rare but possible), and because different video modes may render the
    same bbox at different zooms. Using only the bbox would silently produce
    wrong-zoom map images for some frames.

    Args:
        bbox: Tuple (lon_min, lon_max, lat_min, lat_max) in GPS degrees.
        zoom_level: Integer zoom level.

    Returns:
        str: Cache key combining the base64 bbox encoding with the zoom level.
    """
    return f"{encode_coords(*bbox)}|z{int(zoom_level)}"