"""
Map tile utilities for handling different map styles and zoom levels.
"""

# Standard library imports
import os
import threading
import time
from datetime import datetime
from typing import List, Tuple, Optional

# Third-party imports
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import numpy as np

STADIA_API_KEY = "2413a338-8b10-4302-a96c-439cb795b285"
GEOAPIFY_API_KEY = "1180c192019b43b9a366c498deadcc4b"
THUNDERFOREST_API_KEY = "43413e30756b4dd489d7e62d5c0a9245"
LOGGING_INTO_FILE = False   #render.log

# Service-specific rate limits (seconds between tile download requests).
# Used when fetching tiles; if a request takes longer than the delay, the wait is skipped.
RATE_LIMIT_DEFAULT = 0.05   # OSM, Stadia, and any unspecified service
RATE_LIMIT_OTM = 0.10      # OpenTopoMap
RATE_LIMIT_CYCLOSM = 0.10  # CyclOSM
RATE_LIMIT_GEOAPIFY = 0.20 # Geoapify
RATE_LIMIT_THUNDERFOREST = 0.01  # Thunderforest

# User-Agent for free OSM-based tile services (OSM, OpenTopoMap, CyclOSM).
# Required by OpenStreetMap tile usage policy: https://operations.osmfoundation.org/policies/tiles/
FREE_TILE_USER_AGENT = "RouteSquiggler (+https://routesquiggler.com; contact: juhgu@hotmail.com)"

# Thread-safe rate limiter for tile downloads
_tile_download_lock = threading.Lock()
_last_tile_download_time = 0.0

# Import the debug flag from the main file
try:
    from video_generator_cache_map_tiles import MAP_TILE_CACHING_DEBUG
except ImportError:
    # Fallback if the import fails
    MAP_TILE_CACHING_DEBUG = False

def debug_log(message: str):
    if LOGGING_INTO_FILE:
        """Write debug messages to a log file with timestamp."""
        try:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(current_directory, 'log', "render.log")
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except:
            pass

def get_rate_limit_delay(map_style: str) -> float:
    """Return the rate limit delay in seconds for the given map style."""
    if map_style == "otm":
        return RATE_LIMIT_OTM
    if map_style == "cyclosm":
        return RATE_LIMIT_CYCLOSM
    if map_style.startswith("geoapify_"):
        return RATE_LIMIT_GEOAPIFY
    if map_style.startswith("thunderforest_"):
        return RATE_LIMIT_THUNDERFOREST
    return RATE_LIMIT_DEFAULT


def _rate_limit_tile_download(delay: Optional[float] = None):
    """
    Thread-safe rate limiter for tile downloads.
    Ensures at least `delay` seconds between tile download requests.
    If delay is None, uses RATE_LIMIT_DEFAULT. Only affects actual downloads (not cached).
    """
    if delay is None:
        delay = RATE_LIMIT_DEFAULT
    global _last_tile_download_time
    with _tile_download_lock:
        current_time = time.time()
        time_since_last = current_time - _last_tile_download_time
        if time_since_last < delay:
            time.sleep(delay - time_since_last)
        _last_tile_download_time = time.time()

def set_cache_directory(map_style: str):
    """Set the appropriate cache directory for the given map style."""
    debug_log(f"set_cache_directory called with map_style: '{map_style}'")
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    base_cache_dir = os.path.join(current_directory, 'map tile cache')
    
    # Map styles to their cache subdirectories
    style_cache_mapping = {
        'osm': 'OSM',
        'otm': 'OpenTopoMap',
        'cyclosm': 'CyclOSM',
        'stadia_light': 'StadiaLight',
        'stadia_dark': 'StadiaDark',
        'stadia_outdoors': 'StadiaOutdoors',
        'stadia_toner': 'StadiaToner',
        'stadia_watercolor': 'StadiaWatercolor',
        'geoapify_carto': 'GeoapifyCarto',
        'geoapify_bright': 'GeoapifyBright',
        'geoapify_bright_grey': 'GeoapifyBrightGrey',
        'geoapify_bright_smooth': 'GeoapifyBrightSmooth',
        'geoapify_klockantech': 'GeoapifyKlokantech',
        'geoapify_liberty': 'GeoapifyLiberty',
        'geoapify_maptiler': 'GeoapifyMaptiler',
        'geoapify_toner': 'GeoapifyToner',
        'geoapify_toner_grey': 'GeoapifyTonerGrey',
        'geoapify_positron': 'GeoapifyPositron',
        'geoapify_positron_blue': 'GeoapifyPositronBlue',
        'geoapify_positron_red': 'GeoapifyPositronRed',
        'geoapify_dark': 'GeoapifyDark',
        'geoapify_dark_brown': 'GeoapifyDarkBrown',
        'geoapify_grey': 'GeoapifyGrey',
        'geoapify_purple': 'GeoapifyPurple',
        'geoapify_purple_roads': 'GeoapifyPurpleRoads',
        'geoapify_yellow_roads': 'GeoapifyYellowRoads',
        'thunderforest_atlas': 'ThunderforestAtlas',
        'thunderforest_mobile_atlas': 'ThunderforestMobileAtlas',
        'thunderforest_cycle': 'ThunderforestCycle',
        'thunderforest_landscape': 'ThunderforestLandscape',
        'thunderforest_neighbourhood': 'ThunderforestNeighbourhood',
        'thunderforest_outdoors': 'ThunderforestOutdoors',
        'thunderforest_pioneer': 'ThunderforestPioneer',
        'thunderforest_spinal': 'ThunderforestSpinal',
        'thunderforest_transport': 'ThunderforestTransport',
        'thunderforest_transport_dark': 'ThunderforestTransportDark'
    }
    
    cache_subdir = style_cache_mapping.get(map_style, 'OSM')
    cache_dir = os.path.join(base_cache_dir, cache_subdir)
    
    debug_log(f"Cache subdirectory mapping: {cache_subdir}")
    debug_log(f"Full cache directory path: {cache_dir}")
    
    # Create the directory if it doesn't exist
    try:
        os.makedirs(cache_dir, exist_ok=True)
        debug_log(f"Successfully created/verified cache directory: {cache_dir}")
    except Exception as e:
        debug_log(f"Error creating cache directory: {e}")
    
    # Set the cartopy cache directory
    cartopy.config['cache_dir'] = cache_dir
    debug_log(f"Set cartopy.config['cache_dir'] to: {cartopy.config['cache_dir']}")
    if MAP_TILE_CACHING_DEBUG:
        print(f"Set cache directory to: {cache_dir}")

def create_map_tiles(map_style: str):
    """
    Create map tiles based on the selected style.
    
    Args:
        map_style: String, one of 'osm', 'otm', 'cyclosm', 'stadia_light', 'stadia_dark', 
                  'stadia_outdoors', 'stadia_toner', 'stadia_watercolor', or various 'geoapify_*' styles
    
    Returns:
        CartoPy image tiles object
    """
    debug_log(f"create_map_tiles called with map_style: '{map_style}'")
    
    # Stadia style URL mapping
    stadia_style_url_mapping = {
        "stadia_light": "alidade_smooth",
        "stadia_dark": "alidade_smooth_dark", 
        "stadia_outdoors": "outdoors",
        "stadia_toner": "stamen_toner",
        "stadia_watercolor": "stamen_watercolor"
    }
    
    # Geoapify style URL mapping (map_style -> tile style name in URL)
    geoapify_style_url_mapping = {
        "geoapify_carto": "osm-carto",
        "geoapify_bright": "osm-bright",
        "geoapify_bright_grey": "osm-bright-grey",
        "geoapify_bright_smooth": "osm-bright-smooth",
        "geoapify_klockantech": "klokantech-basic",
        "geoapify_liberty": "osm-liberty",
        "geoapify_maptiler": "maptiler-3d",
        "geoapify_toner": "toner",
        "geoapify_toner_grey": "toner-grey",
        "geoapify_positron": "positron",
        "geoapify_positron_blue": "positron-blue",
        "geoapify_positron_red": "positron-red",
        "geoapify_dark": "dark-matter",
        "geoapify_dark_brown": "dark-matter-brown",
        "geoapify_grey": "dark-matter-dark-grey",
        "geoapify_purple": "dark-matter-dark-purple",
        "geoapify_purple_roads": "dark-matter-purple-roads",
        "geoapify_yellow_roads": "dark-matter-yellow-roads"
    }
    
    # Thunderforest style URL mapping (map_style -> tile style name in URL)
    thunderforest_style_url_mapping = {
        "thunderforest_atlas": "atlas",
        "thunderforest_mobile_atlas": "mobile-atlas",
        "thunderforest_cycle": "cycle",
        "thunderforest_landscape": "landscape",
        "thunderforest_neighbourhood": "neighbourhood",
        "thunderforest_outdoors": "outdoors",
        "thunderforest_pioneer": "pioneer",
        "thunderforest_spinal": "spinal-map",
        "thunderforest_transport": "transport",
        "thunderforest_transport_dark": "transport-dark"
    }
    
    debug_log(f"Available Stadia styles: {list(stadia_style_url_mapping.keys())}")
    
    if map_style.startswith('stadia_'):
        debug_log(f"Processing Stadia map style: {map_style}")
        tile_delay = get_rate_limit_delay(map_style)
        if map_style in stadia_style_url_mapping:
            style = stadia_style_url_mapping[map_style]
            debug_log(f"Mapped {map_style} to Stadia style: {style}")
            
            class StadiaTiles(cimgt.GoogleTiles):
                # Set explicit name to ensure consistent cache subdirectory naming
                _name = 'StadiaTiles'
                
                def _image_url(self, tile):
                    x, y, z = tile
                    url = f'https://tiles.stadiamaps.com/tiles/{style}/{z}/{x}/{y}.png?api_key={STADIA_API_KEY}'
                    debug_log(f"Generated Stadia URL: {url}")
                    return url
                
                def get_image(self, tile):
                    _rate_limit_tile_download(tile_delay)
                    return super().get_image(tile)
            
            tiles = StadiaTiles(cache=True)
            debug_log(f"Created StadiaTiles object: {type(tiles)}")
            return tiles
        else:
            debug_log(f"Unknown Stadia style: {map_style}, defaulting to stadia_light")
            style = stadia_style_url_mapping["stadia_light"]
            
            class StadiaTiles(cimgt.GoogleTiles):
                # Set explicit name to ensure consistent cache subdirectory naming
                _name = 'StadiaTiles'
                
                def _image_url(self, tile):
                    x, y, z = tile
                    url = f'https://tiles.stadiamaps.com/tiles/{style}/{z}/{x}/{y}.png?api_key={STADIA_API_KEY}'
                    debug_log(f"Generated fallback Stadia URL: {url}")
                    return url
                
                def get_image(self, tile):
                    _rate_limit_tile_download(tile_delay)
                    return super().get_image(tile)
            
            tiles = StadiaTiles(cache=True)
            debug_log(f"Created fallback StadiaTiles object: {type(tiles)}")
            return tiles
            
    elif map_style == "otm":
        debug_log("Processing OpenTopoMap style")
        tile_delay = get_rate_limit_delay(map_style)
        
        class OpenTopoMapTiles(cimgt.OSM):
            def _image_url(self, tile):
                x, y, z = tile
                url = f"https://a.tile.opentopomap.org/{z}/{x}/{y}.png"
                debug_log(f"Generated OTM URL: {url}")
                return url
            
            def get_image(self, tile):
                _rate_limit_tile_download(tile_delay)
                return super().get_image(tile)
        
        tiles = OpenTopoMapTiles(cache=True, user_agent=FREE_TILE_USER_AGENT)
        debug_log(f"Created OpenTopoMapTiles object: {type(tiles)}")
        return tiles
        
    elif map_style == "cyclosm":
        debug_log("Processing CyclOSM style")
        tile_delay = get_rate_limit_delay(map_style)
        
        class CyclOSMTiles(cimgt.OSM):
            def _image_url(self, tile):
                x, y, z = tile
                url = f"https://a.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png"
                debug_log(f"Generated CyclOSM URL: {url}")
                return url
            
            def get_image(self, tile):
                _rate_limit_tile_download(tile_delay)
                return super().get_image(tile)
        
        tiles = CyclOSMTiles(cache=True, user_agent=FREE_TILE_USER_AGENT)
        debug_log(f"Created CyclOSMTiles object: {type(tiles)}")
        return tiles
        
    elif map_style.startswith('geoapify_'):
        debug_log(f"Processing Geoapify map style: {map_style}")
        tile_delay = get_rate_limit_delay(map_style)
        if map_style in geoapify_style_url_mapping:
            tile_style = geoapify_style_url_mapping[map_style]
            debug_log(f"Mapped {map_style} to Geoapify tile style: {tile_style}")
            
            class GeoapifyTiles(cimgt.GoogleTiles):
                def _image_url(self, tile):
                    x, y, z = tile
                    url = f'https://maps.geoapify.com/v1/tile/{tile_style}/{z}/{x}/{y}.png?apiKey={GEOAPIFY_API_KEY}'
                    debug_log(f"Generated Geoapify URL: {url}")
                    return url
                
                def get_image(self, tile):
                    _rate_limit_tile_download(tile_delay)
                    return super().get_image(tile)
            
            tiles = GeoapifyTiles(cache=True)
            debug_log(f"Created GeoapifyTiles object: {type(tiles)}")
            return tiles
        else:
            debug_log(f"Unknown Geoapify style: {map_style}, defaulting to geoapify_carto")
            tile_style = geoapify_style_url_mapping["geoapify_carto"]
            
            class GeoapifyTiles(cimgt.GoogleTiles):
                def _image_url(self, tile):
                    x, y, z = tile
                    url = f'https://maps.geoapify.com/v1/tile/{tile_style}/{z}/{x}/{y}.png?apiKey={GEOAPIFY_API_KEY}'
                    debug_log(f"Generated fallback Geoapify URL: {url}")
                    return url
                
                def get_image(self, tile):
                    _rate_limit_tile_download(tile_delay)
                    return super().get_image(tile)
            
            tiles = GeoapifyTiles(cache=True)
            debug_log(f"Created fallback GeoapifyTiles object: {type(tiles)}")
            return tiles
        
    elif map_style.startswith('thunderforest_'):
        debug_log(f"Processing Thunderforest map style: {map_style}")
        tile_delay = get_rate_limit_delay(map_style)
        if map_style in thunderforest_style_url_mapping:
            tile_style = thunderforest_style_url_mapping[map_style]
            debug_log(f"Mapped {map_style} to Thunderforest tile style: {tile_style}")
            
            class ThunderforestTiles(cimgt.GoogleTiles):
                def _image_url(self, tile):
                    x, y, z = tile
                    url = f'https://tile.thunderforest.com/{tile_style}/{z}/{x}/{y}.png?apikey={THUNDERFOREST_API_KEY}'
                    debug_log(f"Generated Thunderforest URL: {url}")
                    return url
                
                def get_image(self, tile):
                    _rate_limit_tile_download(tile_delay)
                    return super().get_image(tile)
            
            tiles = ThunderforestTiles(cache=True)
            debug_log(f"Created ThunderforestTiles object: {type(tiles)}")
            return tiles
        else:
            debug_log(f"Unknown Thunderforest style: {map_style}, defaulting to thunderforest_atlas")
            tile_style = thunderforest_style_url_mapping["thunderforest_atlas"]
            
            class ThunderforestTiles(cimgt.GoogleTiles):
                def _image_url(self, tile):
                    x, y, z = tile
                    url = f'https://tile.thunderforest.com/{tile_style}/{z}/{x}/{y}.png?apikey={THUNDERFOREST_API_KEY}'
                    debug_log(f"Generated fallback Thunderforest URL: {url}")
                    return url
                
                def get_image(self, tile):
                    _rate_limit_tile_download(tile_delay)
                    return super().get_image(tile)
            
            tiles = ThunderforestTiles(cache=True)
            debug_log(f"Created fallback ThunderforestTiles object: {type(tiles)}")
            return tiles
        
    elif map_style == "osm":
        debug_log("Processing OpenStreetMap style")
        tile_delay = get_rate_limit_delay(map_style)
        
        class RateLimitedOSM(cimgt.OSM):
            def get_image(self, tile):
                _rate_limit_tile_download(tile_delay)
                return super().get_image(tile)
        
        tiles = RateLimitedOSM(cache=True, user_agent=FREE_TILE_USER_AGENT)
        debug_log(f"Created OSM tiles object: {type(tiles)}")
        return tiles
        
    else:
        debug_log(f"Invalid map style '{map_style}', defaulting to OSM")
        print(f"Invalid map style '{map_style}', defaulting to OSM")
        tile_delay = get_rate_limit_delay(map_style)
        
        class RateLimitedOSM(cimgt.OSM):
            def get_image(self, tile):
                _rate_limit_tile_download(tile_delay)
                return super().get_image(tile)
        
        tiles = RateLimitedOSM(cache=True, user_agent=FREE_TILE_USER_AGENT)
        debug_log(f"Created default OSM tiles object: {type(tiles)}")
        return tiles

def calculate_tile_count(map_bounds: Tuple[float, float, float, float], zoom_level: int) -> int:
    """
    Calculate the number of map tiles needed for a given geographic area at a specific zoom level.
    
    Args:
        map_bounds: Tuple (lon_min, lon_max, lat_min, lat_max) in decimal degrees
        zoom_level: Zoom level to calculate tiles for
        
    Returns:
        Number of tiles required
    """
    lon_min, lon_max, lat_min, lat_max = map_bounds
    
    def deg2num(lat_deg, lon_deg, zoom):
        try:
            # Validate input coordinates
            if not (isinstance(lat_deg, (int, float)) and isinstance(lon_deg, (int, float))):
                raise ValueError(f"Invalid coordinate types: lat_deg={type(lat_deg)}, lon_deg={type(lon_deg)}")
            
            if not (-90 <= lat_deg <= 90):
                raise ValueError(f"Latitude out of range: {lat_deg}")
            
            if not (-180 <= lon_deg <= 180):
                raise ValueError(f"Longitude out of range: {lon_deg}")
            
            lat_rad = np.radians(lat_deg)
            n = 2.0 ** zoom
            
            # Calculate tile coordinates
            xtile = int((lon_deg + 180.0) / 360.0 * n)
            
            # Calculate ytile with error handling for edge cases
            tan_lat = np.tan(lat_rad)
            cos_lat = np.cos(lat_rad)
            
            # Check for division by zero or invalid values
            if cos_lat == 0 or np.isnan(cos_lat):
                raise ValueError(f"Invalid latitude for tile calculation: {lat_deg}")
            
            log_term = np.log(tan_lat + (1 / cos_lat))
            
            # Check for invalid log result
            if np.isnan(log_term) or np.isinf(log_term):
                raise ValueError(f"Invalid log calculation for latitude: {lat_deg}")
            
            ytile = int((1.0 - log_term / np.pi) / 2.0 * n)
            
            # Validate tile coordinates
            if np.isnan(xtile) or np.isnan(ytile) or np.isinf(xtile) or np.isinf(ytile):
                raise ValueError(f"Invalid tile coordinates calculated: xtile={xtile}, ytile={ytile}")
            
            return (xtile, ytile)
        except Exception as e:
            debug_log(f"Error in deg2num: lat_deg={lat_deg}, lon_deg={lon_deg}, zoom={zoom}, error={str(e)}")
            raise e
    
    try:
        # Get tile coordinates for the corners of our region
        northwest = deg2num(lat_max, lon_min, zoom_level)
        southeast = deg2num(lat_min, lon_max, zoom_level)
        
        # Extract min/max tile coordinates
        x_min, y_min = northwest[0], northwest[1]
        x_max, y_max = southeast[0], southeast[1]
        
        # Ensure proper ordering
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
        
        # Calculate the total number of tiles
        tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
        
        return tile_count
    except Exception as e:
        debug_log(f"Error calculating tile count for bounds {map_bounds} at zoom {zoom_level}: {str(e)}")
        raise e

def detect_zoom_level(map_bounds: Tuple[float, float, float, float], 
                     min_tiles: int = 10,
                     max_tiles: int = 200,
                     map_style: str = "osm",
                     resolution_x: Optional[int] = None,
                     resolution_y: Optional[int] = None) -> List[int]:
    """
    Detect all suitable zoom levels for map bounds based on the number of tiles required.
    
    Args:
        map_bounds: Tuple (lon_min, lon_max, lat_min, lat_max)
        min_tiles: Minimum acceptable number of tiles
        max_tiles: Maximum acceptable number of tiles
        map_style: Map tile style being used (e.g., "osm", "otm")
        resolution_x: Optional image width in pixels (for padded bounds calculation)
        resolution_y: Optional image height in pixels (for padded bounds calculation)
    
    Returns:
        List of suitable zoom levels where tile count is between min_tiles and max_tiles
    """
    if not map_bounds:
        debug_log("ERROR: no coordinates found for detecting zoom level!")
        print("Error: no coordinates found for detecting zoom level!")
        return []
    
    # Apply minimum latitude distance of 0.01 degrees to ensure consistent zoom level calculation
    # This matches the logic in calculate_bounding_box_for_wrapped_coordinates()
    lon_min, lon_max, lat_min, lat_max = map_bounds
    lat_distance = lat_max - lat_min
    if lat_distance < 0.01:
        # Increase lat_max and decrease lat_min by equal amounts to reach 0.01 distance
        adjustment = (0.01 - lat_distance) / 2
        lat_min -= adjustment
        lat_max += adjustment
        map_bounds = (lon_min, lon_max, lat_min, lat_max)
        debug_log(f"Applied minimum latitude distance adjustment: {lat_distance:.6f}° -> 0.01°")
    
    # If resolution is provided, calculate padded bounds (same as image generation uses)
    # This ensures zoom level selection matches actual tile usage
    bounds_for_tile_calculation = map_bounds
    if resolution_x is not None and resolution_y is not None:
        try:
            from image_generator_utils import ImageGenerator
            generator = ImageGenerator()
            target_aspect_ratio = resolution_x / resolution_y
            padded_bounds = generator.calculate_aspect_ratio_bounds(
                map_bounds, target_aspect_ratio=target_aspect_ratio
            )
            bounds_for_tile_calculation = padded_bounds
            debug_log(f"Using padded bounds for zoom level detection (resolution: {resolution_x}x{resolution_y})")
        except Exception as e:
            debug_log(f"Warning: Failed to calculate padded bounds, using raw bounds: {e}")
            # Fall back to raw bounds if padding calculation fails
    
    # Set maximum zoom level based on map style
    if map_style == "otm":
        max_zoom = 15
    elif map_style == "cyclosm":
        max_zoom = 17
    else:
        max_zoom = 20
    
    # Find all suitable zoom levels
    suitable_zooms = []
    
    for zoom in range(1, max_zoom + 1):
        try:
            tile_count = calculate_tile_count(bounds_for_tile_calculation, zoom)
            debug_log(f"Zoom level {zoom}: tile count {tile_count}")
            
            if min_tiles <= tile_count <= max_tiles:
                suitable_zooms.append(zoom)
            elif tile_count > max_tiles:
                break  # No need to check higher zooms as they'll have even more tiles
        except Exception as e:
            print(f"Error calculating tiles for zoom {zoom}: {e}")
            break
    
    if suitable_zooms:
        debug_log(f"Found {len(suitable_zooms)} suitable zoom levels between {min_tiles}-{max_tiles} tiles: {suitable_zooms}")
    else:
        # No suitable zoom levels found - use max_zoom as fallback
        # This ensures we always return at least one zoom level, even if it exceeds max_tiles
        debug_log(f"No suitable zoom levels found within {min_tiles}-{max_tiles} tile count constraints, using max_zoom {max_zoom} as fallback")
        suitable_zooms = [max_zoom]
        
    return suitable_zooms 