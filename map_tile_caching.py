"""
Unified map tile caching system for the Route Squiggler render client.

Three-tier tile resolution:
  1. Local cache  – tiles already on disk.
  2. Remote cache – tiles on the Hetzner Storage Box (FTP), downloaded in parallel.
  3. Tile providers – fresh downloads via CartoPy with rate-limiting and API keys.

After provider downloads, new tiles are verified and uploaded to the remote cache.
"""

import ftplib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import config
from image_generator_maptileutils import create_map_tiles, set_cache_directory
from map_tile_cache_sweep import is_blank_or_erroneous_tile
from map_tile_lock import (
    MAP_TILE_LOCK_API_URL,
    LOCK_REQUEST_TIMEOUT,
    UNLOCK_REQUEST_TIMEOUT,
    LOCK_RETRY_INTERVAL,
    MAX_LOCK_WAIT_TIME,
)
from write_log import write_debug_log, write_log

REMOTE_BASE = "map tile cache"
MAX_FTP_WORKERS = 8

# ---------------------------------------------------------------------------
# Tile path helpers
# ---------------------------------------------------------------------------

# Maps map_style -> the subdirectory that set_cache_directory() creates
_STYLE_SUBDIR = {
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
    'geoapify_klokantech': 'GeoapifyKlokantech',
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
    'thunderforest_transport_dark': 'ThunderforestTransportDark',
    'esri256_natgeo': 'Esri256NatGeo',
    'esri256_satellite': 'Esri256Satellite',
    'esri256_topo': 'Esri256Topo',
    'esri256_transport_nomap': 'Esri256TransportNomap',
    'esri256_elevation_nomap': 'Esri256ElevationNomap',
    'esri512_community': 'Esri512Community',
    'esri512_darkgray_nolabels': 'Esri512DarkgrayNolabels',
    'esri512_darkgray': 'Esri512Darkgray',
    'esri512_lightgray': 'Esri512Lightgray',
    'esri512_lightgray_nolabels': 'Esri512LightgrayNolabels',
    'esri512_midcentury': 'Esri512Midcentury',
    'esri512_newspaper': 'Esri512Newspaper',
    'esri512_nova': 'Esri512Nova',
    'esri512_outdoor': 'Esri512Outdoor',
    'esri512_streets': 'Esri512Streets',
    'esri512_transport_nomap': 'Esri512TransportNomap',
    'esri512_roads_nomap': 'Esri512RoadsNomap',
    'esri512_navigation': 'Esri512Navigation',
    'esri512_navigation_night': 'Esri512NavigationNight',
    'mapbox_outdoors': 'MapboxOutdoors',
    'mapbox_satellite': 'MapboxSatellite',
    'mapbox_hybrid': 'MapboxHybrid',
    'mapbox_pencil': 'MapboxPencil',
    'mapbox_oilcompany': 'MapboxOilcompany',
    'mapbox_japanese': 'MapboxJapanese',
    'mapbox_blueprint': 'MapboxBlueprint',
}


def _cartopy_class_subdir(map_style: str) -> str:
    """Return the subdirectory that CartoPy's tile source class creates inside
    the cache dir.  This mirrors the logic in
    ``video_generator_cache_map_tiles.is_tile_cached``."""
    if map_style.startswith('stadia'):
        return 'StadiaTiles'
    if map_style.startswith('geoapify'):
        return 'GeoapifyTiles'
    if map_style.startswith('thunderforest'):
        return 'ThunderforestTiles'
    if map_style.startswith('esri256'):
        return 'Esri256Tiles'
    if map_style.startswith('esri512'):
        return 'Esri512Tiles'
    if map_style.startswith('mapbox'):
        return 'MapboxTiles'
    fallback = {
        'osm': 'OSM',
        'otm': 'OpenTopoMap',
        'cyclosm': 'CyclOSM',
    }
    return fallback.get(map_style, map_style.upper())


def _tile_filename(x: int, y: int, zoom: int) -> str:
    return f"{x}_{y}_{zoom}.npy"


def _tile_local_path(map_style: str, x: int, y: int, zoom: int) -> str:
    """Full local path for a tile."""
    style_dir = _STYLE_SUBDIR.get(map_style, 'OSM')
    class_dir = _cartopy_class_subdir(map_style)
    return os.path.join(config.map_tile_cache_path, style_dir, class_dir,
                        _tile_filename(x, y, zoom))


def _tile_relative_path(map_style: str, x: int, y: int, zoom: int) -> str:
    """Path relative to the cache root (for storage box operations).
    Uses forward slashes for FTP compatibility."""
    style_dir = _STYLE_SUBDIR.get(map_style, 'OSM')
    class_dir = _cartopy_class_subdir(map_style)
    return f"{style_dir}/{class_dir}/{_tile_filename(x, y, zoom)}"


def _tile_remote_path(map_style: str, x: int, y: int, zoom: int) -> str:
    """Full remote FTP path on the storage box."""
    return f"{REMOTE_BASE}/{_tile_relative_path(map_style, x, y, zoom)}"


# ---------------------------------------------------------------------------
# Cache lock (uses the same lock API as map tile provider locks)
# ---------------------------------------------------------------------------

_CACHE_SERVICE = "cache"


def _acquire_cache_lock(debug_callback=None):
    """Acquire an exclusive lock on the remote tile cache.

    Follows the same protocol as the per-provider locks in ``map_tile_lock.py``:
    POST ``"cache lock"`` and wait for 200 (acquired) or retry on 423 (busy).
    On timeout or network error we proceed (fail-open) so the job is never
    blocked by an unreachable lock server.

    Returns:
        ``True`` if the lock was acquired (or we decided to proceed anyway).
    """
    _debug = debug_callback or write_debug_log
    _debug("Requesting cache lock for storage box access")

    start = time.time()
    attempt = 0

    while True:
        attempt += 1
        elapsed = time.time() - start
        if elapsed > MAX_LOCK_WAIT_TIME:
            _debug("Cache lock wait exceeded 60 minutes — proceeding without lock")
            return True

        try:
            resp = requests.post(
                MAP_TILE_LOCK_API_URL,
                data=f"{_CACHE_SERVICE} lock",
                headers={"Content-Type": "text/plain"},
                timeout=LOCK_REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                _debug("Cache lock acquired")
                return True
            if resp.status_code == 423:
                remaining = int((MAX_LOCK_WAIT_TIME - elapsed) / 60)
                _debug(f"Cache is locked by another client. "
                       f"Retrying in {LOCK_RETRY_INTERVAL}s "
                       f"({remaining} min remaining)")
                time.sleep(LOCK_RETRY_INTERVAL)
                continue
            _debug(f"Unexpected cache lock response {resp.status_code} — proceeding")
            return True

        except requests.Timeout:
            _debug("Cache lock request timed out — proceeding")
            return True
        except requests.RequestException as e:
            _debug(f"Cache lock request failed ({e}) — proceeding")
            return True


def _release_cache_lock(debug_callback=None):
    """Release the remote tile cache lock."""
    _debug = debug_callback or write_debug_log
    try:
        requests.post(
            MAP_TILE_LOCK_API_URL,
            data=f"{_CACHE_SERVICE} unlock",
            headers={"Content-Type": "text/plain"},
            timeout=UNLOCK_REQUEST_TIMEOUT,
        )
        _debug("Cache lock released")
    except Exception as e:
        _debug(f"Cache lock release failed ({e}) — lock will expire on server")


# ---------------------------------------------------------------------------
# FTP helpers (thread-local connections, parallel workers)
# ---------------------------------------------------------------------------

_worker_ftp = threading.local()


def _get_ftp(creds: dict):
    """Return this thread's FTP connection, creating it on first use."""
    ftp = getattr(_worker_ftp, 'ftp', None)
    if ftp is None:
        ftp = ftplib.FTP(creds['address'], timeout=30)
        ftp.login(creds['user'], creds['password'])
        _worker_ftp.ftp = ftp
    return ftp


def _reset_ftp():
    """Close and clear this thread's FTP connection so the next call
    to _get_ftp will create a fresh one."""
    ftp = getattr(_worker_ftp, 'ftp', None)
    if ftp is not None:
        try:
            ftp.quit()
        except Exception:
            pass
        _worker_ftp.ftp = None


def _ftp_download_one(args):
    """Download one tile from the storage box.
    Returns (x, y, zoom, error_or_None)."""
    x, y, zoom, remote_path, local_path, creds = args
    try:
        ftp = _get_ftp(creds)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f"RETR {remote_path}", f.write)
        if os.path.getsize(local_path) > 0:
            return (x, y, zoom, None)
        os.remove(local_path)
        return (x, y, zoom, "empty file")
    except ftplib.all_errors as e:
        # Remove partial file
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except OSError:
            pass
        # Reset connection on error so next call reconnects
        _reset_ftp()
        return (x, y, zoom, str(e))


def _ftp_upload_one(args):
    """Upload one tile to the storage box.
    Returns (x, y, zoom, error_or_None)."""
    x, y, zoom, remote_path, local_path, creds = args
    try:
        ftp = _get_ftp(creds)
        # Ensure remote directories exist
        parts = remote_path.rsplit('/', 1)
        if len(parts) == 2:
            remote_dir = parts[0]
            _ftp_mkdirs(ftp, remote_dir)
        with open(local_path, 'rb') as f:
            ftp.storbinary(f"STOR {remote_path}", f)
        return (x, y, zoom, None)
    except ftplib.all_errors as e:
        _reset_ftp()
        return (x, y, zoom, str(e))


def _ftp_mkdirs(ftp, remote_dir):
    """Recursively create remote directories, ignoring 'already exists' errors."""
    parts = remote_dir.split('/')
    current = ''
    for part in parts:
        current = f"{current}/{part}" if current else part
        try:
            ftp.mkd(current)
        except ftplib.error_perm:
            pass  # directory likely already exists


# ---------------------------------------------------------------------------
# Core caching pipeline
# ---------------------------------------------------------------------------

def cache_required_tiles(
    required_tiles,
    map_style: str,
    storage_box_credentials: dict = None,
    log_callback=None,
    debug_callback=None,
    progress_callback=None,
) -> bool:
    """Cache all tiles required for a job using the three-tier system.

    Args:
        required_tiles: Iterable of ``(x, y, zoom)`` tuples.
        map_style: Map style identifier (e.g. ``'osm'``, ``'stadia_dark'``).
        storage_box_credentials: ``{'address', 'user', 'password'}`` or *None*
            to skip remote cache operations.
        log_callback: Regular (always-visible) log function.
        debug_callback: Debug-only log function.
        progress_callback: ``(bar_name, percentage, text)`` progress function.

    Returns:
        ``True`` if caching succeeded (all or most tiles available).
    """
    _log = log_callback or write_log
    _debug = debug_callback or write_debug_log

    required_list = list(required_tiles)
    total_required = len(required_list)
    _debug(f"{total_required} tiles required total")

    if total_required == 0:
        _log("0 map tiles required, 0 in local cache, 0 in remote cache, "
             "0 downloaded from map providers, 0 uploaded to remote cache")
        return True

    # Ensure cache directory and CartoPy are configured
    set_cache_directory(map_style)

    # ------------------------------------------------------------------
    # Step 1: Check local cache
    # ------------------------------------------------------------------
    tiles_in_local_cache = []
    tiles_not_local = []
    for tile in required_list:
        x, y, zoom = tile
        local_path = _tile_local_path(map_style, x, y, zoom)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            tiles_in_local_cache.append(tile)
        else:
            tiles_not_local.append(tile)

    count_local = len(tiles_in_local_cache)
    _debug(f"{count_local} tiles in local cache")

    if not tiles_not_local:
        _log(f"{total_required} map tiles required, {count_local} in local cache, "
             "0 in remote cache, 0 downloaded from map providers, "
             "0 uploaded to remote cache")
        return True

    # ------------------------------------------------------------------
    # Step 2: Download from remote cache (storage box)
    # ------------------------------------------------------------------
    has_credentials = (
        storage_box_credentials
        and storage_box_credentials.get('address')
        and storage_box_credentials.get('user')
        and storage_box_credentials.get('password')
    )

    downloaded_from_remote = set()
    remote_elapsed = 0.0

    if has_credentials and tiles_not_local:
        _acquire_cache_lock(_debug)
        try:
            t0 = time.perf_counter()
            tasks = []
            for tile in tiles_not_local:
                x, y, zoom = tile
                tasks.append((
                    x, y, zoom,
                    _tile_remote_path(map_style, x, y, zoom),
                    _tile_local_path(map_style, x, y, zoom),
                    storage_box_credentials,
                ))

            if progress_callback:
                progress_callback("progress_bar_tiles", 0,
                                  "Checking remote tile cache")

            with ThreadPoolExecutor(max_workers=MAX_FTP_WORKERS) as pool:
                futures = {pool.submit(_ftp_download_one, t): t for t in tasks}
                done_count = 0
                for future in as_completed(futures):
                    x, y, zoom, err = future.result()
                    if err is None:
                        downloaded_from_remote.add((x, y, zoom))
                    done_count += 1
                    if progress_callback and done_count % 50 == 0:
                        pct = int(done_count / len(tasks) * 30)
                        progress_callback("progress_bar_tiles", pct,
                                          f"Remote cache: {done_count}/{len(tasks)}")

            remote_elapsed = time.perf_counter() - t0
            _debug(f"{len(downloaded_from_remote)} tiles downloaded from remote "
                   f"cache in {remote_elapsed:.1f} seconds")
        finally:
            _release_cache_lock(_debug)
    else:
        _debug("Skipping remote cache (no storage box credentials)")

    count_remote = len(downloaded_from_remote)

    # Remaining tiles that still need provider downloads
    tiles_need_provider = [
        t for t in tiles_not_local if t not in downloaded_from_remote
    ]

    # ------------------------------------------------------------------
    # Step 3: Download from tile providers
    # ------------------------------------------------------------------
    count_provider = 0
    provider_elapsed = 0.0

    if tiles_need_provider:
        t0 = time.perf_counter()
        map_tiles = create_map_tiles(map_style)
        consecutive_errors = 0
        max_consecutive_errors = 20

        if progress_callback:
            progress_callback("progress_bar_tiles", 30,
                              f"Downloading {len(tiles_need_provider)} tiles from provider")

        for idx, tile in enumerate(tiles_need_provider):
            x, y, zoom = tile
            try:
                map_tiles.get_image((x, y, zoom))
                count_provider += 1
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    _debug(f"Too many consecutive provider errors ({consecutive_errors}), "
                           "pausing 10 seconds")
                    time.sleep(10)
                    consecutive_errors = 0

            if progress_callback and (idx + 1) % 10 == 0:
                pct = 30 + int((idx + 1) / len(tiles_need_provider) * 40)
                progress_callback("progress_bar_tiles", pct,
                                  f"Provider: {idx + 1}/{len(tiles_need_provider)}")

        provider_elapsed = time.perf_counter() - t0
        _debug(f"{count_provider} tiles downloaded from providers "
               f"in {provider_elapsed:.1f} seconds")

    # ------------------------------------------------------------------
    # Step 4: Verify tiles downloaded from providers
    # ------------------------------------------------------------------
    verified_tiles = []
    verify_elapsed = 0.0

    if tiles_need_provider:
        t0 = time.perf_counter()
        suspect_tiles = []

        if progress_callback:
            progress_callback("progress_bar_tiles", 70, "Verifying downloaded tiles")

        for tile in tiles_need_provider:
            x, y, zoom = tile
            local_path = _tile_local_path(map_style, x, y, zoom)
            if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                continue
            if is_blank_or_erroneous_tile(local_path):
                suspect_tiles.append(tile)
            else:
                verified_tiles.append(tile)

        verify_elapsed = time.perf_counter() - t0
        _debug(f"Downloaded tiles verified in {verify_elapsed:.1f} seconds")
        if suspect_tiles:
            _debug(f"{len(suspect_tiles)} tiles did not pass verification "
                   "(kept locally but not uploaded to remote cache)")

    # ------------------------------------------------------------------
    # Step 5: Upload verified new tiles to remote cache
    # ------------------------------------------------------------------
    count_uploaded = 0
    upload_elapsed = 0.0

    if has_credentials and verified_tiles:
        _acquire_cache_lock(_debug)
        try:
            t0 = time.perf_counter()
            tasks = []
            for tile in verified_tiles:
                x, y, zoom = tile
                tasks.append((
                    x, y, zoom,
                    _tile_remote_path(map_style, x, y, zoom),
                    _tile_local_path(map_style, x, y, zoom),
                    storage_box_credentials,
                ))

            if progress_callback:
                progress_callback("progress_bar_tiles", 80,
                                  "Uploading new tiles to remote cache")

            with ThreadPoolExecutor(max_workers=MAX_FTP_WORKERS) as pool:
                futures = {pool.submit(_ftp_upload_one, t): t for t in tasks}
                done_count = 0
                for future in as_completed(futures):
                    x, y, zoom, err = future.result()
                    if err is None:
                        count_uploaded += 1
                    done_count += 1
                    if progress_callback and done_count % 50 == 0:
                        pct = 80 + int(done_count / len(tasks) * 20)
                        progress_callback("progress_bar_tiles", pct,
                                          f"Uploading: {done_count}/{len(tasks)}")

            upload_elapsed = time.perf_counter() - t0
            _debug(f"{count_uploaded} new tiles were uploaded to remote cache "
                   f"in {upload_elapsed:.1f} seconds")
        finally:
            _release_cache_lock(_debug)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("progress_bar_tiles", 100, "Tile caching complete")

    _log(f"{total_required} map tiles required, {count_local} in local cache, "
         f"{count_remote} in remote cache, {count_provider} downloaded from "
         f"map providers, {count_uploaded} uploaded to remote cache")

    return True
