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
    acquire_map_tile_lock,
    release_map_tile_lock,
)
from write_log import write_debug_log, write_log

REMOTE_BASE = "map tile cache"
MAX_FTP_WORKERS = 8

# Map tile download retry policy (used by image/video workers).
MAP_TILE_CACHE_MAX_ATTEMPTS = 3
MAP_TILE_CACHE_FAILURE_COOLDOWN_SEC = 60.0

class MapTileCachingExhaustedError(Exception):
    """Raised after all map-tile download retries failed.

    Callers should set job status to *error* before raising (or in the
    ``on_retries_exhausted`` callback of ``run_map_tile_caching_with_retries``).
    """

def run_map_tile_caching_with_retries(
    fetch_cache_result,
    *,
    log_callback=None,
    max_attempts=MAP_TILE_CACHE_MAX_ATTEMPTS,
    cooldown_sec=MAP_TILE_CACHE_FAILURE_COOLDOWN_SEC,
    on_retries_exhausted=None,
):
    """Run ``fetch_cache_result()`` until it returns a dict with ``success`` True.

    ``fetch_cache_result`` must be a no-argument callable returning either
    ``None`` or a dict that includes ``success`` (bool).

    After each failed attempt except the last, sleeps ``cooldown_sec`` (for
    provider rate limits). On the final failure, calls ``on_retries_exhausted``
    if provided, then raises :class:`MapTileCachingExhaustedError`.
    """
    _log = log_callback or write_log
    last_result = None
    for attempt in range(1, max_attempts + 1):
        last_result = fetch_cache_result()
        if last_result is not None and last_result.get("success"):
            return last_result
        if attempt < max_attempts:
            _log(
                f"Map tile downloads did not complete successfully "
                f"(attempt {attempt}/{max_attempts}). "
                f"Waiting {int(cooldown_sec)}s before retrying the full tile caching step."
            )
            time.sleep(cooldown_sec)
    if on_retries_exhausted:
        on_retries_exhausted()
    raise MapTileCachingExhaustedError(
        f"Map tile caching failed after {max_attempts} attempts"
    )

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
        'otm': 'OpenTopoMapTiles',
        'cyclosm': 'CyclOSMTiles',
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

def _download_tiles_from_remote_cache(
    tiles_to_check,
    map_style: str,
    storage_box_credentials: dict,
    debug_callback=None,
    progress_callback=None,
) -> set:
    """Download tiles from the remote cache (storage box) via FTP.

    Acquires/releases the cache lock internally so callers don't have to.

    Args:
        tiles_to_check: Iterable of ``(x, y, zoom)`` tuples to look for.
        map_style: Map style identifier.
        storage_box_credentials: ``{'address', 'user', 'password'}``.
        debug_callback: Debug-only log function.
        progress_callback: Optional ``(bar_name, pct, text)`` function.

    Returns:
        Set of ``(x, y, zoom)`` tuples that were successfully downloaded.
    """
    _debug = debug_callback or write_debug_log
    downloaded = set()

    _acquire_cache_lock(_debug)
    try:
        t0 = time.perf_counter()
        tasks = []
        for tile in tiles_to_check:
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

        total = len(tasks)
        with ThreadPoolExecutor(max_workers=MAX_FTP_WORKERS) as pool:
            futures = {pool.submit(_ftp_download_one, t): t for t in tasks}
            done_count = 0
            for future in as_completed(futures):
                x, y, zoom, err = future.result()
                if err is None:
                    downloaded.add((x, y, zoom))
                done_count += 1
                if done_count % 50 == 0:
                    _debug(f"Remote cache: checked {done_count}/{total}")
                    if progress_callback:
                        pct = int(done_count / total * 30)
                        progress_callback("progress_bar_tiles", pct,
                                          f"Remote cache: {done_count}/{total}")

        elapsed = time.perf_counter() - t0
        _debug(f"{len(downloaded)} tiles downloaded from remote "
               f"cache in {elapsed:.1f} seconds")
    finally:
        _release_cache_lock(_debug)

    return downloaded


def cache_required_tiles(
    required_tiles,
    map_style: str,
    storage_box_credentials: dict = None,
    log_callback=None,
    debug_callback=None,
    progress_callback=None,
    provider_queue_callback=None,
    provider_start_callback=None,
    provider_progress_callback=None,
) -> dict:
    """Cache all tiles required for a job using the three-tier system.

    The provider lock (service lock) for step 3 is managed internally.  When
    the renderer has to wait behind the lock (another renderer is downloading),
    the lock is released immediately and step 2 (remote cache check) is
    repeated — the other renderer may have already uploaded the tiles we need.

    Args:
        required_tiles: Iterable of ``(x, y, zoom)`` tuples.
        map_style: Map style identifier (e.g. ``'osm'``, ``'stadia_dark'``).
        storage_box_credentials: ``{'address', 'user', 'password'}`` or *None*
            to skip remote cache operations.
        log_callback: Regular (always-visible) log function.
        debug_callback: Debug-only log function.
        progress_callback: ``(bar_name, percentage, text)`` progress function.
        provider_queue_callback: Called (no args) when waiting in the provider
            lock queue — callers typically set a "queued" status here.
        provider_start_callback: Called (no args) when provider downloads are
            about to begin — callers typically set a "downloading" status here.
        provider_progress_callback: Called as ``callback(milestone_percent)``
            on provider download milestones (10, 20, ... 100).

    Returns:
        Dict with caching success and tile counts:
        ``success``, ``count_local``, ``count_remote``, ``count_provider``, ``count_uploaded``.
    """
    _log = log_callback or write_log
    _debug = debug_callback or write_debug_log

    required_list = list(required_tiles)
    total_required = len(required_list)
    _debug(f"{total_required} tiles required total")

    if total_required == 0:
        _log("0 map tiles required, 0 in local cache, 0 in remote cache, "
             "0 downloaded from map providers, 0 uploaded to remote cache")
        return {
            'success': True,
            'total_required': 0,
            'count_local': 0,
            'count_remote': 0,
            'count_provider': 0,
            'count_uploaded': 0,
        }

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
        return {
            'success': True,
            'total_required': total_required,
            'count_local': count_local,
            'count_remote': 0,
            'count_provider': 0,
            'count_uploaded': 0,
        }

    # ------------------------------------------------------------------
    # Step 2: Download from remote cache (storage box)
    # ------------------------------------------------------------------
    has_credentials = (
        storage_box_credentials
        and storage_box_credentials.get('address')
        and storage_box_credentials.get('user')
        and storage_box_credentials.get('password')
    )

    count_remote = 0

    if has_credentials and tiles_not_local:
        downloaded_from_remote = _download_tiles_from_remote_cache(
            tiles_not_local, map_style, storage_box_credentials,
            _debug, progress_callback,
        )
        count_remote = len(downloaded_from_remote)
    else:
        downloaded_from_remote = set()
        _debug("Skipping remote cache (no storage box credentials)")

    # Remaining tiles that still need provider downloads
    tiles_need_provider = [
        t for t in tiles_not_local if t not in downloaded_from_remote
    ]

    # ------------------------------------------------------------------
    # Step 3: Download from tile providers (with service lock + retry)
    # ------------------------------------------------------------------
    count_provider = 0
    provider_elapsed = 0.0

    if tiles_need_provider:
        json_data_for_lock = {'map_style': map_style}

        while tiles_need_provider:
            lock_acquired, lock_error, waited = acquire_map_tile_lock(
                json_data_for_lock,
                log_callback=_log,
                debug_callback=_debug,
                status_callback=provider_queue_callback,
            )

            if not lock_acquired:
                _log(f"Map tile lock acquisition failed: {lock_error}")
                return {
                    'success': False,
                    'total_required': total_required,
                    'count_local': count_local,
                    'count_remote': count_remote,
                    'count_provider': 0,
                    'count_uploaded': 0,
                }

            if waited and has_credentials:
                # Another renderer was active while we waited — tiles we need
                # may now be in the remote cache.  Release the provider lock
                # (we're not downloading yet) and re-check before committing.
                release_map_tile_lock(json_data_for_lock, debug_callback=_debug)

                _debug(f"Provider lock was queued — re-checking remote cache "
                       f"for {len(tiles_need_provider)} remaining tiles")

                newly_downloaded = _download_tiles_from_remote_cache(
                    tiles_need_provider, map_style, storage_box_credentials,
                    _debug,
                )
                count_remote += len(newly_downloaded)

                if newly_downloaded:
                    _debug(f"{len(newly_downloaded)} additional tiles found in "
                           f"remote cache after re-check")

                tiles_need_provider = [
                    t for t in tiles_need_provider
                    if t not in newly_downloaded
                ]

                if not tiles_need_provider:
                    _debug("All remaining tiles found in remote cache after "
                           "re-check — no provider downloads needed")
                    break

                _debug(f"{len(tiles_need_provider)} tiles still need provider "
                       f"downloads — retrying provider lock")
                continue

            # Lock acquired without waiting (or no remote-cache credentials
            # to make a re-check worthwhile) — download from providers.
            try:
                if provider_start_callback:
                    provider_start_callback()

                t0 = time.perf_counter()
                map_tiles = create_map_tiles(map_style)
                consecutive_errors = 0
                max_consecutive_errors = 20

                if progress_callback:
                    progress_callback(
                        "progress_bar_tiles", 30,
                        f"Downloading {len(tiles_need_provider)} tiles from "
                        f"provider")

                last_provider_milestone = 0
                for idx, tile in enumerate(tiles_need_provider):
                    x, y, zoom = tile
                    try:
                        map_tiles.get_image((x, y, zoom))
                        count_provider += 1
                        consecutive_errors = 0
                    except Exception:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            _debug(
                                f"Too many consecutive provider errors "
                                f"({consecutive_errors}), pausing 10 seconds")
                            time.sleep(10)
                            consecutive_errors = 0

                    if progress_callback and (idx + 1) % 10 == 0:
                        pct = 30 + int(
                            (idx + 1) / len(tiles_need_provider) * 40)
                        progress_callback(
                            "progress_bar_tiles", pct,
                            f"Provider: {idx + 1}/{len(tiles_need_provider)}")

                    # Provider-only milestones for external status reporting.
                    if provider_progress_callback:
                        provider_pct = int((idx + 1) / len(tiles_need_provider) * 100)
                        milestone = (provider_pct // 10) * 10
                        if milestone > last_provider_milestone and 10 <= milestone <= 100:
                            try:
                                provider_progress_callback(milestone)
                            except Exception:
                                pass
                            last_provider_milestone = milestone

                provider_elapsed = time.perf_counter() - t0
                _debug(f"{count_provider} tiles downloaded from providers "
                       f"in {provider_elapsed:.1f} seconds")
            finally:
                release_map_tile_lock(json_data_for_lock, debug_callback=_debug)

            break

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
    # Final verification: every required tile must exist locally with data.
    # Deliberately no blank/validity check here (ocean tiles etc. may look
    # "invalid" but are correct). Quality checks apply only before remote
    # upload (Step 4).
    # ------------------------------------------------------------------
    incomplete_tiles = []
    for tile in required_list:
        x, y, zoom = tile
        local_path = _tile_local_path(map_style, x, y, zoom)
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            incomplete_tiles.append(tile)

    if incomplete_tiles:
        _log(
            f"Map tile caching incomplete: {len(incomplete_tiles)} of "
            f"{total_required} required tiles missing or empty after "
            f"remote cache and provider steps"
        )
        if progress_callback:
            progress_callback(
                "progress_bar_tiles",
                100,
                "Tile caching failed — incomplete tiles",
            )
        return {
            'success': False,
            'failure_reason': 'incomplete_tiles',
            'incomplete_count': len(incomplete_tiles),
            'total_required': total_required,
            'count_local': count_local,
            'count_remote': count_remote,
            'count_provider': count_provider,
            'count_uploaded': count_uploaded,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if progress_callback:
        progress_callback("progress_bar_tiles", 100, "Tile caching complete")

    _log(f"{total_required} map tiles required, {count_local} in local cache, "
         f"{count_remote} in remote cache, {count_provider} downloaded from "
         f"map providers, {count_uploaded} uploaded to remote cache")

    return {
        'success': True,
        'total_required': total_required,
        'count_local': count_local,
        'count_remote': count_remote,
        'count_provider': count_provider,
        'count_uploaded': count_uploaded,
    }
