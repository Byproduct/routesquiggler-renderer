"""
Cache size limit enforcement for the map tile cache.

On each run, checks the total size of the cache folder against the configured
limit (map_tile_cache_size in config.txt, in gigabytes). If the cache exceeds
the limit, the oldest files are deleted until the total is back under quota.

The last-run timestamp is kept in memory so callers can skip re-runs within
a 1-hour window. It is not persisted across restarts (bootup always runs it).
"""

import os
import time

from write_log import write_debug_log, write_log

_last_run_time = None

RERUN_INTERVAL_SECONDS = 3600  # 1 hour


def enforce_cache_size_limit(config, log_callback=None):
    """Check cache folder size and delete oldest files if over the configured limit.

    Args:
        config: Config object (needs .map_tile_cache_path and .map_tile_cache_size).
        log_callback: Function used for user-visible log messages.
                      Falls back to write_log if not provided.

    Returns:
        True when finished (including when skipped due to unlimited), False on error.
    """
    global _last_run_time

    if log_callback is None:
        log_callback = write_log

    max_size_gb = config.map_tile_cache_size

    if max_size_gb is None:
        write_debug_log("Cache size limit is unlimited; skipping enforcement")
        _last_run_time = time.time()
        return True

    cache_path = config.map_tile_cache_path

    if not os.path.exists(cache_path):
        write_debug_log(f"Cache folder does not exist yet: {cache_path}")
        _last_run_time = time.time()
        return True

    max_size_bytes = max_size_gb * 1024 * 1024 * 1024

    # Single walk using scandir (faster than os.walk + os.stat on large trees:
    # DirEntry.stat() can use directory-entry data and avoid extra syscalls).
    files = []
    total_size = 0
    stack = [cache_path]
    while stack:
        dirpath = stack.pop()
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            st = entry.stat(follow_symlinks=False)
                            total_size += st.st_size
                            files.append((entry.path, st.st_size, st.st_mtime))
                    except OSError:
                        pass
        except OSError:
            pass

    total_size_gb = total_size / (1024 ** 3)

    if total_size <= max_size_bytes:
        log_callback(f"Cache size is within limit ({total_size_gb:.2f} GB / {max_size_gb} GB).")
        _last_run_time = time.time()
        return True

    log_callback(
        f"Cache size ({total_size_gb:.2f} GB) exceeds limit ({max_size_gb} GB). "
        "Pruning oldest files."
    )

    t_start = time.time()

    # Sort oldest-first and delete until under quota
    files.sort(key=lambda f: f[2])

    deleted_count = 0
    deleted_size = 0
    for filepath, size, _mtime in files:
        if total_size <= max_size_bytes:
            break
        try:
            os.remove(filepath)
            total_size -= size
            deleted_count += 1
            deleted_size += size
        except OSError as exc:
            write_debug_log(f"Failed to delete {filepath}: {exc}")

    elapsed = time.time() - t_start
    deleted_gb = deleted_size / (1024 ** 3)
    remaining_gb = total_size / (1024 ** 3)
    log_callback(
        f"Cache was trimmed to limit in {elapsed:.1f} seconds. "
        f"Deleted {deleted_count} files ({deleted_gb:.2f} GB). "
        f"Cache is now {remaining_gb:.2f} GB."
    )

    _last_run_time = time.time()
    return True


def check_and_enforce_if_due(config, log_callback=None):
    """Run enforcement only if more than 1 hour has elapsed since the last run.

    Silently returns True when the limit is unlimited or the interval has not
    elapsed yet. Intended to be called before every job request.
    """
    if config.map_tile_cache_size is None:
        return True

    if _last_run_time is not None:
        if time.time() - _last_run_time < RERUN_INTERVAL_SECONDS:
            return True

    return enforce_cache_size_limit(config, log_callback)
