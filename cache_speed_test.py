"""
Storage box map tile cache download speed test.

Picks 500 random .npy files from the local map tile cache, then downloads
the same files from the storage box via FTP to measure transfer speed.

Uses multiple parallel FTP connections (one per worker) to reduce total time
when many small files are transferred; latency per file is then amortized.
"""

import ftplib
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SAMPLE_SIZE = 500
REMOTE_BASE = "map tile cache"
LOCAL_CACHE = "map tile cache"
OUTPUT_DIR = os.path.join("temporary files", "cache test")
# Number of parallel FTP connections. Hetzner Storage Box allows 10 simultaneous connections
# per account (FTP, SMB mount, SSH, etc. all count). If the box is mounted elsewhere, use
# 5–8 to leave headroom so mount + this test don't exceed 10.
MAX_WORKERS = 5


def load_storage_box_credentials():
    """Read storage box connection details from config.txt."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.txt")

    creds = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if key == "storage_box_address":
                creds["address"] = value
            elif key == "storage_box_user":
                creds["user"] = value
            elif key == "storage_box_password":
                creds["password"] = value
    return creds


def collect_npy_files(cache_dir: str) -> list[str]:
    """Walk the local cache and return relative paths of all .npy files."""
    cache_path = Path(cache_dir)
    return [
        str(p.relative_to(cache_path))
        for p in cache_path.rglob("*.npy")
    ]


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes} min {secs:.2f} s"


# One FTP connection per worker thread (avoids connection setup per file).
_worker_ftp = threading.local()


def _get_worker_ftp(creds: dict):
    """Return this thread's FTP connection, creating it on first use."""
    if not getattr(_worker_ftp, "ftp", None):
        _worker_ftp.ftp = ftplib.FTP(creds["address"], timeout=30)
        _worker_ftp.ftp.login(creds["user"], creds["password"])
    return _worker_ftp.ftp


def _download_one(args: tuple) -> tuple[str, str | None]:
    """Download a single file. Returns (rel_path, error_message); (rel_path, None) on success."""
    rel_path, remote_path, local_dest, creds = args
    try:
        ftp = _get_worker_ftp(creds)
        os.makedirs(os.path.dirname(local_dest), exist_ok=True)
        with open(local_dest, "wb") as f:
            ftp.retrbinary(f"RETR {remote_path}", f.write)
        return (rel_path, None)
    except ftplib.all_errors as e:
        return (rel_path, str(e))


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, LOCAL_CACHE)
    output_dir = os.path.join(base_dir, OUTPUT_DIR)

    # --- Step 1: collect local file listing ---
    print("Scanning local map tile cache ...")
    all_files = collect_npy_files(cache_dir)
    print(f"Found {len(all_files)} .npy files in local cache.")

    if len(all_files) < SAMPLE_SIZE:
        print(f"Warning: fewer than {SAMPLE_SIZE} files available; using all {len(all_files)}.")
        selected = all_files
    else:
        selected = random.sample(all_files, SAMPLE_SIZE)

    count = len(selected)
    print(f"Selected {count} files for download test.\n")

    # --- Step 2: connect to storage box ---
    creds = load_storage_box_credentials()
    for key in ("address", "user", "password"):
        if key not in creds:
            print(f"Error: missing '{key}' in config.txt storage box settings.")
            return

    print(f"Using up to {MAX_WORKERS} parallel FTP connections.")
    print(f"Connecting to storage box at {creds['address']} ...")
    # Pre-warm one connection to verify credentials
    _ = ftplib.FTP(creds["address"], timeout=30)
    _.login(creds["user"], creds["password"])
    _.quit()
    print("Connected.\n")

    # --- Step 3: download files in parallel ---
    os.makedirs(output_dir, exist_ok=True)
    tasks = [
        (
            rel_path,
            REMOTE_BASE + "/" + rel_path.replace("\\", "/"),
            os.path.join(output_dir, rel_path),
            creds,
        )
        for rel_path in selected
    ]

    print(f"Downloading {count} tiles to: {OUTPUT_DIR}")
    print("-" * 50)

    start_time = time.perf_counter()
    downloaded = 0
    errors = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_download_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            completed += 1
            rel_path, err = future.result()
            if err:
                errors += 1
                print(f"  FAILED {rel_path}: {err}")
            else:
                downloaded += 1
            if completed % 50 == 0 or completed == count:
                elapsed = time.perf_counter() - start_time
                print(f"  [{completed}/{count}] {downloaded} OK, {errors} failed  ({format_duration(elapsed)} elapsed)")

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # --- Step 4: report ---

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Files attempted : {count}")
    print(f"Downloaded OK   : {downloaded}")
    print(f"Errors          : {errors}")
    print(f"Total time      : {format_duration(total_time)}")
    if downloaded > 0:
        avg = total_time / downloaded
        print(f"Avg per tile    : {format_duration(avg)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
