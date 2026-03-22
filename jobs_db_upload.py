#!/usr/bin/env python3
"""
Upload local jobs.db to the Hetzner Storage Box (FTP).

Remote path: renderer stats/{user}-YYYY-MM-DD.db
Existing files with the same name are replaced (FTP STOR).

Reads user and storage_box_* from config.txt via config.py.
"""

from __future__ import annotations

import ftplib
import sys
from datetime import date
from pathlib import Path

from config import config

REMOTE_DIR = "renderer stats"
SCRIPT_DIR = Path(__file__).resolve().parent
JOBS_DB = SCRIPT_DIR / "jobs.db"


def _ftp_ensure_dir(ftp: ftplib.FTP, remote_dir: str) -> None:
    """Create remote directory segments; ignore 'already exists' errors."""
    parts = remote_dir.split("/")
    current = ""
    for part in parts:
        if not part:
            continue
        current = f"{current}/{part}" if current else part
        try:
            ftp.mkd(current)
        except ftplib.error_perm:
            pass


def upload_jobs_db_to_storage(log_callback=None) -> bool:
    """
    Upload jobs.db to the storage box. Optional log_callback for messages (e.g. write_log).

    Returns:
        True if upload completed, False if skipped due to config/path or on error.
    """
    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def _err(msg: str) -> None:
        if log_callback:
            log_callback(msg)
        else:
            print(msg, file=sys.stderr)

    if not config.user:
        _err("Error: 'user' is not set in config.txt.")
        return False
    if not all(
        [config.storage_box_address, config.storage_box_user, config.storage_box_password]
    ):
        _err(
            "Error: storage box settings (storage_box_address, storage_box_user, "
            "storage_box_password) are missing in config.txt."
        )
        return False
    if not JOBS_DB.is_file():
        _err(f"Error: jobs database not found: {JOBS_DB}")
        return False

    remote_name = f"{config.user}-{date.today().isoformat()}.db"

    ftp: ftplib.FTP | None = None
    try:
        ftp = ftplib.FTP(config.storage_box_address, timeout=60)
        ftp.login(config.storage_box_user, config.storage_box_password)
        _ftp_ensure_dir(ftp, REMOTE_DIR)
        ftp.cwd(REMOTE_DIR)
        with open(JOBS_DB, "rb") as f:
            ftp.storbinary(f"STOR {remote_name}", f)
    except ftplib.all_errors as e:
        _err(f"FTP error uploading jobs.db: {e}")
        return False
    except OSError as e:
        _err(f"Local file error uploading jobs.db: {e}")
        return False
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except Exception:
                try:
                    ftp.close()
                except Exception:
                    pass

    _log(f"Uploaded jobs.db to storage box: {REMOTE_DIR}/{remote_name}")
    return True


def main() -> int:
    return 0 if upload_jobs_db_to_storage() else 1


if __name__ == "__main__":
    raise SystemExit(main())
