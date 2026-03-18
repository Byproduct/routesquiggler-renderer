"""
SQLite job logging helpers for the terminal-only renderer.

This module is intentionally lightweight:
- Create `jobs.db` + `jobs` table if needed
- Insert one row per successfully completed job (server status == "ok")
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


def _default_db_path() -> Path:
    # Keep the DB next to the client code so relative working directories
    # don't affect where the file ends up.
    return Path(__file__).resolve().parent / "jobs.db"


def _normalize_starting_time(starting_time: Union[float, int, str, datetime]) -> str:
    if isinstance(starting_time, datetime):
        dt = starting_time
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.isoformat()

    if isinstance(starting_time, (float, int)):
        return datetime.fromtimestamp(float(starting_time), tz=timezone.utc).isoformat()

    # Assume it's already a string timestamp.
    return str(starting_time)


def _ensure_jobs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            starting_time TEXT NOT NULL,
            elapsed_time REAL NOT NULL,
            user_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            job_type TEXT NOT NULL,
            tiles_local INTEGER NOT NULL,
            tiles_remote INTEGER NOT NULL,
            tiles_service INTEGER NOT NULL
        )
        """
    )


def log_completed_job_to_db(
    *,
    starting_time: Union[float, int, str, datetime],
    elapsed_time: Union[float, int],
    user_id: int,
    job_id: int,
    job_type: str,
    tiles_local: int,
    tiles_remote: int,
    tiles_service: int,
    db_path: Optional[Path] = None,
) -> None:
    """
    Insert a completed job row into `jobs.db`.

    Only the insert side is handled here; deciding "ok vs error" is done
    by the caller (terminal loop + worker job_completed_ok flag).
    """

    db_file = Path(db_path) if db_path is not None else _default_db_path()
    starting_time_str = _normalize_starting_time(starting_time)

    # Ensure parent dir exists (it should, but keep it safe).
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file), timeout=30)
    try:
        _ensure_jobs_table(conn)
        conn.execute(
            """
            INSERT INTO jobs (
                starting_time,
                elapsed_time,
                user_id,
                job_id,
                job_type,
                tiles_local,
                tiles_remote,
                tiles_service
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                starting_time_str,
                float(elapsed_time),
                int(user_id),
                int(job_id),
                str(job_type),
                int(tiles_local),
                int(tiles_remote),
                int(tiles_service),
            ),
        )
        conn.commit()
    finally:
        conn.close()

