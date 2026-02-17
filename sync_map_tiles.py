"""
Map Tile Cache Syncing Module
This module handles syncing of map tile cache files between local machine and storage box.
Syncs only folders that have changed (by last modified time), using a manual folder list
and a synctime tracking file. Uses rsync over SSH for efficient file synchronization.
"""

# Standard library imports
import subprocess
import sys
import os
import re
from pathlib import Path


# Paths relative to this file's directory (project root)
def _utils_dir():
    return Path(__file__).resolve().parent / "utils"


def _folders_list_path():
    return _utils_dir() / "map_tile_folders.txt"


def _synctime_path():
    return _utils_dir() / "map_tile_folders_synctime.txt"


def _load_folder_list():
    """Load list of folder paths (relative to map tile cache) from utils/map_tile_folders.txt."""
    path = _folders_list_path()
    if not path.exists():
        return []
    folders = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                folders.append(line)
    return folders


def _load_synctime():
    """
    Load local and remote last-modified times from utils/map_tile_folders_synctime.txt.
    Returns (dict_local, dict_remote) where each dict maps folder -> float timestamp.
    """
    path = _synctime_path()
    local_times = {}
    remote_times = {}
    if not path.exists():
        return local_times, remote_times
    section = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "[local]":
                section = "local"
                continue
            if line.strip() == "[remote]":
                section = "remote"
                continue
            if not line.strip() or section is None:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            folder, ts_str = parts[0].strip(), parts[1].strip()
            try:
                ts = float(ts_str)
            except ValueError:
                continue
            if section == "local":
                local_times[folder] = ts
            elif section == "remote":
                remote_times[folder] = ts
    return local_times, remote_times


def _save_synctime(local_times, remote_times):
    """Write utils/map_tile_folders_synctime.txt with [local] and [remote] sections."""
    path = _synctime_path()
    _utils_dir().mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[local]\n")
        for folder in sorted(local_times.keys()):
            f.write(f"{folder}\t{local_times[folder]}\n")
        f.write("[remote]\n")
        for folder in sorted(remote_times.keys()):
            f.write(f"{folder}\t{remote_times[folder]}\n")


def _get_local_folder_mtime(local_cache_dir, folder_path):
    """
    Return last modified time of a folder under local_cache_dir.
    Works on both Windows and Linux. Returns None if path is not a directory.
    """
    full = Path(local_cache_dir) / folder_path
    if not full.is_dir():
        return None
    try:
        return full.stat().st_mtime
    except OSError:
        return None


class MapTileSyncer:
    """Handles syncing of map tile cache files between local and remote storage.
    Only syncs folders that have changed (by mtime) according to the synctime file.
    """

    def __init__(self, storage_box_address, storage_box_user, storage_box_password,
                 local_cache_dir="map tile cache", log_callback=None, progress_callback=None,
                 sync_state_callback=None, debug_callback=None, debug_logging=False):
        self.storage_box_address = storage_box_address
        self.storage_box_user = storage_box_user
        self.storage_box_password = storage_box_password
        self.local_cache_dir = Path(local_cache_dir)
        self.log_callback = log_callback or (lambda msg: None)
        self.progress_callback = progress_callback or (lambda msg: None)
        self.sync_state_callback = sync_state_callback or (lambda state: None)
        self.debug_callback = debug_callback or log_callback or (lambda msg: None)
        self.debug_logging = debug_logging

        self.remote_base_path = "map tile cache"
        self._rsync_cmd = None
        self._ssh_key_message_shown = False
        self._wsl_sshpass_available = None

    def sync(self, max_workers=10, dry_run=False, upload_only=False):
        """
        Determine which folders need syncing (by mtime), then rsync only those folders.
        Before logic: run cache sweep (caller's responsibility). We ensure local cache exists.
        Returns (success, uploaded_count, downloaded_count).
        """
        try:
            rsync_cmd = self._find_rsync()
            if not rsync_cmd:
                self.log_callback("rsync is not available. Please install rsync (Git for Windows includes it) or set up WSL.")
                self.sync_state_callback("complete")
                return False, 0, 0
            self._rsync_cmd = rsync_cmd

            self.sync_state_callback("start")

            if dry_run:
                self.debug_callback("Starting map tile cache file count check")
            elif upload_only:
                self.debug_callback("Starting map tile cache upload-only sync")
            else:
                self.debug_callback("Starting map tile cache sync")

            if not self.local_cache_dir.exists():
                self.debug_callback("Creating local map tile cache directory.")
                self.local_cache_dir.mkdir(parents=True, exist_ok=True)

            folder_list = _load_folder_list()
            if not folder_list:
                self.log_callback("No folders listed in utils/map_tile_folders.txt. Skipping sync.")
                self.sync_state_callback("complete")
                return True, 0, 0

            local_times, remote_times = _load_synctime()

            # Compare remote folder mtimes; mark need_sync and update remote_times in memory
            need_sync = set()
            remote_mtimes = self._get_remote_folder_mtimes(folder_list)
            for folder in folder_list:
                rmt = remote_mtimes.get(folder)
                if rmt is None:
                    continue
                prev = remote_times.get(folder)
                if prev is None or rmt > prev:
                    need_sync.add(folder)
                    remote_times[folder] = rmt

            # Compare local folder mtimes; mark need_sync (do not update file yet)
            for folder in folder_list:
                lmt = _get_local_folder_mtime(self.local_cache_dir, folder)
                if lmt is None:
                    continue
                prev = local_times.get(folder)
                if prev is None or lmt > prev:
                    need_sync.add(folder)

            need_sync_list = sorted(need_sync)

            if self.debug_logging:
                if need_sync_list:
                    self.debug_callback("Folders needing sync: " + ", ".join(need_sync_list))
                else:
                    self.debug_callback("No folders need syncing.")
            else:
                if need_sync_list:
                    self.log_callback(f"{len(need_sync_list)} folder(s) need syncing.")

            uploaded_count = 0
            downloaded_count = 0

            if need_sync_list:
                if dry_run:
                    uploaded_count, downloaded_count = self._rsync_folders_dry_run(
                        need_sync_list, upload_only=upload_only
                    )
                    self.progress_callback(
                        f"Map tile cache check: {uploaded_count} files would be uploaded"
                        + (f", {downloaded_count} would be downloaded" if not upload_only else "")
                    )
                else:
                    # Always sync both ways so local and remote both get all files
                    uploaded_count, downloaded_count = self._rsync_folders(
                        need_sync_list, dry_run=False, upload_only=False
                    )
                    # Update local mtimes in tracking file after sync
                    for folder in need_sync_list:
                        lmt = _get_local_folder_mtime(self.local_cache_dir, folder)
                        if lmt is not None:
                            local_times[folder] = lmt
                    _save_synctime(local_times, remote_times)
                    self.progress_callback(
                        f"Syncing map tile cache: completed ({uploaded_count} uploaded, {downloaded_count} downloaded)"
                    )
            else:
                if dry_run:
                    self.progress_callback("Map tile cache check: no folders need syncing.")

            if not dry_run:
                if upload_only:
                    self.debug_callback(f"Map tile cache upload-only sync completed. Uploaded: {uploaded_count}")
                else:
                    self.debug_callback(f"Map tile cache sync completed. Uploaded: {uploaded_count}, downloaded: {downloaded_count}")

            self.sync_state_callback("complete")
            return True, uploaded_count, downloaded_count

        except Exception as e:
            self.log_callback(f"Map tile cache sync failed: {str(e)}")
            self.sync_state_callback("complete")
            return False, 0, 0

    def _find_rsync(self):
        """Find rsync command (Git for Windows or WSL). Returns list of command parts or None."""
        try:
            subprocess.run(["rsync", "--version"], capture_output=True, check=True, timeout=5)
            return ["rsync"]
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if sys.platform == "win32":
            for git_path in [
                r"C:\Program Files\Git\usr\bin\rsync.exe",
                r"C:\Program Files (x86)\Git\usr\bin\rsync.exe",
                r"C:\Program Files\Git\bin\rsync.exe",
            ]:
                if Path(git_path).exists():
                    try:
                        subprocess.run([git_path, "--version"], capture_output=True, check=True, timeout=5)
                        self.debug_callback(f"Found rsync at: {git_path}")
                        return [git_path]
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        continue
            try:
                r = subprocess.run(["wsl", "which", "rsync"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    self.debug_callback("Using rsync via WSL")
                    try:
                        r2 = subprocess.run(["wsl", "which", "sshpass"], capture_output=True, timeout=5)
                        self._wsl_sshpass_available = r2.returncode == 0
                    except Exception:
                        self._wsl_sshpass_available = False
                    return ["wsl", "rsync"]
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return None

    def _convert_path_for_wsl(self, path):
        """Convert Windows path to WSL path."""
        path = Path(path).resolve()
        drive = path.drive.replace(":", "").lower()
        rest = str(path.relative_to(path.anchor)).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    def _run_ssh_command(self, remote_command, input_data=None):
        """Run SSH command with same auth as rsync (sshpass, port 23). Returns CompletedProcess."""
        using_wsl = self._rsync_cmd and self._rsync_cmd[0] == "wsl"

        if using_wsl:
            if self._wsl_sshpass_available is None:
                try:
                    r = subprocess.run(["wsl", "which", "sshpass"], capture_output=True, timeout=5)
                    self._wsl_sshpass_available = r.returncode == 0
                except Exception:
                    self._wsl_sshpass_available = False
            if self._wsl_sshpass_available:
                cmd = [
                    "wsl", "sshpass", "-p", self.storage_box_password,
                    "ssh", "-p", "23", "-o", "StrictHostKeyChecking=no",
                    "-o", "Compression=yes", "-o", "CompressionLevel=6",
                    f"{self.storage_box_user}@{self.storage_box_address}",
                    remote_command,
                ]
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=False,
                    input=input_data.encode("utf-8") if input_data and isinstance(input_data, str) else input_data,
                    timeout=60,
                )
            # No sshpass in WSL
            if not self._ssh_key_message_shown:
                self.log_callback("sshpass not found in WSL. Please install: wsl sudo apt-get install -y sshpass")
                self._ssh_key_message_shown = True
            class F:
                returncode = 1
                stdout = b""
                stderr = b"sshpass not found"
            return F()

        # Native Windows or Linux
        ssh_cmd = [
            "ssh", "-p", "23", "-o", "StrictHostKeyChecking=no",
            "-o", "Compression=yes", "-o", "CompressionLevel=6",
            f"{self.storage_box_user}@{self.storage_box_address}",
            remote_command,
        ]
        sshpass_cmd = None
        if sys.platform == "win32":
            for p in [r"C:\Program Files\Git\usr\bin\sshpass.exe", r"C:\Program Files (x86)\Git\usr\bin\sshpass.exe"]:
                if Path(p).exists():
                    sshpass_cmd = p
                    break
        if not sshpass_cmd:
            try:
                subprocess.run(["sshpass", "-V"], capture_output=True, check=True, timeout=5)
                sshpass_cmd = "sshpass"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
        if sshpass_cmd:
            cmd = [sshpass_cmd, "-p", self.storage_box_password] + ssh_cmd
        else:
            if not self._ssh_key_message_shown:
                self.log_callback("sshpass not found. Install sshpass or set up SSH keys.")
                self._ssh_key_message_shown = True
            cmd = ssh_cmd
        return subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            input=input_data.encode("utf-8") if input_data and isinstance(input_data, str) else input_data,
            timeout=60,
        )

    def _get_remote_folder_mtimes(self, folder_list):
        """Get last modified time of each folder on the remote (over SSH). Returns dict folder -> float mtime."""
        if not folder_list:
            return {}
        # Remote script: read folder names from stdin, output "mtime folder" per line for existing dirs
        base = "map tile cache"
        script = (
            f'base="{base}"; while IFS= read -r folder; do '
            '[ -d "$base/$folder" ] && stat -c "%Y $folder" "$base/$folder"; done'
        )
        stdin = "\n".join(folder_list)
        result = self._run_ssh_command(script, input_data=stdin)
        out = {}
        if result.returncode != 0:
            return out
        for line in (result.stdout or b"").decode("utf-8", errors="ignore").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                ts = float(parts[0])
                folder = parts[1]
                out[folder] = ts
            except ValueError:
                continue
        return out

    def _run_rsync_with_password(self, cmd, capture_output=False, input_data=None):
        """Run rsync with password (sshpass). Same behavior as before for WSL vs native."""
        using_wsl = cmd and cmd[0] == "wsl"
        if using_wsl:
            if self._wsl_sshpass_available is None:
                try:
                    r = subprocess.run(["wsl", "which", "sshpass"], capture_output=True, timeout=5)
                    self._wsl_sshpass_available = r.returncode == 0
                except Exception:
                    self._wsl_sshpass_available = False
            if self._wsl_sshpass_available:
                full_cmd = ["wsl", "sshpass", "-p", self.storage_box_password] + cmd[1:]
                return subprocess.run(
                    full_cmd,
                    capture_output=capture_output,
                    text=False if capture_output else True,
                    input=input_data.encode("utf-8") if input_data and isinstance(input_data, str) else input_data,
                    timeout=300,
                )
            if not self._ssh_key_message_shown:
                self.log_callback("sshpass not found in WSL. Please install: wsl sudo apt-get install -y sshpass")
                self._ssh_key_message_shown = True
            class F:
                returncode = 1
                stdout = b""
                stderr = b"sshpass not found"
            return F()

        sshpass_cmd = None
        if sys.platform == "win32":
            for p in [r"C:\Program Files\Git\usr\bin\sshpass.exe", r"C:\Program Files (x86)\Git\usr\bin\sshpass.exe"]:
                if Path(p).exists():
                    sshpass_cmd = p
                    break
        if not sshpass_cmd:
            try:
                subprocess.run(["sshpass", "-V"], capture_output=True, check=True, timeout=5)
                sshpass_cmd = "sshpass"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
        if sshpass_cmd:
            full_cmd = [sshpass_cmd, "-p", self.storage_box_password] + cmd
        else:
            if not self._ssh_key_message_shown:
                self.log_callback("sshpass not found. Install sshpass or set up SSH keys.")
                self._ssh_key_message_shown = True
            full_cmd = cmd
        return subprocess.run(
            full_cmd,
            capture_output=capture_output,
            text=False if capture_output else True,
            input=input_data.encode("utf-8") if input_data and isinstance(input_data, str) else input_data,
            timeout=300,
        )

    def _rsync_one_folder(self, folder, dry_run=False, upload_only=False):
        """
        Rsync a single folder: upload then download so both sides get all files.
        upload_only is only used for dry_run (e.g. exit check: count files that would be uploaded).
        Tiles are immutable (no version history), so we use --ignore-existing to only transfer
        files that don't exist on the destination (no modification checks).
        Returns (uploaded, downloaded).
        """
        local_path = self.local_cache_dir / folder
        remote_path = f"{self.storage_box_user}@{self.storage_box_address}:{self.remote_base_path}/{folder}/"
        if self._rsync_cmd[0] == "wsl":
            local_path_str = self._convert_path_for_wsl(local_path) + "/"
        else:
            local_path_str = str(local_path) + "/"

        # Only transfer files that don't exist on destination; no modification checks (tiles are immutable)
        base_opts = [
            "--recursive", "--links", "--perms", "--partial", "--stats", "--no-motd", "--human-readable",
            "--prune-empty-dirs", "--ignore-existing", "--no-inc-recursive",
            "-e", "ssh -p 23 -o StrictHostKeyChecking=no -o Compression=yes -o CompressionLevel=6",
        ]
        if dry_run:
            base_opts.append("--dry-run")
            base_opts.append("--itemize-changes")

        uploaded = 0
        downloaded = 0

        # Ensure local folder exists for upload
        local_path.mkdir(parents=True, exist_ok=True)

        if self.debug_logging and not dry_run:
            self.debug_callback(f"Syncing folder: {folder}")

        upload_cmd = self._rsync_cmd + base_opts + [local_path_str, remote_path]
        res = self._run_rsync_with_password(upload_cmd, capture_output=True)
        if res.returncode == 0:
            uploaded = self._count_rsync_transfers(res.stdout.decode("utf-8", errors="ignore"), "upload")
        else:
            err = res.stderr.decode("utf-8", errors="ignore") if res.stderr else "Unknown error"
            self.log_callback(f"rsync upload failed for {folder}: {err}")
            return 0, 0

        # Always download as well when not dry_run (both sides get all files)
        if not dry_run or not upload_only:
            download_cmd = self._rsync_cmd + base_opts + [remote_path, local_path_str]
            res = self._run_rsync_with_password(download_cmd, capture_output=True)
            if res.returncode == 0:
                downloaded = self._count_rsync_transfers(res.stdout.decode("utf-8", errors="ignore"), "download")
            else:
                err = res.stderr.decode("utf-8", errors="ignore") if res.stderr else "Unknown error"
                self.log_callback(f"rsync download failed for {folder}: {err}")

        return uploaded, downloaded

    def _rsync_folders(self, folder_list, dry_run=False, upload_only=False):
        """Rsync each folder in the list. Returns (total_uploaded, total_downloaded)."""
        total_up = 0
        total_down = 0
        for folder in folder_list:
            up, down = self._rsync_one_folder(folder, dry_run=dry_run, upload_only=upload_only)
            total_up += up
            total_down += down
        return total_up, total_down

    def _rsync_folders_dry_run(self, folder_list, upload_only=False):
        """Run rsync --dry-run on each folder and sum counts."""
        return self._rsync_folders(folder_list, dry_run=True, upload_only=upload_only)

    def _count_rsync_transfers(self, output, direction="upload"):
        """Count files transferred from rsync --stats output."""
        for pattern in [
            r"Number of regular files transferred:\s*([\d,]+)",
            r"Number of files transferred:\s*([\d,]+)",
        ]:
            m = re.search(pattern, output, re.IGNORECASE)
            if m:
                return int(m.group(1).replace(",", ""))
        count = 0
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith(">f") or line.startswith("<f"):
                if not line.startswith(">d") and not line.startswith("<d"):
                    count += 1
        return count


def sync_map_tiles(
    storage_box_address,
    storage_box_user,
    storage_box_password,
    local_cache_dir="map tile cache",
    log_callback=None,
    progress_callback=None,
    sync_state_callback=None,
    max_workers=10,
    dry_run=False,
    upload_only=False,
    debug_callback=None,
    debug_logging=False,
):
    """
    Sync map tiles: only folders that have changed (by mtime) are rsynced.
    Folder list: utils/map_tile_folders.txt. Synctime tracking: utils/map_tile_folders_synctime.txt.
    Returns (success, uploaded_count, downloaded_count).
    """
    syncer = MapTileSyncer(
        storage_box_address=storage_box_address,
        storage_box_user=storage_box_user,
        storage_box_password=storage_box_password,
        local_cache_dir=local_cache_dir,
        log_callback=log_callback,
        progress_callback=progress_callback,
        sync_state_callback=sync_state_callback,
        debug_callback=debug_callback,
        debug_logging=debug_logging,
    )
    return syncer.sync(max_workers=max_workers, dry_run=dry_run, upload_only=upload_only)
