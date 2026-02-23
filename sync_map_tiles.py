"""
Map Tile Cache Syncing Module
This module handles syncing of map tile cache files between local machine and storage box.
Syncs only folders that have changed (by last modified time), using a manual folder list
and a synctime tracking file. Uses rsync over SSH for efficient file synchronization.

Remote folder mtimes are read from a server-written mtimes.txt in the map tile cache root
(storage box SSH does not provide a shell). The client rsyncs that file and parses it.

SSH/rsync use IPv4 only on Windows (AddressFamily=inet) to avoid the IPV6_TCLASS
"Operation not permitted" socket error on some environments. On Linux, both IPv4 and
IPv6 are allowed for storage box connections.

Typically runs integrated into the rendering client, but can also be run standalone.
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


def _local_mtimes_path():
    return _utils_dir() / "mtimes_local.txt"


def _remote_mtimes_path():
    return _utils_dir() / "mtimes_remote.txt"


def _remote_mtimes_last_path():
    return _utils_dir() / "mtimes_remote_last.txt"


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


def _load_local_mtimes():
    """Load local folder mtimes from utils/mtimes_local.txt. Returns dict folder -> float timestamp."""
    path = _local_mtimes_path()
    out = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            folder, ts_str = parts[0].strip(), parts[1].strip()
            try:
                out[folder] = float(ts_str)
            except ValueError:
                continue
    return out


def _save_local_mtimes(local_times):
    """Write utils/mtimes_local.txt (folder\\ttimestamp per line)."""
    path = _local_mtimes_path()
    _utils_dir().mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for folder in sorted(local_times.keys()):
            f.write(f"{folder}\t{local_times[folder]}\n")


def _load_remote_mtimes_last():
    """Load last known remote mtimes from utils/mtimes_remote_last.txt (for comparison). Returns dict."""
    path = _remote_mtimes_last_path()
    out = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            folder, ts_str = parts[0].strip(), parts[1].strip()
            try:
                out[folder] = float(ts_str)
            except ValueError:
                continue
    return out


def _save_remote_mtimes_last(remote_times):
    """Write utils/mtimes_remote_last.txt (folder\\ttimestamp per line) for next-run comparison."""
    path = _remote_mtimes_last_path()
    _utils_dir().mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
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

            local_times = _load_local_mtimes()
            remote_times = _load_remote_mtimes_last()

            # Compare remote folder mtimes; mark need_sync and update remote_times in memory
            need_sync = set()
            remote_mtimes = self._get_remote_folder_mtimes(folder_list)
            for folder in folder_list:
                rmt = remote_mtimes.get(folder)
                prev_remote = remote_times.get(folder)
                if prev_remote is None:
                    # We don't know the remote state (e.g. [remote] empty or new folder) → need sync
                    need_sync.add(folder)
                    if rmt is not None:
                        remote_times[folder] = rmt
                elif rmt is not None and rmt > prev_remote:
                    need_sync.add(folder)
                    remote_times[folder] = rmt

            # Compare local folder mtimes; mark need_sync (do not update file yet)
            for folder in folder_list:
                lmt = _get_local_folder_mtime(self.local_cache_dir, folder)
                prev_local = local_times.get(folder)
                if prev_local is None:
                    # We don't know the local state or never recorded → need sync
                    need_sync.add(folder)
                elif lmt is not None and lmt > prev_local:
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
                    # Re-fetch mtimes.txt so we have current server state; save as last-known for next run
                    post_remote = self._get_remote_folder_mtimes(need_sync_list)
                    for folder in need_sync_list:
                        rmt = post_remote.get(folder)
                        if rmt is not None:
                            remote_times[folder] = rmt
                    _save_local_mtimes(local_times)
                    _save_remote_mtimes_last(remote_times)
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

    def _ssh_options(self):
        """SSH -o options list. On Windows we add AddressFamily=inet to avoid IPV6_TCLASS errors."""
        opts = ["-o", "StrictHostKeyChecking=no", "-o", "Compression=yes"]
        if sys.platform == "win32":
            opts.extend(["-o", "AddressFamily=inet"])
        return opts

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
                    "ssh", "-p", "23",
                ] + self._ssh_options() + [
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
            "ssh", "-p", "23",
        ] + self._ssh_options() + [
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
        """
        Get last modified times from the server-written mtimes.txt in the map tile cache root.
        Storage box SSH does not provide a shell, so we rsync that file and parse it.
        Format: one line per folder, "folder_path\\ttimestamp" (tab-separated).
        Returns dict folder -> float mtime.
        """
        if not folder_list:
            return {}
        n = len(folder_list)
        if self.debug_logging:
            self.debug_callback("Getting remote folder mtimes from mtimes.txt (rsync)...")
        local_file = _remote_mtimes_path()
        _utils_dir().mkdir(parents=True, exist_ok=True)
        remote_spec = f"{self.storage_box_user}@{self.storage_box_address}:map tile cache/mtimes.txt"
        if self._rsync_cmd and self._rsync_cmd[0] == "wsl":
            local_path_str = self._convert_path_for_wsl(local_file)
        else:
            local_path_str = str(local_file)
        cmd = self._rsync_cmd + [
            "-e", "ssh -p 23 " + " ".join(self._ssh_options()),
            "--no-motd",
            remote_spec,
            local_path_str,
        ]
        result = self._run_rsync_with_password(cmd, capture_output=True)
        out = {}
        if result.returncode != 0:
            if self.debug_logging:
                err = (result.stderr or b"").decode("utf-8", errors="ignore").strip()
                self.debug_callback(f"Failed to fetch mtimes.txt (returncode={result.returncode}). {err[:300]}")
            return out
        if not local_file.exists():
            if self.debug_logging:
                self.debug_callback("mtimes.txt not found after rsync.")
            return out
        try:
            raw = local_file.read_text(encoding="utf-8")
        except Exception as e:
            if self.debug_logging:
                self.debug_callback(f"Could not read mtimes_remote.txt: {e}")
            return out
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            folder, ts_str = parts[0].strip(), parts[1].strip()
            try:
                out[folder] = float(ts_str)
            except ValueError:
                continue
        if self.debug_logging:
            got = len(out)
            self.debug_callback(f"Remote mtimes: got {got} folders from mtimes.txt.")
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

        # Only transfer files that don't exist on destination. No modification checks because tiles are immutable
        base_opts = [
            "--recursive", "--links", "--perms", "--partial", "--no-motd",
            "--prune-empty-dirs", "--ignore-existing", "--no-inc-recursive",
            "-e", "ssh -p 23 " + " ".join(self._ssh_options()),
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
    Folder list: utils/map_tile_folders.txt. Local mtimes: utils/mtimes_local.txt. Remote mtimes: server writes mtimes.txt, client uses utils/mtimes_remote.txt and utils/mtimes_remote_last.txt.
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


def _run_standalone():
    """
    Run map tile sync as a standalone script.
    Loads config, runs cache sweep, then syncs - same flow as bootup.
    """
    from config import config
    from write_log import write_debug_log, write_log

    write_log("Map tile cache sync (standalone)")

    if not config.sync_map_tile_cache:
        write_log("Map tile cache syncing is disabled in config. Exiting.")
        sys.exit(0)

    if not all([config.storage_box_address, config.storage_box_user, config.storage_box_password]):
        write_log("Cannot sync: missing storage box credentials in config.txt.")
        sys.exit(1)

    try:
        write_log("Cleaning bad map tiles from local cache.")
        import map_tile_cache_sweep
        map_tile_cache_sweep.main()
    except Exception as sweep_error:
        write_log(f"Warning: Cache cleaning failed: {str(sweep_error)}")
        # Continue with sync even if sweep fails

    write_log("Checking and uploading map tiles")
    success, uploaded_count, downloaded_count = sync_map_tiles(
        storage_box_address=config.storage_box_address,
        storage_box_user=config.storage_box_user,
        storage_box_password=config.storage_box_password,
        local_cache_dir="map tile cache",
        log_callback=write_log,
        progress_callback=write_log,
        sync_state_callback=lambda state: None,
        max_workers=10,
        dry_run=False,
        upload_only=False,
        debug_callback=write_debug_log,
        debug_logging=config.debug_logging,
    )

    if success:
        if uploaded_count > 0 or downloaded_count > 0:
            write_log(f"Uploaded {uploaded_count} tiles, downloaded {downloaded_count} tiles")
        write_log("Map tile cache sync completed successfully.")
        sys.exit(0)
    else:
        write_log("Map tile cache sync failed.")
        sys.exit(1)


if __name__ == "__main__":
    _run_standalone()
