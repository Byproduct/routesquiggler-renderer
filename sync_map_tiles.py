"""
Map Tile Cache Syncing Module
This module handles syncing of map tile cache files between local machine and storage box.
Can be used independently or called from other parts of the application.
Uses rsync over SSH for efficient file synchronization.
"""

# Standard library imports
import subprocess
import sys
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


class MapTileSyncer:
    """Handles syncing of map tile cache files between local and remote storage."""
    
    def __init__(self, storage_box_address, storage_box_user, storage_box_password, 
                 local_cache_dir="map tile cache", log_callback=None, progress_callback=None, sync_state_callback=None, debug_callback=None):
        """
        Initialize the MapTileSyncer.
        
        Args:
            storage_box_address: Storage box SSH server address (e.g., u465900.your-storagebox.de)
            storage_box_user: SSH username (same as FTP username)
            storage_box_password: SSH password (same as FTP password)
            local_cache_dir: Local directory for map tile cache (default: "map tile cache")
            log_callback: Function to call for main logging messages (upload/download counts)
            progress_callback: Function to call for progress updates
            sync_state_callback: Function to call when syncing starts/completes (called with 'start' or 'complete')
            debug_callback: Function to call for debug logging messages (defaults to log_callback if not provided)
        """
        self.storage_box_address = storage_box_address
        self.storage_box_user = storage_box_user
        self.storage_box_password = storage_box_password
        self.local_cache_dir = Path(local_cache_dir)
        self.log_callback = log_callback or (lambda msg: None)
        self.progress_callback = progress_callback or (lambda msg: None)
        self.sync_state_callback = sync_state_callback or (lambda state: None)
        self.debug_callback = debug_callback or log_callback or (lambda msg: None)
        
        # Remote path on storage box
        self.remote_base_path = "map tile cache"
        
        # Find rsync command (will be set on first use)
        self._rsync_cmd = None
        
        # Track if we've shown Windows/SSH key message (to avoid spam)
        self._ssh_key_message_shown = False
        
        # Track sshpass availability for WSL
        self._wsl_sshpass_available = None

    def sync(self, max_workers=10, dry_run=False, upload_only=False):
        """
        Perform the complete sync operation using rsync.
        
        Args:
            max_workers: Maximum number of parallel threads for file operations (used for directory grouping)
            dry_run: If True, only check file counts without performing sync operations
            upload_only: If True, only upload files to remote server (no downloads)
            
        Returns:
            tuple: (success, uploaded_count, downloaded_count)
        """
        try:
            # Find rsync command (checks common locations and WSL)
            rsync_cmd = self._find_rsync()
            if not rsync_cmd:
                error_msg = "rsync is not available. Please install rsync (Git for Windows includes it) or set up WSL."
                self.log_callback(error_msg)
                self.sync_state_callback('complete')
                return False, 0, 0
            self._rsync_cmd = rsync_cmd
            
            # Notify that syncing is starting
            self.sync_state_callback('start')
            
            if dry_run:
                self.debug_callback("Starting map tile cache file count check")
            elif upload_only:
                self.debug_callback("Starting map tile cache upload-only sync")
            else:
                self.debug_callback("Starting map tile cache sync")
            
            # Ensure local cache directory exists
            if not self.local_cache_dir.exists():
                self.debug_callback("Creating local map tile cache directory.")
                self.local_cache_dir.mkdir(parents=True, exist_ok=True)

            # Use rsync's native sync capabilities - sync entire cache directory at once
            # rsync will automatically figure out what needs to be transferred
            uploaded_count, downloaded_count = self._sync_cache_directory_rsync(dry_run=dry_run, upload_only=upload_only)

            if dry_run:
                # For dry run, rsync --dry-run tells us what would be transferred
                if upload_only:
                    self.debug_callback(f"File count check completed (dry run, upload-only mode - no files were transferred)")
                    self.progress_callback(f"Map tile cache check: {uploaded_count} files would be uploaded, no downloads")
                else:
                    self.debug_callback(f"File count check completed (dry run - no files were transferred)")
                    self.progress_callback(f"Map tile cache check: {uploaded_count} files would be uploaded, {downloaded_count} would be downloaded")

            if upload_only:
                self.debug_callback(f"Map tile cache upload-only sync completed. Uploaded: {uploaded_count}")
                self.progress_callback(f"Syncing map tile cache: completed ({uploaded_count} uploaded, upload-only mode)")
            else:
                self.debug_callback(f"Map tile cache sync completed. Uploaded: {uploaded_count}, downloaded: {downloaded_count}")
                self.progress_callback(f"Syncing map tile cache: completed ({uploaded_count} uploaded, {downloaded_count} downloaded)")
            
            # Notify that syncing is complete
            self.sync_state_callback('complete')
            
            return True, uploaded_count, downloaded_count

        except Exception as e:
            self.log_callback(f"Map tile cache sync failed: {str(e)}")
            # Notify that syncing is complete (even on error)
            self.sync_state_callback('complete')
            return False, 0, 0

    def _find_rsync(self):
        """
        Find rsync command, checking common Git for Windows locations and WSL.
        
        Returns:
            List of command parts (e.g., ['rsync'] or ['wsl', 'rsync']) or None if not found
        """
        # First, try standard 'rsync' command (if in PATH)
        try:
            subprocess.run(['rsync', '--version'], capture_output=True, check=True, timeout=5)
            return ['rsync']
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Check common Git for Windows locations
        if sys.platform == 'win32':
            common_git_paths = [
                r'C:\Program Files\Git\usr\bin\rsync.exe',
                r'C:\Program Files (x86)\Git\usr\bin\rsync.exe',
                r'C:\Program Files\Git\bin\rsync.exe',
            ]
            
            for git_path in common_git_paths:
                if Path(git_path).exists():
                    try:
                        subprocess.run([git_path, '--version'], capture_output=True, check=True, timeout=5)
                        self.debug_callback(f"Found rsync at: {git_path}")
                        return [git_path]
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        continue
            
            # Try WSL as fallback
            try:
                # Check if WSL is available and has rsync
                result = subprocess.run(['wsl', 'which', 'rsync'], capture_output=True, timeout=5)
                if result.returncode == 0:
                    self.debug_callback("Using rsync via WSL")
                    # Check if sshpass is available in WSL
                    try:
                        sshpass_result = subprocess.run(['wsl', 'which', 'sshpass'], capture_output=True, timeout=5)
                        if sshpass_result.returncode == 0:
                            self._wsl_sshpass_available = True
                            self.debug_callback("sshpass available in WSL - will use for password authentication")
                        else:
                            self._wsl_sshpass_available = False
                            if not self._ssh_key_message_shown:
                                self.debug_callback("sshpass not found in WSL. Installing sshpass in WSL for password authentication...")
                                self.debug_callback("Run this in WSL to install: sudo apt-get update && sudo apt-get install -y sshpass")
                                self._ssh_key_message_shown = True
                    except:
                        self._wsl_sshpass_available = False
                    return ['wsl', 'rsync']
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        return None

    def _convert_path_for_wsl(self, windows_path):
        """
        Convert Windows path to WSL path format.
        Only needed when using WSL rsync.
        
        Args:
            windows_path: Windows path string
            
        Returns:
            WSL path string (e.g., /mnt/c/Users/...)
        """
        if not isinstance(windows_path, str):
            windows_path = str(windows_path)
        
        # Convert Windows path to WSL path
        # C:\Users\... -> /mnt/c/Users/...
        path = Path(windows_path).resolve()
        drive = path.drive.replace(':', '').lower()
        rest = str(path.relative_to(path.anchor)).replace('\\', '/')
        return f'/mnt/{drive}/{rest}'

    def _run_rsync_with_password(self, cmd, capture_output=False, input_data=None):
        """
        Run rsync command with password authentication using sshpass.
        Hardcodes password for convenience (user's personal project).
        
        Args:
            cmd: List of command arguments for rsync
            capture_output: If True, capture stdout/stderr
            input_data: Optional bytes/string to send to stdin (for --files-from=-)
            
        Returns:
            CompletedProcess object
        """
        # Check if using WSL rsync
        using_wsl = cmd and cmd[0] == 'wsl'
        
        if using_wsl:
            # Using WSL rsync - check if sshpass is available in WSL
            if self._wsl_sshpass_available is None:
                # Check if sshpass exists in WSL
                try:
                    check_result = subprocess.run(['wsl', 'which', 'sshpass'], capture_output=True, timeout=5)
                    self._wsl_sshpass_available = (check_result.returncode == 0)
                except:
                    self._wsl_sshpass_available = False
            
            if self._wsl_sshpass_available:
                # Use WSL sshpass for password authentication
                # cmd is ['wsl', 'rsync', ...], we need ['wsl', 'sshpass', '-p', password, 'rsync', ...]
                full_cmd = ['wsl', 'sshpass', '-p', self.storage_box_password] + cmd[1:]  # Skip 'wsl', add sshpass
                return subprocess.run(
                    full_cmd,
                    capture_output=capture_output,
                    text=False if capture_output else True,
                    input=input_data.encode() if input_data and isinstance(input_data, str) else input_data,
                    timeout=300
                )
            else:
                # sshpass not available in WSL - show helpful error
                if not self._ssh_key_message_shown:
                    error_msg = "sshpass not found in WSL. Please install it: wsl sudo apt-get update && wsl sudo apt-get install -y sshpass"
                    self.log_callback(error_msg)
                    self._ssh_key_message_shown = True
                # Return error result
                class FakeResult:
                    def __init__(self):
                        self.returncode = 1
                        self.stdout = b''
                        self.stderr = b'sshpass not found in WSL'
                return FakeResult()
        
        # For native Windows rsync (Git for Windows) or Linux/Mac
        # Try to find sshpass
        sshpass_cmd = None
        
        # Check common locations for sshpass
        if sys.platform == 'win32':
            # Check Git for Windows locations
            git_sshpass_paths = [
                r'C:\Program Files\Git\usr\bin\sshpass.exe',
                r'C:\Program Files (x86)\Git\usr\bin\sshpass.exe',
            ]
            for sshpass_path in git_sshpass_paths:
                if Path(sshpass_path).exists():
                    sshpass_cmd = sshpass_path
                    break
        
        # If not found in Git paths, try standard 'sshpass' command
        if not sshpass_cmd:
            try:
                subprocess.run(['sshpass', '-V'], capture_output=True, check=True, timeout=5)
                sshpass_cmd = 'sshpass'
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        if sshpass_cmd:
            # Use sshpass for password authentication
            full_cmd = [sshpass_cmd, '-p', self.storage_box_password] + cmd
            return subprocess.run(
                full_cmd,
                capture_output=capture_output,
                text=False if capture_output else True,
                input=input_data.encode() if input_data and isinstance(input_data, str) else input_data,
                timeout=300
            )
        else:
            # sshpass not available - show error
            if not self._ssh_key_message_shown:
                error_msg = "sshpass not found. Please install sshpass for password authentication, or set up SSH keys."
                if sys.platform == 'win32':
                    error_msg += " For WSL: run 'wsl sudo apt-get install sshpass'. For Git: sshpass may need to be installed separately."
                self.log_callback(error_msg)
                self._ssh_key_message_shown = True
            # Try to run without sshpass (will prompt for password)
            return subprocess.run(
                cmd,
                capture_output=capture_output,
                text=False if capture_output else True,
                input=input_data.encode() if input_data and isinstance(input_data, str) else input_data,
                timeout=300
            )

    def _sync_cache_directory_rsync(self, dry_run=False, upload_only=False):
        """
        Sync the entire map tile cache directory using rsync's native capabilities.
        rsync automatically determines what needs to be transferred.
        
        Args:
            dry_run: If True, only show what would be transferred
            upload_only: If True, only upload (no download)
            
        Returns:
            tuple: (uploaded_count, downloaded_count)
        """
        uploaded_count = 0
        downloaded_count = 0
        
        try:
            # Build paths - sync entire cache directory
            local_path = self.local_cache_dir
            remote_path = f"{self.storage_box_user}@{self.storage_box_address}:{self.remote_base_path}/"
            
            # Convert path for WSL if needed
            if self._rsync_cmd and self._rsync_cmd[0] == 'wsl':
                local_path_str = self._convert_path_for_wsl(local_path) + '/'
            else:
                local_path_str = str(local_path) + '/'
            
            # Base rsync options optimized for remote scanning of many immutable .npy files (~200KB each)
            # Key optimizations: minimize remote stat() calls, skip unnecessary metadata operations
            base_opts = [
                # Don't use --archive since we don't need to preserve timestamps for immutable files
                # Use lighter options: --recursive, --links, --perms (skip --times to reduce remote stat calls)
                '--recursive',
                '--links',  # Preserve symlinks
                '--perms',  # Preserve permissions (if needed)
                '--partial',  # Keep partial files for resuming
                '--stats',  # Show transfer statistics
                '--no-motd',
                '--human-readable',  # Human-readable sizes
                # Prune empty directories - reduces directory traversal overhead during scanning
                '--prune-empty-dirs',
                # Use size-only comparison (CRITICAL for remote scanning performance)
                # Since files are immutable, size comparison is sufficient and fastest
                # This skips checksum computation (no file reads) and timestamp comparison on remote
                # Each stat() call only needs to get file size, not read file contents
                '--size-only',
                # Use non-incremental recursion - can be faster for large directory trees
                # This processes directories in a more efficient order
                '--no-inc-recursive',
                # Optimize SSH connection with compression (for metadata transfer, not file data)
                # Compression helps with the metadata/listing phase, not the actual file transfer
                '-e', 'ssh -p 23 -o StrictHostKeyChecking=no -o Compression=yes -o CompressionLevel=6',
            ]
            
            if dry_run:
                base_opts.append('--dry-run')
                # For dry-run, we need itemize-changes to count files
                base_opts.append('--itemize-changes')
            else:
                # For actual sync, remove verbose output to speed things up
                # We'll use --stats to get summary at the end
                pass
            
            # Upload: sync local -> remote
            upload_cmd = self._rsync_cmd + base_opts + [
                local_path_str,
                remote_path
            ]
            
            result = self._run_rsync_with_password(upload_cmd, capture_output=True)
            
            if result.returncode == 0:
                output = result.stdout.decode('utf-8', errors='ignore')
                uploaded_count = self._count_rsync_transfers(output, direction='upload')
                if not dry_run:
                    self.debug_callback(f"Uploaded {uploaded_count} files to storage box")
            else:
                error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
                self.log_callback(f"rsync upload failed: {error_msg}")
                return 0, 0
            
            # Download: sync remote -> local (unless upload_only)
            if not upload_only:
                download_cmd = self._rsync_cmd + base_opts + [
                    remote_path,
                    local_path_str
                ]
                
                result = self._run_rsync_with_password(download_cmd, capture_output=True)
                
                if result.returncode == 0:
                    output = result.stdout.decode('utf-8', errors='ignore')
                    downloaded_count = self._count_rsync_transfers(output, direction='download')
                    if not dry_run:
                        self.debug_callback(f"Downloaded {downloaded_count} files from storage box")
                else:
                    error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
                    self.log_callback(f"rsync download failed: {error_msg}")
                    
        except Exception as e:
            self.log_callback(f"Error syncing map tile cache: {str(e)}")
        
        return uploaded_count, downloaded_count

    def _count_rsync_transfers(self, output, direction='upload'):
        """
        Count files transferred from rsync output.
        Uses rsync's --stats output which is more reliable.
        
        Args:
            output: rsync stdout/stderr output
            direction: 'upload' or 'download' (not used, rsync stats show both)
            
        Returns:
            Number of files transferred
        """
        # Look for rsync stats: "Number of regular files transferred: X" or "Number of files transferred: X".
        # rsync 3.1.0+ with --human-readable uses comma separators (e.g. "1,234"); capture [\d,]+ and strip commas.
        # Do NOT match "Number of files: X (reg: Y, dir: Z)" - that is total file count, not transferred count.
        patterns = [
            r'Number of regular files transferred:\s*([\d,]+)',
            r'Number of files transferred:\s*([\d,]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                num_str = match.group(1).replace(',', '')
                return int(num_str) if num_str else 0
        
        # Fallback: count itemize-changes lines if present (for dry-run)
        count = 0
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Count file transfer lines: >f (sent) or <f (received)
            # Format: ">f+++++++++ filename" or ">f.st...... filename"
            if line.startswith('>f') or line.startswith('<f'):
                # Skip directory lines (start with 'd')
                if not line.startswith('>d') and not line.startswith('<d'):
                    count += 1
        
        return count


# Convenience function for easy usage
def sync_map_tiles(storage_box_address, storage_box_user, storage_box_password, 
                   local_cache_dir="map tile cache", log_callback=None, progress_callback=None, 
                   sync_state_callback=None, max_workers=10, dry_run=False, upload_only=False, debug_callback=None):
    """
    Convenience function to sync map tiles using rsync over SSH.
    
    Args:
        storage_box_address: Storage box SSH server address (e.g., u465900.your-storagebox.de)
        storage_box_user: SSH username (same as FTP username)
        storage_box_password: SSH password (same as FTP password)
        local_cache_dir: Local directory for map tile cache
        log_callback: Function to call for main logging messages (upload/download counts)
        progress_callback: Function to call for progress updates
        sync_state_callback: Function to call when syncing starts/completes
        max_workers: Maximum number of parallel threads (used for directory grouping)
        dry_run: If True, only check file counts without performing sync operations
        upload_only: If True, only upload files to remote server (no downloads)
        debug_callback: Function to call for debug logging messages (defaults to log_callback if not provided)
        
    Returns:
        tuple: (success, uploaded_count, downloaded_count)
        
    Note:
        Requires rsync to be installed and available in PATH.
        On Linux/Mac, sshpass is recommended for password authentication.
        On Windows, SSH keys are recommended for passwordless authentication.
    """
    syncer = MapTileSyncer(
        storage_box_address=storage_box_address,
        storage_box_user=storage_box_user,
        storage_box_password=storage_box_password,
        local_cache_dir=local_cache_dir,
        log_callback=log_callback,
        progress_callback=progress_callback,
        sync_state_callback=sync_state_callback,
        debug_callback=debug_callback
    )
    return syncer.sync(max_workers=max_workers, dry_run=dry_run, upload_only=upload_only) 