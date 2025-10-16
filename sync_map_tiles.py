"""
Map Tile Cache Syncing Module
This module handles syncing of map tile cache files between local machine and storage box.
Can be used independently or called from other parts of the application.
"""

import ftplib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


class MapTileSyncer:
    """Handles syncing of map tile cache files between local and remote storage."""
    
    def __init__(self, storage_box_address, storage_box_user, storage_box_password, 
                 local_cache_dir="map tile cache", log_callback=None, progress_callback=None, sync_state_callback=None):
        """
        Initialize the MapTileSyncer.
        
        Args:
            storage_box_address: FTP server address
            storage_box_user: FTP username
            storage_box_password: FTP password
            local_cache_dir: Local directory for map tile cache (default: "map tile cache")
            log_callback: Function to call for logging messages
            progress_callback: Function to call for progress updates
            sync_state_callback: Function to call when syncing starts/completes (called with 'start' or 'complete')
        """
        self.storage_box_address = storage_box_address
        self.storage_box_user = storage_box_user
        self.storage_box_password = storage_box_password
        self.local_cache_dir = Path(local_cache_dir)
        self.log_callback = log_callback or (lambda msg: None)
        self.progress_callback = progress_callback or (lambda msg: None)
        self.sync_state_callback = sync_state_callback or (lambda state: None)
        
        # Known directory structure for map tile cache
        self.known_directories = [
            "OpenTopoMap/OpenTopoMapTiles",
            "OSM/OSM", 
            "StadiaDark/CustomStadiaTiles",
            "StadiaLight/CustomStadiaTiles",
            "StadiaOutdoors/CustomStadiaTiles",
            "StadiaToner/CustomStadiaTiles",
            "StadiaWatercolor/CustomStadiaTiles"
        ]

    def sync(self, max_workers=10, dry_run=False, upload_only=False):
        """
        Perform the complete sync operation.
        
        Args:
            max_workers: Maximum number of parallel threads for file operations
            dry_run: If True, only check file counts without performing sync operations
            upload_only: If True, only upload files to remote server (no downloads)
            
        Returns:
            tuple: (success, uploaded_count, downloaded_count)
        """
        try:
            # Notify that syncing is starting
            self.sync_state_callback('start')
            
            if dry_run:
                self.log_callback("Starting map tile cache file count check...")
            elif upload_only:
                self.log_callback("Starting map tile cache upload-only sync...")
            else:
                self.log_callback("Starting map tile cache sync...")
            
            # Ensure local cache directory exists
            if not self.local_cache_dir.exists():
                self.log_callback("Creating local map tile cache directory.")
                self.local_cache_dir.mkdir(parents=True, exist_ok=True)

            # Get list of remote files (parallel)
            remote_files = self._get_remote_files_parallel()
            
            # Get list of local files (recursive)
            local_files = self._get_local_files()
            
            self.log_callback(f"Found {len(local_files)} local files and {len(remote_files)} remote files")

            # Prepare files for upload and download
            files_to_upload = [f for f in local_files if f not in remote_files]
            
            if upload_only:
                files_to_download = []
                self.log_callback(f"Upload-only mode: {len(files_to_upload)} files to upload, no downloads")
            else:
                files_to_download = [f for f in remote_files if f not in local_files]
                self.log_callback(f"Files to upload: {len(files_to_upload)}, Files to download: {len(files_to_download)}")

            if dry_run:
                # For dry run, just return the counts without actually syncing
                if upload_only:
                    self.log_callback("File count check completed (dry run, upload-only mode - no files were transferred)")
                    self.progress_callback(f"Map tile cache check: {len(local_files)} local, {len(remote_files)} remote, {len(files_to_upload)} to upload, no downloads")
                else:
                    self.log_callback("File count check completed (dry run - no files were transferred)")
                    self.progress_callback(f"Map tile cache check: {len(local_files)} local, {len(remote_files)} remote, {len(files_to_upload)} to upload, {len(files_to_download)} to download")
                
                # Notify that syncing is complete
                self.sync_state_callback('complete')
                
                return True, len(files_to_upload), len(files_to_download)

            # Use ThreadPoolExecutor for parallel operations
            max_workers = min(max_workers, max(len(files_to_upload), len(files_to_download), 1))
            self.log_callback(f"Using {max_workers} parallel threads for syncing")

            uploaded_count = 0
            downloaded_count = 0

            # Upload files in parallel
            if files_to_upload:
                uploaded_count = self._upload_files_parallel(files_to_upload, max_workers)

            # Download files in parallel (only if not upload_only mode)
            if files_to_download and not upload_only:
                downloaded_count = self._download_files_parallel(files_to_download, max_workers)

            if upload_only:
                self.log_callback(f"Map tile cache upload-only sync completed. Uploaded: {uploaded_count}")
                self.progress_callback(f"Syncing map tile cache: completed ({uploaded_count} uploaded, upload-only mode)")
            else:
                self.log_callback(f"Map tile cache sync completed. Uploaded: {uploaded_count}, downloaded: {downloaded_count}")
                self.progress_callback(f"Syncing map tile cache: completed ({uploaded_count} uploaded, {downloaded_count} downloaded)")
            
            # Notify that syncing is complete
            self.sync_state_callback('complete')
            
            return True, uploaded_count, downloaded_count

        except Exception as e:
            self.log_callback(f"Map tile cache sync failed: {str(e)}")
            # Notify that syncing is complete (even on error)
            self.sync_state_callback('complete')
            return False, 0, 0

    def _get_local_files(self):
        """Get list of local files recursively."""
        local_files = []
        for file_path in self.local_cache_dir.rglob("*"):
            if file_path.is_file():
                # Get relative path from cache directory using forward slashes for consistency
                rel_path = str(file_path.relative_to(self.local_cache_dir)).replace('\\', '/')
                local_files.append(rel_path)
        return local_files

    def _get_remote_files_parallel(self):
        """Get all remote files by reading the known directories in parallel."""
        all_files = []
        
        # Use ThreadPoolExecutor to read directories in parallel
        with ThreadPoolExecutor(max_workers=7) as executor:
            # Submit tasks to read each directory with separate FTP connections
            future_to_dir = {
                executor.submit(self._read_remote_directory_with_new_connection, directory): directory 
                for directory in self.known_directories
            }
            
            # Collect results
            for future in as_completed(future_to_dir):
                directory = future_to_dir[future]
                try:
                    files = future.result()
                    all_files.extend(files)
                    self.log_callback(f"Found {len(files)} files in {directory}")
                except Exception as e:
                    self.log_callback(f"Error reading directory {directory}: {str(e)}")
        
        return all_files

    def _read_remote_directory_with_new_connection(self, directory):
        """Read files from a specific remote directory using a new FTP connection."""
        files = []
        try:
            # Create a new FTP connection for this thread
            with ftplib.FTP(self.storage_box_address) as ftp:
                ftp.login(self.storage_box_user, self.storage_box_password)
                ftp.timeout = 30
                
                # Navigate to the map tile cache directory
                ftp.cwd("map tile cache")
                
                # Navigate to the specific directory
                ftp.cwd(directory)
                
                # List all files in the directory
                items = ftp.nlst()
                
                # Filter out . and .. and add directory prefix to filenames
                for item in items:
                    if item not in ['.', '..']:
                        files.append(f"{directory}/{item}")
            
        except ftplib.error_perm as e:
            self.log_callback(f"Permission error reading directory {directory}: {str(e)}")
        except Exception as e:
            self.log_callback(f"Error reading directory {directory}: {str(e)}")
        
        return files

    def _upload_files_parallel(self, files_to_upload, max_workers):
        """Upload files in parallel."""
        self.log_callback("Starting parallel upload...")
        self.progress_callback("Syncing map tile cache: Starting upload...")
        
        uploaded_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit upload tasks
            upload_futures = {
                executor.submit(self._upload_file, file_path): file_path 
                for file_path in files_to_upload
            }
            
            # Process completed uploads
            for future in as_completed(upload_futures):
                file_path = upload_futures[future]
                try:
                    success = future.result()
                    if success:
                        uploaded_count += 1
                        if uploaded_count % 100 == 0:  # Log progress every 100 files
                            self.log_callback(f"Uploaded {uploaded_count}/{len(files_to_upload)} files")
                            self.progress_callback(f"Syncing map tile cache: Uploaded {uploaded_count}/{len(files_to_upload)} files")
                    else:
                        self.log_callback(f"Failed to upload {file_path}")
                except Exception as e:
                    self.log_callback(f"Error uploading {file_path}: {str(e)}")
        
        return uploaded_count

    def _download_files_parallel(self, files_to_download, max_workers):
        """Download files in parallel."""
        self.progress_callback("Syncing map tile cache: Starting download.")
        
        downloaded_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks
            download_futures = {
                executor.submit(self._download_file, file_path): file_path 
                for file_path in files_to_download
            }
            
            # Process completed downloads
            for future in as_completed(download_futures):
                file_path = download_futures[future]
                try:
                    success = future.result()
                    if success:
                        downloaded_count += 1
                        if downloaded_count % 100 == 0:  # Log progress every 100 files
                            self.log_callback(f"Downloaded {downloaded_count}/{len(files_to_download)} files")
                            self.progress_callback(f"Syncing map tile cache: downloaded {downloaded_count}/{len(files_to_download)} files")
                    else:
                        self.log_callback(f"Failed to download {file_path}")
                except Exception as e:
                    self.log_callback(f"Error downloading {file_path}: {str(e)}")
        
        return downloaded_count

    def _upload_file(self, file_path):
        """Upload a single file to the storage box."""
        try:
            with ftplib.FTP(self.storage_box_address) as ftp:
                ftp.login(self.storage_box_user, self.storage_box_password)
                ftp.timeout = 30  # Set 30 second timeout
                ftp.cwd("map tile cache")
                
                # Use Path objects for cross-platform compatibility
                local_file_path = self.local_cache_dir / file_path
                
                # Ensure remote directory structure exists
                remote_dir_path = str(Path(file_path).parent)
                if remote_dir_path != ".":
                    self._ensure_remote_directory(ftp, remote_dir_path)
                
                # Get just the filename for the STOR command since we're already in the correct directory
                filename = Path(file_path).name
                
                if not local_file_path.exists():
                    self.log_callback(f"Upload failed: Local file not found - {file_path}")
                    return False
                
                with open(local_file_path, 'rb') as f:
                    ftp.storbinary(f'STOR {filename}', f)
                return True
        except Exception as e:
            self.log_callback(f"Upload failed for {file_path}: {str(e)}")
            return False

    def _download_file(self, file_path):
        """Download a single file from the storage box."""
        try:
            with ftplib.FTP(self.storage_box_address) as ftp:
                ftp.login(self.storage_box_user, self.storage_box_password)
                ftp.timeout = 30  # Set 30 second timeout
                ftp.cwd("map tile cache")
                
                # Use Path objects for cross-platform compatibility
                local_file_path = self.local_cache_dir / file_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(local_file_path, 'wb') as f:
                    ftp.retrbinary(f'RETR {file_path}', f.write)
                return True
        except Exception as e:
            self.log_callback(f"Download failed for {file_path}: {str(e)}")
            return False

    def _ensure_remote_directory(self, ftp, dir_path):
        """Ensure remote directory structure exists."""
        if not dir_path or dir_path == ".":
            return
            
        # Convert Windows path to Unix path and split into components
        dir_path = dir_path.replace('\\', '/')
        path_parts = dir_path.split('/')
        
        for part in path_parts:
            if not part:
                continue
                
            try:
                ftp.cwd(part)
            except ftplib.error_perm:
                # Directory doesn't exist, create it
                ftp.mkd(part)
                ftp.cwd(part)


# Convenience function for easy usage
def sync_map_tiles(storage_box_address, storage_box_user, storage_box_password, 
                   local_cache_dir="map tile cache", log_callback=None, progress_callback=None, 
                   sync_state_callback=None, max_workers=10, dry_run=False, upload_only=False):
    """
    Convenience function to sync map tiles.
    
    Args:
        storage_box_address: FTP server address
        storage_box_user: FTP username
        storage_box_password: FTP password
        local_cache_dir: Local directory for map tile cache
        log_callback: Function to call for logging messages
        progress_callback: Function to call for progress updates
        sync_state_callback: Function to call when syncing starts/completes
        max_workers: Maximum number of parallel threads
        dry_run: If True, only check file counts without performing sync operations
        upload_only: If True, only upload files to remote server (no downloads)
        
    Returns:
        tuple: (success, uploaded_count, downloaded_count)
    """
    syncer = MapTileSyncer(
        storage_box_address=storage_box_address,
        storage_box_user=storage_box_user,
        storage_box_password=storage_box_password,
        local_cache_dir=local_cache_dir,
        log_callback=log_callback,
        progress_callback=progress_callback,
        sync_state_callback=sync_state_callback
    )
    return syncer.sync(max_workers=max_workers, dry_run=dry_run, upload_only=upload_only) 