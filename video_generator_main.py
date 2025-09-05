"""
Video generation worker classes for the Route Squiggler render client.
This module handles the main video generation workflow in separate threads.
"""

import time
import os
import ftplib
import re
from io import BytesIO
from PySide6.QtCore import QObject, Signal, QThread

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for video generation

from job_request import update_job_status
from video_generator_sort_files_chronologically import get_sorted_gpx_list
from video_generator_create_combined_route import create_combined_route
from video_generator_cache_map_tiles import cache_map_tiles
from video_generator_cache_map_images import cache_map_images
from video_generator_cache_video_frames import cache_video_frames


def upload_video_to_storage_box(
    video_path: str,
    thumbnail_path: str,
    storage_box_address: str,
    storage_box_user: str,
    storage_box_password: str,
    job_id: str,
    folder: str,
    filename: str,
    log_callback=None,
    progress_callback=None
) -> bool:
    """
    Upload a video and its thumbnail to the storage box and verify they exist.
    
    Args:
        video_path: Path to the video file to upload
        thumbnail_path: Path to the thumbnail file to upload
        storage_box_address: FTP server address
        storage_box_user: FTP username
        storage_box_password: FTP password
        job_id: Job ID for the folder structure
        folder: Subfolder name
        filename: Name of the video file to save
        log_callback: Optional callback for logging messages
        progress_callback: Optional callback for progress updates (progress_bar_name, percentage, progress_text)
        
    Returns:
        bool: True if upload successful and verified, False otherwise
    """
    
    class ProgressFTP(ftplib.FTP):
        """FTP class with progress tracking."""
        def __init__(self, progress_callback=None, total_size=0):
            super().__init__()
            self.progress_callback = progress_callback
            self.total_size = total_size
            self.uploaded_size = 0
            
        def storbinary(self, cmd, fp, blocksize=8192, callback=None, rest=None):
            """Override storbinary to track progress."""
            if self.progress_callback:
                def progress_callback_wrapper(data):
                    self.uploaded_size += len(data)
                    if self.total_size > 0:
                        percentage = int((self.uploaded_size / self.total_size) * 100)
                        self.progress_callback("progress_bar_upload", percentage, f"Uploading: {self.uploaded_size}/{self.total_size} bytes")
                    if callback:
                        callback(data)
                return super().storbinary(cmd, fp, blocksize, progress_callback_wrapper, rest)
            else:
                return super().storbinary(cmd, fp, blocksize, callback, rest)
    
    ftp = None
    try:
        if log_callback:
            log_callback("Starting video upload to storage box...")
            
        # Validate video file exists and has sufficient size (> 10KB)
        if not os.path.exists(video_path):
            if log_callback:
                log_callback(f"Error: Video file not found at {video_path}")
            return False
            
        video_size = os.path.getsize(video_path)
        if video_size < 10240:  # 10KB in bytes
            if log_callback:
                log_callback(f"Error: Video file too small ({video_size} bytes), minimum 10KB required")
            return False
            
        if log_callback:
            log_callback(f"Video file validated: {video_size} bytes")
        
        # Calculate total upload size
        total_upload_size = video_size
        thumbnail_size = 0
        if os.path.exists(thumbnail_path):
            thumbnail_size = os.path.getsize(thumbnail_path)
            total_upload_size += thumbnail_size
            
        if progress_callback:
            progress_callback("progress_bar_upload", 0, "Connecting to storage box...")
            
        # Connect to FTP with progress tracking
        ftp = ProgressFTP(progress_callback, total_upload_size)
        ftp.connect(storage_box_address)
        ftp.login(storage_box_user, storage_box_password)
        
        # Create or enter jobs directory
        try:
            ftp.mkd('jobs')
        except ftplib.error_perm:
            pass  # Directory might already exist
        ftp.cwd('jobs')
        
        # Create job directory if it doesn't exist
        try:
            ftp.mkd(job_id)
        except ftplib.error_perm:
            pass  # Directory might already exist
            
        # Change to job directory
        ftp.cwd(job_id)
        
        # Create folder if it doesn't exist
        try:
            ftp.mkd(folder)
        except ftplib.error_perm:
            pass  # Directory might already exist
            
        # Change to folder
        ftp.cwd(folder)
        
        # Upload the video file
        if log_callback:
            log_callback(f"Uploading video file: {filename}")
            
        if progress_callback:
            progress_callback("progress_bar_upload", 0, f"Uploading video: {filename}")
            
        with open(video_path, 'rb') as video_file:
            ftp.storbinary(f'STOR {filename}', video_file)
        
        # Verify video file exists and has size
        file_size = 0
        try:
            file_size = ftp.size(filename)
        except:
            if log_callback:
                log_callback(f"Failed to get size for {filename}")
            return False
            
        if file_size <= 0:
            if log_callback:
                log_callback(f"File {filename} has zero or negative size")
            return False
            
        if log_callback:
            log_callback(f"Video file uploaded successfully: {file_size} bytes")

        # Upload thumbnail if it exists
        if os.path.exists(thumbnail_path):
            if log_callback:
                log_callback("Uploading thumbnail file: thumbnail.png")
                
            if progress_callback:
                progress_callback("progress_bar_upload", int((video_size / total_upload_size) * 100), "Uploading thumbnail...")
                
            with open(thumbnail_path, 'rb') as thumb_file:
                ftp.storbinary('STOR thumbnail.png', thumb_file)
            
            # Verify thumbnail exists and has size
            thumb_size = 0
            try:
                thumb_size = ftp.size('thumbnail.png')
            except:
                if log_callback:
                    log_callback("Failed to get size for thumbnail.png")
                return False
                
            if thumb_size <= 0:
                if log_callback:
                    log_callback("File thumbnail.png has zero or negative size")
                return False
                
            if log_callback:
                log_callback(f"Thumbnail uploaded successfully: {thumb_size} bytes")
        else:
            if log_callback:
                log_callback("Warning: Thumbnail file not found, skipping thumbnail upload")
        
        if progress_callback:
            progress_callback("progress_bar_upload", 100, "Upload completed")
            
        if log_callback:
            log_callback("Video upload completed successfully")
        
        return True
        
    except Exception as e:
        if log_callback:
            log_callback(f"Upload failed: {str(e)}")
        if progress_callback:
            progress_callback("progress_bar_upload", 0, "Upload failed")
        return False
    finally:
        if ftp:
            try:
                ftp.quit()
            except:
                pass

class VideoGeneratorWorker(QObject):
    """Worker class to handle video generation in a separate thread."""
    finished = Signal()
    error = Signal(str)
    log_message = Signal(str)
    job_completed = Signal(str)  # Emits job ID when all uploads are successful
    progress_update = Signal(str, int, str)  # Emits (progress_bar_name, percentage, progress_text) for UI updates

    def __init__(self, json_data, gpx_files_info, storage_box_credentials, api_url, user, hardware_id, app_version, max_workers=None, is_test=False, gpu_rendering=True):
        super().__init__()
        self.json_data = json_data
        self.gpx_files_info = gpx_files_info
        self.storage_box_address = storage_box_credentials.get('address')
        self.storage_box_user = storage_box_credentials.get('user')
        self.storage_box_password = storage_box_credentials.get('password')
        self.api_url = api_url
        self.user = user
        self.hardware_id = hardware_id
        self.app_version = app_version
        self.max_workers = max_workers
        self.is_test = is_test  # Flag to indicate if this is a test job
        self.gpu_rendering = gpu_rendering  # GPU rendering setting from config
        self.results = None
        self.sorted_gpx_files = None  # Will store chronologically sorted GPX files
        self.combined_route_data = None  # Will store combined route data

    def format_duration(self, seconds):
        """
        Format duration in seconds to human-readable format.
        
        Args:
            seconds (float): Duration in seconds
            
        Returns:
            str: Human-readable duration (e.g., "5 minutes 20.52 seconds")
        """
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} {remaining_seconds:.2f} seconds"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours} hour{'s' if hours != 1 else ''} {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''} {remaining_seconds:.2f} seconds"

    def video_generator_process(self):
        """Main processing method that runs in the worker thread."""
        try:
            self.log_message.emit("Video generation triggered.")
            
            # Add test job flag to json_data for use throughout the pipeline
            self.json_data['test_job'] = self.is_test
            
            # Step 1: Sort GPX files chronologically
            self.log_message.emit("Step 1: Sorting GPX files chronologically")           
            self.sorted_gpx_files = get_sorted_gpx_list(
                self.gpx_files_info, 
                log_callback=self.log_message.emit
            )
            
            if not self.sorted_gpx_files:
                raise ValueError("Failed to sort GPX files chronologically")
            
            self.log_message.emit(f"Chronological sorting completed with {len(self.sorted_gpx_files)} files")
            
            # Step 2: Create combined route
            self.log_message.emit("Step 2: Creating combined route")           
            
            self.combined_route_data = create_combined_route(
                self.sorted_gpx_files,
                self.json_data,
                progress_callback=self.progress_update.emit,
                log_callback=self.log_message.emit
            )
            
            # Update progress
            if self.progress_update:
                self.progress_update.emit("progress_bar_combined_route", 100, "Route creation completed")
            
            if not self.combined_route_data:
                raise ValueError("Failed to create combined route")
            
            # Log summary
            total_points = self.combined_route_data.get('total_points', 0)
            total_distance_km = self.combined_route_data.get('total_distance', 0) / 1000
            self.log_message.emit(f"Combined route created successfully:")
            self.log_message.emit(f"  - Total points: {total_points:,}")
            self.log_message.emit(f"  - Total distance: {total_distance_km:.2f} km")
            self.log_message.emit(f"  - Files processed: {len(self.sorted_gpx_files)}")
            
            # Step 3: Calculate unique bounding boxes and cache map tiles
            self.log_message.emit("Step 3: Calculating unique bounding boxes and caching map tiles")           
            cache_result = cache_map_tiles(
                self.json_data,
                combined_route_data=self.combined_route_data,
                progress_callback=self.progress_update.emit,
                log_callback=self.log_message.emit,
                max_workers=self.max_workers
            )
            
            if not cache_result:
                raise ValueError("Map tile caching failed")
            
            # Summary already logged in cache_map_tiles function - no need to repeat
            
            # START TIMER: Begin timing the rendering process
            render_start_time = time.time()
            self.log_message.emit("Starting render timer - beginning map image caching and video generation")
            
            # Create shared map cache for faster image loading between steps
            from multiprocessing import Manager
            with Manager() as manager:
                shared_map_cache = manager.dict()
                shared_route_cache = manager.dict()  # Add shared route cache for performance
                
                # Step 4: Cache map images for unique bounding boxes
                self.log_message.emit("Step 4: Caching map images for unique bounding boxes")           
                
                map_images_result = cache_map_images(
                    self.json_data,
                    combined_route_data=self.combined_route_data,
                    progress_callback=self.progress_update.emit,
                    log_callback=self.log_message.emit,
                    max_workers=self.max_workers,
                    shared_map_cache=shared_map_cache
                )
                
                if not map_images_result:
                    raise ValueError("Map image caching failed")
                
                # Log summary of map image caching
                cache_info = map_images_result.get('cache_result', {})
                total_images = cache_info.get('total_images_created', 0)
                failed_images = cache_info.get('total_images_failed', 0)
                total_bboxes = cache_info.get('total_bboxes', 0)
                self.log_message.emit(f"Map image caching completed:")
                self.log_message.emit(f"  - Images created: {total_images}")
                self.log_message.emit(f"  - Images failed: {failed_images}")
                self.log_message.emit(f"  - Total bounding boxes: {total_bboxes}")
                
                # Step 5: Generate video frames using shared map cache
                self.log_message.emit("Step 5: Generating video frames")
                
                video_frames_result = cache_video_frames(
                    self.json_data,
                    combined_route_data=self.combined_route_data,
                    progress_callback=self.progress_update.emit,
                    log_callback=self.log_message.emit,
                    max_workers=self.max_workers,
                    shared_map_cache=shared_map_cache,
                    shared_route_cache=shared_route_cache,
                    gpx_time_per_video_time=self.combined_route_data.get('gpx_time_per_video_time', None),
                    gpu_rendering=self.gpu_rendering
                )
                
                if not video_frames_result:
                    raise ValueError("Video frame generation failed")
                
                # Log summary of video frame generation
                frame_cache_info = video_frames_result.get('cache_result', {})
                total_frames_created = frame_cache_info.get('total_frames_created', 0)
                total_frames_failed = frame_cache_info.get('total_frames_failed', 0)
                total_frames = frame_cache_info.get('total_frames', 0)
                video_path = frame_cache_info.get('video_path', 'Unknown')
                self.log_message.emit(f"Video frame generation completed:")
                self.log_message.emit(f"  - Frames created: {total_frames_created}")
                self.log_message.emit(f"  - Frames failed: {total_frames_failed}")
                self.log_message.emit(f"  - Total frames: {total_frames}")
                self.log_message.emit(f"  - Video file: {video_path}")
                
                # STOP TIMER: End timing and log total rendering time
                render_end_time = time.time()
                total_render_time = render_end_time - render_start_time
                formatted_render_time = self.format_duration(total_render_time)
                self.log_message.emit(f"🎬 Total rendering time: {formatted_render_time}")
                
                # Step 6: Upload video and thumbnail to storage box (for non-test jobs)
                if not self.is_test:
                    self.log_message.emit("Step 6: Uploading video and thumbnail to storage box")
                    
                    # Get upload parameters
                    job_id = str(self.json_data.get('job_id', ''))
                    folder = self.json_data.get('folder', '')
                    route_name = self.json_data.get('route_name', 'route_video')
                    
                    # Determine the video filename based on route name
                    clean_route_name = re.sub(r'[<>:"/\\|?*]', '_', route_name)
                    clean_route_name = clean_route_name.strip()
                    
                    if not clean_route_name or clean_route_name == '_':
                        # Fallback to timestamp-based filename
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        video_filename = f"route_video_{timestamp}.mp4"
                    else:
                        video_filename = f"{clean_route_name}.mp4"
                    
                    # Determine thumbnail path
                    video_dir = os.path.dirname(video_path)
                    thumbnail_path = os.path.join(video_dir, 'thumbnail.png')
                    
                    # Upload to storage box
                    upload_success = upload_video_to_storage_box(
                        video_path=video_path,
                        thumbnail_path=thumbnail_path,
                        storage_box_address=self.storage_box_address,
                        storage_box_user=self.storage_box_user,
                        storage_box_password=self.storage_box_password,
                        job_id=job_id,
                        folder=folder,
                        filename=video_filename,
                        log_callback=self.log_message.emit,
                        progress_callback=self.progress_update.emit
                    )
                    
                    if upload_success:
                        self.log_message.emit("✅ Video upload completed successfully")
                        
                        # Report success to main server
                        self.job_completed.emit(job_id)
                        update_job_status(
                            self.api_url, 
                            self.user, 
                            self.hardware_id, 
                            self.app_version, 
                            job_id, 
                            'ok',
                            self.log_message.emit
                        )
                    else:
                        self.log_message.emit("❌ Video upload failed")
                        raise ValueError("Video upload to storage box failed")
                else:
                    # Test job - just log completion without upload
                    self.log_message.emit("Test job completed - skipping upload")
                    job_id = str(self.json_data.get('job_id', ''))
                    self.job_completed.emit(job_id)
                
                self.finished.emit()
                
        except Exception as e:
            self.log_message.emit(f"Error in video generation: {str(e)}")
            
            # Only update job status to error via API if this is not a test job
            if not self.is_test:
                update_job_status(
                    self.api_url, 
                    self.user, 
                    self.hardware_id, 
                    self.app_version, 
                    self.json_data.get('job_id', ''), 
                    'error',
                    self.log_message.emit
                )
            else:
                self.log_message.emit("Test job error - skipping API status update")
                
            self.error.emit(str(e))
        finally:
            # Always clean up resources to prevent Qt painter conflicts
            self.cleanup_resources()

    def cleanup_resources(self):
        """Clean up matplotlib and Qt resources to prevent painter conflicts."""
        try:
            import matplotlib.pyplot as plt
            
            # Close all matplotlib figures to release Qt painters
            plt.close('all')
            
            # Clear matplotlib cache
            matplotlib.pyplot.clf()
            matplotlib.pyplot.cla()
            
            # Force garbage collection to clean up any remaining Qt objects
            import gc
            gc.collect()
            
            self.log_message.emit("Matplotlib and Qt resources cleaned up")
            
        except Exception as e:
            self.log_message.emit(f"Warning: Error during resource cleanup: {str(e)}")
        
        # Clean up MoviePy and FFmpeg processes
        try:
            import psutil
            import subprocess
            
            # Find and terminate any remaining FFmpeg processes that might be hanging
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check if this is an FFmpeg process related to our video generation
                    if proc.info['name'] and 'ffmpeg' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any('moviepy' in str(arg).lower() or 'temp' in str(arg).lower() for arg in cmdline):
                            self.log_message.emit(f"Terminating hanging FFmpeg process: PID {proc.info['pid']}")
                            proc.terminate()
                            proc.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    # Process already terminated or we don't have permission
                    pass
                except Exception as e:
                    self.log_message.emit(f"Warning: Error checking FFmpeg process: {str(e)}")
            
            # Also try to clean up any temporary files that MoviePy might have left
            import os
            import tempfile
            import glob
            
            # Clean up potential MoviePy temporary files
            temp_patterns = [
                os.path.join(tempfile.gettempdir(), 'moviepy_*'),
                os.path.join(tempfile.gettempdir(), 'ffmpeg_*'),
                'temp_*',
                '*.tmp'
            ]
            
            for pattern in temp_patterns:
                try:
                    for temp_file in glob.glob(pattern):
                        if os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                                self.log_message.emit(f"Cleaned up temporary file: {temp_file}")
                            except Exception as e:
                                # File might be in use, ignore
                                pass
                except Exception as e:
                    # Pattern might be invalid, ignore
                    pass
            
            # Force kill any remaining FFmpeg processes as a last resort
            try:
                if os.name == 'nt':  # Windows
                    subprocess.run(['taskkill', '/f', '/im', 'ffmpeg.exe'], 
                                 capture_output=True, timeout=10)
                else:  # Linux/Mac
                    subprocess.run(['pkill', '-f', 'ffmpeg'], 
                                 capture_output=True, timeout=10)
                self.log_message.emit("Force-killed any remaining FFmpeg processes")
            except Exception as e:
                self.log_message.emit(f"Warning: Error force-killing FFmpeg: {str(e)}")
            
            self.log_message.emit("MoviePy and FFmpeg processes cleaned up")
            
        except Exception as e:
            self.log_message.emit(f"Warning: Error during MoviePy cleanup: {str(e)}")

class VideoWorkerThread(QThread):
    """Thread class to run the video generator worker."""
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.worker.moveToThread(self)
        
    def run(self):
        """Execute the worker's video_generator_process method."""
        self.worker.video_generator_process()
