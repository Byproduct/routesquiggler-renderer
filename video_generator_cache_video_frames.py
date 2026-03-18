"""
Functions related to generating video frames.
This file contains things related to preparation and multiprocessing.
The actual map plotting is in video_generator_create_single_frame.py.
"""

import json
from multiprocessing import Manager
import os
from pathlib import Path
import re
import time

# Third-party imports
from moviepy import VideoClip
from PIL import Image

# Increase PIL image size limit to 200 megapixels to handle large renders
Image.MAX_IMAGE_PIXELS = 200_000_000

# Local imports
from config import config
from job_request import set_attribution_from_theme
from video_generator_calculate_bounding_boxes import calculate_route_time_per_frame
from video_generator_cache_video_frames_util import (
    MoviePyDebugLogger,
    get_ffmpeg_executable,
    suppress_moviepy_output,
)
from video_generator_create_combined_route import RoutePoint
from video_generator_streaming_frame_generator import StreamingFrameGenerator
from video_generator_utils import (
    compute_sequential_ending_lengths,
    compute_sequential_frames_to_skip,
    compute_simultaneous_ending_lengths,
    is_simultaneous_mode,
)
from write_log import write_debug_log, write_log


def create_video_streaming(json_data, route_time_per_frame, combined_route_data, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True, user=None, simultaneous_mode=None):
    """
    Create video by streaming frames directly to ffmpeg without saving to disk.
    
    Args:
        json_data (dict): Job data containing video parameters
        route_time_per_frame (float): Time per frame in seconds
        combined_route_data (dict): Combined route data
        progress_callback (callable, optional): Function to call with progress updates
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for map images
        shared_route_cache (dict, optional): Shared memory cache for route images
        gpx_time_per_video_time (float, optional): GPX time per video time ratio
        gpu_rendering (bool, optional): Whether to use GPU rendering (default: True)
        user (str, optional): API key for status updates at 10% intervals (username misleading here, it's the "username" of the renderer used as an API key - not the user to which the job belongs)
    
    Returns:
        str: Path to the created video file, or None if failed
    """
    try:
        if debug_callback:
            debug_callback("Starting streaming video creation")
        
        # Calculate video parameters (simultaneous_mode passed from caller)
        video_length = float(json_data.get('video_length', 30))
        video_fps = float(json_data.get('video_fps', 30))
        if simultaneous_mode is None:
            simultaneous_mode = is_simultaneous_mode(combined_route_data)
        is_simultaneous_mode = simultaneous_mode
        
        # Calculate ending parameters based on mode
        if is_simultaneous_mode:
            # SIMULTANEOUS MODE: Each route handles its own completion and tail fade-out
            extended_video_length, final_video_length, cloned_ending_duration = compute_simultaneous_ending_lengths(video_length, tail_length=0)
            tail_length = 0
            
            if debug_callback:
                debug_callback(f"SIMULTANEOUS MODE: Creating video: {video_length}s route + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
                debug_callback(f"Each route handles its own completion and tail fade-out individually")
        else:
            # SEQUENTIAL MODE: Use the original tail-only phase logic
            tail_length = int(json_data.get('tail_length', 0))
            extended_video_length, final_video_length, cloned_ending_duration = compute_sequential_ending_lengths(video_length, tail_length)
            
            if debug_callback:
                if tail_length > 0 and cloned_ending_duration > 0:
                    debug_callback(f"SEQUENTIAL MODE: Creating video: {video_length}s route + {tail_length}s tail + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
                elif tail_length > 0:
                    debug_callback(f"SEQUENTIAL MODE: Creating video: {video_length}s route + {tail_length}s tail = {final_video_length}s total (no additional frames needed)")
                else:
                    debug_callback(f"SEQUENTIAL MODE: Creating video: {video_length}s route + {cloned_ending_duration:.1f}s cloned = {final_video_length:.1f}s total")
        
        # Calculate frame counts
        original_frames = int(video_length * video_fps)
        tail_frames = int(tail_length * video_fps) if tail_length > 0 else 0  
        cloned_frames = int(cloned_ending_duration * video_fps)
        total_frames = int(final_video_length * video_fps)

        if debug_callback:
            if tail_length > 0 and cloned_ending_duration > 0:
                debug_callback(f"Frame breakdown: {original_frames} route + {tail_frames} tail + {cloned_frames} cloned = {total_frames} total frames")
            elif tail_length > 0:
                debug_callback(f"Frame breakdown: {original_frames} route + {tail_frames} tail = {total_frames} total frames")
            else:
                debug_callback(f"Frame breakdown: {original_frames} route + {cloned_frames} cloned = {total_frames} total frames")
        
        # Generate output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Determine output directory and filename based on test job flag
        is_test_job = json_data.get('test_job', False)
        job_id = json_data.get('job_id', '')
        
        if is_test_job:
            # Test job: save in temporary_files/test jobs with timestamp filename
            output_dir = os.path.join('temporary files', 'test jobs')
            output_filename = f"route_video_{timestamp}.mp4"
        else:
            # Regular job: save in temporary_files/job_id with route_name filename
            output_dir = os.path.join('temporary files', str(job_id))
            route_name = json_data.get('route_name', 'route_video')
            
            # Clean the route name for use as filename (remove invalid characters)
            clean_route_name = re.sub(r'[<>:"/\\|?*]', '_', route_name)
            clean_route_name = clean_route_name.strip()
            
            # If route name is empty or only contains invalid characters, fall back to timestamp
            if not clean_route_name or clean_route_name == '_':
                output_filename = f"route_video_{timestamp}.mp4"
            else:
                output_filename = f"{clean_route_name}.mp4"
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        
        if debug_callback:
            debug_callback(f"Output video will be saved to: {output_path}")
            if is_test_job:
                debug_callback("Test job mode: Video will be saved in test jobs directory")
            else:
                debug_callback(f"Regular job mode: Video will be saved in job directory: {job_id}")
                debug_callback(f"Video filename: {output_filename}")
        
        # Create the frame generator
        frame_generator = StreamingFrameGenerator(json_data, route_time_per_frame, combined_route_data, max_workers, shared_map_cache, shared_route_cache, progress_callback, gpx_time_per_video_time, debug_callback, user, simultaneous_mode=is_simultaneous_mode)
        
        # Initial progress update
        if progress_callback:
            progress_callback("progress_bar_frames", 0, "Starting frame generation")
        
        if frame_generator.frames_to_skip > 0 and debug_callback:
            debug_callback(f"Skipping {frame_generator.frames_to_skip} leading frames with < 2 points (video shortened by {frame_generator.frames_to_skip / video_fps:.2f}s)")
        
        # Create the video clip
        clip = VideoClip(frame_generator.make_frame, duration=frame_generator.final_video_length)
        
        # Write the video file
        if debug_callback:
            debug_callback("Starting video encoding")
        
        # Use GPU rendering setting from config
        if debug_callback:
            if gpu_rendering:
                debug_callback("Using GPU rendering (NVIDIA NVENC)")
            else:
                debug_callback("Using CPU rendering (libx264)")
        
        try:
            # Use moviepy to write the video
            if gpu_rendering:
                # GPU rendering with NVIDIA NVENC
                # Quality/File Size Options:
                # - Lower bitrate = smaller file, lower quality
                # - Higher bitrate = larger file, higher quality
                # - Presets p1-p7: p1 = fastest, p7 = best quality (slowest)
                # - p4 is a good balance between speed and quality
                try:
                    # Suppress MoviePy output when debug is off
                    with suppress_moviepy_output():
                        # Create custom logger that respects debug settings
                        moviepy_logger = MoviePyDebugLogger(debug_callback) if debug_callback else None
                        clip.write_videofile(
                            output_path,
                            fps=video_fps,
                            codec='h264_nvenc',  # NVIDIA GPU encoder
                            bitrate='15000k',    # Can be reduced for smaller files (e.g., '8000k', '5000k')
                            audio=False,
                            threads=max_workers,
                            logger=moviepy_logger if config.debug_logging else None,  # Only use logger when debug is enabled
                            ffmpeg_params=[
                                '-pix_fmt', 'yuv420p',
                                '-preset', 'p5',          # Options: p1 (fastest) to p7 (best quality)
                                '-rc', 'vbr',             # Rate control: vbr (variable), cbr (constant), cqp (constant quality)
                                '-cq', '23',              # Constant quality (18-51, lower = better quality, larger file)
                                '-b:v', '15000k',         # Target bitrate
                                '-maxrate', '20000k',     # Maximum bitrate
                                '-bufsize', '30000k'      # Buffer size for rate control
                            ]
                        )
                except Exception as nvenc_error:
                    # Fallback to CPU encoding if NVENC fails
                    if debug_callback:
                        debug_callback(f"NVENC encoding failed ({nvenc_error}), falling back to CPU encoding")
                    gpu_rendering = False  # Will trigger CPU encoding below
            
            # CPU rendering with libx264 (either by choice or as fallback from failed NVENC)
            if not gpu_rendering:
                # Suppress MoviePy output when debug is off
                with suppress_moviepy_output():
                    # Create custom logger that respects debug settings
                    moviepy_logger = MoviePyDebugLogger(debug_callback) if debug_callback else None
                    clip.write_videofile(
                        output_path,
                        fps=video_fps,
                        codec='libx264',
                        bitrate='15000k',
                        audio=False,
                        threads=max_workers,
                        logger=moviepy_logger if config.debug_logging else None,  # Only use logger when debug is enabled
                        ffmpeg_params=[
                            '-pix_fmt', 'yuv420p',
                            '-preset', 'medium',        # Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
#                        '-tune', 'animation',       # Options: film, animation, grain, stillimage, fastdecode, zerolatency
                            '-crf', '25',               # Constant Rate Factor (0-51, lower = better quality, larger file)
                            '-profile:v', 'high',       # Profile: baseline, main, high, high10, high422, high444
                            '-level', '4.1'             # H.264 level for compatibility
                        ]
                    )
            
            # Final frame generation progress update
            if progress_callback:
                progress_callback("progress_bar_frames", 100, "Frame generation complete")
            
            if debug_callback:
                debug_callback(f"Streaming video creation complete: {output_path}")
            
            # Generate thumbnail for non-test jobs
            # Skip thumbnail generation if hide_complete_routes is enabled (will be generated after black frame removal)
            hide_complete_routes_raw = json_data.get('hide_complete_routes', False)
            if isinstance(hide_complete_routes_raw, str):
                hide_complete_routes = hide_complete_routes_raw.lower() in ('true', '1', 'yes')
            else:
                hide_complete_routes = bool(hide_complete_routes_raw)
            
            if not is_test_job and not hide_complete_routes:
                try:
                    if debug_callback:
                        debug_callback("Generating thumbnail from last frame")
                    
                    # Extract the last frame using ffmpeg
                    import subprocess
                    
                    # Create temporary frame file
                    temp_frame_path = os.path.join(output_dir, 'temp_last_frame.png')
                    
                    # Get the correct ffmpeg executable for the current platform
                    ffmpeg_executable = get_ffmpeg_executable()
                    
                    # Use ffmpeg to extract the last frame
                    cmd = [
                        ffmpeg_executable,
                        '-sseof', '-1',  # Seek to 1 second before end
                        '-i', output_path,
                        '-vframes', '1',
                        '-y',  # Overwrite output file
                        temp_frame_path
                    ]
                    
                    if debug_callback:
                        debug_callback(f"Running FFmpeg command: {' '.join(cmd)}")
                    
                    # Run ffmpeg command
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    if debug_callback:
                        debug_callback(f"FFmpeg command executed successfully")
                    
                    # Load the extracted frame and resize it
                    from PIL import Image
                    pil_image = Image.open(temp_frame_path)
                    
                    # Calculate new width to maintain aspect ratio with 320px height
                    original_width, original_height = pil_image.size
                    new_height = 320
                    new_width = int((original_width / original_height) * new_height)
                    
                    # Resize the image
                    resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save as thumbnail.png in the same directory
                    thumbnail_path = os.path.join(output_dir, 'thumbnail.png')
                    resized_image.save(thumbnail_path, 'PNG')
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_frame_path)
                    except:
                        pass  # Ignore cleanup errors
                    
                    if debug_callback:
                        debug_callback(f"Thumbnail generated: {thumbnail_path} ({new_width}x{new_height})")
                        
                except subprocess.CalledProcessError as e:
                    if log_callback:
                        log_callback(f"Warning: FFmpeg failed to extract last frame: {e}")
                        log_callback(f"FFmpeg stderr: {e.stderr}")
                except Exception as e:
                    if log_callback:
                        log_callback(f"Warning: Failed to generate thumbnail: {str(e)}")
            
            return output_path
            
        except Exception as e:
            if log_callback:
                log_callback(f"Error during video encoding: {str(e)}")
            raise e
        finally:
            # Always clean up resources
            clip.close()
            frame_generator.cleanup()
            
            # Additional cleanup for MoviePy and FFmpeg processes
            try:
                import psutil
                
                # Find and terminate any remaining FFmpeg processes that might be hanging
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        # Check if this is an FFmpeg process related to our video generation
                        if proc.info['name'] and 'ffmpeg' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline']
                            if cmdline and any('moviepy' in str(arg).lower() or 'temp' in str(arg).lower() for arg in cmdline):
                                if debug_callback:
                                    debug_callback(f"Terminating hanging FFmpeg process: PID {proc.info['pid']}")
                                proc.terminate()
                                proc.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                        # Process already terminated or we don't have permission
                        pass
                    except Exception as e:
                        if log_callback:
                            log_callback(f"Warning: Error checking FFmpeg process: {str(e)}")
                
                if debug_callback:
                    debug_callback("MoviePy and FFmpeg processes cleaned up")
                    
            except Exception as e:
                if log_callback:
                    log_callback(f"Warning: Error during MoviePy cleanup: {str(e)}")
            
    except Exception as e:
        if log_callback:
            log_callback(f"Error in streaming video creation: {str(e)}")
        return None


def cache_video_frames_for_video(json_data, route_time_per_frame, combined_route_data, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True, user=None, simultaneous_mode=None):
    """
    Cache video frames for video generation using streaming approach.

    Args:
        json_data (dict): Job data containing video parameters
        route_time_per_frame (float): Time per frame in seconds
        combined_route_data (dict): Combined route data
        progress_callback (callable, optional): Function to call with progress updates
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for map images
        shared_route_cache (dict, optional): Shared memory cache for route images
        gpx_time_per_video_time (float, optional): GPX time per video time ratio
        gpu_rendering (bool, optional): Whether to use GPU rendering (default: True)
        user (str, optional): API key for status updates at 10% intervals
        simultaneous_mode (bool, optional): True if multiple distinct routes; computed from combined_route_data if None

    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if simultaneous_mode is None:
            simultaneous_mode = is_simultaneous_mode(combined_route_data)
        is_simultaneous_mode = simultaneous_mode
        if debug_callback:
            debug_callback("Starting video frame generation with streaming approach")
        
        # Use provided shared map cache or create new one if not provided
        if shared_map_cache is None:
            with Manager() as manager:
                shared_map_cache = manager.dict()
        
        # Create video using streaming approach
        video_path = create_video_streaming(
            json_data, route_time_per_frame, combined_route_data,
            progress_callback, log_callback, debug_callback, max_workers, shared_map_cache, shared_route_cache, gpx_time_per_video_time, gpu_rendering, user,
            simultaneous_mode=is_simultaneous_mode
        )
        
        if video_path:
            # Calculate extended video length including tail and cloned ending
            video_length = float(json_data.get('video_length', 30))
            video_fps = float(json_data.get('video_fps', 30))
            
            # Calculate ending parameters based on mode (is_simultaneous_mode passed from caller)
            if is_simultaneous_mode:
                # SIMULTANEOUS MODE: Each route handles its own completion and tail fade-out
                extended_video_length, final_video_length, cloned_ending_duration = compute_simultaneous_ending_lengths(video_length, tail_length=0)
                tail_length = 0
            else:
                # SEQUENTIAL MODE: Use the original tail-only phase logic
                tail_length = int(json_data.get('tail_length', 0))
                extended_video_length, final_video_length, cloned_ending_duration = compute_sequential_ending_lengths(video_length, tail_length)
            
            # Account for skipped leading frames in sequential mode (mirrors
            # the geographic-diversity scan in StreamingFrameGenerator.__init__)
            frames_to_skip = 0
            if not is_simultaneous_mode:
                combined_route = combined_route_data.get('combined_route', [])
                route_time_per_frame_local = gpx_time_per_video_time / video_fps if gpx_time_per_video_time else 0
                frames_to_generate_total = int(extended_video_length * video_fps)
                frames_to_skip = compute_sequential_frames_to_skip(
                    combined_route, route_time_per_frame_local, frames_to_generate_total
                )
                if frames_to_skip > 0:
                    skipped_duration = frames_to_skip / video_fps
                    final_video_length -= skipped_duration
            
            total_frames = int(final_video_length * video_fps)
            frames_generated = int(extended_video_length * video_fps) - frames_to_skip
            cloned_frames = int(cloned_ending_duration * video_fps)
            
            if debug_callback:
                if is_simultaneous_mode:
                    debug_callback(f"SIMULTANEOUS MODE: Video creation completed: {frames_generated} frames generated + {cloned_frames} frames cloned = {total_frames} total frames")
                else:
                    skip_info = f" (skipped {frames_to_skip} leading frames)" if frames_to_skip > 0 else ""
                    debug_callback(f"SEQUENTIAL MODE: Video creation completed: {frames_generated} frames generated + {cloned_frames} frames cloned = {total_frames} total frames{skip_info}")
            
            return {
                'total_frames_created': frames_generated,  # Frames actually generated (not including cloned)
                'total_frames_failed': 0,
                'total_frames': total_frames,  # Total frames in final video (including cloned)
                'cloned_frames': cloned_frames,  # New: number of cloned frames
                'success': True,
                'video_path': video_path,
                'shared_map_cache': shared_map_cache,
                'results': []
            }
        else:
            return None
                
    except Exception as e:
        if log_callback:
            log_callback(f"Error in cache_video_frames_for_video: {str(e)}")
        return None


def cache_video_frames(json_data=None, combined_route_data=None, progress_callback=None, log_callback=None, debug_callback=None, max_workers=None, shared_map_cache=None, shared_route_cache=None, gpx_time_per_video_time=None, gpu_rendering=True, user=None):
    """
    Cache video frames for video generation.
    
    Args:
        json_data (dict, optional): Job data containing video parameters
        combined_route_data (dict, optional): Combined route data containing gpx_time_per_video_time
        progress_callback (callable, optional): Function to call with progress updates (progress_bar_name, percentage, progress_text)
        log_callback (callable, optional): Function to call for logging messages
        debug_callback (callable, optional): Function to call for debug logging messages
        max_workers (int, optional): Maximum number of worker processes to use
        shared_map_cache (dict, optional): Shared memory cache for map images
        shared_route_cache (dict, optional): Shared memory cache for route images
        gpx_time_per_video_time (float, optional): GPX time per video time ratio
        gpu_rendering (bool, optional): Whether to use GPU rendering (default: True)
        user (str, optional): API key for status updates at 10% intervals
    
    Returns:
        dict: Cache results with timing information, or None if error
    """
    try:
        if debug_callback:
            debug_callback("Starting video generation")
        
        if progress_callback:
            progress_callback("progress_bar_frames", 0, "Starting video generation")
        
        # Load job data if not provided
        if json_data is None:
            data_file = Path("data.json")
            if not data_file.exists():
                if log_callback:
                    log_callback("Error: data.json not found")
                return None
            
            with open(data_file, 'r') as f:
                json_data = json.load(f)
            set_attribution_from_theme(json_data)
        
        # Use gpx_time_per_video_time from combined_route_data if available, otherwise calculate it
        if combined_route_data and 'gpx_time_per_video_time' in combined_route_data:
            gpx_time_per_video_time = combined_route_data['gpx_time_per_video_time']
            video_fps = float(json_data.get('video_fps', 30))
            route_time_per_frame = gpx_time_per_video_time / video_fps
            
            if debug_callback:
                # debug_callback(f"Using gpx_time_per_video_time from combined_route_data: {gpx_time_per_video_time}")
                # debug_callback(f"Calculated route_time_per_frame: {route_time_per_frame:.6f} seconds")
                pass
        else:
            # Fallback to calculating route time per frame
            route_time_per_frame = calculate_route_time_per_frame(json_data, combined_route_data, log_callback, debug_callback)
        
        if route_time_per_frame is None:
            if log_callback:
                log_callback("Error: Could not calculate route time per frame")
            return None
        
        if debug_callback:
            debug_callback(f"Route time per frame: {route_time_per_frame:.4f} seconds")
        
        if combined_route_data is None:
            if log_callback:
                log_callback("Error: combined_route_data parameter is required")
            return None
        
        # Detect simultaneous mode once and pass through the chain
        simultaneous_mode = is_simultaneous_mode(combined_route_data)
        
        # Create video using streaming approach
        cache_result = cache_video_frames_for_video(
            json_data, route_time_per_frame, combined_route_data,
            progress_callback, log_callback, debug_callback, max_workers, shared_map_cache, shared_route_cache, gpx_time_per_video_time, gpu_rendering, user,
            simultaneous_mode=simultaneous_mode
        )
        
        if cache_result is None:
            if log_callback:
                log_callback("Error: Could not create video")
            return None
        
        # Return the result directly since video creation is already handled
        return {
            'route_time_per_frame': route_time_per_frame,
            'combined_route_data': combined_route_data,
            'cache_result': cache_result,
            'success': True
        }
        
    except Exception as e:
        if log_callback:
            log_callback(f"Error in video generation: {str(e)}")
        return None 