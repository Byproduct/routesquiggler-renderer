"""
Remove black/empty frames from videos when hide_complete_routes is enabled.
This module analyzes video frames to detect black frames and removes them
using ffmpeg's copy stream (no re-encoding) for efficiency.
"""

# Standard library imports
import multiprocessing
import os
import subprocess
import tempfile
from pathlib import Path

# Third-party imports
import cv2
import numpy as np

# Local imports
from video_generator_cache_video_frames import get_ffmpeg_executable
from write_log import write_debug_log, write_log


def get_ffprobe_executable():
    """
    Get the correct ffprobe executable path for the current platform.
    Uses the same approach as get_ffmpeg_executable for consistency.
    
    Returns:
        str: Path to ffprobe executable
    """
    # First, try to use the same ffmpeg that MoviePy/imageio uses
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            # Replace 'ffmpeg' with 'ffprobe' in the path
            ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
            if os.path.exists(ffprobe_path):
                return ffprobe_path
    except Exception:
        pass
    
    # Fallback: platform-specific defaults
    if os.name == 'nt':  # Windows
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_ffprobe = os.path.join(script_dir, 'ffprobe.exe')
        if os.path.exists(local_ffprobe):
            return local_ffprobe
        return 'ffprobe.exe'
    else:  # Linux/Mac
        return 'ffprobe'


def detect_keyframes(video_path, fps, debug_callback=None):
    """
    Detect keyframe positions in a video using ffprobe.
    
    Args:
        video_path (str): Path to the video file
        fps (float): Frames per second of the video
        debug_callback (callable, optional): Function to call for debug messages
    
    Returns:
        set: Set of frame numbers that are keyframes
    """
    try:
        ffprobe_executable = get_ffprobe_executable()
        
        # Use ffprobe to get keyframe positions
        # Select frames where key_frame == 1
        cmd = [
            ffprobe_executable,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'frame=key_frame,pkt_pts_time',
            '-of', 'csv=p=0',
            video_path
        ]
        
        if debug_callback:
            debug_callback("Detecting keyframes in video...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        keyframes = set()
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                is_keyframe = parts[0].strip()
                pts_time_str = parts[1].strip()
                
                # Skip if timestamp is empty or invalid
                if not pts_time_str:
                    continue
                
                try:
                    pts_time = float(pts_time_str)
                except ValueError:
                    # Skip invalid timestamp values
                    continue
                
                if is_keyframe == '1':
                    # Convert timestamp to frame number
                    frame_num = int(pts_time * fps)
                    keyframes.add(frame_num)
        
        if debug_callback:
            debug_callback(f"Detected {len(keyframes)} keyframes")
        
        return keyframes
        
    except Exception as e:
        if debug_callback:
            debug_callback(f"Warning: Could not detect keyframes: {e}. Will re-encode all segments.")
        return set()  # Return empty set, will trigger re-encoding for all segments


def _analyze_frame_chunk(args):
    """
    Worker function to analyze a chunk of frames for black frames.
    
    Args:
        args: Tuple of (video_path, start_frame, end_frame, fps, frame_chunk_size)
    
    Returns:
        list: List of frame numbers that are black
    """
    video_path, start_frame, end_frame, fps, frame_chunk_size = args
    
    black_frames = []
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return black_frames
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set frame position to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Analyze frames in this chunk
        frame_num = start_frame
        while frame_num < min(end_frame, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if frame is effectively black (max pixel value <= 5)
            # A max pixel value of 5/255 (~2% brightness) appears completely black to human eye
            # This accounts for compression noise/artifacts while still detecting truly black frames
            # Fast check: if max pixel value <= threshold, the frame is effectively black
            max_pixel_value = np.max(frame)
            
            # Threshold of 5 accounts for video compression noise while detecting effectively black frames
            BLACK_FRAME_THRESHOLD = 5
            if max_pixel_value <= BLACK_FRAME_THRESHOLD:
                black_frames.append(frame_num)
            
            frame_num += 1
        
        cap.release()
        
    except Exception as e:
        write_debug_log(f"Error analyzing frame chunk {start_frame}-{end_frame}: {e}")
    
    return black_frames


def detect_black_frames(video_path, fps, debug_callback=None, max_workers=None):
    """
    Detect black frames in a video using multiprocessing.
    
    Args:
        video_path (str): Path to the video file
        fps (float): Frames per second of the video
        debug_callback (callable, optional): Function to call for debug messages
        max_workers (int, optional): Maximum number of worker processes
    
    Returns:
        list: Sorted list of frame numbers that are black
    """
    if debug_callback:
        debug_callback("Starting black frame detection...")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if debug_callback:
            debug_callback(f"Error: Could not open video file: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if debug_callback:
        debug_callback(f"Analyzing {total_frames} frames for black frames...")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(4, multiprocessing.cpu_count())
    
    # Divide work into chunks of 100 frames
    chunk_size = 100
    chunks = []
    for start_frame in range(0, total_frames, chunk_size):
        end_frame = min(start_frame + chunk_size, total_frames)
        chunks.append((video_path, start_frame, end_frame, fps, chunk_size))
    
    if debug_callback:
        debug_callback(f"Dividing work into {len(chunks)} chunks of ~{chunk_size} frames each")
    
    # Process chunks in parallel
    all_black_frames = []
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.map(_analyze_frame_chunk, chunks)
        
        # Combine results from all chunks
        for chunk_black_frames in results:
            all_black_frames.extend(chunk_black_frames)
    
    # Sort and remove duplicates
    all_black_frames = sorted(set(all_black_frames))
    
    if debug_callback:
        debug_callback(f"Detected {len(all_black_frames)} black frames")
        if all_black_frames:
            # Print black frame numbers in debug output
            # Group consecutive frames for cleaner output
            if len(all_black_frames) <= 50:
                # If few frames, list them all
                frame_list = ', '.join(map(str, all_black_frames))
                debug_callback(f"Black frame numbers: {frame_list}")
            else:
                # If many frames, show ranges and count
                ranges = []
                start = all_black_frames[0]
                end = all_black_frames[0]
                
                for frame_num in all_black_frames[1:]:
                    if frame_num == end + 1:
                        end = frame_num
                    else:
                        if start == end:
                            ranges.append(str(start))
                        else:
                            ranges.append(f"{start}-{end}")
                        start = frame_num
                        end = frame_num
                
                # Add the last range
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                
                debug_callback(f"Black frame numbers: {', '.join(ranges)} ({len(all_black_frames)} total)")
    
    return all_black_frames


def remove_black_frames_ffmpeg(video_path, black_frames, output_path, debug_callback=None, gpu_rendering=True, max_workers=None):
    """
    Remove black frames from video using ffmpeg.
    
    Note: Frame-accurate removal requires re-encoding. Copy codec can only cut at
    keyframes, which may include black frames that fall within GOP boundaries.
    This function uses re-encoding to ensure all black frames are removed.
    
    Note: Always uses CPU encoding (libx264) for black frame removal, regardless of
    gpu_rendering setting, to avoid NVENC compatibility issues.
    
    Args:
        video_path (str): Path to input video file
        black_frames (list): Sorted list of frame numbers to remove
        output_path (str): Path to output video file
        debug_callback (callable, optional): Function to call for debug messages
        gpu_rendering (bool, optional): Ignored - always uses CPU encoding for black frame removal
        max_workers (int, optional): Maximum number of worker threads for encoding
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not black_frames:
        if debug_callback:
            debug_callback("No black frames to remove, skipping video processing")
        return False
    
    if debug_callback:
        debug_callback(f"Removing {len(black_frames)} black frames from video...")
    
    try:
        # Get ffmpeg executable
        ffmpeg_executable = get_ffmpeg_executable()
        
        # Create a filter complex expression to skip black frames
        # We'll use ffmpeg's select filter to skip specific frame numbers
        # However, select works with timestamps, so we need to convert frame numbers to timestamps
        
        # Get video FPS to convert frame numbers to timestamps
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if debug_callback:
                debug_callback(f"Error: Could not open video file: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Convert black frame numbers to timestamps (in seconds)
        black_timestamps = [frame_num / fps for frame_num in black_frames]
        
        # Create filter complex to select frames that are NOT in the black frames list
        # We'll use the select filter with a custom expression
        # select='not(eq(n\,FRAME_NUM))' for each black frame
        
        # Build select expression: select frames where frame number is not in black_frames
        # We'll use multiple select filters chained together, or use a more efficient approach
        
        # More efficient: use select with a Python-like expression
        # select='not(n in [FRAME_LIST])' - but ffmpeg doesn't support this directly
        
        # Alternative approach: use select with multiple conditions
        # select='not(eq(n\,100)*not(eq(n\,120))*not(eq(n\,200))...'
        # But this gets complex with many frames
        
        # Best approach: use ffmpeg's concat demuxer with segments
        # Create segments that exclude black frames, then concat them
        
        # Create temporary directory for segments
        temp_dir = tempfile.mkdtemp(prefix='ffmpeg_segments_')
        
        try:
            # Build segments: ranges of frames that should be included
            segments = []
            last_frame = -1
            
            for black_frame in black_frames:
                # If there's a gap before this black frame, create a segment
                if black_frame > last_frame + 1:
                    start_frame = last_frame + 1
                    end_frame = black_frame - 1
                    # Only add segment if it has at least one frame
                    if end_frame >= start_frame:
                        start_time = start_frame / fps
                        end_time = (end_frame + 1) / fps  # End time is exclusive
                        segments.append((start_time, end_time))
                
                last_frame = black_frame
            
            # Add final segment if there are frames after the last black frame
            if last_frame < total_frames - 1:
                start_frame = last_frame + 1
                end_frame = total_frames - 1
                # Only add segment if it has at least one frame
                if end_frame >= start_frame:
                    start_time = start_frame / fps
                    end_time = (end_frame + 1) / fps
                    segments.append((start_time, end_time))
            
            if debug_callback:
                debug_callback(f"Created {len(segments)} segments to concatenate")
            
            # Detect keyframes to optimize segment extraction
            # Segments that start/end at keyframes can use copy codec (fast)
            # Segments that don't align with keyframes need re-encoding (frame-accurate)
            keyframes = detect_keyframes(video_path, fps, debug_callback)
            
            # Create segment files using ffmpeg
            segment_files = []
            segments_using_copy = 0
            segments_using_reencode = 0
            
            for i, (start_time, end_time) in enumerate(segments):
                # Convert times to frame numbers to check keyframe alignment
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                # OPTIMIZATION: Split segments at keyframes to maximize copy codec usage
                # Find the last keyframe before the end of this segment
                # If found, we can copy up to that keyframe and only re-encode the remainder
                if keyframes and start_frame == 0:
                    # For segments starting from the beginning, find last keyframe before end
                    keyframes_before_end = [kf for kf in keyframes if start_frame <= kf < end_frame]
                    if keyframes_before_end:
                        last_keyframe = max(keyframes_before_end)
                        # Split into two parts:
                        # 1. Copy from start to last keyframe (copy codec)
                        # 2. Re-encode from last keyframe to end (frame-accurate)
                        
                        # Part 1: Copy codec segment
                        keyframe_time = last_keyframe / fps
                        copy_duration = keyframe_time - start_time
                        if copy_duration > 0:
                            copy_segment_file = os.path.join(temp_dir, f'segment_{i:04d}_copy.mp4')
                            cmd = [
                                ffmpeg_executable,
                                '-ss', str(start_time),
                                '-i', video_path,
                                '-t', str(copy_duration),
                                '-c', 'copy',
                                '-avoid_negative_ts', 'make_zero',
                                '-y',
                                copy_segment_file
                            ]
                            
                            if debug_callback:
                                debug_callback(f"Extracting segment {i+1} part 1/2 (copy): {start_time:.3f}s - {keyframe_time:.3f}s")
                            
                            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                            if result.returncode == 0 and os.path.exists(copy_segment_file) and os.path.getsize(copy_segment_file) > 0:
                                segment_files.append(copy_segment_file)
                                segments_using_copy += 1
                            else:
                                if debug_callback:
                                    debug_callback(f"Warning: Failed to extract copy segment {i+1} part 1: {result.stderr}")
                        
                        # Part 2: Re-encode segment from keyframe to end
                        reencode_start_time = keyframe_time
                        reencode_duration = end_time - reencode_start_time
                        if reencode_duration > 0:
                            reencode_segment_file = os.path.join(temp_dir, f'segment_{i:04d}_reencode.mp4')
                            cmd = [
                                ffmpeg_executable,
                                '-i', video_path,
                                '-ss', str(reencode_start_time),
                                '-t', str(reencode_duration),
                                '-c:v', 'libx264',
                                '-pix_fmt', 'yuv420p',
                                '-preset', 'medium',
                                '-crf', '25',
                                '-profile:v', 'high',
                                '-level', '4.1',
                                '-c:a', 'copy',
                                '-avoid_negative_ts', 'make_zero',
                                '-y',
                                reencode_segment_file
                            ]
                            if max_workers:
                                cmd.insert(3, '-threads')
                                cmd.insert(4, str(max_workers))
                            
                            if debug_callback:
                                debug_callback(f"Extracting segment {i+1} part 2/2 (re-encode): {reencode_start_time:.3f}s - {end_time:.3f}s")
                            
                            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                            if result.returncode == 0 and os.path.exists(reencode_segment_file) and os.path.getsize(reencode_segment_file) > 0:
                                segment_files.append(reencode_segment_file)
                                segments_using_reencode += 1
                            else:
                                if debug_callback:
                                    debug_callback(f"Warning: Failed to extract re-encode segment {i+1} part 2: {result.stderr}")
                        
                        continue  # Skip the normal segment extraction for this segment
                
                # Normal segment extraction (for segments not starting at 0, or when no keyframes found)
                segment_file = os.path.join(temp_dir, f'segment_{i:04d}.mp4')
                duration = end_time - start_time
                
                # Check if segment boundaries align with keyframes
                # If both start and end are at keyframes (or very close), we can use copy codec
                # Allow 1 frame tolerance for floating point precision
                start_is_keyframe = start_frame in keyframes or (start_frame - 1) in keyframes or (start_frame + 1) in keyframes
                end_is_keyframe = end_frame in keyframes or (end_frame - 1) in keyframes or (end_frame + 1) in keyframes
                
                use_copy_codec = start_is_keyframe and end_is_keyframe and keyframes
                
                if use_copy_codec:
                    # Use copy codec for segments aligned with keyframes (fast, no re-encoding)
                    cmd = [
                        ffmpeg_executable,
                        '-ss', str(start_time),
                        '-i', video_path,
                        '-t', str(duration),
                        '-c', 'copy',  # Copy codec (no re-encoding)
                        '-avoid_negative_ts', 'make_zero',
                        '-y',
                        segment_file
                    ]
                    segments_using_copy += 1
                    method = "copy"
                else:
                    # Re-encode for frame-accurate cutting when not aligned with keyframes
                    # Always use CPU encoding (libx264) for black frame removal to avoid NVENC issues
                    # Use the same CPU encoding settings as the original video creation
                    cmd = [
                        ffmpeg_executable,
                        '-i', video_path,
                        '-ss', str(start_time),
                        '-t', str(duration),
                        '-c:v', 'libx264',  # CPU encoder (always use CPU for black frame removal)
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'medium',  # Same as video creation
                        '-crf', '25',  # Constant Rate Factor (same as video creation)
                        '-profile:v', 'high',  # Profile (same as video creation)
                        '-level', '4.1',  # H.264 level (same as video creation)
                        '-c:a', 'copy',  # Copy audio codec
                        '-avoid_negative_ts', 'make_zero',
                        '-y',
                        segment_file
                    ]
                    # Add threads parameter if max_workers is specified
                    if max_workers:
                        cmd.insert(3, '-threads')
                        cmd.insert(4, str(max_workers))
                    
                    segments_using_reencode += 1
                    method = "re-encode"
                
                if debug_callback:
                    debug_callback(f"Extracting segment {i+1}/{len(segments)} ({method}): {start_time:.3f}s - {end_time:.3f}s")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    if debug_callback:
                        debug_callback(f"Warning: Failed to extract segment {i+1}: {result.stderr}")
                    continue
                
                if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                    segment_files.append(segment_file)
            
            if debug_callback:
                debug_callback(f"Segment extraction: {segments_using_copy} using copy codec, {segments_using_reencode} using re-encoding")
            
            if not segment_files:
                if debug_callback:
                    debug_callback("Error: No valid segments created")
                return False
            
            # Create concat file for ffmpeg
            concat_file = os.path.join(temp_dir, 'concat.txt')
            with open(concat_file, 'w') as f:
                for segment_file in segment_files:
                    # Use absolute path and escape single quotes if needed
                    abs_path = os.path.abspath(segment_file)
                    # Replace backslashes with forward slashes for cross-platform compatibility
                    abs_path = abs_path.replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")
            
            if debug_callback:
                debug_callback(f"Concatenating {len(segment_files)} segments...")
            
            # Concatenate segments using ffmpeg concat demuxer
            # Since segments are already re-encoded, we can use copy codec for concatenation
            cmd = [
                ffmpeg_executable,
                '-f', 'concat',
                '-safe', '0',  # Allow absolute paths
                '-i', concat_file,
                '-c', 'copy',  # Copy codec for concatenation (segments are already encoded)
                '-y',  # Overwrite output
                output_path
            ]
            
            if debug_callback:
                debug_callback(f"Running FFmpeg concat command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if debug_callback:
                debug_callback("FFmpeg concat completed successfully")
            
            return True
            
        finally:
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
                if debug_callback:
                    debug_callback(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                if debug_callback:
                    debug_callback(f"Warning: Could not clean up temp directory: {e}")
        
    except subprocess.CalledProcessError as e:
        if debug_callback:
            debug_callback(f"Error: FFmpeg command failed: {e}")
            debug_callback(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        if debug_callback:
            debug_callback(f"Error removing black frames: {e}")
        return False


def remove_empty_frames_from_video(video_path, json_data=None, debug_callback=None, log_callback=None, max_workers=None, gpu_rendering=True):
    """
    Remove black/empty frames from a video when hide_complete_routes is enabled.
    
    Args:
        video_path (str): Path to the input video file
        json_data (dict, optional): Job data containing video parameters
        debug_callback (callable, optional): Function to call for debug messages
        log_callback (callable, optional): Function to call for log messages
        max_workers (int, optional): Maximum number of worker processes/threads
        gpu_rendering (bool, optional): Whether to use GPU rendering for re-encoding (default: True)
    
    Returns:
        str: Path to the output video file (same as input if no processing needed), or None if error
    """
    try:
        # Check if hide_complete_routes is enabled
        if json_data:
            hide_complete_routes_raw = json_data.get('hide_complete_routes', False)
            if isinstance(hide_complete_routes_raw, str):
                hide_complete_routes = hide_complete_routes_raw.lower() in ('true', '1', 'yes')
            else:
                hide_complete_routes = bool(hide_complete_routes_raw)
        else:
            hide_complete_routes = False
        
        if not hide_complete_routes:
            if debug_callback:
                debug_callback("hide_complete_routes is not enabled, skipping black frame removal")
            return video_path
        
        if not os.path.exists(video_path):
            if debug_callback:
                debug_callback(f"Error: Video file not found: {video_path}")
            return None
        
        if debug_callback:
            debug_callback(f"hide_complete_routes is enabled, removing black frames from: {video_path}")
        
        # Get video FPS from json_data or detect it
        if json_data:
            fps = float(json_data.get('video_fps', 30))
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                if debug_callback:
                    debug_callback(f"Error: Could not open video file: {video_path}")
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        
        # Detect black frames
        black_frames = detect_black_frames(video_path, fps, debug_callback, max_workers)
        
        if not black_frames:
            if debug_callback:
                debug_callback("No black frames detected, video does not need processing")
            return video_path
        
        # Create output path (replace original file)
        # Use a temporary file first, then replace the original
        video_dir = os.path.dirname(video_path)
        video_basename = os.path.basename(video_path)
        temp_output_path = os.path.join(video_dir, f'temp_{video_basename}')
        
        # Remove black frames
        success = remove_black_frames_ffmpeg(
            video_path,
            black_frames,
            temp_output_path,
            debug_callback,
            gpu_rendering,
            max_workers
        )
        
        if not success:
            if debug_callback:
                debug_callback("Error: Failed to remove black frames from video")
            # Clean up temp file if it exists
            if os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except:
                    pass
            return None
        
        # Replace original file with processed file
        try:
            # On Windows, we need to remove the original file first
            if os.path.exists(video_path):
                os.remove(video_path)
            os.rename(temp_output_path, video_path)
            
            if debug_callback:
                debug_callback(f"Successfully removed {len(black_frames)} black frames from video")
                debug_callback(f"Output video: {video_path}")
            
            return video_path
            
        except Exception as e:
            if debug_callback:
                debug_callback(f"Error replacing video file: {e}")
            # Clean up temp file
            if os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except:
                    pass
            return None
        
    except Exception as e:
        if debug_callback:
            debug_callback(f"Error in remove_empty_frames_from_video: {e}")
        return None

