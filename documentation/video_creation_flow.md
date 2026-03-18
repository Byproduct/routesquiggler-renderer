# Video Creation Flow

Program flow for the Route Squiggler video generation pipeline,
from receiving a job to uploading the finished video.

All calls originate from `VideoGeneratorWorker.video_generator_process()` in `video_generator_main.py`.

---

## High-level overview (the 6 steps)
```
Step 1  Sort GPX files chronologically
Step 2  Create combined route (parse GPX -> route points)
Step 3  Calculate bounding boxes + download map tiles
Step 4  Render map images into shared memory
Step 5  Generate video frames -> encode video 
Step 6  Upload video + thumbnail to storage box
```

---

## Detailed call flow

```
VideoGeneratorWorker.video_generator_process()                    [video_generator_main.py]
|
+- Step 1 - Sort GPX files chronologically
|  |
|  +- get_sorted_gpx_list(gpx_files_info)                        [video_generator_sort_files_chronologically.py]
|      +- sort_gpx_files_chronologically(gpx_files_info)        
|
+- Step 2 - Create combined route
|  |
|  +- create_combined_route(sorted_gpx_files, json_data)          [video_generator_create_combined_route.py]
|      |
|      +- Decides single vs multi-route mode (from track_objects names)
|      |
|      +- _create_multiple_routes(sorted_gpx_files, track_name_map, ...)
|          |
|          +- Groups files by track name
|          |
|          +- For each track:
|          |   +- _create_route_for_track(track_files, route_index, ...)
|          |       +- Parses GPX with gpxpy, builds RoutePoint list
|          |       |   +- _extract_heart_rate_from_gpxpy_point(point)
|          |       |   +- normalize_timestamp / convert_timestamp_to_job_timezone
|          |       |   +- calculate_haversine_distance (from image_generator_utils)
|          |       |
|          |       +- If pruning enabled:
|          |       |   +- prune_route_by_interval(temp_route_data, interval)
|          |       |       +- _reset_track_accumulated_values(...)
|          |       |   +- interpolate_route_by_interval(pruned_route, interval)
|          |       |
|          |       +- Returns route dict per track
|          |
|          +- Computes timing: total_accumulated_time, route_time_per_frame, gpx_time_per_video_time
|          |
|          +- _apply_smoothed_statistics_to_routes(all_routes, ...)
|          |   +- For each route:
|          |       +- _calculate_smoothed_statistics_for_route(route_points, ...)
|          |           (fills heart_rate_smoothed, current_speed_smoothed)
|          |
|          +- update_speed_based_color_range(...)                  [speed_based_color.py]
|          +- update_hr_based_color_range(...)
|          +- update_hr_based_width_range(...)
|          +- create_speed_based_color_label(...)                  (optional label images stored in json_data)
|          +- create_hr_based_color_label(...)
|          +- create_hr_based_width_label(...)
|          |
|          +- Returns first_route dict with all_routes attached
|
+- Step 3 - Calculate bounding boxes + cache map tiles
|  |
|  +- acquire_map_tile_lock(json_data)                            [map_tile_lock.py]
|  |
|  +- cache_map_tiles(json_data, combined_route_data)             [video_generator_cache_map_tiles.py]
|      |
|      +- calculate_unique_bounding_boxes(...)                    [video_generator_calculate_bounding_boxes.py]
|      |   +- calculate_route_time_per_frame(...)
|      |   +- save_final_bounding_box(...)
|      |   +- Multiprocessing:
|      |       +- process_frame_chunk(...)                        (per chunk of frames)
|      |           +- calculate_bounding_box_for_points(...)
|      |               +- split_coordinates_at_longitude_wrap(...)
|      |               +- calculate_bounding_box_for_wrapped_coordinates(...)
|      |
|      +- pre_cache_map_tiles_for_video(unique_bboxes, ...)
|          +- Collects all required (x, y, zoom) tiles across bounding boxes
|          +- cache_required_tiles(required_tiles, map_style, ...)  [map_tile_caching.py]
|              +- is_tile_cached(cache_dir, x, y, zoom, map_style)
|              +- Downloads missing tiles from tile server (or storage box fallback)
|
|  +- release_map_tile_lock(json_data)
|
+- Step 4 - Cache map images (render map tiles -> composited images in shared memory)
|  |
|  +- cache_map_images(json_data, combined_route_data, shared_map_cache, unique_bboxes)
|                                                                 [video_generator_cache_map_images.py]
|      +- cache_map_images_for_video(unique_bboxes, json_data, shared_map_cache)
|          +- Multiprocessing pool:
|              +- create_map_image_worker(args)                   (per unique bounding box)
|                  +- Composites tiles into a single map image
|                  +- Stores result in shared_map_cache dict
|
+- Step 5 - Generate video frames + encode video
|  |
|  +- cache_video_frames(json_data, combined_route_data, shared_map_cache, ...)
|                                                                 [video_generator_cache_video_frames.py]
|      |
|      +- is_simultaneous_mode(combined_route_data)               [video_generator_utils.py]
|      |
|      +- cache_video_frames_for_video(...)
|          |
|          +- create_video_streaming(...)
|              |
|              +- compute_simultaneous_ending_lengths(...)         [video_generator_utils.py]
|              |   or compute_sequential_ending_lengths(...)
|              |
|              +- StreamingFrameGenerator(...)                     [video_generator_streaming_frame_generator.py]
|              |   |
|              |   +- __init__: pre-computes color map, loads stamp, pickles shared data
|              |   |   +- compute_sequential_frames_to_skip(...)   [video_generator_utils.py]
|              |   |   +- _save_shared_worker_data(...)
|              |   |
|              |   +- Multiprocessing Pool spawns workers:
|              |   |   +- _streaming_frame_worker(args)            (per frame)
|              |   |       +- _load_shared_worker_data(filepath)
|              |   |       +- get_route_start_times(all_routes)    [video_generator_utils.py]
|              |   |       +- get_route_delay_seconds(...)
|              |   |       +- binary_search_cutoff_index(...)
|              |   |       +- _timestamp_to_accumulated_time(...)  (POI filtering)
|              |   |       +- generate_video_frame_in_memory(...)  [video_generator_create_single_frame.py]
|              |   |           +- Map background: lookup cached map image (shared_map_cache)
|              |   |           +- Route line: draw route(s), optional speed/HR-based color, HR-based width
|              |   |           +- Route tail: _draw_route_tail / _draw_multi_route_tail / _draw_speed_based_tail
|              |   |           +- Legend (if enabled): create_legend (file_name, year, month, day, or people)
|              |   |           +- Video title: _draw_video_title(route_name)
|              |   |           +- Statistics: _calculate_video_statistics, _draw_video_statistics;
|              |   |               current speed/elevation/HR at point when enabled
|              |   |           +- Attribution: _draw_video_attribution(map style text)
|              |   |           +- Points of interest: _draw_points_of_interest(...)
|              |   |           +- After fig to array: stamp (if enabled), color/width labels,
|              |   |               clock overlay (composite_clock_onto_frame_array)
|              |   |           +- Returns frame as numpy array
|              |   |
|              |   +- make_frame(t): returns frame array to MoviePy
|              |       (handles frame buffering, cloned ending frames, progress updates)
|              |
|              +- VideoClip(frame_generator.make_frame, duration=...)   [moviepy]
|              |
|              +- clip.write_videofile(...)                        (GPU: NVENC or CPU: libx264)
|              |   +- Uses get_ffmpeg_executable()                 [video_generator_cache_video_frames_util.py]
|              |   +- suppress_moviepy_output()
|              |   +- MoviePyDebugLogger(...)
|              |
|              +- Generates thumbnail from last frame (ffmpeg extract + PIL resize)
|              |
|              +- frame_generator.cleanup()
|                  +- _cleanup_shared_worker_data(filepath)
|
|  +- Step 5.5 (optional) - Remove black frames (if hide_complete_routes enabled)
|  |
|  |  +- remove_empty_frames_from_video(video_path, ...)          [video_generator_remove_empty_frames.py]
|  |      +- detect_black_frames(video_path, fps)
|  |      |   +- detect_keyframes(video_path, fps)
|  |      |   |   +- get_ffprobe_executable()
|  |      |   +- Multiprocessing:
|  |      |       +- _analyze_frame_chunk(args)
|  |      |
|  |      +- remove_black_frames_ffmpeg(video_path, black_frames, output)
|  |
|  |  +- Re-generates thumbnail from processed video
|  |
|
+- Step 6 - Upload video + thumbnail (non-test jobs only)
|  |
|  +- upload_video_to_storage_box(video_path, thumbnail_path, ...)  [video_generator_main.py]
|      +- ProgressFTP (FTP with upload progress tracking)
|      +- retry_operation(_do_upload, max_attempts=50)              [network_retry.py]
|
|  +- update_job_status(api_url, user, ..., 'ok')                   [job_request.py]
|  +- cleanup_temporary_job_folder(job_id)
|
+- cleanup_resources()
    +- plt.close('all'), gc.collect()
    +- Terminates lingering ffmpeg processes
```

---

## Key shared data flowing through the pipeline

| Data | Created in | Consumed by |
|------|-----------|-------------|
| `sorted_gpx_files` | Step 1 | Step 2 |
| `combined_route_data` (with `all_routes`, `gpx_time_per_video_time`) | Step 2 | Steps 3, 4, 5 |
| `unique_bounding_boxes` | Step 3 | Steps 3, 4 |
| `shared_map_cache` (multiprocessing Manager dict) | Step 4 | Step 5 (looked up per frame) |
| `shared_route_cache` (multiprocessing Manager dict) | Step 5 | Step 5 (frame workers) |
| `video_path` | Step 5 | Steps 5.5, 6 |