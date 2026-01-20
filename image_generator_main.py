"""
Image generation worker classes for the Route Squiggler render client.
This module handles the main image generation workflow in separate threads.
"""

# Standard library imports
import multiprocessing as mp

# Third-party imports (matplotlib backend must be set before pyplot import)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for image generation

from PySide6.QtCore import QObject, QThread, Signal

# Local imports
import image_generator_utils
from image_generator_maptileutils import detect_zoom_level
from image_generator_multiprocess import generate_images_parallel
from job_request import update_job_status
from map_tile_lock import acquire_map_tile_lock, release_map_tile_lock
from video_generator_create_combined_route import create_combined_route
from video_generator_sort_files_chronologically import get_sorted_gpx_list

class ImageGeneratorWorker(QObject):
    """Worker class to handle image generation in a separate thread."""
    finished = Signal()
    error = Signal(str)
    log_message = Signal(str)
    debug_message = Signal(str)  # Emits debug messages (only logged if debug_logging is enabled)
    status_queue_ready = Signal(object)  # Emits the status queue for UI updates
    zoom_levels_ready = Signal(list)  # Emits the zoom levels as soon as they're calculated
    job_completed = Signal(str)  # Emits job ID when all uploads are successful

    def __init__(self, json_data, gpx_files_info, storage_box_credentials, api_url, user, hardware_id, app_version, max_workers=None):
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
        self.results = None

    def image_generator_process(self):
        """Main processing method that runs in the worker thread."""
        try:
            import time
            start_time = time.time()
            
            self.log_message.emit("Starting image generation.")
                      
            # Create track lookup
            track_objects = self.json_data.get('track_objects', [])
            track_lookup = {}
            for track_obj in track_objects:
                if 'filename' in track_obj:
                    track_lookup[track_obj['filename']] = track_obj
            
            # Check if speed-based or HR-based coloring/width is enabled
            use_speed_based_color = self.json_data.get('speed_based_color', False)
            use_hr_based_color = self.json_data.get('hr_based_color', False)
            use_hr_based_width = self.json_data.get('hr_based_width', False)
            use_hr_based_width_label = self.json_data.get('hr_based_width_label', False)
            # Use RoutePoint system if any of these features need HR or speed data
            use_data_based_color = use_speed_based_color or use_hr_based_color or use_hr_based_width or use_hr_based_width_label
            
            # Initialize processor for potential fallback use
            processor = image_generator_utils.GPXProcessor()
            
            # Process GPX files - use RoutePoint-based system if any data-based feature is enabled
            route_points_data = None
            if use_data_based_color:
                # DATA-BASED MODE: Use RoutePoint-based system
                if use_hr_based_color:
                    feature_type = "HR-based coloring"
                elif use_speed_based_color:
                    feature_type = "Speed-based coloring"
                elif use_hr_based_width or use_hr_based_width_label:
                    feature_type = "HR-based width"
                else:
                    feature_type = "Data-based"
                self.debug_message.emit(f"{feature_type} enabled - using RoutePoint-based system")
                
                # Sort GPX files chronologically (required for create_combined_route)
                sorted_gpx_files = get_sorted_gpx_list(
                    self.gpx_files_info,
                    log_callback=self.log_message.emit,
                    debug_callback=(self.debug_message.emit if hasattr(self, 'debug_message') else self.log_message.emit)
                )
                
                if not sorted_gpx_files:
                    self.log_message.emit(f"Warning: Could not sort GPX files chronologically for {feature_type.lower()}. Falling back to standard mode.")
                    use_data_based_color = False
                    use_speed_based_color = False
                    use_hr_based_color = False
                    use_hr_based_width = False
                    use_hr_based_width_label = False
                else:
                    # Create combined route using RoutePoint system
                    # For images, disable pruning by setting route_accuracy to 'maximum'
                    # This ensures we get all points from the GPX files
                    # Temporarily set route_accuracy, then restore it after route creation
                    original_route_accuracy = self.json_data.get('route_accuracy') if self.json_data else None
                    if self.json_data:
                        self.json_data['route_accuracy'] = 'maximum'  # Disable pruning for images
                    
                    self.log_message.emit(f"Creating combined route for {feature_type.lower()}...")
                    route_points_data = create_combined_route(
                        sorted_gpx_files,
                        self.json_data,  # Pass original json_data so label gets stored correctly
                        progress_callback=None,  # No progress callback for image generation
                        log_callback=self.log_message.emit,
                        debug_callback=(self.debug_message.emit if hasattr(self, 'debug_message') else self.log_message.emit)
                    )
                    
                    # Restore original route_accuracy if it was set
                    if self.json_data:
                        if original_route_accuracy is not None:
                            self.json_data['route_accuracy'] = original_route_accuracy
                        elif 'route_accuracy' in self.json_data:
                            del self.json_data['route_accuracy']
                    
                    if not route_points_data:
                        self.log_message.emit(f"Warning: Could not create combined route for {feature_type.lower()}. Falling back to standard mode.")
                        use_data_based_color = False
                        use_speed_based_color = False
                        use_hr_based_color = False
                        use_hr_based_width = False
                        use_hr_based_width_label = False
                    else:
                        # Extract all coordinates for bounds calculation
                        all_routes = route_points_data.get('all_routes', [])
                        all_lats = []
                        all_lons = []
                        for route in all_routes:
                            combined_route = route.get('combined_route', [])
                            for point in combined_route:
                                all_lats.append(point.lat)
                                all_lons.append(point.lon)
                        
                        if not all_lats or not all_lons:
                            self.log_message.emit("Warning: No valid coordinates in combined route. Falling back to standard mode.")
                            use_data_based_color = False
                            use_speed_based_color = False
                            use_hr_based_color = False
                            use_hr_based_width = False
                            use_hr_based_width_label = False
            
            if not use_data_based_color:
                # STANDARD MODE: Use existing coordinate extraction system
                track_coords_with_metadata = []
                all_lats = []
                all_lons = []
                
                for gpx_info in self.gpx_files_info:
                    filename = gpx_info.get('filename')
                    if not filename:
                        continue
                    
                    lats, lons = processor.extract_coordinates_from_gpx_content(gpx_info['content'])
                    if lats and lons:
                        track_metadata = track_lookup.get(filename, {})
                        color = track_metadata.get('color', '#2E8B57')
                        name = track_metadata.get('name', 'Unknown Track')
                        
                        track_coords_with_metadata.append((lats, lons, color, name, filename))
                        all_lats.extend(lats)
                        all_lons.extend(lons)
                
                if not track_coords_with_metadata:
                    raise ValueError("No valid coordinates found in any GPX file")
            else:
                # Speed-based mode: create track_coords_with_metadata from RoutePoint data for legends, because legends still use track_coords_with_metadata
                # Note: this is probably unnecessary, for now legends are disabled if speed-based color is enabled, and they will likely stay that way.
                track_coords_with_metadata = []
                all_routes = route_points_data.get('all_routes', [])
                for route in all_routes:
                    combined_route = route.get('combined_route', [])
                    if not combined_route:
                        continue
                    
                    # Get filename from first point
                    filename = combined_route[0].filename if combined_route else 'Unknown'
                    # Try lookup with filename as-is, then with .gpx extension (track_lookup keys may include .gpx)
                    track_metadata = track_lookup.get(filename, {})
                    if not track_metadata and not filename.endswith('.gpx'):
                        track_metadata = track_lookup.get(filename + '.gpx', {})
                    color = track_metadata.get('color', '#2E8B57')
                    name = track_metadata.get('name', 'Unknown Track')
                    
                    # Extract coordinates for this route
                    route_lats = [point.lat for point in combined_route]
                    route_lons = [point.lon for point in combined_route]
                    
                    track_coords_with_metadata.append((route_lats, route_lons, color, name, filename))
            
            # Calculate statistics if enabled
            statistics_data = None
            statistics_setting = self.json_data.get('statistics', 'off')
            if statistics_setting in ['light', 'dark'] and self.json_data.get('job_type') == 'image':
                try:
                    self.debug_message.emit("Calculating statistics from GPX files...")
                    statistics_data = processor.calculate_statistics_from_gpx_files(self.gpx_files_info, track_lookup, self.json_data)
                    if statistics_data:
                        self.debug_message.emit("Statistics calculated successfully")
                        # Determine units based on imperial_units setting
                        imperial_units = self.json_data and self.json_data.get('imperial_units', False) is True
                        distance_unit = "miles" if imperial_units else "km"
                        speed_unit = "mph" if imperial_units else "km/h"
                        # Log calculated values for debugging
                        self.debug_message.emit(f"  Starting time: {statistics_data.get('starting_time', 'N/A')}")
                        self.debug_message.emit(f"  Ending time: {statistics_data.get('ending_time', 'N/A')}")
                        self.debug_message.emit(f"  Elapsed time: {statistics_data.get('elapsed_time', 'N/A')}")
                        self.debug_message.emit(f"  Distance: {statistics_data.get('distance', 'N/A')} {distance_unit}")
                        self.debug_message.emit(f"  Average speed: {statistics_data.get('average_speed', 'N/A')} {speed_unit}")
                        avg_hr = statistics_data.get('average_hr', '0')
                        if avg_hr and avg_hr != '0':
                            self.debug_message.emit(f"  Average HR: {avg_hr} bpm")
                    else:
                        self.log_message.emit("Warning: Could not calculate statistics (no timestamps found)")
                except Exception as e:
                    self.log_message.emit(f"Warning: Error calculating statistics: {str(e)}")
                    statistics_data = None
            
            # Apply newest_on_top sorting if specified
            newest_on_top = self.json_data.get('newest_on_top', False)
            if newest_on_top:
                # Sort tracks by date (oldest first for plotting, so newest end up plotted last and on top)
                # self.log_message.emit(f"Sorting tracks: newest on top (newest_on_top=True)")
                track_coords_with_metadata.sort(key=lambda track: track_lookup.get(track[4], {}).get('date', ''))
            else:
                # Sort tracks by date (newest first for plotting, so oldest end up plotted last and on top)
                # self.log_message.emit(f"Sorting tracks: oldest on top (newest_on_top=False)")
                track_coords_with_metadata.sort(key=lambda track: track_lookup.get(track[4], {}).get('date', ''), reverse=True)
            
            # Log the track order for debugging
            track_order = [f"{track[4]} ({track_lookup.get(track[4], {}).get('date', 'no date')})" for track in track_coords_with_metadata]
            # self.log_message.emit(f"Track plotting order: {', '.join(track_order)}")
            
            # Extract resolution to decide appropriate tile limits
            def _parse_dimension(raw_value, default_value):
                try:
                    if isinstance(raw_value, str):
                        cleaned = raw_value.strip().lower().replace('px', '').replace(',', '').replace(' ', '')
                        return int(float(cleaned))
                    return int(float(raw_value))
                except Exception:
                    return default_value

            raw_x = self.json_data.get('image_resolution_x', 1920)
            raw_y = self.json_data.get('image_resolution_y', 1080)
            resolution_x = _parse_dimension(raw_x, 1920)
            resolution_y = _parse_dimension(raw_y, 1080)
            try:
                self.debug_message.emit(f"Requested image_resolution: raw=({raw_x}x{raw_y}) parsed=({resolution_x}x{resolution_y})")
            except Exception:
                pass

            total_pixels = resolution_x * resolution_y
            if total_pixels < 8_000_000:  # Less than 8MP
                min_tiles = 5
                max_tiles = 100
            elif total_pixels < 20_000_000:  # 8MP to 20MP
                min_tiles = 10
                max_tiles = 300
            else:  # More than 20MP
                min_tiles = 20
                max_tiles = 400

            # Calculate map bounds
            map_bounds = (
                min(all_lons), max(all_lons),
                min(all_lats), max(all_lats)
            )
            
            # Get suitable zoom levels
            zoom_levels = detect_zoom_level(
                map_bounds,
                min_tiles=min_tiles,
                max_tiles=max_tiles,
                map_style=self.json_data.get('map_style', 'osm')
            )
            
            if not zoom_levels:
                raise ValueError("No suitable zoom levels found")
            
            # Emit zoom levels immediately so UI can create status labels
            self.zoom_levels_ready.emit(zoom_levels)
            
            # Create status queue and inform UI immediately so it can start receiving updates
            status_queue = mp.Queue()
            self.status_queue_ready.emit(status_queue)

            # Generate images in parallel, passing the pre-created status queue
            # Ensure all numeric parameters are properly converted from JSON
            try:
                map_transparency_value = int(self.json_data.get('map_transparency', 30))
            except (ValueError, TypeError):
                map_transparency_value = 30
            
            try:
                line_width_value = int(self.json_data.get('line_width', 3))
            except (ValueError, TypeError):
                line_width_value = 3
            
            resolution_x_value = _parse_dimension(self.json_data.get('image_resolution_x', 1920), 1920)
            resolution_y_value = _parse_dimension(self.json_data.get('image_resolution_y', 1080), 1080)
            
            # Log the resolved output resolution for debugging
            try:
                self.debug_message.emit(f"Output image resolution: {resolution_x_value}x{resolution_y_value}")
            except Exception:
                pass
            
            # Acquire map tile lock before generating images (tiles are downloaded during generation)
            lock_acquired, lock_error = acquire_map_tile_lock(
                self.json_data,
                log_callback=self.log_message.emit,
                debug_callback=(self.debug_message.emit if hasattr(self, 'debug_message') else None)
            )
            
            if not lock_acquired:
                # Lock acquisition failed after 60 minutes - mark job as error
                raise ValueError(f"Map tile lock acquisition failed: {lock_error}")
            
            # Update status to "downloading maps (job_id)"
            job_id = str(self.json_data.get('job_id', ''))
            from update_status import update_status
            update_status(f"downloading maps ({job_id})", api_key=self.user)
            
            try:
                self.results, _ = generate_images_parallel(
                    zoom_levels=zoom_levels,
                    map_style=self.json_data.get('map_style', 'osm'),
                    map_bounds=map_bounds,
                    track_coords_with_metadata=track_coords_with_metadata,
                    track_lookup=track_lookup,
                    map_transparency=map_transparency_value,
                    line_width=line_width_value,
                    resolution_x=resolution_x_value,
                    resolution_y=resolution_y_value,
                    legend=self.json_data.get('legend', 'off'),
                    statistics=statistics_setting,
                    statistics_data=statistics_data,
                    json_data=self.json_data,
                    status_queue=status_queue,
                    storage_box_address=self.storage_box_address,
                    storage_box_user=self.storage_box_user,
                    storage_box_password=self.storage_box_password,
                    job_id=str(self.json_data.get('job_id', '')),
                    folder=self.json_data.get('folder', ''),
                    route_name=self.json_data.get('route_name', 'unknown'),
                    max_workers=self.max_workers,
                    route_points_data=route_points_data  # Pass RoutePoint data for speed-based coloring
                )
            finally:
                # Always release the lock after image generation, even if it fails
                release_map_tile_lock(
                    self.json_data,
                    log_callback=self.log_message.emit,
                    debug_callback=(self.debug_message.emit if hasattr(self, 'debug_message') else None)
                )
            
            # Update status to "rendering (job_id)" after map tile download completes
            update_status(f"rendering ({job_id})", api_key=self.user)
            
            # After all workers have completed, check if we have all results
            if self.results and len(self.results) == len(zoom_levels):
                # All workers succeeded
                self.job_completed.emit(str(self.json_data.get('job_id', '')))
                update_job_status(
                    self.api_url, 
                    self.user, 
                    self.hardware_id, 
                    self.app_version, 
                    self.json_data.get('job_id', ''), 
                    'ok',
                    self.log_message.emit
                )
            else:
                # Some workers failed or missing results
                update_job_status(
                    self.api_url, 
                    self.user, 
                    self.hardware_id, 
                    self.app_version, 
                    self.json_data.get('job_id', ''), 
                    'error',
                    self.log_message.emit
                )
            
            # Calculate and emit completion time
            elapsed_time = int(time.time() - start_time)
            self.log_message.emit(f"Image generation completed in {elapsed_time} seconds.")
            
            self.finished.emit()
            
        except Exception as e:
            import traceback
            error_msg = f"Error in worker thread: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            update_job_status(
                self.api_url, 
                self.user, 
                self.hardware_id, 
                self.app_version, 
                self.json_data.get('job_id', ''), 
                'error',
                self.log_message.emit
            )
        finally:
            # Always clean up resources to prevent Qt painter conflicts
            self.cleanup_resources()

    def cleanup_resources(self):
        """Clean up matplotlib and Qt resources to prevent painter conflicts."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # Close all matplotlib figures to release Qt painters
            plt.close('all')
            
            # Clear matplotlib cache
            matplotlib.pyplot.clf()
            matplotlib.pyplot.cla()
            
            # Force garbage collection to clean up any remaining Qt objects
            import gc
            gc.collect()
            
            self.debug_message.emit("Matplotlib and Qt resources cleaned up")
            
        except Exception as e:
            self.log_message.emit(f"Warning: Error during resource cleanup: {str(e)}")

class ImageWorkerThread(QThread):
    """Thread class to run the worker."""
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.worker.moveToThread(self)
        
    def run(self):
        """Execute the worker's image_generator_process method."""
        self.worker.image_generator_process() 