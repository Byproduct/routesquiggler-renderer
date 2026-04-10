"""
Follow-3D camera mode for Route Squiggler.

Builds on follow_2d with a perspective tilt applied as a post-process to each
rendered frame, producing a modest fake-3D effect without a full 3D pipeline.

Step 1 (implemented here):
  - Same EMA camera tracking as follow_2d (same bboxes, same zoom).
  - Each rendered 2D frame is warped by a perspective transform that contracts
    the top edge inward and shifts it slightly downward, simulating a camera
    looking at the map from a slight forward angle.
  - Small black triangles may appear at the top-left/right corners (accepted
    artefact for step 1; future steps will expand the bbox to compensate).

Planned future steps (not implemented here):
  - Larger bboxes to fill the frame after tilt without black corners.
  - Camera heading aligned to direction of travel (camera rotates to look along
    the route, not always "north up").
  - Higher resolution input to compensate for near-camera stretch.

Parameters (read from json_data):
  video_tilt        (float, degrees) — tilt angle from vertical. 0 = no tilt.
                    Reasonable range: 10–30°.  Default: 20°.
  video_zoom        (float, km) — same as follow_2d: vertical view diameter.
  camera_smoothing  (float, seconds) — EMA time constant; same as follow_2d.
"""

import math

import numpy as np
from PIL import Image

from video_generator_follow_2d import precompute_follow_2d_bboxes
from video_generator_utils import binary_search_cutoff_index, compute_sequential_ending_lengths


def get_most_recent_point(points_for_frame):
    """Return the most recent route point across all active routes (by accumulated_time)."""
    last_point = None
    for route_points in points_for_frame:
        if not route_points:
            continue
        route_last = route_points[-1]
        if last_point is None or route_last.accumulated_time > last_point.accumulated_time:
            last_point = route_last
    return last_point


def follow_3d_rotate_angle_degrees(bbox, points_for_frame):
    """
    CCW rotation (degrees) so the vector from camera center to current point points straight up.

    Camera center is the midpoint of the GPS bbox (lon_min/max, lat_min/max).
    """
    if not bbox:
        return 0.0

    last_point = get_most_recent_point(points_for_frame)
    if last_point is None:
        return 0.0

    camera_lon = (bbox[0] + bbox[1]) * 0.5
    camera_lat = (bbox[2] + bbox[3]) * 0.5
    dx = float(last_point.lon) - camera_lon
    dy = float(last_point.lat) - camera_lat
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0

    # Map coordinates: +x east, +y north.
    current_angle = math.degrees(math.atan2(dy, dx))
    return 90.0 - current_angle


def prerot_figure_pixel_offset_for_screen_down(stack_spacing_px, heading_degrees_ccw):
    """
    Figure/buffer pixel offset (x right, y down) between two points such that, after
    apply_heading_rotation_to_frame(..., heading_degrees_ccw), the offset in the
    output frame is purely downward (+y in the raster), i.e. a vertical stack on screen.

    PIL positive angles rotate the image CCW; relative vectors on the bitmap rotate
    with the image by the same angle.
    """
    th = math.radians(heading_degrees_ccw)
    s = float(stack_spacing_px)
    sin_t = math.sin(th)
    cos_t = math.cos(th)
    pre_x = -s * sin_t
    pre_y = s * cos_t
    return pre_x, pre_y


def follow_3d_rotate_point_stats_stack_step_mercator(
    x_range, y_range, width_px, height_px, stack_spacing_px, heading_degrees_ccw
):
    """
    Mercator (data-space) delta per stacked label for follow_3d_rotate point statistics.

    Uses the same linear pixel↔Mercator mapping as the point-statistics draw helpers.
    """
    pre_x, pre_y = prerot_figure_pixel_offset_for_screen_down(stack_spacing_px, heading_degrees_ccw)
    w = float(width_px)
    h = float(height_px)
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    dx_merc = (pre_x / w) * x_range
    dy_merc = -(pre_y / h) * y_range
    return dx_merc, dy_merc


def precompute_follow_3d_rotate_angles(json_data, combined_route_data, log_callback=None, debug_callback=None):
    """
    Pre-compute per-frame EMA-smoothed camera rotation angles for follow_3d_rotate.

    Uses the same camera_smoothing time constant and EMA alpha as follow_2d camera
    position smoothing. Rotation is smoothed in unit-vector space to handle
    wrap-around near +/-180 degrees without discontinuities.
    """
    try:
        video_fps = float(json_data.get('video_fps', 30))
        video_length = float(json_data.get('video_length', 30))
        tail_length = int(json_data.get('tail_length', 0))
        gpx_time_per_video_time = float(combined_route_data.get('gpx_time_per_video_time', 1.0))

        tau = float(json_data.get('camera_smoothing', 1.5))
        dt = 1.0 / video_fps
        alpha = 1.0 - math.exp(-dt / tau)

        extended_video_length, _, _ = compute_sequential_ending_lengths(video_length, tail_length)
        total_frames = int(extended_video_length * video_fps)

        combined_route = combined_route_data.get('combined_route', [])
        follow_2d_bboxes = combined_route_data.get('follow_2d_bboxes_per_frame', [])
        if not combined_route:
            if log_callback:
                log_callback("Error: follow_3d_rotate mode requires combined_route in combined_route_data")
            return None
        if not follow_2d_bboxes or len(follow_2d_bboxes) < total_frames:
            if log_callback:
                log_callback(
                    "Error: follow_3d_rotate mode requires follow_2d_bboxes_per_frame "
                    "precomputed for all frames"
                )
            return None

        first_point = combined_route[0]
        last_route_time = combined_route[-1].accumulated_time

        raw_angles = []
        for frame_number in range(1, total_frames + 1):
            bbox = follow_2d_bboxes[frame_number - 1]
            camera_lon = (bbox[0] + bbox[1]) * 0.5
            camera_lat = (bbox[2] + bbox[3]) * 0.5

            target_time_video = (frame_number - 1) / video_fps
            target_time_route = min(target_time_video * gpx_time_per_video_time, last_route_time)
            cutoff = binary_search_cutoff_index(combined_route, target_time_route)
            target_point = combined_route[cutoff - 1] if cutoff > 0 else first_point

            dx = float(target_point.lon) - camera_lon
            dy = float(target_point.lat) - camera_lat
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                angle = raw_angles[-1] if raw_angles else 0.0
            else:
                angle = 90.0 - math.degrees(math.atan2(dy, dx))
            raw_angles.append(angle)

        if not raw_angles:
            return []

        # EMA in unit-vector space avoids angle wrap jumps.
        theta0 = math.radians(raw_angles[0])
        sx = math.cos(theta0)
        sy = math.sin(theta0)
        smoothed_angles = []
        for angle in raw_angles:
            theta = math.radians(angle)
            tx = math.cos(theta)
            ty = math.sin(theta)
            sx = alpha * tx + (1.0 - alpha) * sx
            sy = alpha * ty + (1.0 - alpha) * sy
            smoothed_angles.append(math.degrees(math.atan2(sy, sx)))

        if debug_callback:
            debug_callback(
                f"follow_3d_rotate: precomputed {len(smoothed_angles)} angles "
                f"(tau={tau} s, alpha/frame={alpha:.5f})"
            )
        return smoothed_angles

    except Exception as e:
        if log_callback:
            log_callback(f"Error in precompute_follow_3d_rotate_angles: {str(e)}")
        return None


def precompute_follow_3d_bboxes(json_data, combined_route_data, log_callback=None, debug_callback=None):
    """
    Pre-compute per-frame bboxes for follow_3d mode.

    Step 1 uses the same EMA-smoothed follow_2d bboxes.  Future steps will
    add a larger bbox here to compensate for the tilt cutting off corners.

    Returns a list of bbox tuples indexed by [frame_number - 1], or None on error.
    """
    return precompute_follow_2d_bboxes(
        json_data, combined_route_data,
        log_callback=log_callback,
        debug_callback=debug_callback,
    )


def compute_bg_canvas_size(video_w: int, video_h: int) -> int:
    """
    Minimum square canvas size (px) so that after any in-plane rotation the
    centre video_w × video_h crop is fully covered by map content (no black
    corners).

    Equals the diagonal of the video frame: ceil(sqrt(video_w² + video_h²)).
    For 1920×1080 this is 2204 px.
    """
    return math.ceil(math.sqrt(video_w * video_w + video_h * video_h))


def expand_bbox_for_rotation(bbox, video_w: int, video_h: int):
    """
    Expand a follow_2d bbox to cover the oversized square canvas needed by
    follow_3d_rotate so that heading rotation never exposes black corners.

    The expanded bbox keeps the same geographic centre and scales the lon/lat
    spans proportionally so the pixel-per-metre ratio (zoom level) is
    unchanged:
        lon_span_new = lon_span × (canvas_size / video_w)
        lat_span_new = lat_span × (canvas_size / video_h)

    Args:
        bbox: (lon_min, lon_max, lat_min, lat_max) in GPS degrees.
        video_w: Output frame width in pixels.
        video_h: Output frame height in pixels.

    Returns:
        tuple: Expanded (lon_min, lon_max, lat_min, lat_max), clamped to
               valid GPS/Web-Mercator ranges.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    canvas_size = compute_bg_canvas_size(video_w, video_h)

    lon_center = (lon_min + lon_max) * 0.5
    lat_center = (lat_min + lat_max) * 0.5
    lon_half = (lon_max - lon_min) * 0.5 * (canvas_size / video_w)
    lat_half = (lat_max - lat_min) * 0.5 * (canvas_size / video_h)

    bg_lon_min = max(-180.0, lon_center - lon_half)
    bg_lon_max = min(180.0,  lon_center + lon_half)
    bg_lat_min = max(-85.0,  lat_center - lat_half)   # Web Mercator clamp
    bg_lat_max = min(85.0,   lat_center + lat_half)

    return (
        round(bg_lon_min, 6),
        round(bg_lon_max, 6),
        round(bg_lat_min, 6),
        round(bg_lat_max, 6),
    )


def precompute_follow_3d_rotate_bg_bboxes(json_data, combined_route_data, log_callback=None, debug_callback=None):
    """
    Pre-compute per-frame background bboxes for follow_3d_rotate.

    Each bg_bbox is an expanded version of the corresponding follow_2d bbox
    sized for a square canvas of sqrt(W²+H²) pixels.  Rendering onto this
    canvas and then cropping the rotated result back to W×H guarantees full
    coverage at any heading angle.

    Returns a list indexed by [frame_number - 1], or None on error.
    """
    follow_2d_bboxes = combined_route_data.get('follow_2d_bboxes_per_frame') if combined_route_data else None
    if not follow_2d_bboxes:
        if log_callback:
            log_callback(
                "Error: precompute_follow_3d_rotate_bg_bboxes requires "
                "follow_2d_bboxes_per_frame in combined_route_data"
            )
        return None

    video_w = int(json_data.get('video_resolution_x', 1920))
    video_h = int(json_data.get('video_resolution_y', 1080))
    canvas_size = compute_bg_canvas_size(video_w, video_h)

    bg_bboxes = [expand_bbox_for_rotation(b, video_w, video_h) for b in follow_2d_bboxes]

    if debug_callback:
        debug_callback(
            f"follow_3d_rotate: precomputed {len(bg_bboxes)} bg bboxes "
            f"(canvas {canvas_size}px for {video_w}×{video_h} output)"
        )
    return bg_bboxes


def apply_tilt_to_frame(frame_array, tilt_degrees):
    """
    Apply a perspective tilt warp to a rendered 2D frame.

    The warp contracts the top edge of the image inward and shifts it slightly
    downward, simulating a top-down camera tilted forward by tilt_degrees.

    Transform model (PIL inverse-mapping):
      Output top-left  (0,  0) → Source (0,    0)      top stays full-width (far)
      Output top-right (W,  0) → Source (W,    0)
      Output bot-left  (0,  H) → Source (cx,   H-cy)   bottom contracts (near/magnified)
      Output bot-right (W,  H) → Source (W-cx, H-cy)

    Closed-form PIL perspective coefficients (derived analytically, g=0):
      a = 1
      b = cx·W / (H·(W-2·cx))
      c = 0
      d = 0
      e = (H-cy)·W / (H·(W-2·cx))
      f = 0
      g = 0
      h = 2·cx / (H·(W-2·cx))

    These map each output pixel back to its source pixel without needing a
    matrix solve.

    At tilt_degrees=20° (sin≈0.342):
      - Bottom edge contracts to ~66% of source width (kx=0.5 factor)
      - The very bottom ~14% of the source is not shown (ky=0.40 factor)
      - Small black triangles may appear at the bottom-left/right corners

    Args:
        frame_array (np.ndarray): Rendered frame (H, W, 3), uint8.
        tilt_degrees (float): Tilt angle in degrees (0 = no effect).

    Returns:
        np.ndarray: Warped frame (H, W, 3), uint8.
    """
    if tilt_degrees <= 0:
        return frame_array

    H, W = frame_array.shape[:2]
    t = math.sin(math.radians(tilt_degrees))

    # Empirical scale factors — control the visual strength of the effect.
    # kx: horizontal contraction at top (top width = W*(1 - 2*kx*t))
    # ky: vertical shift of the top edge (top appears ky*t*H below output top)
    kx = 0.50
    ky = 0.40

    cx = W * t * kx
    cy = H * t * ky

    # PIL perspective coefficients (output → source inverse mapping).
    # Positive tilt: bottom is near (magnified), top is far (normal scale).
    denom = H * (W - 2.0 * cx)   # shared denominator; W-2*cx > 0 for tilt < ~60°
    a = 1.0
    b = cx * W / denom
    c = 0.0
    d = 0.0
    e = (H - cy) * W / denom
    f = 0.0
    g = 0.0
    h = 2.0 * cx / denom

    img = Image.fromarray(frame_array)
    img_warped = img.transform(
        (W, H),
        Image.Transform.PERSPECTIVE,
        (a, b, c, d, e, f, g, h),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(img_warped)


def apply_heading_rotation_to_frame(frame_array, rotation_degrees):
    """
    Rotate a frame around its center before 3D tilt.

    Args:
        frame_array (np.ndarray): Rendered frame (H, W, 3), uint8.
        rotation_degrees (float): CCW rotation in degrees.

    Returns:
        np.ndarray: Rotated frame (H, W, 3), uint8.
    """
    if abs(rotation_degrees) < 1e-6:
        return frame_array

    img = Image.fromarray(frame_array)
    rotated = img.rotate(
        rotation_degrees,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0),
    )
    return np.asarray(rotated)
