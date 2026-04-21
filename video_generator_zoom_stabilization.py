"""
Per-frame zoom-level stabilization for video generation.

For follow_2d / follow_3d / follow_3d_rotate modes the map bounding box has a
fixed geographic size, but `detect_zoom_level()` is a discrete function of the
bbox — small frame-to-frame shifts (tile-grid alignment + latitude scaling)
can flip the chosen zoom level between N and N+1 near the max_tiles threshold.
Unchecked this produces a very visible zoom-flicker in the final video.

To prevent that, this module computes a stabilized per-frame zoom from the
precomputed per-frame bboxes using a **direction-aware hysteresis** rule:

  * First change ever: transition instantly (so a genuine fast zoom that
    starts at frame 0 does not wait for dwell_frames).
  * New natural zoom continues the same direction as the most recent
    transition (e.g. 12 -> 13 -> 14 when the last change was 11 -> 12):
    transition instantly. This keeps monotonic sweeps snappy.
  * New natural zoom reverses direction (e.g. just went up to 13 and now
    natural wants 12 again): require `dwell_frames` consecutive frames of
    agreement before transitioning, where
        dwell_frames = camera_zoom_stabilization (seconds) * fps.

This eliminates the 12 <-> 13 flicker pattern (each round-trip requires two
full dwell periods), while still allowing a genuine directional zoom sweep
to pass through a chain of zoom levels as fast as the data demands.

Tile-budget reasoning:
  * When we actually transition, the target zoom equals the current frame's
    natural zoom, which `detect_zoom_level()` has just confirmed fits within
    the max_tiles budget — so transitions never exceed the budget.
  * When we refuse a reversal transition, the frame keeps rendering at the
    previously-stabilized zoom. For reversals from higher to lower zoom that
    means temporarily rendering at one zoom level above natural for the
    current bbox, which can nudge the tile count slightly above max_tiles.
    The overshoot is small (adjacent-zoom tile-count ratios near the
    threshold are O(1)) and bounded in time (it resolves as soon as natural
    returns to current_zoom or the dwell completes), so the trade of a
    temporary, minor over-budget render in exchange for no flicker is an
    acceptable one.

Dynamic mode also uses this helper for uniformity: its bbox usually changes
monotonically, so most frames simply track natural. The hysteresis only ever
kicks in on the rare threshold-flicker frames.
"""

from image_generator_maptileutils import detect_zoom_level
from image_generator_utils import compute_video_max_tiles


def compute_stabilized_zooms(
    bboxes,
    json_data,
    map_style,
    canvas_size_override=None,
    debug_callback=None,
    mode_label=None,
    bypass_hysteresis_from_index=None,
):
    """
    Produce a stabilized per-frame zoom-level list for the given bbox list,
    using a direction-aware hysteresis (see the module docstring for the full
    rule).

    The returned list has the same length as `bboxes`, and each entry is the
    zoom level to use for that frame (or None where the input bbox is None,
    e.g. dynamic-mode empty frames).

    Args:
        bboxes: list of bbox tuples or None entries.
        json_data: job data (read for video_fps, map_detail, video_tilt,
            resolution, and camera_zoom_stabilization).
        map_style: map style name (same value that step 3 / step 4 / step 5
            will use, so the max_tiles budget matches).
        canvas_size_override: optional (width, height) for follow_3d_rotate
            background rendering. Forwarded to `compute_video_max_tiles()` so
            the stabilized zoom matches what the oversized-canvas render will
            actually produce.
        debug_callback: optional callable for debug logging.
        mode_label: optional string used in debug messages only.
        bypass_hysteresis_from_index: optional 0-based frame index from which
            onward hysteresis is disabled and zoom simply tracks the natural
            (per-frame `detect_zoom_level`) value. Intended for the final
            pan-out window in follow modes, where the bbox grows rapidly and
            the hysteresis rule would otherwise lock zoom at a much too high
            level — rendering a huge area at zoom-in tiles, blowing the tile
            budget. All frames before this index still use the normal rule.

    Returns:
        list[int | None] of length len(bboxes).
    """
    if not bboxes:
        return []

    max_tiles = compute_video_max_tiles(
        json_data, map_style, canvas_size_override=canvas_size_override
    )
    video_fps = float(json_data.get('video_fps', 30))

    dwell_seconds_raw = json_data.get('camera_zoom_stabilization', 5)  # Always 5 at the moment, not saved in the json, but could potentially be
    try:
        dwell_seconds = float(dwell_seconds_raw)
    except (TypeError, ValueError):
        dwell_seconds = 2.0

    # dwell_seconds <= 0 disables stabilization: zoom tracks natural every frame.
    if dwell_seconds <= 0:
        dwell_frames = 1
        stabilization_enabled = False
    else:
        dwell_frames = max(1, int(round(dwell_seconds * video_fps)))
        stabilization_enabled = True

    def _natural_zoom(bbox):
        if bbox is None:
            return None
        return detect_zoom_level(
            bbox,
            max_tiles=max_tiles,
            map_style=map_style,
        )

    current_zoom = None
    # +1 = last transition increased the zoom level, -1 = decreased it,
    # 0 = no transitions yet (used to allow the very first change instantly).
    last_change_direction = 0
    pending_zoom = None
    pending_count = 0
    result = []

    bypass_idx = bypass_hysteresis_from_index if bypass_hysteresis_from_index is not None else None
    pan_bypass_frame_count = 0

    for i, bbox in enumerate(bboxes):
        natural = _natural_zoom(bbox)

        if natural is None:
            result.append(None)
            continue

        # Final-pan-out bypass: once we enter the pan window, the bbox grows
        # rapidly on every frame and the hysteresis rule would hold zoom at
        # the prior (much higher) level while the viewport expands — pulling
        # in huge numbers of zoom-in tiles to cover the now-large area. Track
        # natural zoom for the rest of the video so the tile budget is
        # respected and the zoom-out reads visually natural. The same zoom
        # list is consumed by steps 3, 4, and 5, so cache keys stay in sync.
        in_pan_bypass = bypass_idx is not None and i >= bypass_idx

        if current_zoom is None:
            # First valid frame — adopt natural without establishing a direction.
            current_zoom = natural
            last_change_direction = 0
            pending_zoom = None
            pending_count = 0
        elif in_pan_bypass:
            if natural != current_zoom:
                last_change_direction = 1 if natural > current_zoom else -1
                current_zoom = natural
            pending_zoom = None
            pending_count = 0
            pan_bypass_frame_count += 1
        elif not stabilization_enabled:
            if natural != current_zoom:
                last_change_direction = 1 if natural > current_zoom else -1
                current_zoom = natural
            pending_zoom = None
            pending_count = 0
        elif natural == current_zoom:
            # On-target: clear any pending reversal attempt. (A brief return to
            # current_zoom resets the dwell counter, which is the desired behavior
            # for flicker suppression.)
            pending_zoom = None
            pending_count = 0
        else:
            request_direction = 1 if natural > current_zoom else -1
            continues_trend = (
                last_change_direction == 0
                or request_direction == last_change_direction
            )

            if continues_trend:
                # First change ever, or a step in the same direction as the
                # most recent transition — let it through instantly.
                current_zoom = natural
                last_change_direction = request_direction
                pending_zoom = None
                pending_count = 0
            else:
                # Reversal of direction — gate with dwell.
                if natural == pending_zoom:
                    pending_count += 1
                else:
                    pending_zoom = natural
                    pending_count = 1

                if pending_count >= dwell_frames:
                    current_zoom = natural
                    last_change_direction = request_direction
                    pending_zoom = None
                    pending_count = 0

        result.append(current_zoom)

    if debug_callback:
        valid = [z for z in result if z is not None]
        if valid:
            transitions = sum(1 for a, b in zip(valid, valid[1:]) if a != b)
            unique_sorted = sorted(set(valid))
            label = mode_label or "zoom"
            bypass_suffix = (
                f"; pan-out bypass from frame {bypass_idx + 1}"
                f" ({pan_bypass_frame_count} frames tracked natural)"
                if bypass_idx is not None
                else ""
            )
            if stabilization_enabled:
                debug_callback(
                    f"{label} zoom stabilization: dwell={dwell_frames} frames "
                    f"({dwell_seconds:g}s @ {video_fps:g}fps), max_tiles={max_tiles}; "
                    f"zoom levels used: {unique_sorted}; transitions: {transitions}"
                    f"{bypass_suffix}"
                )
            else:
                debug_callback(
                    f"{label} zoom stabilization: DISABLED "
                    f"(camera_zoom_stabilization={dwell_seconds_raw}); "
                    f"zoom levels used: {unique_sorted}; transitions: {transitions}"
                    f"{bypass_suffix}"
                )

    return result
