"""
Static overlay helpers for video frame rendering.

These overlays are screen-space HUD elements (legend/title/statistics/attribution)
that can be composited after world-space transforms (for example follow_3d tilt).
"""

import numpy as np

from image_generator_utils import get_attribution_text, get_text_theme_colors
from video_generator_create_single_frame_legend import (
    create_legend,
    get_day_legend_data,
    get_filename_legend_data,
    get_month_legend_data,
    get_people_legend_data,
    get_year_legend_data,
)
from video_generator_route_statistics import _draw_video_attribution, _draw_video_statistics


def get_legend_theme_colors(legend_theme):
    """Get legend colors based on theme. Uses central text theme colors."""
    theme = 'dark' if legend_theme == 'dark' else 'light'
    bg_color, border_color, text_color = get_text_theme_colors(theme)
    return {
        'facecolor': bg_color,
        'edgecolor': border_color,
        'textcolor': text_color,
        'framealpha': 0.9 if theme == 'dark' else 0.8,
    }


def draw_video_title(ax, title_text, effective_line_width, json_data, resolution_scale=None, image_height=None, theme='light'):
    """
    Draw video title at the top center of the frame.
    Uses the same styling as statistics/attribution but at larger font size.
    """
    if not title_text:
        return

    if resolution_scale is None:
        from image_generator_utils import calculate_resolution_scale
        resolution_x = int(json_data.get('video_resolution_x', 1920))
        resolution_y = int(json_data.get('video_resolution_y', 1080))
        resolution_scale = calculate_resolution_scale(resolution_x, resolution_y)

    bg_color, border_color, text_color = get_text_theme_colors(theme)

    base_font_size = 24
    font_size = base_font_size * resolution_scale

    base_padding_pixels = 20
    padding_pixels = base_padding_pixels * resolution_scale
    padding_y = (padding_pixels / image_height) if image_height else 0.01

    bbox_linewidth = 0.5 * resolution_scale
    ax.text(
        0.5,
        1.0 - padding_y,
        title_text,
        transform=ax.transAxes,
        color=text_color,
        fontsize=font_size,
        fontweight='bold',
        ha='center',
        va='top',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor=bg_color,
            edgecolor=border_color,
            alpha=0.9,
            linewidth=bbox_linewidth,
        ),
        zorder=110,
    )


def draw_static_overlays_to_axis(
    target_ax,
    points_for_frame,
    json_data,
    effective_line_width,
    resolution_scale,
    width,
    height,
    statistics_data,
    exclude_speed,
    exclude_elevation,
):
    """
    Draw screen-space static overlays to target axis.
    """
    legend_type = json_data.get('legend', '')
    if legend_type in ['file_name', 'year', 'month', 'day', 'people']:
        legend_theme = json_data.get('legend_theme', 'light')
        theme_colors = get_legend_theme_colors(legend_theme)
        flattened_points = [point for route_points in points_for_frame for point in route_points]

        if legend_type == 'file_name':
            legend_handles, legend_labels = get_filename_legend_data(flattened_points, json_data, effective_line_width)
        elif legend_type == 'year':
            legend_handles, legend_labels = get_year_legend_data(flattened_points, json_data, effective_line_width)
        elif legend_type == 'month':
            legend_handles, legend_labels = get_month_legend_data(flattened_points, json_data, effective_line_width)
        elif legend_type == 'day':
            legend_handles, legend_labels = get_day_legend_data(flattened_points, json_data, effective_line_width)
        else:
            legend_handles, legend_labels = get_people_legend_data(flattened_points, json_data, effective_line_width)

        if legend_handles and legend_labels:
            create_legend(target_ax, legend_handles, legend_labels, theme_colors, effective_line_width, resolution_scale=resolution_scale)

    title_text_setting = json_data.get('title_text', 'off')
    if isinstance(title_text_setting, bool):
        title_text_setting = 'light' if title_text_setting else 'off'
    if title_text_setting in ['light', 'dark']:
        title_text_value = (json_data.get('route_name') or '').strip()
        if title_text_value:
            draw_video_title(
                target_ax,
                title_text_value,
                effective_line_width,
                json_data,
                resolution_scale=resolution_scale,
                image_height=height,
                theme=title_text_setting,
            )

    statistics_theme = json_data.get('statistics', 'off')
    if statistics_theme in ['light', 'dark'] and statistics_data:
        _draw_video_statistics(
            target_ax,
            statistics_data,
            json_data,
            effective_line_width,
            statistics_theme,
            exclude_current_speed=exclude_speed,
            exclude_current_elevation=exclude_elevation,
            resolution_scale=resolution_scale,
        )

    attribution_setting = json_data.get('attribution', 'off')
    if attribution_setting in ['light', 'dark']:
        map_style = json_data.get('map_style', '')
        attribution_text = get_attribution_text(map_style)
        _draw_video_attribution(target_ax, attribution_text, attribution_setting, resolution_scale, width, height)


def composite_bottom_center_helper_labels(frame_array, json_data, image_scale, width, height, frame_number=None):
    """
    Composite bottom-center helper labels (speed/HR color + HR width) onto frame.
    """
    labels_to_draw = []

    if json_data.get('speed_based_color_label', False):
        color_label = json_data.get('_speed_based_color_label_image')
        if color_label is not None:
            labels_to_draw.append((color_label, "Speed-based color"))
    elif json_data.get('hr_based_color_label', False):
        color_label = json_data.get('_hr_based_color_label_image')
        if color_label is not None:
            labels_to_draw.append((color_label, "HR-based color"))

    if json_data.get('hr_based_width_label', False):
        width_label = json_data.get('_hr_based_width_label_image')
        if width_label is not None:
            labels_to_draw.append((width_label, "HR-based width"))

    if not labels_to_draw:
        return

    base_padding_bottom = 20
    padding_bottom = int(round(base_padding_bottom * image_scale))
    base_gap_between_labels = 100
    gap_between_labels = int(round(base_gap_between_labels * image_scale))

    def draw_label_at_position(label_array, label_name, x_start):
        label_height, label_width = label_array.shape[:2]
        y_end = height - padding_bottom
        y_start = y_end - label_height
        x_end = x_start + label_width

        if y_start >= 0 and x_start >= 0 and y_end <= height and x_end <= width:
            if label_array.shape[2] == 4:
                alpha = label_array[:, :, 3:4] / 255.0
                rgb_label = label_array[:, :, :3] / 255.0
                frame_region = frame_array[y_start:y_end, x_start:x_end].astype(np.float32) / 255.0
                blended = alpha * rgb_label + (1 - alpha) * frame_region
                frame_array[y_start:y_end, x_start:x_end] = (blended * 255).astype(np.uint8)
            else:
                frame_array[y_start:y_end, x_start:x_end] = label_array[:, :, :3]
        else:
            print(f"Warning: {label_name} label does not fit within frame bounds for frame {frame_number}")

    if len(labels_to_draw) == 1:
        label_array, label_name = labels_to_draw[0]
        label_width = label_array.shape[1]
        x_start = (width - label_width) // 2
        draw_label_at_position(label_array, label_name, x_start)
    else:
        label1_array, label1_name = labels_to_draw[0]
        label2_array, label2_name = labels_to_draw[1]
        label1_width = label1_array.shape[1]
        label2_width = label2_array.shape[1]

        total_width = label1_width + gap_between_labels + label2_width
        start_x = (width - total_width) // 2
        draw_label_at_position(label1_array, label1_name, start_x)
        draw_label_at_position(label2_array, label2_name, start_x + label1_width + gap_between_labels)
