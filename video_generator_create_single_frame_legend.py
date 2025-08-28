"""
Legend-related functions for video generation single frame creation.
This module contains functions for creating and managing legends in video frames.
"""

from video_generator_create_single_frame_utils import hex_to_rgba


def create_legend(ax, legend_handles, legend_labels, theme_colors, effective_line_width):
    """Create and configure legend with given handles, labels, and theme colors"""
    from matplotlib.lines import Line2D
    
    # Create legend
    legend = ax.legend(
        legend_handles, 
        legend_labels,
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0),
        bbox_transform=ax.transAxes,
        facecolor=theme_colors['facecolor'],
        edgecolor=theme_colors['edgecolor'],
        framealpha=theme_colors['framealpha'],
        fontsize=10,
        title=None
    )
    
    # Set zorder on the legend object after creation
    legend.set_zorder(45)  # Above tails but below statistics
    
    # Set text color
    for text in legend.get_texts():
        text.set_color(theme_colors['textcolor'])
    
    # Adjust legend position with padding
    legend.set_bbox_to_anchor((0.99, 0.01))  # 10 pixels from bottom and right
    
    return legend


def get_filename_legend_data(points_for_frame, json_data, effective_line_width):
    """Get legend data for filename-based legend"""
    # Gather unique filenames from points_for_frame
    unique_filenames = set()
    for point in points_for_frame:
        if len(point) > 7:
            filename = point[7]
            if filename:
                unique_filenames.add(filename)
    
    if not unique_filenames:
        return None, None
    
    # Create legend entries
    legend_handles = []
    legend_labels = []
    
    # Get track_objects for color mapping
    track_objects = json_data.get('track_objects', [])
    filename_to_color = {}
    for track_obj in track_objects:
        filename = track_obj.get('filename', '')
        if filename.endswith('.gpx'):
            filename = filename[:-4]  # Remove .gpx extension
        filename_to_color[filename] = track_obj.get('color', '#ff0000')
    
    # Sort filenames alphabetically
    sorted_filenames = sorted(unique_filenames)
    
    for filename in sorted_filenames:
        # Get color for this filename
        color = filename_to_color.get(filename, '#ff0000')
        rgba_color = hex_to_rgba(color)
        
        # Create a line for the legend
        from matplotlib.lines import Line2D
        legend_line = Line2D([0], [0], color=rgba_color, linewidth=effective_line_width)
        legend_handles.append(legend_line)
        legend_labels.append(filename)
    
    return legend_handles, legend_labels


def get_year_legend_data(points_for_frame, json_data, effective_line_width):
    """Get legend data for year-based legend"""
    # Gather unique years and their corresponding filenames from points_for_frame
    year_to_filename = {}
    for point in points_for_frame:
        if len(point) > 7:
            timestamp = point[3]  # Element 3 is timestamp
            filename = point[7]   # Element 7 is filename
            if timestamp and filename:
                # Extract year from timestamp (first 4 characters)
                year = str(timestamp)[:4]
                if year.isdigit():  # Ensure it's a valid year
                    year_to_filename[year] = filename
    
    if not year_to_filename:
        return None, None
    
    # Create legend entries
    legend_handles = []
    legend_labels = []
    
    # Get track_objects for color mapping
    track_objects = json_data.get('track_objects', [])
    filename_to_color = {}
    for track_obj in track_objects:
        filename = track_obj.get('filename', '')
        if filename.endswith('.gpx'):
            filename = filename[:-4]  # Remove .gpx extension
        filename_to_color[filename] = track_obj.get('color', '#ff0000')
    
    # Sort years numerically
    sorted_years = sorted(year_to_filename.keys())
    
    for year in sorted_years:
        # Get filename for this year and then get its color
        filename = year_to_filename[year]
        color = filename_to_color.get(filename, '#ff0000')
        rgba_color = hex_to_rgba(color)
        
        # Create a line for the legend
        from matplotlib.lines import Line2D
        legend_line = Line2D([0], [0], color=rgba_color, linewidth=effective_line_width)
        legend_handles.append(legend_line)
        legend_labels.append(year)
    
    return legend_handles, legend_labels


def get_month_legend_data(points_for_frame, json_data, effective_line_width):
    """Get legend data for month-based legend"""
    # Gather unique months and their corresponding filenames from points_for_frame
    month_to_filename = {}
    for point in points_for_frame:
        if len(point) > 7:
            timestamp = point[3]  # Element 3 is timestamp
            filename = point[7]   # Element 7 is filename
            if timestamp and filename:
                # Extract year and month from timestamp (first 7 characters: YYYY-MM)
                timestamp_str = str(timestamp)
                if len(timestamp_str) >= 7 and timestamp_str[4] == '-':
                    month = timestamp_str[:7]  # YYYY-MM format
                    if month[:4].isdigit() and month[5:7].isdigit():  # Ensure valid format
                        month_to_filename[month] = filename
    
    if not month_to_filename:
        return None, None
    
    # Create legend entries
    legend_handles = []
    legend_labels = []
    
    # Get track_objects for color mapping
    track_objects = json_data.get('track_objects', [])
    filename_to_color = {}
    for track_obj in track_objects:
        filename = track_obj.get('filename', '')
        if filename.endswith('.gpx'):
            filename = filename[:-4]  # Remove .gpx extension
        filename_to_color[filename] = track_obj.get('color', '#ff0000')
    
    # Sort months chronologically
    sorted_months = sorted(month_to_filename.keys())
    
    for month in sorted_months:
        # Get filename for this month and then get its color
        filename = month_to_filename[month]
        color = filename_to_color.get(filename, '#ff0000')
        rgba_color = hex_to_rgba(color)
        
        # Create a line for the legend
        from matplotlib.lines import Line2D
        legend_line = Line2D([0], [0], color=rgba_color, linewidth=effective_line_width)
        legend_handles.append(legend_line)
        legend_labels.append(month)
    
    return legend_handles, legend_labels


def get_day_legend_data(points_for_frame, json_data, effective_line_width):
    """Get legend data for day-based legend"""
    # Gather unique days and their corresponding filenames from points_for_frame
    day_to_filename = {}
    for point in points_for_frame:
        if len(point) > 7:
            timestamp = point[3]  # Element 3 is timestamp
            filename = point[7]   # Element 7 is filename
            if timestamp and filename:
                # Extract year, month, and day from timestamp (first 10 characters: YYYY-MM-DD)
                timestamp_str = str(timestamp)
                if len(timestamp_str) >= 10 and timestamp_str[4] == '-' and timestamp_str[7] == '-':
                    day = timestamp_str[:10]  # YYYY-MM-DD format
                    if day[:4].isdigit() and day[5:7].isdigit() and day[8:10].isdigit():  # Ensure valid format
                        day_to_filename[day] = filename
    
    if not day_to_filename:
        return None, None
    
    # Create legend entries
    legend_handles = []
    legend_labels = []
    
    # Get track_objects for color mapping
    track_objects = json_data.get('track_objects', [])
    filename_to_color = {}
    for track_obj in track_objects:
        filename = track_obj.get('filename', '')
        if filename.endswith('.gpx'):
            filename = filename[:-4]  # Remove .gpx extension
        filename_to_color[filename] = track_obj.get('color', '#ff0000')
    
    # Sort days chronologically
    sorted_days = sorted(day_to_filename.keys())
    
    for day in sorted_days:
        # Get filename for this day and then get its color
        filename = day_to_filename[day]
        color = filename_to_color.get(filename, '#ff0000')
        rgba_color = hex_to_rgba(color)
        
        # Create a line for the legend
        from matplotlib.lines import Line2D
        legend_line = Line2D([0], [0], color=rgba_color, linewidth=effective_line_width)
        legend_handles.append(legend_line)
        legend_labels.append(day)
    
    return legend_handles, legend_labels 


def get_people_legend_data(points_for_frame, json_data, effective_line_width):
    """Get legend data for people-based legend (uses track object 'name' or falls back to filename)"""
    # Gather unique filenames from points_for_frame
    unique_filenames = set()
    for point in points_for_frame:
        if len(point) > 7:
            filename = point[7]
            if filename:
                unique_filenames.add(filename)
    
    if not unique_filenames:
        return None, None
    
    # Build mappings from filename to color and to person name
    track_objects = json_data.get('track_objects', [])
    filename_to_color = {}
    filename_to_name = {}
    for track_obj in track_objects:
        filename = track_obj.get('filename', '')
        if filename.endswith('.gpx'):
            filename = filename[:-4]  # Remove .gpx extension
        filename_to_color[filename] = track_obj.get('color', '#ff0000')
        filename_to_name[filename] = (track_obj.get('name') or '').strip()
    
    # Map person label to representative filename (deduplicate labels)
    label_to_filename = {}
    for filename in sorted(unique_filenames):
        person_label = filename_to_name.get(filename, '') or filename
        if person_label not in label_to_filename:
            label_to_filename[person_label] = filename
    
    # Create legend entries sorted by person label
    legend_handles = []
    legend_labels = []
    from matplotlib.lines import Line2D
    for label in sorted(label_to_filename.keys()):
        filename = label_to_filename[label]
        color = filename_to_color.get(filename, '#ff0000')
        rgba_color = hex_to_rgba(color)
        legend_line = Line2D([0], [0], color=rgba_color, linewidth=effective_line_width)
        legend_handles.append(legend_line)
        legend_labels.append(label)
    
    return legend_handles, legend_labels