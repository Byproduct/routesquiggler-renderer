"""
HTML file generator for image galleries.
Creates a simple HTML page that displays resized images linking to their full-size versions.
"""

# Standard library imports
import ftplib
from io import BytesIO
from typing import List

def generate_image_gallery_html(route_name: str, zoom_levels: List[int]) -> str:
    """
    Generate HTML content for the image gallery.
    
    Args:
        route_name: Base name for the image files
        zoom_levels: List of zoom levels to include
        
    Returns:
        str: HTML content
    """
    html_parts = []
    
    # HTML document structure
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html lang="en">')
    html_parts.append('<head>')
    html_parts.append('    <meta charset="UTF-8">')
    html_parts.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html_parts.append('    <meta name="robots" content="noindex">')
    html_parts.append(f'    <title>{route_name}</title>')
    html_parts.append('    <style>')
    html_parts.append('        body {')
    html_parts.append('            background-color: #f0f0f0;')
    html_parts.append('            font-family: Verdana, sans-serif;')
    html_parts.append('            margin: 20px;')
    html_parts.append('            padding: 0;')
    html_parts.append('        }')
    html_parts.append('        h1 {')
    html_parts.append('            color: #333;')
    html_parts.append('            margin-bottom: 20px;')
    html_parts.append('        }')
    html_parts.append('        .instruction {')
    html_parts.append('            font-size: 14px;')
    html_parts.append('            color: #666;')
    html_parts.append('            margin-bottom: 30px;')
    html_parts.append('        }')
    html_parts.append('        .zoom-level {')
    html_parts.append('            font-weight: bold;')
    html_parts.append('            margin-top: 30px;')
    html_parts.append('            margin-bottom: 10px;')
    html_parts.append('            color: #333;')
    html_parts.append('        }')
    html_parts.append('        .image-container {')
    html_parts.append('            margin-bottom: 20px;')
    html_parts.append('        }')
    html_parts.append('        img {')
    html_parts.append('            border: 1px solid #ccc;')
    html_parts.append('            box-shadow: 0 2px 4px rgba(0,0,0,0.1);')
    html_parts.append('        }')
    html_parts.append('        a {')
    html_parts.append('            text-decoration: none;')
    html_parts.append('        }')
    html_parts.append('        a:hover img {')
    html_parts.append('            box-shadow: 0 4px 8px rgba(0,0,0,0.2);')
    html_parts.append('        }')
    html_parts.append('    </style>')
    html_parts.append('</head>')
    html_parts.append('<body>')
    html_parts.append(f'    <h1>{route_name}</h1>')
    html_parts.append('    <div class="instruction">Click to enlarge or save</div>')

    # Add each image
    for zoom_level in sorted(zoom_levels):
        html_parts.append(f'    <div class="zoom-level">Zoom level {zoom_level}</div>')
        image_name = f"{route_name}_zoom{zoom_level}.png"
        html_parts.append('    <div class="image-container">')
        html_parts.append(f'        <a href="{image_name}"><img src="{image_name}" height="640" alt="Zoom level {zoom_level}"></a>')
        html_parts.append('    </div>')
    
    # Close HTML document
    html_parts.append('</body>')
    html_parts.append('</html>')
    
    return '\n'.join(html_parts)

def upload_gallery_html(
    html_content: str,
    storage_box_address: str,
    storage_box_user: str,
    storage_box_password: str,
    job_id: str,
    folder: str
) -> bool:
    """
    Upload the gallery HTML file to the storage box.
    
    Args:
        html_content: The HTML content to upload
        storage_box_address: FTP server address
        storage_box_user: FTP username
        storage_box_password: FTP password
        job_id: Job ID for the folder structure
        folder: Subfolder name
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    ftp = None
    try:
        # Connect to FTP
        ftp = ftplib.FTP(storage_box_address)
        ftp.login(storage_box_user, storage_box_password)
        
        # Navigate to target directory
        ftp.cwd('media')
        ftp.cwd(job_id)
        ftp.cwd(folder)
        
        # Upload HTML file
        ftp.storbinary('STOR images.html', BytesIO(html_content.encode('utf-8')))
        
        # Verify file exists and has size
        try:
            file_size = ftp.size('images.html')
            if file_size <= 0:
                return False
        except:
            return False
        
        return True
        
    except Exception as e:
        print(f"Failed to upload gallery HTML: {str(e)}")
        return False
        
    finally:
        if ftp:
            try:
                ftp.quit()
            except:
                pass 