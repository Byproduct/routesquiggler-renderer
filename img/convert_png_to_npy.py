# Convert .png image to a numpy array preserving the alpha channel

from PIL import Image
import numpy as np
import sys
import os

def png_to_rgba_npy(input_path, output_path=None):
    # Load image using PIL
    image = Image.open(input_path).convert('RGBA')  # Preserve alpha channel
    
    # Convert to numpy array
    rgba_array = np.array(image)
    
    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.npy'
    
    # Save array
    np.save(output_path, rgba_array)
    print(f"Saved RGBA array to: {output_path}")

# Example usage
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python png_to_npy.py <image.png> [output.npy]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        png_to_rgba_npy(input_file, output_file)
