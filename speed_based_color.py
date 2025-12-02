from PIL import Image

def speed_based_color(value: float):
    """
    Return an RGB tuple (0-1 range, matplotlib format) for a value between 0 and 1
    using a Garmin-style speed-based gradient.
    """
    v = max(0.0, min(1.0, float(value)))

    # Key points for interpolation (0-255 range for reference)
    scale_255 = [
        (0.0, (0, 75, 155)),     # dark blue
        (0.25, (60, 150, 245)),  # lighter blue
        (0.5, (125, 225, 25)),   # green
        (0.75, (255, 170, 76)),  # yellow/orange
        (1.0, (175, 0, 0)),      # dark red
    ]
    
    # Convert to 0-1 range for matplotlib
    scale = [(v0, (c0[0] / 255.0, c0[1] / 255.0, c0[2] / 255.0)) for v0, c0 in scale_255]

    for i in range(len(scale) - 1):
        v0, c0 = scale[i]
        v1, c1 = scale[i + 1]

        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0)
            r = c0[0] + (c1[0] - c0[0]) * t
            g = c0[1] + (c1[1] - c0[1]) * t
            b = c0[2] + (c1[2] - c0[2]) * t
            return (r, g, b)

    return scale[-1][1]


# Create scale as output.png if executed independently
if __name__ == "__main__":
    width = 200
    height = 20

    img = Image.new("RGB", (width, height))
    pixels = img.load()

    for x in range(width):
        v = x / (width - 1)
        color_01 = speed_based_color(v)  # Returns 0-1 range
        # Convert to 0-255 range for PIL Image
        color_255 = (int(color_01[0] * 255), int(color_01[1] * 255), int(color_01[2] * 255))
        for y in range(height):
            pixels[x, y] = color_255

    img.save("output.png")
    print("Saved output.png")