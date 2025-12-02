from PIL import Image

def speed_based_color(value: float):
    """
    Return an RGB tuple (0-255) for a value between 0 and 1
    using a Garmin-style speed-based gradient.
    """
    v = max(0.0, min(1.0, float(value)))

    # Key points for interpolation
    scale = [
        (0.0, (0, 75, 155)),     # dark blue
        (0.25, (60, 150, 245)),  # lighter blue
        (0.5, (125, 225, 25)),   # green
        (0.75, (255, 170, 76)),  # yellow/orange
        (1.0, (175, 0, 0)),      # dark red
    ]

    for i in range(len(scale) - 1):
        v0, c0 = scale[i]
        v1, c1 = scale[i + 1]

        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0)
            r = int(c0[0] + (c1[0] - c0[0]) * t)
            g = int(c0[1] + (c1[1] - c0[1]) * t)
            b = int(c0[2] + (c1[2] - c0[2]) * t)
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
        color = speed_based_color(v)
        for y in range(height):
            pixels[x, y] = color

    img.save("output.png")
    print("Saved output.png")