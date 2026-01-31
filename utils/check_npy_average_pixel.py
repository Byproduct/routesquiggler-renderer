#!/usr/bin/env python3
"""
Standalone script to load an .npy file and output its average pixel value.

Uses the same logic as the map tile cache sweep corrupt-file check:
- Normalizes data to 0-255 scale (uint8 as-is, float32/64 * 255).
- For 3D arrays with >= 3 channels: averages over RGB (first 3 channels).
- For 2D arrays: averages over the whole array.

Usage:
    python check_npy_average_pixel.py path/to/file.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def average_pixel_value(npy_path: str) -> float:
    """
    Compute average pixel value of an npy array on a 0-255 scale.

    Args:
        npy_path: Path to the .npy file.

    Returns:
        Average pixel value (0-255).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the array shape/dtype cannot be handled.
    """
    path = Path(npy_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {npy_path}")

    tile_data = np.load(str(path))

    # Normalize to 0-255 scale (same as map_tile_cache_sweep)
    if tile_data.dtype == np.uint8:
        data_255 = tile_data
    elif tile_data.dtype in (np.float32, np.float64):
        data_255 = (tile_data * 255).astype(np.float64)
    else:
        data_255 = np.asarray(tile_data, dtype=np.float64) * 255

    if len(tile_data.shape) == 3 and tile_data.shape[2] >= 3:
        rgb = data_255[:, :, :3]
        return float(np.mean(rgb))
    elif len(tile_data.shape) == 2:
        return float(np.mean(data_255))
    else:
        raise ValueError(
            f"Unsupported shape {tile_data.shape}. Expected 2D (H, W) or 3D (H, W, C) with C >= 3."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Output the average pixel value (0-255) of an .npy file."
    )
    parser.add_argument(
        "npy_path",
        type=str,
        help="Path to the .npy file",
    )
    args = parser.parse_args()

    try:
        avg = average_pixel_value(args.npy_path)
        print(avg)
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
