#!/usr/bin/env python3
# File       : analogueClock.py
# Description: Analogue clock using Python closure.
# Copyright 2022 Harvard University. All Rights Reserved.
import io
import multiprocessing
import os
import time

# Project img/clock dir (script lives in utils/, so project root is parent of utils)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_CLOCK_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "img", "clock")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import datetime as dt
from PIL import Image

# Optional bundled font (e.g. Verdana.ttf) in the same folder as this script.
# If present, used on Linux etc. where Verdana may not be installed.
_BUNDLED_FONT_NAMES = ("Verdana.ttf", "verdana.ttf", "clock_font.ttf")
_cached_clock_font = None


def _get_clock_font():
    """Return a FontProperties for clock labels: bundled font in script dir if present, else Verdana."""
    global _cached_clock_font
    if _cached_clock_font is not None:
        return _cached_clock_font
    for name in _BUNDLED_FONT_NAMES:
        path = os.path.join(_SCRIPT_DIR, name)
        if os.path.isfile(path):
            try:
                _cached_clock_font = fm.FontProperties(fname=path)
                return _cached_clock_font
            except Exception:
                continue
    _cached_clock_font = fm.FontProperties(family="Verdana")
    return _cached_clock_font

# Define your closure here
def clock_hand(r: float):  # do not change this function name and parameter list
    """Function for a clock hand
    Parameters
    ----------
    r : float
        Length of the clock hand
    Returns
    -------
    callable
        Returns the callable closure function.  The returned function takes a
        single floating point argument `theta` and returns a list-like object
        (x, y) for the `x` and `y` Cartesian coordinates of the clock hand
        position.
    """
    # TODO: implement the closure.  Replace the `lambda theta: (0, 0)`
    # expression below with the name of your implemented closure.

    def cart_coords(theta: float):
        x = r * np.cos(theta * np.pi / 180)
        y = r * np.sin(theta * np.pi / 180)
        return (x, y)

    return cart_coords  # replace the lambda with your closure


# Base output size in pixels (folder "2"); figsize (5,5) @ 100 dpi = 500
BASE_SIZE = 500
DPI = 100


def render_clock(hour: int, minute: int) -> np.ndarray:
    """Draw the clock for the given time and return RGBA image as uint8 array (BASE_SIZE x BASE_SIZE x 4)."""
    r_h, r_m = 6, 8

    hour_fn = clock_hand(r_h)
    x_h, y_h = hour_fn(90 - 30 * hour - (minute / 2))
    minute_fn = clock_hand(r_m)
    x_m, y_m = minute_fn(90 - 6 * minute)

    fig = plt.figure(figsize=(BASE_SIZE / DPI, BASE_SIZE / DPI))
    fig.patch.set_facecolor("none")
    rect = [0, 0, 1, 1]

    ax_carthesian = fig.add_axes(rect)
    ax_carthesian.set_facecolor("none")
    ax_polar = fig.add_axes(rect, polar=True, frameon=False)
    ax_polar.set_facecolor("none")

    face_radius = r_m + 3
    ax_carthesian.set_xlim(-face_radius, face_radius)
    ax_carthesian.set_ylim(-face_radius, face_radius)
    ax_carthesian.set_aspect("equal")

    face = mpatches.Circle((0, 0), face_radius, facecolor="#dddddd", edgecolor="none", zorder=0)
    ax_carthesian.add_patch(face)
    clip_circle = mpatches.Circle((0, 0), face_radius, transform=ax_carthesian.transData)
    ax_carthesian.set_clip_path(clip_circle)

    ax_carthesian.plot([0, x_h], [0, y_h], color="black", linewidth=10)
    ax_carthesian.plot([0, x_m], [0, y_m], color="black", linewidth=7)
    hub = mpatches.Circle((0, 0), 0.6, facecolor="black", edgecolor="none", zorder=5)
    ax_carthesian.add_patch(hub)

    number_radius = face_radius * 0.85
    for n in range(1, 13):
        theta = np.pi / 2 - n * np.pi / 6
        x = number_radius * np.cos(theta)
        y = number_radius * np.sin(theta)
        ax_carthesian.text(x, y, str(n), ha="center", va="center", fontproperties=_get_clock_font(), fontsize=18, color="#000000", zorder=10)

    am_pm = "AM" if hour < 12 else "PM"
    ax_carthesian.text(0, number_radius - 4.5, am_pm, ha="center", va="top", fontproperties=_get_clock_font(), fontsize=18, color="#000000", zorder=10)

    ax_carthesian.axis("off")

    ax_polar.set_rlim(0, face_radius)
    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax_polar.set_xticklabels([])
    plt.setp(ax_polar.get_yticklabels(), visible=False)
    ax_polar.set_theta_direction(-1)
    ax_polar.set_theta_offset(np.pi / 3.0)
    ax_polar.grid(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGBA"), dtype=np.uint8)
    return img


def resize_rgba(img: np.ndarray, size: int) -> np.ndarray:
    """Resize RGBA uint8 image to (size, size, 4)."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.uint8)


# Size factors relative to base (folder "3" = BASE_SIZE). xs = 0.33*0.5, s = 0.33*0.7, 1 = 0.33, 2 = 0.67, 3 = 1, 4 = 1.5
SIZE_FOLDERS = [
    ("xs", BASE_SIZE * 0.33 * 0.5),
    ("s", BASE_SIZE * 0.33 * 0.7),
    ("1", BASE_SIZE * 0.33),
    ("2", BASE_SIZE * 0.67),
    ("3", BASE_SIZE * 1),
    ("4", BASE_SIZE * 1.5),
]
# 24 hours * 60 minutes * 6 size folders
EXPECTED_CLOCK_FILE_COUNT = 24 * 60 * len(SIZE_FOLDERS)


def _count_files_in_dir(dirpath: str) -> int:
    """Return total number of regular files under dirpath (recursive)."""
    count = 0
    for _root, _dirs, files in os.walk(dirpath):
        count += sum(1 for f in files if os.path.isfile(os.path.join(_root, f)))
    return count


def _should_generate(path: str) -> bool:
    """Return True if the file should be generated (missing or size 0)."""
    if not os.path.exists(path):
        return True
    try:
        return os.path.getsize(path) <= 0
    except OSError:
        return True


def _process_one_minute(args: tuple) -> None:
    """Worker: render clock for (hour, minute) and save to all size folders. Used by multiprocessing.
    Skips any target file that already exists and has size > 0.
    """
    hour, minute, output_dir = args
    filename = f"{hour:02d}{minute:02d}.npy"
    paths_to_write = []
    for folder, size in SIZE_FOLDERS:
        path = os.path.join(output_dir, folder, filename)
        if _should_generate(path):
            paths_to_write.append((path, int(round(size))))
    if not paths_to_write:
        return
    img = render_clock(hour, minute)
    for path, px in paths_to_write:
        resized = resize_rgba(img, px)
        np.save(path, resized)


def generate_all_clocks(output_dir: str | None = None, num_processes: int | None = None) -> bool:
    """Generate a clock image for each minute of the day; save as HHmm.npy in six size subfolders
    under output_dir (default: project img/clock). Uses one process per size category to parallelize.
    If output_dir already exists and contains exactly EXPECTED_CLOCK_FILE_COUNT files, skips generation.
    Returns True if generation ran, False if skipped (already complete).
    """
    if output_dir is None:
        output_dir = DEFAULT_CLOCK_OUTPUT_DIR
    if os.path.isdir(output_dir) and _count_files_in_dir(output_dir) == EXPECTED_CLOCK_FILE_COUNT:
        return False
    for folder, _ in SIZE_FOLDERS:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    if num_processes is None:
        num_processes = len(SIZE_FOLDERS)

    jobs = [(hour, minute, output_dir) for hour in range(24) for minute in range(60)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(_process_one_minute, jobs)
    return True


if __name__ == "__main__":
    # Optional: save current time as PNG for quick preview
    t = dt.datetime.now()
    img = render_clock(t.hour, t.minute)
    Image.fromarray(img).save("P5a.png")

    # Generate all minutes in six sizes into project img/clock subfolders
    print(f"Output directory: {DEFAULT_CLOCK_OUTPUT_DIR}")
    start = time.perf_counter()
    if not generate_all_clocks():
        print(f"Clock folder already contains {EXPECTED_CLOCK_FILE_COUNT} files; skipping generation.")
    else:
        elapsed = time.perf_counter() - start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"Generated all clocks in {minutes} min {seconds:.1f} s")

