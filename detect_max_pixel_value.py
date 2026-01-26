#!/usr/bin/env python3
"""
Standalone script to analyze a video file and display the maximum pixel value
for each frame. This is useful for debugging black frame detection and understanding
the pixel value distribution across video frames.

Usage:
    python detect_max_pixel_value.py input.mp4

Output:
    Prints frame number and maximum pixel value for each frame, one per line,
    in numerical frame order.
"""

import sys
import cv2
import numpy as np


def analyze_video_max_pixel_values(video_path):
    """
    Analyze a video file and return maximum pixel value for each frame.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        list: List of tuples (frame_number, max_pixel_value)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}", file=sys.stderr)
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:", file=sys.stderr)
    print(f"  Total frames: {total_frames}", file=sys.stderr)
    print(f"  FPS: {fps:.2f}", file=sys.stderr)
    print(f"  Resolution: {width}x{height}", file=sys.stderr)
    print(f"  Duration: {total_frames/fps:.2f} seconds", file=sys.stderr)
    print(f"\nFrame analysis (frame_number: max_pixel_value):", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    
    results = []
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate maximum pixel value across all channels and all pixels
        max_pixel_value = np.max(frame)
        results.append((frame_number, max_pixel_value))
        
        # Print to stdout (one per line)
        print(f"{frame_number}: {max_pixel_value}")
        
        frame_number += 1
    
    cap.release()
    
    print(f"\nAnalysis complete: {len(results)} frames analyzed", file=sys.stderr)
    
    return results


def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <video_file>", file=sys.stderr)
        print(f"Example: {sys.argv[0]} input.mp4", file=sys.stderr)
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Analyze the video
    results = analyze_video_max_pixel_values(video_path)
    
    if results is None:
        sys.exit(1)
    
    # Summary statistics
    max_values = [max_val for _, max_val in results]
    if max_values:
        print(f"\nSummary statistics:", file=sys.stderr)
        print(f"  Minimum max pixel value: {min(max_values)}", file=sys.stderr)
        print(f"  Maximum max pixel value: {max(max_values)}", file=sys.stderr)
        print(f"  Average max pixel value: {np.mean(max_values):.2f}", file=sys.stderr)
        print(f"  Median max pixel value: {np.median(max_values):.2f}", file=sys.stderr)
        
        # Count frames with max pixel value <= 5 (effectively black)
        black_frames = [frame_num for frame_num, max_val in results if max_val <= 5]
        print(f"  Frames with max pixel value <= 5 (effectively black): {len(black_frames)}", file=sys.stderr)
        if black_frames:
            print(f"    Black frame numbers: {black_frames[:20]}{'...' if len(black_frames) > 20 else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()

