#!/usr/bin/env bash
set -e

echo "Setting up Btrfs compressed tile cache"

# Project root (parent of utils/ where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$BASE_DIR/map tile cache"
TEMP_DIR="$BASE_DIR/map tile cache temp"
IMAGE_FILE="$BASE_DIR/map_tile_cache_btrfs.img"

SIZE="500G"

echo "Base directory: $BASE_DIR"

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: 'map tile cache' directory not found."
    exit 1
fi

echo "Renaming existing cache"
mv "$CACHE_DIR" "$TEMP_DIR"

echo "Creating new cache directory"
mkdir "$CACHE_DIR"

echo "Creating sparse Btrfs image ($SIZE)"
# Use truncate (not fallocate): creates a sparse file that only uses disk as data is written
truncate -s $SIZE "$IMAGE_FILE"

echo "Formatting Btrfs filesystem"
mkfs.btrfs -f "$IMAGE_FILE"

echo "Mounting Btrfs filesystem with zstd:5 compression"

sudo mount -o loop,compress=zstd:6,noatime "$IMAGE_FILE" "$CACHE_DIR"

echo "Moving existing tiles into compressed filesystem"
mv "$TEMP_DIR"/* "$CACHE_DIR"/

echo "Removing temporary directory"
rmdir "$TEMP_DIR"

echo ""
echo "Done!"
echo ""
echo "Tile cache is now on a Btrfs filesystem with zstd:6 compression."
echo "Running deduplication is recommended."
echo ""
echo "Mount info:"
mount | grep "$CACHE_DIR"