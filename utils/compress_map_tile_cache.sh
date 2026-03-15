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

# Check for required Btrfs tools (e.g. Raspberry Pi may not have them by default)
if ! command -v mkfs.btrfs &>/dev/null; then
    echo "ERROR: mkfs.btrfs not found. Install Btrfs tools first:"
    echo "  Debian / Ubuntu / Raspberry Pi OS:  sudo apt install btrfs-progs"
    echo "  Fedora:  sudo dnf install btrfs-progs"
    echo "  Arch:    sudo pacman -S btrfs-progs"
    exit 1
fi

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

# Add fstab entry for mount at boot (mount point with spaces as \040)
FSTAB_MOUNTPOINT="${BASE_DIR}/map\\040tile\\040cache"
FSTAB_OPTS="loop,compress=zstd:6,noatime,nofail"
if grep -qF "$IMAGE_FILE" /etc/fstab 2>/dev/null; then
    echo "fstab already contains an entry for this image; skipping."
else
    echo "Adding entry to /etc/fstab for mount at boot."
    printf '%s  %s  btrfs  %s  0  0\n' "$IMAGE_FILE" "$FSTAB_MOUNTPOINT" "$FSTAB_OPTS" | sudo tee -a /etc/fstab > /dev/null
    echo "fstab updated."
fi

echo ""
echo "Done!"
echo ""
echo "Tile cache is now on a Btrfs filesystem with zstd:6 compression."
echo "Running deduplication is recommended."
echo ""
echo "Mount info:"
mount | grep "$CACHE_DIR"