#!/bin/bash
# Uninstallation script for Route Squiggler Render Client systemd service
#
# This script removes the Route Squiggler systemd service.
# Run it with: sudo bash service-uninstall.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Route Squiggler Service Uninstaller${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}ERROR: Please run this script with sudo${NC}"
    echo "Usage: sudo bash service-uninstall.sh"
    exit 1
fi

SERVICE_FILE="/etc/systemd/system/route-squiggler.service"

# Check if service exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${YELLOW}Service file not found at $SERVICE_FILE${NC}"
    echo "The service may already be uninstalled."
    exit 0
fi

echo "This will:"
echo "  1. Stop the Route Squiggler service"
echo "  2. Disable it from starting on boot"
echo "  3. Remove the service file"
echo ""
read -p "Are you sure you want to uninstall? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 0
fi

echo ""

# Stop the service
echo "Stopping service"
if systemctl is-active --quiet route-squiggler.service; then
    systemctl stop route-squiggler.service
    echo -e "${GREEN}✓${NC} Service stopped"
else
    echo -e "${YELLOW}!${NC} Service was not running"
fi

# Disable the service
echo "Disabling service"
if systemctl is-enabled --quiet route-squiggler.service 2>/dev/null; then
    systemctl disable route-squiggler.service
    echo -e "${GREEN}✓${NC} Service disabled"
else
    echo -e "${YELLOW}!${NC} Service was not enabled"
fi

# Remove the service file
echo "Removing service file"
rm "$SERVICE_FILE"
echo -e "${GREEN}✓${NC} Service file removed"

# Reload systemd
echo "Reloading systemd daemon"
systemctl daemon-reload
systemctl reset-failed 2>/dev/null || true
echo -e "${GREEN}✓${NC} Systemd daemon reloaded"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Uninstallation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "The Route Squiggler service has been removed."
echo ""
echo -e "${YELLOW}Note:${NC} Your application files and logs have NOT been deleted."
echo "The service can be reinstalled at any time by running:"
echo "  sudo bash install-service.sh"
echo ""

