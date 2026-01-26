#!/bin/bash
# AI-generated update script for Route Squiggler Render Client
#
# This script pulls the latest code from git and restarts the systemd service.
# Run it with: bash update.sh
#
# Note: Restarting the service requires sudo privileges, so you may be prompted
# for your password when restarting the service.

set -e  # Exit on any error (but we'll handle git pull errors gracefully)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Route Squiggler Update Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}ERROR: Not a git repository!${NC}"
    echo "This script must be run from the Route Squiggler directory."
    exit 1
fi

# Check current git status
echo -e "${BLUE}Checking git status${NC}"
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}WARNING: You have uncommitted changes!${NC}"
    echo "The following files have been modified:"
    git status --short
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Update cancelled."
        exit 0
    fi
    echo ""
fi

# Show current commit
CURRENT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo -e "Current branch: ${BLUE}$CURRENT_BRANCH${NC}"
echo -e "Current commit: ${BLUE}$CURRENT_COMMIT${NC}"
echo ""

# Perform git pull
echo -e "${BLUE}Pulling latest changes from git${NC}"
set +e  # Temporarily disable exit on error to handle git pull failures
GIT_OUTPUT=$(git pull 2>&1)
GIT_EXIT_CODE=$?
set -e  # Re-enable exit on error

if [ $GIT_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Git pull failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Error output:"
    echo "$GIT_OUTPUT"
    echo ""
    echo -e "${YELLOW}The service was NOT restarted.${NC}"
    echo "Please resolve the git pull errors and try again."
    exit 1
fi

# Check if there were any actual updates
NEW_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_COMMIT" = "$NEW_COMMIT" ]; then
    echo -e "${GREEN}✓${NC} Already up to date (no changes pulled)"
    echo ""
    echo -e "${YELLOW}No updates were pulled.${NC}"
    read -p "Restart service anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Service restart cancelled."
        exit 0
    fi
else
    echo -e "${GREEN}✓${NC} Successfully pulled updates"
    echo -e "  Old commit: ${BLUE}$CURRENT_COMMIT${NC}"
    echo -e "  New commit: ${BLUE}$NEW_COMMIT${NC}"
    echo ""
    
    # Show what changed
    echo -e "${BLUE}Recent changes:${NC}"
    git log --oneline "$CURRENT_COMMIT".."$NEW_COMMIT" 2>/dev/null || echo "  (Unable to show changes)"
    echo ""
fi

# Check if service is installed
if [ ! -f "/etc/systemd/system/route-squiggler.service" ]; then
    echo -e "${YELLOW}WARNING: Systemd service not found!${NC}"
    echo "The service may not be installed, or it may be installed with a different name."
    echo ""
    read -p "Continue with service restart attempt anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Service restart cancelled."
        exit 0
    fi
fi

# Restart the service
echo -e "${BLUE}Restarting route-squiggler.service${NC}"
if sudo systemctl restart route-squiggler.service; then
    echo -e "${GREEN}✓${NC} Service restarted successfully"
    echo ""
    
    # Wait a moment and check status
    sleep 2
    echo -e "${BLUE}Service status:${NC}"
    sudo systemctl status route-squiggler.service --no-pager -l || true
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Failed to restart service!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "The git pull succeeded, but restarting the service failed."
    echo "You may need to restart it manually:"
    echo "  sudo systemctl restart route-squiggler.service"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Update Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "The service has been restarted with the latest code."
echo ""
echo "To view logs:"
echo "  sudo journalctl -u route-squiggler.service -f"
echo "  tail -f output.log"
echo ""

