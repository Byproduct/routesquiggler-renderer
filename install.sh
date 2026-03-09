# Installs venv and requirements.txt and create a systemd service for this renderer. Program will automatically start on reboot afterwards.

#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="route-squiggler-renderer"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
PYTHON_BIN="python3"

echo "=== Route Squiggler Renderer - Install ==="
echo "Install directory: ${SCRIPT_DIR}"

# Require root for service installation
if [[ $EUID -ne 0 ]]; then
    echo "Error: This script must be run as root (sudo) to install the systemd service."
    exit 1
fi

# Detect the user who invoked sudo (avoid running venv as root)
REAL_USER="${SUDO_USER:-$(whoami)}"
REAL_GROUP="$(id -gn "$REAL_USER")"

# Check Python 3 is available
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    echo "Error: ${PYTHON_BIN} not found. Please install Python 3.9+ first."
    exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python ${PY_VERSION}"

# ── 1. Create / update virtual environment ──────────────────────────────────

if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at ${VENV_DIR}, updating..."
else
    echo "Creating virtual environment..."
    sudo -u "$REAL_USER" "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Installing / upgrading pip..."
sudo -u "$REAL_USER" "${VENV_DIR}/bin/pip" install --upgrade pip

echo "Installing requirements..."
sudo -u "$REAL_USER" "${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"

echo "Requirements installed."

# ── 2. Create / update systemd service ──────────────────────────────────────

echo "Writing systemd service to ${SERVICE_FILE}..."

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Route Squiggler Renderer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${REAL_USER}
Group=${REAL_GROUP}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${VENV_DIR}/bin/python main.py nogui
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Generous timeout for bootup (map tile sync can be slow)
TimeoutStartSec=300

# Environment
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# ── 3. Reload, enable, and (re)start ────────────────────────────────────────

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling ${SERVICE_NAME} to start on boot..."
systemctl enable "$SERVICE_NAME"

echo "Restarting ${SERVICE_NAME}..."
systemctl restart "$SERVICE_NAME"

# Brief pause to let the service start
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo ""
    echo "=== ${SERVICE_NAME} is running ==="
else
    echo ""
    echo "Warning: service did not start cleanly. Check logs with:"
    echo "  journalctl -u ${SERVICE_NAME} -n 50 --no-pager"
fi

echo ""
echo "Useful commands:"
echo "  sudo systemctl status  ${SERVICE_NAME}   # Check status"
echo "  sudo systemctl stop    ${SERVICE_NAME}   # Stop"
echo "  sudo systemctl start   ${SERVICE_NAME}   # Start"
echo "  sudo systemctl restart ${SERVICE_NAME}   # Restart"
echo "  journalctl -u ${SERVICE_NAME} -f         # Follow logs"
echo ""
echo "Install complete."
