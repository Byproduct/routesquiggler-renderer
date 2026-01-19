# Route Squiggler Render Client - Systemd Service Setup

This directory contains scripts and configuration for running the Route Squiggler Render Client as a systemd service on Linux Mint (and other systemd-based Linux distributions).

## Files

- **`route-squiggler.service`** - Systemd service file (template with instructions)
- **`install-service.sh`** - Automated installation script
- **`uninstall-service.sh`** - Automated uninstallation script

## Quick Start

### Automated Installation (Recommended)

The easiest way to install the service:

```bash
# Make the script executable
chmod +x install-service.sh

# Run the installation script
sudo bash install-service.sh
```

The script will:
- Detect your username and installation directory
- Find your Python installation (system or venv)
- Check for required dependencies
- Create and install the systemd service
- Optionally enable and start the service

### Manual Installation

If you prefer to install manually:

1. Edit `route-squiggler.service`:
   - Replace `/path/to/route-squiggler` with your actual installation path
   - Replace `YOUR_USERNAME_HERE` with your Linux username
   - Update the Python path if using a virtual environment

2. Copy the service file:
   ```bash
   sudo cp route-squiggler.service /etc/systemd/system/
   ```

3. Reload systemd:
   ```bash
   sudo systemctl daemon-reload
   ```

4. Enable and start the service:
   ```bash
   sudo systemctl enable route-squiggler.service
   sudo systemctl start route-squiggler.service
   ```

## Managing the Service

### Check Status
```bash
sudo systemctl status route-squiggler.service
```

### Start Service
```bash
sudo systemctl start route-squiggler.service
```

### Stop Service
```bash
sudo systemctl stop route-squiggler.service
```

### Restart Service
```bash
sudo systemctl restart route-squiggler.service
```

### Enable Autostart on Boot
```bash
sudo systemctl enable route-squiggler.service
```

### Disable Autostart on Boot
```bash
sudo systemctl disable route-squiggler.service
```

## Viewing Logs

The service writes output to two places:

### 1. output.log file (in the application directory)
```bash
tail -f output.log
```

### 2. Systemd journal
```bash
# View recent logs
sudo journalctl -u route-squiggler.service

# Follow logs in real-time
sudo journalctl -u route-squiggler.service -f

# View logs since last boot
sudo journalctl -u route-squiggler.service -b

# View logs from last hour
sudo journalctl -u route-squiggler.service --since "1 hour ago"
```

## Configuration

The service is configured to:
- Run in terminal (nogui) mode
- Use the application's working directory for all relative paths
- Restart automatically on failure (after 10 seconds)
- Wait up to 5 minutes for graceful shutdown (to finish current job)
- Write all output to `output.log` in the application directory

### Environment Variables

If you need to add environment variables, edit the service file:

```ini
[Service]
Environment="VARIABLE_NAME=value"
Environment="ANOTHER_VAR=value"
```

### Python Virtual Environment

If using a Python virtual environment, the install script will automatically detect it. For manual configuration, edit the `ExecStart` line in the service file:

```ini
ExecStart=/path/to/route-squiggler/venv/bin/python main.py nogui
```

## Troubleshooting

### Service won't start

1. Check the status for error messages:
   ```bash
   sudo systemctl status route-squiggler.service
   ```

2. Check the journal for detailed logs:
   ```bash
   sudo journalctl -u route-squiggler.service -n 50
   ```

3. Verify Python dependencies are installed:
   ```bash
   python3 -c "import PySide6; print('OK')"
   ```

4. Test the application manually:
   ```bash
   cd /path/to/route-squiggler
   python3 main.py nogui
   ```

### Permission errors

Make sure the user specified in the service file has:
- Read/write access to the application directory
- Read/write access to output.log
- Access to required system resources

### Service stops unexpectedly

Check the logs to see why:
```bash
sudo journalctl -u route-squiggler.service | tail -100
```

Common issues:
- Missing dependencies
- Invalid configuration in config.txt
- Network connectivity problems
- Insufficient disk space

## Uninstalling

### Automated Uninstallation (Recommended)

```bash
# Make the script executable (if not already)
chmod +x uninstall-service.sh

# Run the uninstallation script
sudo bash uninstall-service.sh
```

### Manual Uninstallation

```bash
# Stop the service
sudo systemctl stop route-squiggler.service

# Disable the service
sudo systemctl disable route-squiggler.service

# Remove the service file
sudo rm /etc/systemd/system/route-squiggler.service

# Reload systemd
sudo systemctl daemon-reload
```

## Notes

- The service runs in **nogui** (terminal-only) mode
- All relative paths in the application are preserved (working directory is set correctly)
- The service will automatically request and process jobs from the server
- You cannot see the output directly when running as a service, but you can:
  - Tail the `output.log` file
  - Use `journalctl` to view systemd logs
- The service uses graceful shutdown (SIGINT) and will finish the current job before stopping
- If the service crashes, it will automatically restart after 10 seconds

## Security Considerations

The service runs as a regular user (not root) for security. If you need additional security hardening, you can uncomment these lines in the service file:

```ini
NoNewPrivileges=true
PrivateTmp=true
```

For production deployments, consider:
- Running as a dedicated service user with minimal permissions
- Using systemd's sandboxing features
- Implementing log rotation for output.log
- Setting up monitoring and alerting

## Additional Resources

- [systemd service documentation](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [systemd unit file examples](https://www.freedesktop.org/software/systemd/man/systemd.unit.html)
- [journalctl documentation](https://www.freedesktop.org/software/systemd/man/journalctl.html)

