"""
Cross-platform Hardware ID Module
Generates a unique hardware identifier for Windows and Linux systems.
"""

import hashlib
import platform
import os
import uuid

def get_hardware_id():
    """
    Get a unique hardware identifier for the current machine.

    On Windows: Combines Machine GUID and WMI System UUID.
    On Linux: Uses /etc/machine-id or fallback to DMI product UUID.

    Returns:
        str: SHA256 hash of the hardware ID components
    """
    system = platform.system()
    identifier = ""

    if system == 'Windows':
        try:
            import winreg
            import wmi

            # Get Machine GUID from Registry
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SOFTWARE\Microsoft\Cryptography")
            machine_guid, _ = winreg.QueryValueEx(key, "MachineGuid")
            winreg.CloseKey(key)

            # Get System UUID from WMI
            system_uuid = ""
            c = wmi.WMI()
            for system in c.Win32_ComputerSystemProduct():
                if system.UUID:
                    system_uuid = system.UUID.strip()
                    break

            identifier = f"{machine_guid}-{system_uuid}"
        except Exception:
            identifier = "windows-unknown"

    elif system == 'Linux':
        try:
            # Try /etc/machine-id (persistent and unique to the OS installation)
            if os.path.exists("/etc/machine-id"):
                with open("/etc/machine-id", "r") as f:
                    identifier = f.read().strip()
            # Fallback: Try DMI product UUID
            elif os.path.exists("/sys/class/dmi/id/product_uuid"):
                with open("/sys/class/dmi/id/product_uuid", "r") as f:
                    identifier = f.read().strip()
            else:
                # Last resort: Use MAC address (can change)
                identifier = str(uuid.getnode())
        except Exception:
            identifier = "linux-unknown"

    else:
        # Other OS types (macOS, BSD, etc.) — use MAC address
        identifier = str(uuid.getnode())

    # Return a SHA256 hash of the identifier for consistency
    return hashlib.sha256(identifier.encode('utf-8')).hexdigest()
