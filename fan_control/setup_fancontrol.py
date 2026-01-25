#!/usr/bin/env python3
"""
setup_fancontrol.py
Idempotent setup for custom fan control service (CPU + GPU)
"""

import os
import shutil
import subprocess
from pathlib import Path
import sys

# Paths / constants
PYTHON_SCRIPT_NAME = "my_fan_control_service.py"
SERVICE_FILE_NAME = "fan-control.service"
PYTHON_SCRIPT_DEST = Path("/usr/sbin/my_fan_control_service.py")
SERVICE_FILE_DEST = Path("/etc/systemd/system/fan-control.service")
SERVICE_NAME = "fan-control.service"


def run(cmd):
    """Run a command and fail hard on error."""
    subprocess.run(cmd, check=True)


def ensure_root():
    if os.geteuid() != 0:
        print("This script must be run as root (use sudo).", file=sys.stderr)
        sys.exit(1)


def main():
    ensure_root()

    print("### Step 1: Installing fan control Python script ###")
    src_script = Path.cwd() / PYTHON_SCRIPT_NAME
    if not src_script.exists():
        print(f"ERROR: Expected {PYTHON_SCRIPT_NAME} in current directory", file=sys.stderr)
        sys.exit(1)

    shutil.copy2(src_script, PYTHON_SCRIPT_DEST)
    PYTHON_SCRIPT_DEST.chmod(0o755)
    print(f"Fan control script installed to {PYTHON_SCRIPT_DEST}")

    print("### Step 2: Installing systemd service file ###")
    src_service = Path.cwd() / SERVICE_FILE_NAME
    if not src_service.exists():
        print(f"ERROR: Expected {SERVICE_FILE_NAME} in current directory", file=sys.stderr)
        sys.exit(1)

    shutil.copy2(src_service, SERVICE_FILE_DEST)
    SERVICE_FILE_DEST.chmod(0o644)
    print(f"Service file installed to {SERVICE_FILE_DEST}")

    print("### Step 3: Reloading systemd daemon ###")
    run(["systemctl", "daemon-reload"])

    print("### Step 4: Enabling and starting fan control service ###")
    run(["systemctl", "enable", "--now", SERVICE_NAME])

    print("\nSetup complete! Fan control service is now running.")
    print(f"Check status with: systemctl status {SERVICE_NAME}")
    print(f"View logs with: journalctl -u {SERVICE_NAME} -f")


if __name__ == "__main__":
    main()