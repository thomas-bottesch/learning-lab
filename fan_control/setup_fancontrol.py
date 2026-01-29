#!/usr/bin/env python3
"""
setup_fancontrol.py
Idempotent setup for custom fan control service (CPU + GPU)
Sets up both system service (fan control) and user service (notifications)
"""

import os
import shutil
import subprocess
from pathlib import Path
import sys
import pwd

# Paths / constants
PYTHON_SCRIPT_NAME = "my_fan_control_service.py"
NOTIFY_SCRIPT_NAME = "fan_notify_watcher.py"
SYSTEM_SERVICE_NAME = "fan-control.service"
USER_SERVICE_NAME = "fan-notify.service"

PYTHON_SCRIPT_DEST = Path("/usr/sbin/my_fan_control_service.py")
NOTIFY_SCRIPT_DEST = Path("/usr/local/bin/fan_notify_watcher.py")
SYSTEM_SERVICE_DEST = Path("/etc/systemd/system/fan-control.service")
USER_SERVICE_DEST = Path("/etc/systemd/user/fan-notify.service")


def run(cmd):
    """Run a command and fail hard on error."""
    subprocess.run(cmd, check=True)


def ensure_root():
    if os.geteuid() != 0:
        print("This script must be run as root (use sudo).", file=sys.stderr)
        sys.exit(1)


def get_real_user():
    """Get the real user who ran sudo."""
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        return sudo_user
    return "tbo"  # Fallback


def main():
    ensure_root()
    real_user = get_real_user()

    print(f"### Installing services for user: {real_user} ###\n")

    print("### Step 1: Installing fan control Python script ###")
    src_script = Path.cwd() / PYTHON_SCRIPT_NAME
    if not src_script.exists():
        print(
            f"ERROR: Expected {PYTHON_SCRIPT_NAME} in current directory",
            file=sys.stderr,
        )
        sys.exit(1)

    shutil.copy2(src_script, PYTHON_SCRIPT_DEST)
    PYTHON_SCRIPT_DEST.chmod(0o755)
    print(f"Fan control script installed to {PYTHON_SCRIPT_DEST}")

    print("\n### Step 2: Installing notification watcher script ###")
    src_notify = Path.cwd() / NOTIFY_SCRIPT_NAME
    if not src_notify.exists():
        print(
            f"ERROR: Expected {NOTIFY_SCRIPT_NAME} in current directory",
            file=sys.stderr,
        )
        sys.exit(1)

    shutil.copy2(src_notify, NOTIFY_SCRIPT_DEST)
    NOTIFY_SCRIPT_DEST.chmod(0o755)
    print(f"Notification watcher installed to {NOTIFY_SCRIPT_DEST}")

    print("\n### Step 3: Installing system service file ###")
    src_system_service = Path.cwd() / SYSTEM_SERVICE_NAME
    if not src_system_service.exists():
        print(
            f"ERROR: Expected {SYSTEM_SERVICE_NAME} in current directory",
            file=sys.stderr,
        )
        sys.exit(1)

    shutil.copy2(src_system_service, SYSTEM_SERVICE_DEST)
    SYSTEM_SERVICE_DEST.chmod(0o644)
    print(f"System service file installed to {SYSTEM_SERVICE_DEST}")

    print("\n### Step 4: Installing user service file ###")
    src_user_service = Path.cwd() / USER_SERVICE_NAME
    if not src_user_service.exists():
        print(
            f"ERROR: Expected {USER_SERVICE_NAME} in current directory",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create user systemd directory if it doesn't exist
    USER_SERVICE_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_user_service, USER_SERVICE_DEST)
    USER_SERVICE_DEST.chmod(0o644)
    print(f"User service file installed to {USER_SERVICE_DEST}")

    print("\n### Step 5: Reloading systemd daemons ###")
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "--user", "--machine", f"{real_user}@.host", "daemon-reload"])

    print("\n### Step 6: Stopping services (if running) ###")
    subprocess.run(["systemctl", "stop", SYSTEM_SERVICE_NAME], check=False)
    subprocess.run(
        [
            "systemctl",
            "--user",
            "--machine",
            f"{real_user}@.host",
            "stop",
            USER_SERVICE_NAME,
        ],
        check=False,
    )

    print("\n### Step 7: Enabling and starting system service ###")
    run(["systemctl", "enable", SYSTEM_SERVICE_NAME])
    run(["systemctl", "start", SYSTEM_SERVICE_NAME])

    print("\n### Step 8: Enabling and starting user service ###")
    run(
        [
            "systemctl",
            "--user",
            "--machine",
            f"{real_user}@.host",
            "enable",
            USER_SERVICE_NAME,
        ]
    )
    run(
        [
            "systemctl",
            "--user",
            "--machine",
            f"{real_user}@.host",
            "start",
            USER_SERVICE_NAME,
        ]
    )

    print("\n" + "=" * 60)
    print("Setup complete! Both services are now running.")
    print("=" * 60)
    print(f"\nSystem service (fan control):")
    print(f"  Status: systemctl status {SYSTEM_SERVICE_NAME}")
    print(f"  Logs:   journalctl -u {SYSTEM_SERVICE_NAME} -f")
    print(f"\nUser service (notifications):")
    print(f"  Status: systemctl --user status {USER_SERVICE_NAME}")
    print(f"  Logs:   journalctl --user -u {USER_SERVICE_NAME} -f")
    print(f"\nDesktop notifications are sent via user service watching the journal.")
    print()


if __name__ == "__main__":
    main()
