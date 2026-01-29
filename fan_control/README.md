# Fan Control Service Setup and Files

This directory contains files to set up a custom systemd service for controlling CPU and GPU fans on systems with an NCT6775/NCT6799 Super I/O chip.

## File Descriptions

- **my_fan_control_service.py**: Python script that runs as a service, reads CPU and GPU temperatures, and sets fan speeds accordingly via the hardware monitoring (hwmon) sysfs interface. It supports both CPU and GPU temperature monitoring (GPU is optional, requires `pynvml`). The script ensures safe operation, restores automatic fan control on exit, and logs status for diagnostics.

- **fan-control.service**: A systemd service unit file that runs the Python script as a background service. It ensures the service starts after kernel modules are loaded, restarts automatically on failure, and applies security hardening options. It grants the necessary permissions to write to `/sys/class/hwmon` for fan control.

- **fan_notify_watcher.py**: A Python script that runs as a user service to monitor the fan-control service journal logs for notification markers and send desktop notifications. This allows the system service to communicate important events to the desktop user.

- **fan-notify.service**: A systemd user service that runs the notification watcher script in the user session, providing access to the desktop notification system.

- **setup_fancontrol.py**: Idempotent setup script to install both the system service (fan control) and user service (notifications) to the correct system locations, reload the systemd daemon, and enable/start both services. Must be run as root (with sudo).

## Required Kernel Module

The `nct6775` kernel module must be loaded for fan control to work. To ensure it loads at boot, create the following file:

    /etc/modules-load.d/nct6775.conf

with the contents:

    nct6775

This ensures the module is available before the service starts.

## Setup Steps

1. Ensure the `nct6775` module is loaded at boot as described above.
2. Run `setup_fancontrol.py` as root to install and enable both services:

	sudo ./setup_fancontrol.py

3. Check the service status:

	systemctl status fan-control.service
	systemctl --user status fan-notify.service

4. View logs for troubleshooting:

	journalctl -u fan-control.service -f
	journalctl --user -u fan-notify.service -f

## Notes

- The fan control service will fail to start if the hardware monitor device is not found or if the required sysfs paths are missing.
- The Python script is designed to be robust and will restore automatic fan control on shutdown or error.
- Desktop notifications are sent via the user service which monitors the system service journal logs for events marked with "NOTIFY:".
- GPU temperature monitoring is optional and requires the `pynvml` package. If not available, only CPU temperatures will be used for fan control.