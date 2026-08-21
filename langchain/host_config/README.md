# Host Setup Instructions

## 1. Docker daemon.json

You must install the provided `daemon.json` file on your host system. This file configures Docker with the necessary settings for Kubernetes (k3s) compatibility, such as cgroup driver and networking options.

**Installation:**
1. Copy `daemon.json` to `/etc/docker/daemon.json` on your host.
2. Restart the Docker service:
   ```sh
   sudo systemctl restart docker
   ```

## 2. k8s_cluster.py

`k8s_cluster.py` is a management script for a local k3s (Kubernetes) cluster. It provides commands to install, start, stop, and uninstall k3s, handling all required host and network setup steps.

### Features
- Installs k3s with Docker as the container runtime
- Ensures correct MTU settings for overlay networking
- Stops and cleans up k3s and related containers
- Handles uninstall and cleanup

### Usage
Run with Python 3.10+:

```sh
python3 k8s_cluster.py --start   # Install and start k3s
python3 k8s_cluster.py --stop    # Uninstall and stop k3s
```

You must specify exactly one of `--start` or `--stop`.

### Notes
- The script will automatically download the k3s installer if not present.
- It manages MTU settings to avoid network fragmentation issues with overlay networks.
- Requires root privileges for some operations (e.g., changing MTU, starting services).

---
For more details, see comments and docstrings in `k8s_cluster.py`.
