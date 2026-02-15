#!/usr/bin/env python3
# This script manages a local k3s Kubernetes cluster for development purposes. Python 3.12

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Sequence

import requests


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

K3S_BIN = Path("/usr/local/bin/k3s")
K3S_VERSION = "v1.35.0+k3s3"

Command = str | Sequence[str]


def format_command_output(
    cmd: Sequence[str],
    cwd: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    msg = [f"cwd={cwd}", f"cmd={' '.join(cmd)}", f"exit={returncode}"]

    if stdout:
        prefix = "stdout:\n" if "\n" in stdout else "stdout: "
        msg.append(prefix + stdout)
    if stderr:
        prefix = "stderr:\n" if "\n" in stderr else "stderr: "
        msg.append(prefix + stderr)
    if not stdout and not stderr:
        msg.append("[no output]")

    return "\n".join(msg)


def run_command(
    cmd: Command,
    cwd: str | Path | None = None,
    skip_errors: bool = False,
) -> str:
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    command_str = " ".join(cmd)
    cwd = str(cwd) if cwd else os.getcwd()

    cp = subprocess.run(cmd, capture_output=True, cwd=cwd)

    try:
        stdout = cp.stdout.decode().strip()
    except UnicodeDecodeError:
        stdout = cp.stdout.decode(errors="replace").strip()
        LOG.error(
            f"Failed to decode stdout for command: {command_str}; replaced invalid bytes.",
            exc_info=True,
        )

    try:
        stderr = cp.stderr.decode().strip()
    except UnicodeDecodeError:
        stderr = cp.stderr.decode(errors="replace").strip()
        LOG.error(
            f"Failed to decode stderr for command: {command_str}; replaced invalid bytes.",
            exc_info=True,
        )

    msg = format_command_output(cmd, cwd, cp.returncode, stdout, stderr)

    if cp.returncode != 0:
        if skip_errors:
            LOG.warning(f"Command failed but skip_errors=True; continuing.\n{msg}")
        else:
            # Provide a shell-ready command string in the error
            shell_cmd = (
                command_str
                if isinstance(cmd, str)
                else " ".join(shlex.quote(str(x)) for x in cmd)
            )
            error_msg = f"Shell command: {shell_cmd}\n{msg}"
            raise subprocess.CalledProcessError(
                cp.returncode, shell_cmd, output=error_msg
            )

    return stdout


def process_exists(name: str) -> bool:
    res = run_command("ps aux")
    return name in res


def kubernetes_bin_exists() -> bool:
    return K3S_BIN.is_file()


def get_running_containers(process_suffix: str = "") -> list[str]:
    try:
        output = run_command("docker ps --format '{{.Names}}'")
    except subprocess.CalledProcessError:
        return []

    containers = output.splitlines()
    if process_suffix:
        containers = [c for c in containers if process_suffix in c]
    return containers


def stop_container(process_suffix: str) -> None:
    containers = get_running_containers(process_suffix)
    if not containers:
        return

    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    for c in containers:
        LOG.info(f"Stopping container: {c}")
        proc = subprocess.Popen(
            ["docker", "stop", c], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        processes.append((c, proc))

    for c, proc in processes:
        stdout_b, stderr_b = proc.communicate()
        stdout = stdout_b.decode(errors="replace").strip()
        stderr = stderr_b.decode(errors="replace").strip()
        if proc.returncode != 0:
            msg = format_command_output(
                ["docker", "stop", c], os.getcwd(), proc.returncode, stdout, stderr
            )
            LOG.warning(f"Failed to stop container: {c}; continuing.\n{msg}")


def download_file_from_url(url: str, dest: Path) -> None:
    r = requests.get(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    LOG.info(f"Downloading {url} to {dest}")
    dest.write_bytes(r.content)


def get_default_gateway_ip() -> str | None:
    route_output = run_command("ip -4 route show default", skip_errors=True)
    if not route_output:
        return None

    for line in route_output.splitlines():
        match = re.search(r"\bvia\s+([0-9]+(?:\.[0-9]+){3})\b", line)
        if match:
            return match.group(1)

    return None


def get_network_interfaces_and_mtu() -> tuple[dict[str, int], int]:
    """
    Determines the MTU (Maximum Transmission Unit) for all physical network interfaces.

    Why MTU is set:
    Overlay networks (like Flannel or other CNI plugins) encapsulate packets, adding extra headers.
    This reduces the effective payload size. If the MTU is too high, packets may exceed the physical
    network’s MTU, causing fragmentation or dropped packets. Setting the MTU lower (to account for
    encapsulation overhead) prevents fragmentation, ensuring stable pod-to-pod and pod-to-service
    communication. This ensures that all network packets, including those with overlay headers, fit
    within the physical network’s limits, preventing connectivity issues and improving cluster reliability.
    """
    res = run_command("find /sys/class/net -type l -not -lname '*virtual*'")
    if res == "":
        raise Exception("Unable to find a physical network interface!")

    interfaces: dict[str, int] = {}
    minimal_mtu: int | None = None

    for interface_path in res.split("\n"):
        interface = Path(interface_path).name
        mtu_info = run_command(f"ip link show {interface}")
        match = re.search(r".*mtu ([0-9]+) .*", mtu_info)
        if not match:
            raise ValueError(f"Unable to parse MTU from: {mtu_info}")
        mtu = int(match.groups()[0])

        if minimal_mtu is None or mtu < minimal_mtu:
            minimal_mtu = mtu

        interfaces[interface] = mtu

    if len(interfaces) == 0:
        raise Exception("Unable to find a physical interface with status UP!")

    if minimal_mtu is None:
        raise Exception("Unable to determine MTU for any interface!")

    return interfaces, minimal_mtu


def restore_mtu(interfaces: dict[str, int]) -> None:
    for interface in interfaces:
        try:
            run_command(
                f"sudo ip link set dev {interface} mtu {interfaces[interface]}",
            )
        except subprocess.CalledProcessError:
            LOG.warning(f"Failed to restore MTU for interface: {interface}")


def stop_kubernetes() -> None:
    if process_exists("k3s server"):
        LOG.info("Stopping k3s service")
        run_command("sudo service k3s stop")
    else:
        LOG.info("k3s service not running; checking for leftover k8s containers.")

    running_containers = get_running_containers(process_suffix="k8s_")
    if len(running_containers) > 0:
        LOG.info(f"Found {len(running_containers)} running k8s containers; stopping.")
        stop_container("k8s_")


def uninstall_kubernetes() -> None:
    uninstall_script = Path("/usr/local/bin/k3s-uninstall.sh")

    if not uninstall_script.is_file():
        LOG.info(f"k3s uninstall script not found at {uninstall_script}; skipping.")
        return

    LOG.info("Stopping k3s if running")
    stop_kubernetes()

    LOG.info("Running k3s uninstall script")
    run_command(f"sh {uninstall_script}")

    # stuff like PVCs are stored here which are not uninstalled with the uninstall script
    storage_path = Path("/var/lib/rancher/k3s/storage")
    if storage_path.is_dir():
        LOG.info(f"Removing k3s storage directory: {storage_path}")
        run_command(f"sudo rm -rf {storage_path}")

    LOG.info("k3s uninstall completed")


def install_kubernetes() -> None:
    if process_exists("k3s server"):
        uninstall_kubernetes()
    elif kubernetes_bin_exists():
        uninstall_kubernetes()
    else:
        running_containers = get_running_containers(process_suffix="k8s_")
        if len(running_containers) > 0:
            LOG.info(
                f"Found {len(running_containers)} leftover k8s containers; stopping."
            )
            stop_container("k8s_")

    k3s_install_script = Path(__file__).parent / "install_k3s.sh"

    if not k3s_install_script.is_file():
        download_file_from_url("https://get.k3s.io", k3s_install_script)

    resolvconf_path = Path("/tmp/k3s_resolv.conf")
    default_gateway_ip = get_default_gateway_ip()
    if default_gateway_ip:
        resolvconf_path.write_text(f"nameserver {default_gateway_ip}\n")
        LOG.info(f"Using default gateway as DNS resolver for k3s: {default_gateway_ip}")
    else:
        resolvconf_path.write_text("nameserver 1.1.1.1\nnameserver 8.8.8.8\n")
        LOG.warning(
            "Could not detect default IPv4 gateway; using fallback DNS resolvers "
            "1.1.1.1 and 8.8.8.8"
        )

    LOG.info(f"Installing k3s using script: {k3s_install_script}")

    docker_info = json.loads(run_command(["docker", "info", "--format", "{{json .}}"]))
    cgroup_driver = docker_info["CgroupDriver"]

    # The cgroup driver must be set to match Docker's configuration.
    # Kubernetes (k3s) and Docker must use the same cgroup driver (systemd or cgroupfs)
    # or kubelet will fail to start pods. This ensures compatibility and prevents errors
    # like 'failed to create pod sandbox' due to mismatched cgroup drivers.
    run_command(
        f"env INSTALL_K3S_SKIP_ENABLE=true K3S_RESOLV_CONF={resolvconf_path} "
        f"INSTALL_K3S_VERSION={K3S_VERSION} "
        f"sh {k3s_install_script} "
        "--node-ip=172.17.0.1 "
        "--bind-address 172.17.0.1 "
        "--advertise-address 172.17.0.1 "
        "--write-kubeconfig-mode 666 --docker "
        f"--kubelet-arg cgroup-driver={cgroup_driver} "
        "--disable=traefik"
    )

    rancher_folder = Path("/etc/rancher")
    if not rancher_folder.is_dir():
        run_command(f"sudo mkdir -p {rancher_folder}")

    control_file = rancher_folder / "vision_control"
    if not control_file.is_file():
        run_command(f"sudo touch {control_file}")

    interfaces, minimal_mtu = get_network_interfaces_and_mtu()
    new_mtu = minimal_mtu - 100

    try:
        for interface in interfaces:
            run_command(f"sudo ip link set dev {interface} mtu {new_mtu}")
    except subprocess.CalledProcessError:
        restore_mtu(interfaces)
        raise

    try:
        run_command("sudo service k3s start")

        flannel_subnet_file = Path("/run/flannel/subnet.env")
        while not flannel_subnet_file.is_file():
            LOG.info(f"Waiting for flannel subnet file: {flannel_subnet_file}")
            time.sleep(1)
    finally:
        restore_mtu(interfaces)

    if subprocess.run(["which", "firewall-cmd"], capture_output=True).returncode == 0:
        # Note: This is a workaround for a common issue where the Kubernetes CNI
        # bridge (cni0) is not in the same firewalld zone as the Docker bridge (docker0).
        #
        # Ensure the Kubernetes CNI bridge (cni0) is added to the same firewalld zone
        # as the Docker bridge (docker0). This allows network traffic rules for Docker
        # containers to also apply to Kubernetes pods, enabling correct communication.
        #
        # If another firewall management tool is used, the user may need to manually
        # add cni0 to the appropriate zone.

        docker_zone = run_command("firewall-cmd --get-zone-of-interface docker0")
        run_command(f"sudo firewall-cmd --zone {docker_zone} --add-interface=cni0")

    subnet_text = flannel_subnet_file.read_text()
    match = re.search(r".*FLANNEL_MTU=([0-9]+).*", subnet_text)
    if not match:
        raise ValueError("Unable to parse FLANNEL_MTU from subnet.env")
    flannel_mtu = match.groups()[0]
    assert int(flannel_mtu) == new_mtu - 50

    LOG.info("k3s install completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage local k3s cluster")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--stop", action="store_true")

    args = parser.parse_args()

    if args.start == args.stop:
        parser.error("You must specify exactly one of --start or --stop")

    previous_dir = os.getcwd()
    os.chdir(Path(__file__).parent)
    try:
        if args.start:
            install_kubernetes()
        if args.stop:
            uninstall_kubernetes()
    finally:
        os.chdir(previous_dir)


if __name__ == "__main__":
    main()
