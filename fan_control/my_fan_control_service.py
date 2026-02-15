#!/usr/bin/env python3

import sys
import time
import os
import logging
import signal
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from types import FrameType

# Try to import pynvml for GPU support (optional dependency)
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetTemperature,
        NVML_TEMPERATURE_GPU,
        nvmlShutdown,
    )

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

HYSTERESIS_PERCENT = 5.0  # Fan speed must change by this % to trigger update
TEMP_SMOOTHING = 0.5  # EMA smoothing factor (0-1): lower = more smoothing

# PWM channel configuration: {channel_num: {'min': min_speed, 'max': max_speed}}
PWM_CONFIG = {
    1: {"min": 50, "max": 180},
    2: {"min": 30, "max": 180},
    4: {"min": 30, "max": 180},
    5: {"min": 30, "max": 180},
}

# Global GPU handle to avoid repeated init/shutdown
gpu_handle = None
gpu_available = False


def init_gpu() -> None:
    """Initialize NVML for GPU temperature monitoring."""
    global gpu_handle, gpu_available

    if not PYNVML_AVAILABLE:
        logging.info("pynvml not available, GPU monitoring disabled")
        gpu_available = False
        return

    try:
        nvmlInit()
        gpu_handle = nvmlDeviceGetHandleByIndex(0)
        gpu_available = True
        logging.info("GPU initialized successfully")
    except Exception as e:
        logging.warning(f"GPU not available or failed to initialize: {e}")
        gpu_available = False


def get_gpu_temp(last_ema: Optional[float] = None) -> float:
    global gpu_handle, gpu_available

    if not gpu_available:
        # No GPU available, return minimum temperature to not affect fan speed
        return 0

    try:
        temp_c = nvmlDeviceGetTemperature(gpu_handle, NVML_TEMPERATURE_GPU)
        logging.debug(f"GPU temperature (raw): {temp_c}°C")
    except Exception as e:
        # Any failure → safe fallback
        logging.error(
            f"Failed to read GPU temperature: {e}. Using fallback temp of 90°C"
        )
        temp_c = 90

    # Apply exponential moving average
    temp_millis = temp_c * 1000
    if last_ema is not None:
        temp_millis = TEMP_SMOOTHING * temp_millis + (1 - TEMP_SMOOTHING) * last_ema
        logging.debug(f"GPU temperature (smoothed): {temp_millis/1000:.1f}°C")

    return temp_millis


def get_cpu_temp(cpu_temp_path: str, last_ema: Optional[float] = None) -> float:
    try:
        with open(cpu_temp_path, "r") as f:
            temp = int(f.read())
            logging.debug(f"CPU temperature (raw): {temp/1000:.1f}°C")

            # Apply exponential moving average
            if last_ema is not None:
                temp = TEMP_SMOOTHING * temp + (1 - TEMP_SMOOTHING) * last_ema
                logging.debug(f"CPU temperature (smoothed): {temp/1000:.1f}°C")

            return temp
    except Exception as e:
        # Safeguard return a temp of 90000 to make
        # the fans spin fast to highlight an issue
        logging.error(
            f"Failed to read CPU temperature from {cpu_temp_path}: {e}. Using fallback temp of 90°C"
        )
        return 90000


def determine_fan_speed_percent(
    cpu_temp_path: str,
    last_percent: Optional[float] = None,
    last_gpu_ema: Optional[float] = None,
    last_cpu_ema: Optional[float] = None,
) -> Tuple[float, float, float]:
    try:
        CPU_MIN = 50 * 1000
        CPU_MAX = 70 * 1000
        GPU_MIN = 55 * 1000
        GPU_MAX = 70 * 1000
        gpu_temp = get_gpu_temp(last_gpu_ema)
        cpu_temp = get_cpu_temp(cpu_temp_path, last_cpu_ema)

        # The interesting ranges are different between gpu and cpu
        # For CPU we want to start spinning at 50°C and at 70°C we reach max spin
        # For GPU we want to start spinning at 55°C and at 80°C we reach max spin

        # We will map the GPU temp range to the CPU temp range so that
        # 55°C -> 50°C and 80°C -> 70°C

        # Need to account for the case where gpu_temp is below GPU_MIN
        # max(gpu_temp, GPU_MIN)

        gpu_percent = (min(max(gpu_temp, GPU_MIN), GPU_MAX) - GPU_MIN) / (
            GPU_MAX - GPU_MIN
        )
        cpu_percent = (min(max(cpu_temp, CPU_MIN), CPU_MAX) - CPU_MIN) / (
            CPU_MAX - CPU_MIN
        )

        max_percent = max(gpu_percent, cpu_percent)

        # Apply hysteresis to prevent oscillation
        if last_percent is not None:
            percent_change = abs(max_percent - last_percent) * 100
            if percent_change < HYSTERESIS_PERCENT:
                # Change is too small, keep current speed
                calculated_percent = max_percent
                max_percent = last_percent
                logging.debug(
                    f"Fan speed calculation: GPU={gpu_percent*100:.1f}%, CPU={cpu_percent*100:.1f}%, Calculated={calculated_percent*100:.1f}% (suppressed by hysteresis, keeping {last_percent*100:.1f}%)"
                )
            else:
                logging.debug(
                    f"Fan speed calculation: GPU={gpu_percent*100:.1f}%, CPU={cpu_percent*100:.1f}%, Using={max_percent*100:.1f}% (changed from {last_percent*100:.1f}%)"
                )
        else:
            logging.debug(
                f"Fan speed calculation: GPU={gpu_percent*100:.1f}%, CPU={cpu_percent*100:.1f}%, Using={max_percent*100:.1f}%"
            )

        # If all temperatures are lower than their minimums, 0% will be returned (MIN_FAN_SPEED)
        # If they are within their ranges, the percentage will be calculated proportionally
        # If they exceed their maximums, 100% will be returned (MAX_FAN_SPEED)

        return max_percent, gpu_temp, cpu_temp
    except Exception as e:
        logging.error(f"Error in determine_fan_speed_percent: {e}")
        # Return safe defaults: max fan speed and high temps to keep fans running
        return 1.0, 90000, 90000


def send_notification(message: str, is_error: bool = False) -> None:
    """Send notification via journal log for user service to pick up."""
    # Log with NOTIFY marker for fan-notify-watcher to pick up and send desktop notification
    if is_error:
        logging.error(f"NOTIFY: {message}")
    else:
        logging.info(f"NOTIFY: {message}")


def cleanup_and_exit(
    main_path: Path, signum: Optional[int] = None, frame: Optional[FrameType] = None
) -> None:
    """Restore automatic fan control before exiting."""
    global gpu_handle, gpu_available

    if signum:
        logging.info(f"Received signal {signum}, shutting down gracefully...")
    else:
        logging.info("Shutting down gracefully...")

    # Restore automatic fan control
    for pwm_num in PWM_CONFIG.keys():
        pwm_enable = main_path / f"pwm{pwm_num}_enable"
        try:
            logging.info(f"Restoring automatic fan control for pwm{pwm_num}")
            with open(pwm_enable, "w") as f:
                f.write("5")  # 5 = automatic fan control
        except Exception as e:
            logging.error(
                f"Failed to restore automatic fan control for pwm{pwm_num}: {e}"
            )

    logging.info("Automatic fan control restoration complete")

    # Shutdown NVML if it was initialized
    if gpu_available and PYNVML_AVAILABLE:
        try:
            nvmlShutdown()
            logging.info("NVML shutdown complete")
        except Exception as e:
            logging.error(f"Failed to shutdown NVML: {e}")

    sys.exit(0)


def enable_manual_input(main_path: Path, initial_run=False) -> bool:
    """Enable manual fan control. Returns True on success, False on failure."""
    # Enable manual fan control but only if not already enabled
    for pwm_num in PWM_CONFIG.keys():
        pwm_enable = main_path / f"pwm{pwm_num}_enable"
        try:
            with open(pwm_enable, "r") as f:
                current_value = f.read().strip()
            if current_value != "1":
                logging.info(f"Enabling manual fan control for pwm{pwm_num}")
                with open(pwm_enable, "w") as f:
                    f.write("1")
            else:
                logging.debug(f"Manual fan control already enabled for pwm{pwm_num}")
        except Exception as e:
            logging.error(f"Failed to configure pwm{pwm_num}_enable: {e}")
            return False

    # Check if enabled if not return failure
    for pwm_num in PWM_CONFIG.keys():
        pwm_enable = main_path / f"pwm{pwm_num}_enable"
        try:
            with open(pwm_enable, "r") as f:
                current_value = f.read().strip()
            if current_value != "1":
                logging.critical(
                    f"Failed to enable manual fan control for pwm{pwm_num}"
                )
                return False
        except Exception as e:
            logging.critical(f"Failed to verify pwm{pwm_num}_enable: {e}")
            return False

    if initial_run:
        logging.info(
            f"Manual fan control enabled successfully for PWM channels: {list(PWM_CONFIG.keys())}"
        )
    else:
        logging.info(
            f"Manual fan control re-enabled successfully for PWM channels: {list(PWM_CONFIG.keys())}"
        )

    return True


def main() -> None:
    # Configure logging to output to stdout (captured by systemd)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )

    logging.info("Fan control service starting...")

    # Validate PWM_CONFIG is not empty
    if not PWM_CONFIG:
        logging.critical("PWM_CONFIG is empty. No fans to control.")
        sys.exit(1)

    # Initialize GPU once at startup
    init_gpu()

    # Determine main device path
    main_path = None
    hwmon_base = Path("/sys/class/hwmon")
    for hwmon in hwmon_base.iterdir():
        name_file = hwmon / "name"
        if name_file.exists():
            try:
                with open(name_file, "r") as f:
                    name = f.read().strip()
                    if name == "nct6799":
                        main_path = hwmon
                        logging.info(f"Found nct6799 device at {hwmon}")
                        break
            except Exception as e:
                logging.warning(f"Failed to read {name_file}: {e}")
                continue

    if main_path is None:
        logging.critical("Could not find nct6799 device in /sys/class/hwmon")
        sys.exit(1)

    cpu_temp_path = main_path / "temp13_input"

    # Verify CPU temperature path exists
    if not cpu_temp_path.exists():
        logging.critical(f"CPU temperature path does not exist: {cpu_temp_path}")
        sys.exit(1)

    # Register signal handlers early for graceful shutdown
    signal.signal(signal.SIGTERM, lambda s, f: cleanup_and_exit(main_path, s, f))
    signal.signal(signal.SIGINT, lambda s, f: cleanup_and_exit(main_path, s, f))
    logging.info("Signal handlers registered")

    if not enable_manual_input(main_path, initial_run=True):
        send_notification(
            "Failed to enable manual fan control at startup. Check logs.", is_error=True
        )
        cleanup_and_exit(main_path)

    logging.info("Entering fan control loop (updates every 5 seconds)")

    iteration = 0
    last_percent = None
    last_gpu_ema = None
    last_cpu_ema = None

    while True:
        percent, gpu_ema, cpu_ema = determine_fan_speed_percent(
            str(cpu_temp_path), last_percent, last_gpu_ema, last_cpu_ema
        )
        last_percent = percent
        last_gpu_ema = gpu_ema
        last_cpu_ema = cpu_ema

        # First attempt: try to write to all PWM channels
        failed_pwms = []
        for pwm_num in PWM_CONFIG.keys():
            pwm_config = PWM_CONFIG[pwm_num]
            fan_speed = int(
                pwm_config["min"] + (pwm_config["max"] - pwm_config["min"]) * percent
            )
            pwm = main_path / f"pwm{pwm_num}"
            try:
                with open(pwm, "w") as f:
                    f.write(str(fan_speed))
            except Exception:
                # Don't log error yet, will retry after re-enabling manual input
                failed_pwms.append(pwm_num)

        # If any writes failed, try to re-enable manual input and retry
        if failed_pwms:
            if not enable_manual_input(main_path):
                # Failed to re-enable manual input
                send_notification(
                    "Failed to re-enable manual fan control. Fan control service stopping.",
                    is_error=True,
                )
                cleanup_and_exit(main_path)

            # Retry writing to the failed PWM channels
            still_failed = []
            for pwm_num in failed_pwms:
                pwm_config = PWM_CONFIG[pwm_num]
                fan_speed = int(
                    pwm_config["min"]
                    + (pwm_config["max"] - pwm_config["min"]) * percent
                )
                pwm = main_path / f"pwm{pwm_num}"
                try:
                    with open(pwm, "w") as f:
                        f.write(str(fan_speed))
                except Exception as e:
                    # Now log the error since retry also failed
                    logging.error(
                        f"Failed to write to pwm{pwm_num} even after re-enabling manual input: {e}"
                    )
                    still_failed.append(pwm_num)

            # If any PWMs still failed after re-enabling and retrying, this is critical
            if still_failed:
                send_notification(
                    f"Failed to write to PWM channels {still_failed}. Fan control service stopping.",
                    is_error=True,
                )
                cleanup_and_exit(main_path)

        # Log status every 12 iterations (every minute)
        if iteration % 12 == 0:
            logging.info(f"Fan speeds set to {percent*100:.1f}%")
        else:
            logging.debug(f"Fan speeds set to {percent*100:.1f}%")

        iteration += 1
        time.sleep(5)  # Update every 5 seconds


if __name__ == "__main__":
    main()
