#!/usr/bin/env python3
"""
fan_notify_watcher.py
Watches the fan-control service journal for NOTIFY: markers and sends desktop notifications.
Runs as user service with access to D-Bus session for desktop notifications.
Tracks last displayed notification to avoid duplicates on restart.
"""

import subprocess
import sys
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timezone


STATE_DIR = Path.home() / ".local" / "state" / "fan-notify"
STATE_DIR.mkdir(parents=True, exist_ok=True)
LAST_CURSOR_FILE = STATE_DIR / "last-cursor.txt"

# Service and journal configuration
FAN_SERVICE = "fan-control.service"
DEFAULT_BACKLOG_ENTRIES = 20


def load_last_cursor():
    """Load the last processed journal cursor."""
    try:
        if LAST_CURSOR_FILE.exists():
            return LAST_CURSOR_FILE.read_text().strip()
    except Exception as e:
        logging.warning(f"Could not read last cursor file: {e}")
    return None


def save_last_cursor(cursor):
    """Save the last processed journal cursor."""
    try:
        LAST_CURSOR_FILE.write_text(cursor)
    except Exception as e:
        logging.error(f"Could not save last cursor file: {e}")


def parse_journal_entry_for_notify(json_line):
    """Parse a journal entry JSON line for NOTIFY messages.

    Args:
        json_line: A JSON string representing a journal entry

    Returns:
        Tuple of (message, cursor, timestamp_us) if entry contains NOTIFY marker,
        None otherwise.
    """
    try:
        entry = json.loads(json_line)
        message_text = entry.get("MESSAGE", "")
        cursor = entry.get("__CURSOR", "")
        timestamp_us = entry.get("__REALTIME_TIMESTAMP", "")

        # Look for NOTIFY: prefix
        match = re.search(r"NOTIFY:\s*(.+)", message_text)
        if match and cursor and timestamp_us:
            return (match.group(1), cursor, int(timestamp_us))
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def build_journalctl_command(after_cursor=None, follow=False, num_entries=None):
    """Build a journalctl command for the fan-control service.

    Args:
        after_cursor: Start after this cursor position
        follow: Whether to follow the journal (tail -f style)
        num_entries: Number of recent entries to show (ignored if after_cursor is set)

    Returns:
        List of command arguments for subprocess
    """
    cmd = [
        "journalctl",
        "--system",
        "-u",
        FAN_SERVICE,
        "-o",
        "json",
    ]

    if follow:
        cmd.append("-f")

    if after_cursor:
        cmd.extend(["--after-cursor", after_cursor])
    elif num_entries is not None:
        cmd.extend(["-n", str(num_entries)])

    return cmd


def get_notify_messages_from_journal(after_cursor=None):
    """Get NOTIFY messages from journal, optionally after a specific cursor.

    Returns list of tuples: [(message, cursor, timestamp_us), ...]
    """
    try:
        cmd = build_journalctl_command(
            after_cursor=after_cursor,
            num_entries=DEFAULT_BACKLOG_ENTRIES if not after_cursor else None,
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            logging.error(
                f"journalctl failed with return code {result.returncode}: {result.stderr}"
            )
            return []

        entries = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parsed = parse_journal_entry_for_notify(line)
            if parsed:
                entries.append(parsed)

        return entries
    except Exception as e:
        logging.error(f"Failed to query journal: {e}")
        return []


def format_time_ago(timestamp_us):
    """Format timestamp as 'Xs ago' or 'Xm ago'.

    Args:
        timestamp_us: Unix timestamp in microseconds

    Returns:
        Formatted string like '(5s ago)' or '(2m ago)'
    """
    # Convert microseconds to seconds
    event_time = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    delta_seconds = (now - event_time).total_seconds()

    if delta_seconds < 60:
        return f"({int(delta_seconds)}s ago)"
    else:
        minutes = int(delta_seconds / 60)
        return f"({minutes}m ago)"


def process_backlog():
    """Process any NOTIFY messages that were missed since last run.

    Returns:
        The last cursor processed, or None if no entries were processed.
    """
    last_cursor = load_last_cursor()

    if last_cursor:
        # Get all entries after our last cursor
        entries = get_notify_messages_from_journal(after_cursor=last_cursor)
    else:
        # No saved cursor, get recent entries
        entries = get_notify_messages_from_journal()
        # Only show the most recent one to avoid spam on first run
        if entries:
            entries = [entries[-1]]

    # Send notifications for all new entries
    last_processed_cursor = None
    for message, cursor, timestamp in entries:
        send_desktop_notification(message, cursor, timestamp)
        last_processed_cursor = cursor

    # If we processed entries, return the last cursor; otherwise return the saved one
    return last_processed_cursor if last_processed_cursor else last_cursor


def send_desktop_notification(message, cursor, timestamp):
    """Send a desktop notification via notify-send and save the cursor."""
    try:
        time_ago = format_time_ago(timestamp)
        formatted_message = f"{time_ago} {message}"

        subprocess.run(
            ["notify-send", "Fan Control", formatted_message],
            timeout=5,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        save_last_cursor(cursor)
        logging.debug(f"Notification sent: {formatted_message}")
    except Exception as e:
        logging.error(f"Failed to send notification: {e}")


def main():
    """Watch the fan-control service journal and send desktop notifications."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )

    logging.info("Fan notification watcher started")

    last_cursor = process_backlog()

    # Follow the fan-control service journal for new messages
    # Use --after-cursor to start exactly where backlog ended (eliminates race condition)
    cmd = build_journalctl_command(
        after_cursor=last_cursor,
        follow=True,
        num_entries=0 if not last_cursor else None,
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            parsed = parse_journal_entry_for_notify(line)
            if parsed:
                message, cursor, timestamp = parsed
                send_desktop_notification(message, cursor, timestamp)

    except KeyboardInterrupt:
        logging.info("Fan notification watcher stopped")
        process.terminate()
        sys.exit(0)


if __name__ == "__main__":
    main()
