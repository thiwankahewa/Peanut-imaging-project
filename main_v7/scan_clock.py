"""A local scan clock that can be corrected without changing the OS clock."""
from datetime import datetime
import subprocess
import time


def system_time_synchronized():
    """Check actual NTP synchronization; a Wi-Fi connection alone is insufficient."""
    try:
        result = subprocess.run(
            ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
            check=True, capture_output=True, text=True, timeout=4,
        )
        return result.stdout.strip() == "yes" and datetime.now().year >= 2024
    except (OSError, subprocess.SubprocessError):
        return False


class ScanClock:
    def __init__(self):
        self.manual_epoch = None
        self.manual_started = None

    def set_manual(self, value):
        if len(value) != 12 or not value.isascii() or not value.isdigit():
            raise ValueError("Enter 12 digits: YYYYMMDDHHMM (24-hour time).")
        try:
            entered = datetime.strptime(value, "%Y%m%d%H%M")
        except ValueError as exc:
            raise ValueError("Enter a valid date and time: YYYYMMDDHHMM.") from exc
        if not 2024 <= entered.year <= 2099:
            raise ValueError("Enter a year between 2024 and 2099.")
        self.manual_epoch = entered.timestamp()
        self.manual_started = time.monotonic()

    def use_system_time(self):
        self.manual_epoch = self.manual_started = None

    def now(self):
        if self.manual_epoch is None:
            return datetime.now()
        return datetime.fromtimestamp(self.manual_epoch + time.monotonic() - self.manual_started)
