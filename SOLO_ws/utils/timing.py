"""Real-time loop rate controller using monotonic clock."""

import time


class RateController:
    """Maintains a fixed-rate control loop with overrun detection.

    Usage::

        rate = RateController(60.0)             # 60 Hz
        while running:
            do_work()
            overrun = rate.sleep()              # blocks until next tick
            if overrun:
                print(f"Loop overrun by {rate.last_dt*1000:.1f} ms")
    """

    def __init__(self, frequency_hz: float):
        self.period = 1.0 / frequency_hz
        self.last_time = None
        self.last_dt = 0.0  # actual dt of the last cycle (seconds)

    def reset(self):
        """Call before the first loop iteration."""
        self.last_time = time.monotonic()

    def sleep(self) -> bool:
        """Sleep for the remaining time in the current period.

        Returns:
            True if the loop overran (took longer than the period).
        """
        now = time.monotonic()
        if self.last_time is None:
            self.last_time = now
            return False

        elapsed = now - self.last_time
        remaining = self.period - elapsed
        overrun = remaining < 0

        if remaining > 0:
            time.sleep(remaining)

        now_after = time.monotonic()
        self.last_dt = now_after - self.last_time
        self.last_time = now_after
        return overrun
