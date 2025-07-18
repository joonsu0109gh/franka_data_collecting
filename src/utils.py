import time


class Rate:
    """
    A precise rate limiter using busy-wait for robotics applications.

    This provides more accurate timing than time.sleep() for small intervals,
    which is crucial for maintaining consistent control loop frequencies.
    """

    def __init__(self, rate_hz: float):
        """
        Initialize the rate limiter.

        Args:
            rate_hz: Desired frequency in Hz
        """
        self.rate = rate_hz
        self.period = 1.0 / rate_hz
        self.last = time.time()

        # For monitoring actual performance
        self.timing_history = []
        self.max_history = 50  # Keep last 50 timings

    def sleep(self):
        """
        Waits for the remainder of the time to maintain the desired rate.
        Uses a busy-wait loop for precise timing.
        """
        target_time = self.last + self.period

        # Busy-wait loop for precise timing
        while time.time() < target_time:
            # Sleep for a very small duration to yield CPU time
            # without oversleeping.
            time.sleep(0.0001)

        current_time = time.time()
        actual_period = current_time - self.last

        # Track timing for monitoring
        self.timing_history.append(actual_period)
        if len(self.timing_history) > self.max_history:
            self.timing_history.pop(0)

        self.last = current_time

    def reset(self):
        """Reset the rate limiter timing."""
        self.last = time.time()
        self.timing_history.clear()

    def get_actual_rate(self):
        """
        Calculate actual rate over recent samples.
        Returns the current actual frequency in Hz.
        """
        if len(self.timing_history) < 5:
            return self.rate  # Not enough data yet

        avg_period = sum(self.timing_history) / len(self.timing_history)
        return 1.0 / avg_period if avg_period > 0 else self.rate

    def is_rate_stable(self, tolerance=0.3):
        """
        Check if the rate is within acceptable bounds (9.7-10.3Hz for 10Hz target).

        Args:
            tolerance: Allowed deviation in Hz (default 0.3 for ±0.3Hz)
        """
        if len(self.timing_history) < 10:
            return True  # Not enough data to judge

        actual_rate = self.get_actual_rate()
        min_rate = self.rate - tolerance
        max_rate = self.rate + tolerance

        return min_rate <= actual_rate <= max_rate
