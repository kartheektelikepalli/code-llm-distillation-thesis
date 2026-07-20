"""
Memory Monitor

Continuously monitors process memory during an experiment.
"""

import threading
import time
import psutil


class MemoryMonitor:

    def __init__(self, interval=0.1):

        self.interval = interval

        self._running = False
        self._thread = None

        self._samples = []

        self.process = psutil.Process()

    def start(self):

        self._running = True

        self._thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self._thread.start()

    def stop(self):

        self._running = False

        if self._thread is not None:
            self._thread.join()

    def peak_memory_gb(self):

        if not self._samples:
            return 0.0

        return max(self._samples)

    def current_memory_gb(self):

        if not self._samples:
            return 0.0

        return self._samples[-1]

    def average_memory_gb(self):

        if not self._samples:
            return 0.0

        return sum(self._samples) / len(self._samples)

    def _monitor(self):

        while self._running:

            memory_bytes = self.process.memory_info().rss

            memory_gb = memory_bytes / (1024 ** 3)

            self._samples.append(memory_gb)

            time.sleep(self.interval)