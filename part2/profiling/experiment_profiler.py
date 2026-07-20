import time
from profiling.memory_monitor import MemoryMonitor
"""
Experiment Profiler

Provides a reusable framework for measuring
runtime and memory consumption for research
experiments.

Author: Kartheek Telikepalli
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExperimentMetrics:
    experiment_name: str

    start_time: datetime | None = None
    end_time: datetime | None = None

    runtime_seconds: float = 0.0

    peak_memory_gb: float = 0.0
    current_memory_gb: float = 0.0


class ExperimentProfiler:

    def __init__(self, experiment_name: str):
        self.metrics = ExperimentMetrics(
            experiment_name=experiment_name
        )
        self.memory_monitor = MemoryMonitor()

    def start(self):
        self.metrics.start_time = datetime.now()
        self._start_timer = time.perf_counter()
        self.memory_monitor.start()

    def stop(self):
        self.metrics.end_time = datetime.now()
        self.memory_monitor.stop()

        self.metrics.runtime_seconds = (
            time.perf_counter() - self._start_timer
    )

    def print_summary(self):

        print("\n==============================")
        print("Experiment Summary")
        print("==============================")

        print(f"Experiment : {self.metrics.experiment_name}")
        print(f"Started    : {self.metrics.start_time}")
        print(f"Finished   : {self.metrics.end_time}")
        print(f"Runtime    : {self.metrics.runtime_seconds:.2f} sec")
        print(f"Peak Memory: {self.memory_monitor.peak_memory_gb():.2f} GB")

    def log_to_mlflow(self):

        import mlflow

        mlflow.log_metric(
            "runtime_sec",
            self.metrics.runtime_seconds
        )

        mlflow.log_metric(
            "peak_ram_gb",
            self.memory_monitor.peak_memory_gb()
        )

        mlflow.log_metric(
            "average_ram_gb",
            self.memory_monitor.average_memory_gb()
        )

        mlflow.log_metric(
            "current_ram_gb",
            self.memory_monitor.current_memory_gb()
        )