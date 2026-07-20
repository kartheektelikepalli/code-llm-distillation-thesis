import time
import numpy as np

from profiling.experiment_profiler import ExperimentProfiler


profiler = ExperimentProfiler("Profiler Test")

profiler.start()

print("Sleeping for 2 seconds...")
time.sleep(2)

print("Allocating memory...")
x = np.random.rand(8000, 8000)

time.sleep(2)

profiler.stop()

print()
profiler.print_summary()