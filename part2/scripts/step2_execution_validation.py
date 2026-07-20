import traceback
from pathlib import Path

import mlflow
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

import sys

sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
    )
)

from config import *
from profiling.experiment_profiler import ExperimentProfiler
from utils.helper_extractor import extract_helper_definitions

# =========================================================
# PATHS
# =========================================================

INPUT_PARQUET = (
    Path(OUTPUT_DIR)
    / "teacher_baseline.parquet"
)

OUTPUT_PARQUET = (
    Path(OUTPUT_DIR)
    / "teacher_execution_validated.parquet"
)


# =========================================================
# MLFLOW
# =========================================================

EXECUTION_VALIDATION_EXPERIMENT = (
    "part2_execution_validation"
)


def main():

    mlflow.set_experiment(
        EXECUTION_VALIDATION_EXPERIMENT
    )

    profiler = ExperimentProfiler(
        "Execution Validation"
    )

    teacher_df = pd.read_parquet(
        INPUT_PARQUET
    )

    dataset = load_dataset(
        DATASET_NAME
    )

    problems = dataset[
        DATASET_SPLIT
    ]

    # =========================================================
    # CREATE TASK LOOKUP
    # =========================================================

    task_lookup = {
        problem["task_id"]: problem
        for problem in problems
    }

    # =========================================================
    # START PROFILER
    # =========================================================

    profiler.start()

    results = []

    for _, row in tqdm(
        teacher_df.iterrows(),
        total=len(teacher_df)
    ):
        problem = task_lookup[
        row["task_id"]
    ]
        
        test_setup_code = (
        problem["test_setup_code"]
    )

        test_list = (
            problem["test_list"]
        )

        reference_code = problem["code"]

        helper_code = extract_helper_definitions(
            reference_code,
            row["expected_function_name"]
        )
        execution_globals = {}

        try:

            exec(helper_code, execution_globals)

            exec(
                row["generated_code"],
                execution_globals
            )

            for test in test_list:
                exec(
                    test,
                    execution_globals
                )

            execution_passed = True
            error_type = None
            error_traceback = None

        except Exception as e:

            execution_passed = False
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()

        results.append({
            **row.to_dict(),
            "execution_passed": execution_passed,
            "error_type": error_type,
            "error_traceback": error_traceback,
        })
    results_df = pd.DataFrame(results)

    results_df.to_parquet(
        OUTPUT_PARQUET,
        index=False
    )

    print(f"Saved {len(results_df)} records.")

    df = pd.read_parquet(OUTPUT_PARQUET)

    print(df.shape)

    print(df["execution_passed"].value_counts())

    print(df["error_type"].value_counts())

    print("=" * 60)
    print(df["error_type"].value_counts())

    print("=" * 60)
    print(df[df["execution_passed"]]["error_type"].value_counts(dropna=False))

    print("=" * 60)
    print(df[df["error_type"] == "AssertionError"].iloc[0])

    print("=" * 60)
    print(df[df["error_type"] == "TypeError"].iloc[0])

    print("=" * 60)
    print(df[df["error_type"] == "NameError"].iloc[0])

if __name__ == "__main__":
    main()