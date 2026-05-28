import os
import traceback
import tempfile
import subprocess
import textwrap

import pandas as pd
import mlflow

from tqdm import tqdm
from datasets import load_dataset

from config import *


# =========================================================
# INPUT
# =========================================================

INPUT_PARQUET = (
    "optimization/artifacts/"
    "teacher_pass5_outputs/"
    "teacher_pass5_dataset_20260528_225244.parquet"
)


# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = (
    "optimization/artifacts/"
    "teacher_pass5_execution_validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =========================================================
# TIMEOUT
# =========================================================

EXECUTION_TIMEOUT = 5


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_experiment(
    "teacher_pass5_execution_validation"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_parquet(
    INPUT_PARQUET
)


# =========================================================
# LOAD MBPP TESTS
# =========================================================

dataset = load_dataset(
    "mbpp"
)["train"]

task_id_to_tests = {}

for sample in dataset:

    task_id_to_tests[
        sample["task_id"]
    ] = sample["test_list"]


# =========================================================
# SAFE EXECUTION
# =========================================================

def run_code_with_timeout(
    generated_code,
    tests,
):

    full_script = generated_code + "\n\n"

    for test_case in tests:

        full_script += (
            test_case + "\n"
        )

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as temp_file:

            temp_file.write(full_script)

            temp_path = temp_file.name

        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
        )

        os.remove(temp_path)

        if result.returncode == 0:

            return True, ""

        else:

            return (
                False,
                result.stderr
            )

    except subprocess.TimeoutExpired:

        try:
            os.remove(temp_path)
        except:
            pass

        return (
            False,
            "Execution Timeout"
        )

    except Exception:

        try:
            os.remove(temp_path)
        except:
            pass

        return (
            False,
            traceback.format_exc()
        )


# =========================================================
# MAIN VALIDATION
# =========================================================

with mlflow.start_run():

    execution_results = []

    pass_count = 0
    fail_count = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        task_id = row["task_id"]

        generated_code = row[
            "generated_code"
        ]

        tests = task_id_to_tests.get(
            task_id,
            []
        )

        execution_passed, error_message = (
            run_code_with_timeout(
                generated_code,
                tests,
            )
        )

        if execution_passed:
            pass_count += 1
        else:
            fail_count += 1

        execution_results.append(
            {
                **row.to_dict(),

                "execution_passed":
                    execution_passed,

                "error_message":
                    error_message,
            }
        )


    # =========================================================
    # SAVE RESULTS
    # =========================================================

    results_df = pd.DataFrame(
        execution_results
    )

    FULL_OUTPUT = os.path.join(
        OUTPUT_DIR,
        "teacher_pass5_execution_full.parquet"
    )

    PASSED_OUTPUT = os.path.join(
        OUTPUT_DIR,
        "teacher_pass5_execution_passed.parquet"
    )

    FAILED_OUTPUT = os.path.join(
        OUTPUT_DIR,
        "teacher_pass5_execution_failed.parquet"
    )

    results_df.to_parquet(
        FULL_OUTPUT
    )

    results_df[
        results_df["execution_passed"]
        == True
    ].to_parquet(
        PASSED_OUTPUT
    )

    results_df[
        results_df["execution_passed"]
        == False
    ].to_parquet(
        FAILED_OUTPUT
    )


    # =========================================================
    # TRUE PASS@5
    # =========================================================

    grouped = results_df.groupby(
        "task_id"
    )["execution_passed"].any()

    true_pass_at_5 = (
        grouped.sum()
        / len(grouped)
    ) * 100


    # =========================================================
    # MLFLOW LOGGING
    # =========================================================

    mlflow.log_param(
        "dataset_name",
        DATASET_NAME
    )

    mlflow.log_param(
        "dataset_split",
        DATASET_SPLIT
    )

    mlflow.log_param(
        "num_return_sequences",
        NUM_RETURN_SEQUENCES
    )

    mlflow.log_param(
        "execution_timeout_seconds",
        EXECUTION_TIMEOUT
    )

    mlflow.log_metric(
        "total_generations",
        len(results_df)
    )

    mlflow.log_metric(
        "execution_passed",
        pass_count
    )

    mlflow.log_metric(
        "execution_failed",
        fail_count
    )

    mlflow.log_metric(
        "sample_level_pass_rate",
        (
            pass_count
            / len(results_df)
        ) * 100
    )

    mlflow.log_metric(
        "true_pass_at_5",
        true_pass_at_5
    )

    mlflow.log_artifact(
        FULL_OUTPUT
    )

    mlflow.log_artifact(
        PASSED_OUTPUT
    )

    mlflow.log_artifact(
        FAILED_OUTPUT
    )


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n===== PASS@5 EXECUTION SUMMARY ====="
)

print(
    f"\nTotal Generations     : "
    f"{len(results_df)}"
)

print(
    f"Execution Passed      : "
    f"{pass_count}"
)

print(
    f"Execution Failed      : "
    f"{fail_count}"
)

print(
    f"Sample-Level Pass Rate: "
    f"{(pass_count / len(results_df)) * 100:.2f}%"
)

print(
    f"TRUE Pass@5           : "
    f"{true_pass_at_5:.2f}%"
)

print("\nSaved to:")

print(FULL_OUTPUT)

print(PASSED_OUTPUT)

print(FAILED_OUTPUT)