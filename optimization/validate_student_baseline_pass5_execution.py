import os
import time
import tempfile
import subprocess
import traceback

import pandas as pd
import mlflow

from tqdm import tqdm
from datasets import load_dataset


# =========================================================
# INPUT
# =========================================================

INPUT_PARQUET = (
    "optimization/artifacts/"
    "distilled_student_pass5/"
    "distilled_student_pass5.parquet"
)


# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = (
    "optimization/artifacts/"
    "distilled_student_pass5_execution_validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

FULL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "distilled_student_pass5_execution_full.parquet"
)

PASSED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "distilled_student_pass5_execution_passed.parquet"
)

FAILED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "distilled_student_pass5_execution_failed.parquet"
)


# =========================================================
# EXECUTION
# =========================================================

EXECUTION_TIMEOUT = 5


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_experiment(
    "distilled_student_pass5_execution_validation"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_parquet(
    INPUT_PARQUET
)

dataset = load_dataset(
    "mbpp"
)["test"]

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
# MAIN
# =========================================================

results = []

execution_pass_count = 0
execution_fail_count = 0

start_time = time.time()


with mlflow.start_run():

    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        if not row["syntax_passed"]:

            execution_fail_count += 1

            results.append(
                {
                    **row.to_dict(),

                    "execution_passed":
                        False,

                    "execution_error":
                        "Syntax Failed"
                }
            )

            continue

        task_id = row["task_id"]

        generated_code = (
            row["generated_code"]
        )

        tests = task_id_to_tests[
            task_id
        ]

        execution_passed, error_message = (
            run_code_with_timeout(
                generated_code,
                tests,
            )
        )

        if execution_passed:

            execution_pass_count += 1

        else:

            execution_fail_count += 1

        results.append(
            {
                **row.to_dict(),

                "execution_passed":
                    execution_passed,

                "execution_error":
                    error_message
            }
        )


    # =====================================================
    # SAVE
    # =====================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_parquet(
        FULL_OUTPUT
    )

    results_df[
        results_df[
            "execution_passed"
        ] == True
    ].to_parquet(
        PASSED_OUTPUT
    )

    results_df[
        results_df[
            "execution_passed"
        ] == False
    ].to_parquet(
        FAILED_OUTPUT
    )


    # =====================================================
    # TRUE PASS@5
    # =====================================================

    grouped = results_df.groupby(
        "task_id"
    )["execution_passed"].any()

    true_pass_at_5 = (
        grouped.sum()
        / len(grouped)
    ) * 100


    # =====================================================
    # MLFLOW
    # =====================================================

    total_samples = len(
        results_df
    )

    execution_pass_rate = (
        execution_pass_count
        / max(total_samples, 1)
    ) * 100

    mlflow.log_param(
        "input_parquet",
        INPUT_PARQUET
    )

    mlflow.log_metric(
        "total_generations",
        total_samples
    )

    mlflow.log_metric(
        "execution_passed",
        execution_pass_count
    )

    mlflow.log_metric(
        "execution_failed",
        execution_fail_count
    )

    mlflow.log_metric(
        "sample_level_pass_rate",
        execution_pass_rate
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

runtime_minutes = (
    time.time()
    - start_time
) / 60


print(
    "\n===== DISTILLED STUDENT PASS@5 EXECUTION SUMMARY ====="
)

print(
    f"\nTotal Generations     : "
    f"{total_samples}"
)

print(
    f"Execution Passed      : "
    f"{execution_pass_count}"
)

print(
    f"Execution Failed      : "
    f"{execution_fail_count}"
)

print(
    f"Sample-Level Pass Rate: "
    f"{execution_pass_rate:.2f}%"
)

print(
    f"TRUE Pass@5           : "
    f"{true_pass_at_5:.2f}%"
)

print(
    f"\nRuntime (minutes)     : "
    f"{runtime_minutes:.2f}"
)

print(
    "\nSaved to:"
)

print(FULL_OUTPUT)

print(PASSED_OUTPUT)

print(FAILED_OUTPUT)