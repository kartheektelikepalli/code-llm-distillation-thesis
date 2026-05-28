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

INPUT_FILE = (
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
    "distilled_student_pass5_execution.parquet"
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
# LOAD
# =========================================================

df = pd.read_parquet(
    INPUT_FILE
)

dataset = load_dataset(
    "mbpp"
)["test"]

task_to_tests = {}

for sample in dataset:

    task_to_tests[
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

pass_count = 0
fail_count = 0

start_time = time.time()


with mlflow.start_run():

    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        if not row["syntax_passed"]:

            fail_count += 1

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

        generated_code = (
            row["generated_code"]
        )

        task_id = row["task_id"]

        tests = task_to_tests[
            task_id
        ]

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
    # METRICS
    # =====================================================

    total = len(results_df)

    sample_level_pass_rate = (
        pass_count / total
    ) * 100

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
        sample_level_pass_rate
    )

    mlflow.log_metric(
        "true_pass_at_5",
        true_pass_at_5
    )

    mlflow.log_artifact(
        FULL_OUTPUT
    )


# =========================================================
# SUMMARY
# =========================================================

runtime_minutes = (
    time.time()
    - start_time
) / 60


print(
    "\n===== DISTILLED PASS@5 EXECUTION SUMMARY ====="
)

print(
    f"\nExecution Passed      : "
    f"{pass_count}"
)

print(
    f"Execution Failed      : "
    f"{fail_count}"
)

print(
    f"Sample-Level Pass Rate: "
    f"{sample_level_pass_rate:.2f}%"
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
    f"\nSaved to:\n"
    f"{FULL_OUTPUT}"
)