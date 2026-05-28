import os
import traceback
import pandas as pd
import mlflow
from tqdm import tqdm
from datasets import load_dataset


# --------------------------------------------------
# Input
# --------------------------------------------------

INPUT_PARQUET = (
    "data/student_baseline_outputs/"
    "student_baseline_mbpp_20260526_230249.parquet"
)

OUTPUT_DIR = "data/student_execution_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
mlflow.set_experiment("student_baseline_mbpp")

# --------------------------------------------------
# Load Student Generations
# --------------------------------------------------

df = pd.read_parquet(INPUT_PARQUET)


# --------------------------------------------------
# Load MBPP TEST Split
# --------------------------------------------------

dataset = load_dataset("mbpp")["test"]

task_id_to_tests = {}

for sample in dataset:

    task_id_to_tests[sample["task_id"]] = sample["test_list"]


# --------------------------------------------------
# Execution Validation
# --------------------------------------------------
with mlflow.start_run():
    execution_results = []

    pass_count = 0
    fail_count = 0


    for _, row in tqdm(df.iterrows(), total=len(df)):

        task_id = row["task_id"]

        generated_code = row["generated_code"]

        tests = task_id_to_tests.get(task_id, [])

        execution_passed = True

        error_message = ""

        try:

            local_scope = {}

            exec(generated_code, local_scope)

            for test_case in tests:

                exec(test_case, local_scope)

        except Exception:

            execution_passed = False

            error_message = traceback.format_exc()

        if execution_passed:
            pass_count += 1
        else:
            fail_count += 1

        execution_results.append(
            {
                "task_id": task_id,
                "prompt": row["prompt"],
                "generated_code": generated_code,
                "latency": row["latency"],
                "execution_passed": execution_passed,
                "error_message": error_message
            }
        )


    # --------------------------------------------------
    # Save Results
    # --------------------------------------------------

    results_df = pd.DataFrame(execution_results)

    FULL_OUTPUT = os.path.join(
        OUTPUT_DIR,
        "student_execution_validated_full.parquet"
    )

    PASSED_OUTPUT = os.path.join(
        OUTPUT_DIR,
        "student_execution_passed.parquet"
    )

    FAILED_OUTPUT = os.path.join(
        OUTPUT_DIR,
        "student_execution_failed.parquet"
    )

    results_df.to_parquet(FULL_OUTPUT)

    results_df[
        results_df["execution_passed"] == True
    ].to_parquet(PASSED_OUTPUT)

    results_df[
        results_df["execution_passed"] == False
    ].to_parquet(FAILED_OUTPUT)
    mlflow.log_artifact(FULL_OUTPUT)

    mlflow.log_artifact(PASSED_OUTPUT)

    mlflow.log_artifact(FAILED_OUTPUT)


    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    total = len(results_df)

    pass_rate = (pass_count / total) * 100
    mlflow.log_metric("total_samples", total)

    mlflow.log_metric("execution_passed", pass_count)

    mlflow.log_metric("execution_failed", fail_count)

    mlflow.log_metric("pass_at_1", pass_rate)

    print("\n===== STUDENT EXECUTION SUMMARY =====")

    print(f"Total Samples     : {total}")

    print(f"Execution Passed  : {pass_count}")

    print(f"Execution Failed  : {fail_count}")

    print(f"Pass@1            : {pass_rate:.2f}%")

    print("\nSaved to:")

    print(FULL_OUTPUT)

    print(PASSED_OUTPUT)

    print(FAILED_OUTPUT)