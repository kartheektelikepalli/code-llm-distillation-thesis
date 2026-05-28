import traceback
import pandas as pd
import mlflow

from tqdm import tqdm
from datasets import load_dataset


# --------------------------------------------------
# MLflow
# --------------------------------------------------

mlflow.set_experiment(
    "mlx_distilled_execution_validation"
)


# --------------------------------------------------
# Input
# --------------------------------------------------

INPUT_FILE = (
    "lora/artifacts/distilled_outputs/"
    "distilled_syntax_passed.parquet"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_parquet(INPUT_FILE)

dataset = load_dataset("mbpp")["test"]


# --------------------------------------------------
# Build Task → Tests Mapping
# --------------------------------------------------

task_to_tests = {}

for sample in dataset:

    task_to_tests[
        sample["task_id"]
    ] = sample["test_list"]


# --------------------------------------------------
# Execution Validation
# --------------------------------------------------

results = []

pass_count = 0
fail_count = 0


with mlflow.start_run():

    for _, row in tqdm(df.iterrows(), total=len(df)):

        generated_output = row["generated_output"]

        task_id = row["task_id"]

        tests = task_to_tests[task_id]

        execution_passed = True

        error_message = ""

        try:

            local_scope = {}

            exec(generated_output, local_scope)

            for test_case in tests:

                exec(test_case, local_scope)

        except Exception:

            execution_passed = False

            error_message = traceback.format_exc()

        if execution_passed:
            pass_count += 1
        else:
            fail_count += 1

        results.append(
            {
                "task_id": task_id,
                "generated_output": generated_output,
                "execution_passed": execution_passed,
                "error_message": error_message
            }
        )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_FILE = (
        "lora/artifacts/distilled_outputs/"
        "distilled_execution_validation.parquet"
    )

    pd.DataFrame(results).to_parquet(
        OUTPUT_FILE
    )


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    total = len(results)

    pass_at_1 = (
        pass_count / total
    ) * 100


    mlflow.log_metric(
        "total_samples",
        total
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
        "pass_at_1",
        pass_at_1
    )

    mlflow.log_artifact(
        OUTPUT_FILE
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n===== DISTILLED EXECUTION SUMMARY =====")

print(f"Total Samples     : {total}")

print(f"Execution Passed  : {pass_count}")

print(f"Execution Failed  : {fail_count}")

print(f"Pass@1            : {pass_at_1:.2f}%")