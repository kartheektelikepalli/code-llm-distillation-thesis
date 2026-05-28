import os
import ast
import traceback
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_parquet", required=True)
args = parser.parse_args()

INPUT_PARQUET = args.input_parquet

OUTPUT_DIR = "data/execution_validated_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PARQUET = os.path.join(
    OUTPUT_DIR,
    "execution_validated_full.parquet"
)

# ---------------------------------------------------------
# Load generated outputs
# ---------------------------------------------------------

df = pd.read_parquet(INPUT_PARQUET)

# ---------------------------------------------------------
# Load MBPP tests
# ---------------------------------------------------------

mbpp = load_dataset("mbpp")["train"]

test_lookup = {}

for item in mbpp:
    test_lookup[item["task_id"]] = item["test_list"]

# ---------------------------------------------------------
# Execution helper
# ---------------------------------------------------------

def run_code_and_tests(code, tests):

    namespace = {}

    try:
        exec(code, namespace)

        for test in tests:
            exec(test, namespace)

        return True, None, None

    except Exception as e:

        tb = traceback.format_exc()

        return (
            False,
            type(e).__name__,
            tb
        )

# ---------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------

results = []

passed_count = 0
failed_count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    task_id = row["task_id"]
    generated_code = row["generated_code"]

    tests = test_lookup.get(task_id, [])

    passed, error_type, error_traceback = run_code_and_tests(
        generated_code,
        tests
    )

    if passed:
        passed_count += 1
    else:
        failed_count += 1

    results.append(
        {
            **row.to_dict(),

            "execution_passed": passed,

            "error_type": error_type,

            "error_traceback": error_traceback,
        }
    )

# ---------------------------------------------------------
# Save full dataset
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_parquet(OUTPUT_PARQUET)
results_df.to_parquet(OUTPUT_PARQUET)

passed_df = results_df[
    results_df["execution_passed"] == True
]

failed_df = results_df[
    results_df["execution_passed"] == False
]

passed_df.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "execution_passed.parquet"
    )
)

failed_df.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "execution_failed.parquet"
    )
)

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

total = len(results_df)

print("\n===== EXECUTION SUMMARY =====")

print(f"Total Samples     : {total}")
print(f"Execution Passed  : {passed_count}")
print(f"Execution Failed  : {failed_count}")

if total > 0:
    print(
        f"Pass Rate         : {100 * passed_count / total:.2f}%"
    )

print(f"\nSaved to:")
print(OUTPUT_PARQUET)