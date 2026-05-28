import ast
import os
import pandas as pd


# --------------------------------------------------
# Input
# --------------------------------------------------

INPUT_FILE = (
    "lora/artifacts/distilled_outputs/"
    "distilled_student_outputs.parquet"
)

OUTPUT_DIR = (
    "lora/artifacts/distilled_outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_parquet(INPUT_FILE)


# --------------------------------------------------
# Syntax Validation
# --------------------------------------------------

results = []

pass_count = 0
fail_count = 0


for _, row in df.iterrows():

    generated_output = row["generated_output"]

    syntax_passed = True

    try:

        ast.parse(generated_output)

    except Exception:

        syntax_passed = False

    if syntax_passed:
        pass_count += 1
    else:
        fail_count += 1

    results.append(
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "generated_output": generated_output,
            "latency": row["latency"],
            "syntax_passed": syntax_passed
        }
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

results_df = pd.DataFrame(results)

FULL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "distilled_syntax_validation.parquet"
)

PASSED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "distilled_syntax_passed.parquet"
)

FAILED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "distilled_syntax_failed.parquet"
)

results_df.to_parquet(FULL_OUTPUT)

results_df[
    results_df["syntax_passed"] == True
].to_parquet(PASSED_OUTPUT)

results_df[
    results_df["syntax_passed"] == False
].to_parquet(FAILED_OUTPUT)


# --------------------------------------------------
# Summary
# --------------------------------------------------

total = len(results_df)

syntax_rate = (pass_count / total) * 100

print("\n===== DISTILLED SYNTAX SUMMARY =====")

print(f"Total Samples : {total}")

print(f"Syntax Passed : {pass_count}")

print(f"Syntax Failed : {fail_count}")

print(f"Syntax Rate   : {syntax_rate:.2f}%")