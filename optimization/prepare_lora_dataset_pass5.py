import os
import json
import pandas as pd


# =========================================================
# INPUT
# =========================================================

INPUT_PARQUET = (
    "optimization/artifacts/"
    "teacher_pass5_execution_validation/"
    "teacher_pass5_execution_passed.parquet"
)


# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = (
    "optimization/lora_datasets"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_JSONL = os.path.join(
    OUTPUT_DIR,
    "teacher_pass5_lora_dataset.jsonl"
)


# =========================================================
# LOAD
# =========================================================

df = pd.read_parquet(
    INPUT_PARQUET
)


# =========================================================
# CONVERT
# =========================================================

with open(
    OUTPUT_JSONL,
    "w"
) as f:

    for _, row in df.iterrows():

        prompt = row["prompt"]

        code = row["generated_code"]

        sample = {

            "messages": [

                {
                    "role": "user",
                    "content": prompt
                },

                {
                    "role": "assistant",
                    "content": code
                }
            ]
        }

        f.write(
            json.dumps(sample)
            + "\n"
        )


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n===== PASS@5 LORA DATASET READY ====="
)

print(
    f"\nTotal Samples : "
    f"{len(df)}"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_JSONL}"
)