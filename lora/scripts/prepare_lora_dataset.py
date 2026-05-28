import os
import pandas as pd


# --------------------------------------------------
# Input
# --------------------------------------------------

INPUT_PARQUET = (
    "data/execution_validated_outputs/"
    "execution_passed.parquet"
)

OUTPUT_JSONL = (
    "lora/datasets/"
    "teacher_lora_dataset.jsonl"
)

os.makedirs("lora/datasets", exist_ok=True)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

df = pd.read_parquet(INPUT_PARQUET)


# --------------------------------------------------
# Convert
# --------------------------------------------------

with open(OUTPUT_JSONL, "w") as f:

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
            pd.Series(sample).to_json()
            + "\n"
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n===== LORA DATASET READY =====")

print(f"Total Samples: {len(df)}")

print(f"Saved to:\n{OUTPUT_JSONL}")