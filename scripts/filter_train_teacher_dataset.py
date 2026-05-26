from datasets import load_dataset
import pandas as pd

# -----------------------------
# LOAD EXISTING TEACHER DATASET
# -----------------------------

teacher_df = pd.read_parquet(
    "data/final_teacher_dataset.parquet"
)
print(teacher_df.columns)
print(f"Original teacher samples: {len(teacher_df)}")

# -----------------------------
# LOAD MBPP TRAIN SPLIT
# -----------------------------

dataset = load_dataset("mbpp")

train_split = dataset["train"]

train_task_ids = set(train_split["task_id"])

print(f"MBPP train task ids: {len(train_task_ids)}")

# -----------------------------
# FILTER ONLY TRAIN SAMPLES
# -----------------------------

train_prompts = set(train_split["text"])

filtered_df = teacher_df[
    teacher_df["prompt"].isin(train_prompts)
]

print(f"Filtered train samples: {len(filtered_df)}")

# -----------------------------
# SAVE CLEAN TRAIN DATASET
# -----------------------------

output_path = "data/final_teacher_train_dataset.parquet"

filtered_df.to_parquet(output_path)

print(f"Saved filtered dataset to: {output_path}")