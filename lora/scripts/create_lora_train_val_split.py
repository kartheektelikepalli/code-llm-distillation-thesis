import os
import json
import random


# --------------------------------------------------
# Config
# --------------------------------------------------

INPUT_JSONL = (
    "lora/datasets/teacher_lora_dataset.jsonl"
)

TRAIN_OUTPUT = (
    "lora/datasets/train.jsonl"
)

VAL_OUTPUT = (
    "lora/datasets/val.jsonl"
)

VAL_RATIO = 0.1

SEED = 42


# --------------------------------------------------
# Load
# --------------------------------------------------

with open(INPUT_JSONL, "r") as f:

    samples = [json.loads(line) for line in f]


# --------------------------------------------------
# Shuffle
# --------------------------------------------------

random.seed(SEED)

random.shuffle(samples)


# --------------------------------------------------
# Split
# --------------------------------------------------

val_size = int(len(samples) * VAL_RATIO)

val_samples = samples[:val_size]

train_samples = samples[val_size:]


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(TRAIN_OUTPUT, "w") as f:

    for sample in train_samples:

        f.write(json.dumps(sample) + "\n")


with open(VAL_OUTPUT, "w") as f:

    for sample in val_samples:

        f.write(json.dumps(sample) + "\n")


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n===== TRAIN / VAL SPLIT COMPLETE =====")

print(f"Train Samples : {len(train_samples)}")

print(f"Val Samples   : {len(val_samples)}")

print(f"\nSaved:")

print(TRAIN_OUTPUT)

print(VAL_OUTPUT)