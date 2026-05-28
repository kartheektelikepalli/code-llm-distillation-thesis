import os
import json
import random


# =========================================================
# CONFIG
# =========================================================

INPUT_JSONL = (
    "optimization/lora_datasets/"
    "teacher_pass5_lora_dataset.jsonl"
)

OUTPUT_DIR = (
    "optimization/lora_datasets"
)

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train.jsonl"
)

VAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "val.jsonl"
)

VAL_RATIO = 0.1

SEED = 42


# =========================================================
# LOAD
# =========================================================

with open(
    INPUT_JSONL,
    "r"
) as f:

    samples = [
        json.loads(line)
        for line in f
    ]


# =========================================================
# SHUFFLE
# =========================================================

random.seed(SEED)

random.shuffle(samples)


# =========================================================
# SPLIT
# =========================================================

val_size = int(
    len(samples)
    * VAL_RATIO
)

val_samples = samples[:val_size]

train_samples = samples[val_size:]


# =========================================================
# SAVE
# =========================================================

with open(
    TRAIN_OUTPUT,
    "w"
) as f:

    for sample in train_samples:

        f.write(
            json.dumps(sample)
            + "\n"
        )


with open(
    VAL_OUTPUT,
    "w"
) as f:

    for sample in val_samples:

        f.write(
            json.dumps(sample)
            + "\n"
        )


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n===== TRAIN / VAL SPLIT COMPLETE ====="
)

print(
    f"\nTrain Samples : "
    f"{len(train_samples)}"
)

print(
    f"Val Samples   : "
    f"{len(val_samples)}"
)

print(
    f"\nSaved:"
)

print(TRAIN_OUTPUT)

print(VAL_OUTPUT)