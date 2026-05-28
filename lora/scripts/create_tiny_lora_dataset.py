import json


INPUT_FILE = "lora/datasets/train.jsonl"

TINY_TRAIN = "lora/datasets/tiny_train.jsonl"

TINY_VAL = "lora/datasets/tiny_val.jsonl"


with open(INPUT_FILE, "r") as f:

    samples = [json.loads(line) for line in f]


tiny_train = samples[:10]

tiny_val = samples[10:12]


with open(TINY_TRAIN, "w") as f:

    for sample in tiny_train:

        f.write(json.dumps(sample) + "\n")


with open(TINY_VAL, "w") as f:

    for sample in tiny_val:

        f.write(json.dumps(sample) + "\n")


print("\n===== TINY DATASET CREATED =====")

print(f"Train: {len(tiny_train)}")

print(f"Val: {len(tiny_val)}")