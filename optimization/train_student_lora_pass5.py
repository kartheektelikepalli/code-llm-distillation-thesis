import os
import time
import mlflow
import pandas as pd

from datasets import Dataset

from mlx_lm import load

from mlx_lm.tuner import (
    TrainingArgs,
    train,
    linear_to_lora_layers
)


# =========================================================
# PATHS
# =========================================================

TEACHER_DATASET = (
    "optimization/artifacts/"
    "teacher_pass5_execution_validation/"
    "teacher_pass5_execution_passed.parquet"
)

BASE_MODEL = (
    "models/student/"
    "deepseek-coder-1.3b-instruct"
)

OUTPUT_DIR = (
    "optimization/artifacts/"
    "student_lora_pass5"
)

ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "adapters"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    ADAPTER_DIR,
    exist_ok=True
)


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_experiment(
    "student_lora_pass5_training"
)


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_parquet(
    TEACHER_DATASET
)

print(
    f"\nLoaded teacher samples: "
    f"{len(df)}"
)


# =========================================================
# FORMAT DATA
# =========================================================

formatted_samples = []

for _, row in df.iterrows():

    prompt = row["prompt"]

    code = row["generated_code"]

    text = (
        f"{prompt}\n"
        f"{code}"
    )

    formatted_samples.append(
        {
            "text": text
        }
    )


dataset = Dataset.from_list(
    formatted_samples
)


# =========================================================
# LOAD MODEL
# =========================================================

model, tokenizer = load(
    BASE_MODEL
)


# =========================================================
# INSERT LORA LAYERS
# =========================================================

linear_to_lora_layers(
    model,
    num_layers=16
)


# =========================================================
# TRAINING ARGS
# =========================================================

training_args = TrainingArgs(

    batch_size=1,

    iters=1000,

    val_batches=0,

    steps_per_report=10,

    steps_per_eval=0,

    steps_per_save=100,

    adapter_file=os.path.join(
        ADAPTER_DIR,
        "adapters.safetensors"
    )
)


# =========================================================
# TRAIN
# =========================================================

start_time = time.time()


with mlflow.start_run():

    mlflow.log_param(
        "base_model",
        BASE_MODEL
    )

    mlflow.log_param(
        "teacher_dataset",
        TEACHER_DATASET
    )

    mlflow.log_param(
        "teacher_dataset_size",
        len(dataset)
    )

    mlflow.log_param(
        "iterations",
        1000
    )

    mlflow.log_param(
        "batch_size",
        1
    )

    mlflow.log_param(
        "lora_layers",
        16
    )


    train(

        model=model,

        tokenizer=tokenizer,

        dataset=dataset,

        args=training_args,

        adapter_path=os.path.join(
            ADAPTER_DIR,
            "adapters.safetensors"
        )
    )


    runtime_minutes = (
        time.time()
        - start_time
    ) / 60


    mlflow.log_metric(
        "training_runtime_minutes",
        runtime_minutes
    )

    mlflow.log_artifact(
        os.path.join(
            ADAPTER_DIR,
            "adapters.safetensors"
        )
    )


# =========================================================
# SUMMARY
# =========================================================

print(
    "\n===== LORA TRAINING COMPLETE ====="
)

print(
    f"\nTeacher Samples Used : "
    f"{len(dataset)}"
)

print(
    f"Runtime (minutes)    : "
    f"{runtime_minutes:.2f}"
)

print(
    f"\nAdapters saved to:\n"
    f"{ADAPTER_DIR}"
)