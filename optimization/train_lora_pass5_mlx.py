import os
import mlflow


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = (
    "lora/artifacts/"
    "deepseek-coder-1.3b-instruct-mlx"
)

DATASET_DIR = (
    "optimization/lora_datasets"
)

ADAPTER_OUTPUT = (
    "optimization/lora_adapters_pass5"
)

os.makedirs(
    ADAPTER_OUTPUT,
    exist_ok=True
)


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_experiment(
    "mlx_lora_distillation_pass5"
)


# =========================================================
# TRAIN
# =========================================================

with mlflow.start_run():

    mlflow.log_param(
        "model",
        "deepseek-coder-1.3b-instruct"
    )

    mlflow.log_param(
        "framework",
        "mlx"
    )

    mlflow.log_param(
        "teacher_dataset_size",
        684
    )

    mlflow.log_param(
        "train_samples",
        616
    )

    mlflow.log_param(
        "val_samples",
        68
    )

    mlflow.log_param(
        "batch_size",
        1
    )

    mlflow.log_param(
        "iterations",
        300
    )

    mlflow.log_param(
        "learning_rate",
        1e-5
    )


    # =====================================================
    # MLX COMMAND
    # =====================================================

    command = f"""
mlx_lm.lora \
--model {MODEL_PATH} \
--train \
--data {DATASET_DIR} \
--batch-size 1 \
--iters 300 \
--learning-rate 1e-5 \
--steps-per-report 10 \
--steps-per-eval 20 \
--save-every 50 \
--adapter-path {ADAPTER_OUTPUT}
"""


    print(
        "\n===== STARTING PASS@5 LORA TRAINING =====\n"
    )

    exit_code = os.system(
        command
    )

    mlflow.log_metric(
        "training_exit_code",
        exit_code
    )


print(
    "\n===== TRAINING FINISHED ====="
)