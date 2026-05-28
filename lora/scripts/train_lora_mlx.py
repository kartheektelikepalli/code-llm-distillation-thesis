import os
import mlflow


# --------------------------------------------------
# Config
# --------------------------------------------------

MODEL_PATH = (
    "lora/artifacts/"
    "deepseek-coder-1.3b-instruct-mlx"
)

TRAIN_FILE = "lora/datasets/train.jsonl"

VAL_FILE = "lora/datasets/val.jsonl"

ADAPTER_OUTPUT = "lora/checkpoints"

os.makedirs(ADAPTER_OUTPUT, exist_ok=True)


# --------------------------------------------------
# MLflow
# --------------------------------------------------

mlflow.set_experiment("mlx_lora_distillation")


with mlflow.start_run():

    # ----------------------------------------------
    # Log Params
    # ----------------------------------------------

    mlflow.log_param("model", "deepseek-coder-1.3b-instruct")

    mlflow.log_param("framework", "mlx")

    mlflow.log_param("learning_rate", 1e-5)

    mlflow.log_param("epochs", 3)

    mlflow.log_param("batch_size", 1)


    # ----------------------------------------------
    # Launch MLX LoRA Training
    # ----------------------------------------------

    command = f"""
mlx_lm.lora \
--model {MODEL_PATH} \
--train \
--data lora/datasets \
--batch-size 1 \
--iters 300 \
--learning-rate 1e-5 \
--steps-per-report 10 \
--steps-per-eval 20 \
--save-every 50 \
--adapter-path {ADAPTER_OUTPUT}
"""

    print("\n===== STARTING MLX LORA TRAINING =====\n")

    exit_code = os.system(command)

    mlflow.log_metric("training_exit_code", exit_code)

    print("\n===== TRAINING FINISHED =====\n")