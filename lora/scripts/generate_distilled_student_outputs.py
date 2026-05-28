import os
import time
import pandas as pd
import mlflow

from tqdm import tqdm
from datasets import load_dataset

from mlx_lm import load, generate


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = (
    "lora/artifacts/"
    "deepseek-coder-1.3b-instruct-mlx"
)

ADAPTER_PATH = (
    "lora/checkpoints"
)

OUTPUT_DIR = (
    "lora/artifacts/distilled_outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# MLflow
# --------------------------------------------------

mlflow.set_experiment(
    "mlx_distilled_inference"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model, tokenizer = load(
    MODEL_PATH,
    adapter_path=ADAPTER_PATH
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = load_dataset("mbpp")["test"]


# --------------------------------------------------
# Prompt
# --------------------------------------------------

def build_prompt(problem):

    messages = [
        {
            "role": "user",
            "content": (
                "Write a Python function to solve the following problem.\n\n"
                f"{problem}"
            )
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# --------------------------------------------------
# Generation
# --------------------------------------------------

results = []


with mlflow.start_run():

    for sample in tqdm(dataset):

        prompt = build_prompt(sample["text"])

        start = time.time()

        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=128
        )

        latency = time.time() - start

        results.append(
            {
                "task_id": sample["task_id"],
                "prompt": prompt,
                "generated_output": output,
                "latency": latency
            }
        )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR,
        "distilled_student_outputs.parquet"
    )

    pd.DataFrame(results).to_parquet(OUTPUT_FILE)


    # --------------------------------------------------
    # MLflow Logging
    # --------------------------------------------------

    avg_latency = (
        sum(r["latency"] for r in results)
        / len(results)
    )

    mlflow.log_param(
        "model",
        "deepseek-coder-1.3b-instruct-mlx"
    )

    mlflow.log_param(
        "adapter_type",
        "vanilla_lora"
    )

    mlflow.log_metric(
        "total_samples",
        len(results)
    )

    mlflow.log_metric(
        "avg_latency",
        avg_latency
    )

    mlflow.log_artifact(OUTPUT_FILE)


print("\n===== DISTILLED INFERENCE COMPLETE =====")

print(f"\nSaved to:\n{OUTPUT_FILE}")