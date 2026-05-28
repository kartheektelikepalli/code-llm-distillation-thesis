import os
import ast
import time
import requests
import pandas as pd
import mlflow

from tqdm import tqdm
from datasets import load_dataset

from config import *


# =========================================================
# LLAMA SERVER
# =========================================================

LLAMA_URL = "http://localhost:8081/completion"


# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = (
    "optimization/artifacts/"
    "student_baseline_pass5"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

OUTPUT_PARQUET = os.path.join(
    OUTPUT_DIR,
    f"student_baseline_pass5_{timestamp}.parquet"
)


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_experiment(
    "student_baseline_pass5_generation"
)


# =========================================================
# DATASET
# =========================================================

dataset = load_dataset(
    "mbpp"
)["test"]


# =========================================================
# PROMPT
# =========================================================

def build_prompt(problem):

    return f"""### Instruction:
Write a Python function to solve the following problem.

### Problem:
{problem}

### Response:
```python
"""


# =========================================================
# CODE EXTRACTION
# =========================================================

def extract_python_code(text):

    if "```python" in text:

        text = text.split(
            "```python"
        )[1]

    if "```" in text:

        text = text.split(
            "```"
        )[0]

    return text.strip()


# =========================================================
# GENERATION
# =========================================================

def generate_code(prompt):

    response = requests.post(
        LLAMA_URL,

        json={

            "prompt":
                prompt,

            "n_predict":
                MAX_NEW_TOKENS,

            "temperature":
                TEMPERATURE,

            "top_p":
                TOP_P,

            "stop": [
                "### Instruction:",
                "### Problem:",
                "### Response:",
                "```",
            ]
        },

        timeout=120,
    )

    response.raise_for_status()

    output = response.json()[
        "content"
    ]

    return output.strip()


# =========================================================
# MAIN LOOP
# =========================================================

results = []

syntax_pass_count = 0
syntax_fail_count = 0

start_time = time.time()


with mlflow.start_run():

    for sample in tqdm(dataset):

        prompt = build_prompt(
            sample["text"]
        )

        for sample_id in range(
            NUM_RETURN_SEQUENCES
        ):

            generation_start = (
                time.time()
            )

            try:

                raw_output = generate_code(
                    prompt
                )

                latency = (
                    time.time()
                    - generation_start
                )

                generated_code = (
                    extract_python_code(
                        raw_output
                    )
                )

                syntax_passed = True

                try:

                    ast.parse(
                        generated_code
                    )

                    syntax_pass_count += 1

                except:

                    syntax_passed = False

                    syntax_fail_count += 1

                results.append(
                    {
                        "task_id":
                            sample["task_id"],

                        "sample_id":
                            sample_id,

                        "prompt":
                            prompt,

                        "raw_output":
                            raw_output,

                        "generated_code":
                            generated_code,

                        "latency":
                            latency,

                        "syntax_passed":
                            syntax_passed
                    }
                )

            except Exception as e:

                syntax_fail_count += 1

                results.append(
                    {
                        "task_id":
                            sample["task_id"],

                        "sample_id":
                            sample_id,

                        "prompt":
                            prompt,

                        "raw_output":
                            "",

                        "generated_code":
                            "",

                        "latency":
                            -1,

                        "syntax_passed":
                            False,

                        "error":
                            str(e)
                    }
                )


    # =========================================================
    # SAVE
    # =========================================================

    df = pd.DataFrame(results)

    df.to_parquet(
        OUTPUT_PARQUET
    )


    # =========================================================
    # MLFLOW
    # =========================================================

    total_generations = len(df)

    syntax_pass_rate = (
        syntax_pass_count
        / max(total_generations, 1)
    ) * 100

    mlflow.log_param(
        "dataset",
        "mbpp_test"
    )

    mlflow.log_param(
        "num_return_sequences",
        NUM_RETURN_SEQUENCES
    )

    mlflow.log_param(
        "inference_backend",
        "llama.cpp"
    )

    mlflow.log_param(
        "student_model",
        "deepseek-coder-1.3b-instruct-f16.gguf"
    )

    mlflow.log_metric(
        "total_generations",
        total_generations
    )

    mlflow.log_metric(
        "syntax_passed",
        syntax_pass_count
    )

    mlflow.log_metric(
        "syntax_failed",
        syntax_fail_count
    )

    mlflow.log_metric(
        "syntax_pass_rate",
        syntax_pass_rate
    )

    mlflow.log_artifact(
        OUTPUT_PARQUET
    )


# =========================================================
# SUMMARY
# =========================================================

runtime_minutes = (
    time.time()
    - start_time
) / 60


print(
    "\n===== STUDENT BASELINE PASS@5 GENERATION COMPLETE ====="
)

print(
    f"\nTotal Generations : "
    f"{total_generations}"
)

print(
    f"Syntax Passed     : "
    f"{syntax_pass_count}"
)

print(
    f"Syntax Failed     : "
    f"{syntax_fail_count}"
)

print(
    f"Syntax Pass Rate  : "
    f"{syntax_pass_rate:.2f}%"
)

print(
    f"\nRuntime (minutes) : "
    f"{runtime_minutes:.2f}"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_PARQUET}"
)