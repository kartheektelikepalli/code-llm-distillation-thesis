import os
import ast
import time
import pandas as pd
import mlflow

from tqdm import tqdm
from datasets import load_dataset

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = (
    "lora/artifacts/"
    "deepseek-coder-1.3b-instruct-mlx"
)

ADAPTER_PATH = (
    "optimization/lora_adapters_pass5"
)

OUTPUT_DIR = (
    "optimization/artifacts/"
    "distilled_student_pass5"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "distilled_student_pass5.parquet"
)


# =========================================================
# CONFIG
# =========================================================

NUM_SAMPLES = 5

MAX_TOKENS = 128


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_experiment(
    "distilled_student_pass5_generation"
)


# =========================================================
# LOAD MODEL
# =========================================================

model, tokenizer = load(
    MODEL_PATH,
    adapter_path=ADAPTER_PATH
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
# EXTRACTION
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
            NUM_SAMPLES
        ):

            generation_start = (
                time.time()
            )

            sampler = make_sampler(
                temp=0.8
            )

            output = generate(

                model,
                tokenizer,

                prompt=prompt,

                max_tokens=MAX_TOKENS,

                sampler=sampler
            )

            latency = (
                time.time()
                - generation_start
            )

            generated_code = (
                extract_python_code(
                    output
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

                    "generated_output":
                        output,

                    "generated_code":
                        generated_code,

                    "latency":
                        latency,

                    "syntax_passed":
                        syntax_passed
                }
            )


    # =====================================================
    # SAVE
    # =====================================================

    df = pd.DataFrame(results)

    df.to_parquet(
        OUTPUT_FILE
    )


    # =====================================================
    # METRICS
    # =====================================================

    total_generations = len(df)

    syntax_pass_rate = (
        syntax_pass_count
        / total_generations
    ) * 100

    avg_latency = (
        df["latency"].mean()
    )

    mlflow.log_metric(
        "total_generations",
        total_generations
    )

    mlflow.log_metric(
        "syntax_pass_rate",
        syntax_pass_rate
    )

    mlflow.log_metric(
        "avg_latency",
        avg_latency
    )

    mlflow.log_artifact(
        OUTPUT_FILE
    )


# =========================================================
# SUMMARY
# =========================================================

runtime_minutes = (
    time.time()
    - start_time
) / 60


print(
    "\n===== DISTILLED PASS@5 GENERATION COMPLETE ====="
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
    f"{OUTPUT_FILE}"
)