import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import warnings

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL",
)

import json
import time
import re
import ast
import requests
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from datasets import load_dataset

from configs.experiment_config import get_args

from utils.mlflow_logger import (
    start_run,
    log_params,
    log_metrics,
    log_artifact,
    set_tags,
    end_run,
)

# =========================================================
# CONFIG
# =========================================================

args = get_args()

MODEL_NAME = args.model_name
MODEL_PATH = args.model_path

DATASET_NAME = args.dataset_name
DATASET_SPLIT = args.dataset_split

TEMPERATURE = args.temperature
MAX_TOKENS = args.max_tokens
REQUEST_TIMEOUT = args.request_timeout

EXPERIMENT_NAME = args.experiment_name
EVALUATION_TYPE = args.evaluation_type

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

OUTPUT_PATH = (
    OUTPUT_DIR
    / f"mbpp_train_teacher_dataset_{TIMESTAMP}.parquet"
)

NUM_SAMPLES = args.num_samples

LLAMA_URL = "http://localhost:8080/completion"

BATCH_SIZE = 25

# =========================================================
# METRICS
# =========================================================

metrics = {
    "processed_samples": 0,
    "accepted_samples": 0,
    "rejected_samples": 0,
    "syntax_failures": 0,
    "timeout_failures": 0,
    "request_failures": 0,
    "runtime_failures": 0,
}

# =========================================================
# LOAD DATASET
# =========================================================

dataset = load_dataset(DATASET_NAME)

problems = dataset[DATASET_SPLIT]

if NUM_SAMPLES > 0:
    problems = problems.select(range(NUM_SAMPLES))

# =========================================================
# FUNCTION SIGNATURE EXTRACTION
# =========================================================


def extract_function_name(test_list):

    first_test = test_list[0]

    match = re.search(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\(",
        first_test,
    )

    if match:
        return match.group(1)

    return None


# =========================================================
# PROMPT
# =========================================================


def build_prompt(problem_text, function_name):

    return f"""[INST]
You are an expert Python programmer.

STRICT REQUIREMENTS:
- Return ONLY valid Python code
- Return ONLY ONE function
- Include all required imports
- Do NOT explain anything
- Use EXACTLY this function name:
{function_name}

Problem:
{problem_text}
[/INST]
"""


# =========================================================
# CALL LLM
# =========================================================


def call_llm(prompt):

    response = requests.post(
        LLAMA_URL,
        json={
            "prompt": prompt,
            "n_predict": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stop": [
                "C++ Solution:",
                "Java Solution:",
                "C# Solution:",
                "Ruby Solution:",
                "PHP Solution:",
                "Swift Solution:",
                "JavaScript Solution:",
                "[/PYTHON]",
                "[TESTS]",
                "Explanation:",
            ],
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()["content"]


# =========================================================
# CLEAN CODE
# =========================================================


def clean_code(code):

    wrappers = [
        "```python",
        "```",
        "[PYTHON]",
        "[/PYTHON]",
    ]

    for wrapper in wrappers:
        code = code.replace(wrapper, "")

    code = code.strip()

    stop_markers = [
        "\nComment:",
        "\n###",
        "\nC++",
        "\nJava",
        "\nRuby",
        "\nPHP",
        "\nSwift",
        "\nJavaScript",
        "\nExplanation:",
        "\nExample:",
        "\n[TESTS]",
    ]

    stop_positions = []

    for marker in stop_markers:

        idx = code.find(marker)

        if idx != -1:
            stop_positions.append(idx)

    if stop_positions:

        code = code[: min(stop_positions)]

    code = code.strip()

    lines = code.splitlines()

    candidate_code = []

    started = False

    for line in lines:

        stripped = line.strip()

        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
        ):
            candidate_code.append(line)

        if stripped.startswith("def "):
            started = True

        if started:
            candidate_code.append(line)

    candidate_code = "\n".join(candidate_code)

    candidate_lines = candidate_code.splitlines()

    for end_idx in range(
        len(candidate_lines),
        0,
        -1,
    ):

        partial_code = "\n".join(
            candidate_lines[:end_idx]
        )

        try:

            ast.parse(partial_code)

            return partial_code.strip()

        except Exception:

            continue

    return ""


# =========================================================
# VALIDATE CODE
# =========================================================


def validate_code(code):

    try:

        compile(code, "<string>", "exec")

        return True

    except Exception:

        return False


# =========================================================
# MAIN
# =========================================================


def main():

    run_name = (
        f"syntax_validation_"
        f"{MODEL_NAME}_"
        f"{DATASET_NAME}_"
        f"{DATASET_SPLIT}"
    )

    start_run(
        run_name=run_name,
        experiment_name=EXPERIMENT_NAME,
    )

    set_tags(
        {
            "stage": "syntax_validation",
            "dataset": DATASET_NAME,
            "split": DATASET_SPLIT,
            "pipeline": "error_aware_refinement",
        }
    )

    log_params(
        {
            "model_name": MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "dataset_split": DATASET_SPLIT,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "num_samples": NUM_SAMPLES,
        }
    )

    total_problems = len(problems)

    buffer = []

    writer = None

    start_time = time.time()

    print("=" * 70)
    print("SYNTAX VALIDATED TEACHER GENERATION")
    print("=" * 70)
    print(f"Total problems: {total_problems}")
    print("-" * 70)

    for idx, problem in enumerate(problems, start=1):

        function_name = extract_function_name(
            problem["test_list"]
        )

        prompt = build_prompt(
            problem["text"],
            function_name,
        )

        try:

            generation_start = time.time()

            output = call_llm(prompt)

            latency = time.time() - generation_start

            cleaned_output = clean_code(output)

            passed = validate_code(cleaned_output)

            result_dict = {
                "task_id": problem["task_id"],
                "prompt": problem["text"],
                "expected_function_name": function_name,
                "raw_output": output,
                "generated_code": cleaned_output,
                "latency": latency,
                "passed": passed,
            }

            metrics["processed_samples"] += 1

            if passed:

                metrics["accepted_samples"] += 1

                buffer.append(result_dict)

                print(
                    f"{idx}/{total_problems} --- PASSED --- "
                    f"{problem['task_id']}"
                )

            else:

                metrics["syntax_failures"] += 1
                metrics["rejected_samples"] += 1

                print(
                    f"{idx}/{total_problems} --- SYNTAX FAILED --- "
                    f"{problem['task_id']}"
                )

            if len(buffer) >= BATCH_SIZE:

                table = pa.Table.from_pandas(
                    pd.DataFrame(buffer)
                )

                if writer is None:

                    writer = pq.ParquetWriter(
                        OUTPUT_PATH,
                        table.schema,
                    )

                writer.write_table(table)

                buffer = []

            if idx % 10 == 0:

                current_metrics = {
                    **metrics,
                    "syntax_pass_rate": (
                        metrics["accepted_samples"]
                        / max(metrics["processed_samples"], 1)
                    ),
                }

                log_metrics(
                    current_metrics,
                    step=idx,
                )

        except requests.exceptions.Timeout:

            metrics["timeout_failures"] += 1
            metrics["rejected_samples"] += 1
            metrics["processed_samples"] += 1

            print(
                f"{idx}/{total_problems} --- TIMEOUT --- "
                f"{problem['task_id']}"
            )

        except Exception as e:

            metrics["runtime_failures"] += 1
            metrics["rejected_samples"] += 1
            metrics["processed_samples"] += 1

            print(
                f"{idx}/{total_problems} --- ERROR --- {e}"
            )

    if buffer:

        table = pa.Table.from_pandas(
            pd.DataFrame(buffer)
        )

        if writer is None:

            writer = pq.ParquetWriter(
                OUTPUT_PATH,
                table.schema,
            )

        writer.write_table(table)

    if writer:

        writer.close()

    end_time = time.time()

    final_metrics = {
        **metrics,
        "syntax_pass_rate": (
            metrics["accepted_samples"]
            / max(metrics["processed_samples"], 1)
        ),
        "total_runtime_minutes": (
            end_time - start_time
        )
        / 60,
    }

    log_metrics(final_metrics)

    log_artifact(str(OUTPUT_PATH))

    end_run()

    print("-" * 70)

    print(
        f"Syntax Pass Rate: "
        f"{final_metrics['syntax_pass_rate']:.4f}"
    )

    print(
        f"Total Runtime: "
        f"{final_metrics['total_runtime_minutes']:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()