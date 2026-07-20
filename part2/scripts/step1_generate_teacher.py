import os
import re
import ast
import time
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import mlflow

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from profiling.experiment_profiler import ExperimentProfiler

from config import *

from rich import traceback
from tqdm import tqdm
from datasets import load_dataset

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = Path(OUTPUT_DIR).resolve()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TIMESTAMP = time.strftime(
    "%Y%m%d_%H%M%S"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "teacher_baseline.parquet"
)


# =========================================================
# MLFLOW
# =========================================================

mlflow.set_experiment(
    TEACHER_EXPERIMENT_NAME
)

mlflow.log_param(
    "experiment_version",
    "baseline_v1"
)
# =========================================================
# LOAD DATASET
# =========================================================

dataset = load_dataset(
    DATASET_NAME
)

problems = dataset[
    DATASET_SPLIT
]

# =========================================================
# FUNCTION NAME EXTRACTION
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

def build_prompt(
    problem_text,
    function_name,
):

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
            "n_predict": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
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
        timeout=120,
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

        code = code.replace(
            wrapper,
            ""
        )

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

    candidate_code = "\n".join(
        candidate_code
    )

    candidate_lines = (
        candidate_code.splitlines()
    )

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

        compile(
            code,
            "<string>",
            "exec"
        )

        return True

    except Exception:

        return False


# =========================================================
# MAIN
# =========================================================

def main():

    results = []

    syntax_passed = 0
    syntax_failed = 0

    total_generations = 0

    start_time = time.time()

    mlflow.end_run()
    profiler = ExperimentProfiler(
    experiment_name="Teacher Baseline Generation"
    )
    profiler.start()
    with mlflow.start_run():

        # =========================================================
        # TAGS
        # =========================================================

        mlflow.set_tags({

            "stage": "teacher_generation",

            "project": "code_llm_distillation",

            "framework": "llama.cpp",

            "hardware": "Apple M5 Pro",

        })


        # =========================================================
        # PARAMETERS
        # =========================================================

        mlflow.log_params({

            "experiment_version": "baseline_v1",

            "dataset": DATASET_NAME,

            "dataset_split": DATASET_SPLIT,

            "teacher_model": "CodeLlama-7B-Instruct",

            "model_format": "GGUF FP16",

            "temperature": TEMPERATURE,

            "top_p": TOP_P,

            "max_new_tokens": MAX_NEW_TOKENS,

            "num_return_sequences": NUM_RETURN_SEQUENCES,

            "prompt_version": "v1",

        })



        for sample in tqdm(problems):

            function_name = (
                extract_function_name(
                    sample["test_list"]
                )
            )

            prompt = build_prompt(
                sample["text"],
                function_name,
            )

            for sample_id in range(
                NUM_RETURN_SEQUENCES
            ):

                generation_start = time.time()

                try:

                    raw_output = call_llm(
                        prompt
                    )

                    latency = (
                        time.time()
                        - generation_start
                    )

                    generated_code = (
                        clean_code(
                            raw_output
                        )
                    )

                    passed = validate_code(
                        generated_code
                    )

                    total_generations += 1

                    if passed:
                        syntax_passed += 1
                    else:
                        syntax_failed += 1

                    results.append(
                        {
                            "task_id": sample["task_id"],
                            "sample_id": sample_id,
                            "prompt": sample["text"],
                            "expected_function_name": function_name,
                            "raw_output": raw_output,
                            "generated_code": generated_code,
                            "latency": latency,
                            "syntax_passed": passed,
                        }
                    )

                except Exception as e:
                    import traceback

                    traceback.print_exc()

                    syntax_failed += 1

                    results.append(
                        {
                            "task_id": sample["task_id"],
                            "sample_id": sample_id,
                            "prompt": sample["text"],
                            "expected_function_name": function_name,
                            "raw_output": "",
                            "generated_code": "",
                            "latency": -1,
                            "syntax_passed": False,
                            "error": str(e),
                        }
                    )
                    

        # =========================================================
        # SAVE
        # =========================================================

        df = pd.DataFrame(results)

        table = pa.Table.from_pandas(df)

        pq.write_table(
            table,
            OUTPUT_PATH
        )

        # =========================================================
        # MLFLOW
        # =========================================================

        mlflow.log_param(
            "dataset_name",
            DATASET_NAME
        )

        mlflow.log_param(
            "dataset_split",
            DATASET_SPLIT
        )

        mlflow.log_metric(
            "temperature",
            TEMPERATURE
        )

        mlflow.log_metric(
            "top_p",
            TOP_P
        )

        mlflow.log_metric(
            "num_return_sequences",
            NUM_RETURN_SEQUENCES
        )

        mlflow.log_metric(
            "total_generations",
            total_generations
        )

        mlflow.log_metric(
            "syntax_passed",
            syntax_passed
        )

        mlflow.log_metric(
            "syntax_failed",
            syntax_failed
        )

        mlflow.log_metric(
            "syntax_pass_rate",
            (
                syntax_passed
                / max(total_generations, 1)
            )
            * 100
        )

        mlflow.log_artifact(
            str(OUTPUT_PATH)
        )

    total_time = (
        time.time() - start_time
    ) / 60
    profiler.stop()
    profiler.log_to_mlflow()

    print("\n===== PASS@5 TEACHER GENERATION COMPLETE =====")

    print(
        f"\nTotal Generations : {total_generations}"
    )

    print(
        f"Syntax Passed     : {syntax_passed}"
    )

    print(
        f"Syntax Failed     : {syntax_failed}"
    )

    print(
        f"Syntax Pass Rate  : "
        f"{(syntax_passed / max(total_generations, 1)) * 100:.2f}%"
    )

    print(
        f"\nTotal Runtime     : {total_time:.2f} minutes"
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":

    main()
    print("Teacher baseline generated successfully.")