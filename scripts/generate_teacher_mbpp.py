"""
MBPP Teacher Dataset Generation
with self-consistency and robustness improvements
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import sys
from io import StringIO
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# -----------------------------
# GLOBAL STATE
# -----------------------------

seen_solutions = set()


# -----------------------------
# LLM GENERATION
# -----------------------------

def generate_with_llama(prompt_text, func_signature):

    url = "http://127.0.0.1:8080/completion"

    formatted_prompt = f"""[INST] Write a Python function with the following signature:

{func_signature}

Problem:
{prompt_text}

Return only valid Python code implementing this function. [/INST]
"""

    payload = {
        "prompt": formatted_prompt,
        "temperature": 0.2,
        "top_p": 0.95,
        "n_predict": 512
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()

        if "choices" in result:
            return result["choices"][0]["text"]
        else:
            return result.get("content", "")

    except requests.exceptions.RequestException:
        return ""


# -----------------------------
# CLEAN MODEL OUTPUT
# -----------------------------

def clean_solution(code):

    return (
        code
        .replace("```python", "")
        .replace("```", "")
        .replace("[PYTHON]", "")
        .replace("[/PYTHON]", "")
        .replace("[TESTS]", "")
        .replace("[/TESTS]", "")
        .strip()
    )


# -----------------------------
# EXECUTE TESTS
# -----------------------------

def execute_test(test_code):

    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        namespace = {}
        exec(test_code, namespace)

        sys.stdout = old_stdout
        return True

    except Exception:
        sys.stdout = old_stdout
        return False


# -----------------------------
# PROCESS SINGLE PROBLEM
# -----------------------------

def process_problem(problem_data):

    idx, total, problem = problem_data

    task_id = problem["task_id"]
    prompt = problem["text"]
    test_list = problem["test_list"]

    func_signature = problem["code"].split("\n")[0]

    # self-consistency attempts
    for attempt in range(3):

        teacher_solution = generate_with_llama(prompt, func_signature)
        teacher_solution = clean_solution(teacher_solution)

        if not teacher_solution:
            continue

        complete_code = teacher_solution + "\n"

        test_setup_code = problem.get("test_setup_code", "")
        if test_setup_code:
            complete_code = test_setup_code + "\n" + complete_code

        passed = True

        for test in test_list:
            if not execute_test(complete_code + test):
                passed = False
                break

        if passed:

            solution_hash = hash(teacher_solution)

            if solution_hash in seen_solutions:
                return idx, None, False

            seen_solutions.add(solution_hash)

            result = {
                "task_id": task_id,
                "prompt": prompt,
                "teacher_solution": teacher_solution,
                "tests": test_list
            }

            return idx, result, True

    return idx, None, False


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():

    print("=" * 70)
    print("MBPP Teacher Solution Generation")
    print("=" * 70)

    dataset = load_dataset("mbpp")

    full_dataset = concatenate_datasets([
        dataset["train"],
        dataset["validation"],
        dataset["test"]
    ])

    total_problems = len(full_dataset)
    passed_count = 0

    print(f"Total problems: {total_problems}")
    print("-" * 70)

    output_path = Path("data/teacher_outputs/mbpp_teacher.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # RESUME SUPPORT
    # -----------------------------

    completed_tasks = set()

    if output_path.exists():
        try:
            existing = pd.read_parquet(output_path)
            completed_tasks = set(existing["task_id"].tolist())
            print(f"Resuming run. Skipping {len(completed_tasks)} completed tasks.")
        except Exception:
            completed_tasks = set()

    # -----------------------------
    # BUFFER WRITES
    # -----------------------------

    buffer = []
    flush_every = 20

    print_lock = Lock()

    max_workers = 3

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {}

        for i, problem in enumerate(full_dataset, 1):

            if problem["task_id"] in completed_tasks:
                continue

            future = executor.submit(process_problem, (i, total_problems, problem))
            futures[future] = i

        for future in as_completed(futures):

            idx, result_dict, passed = future.result()

            if passed:

                buffer.append(result_dict)
                passed_count += 1

                if len(buffer) >= flush_every:

                    df = pd.DataFrame(buffer)

                    if output_path.exists():
                        df.to_parquet(output_path, engine="pyarrow", append=True)
                    else:
                        df.to_parquet(output_path, engine="pyarrow")

                    buffer.clear()

            with print_lock:
                if passed:
                    print(f"{idx}/{total_problems} --- Passed")
                else:
                    print(f"{idx}/{total_problems} --- Failed")

    # -----------------------------
    # FINAL FLUSH
    # -----------------------------

    if buffer:

        df = pd.DataFrame(buffer)

        if output_path.exists():
            df.to_parquet(output_path, engine="pyarrow", append=True)
        else:
            df.to_parquet(output_path, engine="pyarrow")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total problems:  {total_problems}")
    print(f"Passed problems: {passed_count}")
    print(f"Pass rate:       {(passed_count / total_problems) * 100:.2f}%")
    print(f"Saved to:        {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()