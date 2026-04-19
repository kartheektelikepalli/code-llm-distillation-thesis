import os
os.environ["PYTHONWARNINGS"] = "ignore"

import re
import time
import signal
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlflow

# =========================
# CONFIG
# =========================
INPUT_PATH = "data/teacher_mbpp.parquet"
OUTPUT_PATH = "data/final_teacher_dataset.parquet"

MAX_WORKERS = 4
TIMEOUT_SECONDS = 5

# =========================
# TIMEOUT
# =========================
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# =========================
# CLEAN CODE
# =========================
def clean_code(code):
    # remove markdown fences
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)

    # remove JS-style comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)

    # extract first Python function block
    lines = code.split("\n")
    cleaned = []
    inside = False
    indent = None

    for line in lines:
        if line.strip().startswith("def "):
            inside = True
            indent = len(line) - len(line.lstrip())
            cleaned.append(line)
            continue

        if inside:
            cur = len(line) - len(line.lstrip())
            if line.strip() == "":
                cleaned.append(line)
                continue
            if cur > indent:
                cleaned.append(line)
            else:
                break

    if not cleaned:
        return None

    return "\n".join(cleaned)

# =========================
# SAFE INPUTS
# =========================
def get_safe_inputs(prompt):
    p = prompt.lower()

    if "string" in p:
        return [["abc"], ["aabb"]]
    elif "list" in p or "array" in p:
        return [[[1, 2, 3]], [[0]]]
    elif "integer" in p or "number" in p:
        return [[1], [0]]

    return [[1]]

# =========================
# EXECUTION
# =========================
def run_code(code, inp):
    try:
        code = clean_code(code)
        if code is None:
            return False

        # extract function name
        match = re.search(r"def\s+(\w+)\(", code)
        if not match:
            return False

        func_name = match.group(1)

        # append safe call
        call_code = f"\n__result = {func_name}(*{repr(inp)})"
        full_code = code + "\n" + call_code

        local_env = {}

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

        exec(full_code, {"__builtins__": __builtins__}, local_env)

        signal.alarm(0)

        return True

    except TimeoutException:
        return False
    except Exception:
        return False

# =========================
# PROCESS ROW
# =========================
def process_row(row):
    prompt = row["prompt"]
    code = row["output"]

    inputs = get_safe_inputs(prompt)

    # keep if at least one run succeeds
    for inp in inputs:
        if run_code(code, inp):
            return row

    return None

# =========================
# MAIN
# =========================
def main():
    start = time.time()

    df = pd.read_parquet(INPUT_PATH)

    # 🔥 BASE FILTER (critical)
    df = df[df["passed"] == True]

    total = len(df)

    results = []
    passed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]

        for i, future in enumerate(as_completed(futures)):
            res = future.result()

            if res is not None:
                results.append(res)
                passed += 1

            if (i + 1) % 20 == 0 or (i + 1) == total:
                print(f"{i+1}/{total} processed | Kept: {passed}")

    final_df = pd.DataFrame(results)
    final_df.to_parquet(OUTPUT_PATH)

    duration = (time.time() - start) / 60
    keep_ratio = passed / total if total > 0 else 0

    mlflow.set_experiment("teacher_dataset_refinement")

    with mlflow.start_run(run_name="sane_filter"):
        mlflow.log_metric("input_samples", total)
        mlflow.log_metric("kept_samples", passed)
        mlflow.log_metric("keep_ratio", keep_ratio)
        mlflow.log_metric("time_minutes", duration)

    print("\n====================")
    print(f"Kept: {passed}/{total} ({keep_ratio:.2f})")
    print(f"Time Taken: {duration:.2f} mins")
    print("====================")

if __name__ == "__main__":
    main()