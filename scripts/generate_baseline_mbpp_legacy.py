import pandas as pd
import requests
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# CONFIG
LLAMA_URL = "http://localhost:8080/completion"
DATA_PATH = "data/raw/mbpp.jsonl"
OUTPUT_PATH = Path("data/teacher_mbpp.parquet")

BATCH_SIZE = 25

def call_llm(prompt):
    response = requests.post(
        LLAMA_URL,
        json={
            "prompt": prompt,
            "n_predict": 256,
            "temperature": 0.2,
        },
        timeout=120,
    )
    return response.json()["content"]


def load_mbpp(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(eval(line))  # assuming MBPP format
    return data


def main():
    problems = load_mbpp(DATA_PATH)
    total_problems = len(problems)

    passed_count = 0
    buffer = []
    writer = None

    start_time = time.time()

    print("=" * 70)
    print("MBPP CodeLlama Teacher Generation")
    print("=" * 70)
    print(f"Total problems: {total_problems}")
    print("-" * 70)

    for idx, problem in enumerate(problems, start=1):
        prompt = problem["text"]

        try:
            output = call_llm(prompt)
        except Exception as e:
            print(f"{idx}/{total_problems} --- ERROR --- {e}")
            continue

        # TODO: replace with your actual evaluation logic
        passed = "def" in output  # placeholder condition

        result_dict = {
            "task_id": problem["task_id"],
            "prompt": prompt,
            "output": output,
            "passed": passed,
        }

        if passed:
            passed_count += 1
            buffer.append(result_dict)

            # 🔥 WRITE IN BATCHES
            if len(buffer) >= BATCH_SIZE:
                table = pa.Table.from_pandas(pd.DataFrame(buffer))

                if writer is None:
                    writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)

                writer.write_table(table)
                buffer = []

            print(f"{idx}/{total_problems} --- Passed --- {problem['task_id']}")
        else:
            print(f"{idx}/{total_problems} --- Failed --- {problem['task_id']}")

    # 🔥 FINAL FLUSH
    if buffer:
        table = pa.Table.from_pandas(pd.DataFrame(buffer))
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

    end_time = time.time()

    print("-" * 70)
    print(f"Pass@1: {passed_count}/{total_problems} = {passed_count/total_problems:.4f}")
    print(f"Total time: {(end_time - start_time)/60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()