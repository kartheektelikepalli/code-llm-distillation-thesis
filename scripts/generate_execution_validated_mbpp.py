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
import traceback
import time

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

DATASET_NAME = args.dataset_name
DATASET_SPLIT = args.dataset_split

EXPERIMENT_NAME = args.experiment_name

INPUT_PARQUET = args.input_parquet

OUTPUT_DIR = Path(
    "data/execution_validated_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

OUTPUT_PATH = (
    OUTPUT_DIR
    / f"execution_validated_mbpp_{TIMESTAMP}.parquet"
)

NUM_SAMPLES = args.num_samples

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_parquet(INPUT_PARQUET)

if NUM_SAMPLES > 0:
    df = df.head(NUM_SAMPLES)

dataset = load_dataset(DATASET_NAME)

mbpp_split = dataset[DATASET_SPLIT]

task_lookup = {
    item["task_id"]: item
    for item in mbpp_split
}

# =========================================================
# METRICS
# =========================================================

metrics = {
    "processed_samples": 0,
    "execution_passed": 0,
    "execution_failed": 0,
    "assertion_failures": 0,
    "runtime_failures": 0,
}

# =========================================================
# HELPER STRUCTURES
# =========================================================


class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __getitem__(self, idx):
        return [self.a, self.b][idx]

    def __repr__(self):
        return f"Pair({self.a}, {self.b})"


# =========================================================
# EXECUTION VALIDATION
# =========================================================


def validate_execution(code, tests):

    execution_namespace = {
        "Pair": Pair,
    }

    try:

        exec(code, execution_namespace)

        for test in tests:

            exec(test, execution_namespace)

        return True, None

    except AssertionError as e:

        return False, f"AssertionError: {str(e)}"

    except Exception:

        return False, traceback.format_exc()


# =========================================================
# MAIN
# =========================================================


def main():

    run_name = (
        f"execution_validation_"
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
            "stage": "execution_validation",
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
            "input_parquet": INPUT_PARQUET,
            "num_samples": NUM_SAMPLES,
        }
    )

    validated_rows = []

    total_samples = len(df)

    print("=" * 70)
    print("EXECUTION VALIDATION")
    print("=" * 70)
    print(f"Total samples: {total_samples}")
    print("-" * 70)

    for idx, row in df.iterrows():

        task_id = row["task_id"]

        generated_code = row["generated_code"]

        mbpp_item = task_lookup.get(task_id)

        if mbpp_item is None:

            print(
                f"{idx+1}/{total_samples} "
                f"--- TASK NOT FOUND --- "
                f"{task_id}"
            )

            continue

        tests = mbpp_item["test_list"]

        metrics["processed_samples"] += 1

        passed, error_message = validate_execution(
            generated_code,
            tests,
        )

        if passed:

            metrics["execution_passed"] += 1

            validated_rows.append(
                {
                    **row.to_dict(),
                    "execution_passed": True,
                    "execution_error": None,
                }
            )

            print(
                f"{idx+1}/{total_samples} "
                f"--- EXECUTION PASSED --- "
                f"{task_id}"
            )

        else:

            metrics["execution_failed"] += 1

            if (
                error_message
                and "AssertionError"
                in error_message
            ):

                metrics["assertion_failures"] += 1

            else:

                metrics["runtime_failures"] += 1

            print(
                f"{idx+1}/{total_samples} "
                f"--- EXECUTION FAILED --- "
                f"{task_id}"
            )

        if (idx + 1) % 10 == 0:

            current_metrics = {
                **metrics,
                "execution_pass_rate": (
                    metrics["execution_passed"]
                    / max(
                        metrics["processed_samples"],
                        1,
                    )
                ),
            }

            log_metrics(
                current_metrics,
                step=idx + 1,
            )

    # =====================================================
    # SAVE FINAL DATASET
    # =====================================================

    validated_df = pd.DataFrame(validated_rows)

    table = pa.Table.from_pandas(validated_df)

    pq.write_table(
        table,
        OUTPUT_PATH,
    )

    # =====================================================
    # FINAL METRICS
    # =====================================================

    final_metrics = {
        **metrics,
        "execution_pass_rate": (
            metrics["execution_passed"]
            / max(
                metrics["processed_samples"],
                1,
            )
        ),
    }

    log_metrics(final_metrics)

    log_artifact(str(OUTPUT_PATH))

    end_run()

    print("-" * 70)

    print(
        f"Execution Pass Rate: "
        f"{final_metrics['execution_pass_rate']:.4f}"
    )

    print("=" * 70)

    print("\nFinal Metrics:\n")

    print(
        json.dumps(
            final_metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()