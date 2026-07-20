import os
import time
import pandas as pd
import mlflow

from optimization.datasets import load_dataset_split
from optimization.generator import generate_dataset
from optimization.prompts import build_mbpp_prompt
from optimization.code_utils import extract_python_code
from optimization.backends import MLXBackend


def run_generation(
    model_path,
    adapter_path,
    dataset_name,
    dataset_split,
    output_file,
    experiment_name,
    num_samples,
    max_tokens,
    temperature=0.8,
):

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True,
    )

    mlflow.set_experiment(
        experiment_name
    )

    backend = MLXBackend(
        model_path,
        adapter_path,
    )

    dataset = load_dataset_split(
        dataset_name,
        dataset_split,
    )
    start_time = time.time()

    with mlflow.start_run():

        results, syntax_pass_count, syntax_fail_count = generate_dataset(
            backend=backend,
            dataset=dataset,
            prompt_builder=build_mbpp_prompt,
            extract_python_code=extract_python_code,
            num_samples=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        df = pd.DataFrame(results)

        df.to_parquet(output_file)

        total_generations = len(df)

        syntax_pass_rate = (
            syntax_pass_count / total_generations
        ) * 100

        avg_latency = df["latency"].mean()

        mlflow.log_metric(
            "total_generations",
            total_generations,
        )

        mlflow.log_metric(
            "syntax_pass_rate",
            syntax_pass_rate,
        )

        mlflow.log_metric(
            "avg_latency",
            avg_latency,
        )

        mlflow.log_artifact(
            output_file
        )

    runtime_minutes = (
        time.time() - start_time
    ) / 60

    print("\n===== GENERATION COMPLETE =====")

    print(f"\nTotal Generations : {total_generations}")
    print(f"Syntax Passed     : {syntax_pass_count}")
    print(f"Syntax Failed     : {syntax_fail_count}")
    print(f"Syntax Pass Rate  : {syntax_pass_rate:.2f}%")
    print(f"\nRuntime (minutes) : {runtime_minutes:.2f}")
    print(f"\nSaved to:\n{output_file}")