#!/usr/bin/env python3

import argparse
import os
import re
import shlex
import subprocess
import time

from click import parser
import mlflow
import mlx.core as mx

def build_command(args):
    command = [
        "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", args.data,
        "--batch-size", str(args.batch_size),
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--steps-per-report", str(args.steps_per_report),
        "--steps-per-eval", str(args.steps_per_eval),
        "--save-every", str(args.save_every),
        "--adapter-path", args.adapter_path,
        "--lora-rank", str(args.lora_rank),
    ]

    if args.grad_checkpoint:
        command.append("--grad-checkpoint")

    return command

def run_experiment(args):

    gc = "true" if args.grad_checkpoint else "false"

    run_name = (
        f"lora_r{args.lora_rank}_gc_{gc}"
    )

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("model", args.model)
        mlflow.log_param("dataset", args.data)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("iterations", args.iters)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param(
            "gradient_checkpoint",
            args.grad_checkpoint,
        )
        mlflow.log_param(
            "lora_rank",
            args.lora_rank,
        )

        command = build_command(args)

        print()
        print("=" * 80)
        print("Running command:")
        print(shlex.join(command))
        print("=" * 80)
        print()
        #
        # Reset MLX peak memory counters
        #
        mx.reset_peak_memory()

        start_time = time.perf_counter()

        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        print(result.stdout)

        if result.stderr:
            print(result.stderr)

        end_time = time.perf_counter()

        training_time = end_time - start_time

        peak_memory = 0.0
        output = result.stdout + "\n" + result.stderr
        match = re.search(
            r"Peak mem\s+([0-9.]+)\s+GB",
            output,
        )

        if match:
            peak_memory = float(match.group(1)) * 1024**3

        mlflow.log_metric(
            "training_time_seconds",
            training_time,
        )

        if isinstance(peak_memory, dict):

            for name, value in peak_memory.items():
                mlflow.log_metric(
                    f"peak_memory_{name}",
                    float(value),
                )

        else:

            mlflow.log_metric(
                "peak_memory_bytes",
                float(peak_memory),
            )
        mlflow.log_metric("peak_memory_gib", peak_memory / 1024**3)

        mlflow.log_metric(
            "return_code",
            result.returncode,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "LoRA training failed."
            )
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter-path", required=True)

    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
    )

    return parser.parse_args()

def main():

    parser.add_argument(
        "--experiment-name",
        default="LoRA Optimization",
    )

    mlflow.set_experiment(
        args.experiment_name
    )

    args = parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()