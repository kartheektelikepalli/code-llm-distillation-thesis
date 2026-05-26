import argparse
from datetime import datetime


def get_args():

    parser = argparse.ArgumentParser()

    # =========================
    # MODEL CONFIG
    # =========================

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/codellama-7b-instruct-f16.gguf",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="codellama_7b_f16",
    )

    # =========================
    # DATASET CONFIG
    # =========================

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mbpp",
    )

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
    )

    parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    )

    # =========================
    # GENERATION CONFIG
    # =========================

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--request_timeout",
        type=int,
        default=120,
    )

    # =========================
    # EXPERIMENT CONFIG
    # =========================

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="codellama7b_fp16_mbpp",
    )

    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="teacher_baseline",
    )

    parser.add_argument(
    "--input_parquet",
    type=str,
    default="",
    )

    # =========================
    # OUTPUT CONFIG
    # =========================

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/teacher_outputs",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser.add_argument(
        "--output_file",
        type=str,
        default=f"mbpp_train_teacher_dataset_{timestamp}.parquet",
    )

    return parser.parse_args()