from optimization.generate import run_generation


MODEL_PATH = (
    "lora/artifacts/"
    "deepseek-coder-1.3b-instruct-mlx"
)

ADAPTER_PATH = (
    "optimization/lora_adapters_pass5"
)

OUTPUT_FILE = (
    "optimization/artifacts/"
    "distilled_student_pass5/"
    "distilled_student_pass5.parquet"
)

EXPERIMENT_NAME = (
    "distilled_student_pass5_generation"
)

NUM_SAMPLES = 5

MAX_TOKENS = 128

TEMPERATURE = 0.8

run_generation(
    model_path=MODEL_PATH,
    adapter_path=ADAPTER_PATH,
    dataset_name="mbpp",
    dataset_split="test",
    output_file=OUTPUT_FILE,
    experiment_name=EXPERIMENT_NAME,
    num_samples=NUM_SAMPLES,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)