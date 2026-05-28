# --------------------------------------------------
# Teacher Model
# --------------------------------------------------

TEACHER_MODEL_PATH = (
    "models/codellama-7b-instruct-f16.gguf"
)

TEACHER_EXPERIMENT_NAME = (
    "teacher_pass5_generation"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

DATASET_NAME = "mbpp"

DATASET_SPLIT = "train"


# --------------------------------------------------
# Generation
# --------------------------------------------------

MAX_NEW_TOKENS = 128

TEMPERATURE = 0.6

TOP_P = 0.95

NUM_RETURN_SEQUENCES = 5

CTX_SIZE = 4096


# --------------------------------------------------
# Paths
# --------------------------------------------------

OUTPUT_DIR = (
    "optimization/artifacts/"
    "teacher_pass5_outputs"
)