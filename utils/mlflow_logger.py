import warnings

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL",
)
import mlflow
import os
from datetime import datetime


DEFAULT_EXPERIMENT_NAME = "code_llm_distillation"


def setup_experiment(experiment_name=DEFAULT_EXPERIMENT_NAME):
    """
    Create or set MLflow experiment.
    """
    mlflow.set_experiment(experiment_name)


def start_run(
    run_name=None,
    experiment_name=DEFAULT_EXPERIMENT_NAME,
):
    """
    Start a new MLflow run.
    """

    setup_experiment(experiment_name)

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    mlflow.start_run(run_name=run_name)

    return run_name


def log_params(params: dict):
    """
    Log parameters dictionary.
    """

    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict, step=None):
    """
    Log metrics dictionary.
    """

    for key, value in metrics.items():

        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)


def log_artifact(file_path):
    """
    Log artifact if file exists.
    """

    if os.path.exists(file_path):
        mlflow.log_artifact(file_path)


def set_tags(tags: dict):
    """
    Set MLflow tags.
    """

    mlflow.set_tags(tags)


def end_run():
    """
    Safely end MLflow run.
    """

    mlflow.end_run()