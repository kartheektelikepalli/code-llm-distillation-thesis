import mlflow

mlflow.set_experiment("teacher_dataset_generation")

with mlflow.start_run(run_name="codellama_7b_fp16_final"):
    mlflow.log_param("model", "codellama-7b-fp16")
    mlflow.log_param("dataset", "MBPP")

    mlflow.log_metric("pass_at_1", 0.6483)
    mlflow.log_metric("passed", 625)
    mlflow.log_metric("total", 964)
    mlflow.log_metric("time_minutes", 386.46)