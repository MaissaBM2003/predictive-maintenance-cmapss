import mlflow
import yaml

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_experiment(config["mlflow"]["experiment_name"])

with mlflow.start_run(run_name="test"):
    mlflow.log_param("window_size", config["preprocessing"]["window_size"])
    mlflow.log_metric("rmse", 20.5)

print("Run enregistré dans MLflow")