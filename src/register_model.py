import argparse
import json

import mlflow
import numpy as np
import yaml
from mlflow import MlflowClient

from src.metrics import rmse
from src.pyfunc_model import RULModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", default="rul-predictor")
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    d = np.load(f"{config['data']['processed_dir']}/windows.npz")

    local_dir = mlflow.artifacts.download_artifacts(run_id=args.run_id)
    with open(f"{local_dir}/best_params.json") as f:
        params = json.load(f)
    params.update(n_features=int(d["X_train"].shape[2]), window=int(d["X_train"].shape[1]),
                  rul_cap=config["preprocessing"]["rul_cap"])
    with open("model_params.json", "w") as f:
        json.dump(params, f)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="register_best_rnn"):
        mlflow.set_tag("source_run_id", args.run_id)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=RULModel(),
            artifacts={"weights": f"{local_dir}/model/best_model_state.pt",
                       "params": "model_params.json"},
            code_paths=["src"],
            input_example=d["X_test"][:2],
            registered_model_name=args.name,
        )

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(args.name, "champion", version)

    loaded = mlflow.pyfunc.load_model(f"models:/{args.name}@champion")
    p_test = loaded.predict(d["X_test"])
    print(f"{args.name} version {version} -> alias champion | Test RMSE {rmse(d['y_test'], p_test):.2f}")


if __name__ == "__main__":
    main()