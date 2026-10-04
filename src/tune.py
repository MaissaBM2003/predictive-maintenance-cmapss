import copy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import mae, nasa_score, rmse
from src.models import build_model
from src.train_dl import predict


def train_trial(params, data, cfg, cap, device, trial):
    X_train, y_train, X_val, y_val = data
    torch.manual_seed(cfg["seed"])
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train / cap)),
                        batch_size=params["batch_size"], shuffle=True)
    model = build_model(params["model"], X_train.shape[2], X_train.shape[1],
                        hidden=params["hidden"], layers=params["layers"],
                        dropout=params["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        val_rmse = rmse(y_val, predict(model, X_val, device) * cap)
        if val_rmse < best:
            best, best_state, wait = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best, best_state


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    cfg, cap = config["deep_learning"], config["preprocessing"]["rul_cap"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(f"{config['data']['processed_dir']}/windows.npz")
    data = (d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    best_holder = {"rmse": float("inf"), "state": None}

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="optuna_rnn"):

        def objective(trial):
            params = {
                "model": trial.suggest_categorical("model", ["lstm", "gru"]),
                "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
                "layers": trial.suggest_int("layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            }
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                try:
                    val_rmse, state = train_trial(params, data, cfg, cap, device, trial)
                except optuna.TrialPruned:
                    mlflow.set_tag("status", "pruned")
                    raise
                mlflow.log_metric("val_rmse", val_rmse)

            if val_rmse < best_holder["rmse"]:
                best_holder.update(rmse=val_rmse, state=state)
            return val_rmse

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        study.optimize(objective, n_trials=config["tuning"]["n_trials"])

        best = study.best_params
        model = build_model(best["model"], data[0].shape[2], data[0].shape[1],
                            hidden=best["hidden"], layers=best["layers"],
                            dropout=best["dropout"]).to(device)
        model.load_state_dict(best_holder["state"])

        p_val = np.clip(predict(model, d["X_val"], device) * cap, 0, None)
        p_test = np.clip(predict(model, d["X_test"], device) * cap, 0, None)

        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metrics({
            "val_rmse": rmse(d["y_val"], p_val),
            "test_rmse": rmse(d["y_test"], p_test),
            "test_mae": mae(d["y_test"], p_test),
            "test_nasa_score": nasa_score(d["y_test"], p_test),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        })
        mlflow.log_dict(best, "best_params.json")
        torch.save(best_holder["state"], "best_model_state.pt")
        mlflow.log_artifact("best_model_state.pt", artifact_path="model")

        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        mlflow.log_figure(ax.figure, "optimization_history.png")
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        mlflow.log_figure(ax.figure, "param_importances.png")
        plt.close("all")

        print("Meilleurs paramètres :", best)
        print(f"Val RMSE {rmse(d['y_val'], p_val):.2f} | Test RMSE {rmse(d['y_test'], p_test):.2f}")


if __name__ == "__main__":
    main()