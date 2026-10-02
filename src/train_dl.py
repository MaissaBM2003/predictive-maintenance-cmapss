import argparse
import copy

import mlflow
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import mae, nasa_score, rmse
from src.models import build_model


def predict(model, X, device, batch_size=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "gru", "cnn1d", "transformer"])
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    cfg, cap = config["deep_learning"], config["preprocessing"]["rul_cap"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(f"{config['data']['processed_dir']}/windows.npz")
    X_train, y_train = d["X_train"], d["y_train"] / cap
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=cfg["batch_size"], shuffle=True)

    model = build_model(args.model, X_train.shape[2], X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"dl_{args.model}"):
        mlflow.log_params({"model": args.model, "window_size": X_train.shape[1],
                           "n_params": sum(p.numel() for p in model.parameters()), **cfg})

        best_rmse, best_state, best_epoch, wait = float("inf"), None, 0, 0
        for epoch in range(cfg["max_epochs"]):
            model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)

            train_rmse = float(np.sqrt(total / len(X_train)) * cap)
            val_rmse = rmse(y_val, predict(model, X_val, device) * cap)
            scheduler.step(val_rmse)
            mlflow.log_metrics({"train_rmse": train_rmse, "val_rmse_epoch": val_rmse,
                                "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
            print(f"Epoch {epoch:3d} | train RMSE {train_rmse:.2f} | val RMSE {val_rmse:.2f}")

            if val_rmse < best_rmse:
                best_rmse, best_epoch, wait = val_rmse, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    print("Early stopping")
                    break

        model.load_state_dict(best_state)
        p_val = np.clip(predict(model, X_val, device) * cap, 0, None)
        p_test = np.clip(predict(model, X_test, device) * cap, 0, None)

        mlflow.log_metrics({
            "best_epoch": best_epoch,
            "val_rmse": rmse(y_val, p_val),
            "val_mae": mae(y_val, p_val),
            "test_rmse": rmse(y_test, p_test),
            "test_mae": mae(y_test, p_test),
            "test_nasa_score": nasa_score(y_test, p_test),
        })
        torch.save(best_state, "model_state.pt")
        mlflow.log_artifact("model_state.pt", artifact_path="model")
        print(f"{args.model} | Val RMSE {rmse(y_val, p_val):.2f} | Test RMSE {rmse(y_test, p_test):.2f}")


if __name__ == "__main__":
    main()