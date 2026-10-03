import argparse
import copy

import mlflow
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset

from src.anomaly import DenseAutoencoder, detection_metrics, plot_health
from src.features import window_features


def ae_scores(model, X, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.from_numpy(X[i:i + 1024]).to(device)
            scores.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores)


def train_autoencoder(X_healthy, X_val_healthy, cfg, device):
    model = DenseAutoencoder(X_healthy.shape[1], X_healthy.shape[2], cfg["hidden"], cfg["latent"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(TensorDataset(torch.from_numpy(X_healthy)), batch_size=cfg["batch_size"], shuffle=True)

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward()
            optimizer.step()

        val_loss = float(ae_scores(model, X_val_healthy, device).mean())
        mlflow.log_metric("val_recon_loss", val_loss, step=epoch)
        if val_loss < best:
            best, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="autoencoder", choices=["autoencoder", "isolation_forest"])
    args = parser.parse_args()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    cfg, cap = config["anomaly"], config["preprocessing"]["rul_cap"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    d = np.load(f"{config['data']['processed_dir']}/windows.npz")
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val, units_val = d["X_val"], d["y_val"], d["units_val"]
    healthy_train, healthy_val = X_train[y_train >= cap], X_val[y_val >= cap]

    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"anomaly_{args.method}"):
        mlflow.log_params({"method": args.method, "n_healthy_train": len(healthy_train), **cfg})

        if args.method == "autoencoder":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, state = train_autoencoder(healthy_train, healthy_val, cfg, device)
            scores_val = ae_scores(model, X_val, device)
            healthy_scores = ae_scores(model, healthy_val, device)
            torch.save(state, "anomaly_ae.pt")
            mlflow.log_artifact("anomaly_ae.pt", artifact_path="model")
        else:
            iso = IsolationForest(n_estimators=200, random_state=cfg["seed"])
            iso.fit(window_features(healthy_train))
            scores_val = -iso.score_samples(window_features(X_val))
            healthy_scores = -iso.score_samples(window_features(healthy_val))

        threshold = float(np.percentile(healthy_scores, cfg["threshold_percentile"]))
        metrics = detection_metrics(scores_val, units_val, y_val, threshold, cap, cfg["consecutive"])
        metrics["spearman_score_vs_rul"] = float(spearmanr(scores_val, y_val).correlation)
        metrics["threshold"] = threshold
        mlflow.log_metrics(metrics)
        mlflow.log_figure(plot_health(scores_val, units_val, threshold), "health_indicator.png")

        print(args.method, {k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()