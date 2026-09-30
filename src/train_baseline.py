import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from src.features import window_features
from src.metrics import mae, nasa_score, rmse

MODELS = {
    "linear_regression": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
}


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    d = np.load(f"{config['data']['processed_dir']}/windows.npz")
    X_train, y_train, groups = window_features(d["X_train"]), d["y_train"], d["units_train"]
    X_val, y_val = window_features(d["X_val"]), d["y_val"]
    X_test, y_test = window_features(d["X_test"]), d["y_test"]

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    for name, make_model in MODELS.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            cv_scores = []
            for tr_idx, va_idx in GroupKFold(n_splits=5).split(X_train, y_train, groups):
                m = make_model().fit(X_train[tr_idx], y_train[tr_idx])
                cv_scores.append(rmse(y_train[va_idx], m.predict(X_train[va_idx])))

            model = make_model().fit(X_train, y_train)
            p_val = np.clip(model.predict(X_val), 0, None)
            p_test = np.clip(model.predict(X_test), 0, None)

            mlflow.log_params({"model": name, "window_size": config["preprocessing"]["window_size"],
                               "n_features": X_train.shape[1]})
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({
                "cv_rmse_mean": float(np.mean(cv_scores)),
                "cv_rmse_std": float(np.std(cv_scores)),
                "val_rmse": rmse(y_val, p_val),
                "val_mae": mae(y_val, p_val),
                "test_rmse": rmse(y_test, p_test),
                "test_mae": mae(y_test, p_test),
                "test_nasa_score": nasa_score(y_test, p_test),
            })
            mlflow.sklearn.log_model(model, "model")

            print(f"{name:18s} | CV RMSE {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f} "
                  f"| Val RMSE {rmse(y_val, p_val):.2f} | Test RMSE {rmse(y_test, p_test):.2f}")


if __name__ == "__main__":
    main()