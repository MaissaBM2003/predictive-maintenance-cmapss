from pathlib import Path
import joblib
import numpy as np
import yaml
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from src.data_loader import load_cmapss
from src.validation import raw_schema


def add_rul(df, cap):
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df


def split_by_unit(df, val_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["unit"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def make_windows(df, features, window):
    X, y = [], []
    for _, d in df.groupby("unit"):
        values, rul = d[features].values, d["rul"].values
        for i in range(len(d) - window + 1):
            X.append(values[i:i + window])
            y.append(rul[i + window - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_test_windows(test, rul, features, window, cap):
    X = []
    for _, d in test.groupby("unit"):
        values = d[features].values
        if len(values) < window:
            pad = np.repeat(values[:1], window - len(values), axis=0)
            values = np.vstack([pad, values])
        X.append(values[-window:])
    y = rul["rul"].clip(upper=cap).values
    return np.array(X, dtype=np.float32), y.astype(np.float32)


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    data_cfg, p = config["data"], config["preprocessing"]

    train, test, rul = load_cmapss(data_cfg["raw_dir"], data_cfg["dataset"])
    train, test = raw_schema.validate(train), raw_schema.validate(test)

    train = train.drop(columns=p["drop_columns"])
    test = test.drop(columns=p["drop_columns"])
    features = [c for c in train.columns if c.startswith("s_")]

    train = add_rul(train, p["rul_cap"])
    train_df, val_df = split_by_unit(train, p["val_size"], p["seed"])

    scaler = MinMaxScaler().fit(train_df[features])
    for df in (train_df, val_df, test):
        df[features] = scaler.transform(df[features])

    w = p["window_size"]
    X_train, y_train = make_windows(train_df, features, w)
    X_val, y_val = make_windows(val_df, features, w)
    X_test, y_test = make_test_windows(test, rul, features, w, p["rul_cap"])

    out = Path(data_cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "windows.npz", X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
             features=np.array(features))
    joblib.dump(scaler, out / "scaler.joblib")

    print("Train :", X_train.shape, "| Val :", X_val.shape, "| Test :", X_test.shape)


if __name__ == "__main__":
    main()