import numpy as np

STATS = ["mean", "std", "min", "max", "last", "slope"]


def window_features(X):
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    slope = (X * t[None, :, None]).sum(axis=1) / (t ** 2).sum()
    feats = [X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1), X[:, -1, :], slope]
    return np.concatenate(feats, axis=1)


def feature_names(sensors):
    return [f"{s}_{stat}" for stat in STATS for s in sensors]