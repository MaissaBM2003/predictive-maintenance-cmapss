import numpy as np
from src.features import feature_names, window_features


def test_features_shape():
    X = np.random.rand(10, 30, 14).astype(np.float32)
    assert window_features(X).shape == (10, 84)
    assert len(feature_names([f"s_{i}" for i in range(14)])) == 84


def test_slope():
    X = np.arange(30, dtype=np.float32).reshape(1, 30, 1)   # valeur qui monte de 1 par cycle
    assert np.isclose(window_features(X)[0, -1], 1.0)