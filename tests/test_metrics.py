import numpy as np
from src.metrics import mae, nasa_score, rmse


def test_rmse_mae():
    y_true, y_pred = np.array([10.0, 20.0]), np.array([13.0, 17.0])
    assert np.isclose(rmse(y_true, y_pred), 3.0)
    assert np.isclose(mae(y_true, y_pred), 3.0)


def test_nasa_perfect_and_asymmetric():
    y = np.array([50.0])
    assert nasa_score(y, y) == 0.0
    assert nasa_score(y, y + 10) > nasa_score(y, y - 10)