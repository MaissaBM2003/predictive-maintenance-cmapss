import numpy as np
import torch
from src.anomaly import DenseAutoencoder, first_alarm


def test_autoencoder_shape():
    model = DenseAutoencoder(window=30, n_features=14)
    x = torch.randn(8, 30, 14)
    assert model(x).shape == x.shape


def test_first_alarm_needs_consecutive():
    scores = np.array([0, 5, 0, 5, 5, 5, 0])
    assert first_alarm(scores, threshold=1, consecutive=3) == 5
    assert first_alarm(scores, threshold=10, consecutive=3) is None