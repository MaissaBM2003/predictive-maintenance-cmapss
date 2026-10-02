import pytest
import torch
from src.models import build_model


@pytest.mark.parametrize("name", ["lstm", "gru", "cnn1d", "transformer"])
def test_output_shape(name):
    model = build_model(name, n_features=14, window=30)
    out = model(torch.randn(8, 30, 14))
    assert out.shape == (8,)