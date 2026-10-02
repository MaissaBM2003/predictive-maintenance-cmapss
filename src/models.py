import torch
from torch import nn


class RNNRegressor(nn.Module):
    def __init__(self, n_features, cell="lstm", hidden=64, layers=2, dropout=0.2):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class CNN1DRegressor(nn.Module):
    def __init__(self, n_features, channels=64, kernel=5, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=kernel // 2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, n_features, window, d_model=64, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.encoder(self.proj(x) + self.pos)
        return self.head(h.mean(dim=1)).squeeze(-1)


def build_model(name, n_features, window):
    if name in ("lstm", "gru"):
        return RNNRegressor(n_features, cell=name)
    if name == "cnn1d":
        return CNN1DRegressor(n_features)
    if name == "transformer":
        return TransformerRegressor(n_features, window)
    raise ValueError(f"Modèle inconnu : {name}")