import matplotlib.pyplot as plt
import numpy as np
from torch import nn


class DenseAutoencoder(nn.Module):
    def __init__(self, window, n_features, hidden=128, latent=16):
        super().__init__()
        d = window * n_features
        self.window, self.n_features = window, n_features
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.decoder(self.encoder(x)).view(-1, self.window, self.n_features)


def first_alarm(scores, threshold, consecutive):
    run = 0
    for i, flag in enumerate(scores > threshold):
        run = run + 1 if flag else 0
        if run == consecutive:
            return i
    return None


def detection_metrics(scores, units, y_rul, threshold, cap, consecutive=3):
    lead, premature, missed = [], 0, 0
    engines = np.unique(units)
    for u in engines:
        idx = first_alarm(scores[units == u], threshold, consecutive)
        if idx is None:
            missed += 1
        elif y_rul[units == u][idx] >= cap:
            premature += 1
        else:
            lead.append(y_rul[units == u][idx])
    n = len(engines)
    return {
        "detection_rate": len(lead) / n,
        "premature_rate": premature / n,
        "missed_rate": missed / n,
        "mean_lead_time": float(np.mean(lead)) if lead else 0.0,
    }


def plot_health(scores, units, threshold, n_engines=4):
    fig, ax = plt.subplots(figsize=(10, 5))
    for u in np.unique(units)[:n_engines]:
        s = scores[units == u]
        ax.plot(range(len(s)), s, label=f"Moteur {u}")
    ax.axhline(threshold, color="red", linestyle="--", label="Seuil")
    ax.set_xlabel("Fenêtre (temps)")
    ax.set_ylabel("Score d'anomalie")
    ax.legend()
    return fig