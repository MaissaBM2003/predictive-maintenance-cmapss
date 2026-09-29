import pandas as pd
from pathlib import Path

COLUMNS = (
    ["unit", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}" for i in range(1, 22)]
)

def load_cmapss(raw_dir: str, dataset: str = "FD001"):
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / f"train_{dataset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(raw_dir / f"test_{dataset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(raw_dir / f"RUL_{dataset}.txt", sep=r"\s+", header=None, names=["rul"])
    return train, test, rul