import pandas as pd
from src.preprocessing import add_rul, make_windows, split_by_unit


def toy_data():
    return pd.DataFrame({
        "unit": [1] * 40 + [2] * 40,
        "cycle": list(range(1, 41)) * 2,
        "s_2": range(80),
    })


def test_rul_cap():
    df = add_rul(toy_data(), cap=25)
    assert df["rul"].max() == 25 and df["rul"].min() == 0


def test_windows_shape():
    X, y, units = make_windows(add_rul(toy_data(), 125), ["s_2"], 30)
    assert X.shape == (22, 30, 1) and len(y) == 22 and len(units) == 22


def test_no_unit_overlap():
    train, val = split_by_unit(add_rul(toy_data(), 125), 0.5, 42)
    assert set(train["unit"]).isdisjoint(set(val["unit"]))

