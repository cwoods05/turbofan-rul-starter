#!/usr/bin/env python3
"""
Turbofan RUL — Starter (CMAPSS FD001, scikit-learn only)

- Looks for data in "CMaps" by default, but also works with "CMaps/FD001".
- Beginner-friendly: rolling averages + first differences → Ridge + RandomForest.
- Saves: metrics.txt, val_scatter.png, val_preds.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# Headless plotting (works in WSL/remote servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Standard column names for CMAPSS FD001 (3 ops + 21 sensors)
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

def find_fd001_dir(root: Path) -> Path:
    """
    Accepts either:
      <root>/train_FD001.txt, test_FD001.txt, RUL_FD001.txt
    or
      <root>/FD001/train_FD001.txt, ...
    """
    d1 = root
    d2 = root / "FD001"
    for d in (d1, d2):
        if (d / "train_FD001.txt").exists() and (d / "test_FD001.txt").exists() and (d / "RUL_FD001.txt").exists():
            return d
    raise FileNotFoundError(
        f"Could not find FD001 files under {root}. "
        f"Expected train_FD001.txt, test_FD001.txt, RUL_FD001.txt either directly or in {root/'FD001'}."
    )

def read_fd001(root: Path):
    """
    Returns:
      train_df: columns [unit, cycle, op1..op3, s1..s21]
      test_df : same columns (no RUL)
      rul_test: DataFrame with single column 'RUL' (one per unit in test)
    """
    root = find_fd001_dir(root)
    train_df = pd.read_csv(root / "train_FD001.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    test_df  = pd.read_csv(root / "test_FD001.txt",  sep=r"\s+", header=None).dropna(axis=1,  how="all")
    rul_test = pd.read_csv(root / "RUL_FD001.txt",   sep=r"\s+", header=None)

    train_df.columns = COLS[:train_df.shape[1]]
    test_df.columns  = COLS[:test_df.shape[1]]
    rul_test.columns = ["RUL"]
    return train_df, test_df, rul_test

def add_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each engine (unit), compute RUL = max(cycle) - cycle.
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).astype(int)
    return out

def simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add easy features per sensor, per unit:
      - rolling mean over window=3
      - first difference (delta)
    Keeps things simple and fast for a starter project.
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    for c in sensor_cols:
        # Rolling mean (window 3)
        df[f"{c}_mean3"] = (
            df.groupby("unit")[c]
              .rolling(3, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
        # First difference
        df[f"{c}_d1"] = df.groupby("unit")[c].diff().fillna(0)
    return df

def unit_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """
    Split by engine 'unit', not by rows.
    This prevents leakage across the same physical engine.
    """
    units = df["unit"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(val_frac * len(units)))
    val_units = set(units[:n_val])
    train_units = df[~df["unit"].isin(val_units)].copy()
    val_units_df = df[df["unit"].isin(val_units)].copy()
    return train_units, val_units_df

def zscore_fit(df: pd.DataFrame, cols):
    mu = df[cols].mean()
    sd = df[cols].std(ddof=0).replace(0, 1)  # avoid divide-by-zero
    return mu, sd

def zscore_apply(df: pd.DataFrame, cols, mu, sd):
    out = df.copy()
    out[cols] = (out[cols] - mu) / sd
    return out

def make_matrix(df: pd.DataFrame):
    """
    Build X, y from a feature-augmented DataFrame:
      X = ops + all sensor-derived columns
      y = RUL
    """
    feature_cols = ["op1", "op2", "op3"] + [c for c in df.columns if c.startswith("s")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["RUL"].to_numpy(dtype=np.float32)
    return X, y, feature_cols

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM scoring (asymmetric): over/under estimation penalized differently.
    """
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Manual RMSE for compatibility with older scikit-learn (no squared=False arg).
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
