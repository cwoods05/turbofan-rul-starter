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
