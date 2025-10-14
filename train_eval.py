#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# column names for the dataset
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

def find_fd001_dir(root):
    # check if files are in root or root/FD001
    d1 = root
    d2 = root / "FD001"
    for d in (d1, d2):
        if (d / "train_FD001.txt").exists() and (d / "test_FD001.txt").exists() and (d / "RUL_FD001.txt").exists():
            return d
    raise FileNotFoundError(f"Could not find FD001 files under {root}")

def read_fd001(root):
    root = find_fd001_dir(root)
    train_df = pd.read_csv(root / "train_FD001.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    test_df = pd.read_csv(root / "test_FD001.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    rul_test = pd.read_csv(root / "RUL_FD001.txt", sep=r"\s+", header=None)
    
    train_df.columns = COLS[:train_df.shape[1]]
    test_df.columns = COLS[:test_df.shape[1]]
    rul_test.columns = ["RUL"]
    return train_df, test_df, rul_test

def add_rul_train(df):
    # calculate RUL for each engine
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).astype(int)
    return out

def simple_features(df):
    df = df.sort_values(["unit", "cycle"]).copy()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    for c in sensor_cols:
        # rolling mean
        df[f"{c}_mean3"] = (
            df.groupby("unit")[c]
              .rolling(3, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )
        # difference
        df[f"{c}_d1"] = df.groupby("unit")[c].diff().fillna(0)
    return df

def unit_split(df, val_frac=0.2, seed=42):
    # split by unit not by rows
    units = df["unit"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(val_frac * len(units)))
    val_units = set(units[:n_val])
    train_units = df[~df["unit"].isin(val_units)].copy()
    val_units_df = df[df["unit"].isin(val_units)].copy()
    return train_units, val_units_df

def zscore_fit(df, cols):
    mu = df[cols].mean()
    sd = df[cols].std(ddof=0).replace(0, 1)
    return mu, sd

def zscore_apply(df, cols, mu, sd):
    out = df.copy()
    out[cols] = (out[cols] - mu) / sd
    return out

def make_matrix(df):
    feature_cols = ["op1", "op2", "op3"] + [c for c in df.columns if c.startswith("s")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["RUL"].to_numpy(dtype=np.float32)
    return X, y, feature_cols

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_and_eval(train_df, seed=42, val_frac=0.2):
    train_df = add_rul_train(train_df)
    train_df = simple_features(train_df)
    
    feature_cols_all = ["op1", "op2", "op3"] + [c for c in train_df.columns if c.startswith("s")]
    train_df = train_df[["unit", "cycle", "RUL"] + feature_cols_all]
    
    # split and normalize
    tr_units, va_units = unit_split(train_df, val_frac=val_frac, seed=seed)
    mu, sd = zscore_fit(tr_units, feature_cols_all)
    tr_units_n = zscore_apply(tr_units, feature_cols_all, mu, sd)
    va_units_n = zscore_apply(va_units, feature_cols_all, mu, sd)
    
    Xtr, ytr, _ = make_matrix(tr_units_n)
    Xva, yva, _ = make_matrix(va_units_n)
    
    # ridge baseline
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    p_ridge = ridge.predict(Xva)
    mae_r = float(mean_absolute_error(yva, p_ridge))
    rmse_r = rmse(yva, p_ridge)
    nasa_r = nasa_score(yva, p_ridge)
    
    # random forest
    rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    p_rf = rf.predict(Xva)
    mae_rf = float(mean_absolute_error(yva, p_rf))
    rmse_rf = rmse(yva, p_rf)
    nasa_rf = nasa_score(yva, p_rf)
    
    print(f"Ridge  | MAE={mae_r:.2f}  RMSE={rmse_r:.2f}  NASA={nasa_r:.2f}")
    print(f"RF     | MAE={mae_rf:.2f} RMSE={rmse_rf:.2f} NASA={nasa_rf:.2f}")
    
    # plot
    plt.figure()
    plt.scatter(yva, p_rf, s=8, alpha=0.6, label="RF preds")
    lo, hi = float(yva.min()), float(yva.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="ideal")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Validation: RF Predicted vs True RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_scatter.png")
    print("Saved plot: val_scatter.png")
    
    # save predictions
    flag = (p_rf < 20.0)
    out = pd.DataFrame({"true_rul": yva.astype(float),
                        "pred_rul_rf": p_rf.astype(float),
                        "flag_maint_<20": flag})
    out.to_csv("val_preds.csv", index=False)
    print("Saved predictions: val_preds.csv")
    
    with open("metrics.txt", "w") as f:
        f.write("Model,MAE,RMSE,NASA\n")
        f.write(f"Ridge,{mae_r:.4f},{rmse_r:.4f},{nasa_r:.4f}\n")
        f.write(f"RF,{mae_rf:.4f},{rmse_rf:.4f},{nasa_rf:.4f}\n")
    print("Saved metrics: metrics.txt")
    
    return {
        "ridge": {"MAE": mae_r, "RMSE": rmse_r, "NASA": nasa_r},
        "rf": {"MAE": mae_rf, "RMSE": rmse_rf, "NASA": nasa_rf},
    }

def main():
    parser = argparse.ArgumentParser(description="Turbofan RUL training")
    parser.add_argument("--data_dir", type=str, default="CMaps", 
                        help="folder with FD001 files")
    parser.add_argument("--val_frac", type=float, default=0.2, 
                        help="validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    
    data_root = Path(args.data_dir)
    train_df, test_df, rul_test = read_fd001(data_root)
    _ = train_and_eval(train_df, seed=args.seed, val_frac=args.val_frac)

if __name__ == "__main__":
    main()
