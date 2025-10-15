# Turbofan Engine Remaining Useful Life (RUL) Prediction

This project implements a simple machine learning pipeline for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA CMAPSS dataset (FD001 subset). It provides baseline models (Ridge Regression and Random Forest) along with basic feature engineering, normalization, evaluation, and visualization.

## Dataset

The code expects the FD001 data files from the NASA CMAPSS dataset:

```train_FD001.txt```
```test_FD001.txt```
```RUL_FD001.txt```

These can be placed either directly in the specified data directory or inside a subfolder named FD001.

## Features

* **Automatic dataset detection and loading**
* **Rolling mean and first-order difference** feature generation
* **Unit-based train/validation splitting**
* **Z-score normalization**
* Baseline models:
  * Ridge Regression
  * Random Forest Regressor
* Metrics:
  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
  * NASA Scoring Function
* Visualization and outputs:
  * Scatter plot of predicted vs. true RUL (```val_scatter.png```)
  * Validation predictions (```val_preds.csv```)
  * Model metrics summary (```metrics.txt```)

## Usage

Run the training and evaluation script:
```source venv/bin/activate```
```python3 main.py --data_dir CMaps```

## Outputs

After running, you will find:

* ```val_scatter.png``` — scatter plot comparing true vs predicted RUL
* ```val_preds.csv``` — validation predictions and maintenance flags 
* ```metrics.txt``` — summary of evaluation metrics for both models

## Reference

* [NASA Prognostics Data Repository — CMAPSS Turbofan Engine Degradation Simulation Data Set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
