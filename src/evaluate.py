"""Metrics and grouped cross-validation, shared by all model notebooks.

Every model is scored the same way:
  - R2 on log price
  - median absolute percentage error of the PRICE (we undo the log with exp)
  - mean absolute error in rupees
"""
import numpy as np
import pandas as pd

from split import cv_folds


def price_metrics(log_true, log_pred):
    """Scores for one set of predictions. Inputs are on the log-price scale."""
    log_true, log_pred = np.asarray(log_true), np.asarray(log_pred)
    r2 = 1 - np.sum((log_true - log_pred) ** 2) / np.sum((log_true - log_true.mean()) ** 2)
    price_true, price_pred = np.exp(log_true), np.exp(log_pred)
    return {
        "R2_log": r2,
        "median_pct_error": 100 * np.median(np.abs(price_pred / price_true - 1)),
        "MAE_rupees": np.mean(np.abs(price_pred - price_true)),
    }


def grouped_cv_scores(fit_predict, train):
    """Run 5-fold grouped CV (the saved folds) on the TRAINING set only.

    fit_predict(fit_rows, predict_rows) must fit a model on fit_rows and return
    log-price predictions for predict_rows. Writing every model this way lets the
    baselines, statsmodels OLS and sklearn pipelines share one CV loop.
    Returns one row of metrics per fold.
    """
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(cv_folds(train)):
        fit_rows, val_rows = train.iloc[fit_idx], train.iloc[val_idx]
        pred = fit_predict(fit_rows, val_rows)
        rows.append({"fold": fold + 1, **price_metrics(val_rows["log_price"], pred)})
    return pd.DataFrame(rows)


def summarise(folds, name):
    """Mean and standard deviation across folds, as one table row."""
    out = {"model": name}
    for col in ["R2_log", "median_pct_error", "MAE_rupees"]:
        out[col + "_mean"] = folds[col].mean()
        out[col + "_std"] = folds[col].std()
    return out


def pretty(table):
    """Format a table of summarise() rows as 'mean ± std' strings for reading."""
    t = pd.DataFrame(table).set_index("model")
    return pd.DataFrame({
        "R² (log price)": t["R2_log_mean"].map("{:.3f}".format) + " ± " + t["R2_log_std"].map("{:.3f}".format),
        "median % error": t["median_pct_error_mean"].map("{:.1f}".format) + " ± " + t["median_pct_error_std"].map("{:.1f}".format),
        "MAE (₹)": t["MAE_rupees_mean"].map("{:,.0f}".format) + " ± " + t["MAE_rupees_std"].map("{:,.0f}".format),
    })
