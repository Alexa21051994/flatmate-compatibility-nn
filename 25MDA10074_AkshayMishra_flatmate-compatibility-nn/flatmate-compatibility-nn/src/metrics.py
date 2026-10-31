"""Evaluation metrics shared by training, evaluation and tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

from .model import bands_of


def regression_report(y_true, y_pred, y_noiseless=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rep = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
        "Within_5": float(np.mean(np.abs(y_true - y_pred) <= 5)),
        "Within_10": float(np.mean(np.abs(y_true - y_pred) <= 10)),
        "Band_accuracy": float(np.mean(bands_of(y_true) == bands_of(y_pred))),
        "Band_macroF1": float(f1_score(bands_of(y_true), bands_of(y_pred), average="macro")),
    }
    if y_noiseless is not None:
        rep["MAE_vs_noiseless"] = mean_absolute_error(y_noiseless, y_pred)
        rep["R2_vs_noiseless"] = r2_score(y_noiseless, y_pred)
    return rep


def bootstrap_ci(y_true, y_pred, metric, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for ``metric(y_true, y_pred)`` over test pairs."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = [metric(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
