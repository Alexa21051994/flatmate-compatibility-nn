"""Metrics shared by training and reporting."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)

from .config import band


def regression_metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p),
        "Spearman": float(np.nan_to_num(spearmanr(y, p).statistic)) if np.ptp(p) > 0 else 0.0,
        "Band accuracy": accuracy_score(band(y), band(p)),
        "Band macro-F1": f1_score(band(y), band(p), average="macro"),
    }
