"""Neural-network ensemble wrapper used for training, evaluation and the app.

The model is a seed-ensemble of scikit-learn ``MLPRegressor`` networks sharing one
preprocessing pipeline. The ensemble mean is the prediction; the ensemble standard
deviation is a cheap uncertainty signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import build_features, categorical_columns, validate_profile


def make_preprocessor(X: pd.DataFrame, feature_set: str) -> ColumnTransformer:
    cat = [c for c in categorical_columns(feature_set) if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer(
        [("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]
    )


def make_mlp(params: dict, seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation=params["activation"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        batch_size=params["batch_size"],
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        max_iter=300,
        random_state=seed,
    )


def band_of(score: float) -> str:
    lo, hi = C.BAND_EDGES
    return C.BAND_NAMES[0] if score < lo else (C.BAND_NAMES[1] if score < hi else C.BAND_NAMES[2])


def bands_of(scores: np.ndarray) -> np.ndarray:
    return np.digitize(scores, C.BAND_EDGES)  # 0 = Low, 1 = Medium, 2 = High


@dataclass
class FlatmateModel:
    """Preprocessor + ensemble of MLPs. Predicts a 0-100 compatibility score for a pair."""

    feature_set: str
    params: dict
    seeds: tuple = (0, 1, 2, 3, 4)
    preprocessor: ColumnTransformer | None = None
    networks: list = field(default_factory=list)
    y_mean: float = 0.0
    y_std: float = 1.0

    # --- training -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FlatmateModel":
        self.preprocessor = make_preprocessor(X, self.feature_set)
        Z = self.preprocessor.fit_transform(X)
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y))
        yz = (np.asarray(y) - self.y_mean) / self.y_std
        self.networks = [make_mlp(self.params, s).fit(Z, yz) for s in self.seeds]
        return self

    # --- inference ----------------------------------------------------------
    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.preprocessor.transform(X)
        out = np.stack([net.predict(Z) * self.y_std + self.y_mean for net in self.networks])
        return np.clip(out, 0.0, 100.0)

    def predict_features(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_members(X).mean(axis=0)

    def predict_pairs(self, a: pd.DataFrame, b: pd.DataFrame):
        """Return (mean score, ensemble std) for aligned profile DataFrames."""
        members = self.predict_members(build_features(a, b, self.feature_set))
        return members.mean(axis=0), members.std(axis=0)

    def predict_one(self, profile_a: dict, profile_b: dict) -> dict:
        a = pd.DataFrame([validate_profile(profile_a)])
        b = pd.DataFrame([validate_profile(profile_b)])
        mean, std = self.predict_pairs(a, b)
        score = float(mean[0])
        spread = float(std[0])
        return {
            "score": score,
            "uncertainty": spread,
            "low": max(0.0, score - 2 * spread - C.LABEL_NOISE_SD),
            "high": min(100.0, score + 2 * spread + C.LABEL_NOISE_SD),
            "band": band_of(score),
        }
