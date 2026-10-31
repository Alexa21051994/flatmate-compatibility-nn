"""Neural network and baseline models for pair compatibility.

All models are wrapped so that they (i) train on both orderings of each pair
(swap augmentation) and (ii) predict the mean over both orderings, which makes
f(A,B) == f(B,A) by construction.
"""
import copy

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .features import pair_features

Y_SCALE = 100.0


def both_orders(PA, PB, y=None, engineered=True):
    X = np.vstack([pair_features(PA, PB, engineered), pair_features(PB, PA, engineered)])
    return X if y is None else (X, np.concatenate([y, y]))


class SymmetricRegressor:
    """Wrap any sklearn regressor with swap augmentation + symmetric inference."""

    def __init__(self, estimator, scale=True, engineered=True):
        self.estimator, self.scale, self.engineered = estimator, scale, engineered

    def fit(self, PA, PB, y):
        X, yy = both_orders(PA, PB, y, self.engineered)
        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        self.estimator.fit(self.scaler_.transform(X) if self.scale else X, yy)
        return self

    def predict(self, PA, PB):
        n = len(PA)
        X = both_orders(PA, PB, None, self.engineered)
        if self.scale:
            X = self.scaler_.transform(X)
        p = self.estimator.predict(X)
        return np.clip(0.5 * (p[:n] + p[n:]), 0, 100)


class PairMLP:
    """Feed-forward ANN trained with early stopping on a *person-disjoint* validation set.

    Each epoch is one pass of Adam over the (swap-augmented) training pairs. The
    weights with the best validation MAE are restored at the end.
    """

    def __init__(self, hidden=(128, 64), alpha=1e-3, lr=1e-3, batch_size=256,
                 activation="relu", max_epochs=150, patience=15, seed=0):
        self.hidden, self.alpha, self.lr, self.batch_size = tuple(hidden), alpha, lr, batch_size
        self.activation, self.max_epochs, self.patience, self.seed = activation, max_epochs, patience, seed

    def fit(self, PA, PB, y, PAv, PBv, yv, verbose=False):
        X, yy = both_orders(PA, PB, y)
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)
        net = MLPRegressor(hidden_layer_sizes=self.hidden, activation=self.activation,
                           solver="adam", alpha=self.alpha, learning_rate_init=self.lr,
                           batch_size=self.batch_size, random_state=self.seed, shuffle=True)
        best, best_net, wait = np.inf, None, 0
        self.history_ = {"train_mae": [], "val_mae": []}
        for epoch in range(self.max_epochs):
            net.partial_fit(X, yy / Y_SCALE)
            self.net_ = net
            tr = np.mean(np.abs(net.predict(X) * Y_SCALE - yy))
            va = np.mean(np.abs(self.predict(PAv, PBv) - yv))
            self.history_["train_mae"].append(tr)
            self.history_["val_mae"].append(va)
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch:3d}  train MAE {tr:.3f}  val MAE {va:.3f}")
            if va < best - 1e-4:
                best, best_net, wait = va, copy.deepcopy(net), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_ = best_net
        self.best_val_mae_, self.best_epoch_ = best, int(np.argmin(self.history_["val_mae"])) + 1
        return self

    def predict(self, PA, PB):
        n = len(PA)
        X = self.scaler_.transform(both_orders(PA, PB))
        p = self.net_.predict(X) * Y_SCALE
        return np.clip(0.5 * (p[:n] + p[n:]), 0, 100)


class MLPEnsemble:
    """Average of several PairMLPs trained from different random initialisations."""

    def __init__(self, members):
        self.members = members

    def predict(self, PA, PB):
        return np.mean([m.predict(PA, PB) for m in self.members], axis=0)
