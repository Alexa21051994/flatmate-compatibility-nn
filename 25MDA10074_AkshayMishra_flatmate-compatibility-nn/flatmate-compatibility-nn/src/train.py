"""Full training pipeline: baselines -> ablation -> hyperparameter search -> final ensemble.

Run:  python -m src.train            (about 15-25 minutes on 2 CPU cores)
      python -m src.train --quick    (smoke test, ~2 minutes)

Model selection uses ONLY the validation split (people disjoint from training).
The test split is evaluated once at the end.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from .config import DATA_RAW, METRIC_DIR, MODELS_DIR, SEED
from .evaluate import regression_metrics
from .models import MLPEnsemble, PairMLP, SymmetricRegressor
from .data_generation import main as generate_data


class Mean:
    def fit(self, PA, PB, y):
        self.m = y.mean()
        return self

    def predict(self, PA, PB):
        return np.full(len(PA), self.m)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_config(rng):
    return dict(
        hidden=[(64,), (128,), (128, 64), (256, 128), (256, 128, 64), (128, 128, 64), (64, 32)][rng.integers(7)],
        alpha=float(10 ** rng.uniform(-4, -0.5)),
        lr=float(10 ** rng.uniform(-3.7, -2.3)),
        batch_size=int([128, 256, 512][rng.integers(3)]),
        activation=["relu", "tanh"][rng.integers(2)],
    )


def main(quick=False):
    from .features import load_split
    if not (DATA_RAW / "pairs.csv").exists():
        generate_data()
    people, pairs = pd.read_csv(DATA_RAW / "people.csv"), pd.read_csv(DATA_RAW / "pairs.csv")
    tr, va, te = (load_split(s, people, pairs) for s in ("train", "val", "test"))
    log(f"train {len(tr[2])} | val {len(va[2])} | test {len(te[2])} pairs")
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    max_ep = 25 if quick else 150

    results, preds = {}, {}

    def record(name, model, extra=None):
        pv, pt = model.predict(*va[:2]), model.predict(*te[:2])
        results[name] = {"val": regression_metrics(va[2], pv), "test": regression_metrics(te[2], pt), **(extra or {})}
        preds[name] = pt
        log(f"{name:34s} val MAE {results[name]['val']['MAE']:.3f} | test MAE {results[name]['test']['MAE']:.3f} "
            f"| test R2 {results[name]['test']['R2']:.4f}")

    # ---------------- 1. Baselines ----------------
    record("Mean predictor", Mean().fit(*tr[:3]))
    record("Linear regression (raw A|B)", SymmetricRegressor(Ridge(alpha=1e-3), engineered=False).fit(*tr[:3]))
    record("Ridge (pair features)", SymmetricRegressor(Ridge(alpha=1.0)).fit(*tr[:3]))
    record("Random Forest", SymmetricRegressor(
        RandomForestRegressor(n_estimators=60 if quick else 200, min_samples_leaf=3, max_features=0.4,
                              n_jobs=-1, random_state=SEED), scale=False).fit(*tr[:3]))
    record("Gradient Boosting", SymmetricRegressor(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=SEED), scale=False).fit(*tr[:3]))

    # ---------------- 2. ANN refinement ladder (ablation) ----------------
    base_cfg = dict(hidden=(128, 64), alpha=1e-3, lr=1e-3, batch_size=256, activation="relu")
    # v1: reference-style MLP, raw [A|B] features, no swap augmentation, sklearn early stopping
    from .features import pair_features
    from sklearn.preprocessing import StandardScaler

    class PlainANN:
        def fit(self, PA, PB, y):
            X = pair_features(PA, PB, engineered=False)
            self.s = StandardScaler().fit(X)
            self.n = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=max_ep * 3,
                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                  random_state=SEED).fit(self.s.transform(X), y / 100)
            return self

        def predict(self, PA, PB):
            return np.clip(self.n.predict(self.s.transform(pair_features(PA, PB, False))) * 100, 0, 100)

    record("ANN v1: plain MLP on [A|B]", PlainANN().fit(*tr[:3]))
    record("ANN v2: + swap aug + symmetry", _v2(tr, max_ep))
    record("ANN v3: + pair features", _v3(tr, max_ep))
    m4 = PairMLP(**base_cfg, max_epochs=max_ep, seed=SEED).fit(*tr[:3], *va[:3])
    record("ANN v4: + person-disjoint stopping", m4, {"best_epoch": m4.best_epoch_})

    # ---------------- 3. Random search ----------------
    n_trials = 4 if quick else 24
    trials = []
    for t in range(n_trials):
        cfg = sample_config(rng)
        t0 = time.time()
        m = PairMLP(**cfg, max_epochs=max_ep, patience=10, seed=SEED).fit(*tr[:3], *va[:3])
        trials.append({**{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                       "val_MAE": m.best_val_mae_, "best_epoch": m.best_epoch_})
        log(f"trial {t + 1:2d}/{n_trials} val MAE {m.best_val_mae_:.3f} ({time.time() - t0:.0f}s) {cfg}")
    trials_df = pd.DataFrame(trials).sort_values("val_MAE").reset_index(drop=True)
    trials_df.to_csv(METRIC_DIR / "hyperparameter_search.csv", index=False)
    best = trials_df.iloc[0]
    best_cfg = dict(hidden=tuple(best.hidden), alpha=float(best.alpha), lr=float(best.lr),
                    batch_size=int(best.batch_size), activation=best.activation)
    log(f"best config: {best_cfg}")
    m5 = PairMLP(**best_cfg, max_epochs=max_ep * 2, patience=15, seed=SEED).fit(*tr[:3], *va[:3])
    record("ANN v5: + hyperparameter search", m5, {"best_epoch": m5.best_epoch_})

    # ---------------- 4. Final: seed ensemble ----------------
    members = [m5] + [PairMLP(**best_cfg, max_epochs=max_ep * 2, patience=15, seed=SEED + k).fit(*tr[:3], *va[:3])
                      for k in range(1, 3 if quick else 5)]
    ens = MLPEnsemble(members)
    record("ANN v6: + 5-seed ensemble (final)", ens)

    # oracle ceiling: the noise-free score vs the noisy label
    results["Noise ceiling (oracle)"] = {"test": regression_metrics(te[2], te[3]), "val": regression_metrics(va[2], va[3])}

    # ---------------- 5. Save ----------------
    joblib.dump({"ensemble": ens, "config": best_cfg}, MODELS_DIR / "compatibility_ann.joblib")
    json.dump(results, open(METRIC_DIR / "model_results.json", "w"), indent=2, default=float)
    json.dump({"history": [m.history_ for m in members], "best_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_cfg.items()}},
              open(METRIC_DIR / "training_history.json", "w"), default=float)
    np.savez(METRIC_DIR / "test_predictions.npz", y=te[2], y_clean=te[3], **{k: v for k, v in preds.items()})
    rows = [{"Model": k, **{f"test_{m}": v for m, v in r["test"].items()}, **{f"val_{m}": v for m, v in r["val"].items()}}
            for k, r in results.items()]
    pd.DataFrame(rows).to_csv(METRIC_DIR / "model_comparison.csv", index=False)
    log("saved model + metrics")


def _v2(tr, max_ep):
    est = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=max_ep * 3, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
    return _Scaled(est, engineered=False).fit(*tr[:3])


def _v3(tr, max_ep):
    est = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=max_ep * 3, early_stopping=True,
                       validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
    return _Scaled(est, engineered=True).fit(*tr[:3])


class _Scaled(SymmetricRegressor):
    """SymmetricRegressor whose target is divided by 100 (MLP trains better on [0, 1])."""

    def fit(self, PA, PB, y):
        return super().fit(PA, PB, y / 100)

    def predict(self, PA, PB):
        n = len(PA)
        from .models import both_orders
        X = self.scaler_.transform(both_orders(PA, PB, None, self.engineered))
        p = self.estimator.predict(X) * 100
        return np.clip(0.5 * (p[:n] + p[n:]), 0, 100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
