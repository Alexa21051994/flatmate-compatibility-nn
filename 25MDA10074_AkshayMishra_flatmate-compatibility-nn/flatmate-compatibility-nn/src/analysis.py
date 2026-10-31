"""Post-training analysis: figures, permutation importance, symmetry check, learning curve.

Run after training:  python -m src.analysis
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import (BAND_NAMES, DATA_RAW, FIG_DIR, METRIC_DIR, MODELS_DIR, PERSON_FEATURES, SEED,
                     TRAIT_GROUPS, band)
from .evaluate import regression_metrics
from .features import load_split, person_matrix
from .models import PairMLP

BLUE, ORANGE, GREY, INK = "#2a6fdb", "#d9622b", "#9aa5b1", "#1c2430"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a95a3", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def main():
    people, pairs = pd.read_csv(DATA_RAW / "people.csv"), pd.read_csv(DATA_RAW / "pairs.csv")
    te = load_split("test", people, pairs)
    tr, va = load_split("train", people, pairs), load_split("val", people, pairs)
    bundle = joblib.load(MODELS_DIR / "compatibility_ann.joblib")
    ens = bundle["ensemble"]
    res = json.load(open(METRIC_DIR / "model_results.json"))
    P = np.load(METRIC_DIR / "test_predictions.npz")
    y, p = P["y"], P["ANN v6: + 5-seed ensemble (final)"]
    out = {}

    # --- Dataset figure: score distribution -----------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(pairs.score, bins=40, color=BLUE, edgecolor="white", linewidth=0.5)
    for e in (50, 70):
        ax.axvline(e, color=GREY, ls="--", lw=1)
    ax.text(25, ax.get_ylim()[1] * 0.92, "Low", color=INK); ax.text(58, ax.get_ylim()[1] * 0.92, "Moderate", color=INK)
    ax.text(80, ax.get_ylim()[1] * 0.92, "High", color=INK)
    ax.set_xlabel("Compatibility score (0-100)"); ax.set_ylabel("Number of pairs")
    save(fig, "fig_score_distribution.png")

    # --- Predicted vs actual -------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    hb = ax.hexbin(y, p, gridsize=40, cmap="Blues", mincnt=1, linewidths=0.2)
    ax.plot([0, 100], [0, 100], color=ORANGE, lw=1.5, label="Perfect prediction")
    ax.set_xlim(15, 100); ax.set_ylim(15, 100)
    ax.set_xlabel("Actual score"); ax.set_ylabel("Predicted score"); ax.legend(frameon=False, loc="upper left")
    fig.colorbar(hb, ax=ax, label="Pairs per cell", shrink=0.8)
    save(fig, "fig_pred_vs_actual.png")

    # --- Residuals -------------------------------------------------------------
    r = p - y
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(r, bins=50, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=ORANGE, lw=1.5)
    ax.set_xlabel("Prediction error (predicted - actual, points)"); ax.set_ylabel("Number of test pairs")
    save(fig, "fig_residuals.png")
    out["residual_mean"], out["residual_sd"] = float(r.mean()), float(r.std())
    out["within_5"], out["within_10"] = float((abs(r) <= 5).mean()), float((abs(r) <= 10).mean())

    # --- Confusion matrix -----------------------------------------------------
    cm = confusion_matrix(band(y), band(p))
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else INK, fontsize=9)
    ax.set_xticks(range(3), BAND_NAMES); ax.set_yticks(range(3), BAND_NAMES)
    ax.set_xlabel("Predicted band"); ax.set_ylabel("Actual band")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig_confusion.png")
    out["confusion"] = cm.tolist()

    # --- Per-band MAE ---------------------------------------------------------
    out["mae_by_band"] = {BAND_NAMES[k]: float(np.abs(r[band(y) == k]).mean()) for k in range(3)}
    out["n_by_band"] = {BAND_NAMES[k]: int((band(y) == k).sum()) for k in range(3)}

    # --- Model comparison -------------------------------------------------------
    names = ["Mean predictor", "Linear regression (raw A|B)", "Ridge (pair features)", "Random Forest",
             "Gradient Boosting", "ANN v6: + 5-seed ensemble (final)"]
    short = ["Mean", "Linear\n(raw A|B)", "Ridge\n(pair feat.)", "Random\nForest", "Gradient\nBoosting", "ANN\n(final)"]
    maes = [res[n]["test"]["MAE"] for n in names]
    floor = res["Noise ceiling (oracle)"]["test"]["MAE"]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    bars = ax.bar(short, maes, color=[GREY] * 5 + [BLUE], width=0.6)
    ax.axhline(floor, color=ORANGE, ls="--", lw=1.3, label=f"Noise floor ({floor:.2f})")
    for b_, v in zip(bars, maes):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=9, bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_ylabel("Test MAE (points, lower is better)"); ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_model_comparison.png")

    # --- ANN refinement ladder ---------------------------------------------------
    lad = [k for k in res if k.startswith("ANN v")]
    lab = [k.replace("ANN ", "").replace("+ ", "+").replace(" (final)", "") for k in lad]
    lad_mae = [res[k]["test"]["MAE"] for k in lad]
    floor = res["Noise ceiling (oracle)"]["test"]["MAE"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ys = np.arange(len(lad))[::-1]
    ax.barh(ys, lad_mae, color=[GREY] * (len(lad) - 1) + [BLUE], height=0.6)
    for yy, v in zip(ys, lad_mae):
        ax.text(v + 0.05, yy, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(floor, color=ORANGE, ls="--", lw=1.3)
    ax.text(floor + 0.03, ys[0] + 0.5, f"noise floor {floor:.2f}", color=ORANGE, fontsize=9, va="bottom")
    ax.set_yticks(ys, lab); ax.set_xlim(0, max(lad_mae) + 0.6); ax.set_ylim(-0.6, len(lad) - 0.1)
    ax.set_xlabel("Test MAE (points, lower is better)")
    save(fig, "fig_ann_ladder.png")

    # --- Training curve of final model's first member ------------------------------
    hist = json.load(open(METRIC_DIR / "training_history.json"))["history"][0]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(hist["train_mae"], color=GREY, lw=2, label="Training MAE")
    ax.plot(hist["val_mae"], color=BLUE, lw=2, label="Validation MAE (unseen people)")
    be = int(np.argmin(hist["val_mae"]))
    ax.scatter([be], [hist["val_mae"][be]], color=ORANGE, zorder=3, s=30)
    ax.annotate(f"best epoch {be + 1}", (be, hist["val_mae"][be]), textcoords="offset points", xytext=(10, 12), fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (points)"); ax.legend(frameon=False)
    save(fig, "fig_training_curve.png")

    # --- Permutation importance by trait group ------------------------------------
    rng = np.random.default_rng(SEED)
    ppl = people.set_index("person_id")
    tp = pairs[pairs.split == "test"].reset_index(drop=True)
    A = ppl.loc[tp.person_a].reset_index(drop=True)[PERSON_FEATURES]
    B = ppl.loc[tp.person_b].reset_index(drop=True)[PERSON_FEATURES]
    base_mae = np.abs(ens.predict(person_matrix(A), person_matrix(B)) - y).mean()
    imp = {}
    for g, cols in TRAIT_GROUPS.items():
        vals = []
        for _ in range(3):
            perm = rng.permutation(len(A))
            A2, B2 = A.copy(), B.copy()
            A2[cols] = A[cols].to_numpy()[perm]
            B2[cols] = B[cols].to_numpy()[rng.permutation(len(B))]
            vals.append(np.abs(ens.predict(person_matrix(A2), person_matrix(B2)) - y).mean() - base_mae)
        imp[g] = float(np.mean(vals))
    imp = dict(sorted(imp.items(), key=lambda t: t[1]))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(list(imp), list(imp.values()), color=BLUE, height=0.65)
    ax.set_xlabel("Increase in test MAE when the trait group is shuffled (points)")
    save(fig, "fig_importance.png")
    out["permutation_importance"] = imp

    # --- Symmetry check --------------------------------------------------------------
    out["max_symmetry_gap"] = float(np.abs(ens.predict(person_matrix(A), person_matrix(B))
                                           - ens.predict(person_matrix(B), person_matrix(A))).max())

    # --- Learning curve (fraction of training people) ------------------------------
    cfg = bundle["config"]
    ids_tr = people.loc[people.split == "train", "person_id"].to_numpy()
    tr_pairs = pairs[pairs.split == "train"]
    lc = []
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        keep = set(rng.choice(ids_tr, int(frac * len(ids_tr)), replace=False))
        sub = tr_pairs[tr_pairs.person_a.isin(keep) & tr_pairs.person_b.isin(keep)]
        sp = pd.concat([sub, pairs[pairs.split != "train"]])
        s_tr = load_split("train", people, sp)
        m = PairMLP(**cfg, max_epochs=120, patience=12, seed=SEED).fit(*s_tr[:3], *va[:3])
        lc.append({"train_people": len(keep), "train_pairs": len(sub), "test_MAE": float(np.abs(m.predict(*te[:2]) - y).mean())})
        print(lc[-1], flush=True)
    out["learning_curve"] = lc
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot([d["train_people"] for d in lc], [d["test_MAE"] for d in lc], color=BLUE, marker="o", lw=2)
    ax.axhline(res["Noise ceiling (oracle)"]["test"]["MAE"], color=ORANGE, ls="--", lw=1.3)
    ax.text(lc[-1]["train_people"], res["Noise ceiling (oracle)"]["test"]["MAE"] - 0.25, "noise floor", color=ORANGE, ha="right", fontsize=9)
    ax.set_xlabel("People in the training set"); ax.set_ylabel("Test MAE (points)")
    save(fig, "fig_learning_curve.png")

    # --- Sanity probes (scenario tests) -----------------------------------------------
    from .predict import score_pair
    base = {"chronotype": 3, "cleanliness": 4, "noise_level": 2, "noise_sensitivity": 3, "guest_frequency": 2,
            "guest_tolerance": 3, "smoke_tolerance": 2, "alcohol": 2, "kitchen_strictness": 2, "cooking_frequency": 3,
            "chore_sharing": 4, "time_at_home": 3, "extraversion": 3, "pet_tolerance": 3, "ac_preference": 3,
            "bill_responsibility": 4, "smoker": 0, "has_pet": 0, "occupation": 1, "diet": 2, "budget_k": 9.0}
    scen = {"Identical twins": (base, dict(base)),
            "Smoker with smoke-intolerant flatmate": (base, {**base, "smoker": 1, "smoke_tolerance": 5}),
            "Tidy vs messy": (base, {**base, "cleanliness": 1, "chore_sharing": 1}),
            "Early bird vs night owl": ({**base, "chronotype": 1}, {**base, "chronotype": 5}),
            "Pet owner vs pet-averse": ({**base, "has_pet": 1, "pet_tolerance": 5}, {**base, "pet_tolerance": 1}),
            "Budget Rs 6k vs Rs 20k": ({**base, "budget_k": 6.0}, {**base, "budget_k": 20.0})}
    out["scenarios"] = {k: round(score_pair(a, b, ens), 1) for k, (a, b) in scen.items()}
    json.dump(out, open(METRIC_DIR / "analysis.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
