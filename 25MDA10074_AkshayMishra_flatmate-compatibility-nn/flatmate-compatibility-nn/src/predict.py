"""Inference helpers: score a pair, explain friction points, rank candidates.

No external services or LLMs are used; explanations come from the trained network
itself via counterfactual "what if the two were aligned on this trait group?" probes.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import BAND_NAMES, MODELS_DIR, PERSON_FEATURES, TRAIT_GROUPS, band
from .features import person_matrix

DEFAULTS = {c: 3 for c in PERSON_FEATURES}
DEFAULTS.update({"smoker": 0, "has_pet": 0, "occupation": 0, "diet": 2, "budget_k": 9.0})
DEFAULTS.update({c: 0 for c in PERSON_FEATURES if c.startswith("hobby_")})


@lru_cache(maxsize=1)
def load_model(path=None):
    return joblib.load(path or MODELS_DIR / "compatibility_ann.joblib")["ensemble"]


def _frame(person: dict) -> pd.DataFrame:
    row = {**DEFAULTS, **person}
    return pd.DataFrame([row])[PERSON_FEATURES]


def score_pair(person_a: dict, person_b: dict, model=None) -> float:
    model = model or load_model()
    return float(model.predict(person_matrix(_frame(person_a)), person_matrix(_frame(person_b)))[0])


def explain_pair(person_a: dict, person_b: dict, model=None, top_k=4):
    """Return the trait groups whose misalignment costs the most compatibility points.

    For each group, copy A's answers onto B (and B's onto A), re-score, and average the gain.
    A large positive gain means that group is a friction point for this pair.
    """
    model = model or load_model()
    A, B = _frame(person_a), _frame(person_b)
    base = float(model.predict(person_matrix(A), person_matrix(B))[0])
    out = []
    for name, cols in TRAIT_GROUPS.items():
        B2, A2 = B.copy(), A.copy()
        B2[cols], A2[cols] = A[cols].to_numpy(), B[cols].to_numpy()
        gain = 0.5 * (float(model.predict(person_matrix(A), person_matrix(B2))[0]) - base
                      + float(model.predict(person_matrix(A2), person_matrix(B))[0]) - base)
        out.append((name, gain))
    out.sort(key=lambda t: -t[1])
    return base, [(n, g) for n, g in out[:top_k] if g > 0.3], [(n, g) for n, g in out[::-1][:2] if g <= 0.3]


def rank_candidates(seeker: dict, candidates: pd.DataFrame, model=None, top_n=5) -> pd.DataFrame:
    """Rank a candidate pool (rows with the survey columns) for one seeker."""
    model = model or load_model()
    P = person_matrix(candidates.reindex(columns=PERSON_FEATURES))
    S = np.repeat(person_matrix(_frame(seeker)), len(P), axis=0)
    res = candidates.copy()
    res["compatibility"] = model.predict(S, P)
    res["band"] = [BAND_NAMES[i] for i in band(res["compatibility"])]
    return res.sort_values("compatibility", ascending=False).head(top_n)
