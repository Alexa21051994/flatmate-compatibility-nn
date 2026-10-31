"""Pair feature engineering.

A pair (A, B) becomes one row: [A | B | |A-B| | A*B | hobby overlap]. Because a
flatmate pair is unordered, training uses both orderings and inference averages
f(A,B) and f(B,A), so predictions are exactly symmetric.
"""
import numpy as np
import pandas as pd

from .config import HOBBIES, PERSON_FEATURES


def person_matrix(people: pd.DataFrame) -> np.ndarray:
    """People frame -> float matrix (budget log-transformed)."""
    m = people[PERSON_FEATURES].astype(float).copy()
    m["budget_k"] = np.log(m["budget_k"])
    return m.to_numpy()


_HOB_IDX = [PERSON_FEATURES.index("hobby_" + h) for h in HOBBIES]


def pair_features(PA: np.ndarray, PB: np.ndarray, engineered: bool = True) -> np.ndarray:
    """Build the model input for ordered pairs (rows of PA, PB)."""
    if not engineered:
        return np.hstack([PA, PB])
    shared = (PA[:, _HOB_IDX] * PB[:, _HOB_IDX]).sum(1, keepdims=True)
    union = np.clip(PA[:, _HOB_IDX] + PB[:, _HOB_IDX], 0, 1).sum(1, keepdims=True)
    jaccard = shared / np.maximum(union, 1)
    return np.hstack([PA, PB, np.abs(PA - PB), PA * PB, shared, jaccard])


def load_split(split: str, people: pd.DataFrame, pairs: pd.DataFrame):
    """Return (PA, PB, y, y_noise_free) for one split."""
    ppl = people.set_index("person_id")
    p = pairs[pairs.split == split].reset_index(drop=True)
    PA = person_matrix(ppl.loc[p.person_a].reset_index())
    PB = person_matrix(ppl.loc[p.person_b].reset_index())
    return PA, PB, p.score.to_numpy(float), p.score_noise_free.to_numpy(float)
