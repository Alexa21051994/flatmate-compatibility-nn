"""Basic correctness tests. Run after training:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.config import DATA_RAW, MODELS_DIR, PERSON_FEATURES
from src.data_generation import compatibility_score
from src.features import pair_features, person_matrix

needs_model = pytest.mark.skipif(not (MODELS_DIR / "compatibility_ann.joblib").exists(), reason="train first")


def _people(n=200):
    return pd.read_csv(DATA_RAW / "people.csv").head(n)


def test_score_simulator_is_symmetric_and_bounded():
    p = _people()
    a, b = p.iloc[:100].reset_index(drop=True), p.iloc[100:].reset_index(drop=True)
    s1, s2 = compatibility_score(a, b), compatibility_score(b, a)
    assert np.allclose(s1, s2)
    assert s1.min() >= 0 and s1.max() <= 100


def test_pair_feature_shape():
    p = person_matrix(_people(10))
    assert pair_features(p, p).shape[0] == 10
    assert pair_features(p, p, engineered=False).shape[1] == 2 * len(PERSON_FEATURES)


@needs_model
def test_prediction_symmetry_and_range():
    from src.predict import load_model
    m, p = load_model(), person_matrix(_people(300))
    a, b = p[:150], p[150:]
    ab, ba = m.predict(a, b), m.predict(b, a)
    assert np.allclose(ab, ba, atol=1e-9)
    assert ab.min() >= 0 and ab.max() <= 100


@needs_model
def test_deal_breaker_scenarios_rank_sensibly():
    from src.predict import score_pair
    base = {"cleanliness": 4, "chore_sharing": 4, "smoker": 0, "smoke_tolerance": 2}
    twins = score_pair(base, dict(base))
    smoke = score_pair(base, {**base, "smoker": 1, "smoke_tolerance": 5})
    assert twins > smoke
