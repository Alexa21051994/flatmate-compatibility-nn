# Flatmate Compatibility Prediction using an Artificial Neural Network

**Submitted by:** Akshay Mishra (UID: 25MDA10000074)
**Programme:** MBA (Data Science and AI), Chandigarh University
**Submitted to:** Er. Aaqib Rashid Mir
**GitHub:** [Alexa21051994](https://github.com/Alexa21051994)

An end-to-end machine-learning project that estimates how well two people would live together
(a **0-100 compatibility score**, plus a Low / Moderate / High band) from their lifestyle survey answers.
A feed-forward neural network is the core model; it is compared with linear, Random Forest and
Gradient Boosting baselines. A small Flask web app lets you enter two profiles and see the score
and the likely friction points.

**No API keys, no LLMs, no external services.** Everything runs locally with scikit-learn.

> **Important - the data is simulated.** No public dataset of real flatmate outcomes exists, so
> `src/data_generation.py` creates 1,800 synthetic people and 34,000 pairs whose scores come from a documented,
> noisy "friction" model. The network never sees those rules, only (person A, person B, score), but results
> describe how well it learns *this* simulated world, not real-life flatmate success.
> `docs/SURVEY_GUIDE.md` explains how to retrain on real survey data.

## Results (held-out test set: 5,000 pairs of people never seen in training)

| Model | MAE (points) | RMSE | R² | Band accuracy |
|---|---:|---:|---:|---:|
| Mean predictor | 9.76 | 12.01 | -0.006 | 55.8% |
| Linear regression (raw A\|B) | 7.53 | 9.65 | 0.351 | 62.0% |
| Ridge (pair features) | 3.52 | 4.41 | 0.864 | 83.7% |
| Random Forest | 3.80 | 4.78 | 0.841 | 81.8% |
| Gradient Boosting | 3.40 | 4.26 | 0.874 | 83.9% |
| **ANN (final, 5-seed ensemble)** | **3.32** | **4.16** | **0.879** | **84.2%** |
| Noise floor (oracle: true score vs noisy label) | 3.25 | 4.05 | 0.885 | 84.4% |

The final ANN is within 0.08 points of the best error any model could achieve on this data.
The gap to Gradient Boosting (0.07 MAE) is small; the full ablation is in `reports/metrics/`.

## How the network was refined

| Step | Test MAE |
|---|---:|
| v1 plain MLP on [A\|B] | 4.40 |
| v2 + swap augmentation & symmetric inference | 3.61 |
| v3 + pair features (\|A-B\|, A*B, hobby overlap) | 4.16 |
| v4 + early stopping on a person-disjoint validation set | 3.55 |
| v5 + random search over 24 configurations | 3.35 |
| v6 + 5-seed ensemble (final) | 3.32 |

Key design choices: person-disjoint train/val/test splits (no person appears in two splits, which prevents leakage),
order-invariant predictions (`f(A,B) == f(B,A)` exactly), and model selection on validation data only.
v3 is worse than v2 because the pair-feature MLP overfits under scikit-learn's built-in random early-stopping
split; v4's person-disjoint stopping fixes this.

## Project structure

```
app.py                      Flask web app
src/config.py               Trait definitions, paths, score bands
src/data_generation.py      Synthetic people + pair scoring simulator
src/features.py            Pair feature engineering
src/models.py              PairMLP (ANN), ensemble, symmetric wrapper
src/train.py               Baselines, ablation, hyperparameter search, final model
src/analysis.py            Figures, permutation importance, learning curve, scenario checks
src/predict.py             score_pair, explain_pair, rank_candidates
src/evaluate.py            Metrics
tests/test_model.py        Unit tests (symmetry, bounds, sanity scenarios)
data/raw/                  people.csv, pairs.csv (generated); data/survey_template.csv
models/compatibility_ann.joblib   Trained ensemble
reports/figures, reports/metrics  Plots and result tables
docs/                       Survey guide and app screenshots
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_generation     # (optional) regenerate data
python -m src.train               # ~20 min on 2 CPU cores; use --quick for a 3-minute smoke test
python -m src.analysis            # figures + importance + learning curve
pytest -q                         # tests
python app.py                     # open http://127.0.0.1:5000
```

The repository already contains the trained model and results, so `python app.py` works straight away.

## Using it from Python

```python
from src.predict import score_pair, explain_pair
a = {"smoker": 1, "smoke_tolerance": 5, "cleanliness": 5}
b = {"smoke_tolerance": 1, "cleanliness": 1}
score_pair(a, b)                  # 0-100
explain_pair(a, b)                # (score, friction points, ...)
```

Unspecified traits default to the neutral midpoint (3).

## Limitations

- Simulated labels: scores reflect the designer's assumptions about what causes friction.
- Only lifestyle traits are used; gender, religion, caste and region are deliberately excluded.
- 1-5 self-reported answers are subjective; real survey data will be noisier than the simulation.
- Scores are estimates to start a conversation, not a verdict about real people.

## Future work

Collect real survey outcomes, calibrate prediction intervals, add a candidate-ranking view in the app, and try
a PyTorch Siamese network (shared person encoder) once a GPU/PyTorch environment is available.

## Acknowledgements

I gratefully acknowledge the support extended by my classmates during the course of this project:

- **Dharmesh Durgeshi** (UID: 25MDA10087), MBA (Data Science and AI), Chandigarh University
- **Tanishq Singh Guleria** (UID: 25MDA10094), MBA (Data Science and AI), Chandigarh University

Their discussions, feedback and help with reviewing and testing the work contributed to its improvement.
I also thank **Er. Aaqib Rashid Mir** for his guidance and for evaluating this submission.

## Author

**Akshay Mishra** (UID: 25MDA10000074), MBA (Data Science and AI), Chandigarh University.
GitHub: [@Alexa21051994](https://github.com/Alexa21051994)

Submitted to: **Er. Aaqib Rashid Mir**
