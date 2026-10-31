"""Synthetic flatmate-survey generator.

No public dataset of real flatmate outcomes exists, so this module simulates
(1) a population of people with realistic, correlated lifestyle traits and
(2) a compatibility score for pairs of people, produced by a transparent,
domain-informed "living-together friction" model plus random pair-level noise.

The neural network never sees the scoring rules, only (person A, person B, score).
Everything is deterministic given SEED. Replace `data/raw/people.csv` / `pairs.csv`
with real survey data (see docs/SURVEY_GUIDE.md) to retrain on real outcomes.

Run:  python -m src.data_generation
"""
import numpy as np
import pandas as pd

from .config import DATA_RAW, HOBBIES, SEED

N_PEOPLE = 1800
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
PAIRS_PER_SPLIT = {"train": 24000, "val": 5000, "test": 5000}
K_SCALE = 54.0   # sets how quickly friction erodes the score
NOISE_SD = 4.0  # pair "chemistry" noise (points on the 0-100 scale)


# ----------------------------------------------------------------------------
# 1. People
# ----------------------------------------------------------------------------
def _ordinal(rng, mean, sd_noise=0.6, size=None):
    return np.clip(np.rint(mean + rng.normal(0, sd_noise, size)), 1, 5).astype(int)


def generate_people(n=N_PEOPLE, rng=None):
    rng = rng or np.random.default_rng(SEED)
    z_ex, z_con, z_owl, z_agr = (rng.normal(size=n) for _ in range(4))

    chronotype = _ordinal(rng, 3 + 0.9 * z_owl, 0.55)
    cleanliness = _ordinal(rng, 3.1 + 0.8 * z_con, 0.6)
    chore_sharing = _ordinal(rng, 3.3 + 0.5 * z_con + 0.4 * z_agr, 0.6)
    bill_resp = _ordinal(rng, 3.7 + 0.6 * z_con - 0.2 * z_owl, 0.6)
    extraversion = _ordinal(rng, 3 + 0.9 * z_ex, 0.5)
    guest_frequency = _ordinal(rng, 2.7 + 0.7 * z_ex, 0.7)
    guest_tolerance = _ordinal(rng, 3.2 + 0.5 * z_ex + 0.3 * z_agr, 0.8)
    noise_level = _ordinal(rng, 2.8 + 0.4 * z_ex + 0.3 * z_owl, 0.8)
    noise_sens = _ordinal(rng, 3.0 - 0.3 * z_ex + 0.3 * z_con, 1.0)

    smoker = (rng.random(n) < 1 / (1 + np.exp(-(-1.7 + 0.3 * z_owl)))).astype(int)
    smoke_tol = np.where(smoker == 1, _ordinal(rng, 4.5, 0.5), _ordinal(rng, 2.2, 1.1))
    alcohol = np.clip(np.rint(1.8 + 0.5 * z_ex + 1.2 * smoker + rng.normal(0, 0.9, n)), 1, 5).astype(int)

    diet = rng.choice([0, 1, 2], size=n, p=[0.30, 0.12, 0.58])
    kitchen = np.where(diet == 0, _ordinal(rng, 3.8, 0.9),
              np.where(diet == 1, _ordinal(rng, 3.0, 0.8), _ordinal(rng, 1.8, 0.8)))
    cooking = _ordinal(rng, 2.8 + 0.5 * z_con, 0.9)

    budget = np.exp(rng.normal(np.log(9), 0.4, n))
    budget = np.round(np.clip(budget, 4, 25) * 2) / 2

    occupation = (rng.random(n) < 0.55).astype(int)  # 1 = professional
    time_home = np.where(occupation == 0, _ordinal(rng, 3.7, 0.8), _ordinal(rng, 2.6, 0.8))

    has_pet = (rng.random(n) < 0.12).astype(int)
    pet_tol = np.where(has_pet == 1, _ordinal(rng, 4.8, 0.4), _ordinal(rng, 3.3, 1.1))
    ac_pref = _ordinal(rng, 3.0, 1.0)

    base_p = np.array([0.35, 0.35, 0.45, 0.45, 0.30, 0.35, 0.28, 0.35])
    hobbies = {}
    for k, h in enumerate(HOBBIES):
        p = np.clip(base_p[k] + 0.08 * z_ex, 0.05, 0.9)
        hobbies["hobby_" + h] = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "chronotype": chronotype, "cleanliness": cleanliness, "noise_level": noise_level,
        "noise_sensitivity": noise_sens, "guest_frequency": guest_frequency,
        "guest_tolerance": guest_tolerance, "smoke_tolerance": smoke_tol, "alcohol": alcohol,
        "kitchen_strictness": kitchen, "cooking_frequency": cooking, "chore_sharing": chore_sharing,
        "time_at_home": time_home, "extraversion": extraversion, "pet_tolerance": pet_tol,
        "ac_preference": ac_pref, "bill_responsibility": bill_resp,
        "smoker": smoker, "has_pet": has_pet, "occupation": occupation,
        "diet": diet, "budget_k": budget, **hobbies,
    })
    df.insert(0, "person_id", np.arange(n))
    # person-level split: pairs are only formed *within* a split (prevents leakage)
    order = rng.permutation(n)
    n_tr = int(SPLIT_FRACTIONS["train"] * n)
    n_va = int(SPLIT_FRACTIONS["val"] * n)
    split = np.empty(n, dtype=object)
    split[order[:n_tr]] = "train"
    split[order[n_tr:n_tr + n_va]] = "val"
    split[order[n_tr + n_va:]] = "test"
    df["split"] = split
    return df


# ----------------------------------------------------------------------------
# 2. Compatibility scoring ("ground-truth" simulator)
# ----------------------------------------------------------------------------
def _u(x):  # scale 1..5 -> 0..1
    return (x - 1.0) / 4.0


def compatibility_score(a: pd.DataFrame, b: pd.DataFrame, noise_sd=0.0, rng=None):
    """Score 0-100 for aligned rows of person frames a and b (symmetric in a, b).

    Friction terms are penalties (weight x severity in [0, 1]). Several are
    deliberately directional and non-linear (e.g. a smoker with a smoke-intolerant
    flatmate; the tidier person suffers more from mess), so a network must learn
    interactions rather than simple differences.
    """
    A = {c: a[c].to_numpy(float) for c in a.columns if c not in ("person_id", "split")}
    B = {c: b[c].to_numpy(float) for c in b.columns if c not in ("person_id", "split")}

    sched = (abs(A["chronotype"] - B["chronotype"]) / 4) ** 1.3 * (
        0.6 + 0.4 * np.maximum(A["noise_sensitivity"], B["noise_sensitivity"]) / 5)
    gap = abs(A["cleanliness"] - B["cleanliness"]) / 4
    tidier = np.maximum(A["cleanliness"], B["cleanliness"])
    clean = gap ** 1.4 * (0.6 + 0.4 * _u(tidier))
    chore = 0.6 * ((5 - np.minimum(A["chore_sharing"], B["chore_sharing"])) / 4) ** 1.2 \
        + 0.4 * abs(A["chore_sharing"] - B["chore_sharing"]) / 4
    home = 0.7 + 0.3 * (A["time_at_home"] + B["time_at_home"]) / 10
    n_ab = _u(A["noise_level"]) * _u(B["noise_sensitivity"])
    n_ba = _u(B["noise_level"]) * _u(A["noise_sensitivity"])
    noise = (0.5 * (n_ab + n_ba) + 0.5 * np.maximum(n_ab, n_ba)) * home
    g_ab = _u(A["guest_frequency"]) * (5 - B["guest_tolerance"]) / 4
    g_ba = _u(B["guest_frequency"]) * (5 - A["guest_tolerance"]) / 4
    guests = 0.5 * (g_ab + g_ba) + 0.5 * np.maximum(g_ab, g_ba)
    smoke = np.maximum(A["smoker"] * (5 - B["smoke_tolerance"]) / 4,
                       B["smoker"] * (5 - A["smoke_tolerance"]) / 4)
    alcohol = abs(A["alcohol"] - B["alcohol"]) / 4
    diet_gap = abs(A["diet"] - B["diet"]) / 2
    veg_strict = np.where(A["diet"] < B["diet"], A["kitchen_strictness"], B["kitchen_strictness"])
    diet = diet_gap * (0.1 + 0.9 * _u(veg_strict))
    cooking = abs(A["cooking_frequency"] - B["cooking_frequency"]) / 4
    budget = np.tanh(abs(np.log(A["budget_k"] / B["budget_k"])) / np.log(2.0))
    social = abs(A["extraversion"] - B["extraversion"]) / 4 * (0.4 + 0.6 * (A["time_at_home"] + B["time_at_home"]) / 10)
    p_ab = A["has_pet"] * (5 - B["pet_tolerance"]) / 4
    p_ba = B["has_pet"] * (5 - A["pet_tolerance"]) / 4
    pets = np.maximum(p_ab, p_ba)
    ac = abs(A["ac_preference"] - B["ac_preference"]) / 4
    bills = ((5 - np.minimum(A["bill_responsibility"], B["bill_responsibility"])) / 4) ** 1.3

    penalty = (9 * sched + 14 * clean + 8 * chore + 12 * noise + 9 * guests + 20 * smoke
               + 4 * alcohol + 10 * diet + 3 * cooking + 8 * budget + 5 * social
               + 12 * pets + 4 * ac + 9 * bills)

    hob = [c for c in a.columns if c.startswith("hobby_")]
    shared = sum(A[h] * B[h] for h in hob)
    bonus = 7 * np.minimum(1.0, shared / 3.0)
    bonus += 2.0 * (A["occupation"] == B["occupation"])
    bonus += 4 * np.clip((np.minimum(A["chore_sharing"], A["bill_responsibility"]) - 3) / 2, 0, 1) \
        * np.clip((np.minimum(B["chore_sharing"], B["bill_responsibility"]) - 3) / 2, 0, 1)

    # Friction compounds multiplicatively: each extra problem hurts less in absolute terms
    # once the relationship is already strained (saturating, non-linear response).
    score = 100.0 * np.exp(-(penalty - bonus) / K_SCALE)
    score *= np.where(smoke > 0.75, 0.80, 1.0) * np.where(pets > 0.8, 0.88, 1.0)  # deal-breakers
    if noise_sd > 0:
        rng = rng or np.random.default_rng(SEED)
        score = score + rng.normal(0, noise_sd, len(score))  # chemistry noise
    return np.clip(score, 0, 100)


# ----------------------------------------------------------------------------
# 3. Pair sampling
# ----------------------------------------------------------------------------
def sample_pairs(people: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for split, n_pairs in PAIRS_PER_SPLIT.items():
        ids = people.loc[people.split == split, "person_id"].to_numpy()
        seen, out = set(), []
        while len(out) < n_pairs:
            i, j = rng.choice(ids, 2, replace=False)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            out.append((key[0], key[1]))
        rows += [(i, j, split) for i, j in out]
    pairs = pd.DataFrame(rows, columns=["person_a", "person_b", "split"])
    ppl = people.set_index("person_id")
    a, b = ppl.loc[pairs.person_a].reset_index(drop=True), ppl.loc[pairs.person_b].reset_index(drop=True)
    clean = compatibility_score(a, b)
    pairs["score_noise_free"] = clean.round(2)
    pairs["score"] = np.clip(clean + rng.normal(0, NOISE_SD, len(clean)), 0, 100).round(2)
    return pairs


def main():
    rng = np.random.default_rng(SEED)
    people = generate_people(rng=rng)
    pairs = sample_pairs(people, rng)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    people.to_csv(DATA_RAW / "people.csv", index=False)
    pairs.to_csv(DATA_RAW / "pairs.csv", index=False)
    print(f"people: {people.shape}, pairs: {pairs.shape}")
    print(pairs.groupby("split").score.describe().round(2))


if __name__ == "__main__":
    main()
