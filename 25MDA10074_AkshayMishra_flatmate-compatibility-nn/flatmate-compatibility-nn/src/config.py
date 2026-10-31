"""Central configuration: trait definitions, paths and global constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
METRIC_DIR = ROOT / "reports" / "metrics"

SEED = 42

# --- Person-level lifestyle traits (what a flatmate-seeker answers in the survey) ---
# name: (label, min, max, description)
ORDINAL_TRAITS = {
    "chronotype":          ("Sleep schedule", 1, 5, "1 = early bird, 5 = night owl"),
    "cleanliness":         ("Cleanliness", 1, 5, "1 = relaxed, 5 = very tidy"),
    "noise_level":         ("Noise you make", 1, 5, "1 = very quiet, 5 = loud (music, calls)"),
    "noise_sensitivity":   ("Noise sensitivity", 1, 5, "1 = sleeps through anything, 5 = very sensitive"),
    "guest_frequency":     ("Hosting guests", 1, 5, "1 = never, 5 = very often"),
    "guest_tolerance":     ("Guest tolerance", 1, 5, "1 = dislikes guests, 5 = welcomes guests"),
    "smoke_tolerance":     ("Smoke tolerance", 1, 5, "1 = cannot tolerate, 5 = fine with it"),
    "alcohol":             ("Alcohol use", 1, 5, "1 = never, 5 = frequently"),
    "kitchen_strictness":  ("Kitchen strictness", 1, 5, "How strict about food rules / separate utensils"),
    "cooking_frequency":   ("Cooking", 1, 5, "1 = never cooks, 5 = cooks daily"),
    "chore_sharing":       ("Chore sharing", 1, 5, "1 = avoids chores, 5 = shares equally"),
    "time_at_home":        ("Time at home", 1, 5, "1 = rarely home, 5 = always home"),
    "extraversion":        ("Extraversion", 1, 5, "1 = introvert, 5 = extrovert"),
    "pet_tolerance":       ("Pet tolerance", 1, 5, "1 = dislikes pets, 5 = loves pets"),
    "ac_preference":       ("AC / cold preference", 1, 5, "1 = prefers warm, 5 = prefers very cold"),
    "bill_responsibility": ("Bill responsibility", 1, 5, "1 = often late with shared payments, 5 = always on time"),
}
BINARY_TRAITS = {
    "smoker":     "Smokes (0/1)",
    "has_pet":    "Has a pet (0/1)",
    "occupation": "0 = student, 1 = working professional",
}
DIET_LABELS = {0: "Vegetarian", 1: "Eggetarian", 2: "Non-vegetarian"}
HOBBIES = ["cricket_sports", "gaming", "music", "movies", "gym_fitness", "cooking_food", "reading", "travel"]
CONTINUOUS_TRAITS = {"budget_k": "Monthly rent budget (INR thousands)"}

PERSON_NUMERIC = (list(ORDINAL_TRAITS) + list(BINARY_TRAITS) + ["diet", "budget_k"])
PERSON_FEATURES = PERSON_NUMERIC + ["hobby_" + h for h in HOBBIES]
SURVEY_COLUMNS = PERSON_NUMERIC + ["hobby_" + h for h in HOBBIES]

# Trait groups used for permutation importance and the "friction point" explanation
TRAIT_GROUPS = {
    "Sleep schedule":      ["chronotype"],
    "Cleanliness":         ["cleanliness"],
    "Chore sharing":       ["chore_sharing"],
    "Noise":               ["noise_level", "noise_sensitivity"],
    "Guests":              ["guest_frequency", "guest_tolerance"],
    "Smoking":             ["smoker", "smoke_tolerance"],
    "Alcohol":             ["alcohol"],
    "Diet & kitchen":      ["diet", "kitchen_strictness"],
    "Cooking":             ["cooking_frequency"],
    "Budget":              ["budget_k"],
    "Time at home":        ["time_at_home"],
    "Extraversion":        ["extraversion"],
    "Pets":                ["has_pet", "pet_tolerance"],
    "AC preference":       ["ac_preference"],
    "Bill responsibility": ["bill_responsibility"],
    "Occupation":          ["occupation"],
    "Hobbies":             ["hobby_" + h for h in HOBBIES],
}

# Compatibility bands used to turn the 0-100 score into a decision
BAND_EDGES = (50.0, 70.0)
BAND_NAMES = ["Low", "Moderate", "High"]


def band(score):
    """Map a 0-100 score to Low / Moderate / High."""
    import numpy as np
    s = np.asarray(score, dtype=float)
    return np.digitize(s, BAND_EDGES)  # 0,1,2
