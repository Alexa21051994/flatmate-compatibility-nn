# Collecting real survey data (optional upgrade)

The shipped dataset is **simulated** (see `src/data_generation.py`). To train on real outcomes:

1. Ask each participant to fill the columns in `data/survey_template.csv`
   (1-5 scales are described in `src/config.py`; `diet`: 0 vegetarian, 1 eggetarian, 2 non-vegetarian;
   `budget_k` is monthly rent budget in INR thousands; hobbies are 0/1).
2. Collect the outcome for each *pair of people who actually lived together*:
   a 0-100 rating (e.g. the average of both flatmates' "how well did it work out?" ratings after 3+ months).
3. Save `data/raw/people.csv` (person_id, survey columns, `split`) and `data/raw/pairs.csv`
   (`person_a, person_b, split, score, score_noise_free`; set `score_noise_free = score` if unknown).
   Assign `split` per **person**, not per pair, and only form pairs inside a split.
4. Run `python -m src.train` and `python -m src.analysis`.

Collect consent, keep the data anonymous, and do not include gender, religion, caste, region or other
protected attributes as inputs.
