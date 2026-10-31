"""Flask web app: enter two flatmate profiles, get a compatibility score and friction points.

Run:  python app.py      then open http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request

from src.config import BAND_NAMES, DIET_LABELS, HOBBIES, ORDINAL_TRAITS, band
from src.predict import explain_pair, load_model

app = Flask(__name__)

NUMERIC_FORM = {k: (v[0], v[3]) for k, v in ORDINAL_TRAITS.items()}


def parse_person(form, prefix):
    p = {}
    for k in ORDINAL_TRAITS:
        p[k] = int(form.get(f"{prefix}_{k}", 3))
    p["smoker"] = int(form.get(f"{prefix}_smoker", 0))
    p["has_pet"] = int(form.get(f"{prefix}_has_pet", 0))
    p["occupation"] = int(form.get(f"{prefix}_occupation", 0))
    p["diet"] = int(form.get(f"{prefix}_diet", 2))
    p["budget_k"] = max(1.0, float(form.get(f"{prefix}_budget_k", 9)))
    for h in HOBBIES:
        p["hobby_" + h] = 1 if form.get(f"{prefix}_hobby_{h}") else 0
    return p


@app.get("/")
def index():
    return render_template("index.html", traits=NUMERIC_FORM, diets=DIET_LABELS, hobbies=HOBBIES)


@app.post("/predict")
def predict():
    a, b = parse_person(request.form, "a"), parse_person(request.form, "b")
    score, friction, strengths = explain_pair(a, b)
    return jsonify({
        "score": round(score, 1),
        "band": BAND_NAMES[int(band(score))],
        "friction": [{"trait": n, "gain": round(g, 1)} for n, g in friction],
        "aligned": [n for n, _ in strengths],
    })


if __name__ == "__main__":
    load_model()  # fail fast if the model has not been trained
    app.run(debug=False)
