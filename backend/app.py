"""Flask API + static frontend server.  Run from project root:  python backend/app.py"""
import io, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

ART = joblib.load(ROOT / "model" / "manufacturing_model.joblib")
META = json.loads((ROOT / "model" / "model_metadata.json").read_text())
MODEL, FEATURES, STATS = ART["model"], ART["features"], ART["input_stats"]


def risk_level(p):  # UI interpretation thresholds, not statistical guarantees
    return "Low" if p < 0.30 else ("Moderate" if p <= 0.70 else "High")


def clean_frame(df):
    """Check required columns, coerce to numeric, reject bad values. Returns float frame in model feature order."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad = X.isna().any(axis=1)
    if bad.any():
        rows = (X.index[bad][:5] + 1).tolist()
        raise ValueError(f"Non-numeric or empty values found (e.g. in row(s) {rows}).")
    return X.astype(float)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_loaded=MODEL is not None)


@app.get("/api/summary")
def summary():
    return jsonify(records=META["dataset"]["records"], features=len(FEATURES), model=META["model_name"],
                   metrics=META["metrics"], dataset=META["dataset"], feature_names=FEATURES,
                   input_stats=STATS, trained_at=META["trained_at"])


@app.get("/api/insights")
def insights():
    return jsonify({k: META[k] for k in ["model_name", "metrics", "confusion", "comparison", "feature_importance",
                                         "group_analysis", "excluded_features", "leakage_note", "dataset"]})


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Request body must be a JSON object."), 400
    try:
        X = clean_frame(pd.DataFrame([data]))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    prob = float(MODEL.predict_proba(X)[0, 1])
    # Explanation: is each value closer to the defective or the non-defective class average?
    # Score = importance x (distance to non-defective avg - distance to defective avg) / std. Positive => looks defective.
    imp, ga = META["feature_importance"], META["group_analysis"]
    contrib = []
    for f in FEATURES:
        w = max(imp.get(f, 0.0), 0.0)
        sd = (STATS[f]["max"] - STATS[f]["min"]) / 3.46 or 1.0  # std of a uniform range
        v = float(X.iloc[0][f])
        score = w * (abs(v - ga[f]["non_defective"]) - abs(v - ga[f]["defective"])) / sd
        if w > 0:
            contrib.append({"feature": f, "value": v, "effect": float(score),
                            "leans": "defective" if score > 0 else "non-defective"})
    contrib = sorted(contrib, key=lambda d: -abs(d["effect"]))[:5]
    return jsonify(prediction=int(prob >= 0.5), label="Defective" if prob >= 0.5 else "Non-Defective",
                   probability=prob, risk_level=risk_level(prob), contributions=contrib)


@app.post("/api/batch-predict")
def batch_predict():
    f = request.files.get("file")
    if f is None or not f.filename.lower().endswith(".csv"):
        return jsonify(error="Please upload a .csv file."), 400
    try:
        df = pd.read_csv(io.BytesIO(f.read()))
        if df.empty:
            raise ValueError("The CSV file has no rows.")
        X = clean_frame(df)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        return jsonify(error="Could not read the file as a CSV."), 400
    probs = MODEL.predict_proba(X)[:, 1]
    out = df.copy()
    out["PredictedDefectStatus"] = (probs >= 0.5).astype(int)
    out["PredictionLabel"] = np.where(probs >= 0.5, "Defective", "Non-Defective")
    out["DefectProbability"] = probs.round(4)
    out["RiskLevel"] = [risk_level(p) for p in probs]
    risk = out["RiskLevel"].value_counts()
    res = dict(total=len(out), defective=int((probs >= 0.5).sum()), non_defective=int((probs < 0.5).sum()),
               avg_probability=float(probs.mean()), risk_counts={k: int(risk.get(k, 0)) for k in ["Low", "Moderate", "High"]},
               columns=list(out.columns), rows=json.loads(out.to_json(orient="records")))
    if "DefectStatus" in df.columns:
        actual = pd.to_numeric(df["DefectStatus"], errors="coerce")
        res["accuracy_vs_actual"] = float((actual == out["PredictedDefectStatus"]).mean())
    return jsonify(res)


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="File too large (limit 5 MB)."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
