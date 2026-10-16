"""Train, compare and save the defect model.  Run from project root:  python scripts/train_model.py"""
import json, datetime, warnings
from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
TARGET = "DefectStatus"
# LEAKAGE DECISION: DefectRate is named like an outcome measurement, so it is excluded from the
# predictive features (AUC drops only ~0.02 without it). It is still analysed as an association.
EXCLUDE = ["DefectRate"]

df = pd.read_csv(ROOT / "data" / "manufacturing_defect_dataset.csv")
features = [c for c in df.columns if c not in [TARGET] + EXCLUDE]
X, y = df[features], df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def pipe(clf, scale):
    return Pipeline([("scale", StandardScaler()), ("clf", clf)]) if scale else Pipeline([("clf", clf)])

candidates = {
    "Logistic Regression": pipe(LogisticRegression(class_weight="balanced", max_iter=2000), True),
    "Decision Tree": pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42), False),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1), False),
    "KNN": pipe(KNeighborsClassifier(n_neighbors=15, weights="distance"), True),
}
results, fitted = [], {}
for name, m in candidates.items():
    m.fit(Xtr, ytr); p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]; fitted[name] = m
    tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    results.append(dict(model=name, accuracy=accuracy_score(yte, p), precision=precision_score(yte, p),
        recall=recall_score(yte, p), f1=f1_score(yte, p), roc_auc=roc_auc_score(yte, pr),
        confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))))
    print(f"{name:20s} acc={results[-1]['accuracy']:.3f} prec={results[-1]['precision']:.3f} "
          f"rec={results[-1]['recall']:.3f} f1={results[-1]['f1']:.3f} auc={results[-1]['roc_auc']:.3f}")

# imbalance => select by ROC-AUC (threshold-free), not accuracy
best = max(results, key=lambda r: r["roc_auc"]); model = fitted[best["model"]]
print("Selected:", best["model"])

perm = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=42, n_jobs=-1)
importance = dict(sorted(zip(features, perm.importances_mean.round(5).tolist()), key=lambda kv: -kv[1]))
groups = {c: dict(non_defective=float(df[df[TARGET] == 0][c].mean()), defective=float(df[df[TARGET] == 1][c].mean()),
          corr=float(df[c].corr(df[TARGET]))) for c in features + EXCLUDE}
stats = {c: dict(min=float(df[c].min()), max=float(df[c].max()), mean=float(df[c].mean()),
         median=float(df[c].median()), is_int=bool(pd.api.types.is_integer_dtype(df[c]))) for c in features + EXCLUDE}
vc = df[TARGET].value_counts()
meta = dict(model_name=best["model"], trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
    features=features, excluded_features=EXCLUDE,
    leakage_note="DefectRate is named like an outcome measurement, so it was excluded from the model to avoid possible "
                 "leakage. It is still reported in the pattern analysis.",
    metrics={k: best[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}, confusion=best["confusion"],
    comparison=results, feature_importance=importance, group_analysis=groups, input_stats=stats,
    dataset=dict(records=len(df), n_features=len(features), defective=int(vc[1]), non_defective=int(vc[0]),
                 defective_pct=float(vc[1] / len(df)), missing=int(df.isna().sum().sum()),
                 duplicates=int(df.duplicated().sum()), train_size=len(Xtr), test_size=len(Xte)))
joblib.dump(dict(model=model, **{k: meta[k] for k in ["model_name", "features", "metrics", "feature_importance", "input_stats"]}),
            ROOT / "model" / "manufacturing_model.joblib")
(ROOT / "model" / "model_metadata.json").write_text(json.dumps(meta, indent=2))
print("Saved model/manufacturing_model.joblib and model/model_metadata.json")
