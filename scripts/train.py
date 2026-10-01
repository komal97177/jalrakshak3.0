"""Train the flood model on REAL satellite flood-inundation records (block x month, Bhagalpur 2021-2025).
Label: block had >=10% of its area inundated. Validation: leave-one-year-out."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score
from app.core import DATA, MODELS, features

THR = 0.10
d = pd.read_csv(DATA / "inundation_bhagalpur.csv"); d["block_id"] = d.object_id.str[-5:]
bl = {r["block_id"]: r for r in csv.DictReader(open(DATA / "blocks.csv", encoding="utf-8"))}
X = np.array([features(float(bl[b]["lat"]), float(bl[b]["lon"]), m) for b, m in zip(d.block_id, d.month_num)])
y, yf = (d.inundation_pct >= THR).astype(int).values, d.inundation_pct.values
mk = lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1)
mr = lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1)
P, F = np.zeros(len(d)), np.zeros(len(d))
for yr in sorted(d.year.unique()):
    te = (d.year == yr).values
    P[te] = mk().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]; F[te] = mr().fit(X[~te], yf[~te]).predict(X[te])
base = np.full(len(d), y.mean())
metrics = {"rows": len(d), "blocks": int(d.block_id.nunique()), "years": sorted(int(v) for v in d.year.unique()),
           "positive_rate": round(float(y.mean()), 3), "label": f"inundation >= {int(THR*100)}% of block area",
           "loyo_auc": round(roc_auc_score(y, P), 3), "loyo_brier": round(brier_score_loss(y, P), 3),
           "baseline_brier": round(brier_score_loss(y, base), 3), "loyo_r2_inundation": round(r2_score(yf, F), 3),
           "validation": "leave-one-year-out (train on 4 years, test on the 5th)",
           "caveat": "Model has no rainfall feature (no rain history in the records). Rain/river adjustment is hand-set; see model card."}
MODELS.mkdir(exist_ok=True)
joblib.dump({"clf": mk().fit(X, y), "reg": mr().fit(X, yf), "metrics": metrics}, MODELS / "flood_rf.joblib")
print(json.dumps(metrics, indent=1))
