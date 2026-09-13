"""
Builds the JSON data block + SHAP beeswarm PNG for the conversion (win/loss)
model's SHAP-space compass visualization, mirroring adr_shap_umap_3d1.html
but for rfp_conversion_pipeline_filtered_first.py's classifier.
"""
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.cluster import DBSCAN

import sys
sys.path.insert(0, "/home/claude/work")
from rfp_conversion_pipeline_filtered_first import (
    load_data_filtered, engineer_features_filtered, prep_X, RANDOM_STATE
)

RANDOM_STATE = 42

# ---- 1. Reload data + bundle ----
print("Loading model bundle...")
with open("outputs/conversion_filtered_first_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
features = bundle["features"]
print(f"  {len(features)} features")

print("Rebuilding the exact training population (n=1705)...")
ev = load_data_filtered()
ev = engineer_features_filtered(ev)
ev = ev[ev["room_block"] > 0].copy().reset_index(drop=True)
print(f"  {len(ev)} rows")

X = prep_X(ev, features)
y = ev["did_convert"].astype(int).values

proba = model.predict_proba(X)[:, 1]
pred_label = (proba >= 0.5).astype(int)

# ---- 2. Real per-row SHAP values from the actual trained model ----
print("Computing SHAP values (TreeExplainer, margin/log-odds space)...")
explainer = shap.TreeExplainer(model, model_output="raw")
shap_values = explainer.shap_values(X)  # (n, n_features), log-odds contributions
print(f"  shap_values shape: {shap_values.shape}")

# Sanity check: base_value + sum(shap) should equal margin prediction
margin_pred = model.predict(X, output_margin=True)
recon = explainer.expected_value + shap_values.sum(axis=1)
print(f"  max abs reconstruction error vs margin: {np.max(np.abs(recon - margin_pred)):.6f}")

# ---- 3. UMAP embed the full SHAP matrix into 3D ----
print("Running UMAP on all SHAP dimensions -> 3D...")
reducer = umap.UMAP(n_components=3, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
emb = reducer.fit_transform(shap_values)
# Normalize to roughly [-1, 1]-ish range like the reference file
emb_scaled = emb / (np.abs(emb).std(axis=0).mean() * 3)

# ---- 4. Density-based clustering on the embedding (ordered by win rate) ----
print("Clustering (DBSCAN) on the 3D embedding...")
best_labels, best_eps, best_n_noise = None, None, None
for eps in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]:
    db = DBSCAN(eps=eps, min_samples=15).fit(emb_scaled)
    n_noise = int((db.labels_ == -1).sum())
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    if 4 <= n_clusters <= 12 and n_noise < len(emb_scaled) * 0.05:
        best_labels, best_eps, best_n_noise = db.labels_, eps, n_noise
        break
if best_labels is None:
    # fallback: pick eps with fewest noise points among reasonable cluster counts
    candidates = []
    for eps in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        db = DBSCAN(eps=eps, min_samples=15).fit(emb_scaled)
        n_noise = int((db.labels_ == -1).sum())
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        candidates.append((n_noise, eps, db.labels_, n_clusters))
    candidates.sort(key=lambda t: (t[0], -t[3]))
    best_n_noise, best_eps, best_labels, _ = candidates[0]

print(f"  eps={best_eps}, noise points={best_n_noise}, n_clusters={len(set(best_labels)) - (1 if -1 in best_labels else 0)}")

# assign noise points to nearest cluster centroid
raw_labels = best_labels.copy()
unique_clusters = sorted(set(raw_labels) - {-1})
centroids = {c: emb_scaled[raw_labels == c].mean(axis=0) for c in unique_clusters}
for i in np.where(raw_labels == -1)[0]:
    dists = {c: np.linalg.norm(emb_scaled[i] - cen) for c, cen in centroids.items()}
    raw_labels[i] = min(dists, key=dists.get)

# order clusters by mean predicted win probability, low -> high
cluster_mean_proba = {c: proba[raw_labels == c].mean() for c in unique_clusters}
ordered = sorted(unique_clusters, key=lambda c: cluster_mean_proba[c])
remap = {old: new for new, old in enumerate(ordered)}
cluster_labels = np.array([remap[c] for c in raw_labels])

cluster_info = []
for new_id in range(len(ordered)):
    mask = cluster_labels == new_id
    n = int(mask.sum())
    win_rate = float(y[mask].mean())
    mean_p = float(proba[mask].mean())
    cluster_info.append({
        "id": new_id, "n": n,
        "win_rate_pct": round(win_rate * 100, 1),
        "mean_pred_proba": round(mean_p, 3),
        "tag": f"win-rate {win_rate*100:.0f}%",
    })

# ---- 5. Top-5 signed SHAP contributors per row ----
print("Extracting top-5 SHAP contributors per row...")
abs_shap = np.abs(shap_values)
top5_idx = np.argsort(-abs_shap, axis=1)[:, :5]

points = []
for i in range(len(ev)):
    contribs = [
        {"name": features[j], "shap": round(float(shap_values[i, j]), 3)}
        for j in top5_idx[i]
    ]
    points.append({
        "id": str(ev["rfp_id"].iloc[i]),
        "acct": str(ev["account_name"].iloc[i]),
        "x": round(float(emb_scaled[i, 0]), 4),
        "y": round(float(emb_scaled[i, 1]), 4),
        "z": round(float(emb_scaled[i, 2]), 4),
        "won": int(y[i]),
        "actual": int(y[i]),
        "pred": round(float(proba[i]), 4),
        "contribs": contribs,
        "cluster": int(cluster_labels[i]),
    })

# ---- 6. Comparison table: does UMAP-of-SHAP separate outcomes better? ----
print("Computing comparison metrics (5-fold k-NN ROC-AUC)...")
from sklearn.metrics import roc_auc_score

def knn_auc(Z, y, k=5):
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in cv.split(Z, y):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(Z[tr], y[tr])
        p = knn.predict_proba(Z[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))

Xf = X.values.astype(float)
Xstd = (Xf - Xf.mean(axis=0)) / (Xf.std(axis=0) + 1e-9)

pca3 = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(Xstd)
auc_pca3 = knn_auc(pca3, y)

umap_raw3 = umap.UMAP(n_components=3, random_state=RANDOM_STATE).fit_transform(Xstd)
auc_umap_raw3 = knn_auc(umap_raw3, y)

top3_feats = list(pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False).head(3).index)
top3_idx = [features.index(f) for f in top3_feats]
auc_shap_top3 = knn_auc(Xstd[:, top3_idx], y)

auc_umap_shap_all = knn_auc(emb_scaled, y)

comparison = {
    "pca3": round(auc_pca3, 3),
    "umap_raw3": round(auc_umap_raw3, 3),
    "shap_top3": round(auc_shap_top3, 3),
    "shap_top3_names": top3_feats,
    "umap_shap_all": round(auc_umap_shap_all, 3),
    "n_shap_dims": len(features),
}
print("  ", comparison)

# ---- 7. Real model metrics from the bundle ----
metrics = bundle["metrics"]

# ---- 8. SHAP beeswarm summary plot ----
print("Rendering SHAP beeswarm summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, feature_names=features, max_display=18, show=False)
buf = io.BytesIO()
plt.tight_layout()
plt.savefig(buf, format="png", dpi=130)
plt.close()
buf.seek(0)
beeswarm_b64 = base64.b64encode(buf.read()).decode("ascii")
print(f"  PNG size: {len(beeswarm_b64)} base64 chars")

# ---- 9. Assemble final data blob ----
data = {
    "points": points,
    "cluster_info": cluster_info,
    "metrics": metrics,
    "comparison": comparison,
    "n_rows": len(ev),
    "n_won": int(y.sum()),
    "n_lost": int(len(y) - y.sum()),
    "base_value": float(explainer.expected_value),
}

with open("shap_umap_data.json", "w") as f:
    json.dump(data, f)
with open("shap_beeswarm_b64.txt", "w") as f:
    f.write(beeswarm_b64)

print("DONE. Wrote shap_umap_data.json and shap_beeswarm_b64.txt")
