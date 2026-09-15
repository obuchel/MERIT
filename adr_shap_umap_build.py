"""
Builds adr_shap_umap_data.js + shap_beeswarm.png for the ADR model's SHAP-space
compass (adr_shap_umap_3d1_fixed.html), the ADR-model counterpart to
build_shap_umap_data.py (which does the same thing for the win/loss conversion
model). Run this whenever adr_filtered_first_model.pkl or the underlying CSVs
change, then re-open the HTML - it loads its data from the two files this
script writes, it no longer carries the data inline.

adr_shap_umap_data.js now also carries real per-row SHAP + feature values for
the top 18 features (see `beeswarm` below) so the page renders its own
hoverable/clickable beeswarm instead of just embedding shap_beeswarm.png as a
picture. The PNG is still written too, as a static/downloadable fallback -
older copies of the HTML, or anyone opening this data.js before rerunning
against a page that predates this, just keep showing that image untouched.

load_data() / join_transient() / engineer_features() / prep_X() are imported
directly from rfp_adr_pipeline_filtered_first_final.py (the script that
actually produced adr_filtered_first_model.pkl) - not re-derived or guessed.
Only that module's data-loading/feature-engineering functions are used here,
not its training/pruning code, since that part (documented in its own
docstring as having gone through several determinism fixes across revisions)
only matters for retraining the model, not for re-scoring the one that's
already pickled.

ONE thing in here is still NOT recovered from original source:

  The cluster tag rule (tag_cluster()) - reverse-engineered by fitting
  thresholds against the 10 cluster rows already in the existing page and
  confirming an exact match on all 10. It's a plausible rule, not something
  recovered from whatever script originally produced those tags - if you have
  that script, prefer it.

Everything else (SHAP computation, UMAP embedding, DBSCAN sweep, comparison
metrics, beeswarm plot) mirrors build_shap_umap_data.py's actual, working
approach for the conversion model, adapted from classification to regression.
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

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score, mean_absolute_error

# real pipeline functions - this file must sit next to
# rfp_adr_pipeline_filtered_first_final.py (or be on PYTHONPATH)
from rfp_adr_pipeline_filtered_first_final import (
    load_data, join_transient, engineer_features, prep_X, RANDOM_STATE,
)

# ---- 1. load model bundle + rebuild the exact scored population ----
print("Loading ADR model bundle...")
with open("outputs/adr_filtered_first_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
features = bundle["features"]
print(f"  {len(features)} features")

# columns that depend on the transient join (join_transient()) rather than
# being present directly in the raw RFP CSV - flagged in the ranking output
# below the same way this page's footer already describes them
RECONSTRUCTED_COLS = [
    "tr_transient_adr", "tr_transient_revpar", "tr_demand_tier",
    "tr_rooms_to_capacity", "tr_transient_pace_30d", "tr_pace_vs_prior_year",
    "length_of_stay_category_enc",  # + the one-hot market_seg_* pair
]

print("Rebuilding the exact scored population (load_data -> join_transient -> "
      "engineer_features, then room_block>0 - same sequence as "
      "run_adr_pipeline(), minus the training/pruning steps)...")
ev = load_data()
ev = join_transient(ev)
ev = engineer_features(ev)
ev = ev[ev["room_block"] > 0].copy().reset_index(drop=True)
print(f"  {len(ev)} rows")

X = prep_X(ev, features)
actual_adr = ev["quoted_adr"].astype(float).values  # the model's real target column
won = ev["did_convert"].astype(int).values  # kept per-row for the tooltip, no longer a color mode
pred_adr = model.predict(X)

# ---- 2. real per-row SHAP values (regression: raw margin = predicted ADR contributions) ----
print("Computing SHAP values (TreeExplainer, ADR units)...")
explainer = shap.TreeExplainer(model, model_output="raw")
shap_values = explainer.shap_values(X)
recon = explainer.expected_value + shap_values.sum(axis=1)
print(f"  max abs reconstruction error vs model.predict: {np.max(np.abs(recon - pred_adr)):.6f}")

# ---- 3. UMAP embed all SHAP dims -> 3D ----
print("Running UMAP on all SHAP dimensions -> 3D...")
reducer = umap.UMAP(n_components=3, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
emb = reducer.fit_transform(shap_values)
emb_scaled = emb / (np.abs(emb).std(axis=0).mean() * 3)

# ---- 4. DBSCAN price clusters (same eps-sweep approach as build_shap_umap_data.py) ----
print("Clustering (DBSCAN) on the 3D embedding...")
best_labels, best_eps, best_n_noise = None, None, None
for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3]:
    db = DBSCAN(eps=eps, min_samples=15).fit(emb_scaled)
    n_noise = int((db.labels_ == -1).sum())
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    if 8 <= n_clusters <= 12 and n_noise == 0:
        best_labels, best_eps, best_n_noise = db.labels_, eps, n_noise
        break
if best_labels is None:
    candidates = []
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        db = DBSCAN(eps=eps, min_samples=15).fit(emb_scaled)
        n_noise = int((db.labels_ == -1).sum())
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        candidates.append((n_noise, eps, db.labels_, n_clusters))
    candidates.sort(key=lambda t: (t[0], -t[3]))
    best_n_noise, best_eps, best_labels, _ = candidates[0]

print(f"  eps={best_eps}, noise points={best_n_noise}, "
      f"n_clusters={len(set(best_labels)) - (1 if -1 in best_labels else 0)}")

raw_labels = best_labels.copy()
unique_clusters = sorted(set(raw_labels) - {-1})
centroids = {c: emb_scaled[raw_labels == c].mean(axis=0) for c in unique_clusters}
for i in np.where(raw_labels == -1)[0]:
    dists = {c: np.linalg.norm(emb_scaled[i] - cen) for c, cen in centroids.items()}
    raw_labels[i] = min(dists, key=dists.get)

# order clusters by mean actual ADR, low -> high (matches "ordered low->high ADR" in the page copy)
cluster_mean_adr = {c: actual_adr[raw_labels == c].mean() for c in unique_clusters}
ordered = sorted(unique_clusters, key=lambda c: cluster_mean_adr[c])
remap = {old: new for new, old in enumerate(ordered)}
cluster_labels = np.array([remap[c] for c in raw_labels])


def tag_cluster(compression_pct, discount_pct, demand_tier):
    """Reverse-engineered against the 10 clusters already in the existing
    page's data - this exact rule order reproduces all 10 of that page's tags
    with no exceptions. Not recovered from original source; treat as a
    reasonable default and adjust if a re-run's clusters don't fit its
    intent."""
    if compression_pct >= 50:
        return "Compression dates"
    if discount_pct >= 15:
        return "Deep discount"
    if discount_pct <= 5 and demand_tier < 0.5:
        return "Low demand tier"
    if discount_pct <= 1 and demand_tier >= 0.95 and compression_pct < 5:
        return "High demand, list price"
    return "Mixed"


cluster_info = []
for new_id in range(len(ordered)):
    mask = cluster_labels == new_id
    compression_pct = float(ev.loc[mask, "Is_Compression_Date"].mean() * 100) if "Is_Compression_Date" in ev else 0.0
    discount_pct = float(ev.loc[mask, "rate_discount_pct"].mean()) if "rate_discount_pct" in ev else 0.0
    demand_tier = float(ev.loc[mask, "tr_demand_tier"].mean()) if "tr_demand_tier" in ev else 0.0
    cluster_info.append({
        "id": new_id,
        "n": int(mask.sum()),
        "mean_adr": round(float(actual_adr[mask].mean()), 2),
        "win_rate": round(float(won[mask].mean()), 3),
        "compression_pct": round(compression_pct, 1),
        "discount_pct": round(discount_pct, 1),
        "demand_tier": round(demand_tier, 2),
        "tag": tag_cluster(compression_pct, discount_pct, demand_tier),
    })

# ---- 5. top-5 signed SHAP contributors per row ----
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
        "won": int(won[i]),
        "actual": round(float(actual_adr[i]), 2),
        "pred": round(float(pred_adr[i]), 2),
        "contribs": contribs,
        "cluster": int(cluster_labels[i]),
    })

# ---- 6. comparison metrics (5-fold k-NN R², regression) ----
print("Computing comparison metrics (5-fold k-NN R²)...")


def knn_r2(Z, y, k=5):
    cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, te in cv.split(Z):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(Z[tr], y[tr])
        scores.append(r2_score(y[te], knn.predict(Z[te])))
    return float(np.mean(scores))


Xf = X.values.astype(float)
Xstd = (Xf - Xf.mean(axis=0)) / (Xf.std(axis=0) + 1e-9)

pca3 = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(Xstd)
auc_pca3 = knn_r2(pca3, actual_adr)

umap_raw3 = umap.UMAP(n_components=3, random_state=RANDOM_STATE).fit_transform(Xstd)
auc_umap_raw3 = knn_r2(umap_raw3, actual_adr)

top3_feats = list(pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False).head(3).index)
top3_idx = [features.index(f) for f in top3_feats]
auc_shap_top3 = knn_r2(Xstd[:, top3_idx], actual_adr)

auc_umap_shap3 = knn_r2(emb_scaled, actual_adr)
auc_shap_full51 = knn_r2(shap_values, actual_adr)

recon_mae = float(mean_absolute_error(actual_adr, pred_adr))
recon_r2 = float(r2_score(actual_adr, pred_adr))

r2 = {
    "pca3": round(auc_pca3, 3),
    "umap_raw3": round(auc_umap_raw3, 3),
    "shap_top3": round(auc_shap_top3, 3),
    "umap_shap3": round(auc_umap_shap3, 3),
    "shap_full51": round(auc_shap_full51, 3),
    "recon_r2": recon_r2,
    "recon_mae": recon_mae,
    "real_test_r2": bundle["metrics"]["test_r2"],
    "real_test_mae": bundle["metrics"]["test_mae"],
}

mean_abs_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(-mean_abs_shap)
ranking = [
    {
        "name": features[j],
        "mean_abs_shap": round(float(mean_abs_shap[j]), 3),
        "rank": rank + 1,
        "reconstructed": features[j] in RECONSTRUCTED_COLS or features[j].startswith("market_seg_"),
    }
    for rank, j in enumerate(order)
]

# ---- 7. interactive beeswarm data: real per-row SHAP + feature value for the
# top BEESWARM_N features, so the page can draw its own hoverable/clickable
# beeswarm instead of just showing a picture of one. Each feature's "shap"
# and "val" arrays are in the same row order as `points` above (both built
# from the same `range(len(ev))` loop), so the page looks up account/actual/
# predicted by array index rather than duplicating those fields per feature.
BEESWARM_N = 18
beeswarm_idx = order[:BEESWARM_N]
beeswarm = {
    "features": [
        {
            "name": features[j],
            "mean_abs_shap": round(float(mean_abs_shap[j]), 3),
            "shap": [round(float(v), 3) for v in shap_values[:, j]],
            "val": [round(float(v), 4) for v in X.values[:, j].astype(float)],
        }
        for j in beeswarm_idx
    ],
}

# ---- 7b. beeswarm plot -> real PNG file too (no base64 - loaded via <img src>),
# kept as a static, downloadable fallback; the page's own interactive chart is
# built client-side from `beeswarm` above ----
print("Rendering SHAP beeswarm summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, feature_names=features, max_display=18, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=130)
plt.close()

# ---- 8. assemble + write ----
data = {
    "points": points,
    "median_adr": round(float(np.median(actual_adr)), 2),
    "default_idx": int(np.argmin(np.abs(actual_adr - np.median(actual_adr)))),
    "base_value": round(float(explainer.expected_value), 2),
    "ranking": ranking,
    "beeswarm": beeswarm,
    "r2": r2,
    "n_total": len(ev),
    "cluster_info": cluster_info,
}

with open("adr_shap_umap_data.js", "w") as f:
    f.write("// Generated by adr_shap_umap_build.py - do not hand-edit.\n")
    f.write("window.ADR_VIZ_DATA = ")
    json.dump(data, f, separators=(",", ":"))
    f.write(";\n")

print("DONE. Wrote adr_shap_umap_data.js and shap_beeswarm.png "
      "- put both next to adr_shap_umap_3d1_fixed.html and reopen it.")
