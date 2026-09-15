"""
adr_shap_umap_build_rebuilt.py

Adapted from adr_shap_umap_build.py for the REBUILT ADR model
(adr_model_rebuilt.pkl / adr_model_rebuilt_v2.pkl — whichever bundle you
point --model at), i.e. the model produced by build_funnel_adr_website.py /
rebuild_funnel_adr.py, NOT the original adr_filtered_first_model.pkl.

Same output, same purpose: builds adr_shap_umap_data.js + shap_beeswarm.png
for the ADR model's SHAP-space compass (adr_shap_umap_3d1_fixed.html).
Run this whenever your rebuilt model bundle or the underlying CSVs change,
then re-open the HTML.

Why this couldn't just reuse the original script unmodified
==============================================================
The original script rebuilds "the exact scored population" by calling
load_data() -> join_transient() -> engineer_features() straight from
rfp_adr_pipeline_filtered_first_final.py with NO arguments — those
functions read two raw files (V3_FILE, EVENTS_FILE) via hardcoded global
paths that only exist in the original training environment. We don't have
those two files; we only have the already-merged
rfp_training_data_complete_v3_with_transient.csv. That's exactly the
constraint build_funnel_adr_website.py's run_pipeline() was written
around, and THIS model bundle was produced by that exact reconstruction
(real_join_transient() + engineer_features_full()), not by the original
run_adr_pipeline(). So to reconstruct the SAME scored population this
model bundle was actually trained/evaluated on, this script mirrors
run_pipeline()'s data-prep exactly rather than calling the original
load_data()/engineer_features():

  1. read --data directly (no V3_FILE/EVENTS_FILE merge — already merged)
  2. real_join_transient(): the pipeline's own join_transient(), run
     verbatim against --transient-data (skip with --no-transient-join if
     your bundle's not_reconstructable_features isn't empty — see below)
  3. engineer_features_full(): the same blanket per-object-column label
     encoding + market_segment one-hot + group_size_tier split used to
     build this model's candidate pool (NOT the original engineer_features(),
     which needs columns only the original V3_FILE/EVENTS_FILE merge
     produces)
  4. exclude the 3 flagged rows (EXCLUDE_RFP_IDS), then room_block>0 —
     same order as run_pipeline()
  5. pipe.prep_X(ev, bundle["features"]) — the real pipeline's own prep_X,
     imported live, unmodified

Two smaller adaptations from the original script:
  - ev["did_convert"] doesn't exist in this CSV (it's a column the original
    V3_FILE/EVENTS_FILE merge apparently produced that isn't present here).
    ev["is_won"] is the closest real substitute (same win/loss semantics)
    and IS present, so the "won" tooltip field uses that instead. If your
    CSV has neither, it falls back to all-zero rather than guessing.
  - RECONSTRUCTED_COLS (used to flag engineered/derived features in the
    ranking output) is computed dynamically from bundle["features"] instead
    of hardcoded, since which columns are "derived" (tr_*, market_seg_*,
    *_enc) differs between model versions (e.g. adr_model_rebuilt_v2.pkl
    has 16 genuinely-missing tr_* columns; adr_model_rebuilt.pkl, built
    after the real transient join, has none).

Everything else — SHAP computation, UMAP embedding, the DBSCAN eps-sweep,
cluster tagging (same reverse-engineered rule, now flagged if it can't find
the columns it needs), comparison metrics, and the interactive beeswarm
data — is unchanged from adr_shap_umap_build.py.

Usage
=====
    python adr_shap_umap_build_rebuilt.py \\
        --model adr_model_rebuilt.pkl \\
        --data rfp_training_data_complete_v3_with_transient.csv \\
        --pipeline rfp_adr_pipeline_filtered_first_final.py \\
        --transient-data Nexus_Transient_Demand_v2.csv \\
        --out-js adr_shap_umap_data.js \\
        --out-png shap_beeswarm.png

If your bundle's not_reconstructable_features is non-empty (e.g. the older
adr_model_rebuilt_v2.pkl, pool_size 282), either still pass --transient-data
(the extra real tr_* columns just won't be in bundle["features"], so they're
computed but unused — harmless) or pass --no-transient-join to skip the
join entirely and use whatever partial tr_* columns --data already has,
matching exactly how that older bundle was actually built.
"""
import argparse
import importlib.util
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import umap

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score, mean_absolute_error

EXCLUDE_RFP_IDS = ["B-202309-62288", "B-202310-34967", "B-202511-72086"]


# ═══════════════════════════════════════════════════════════════════════════
# Data prep — mirrors build_funnel_adr_website.py's run_pipeline() exactly,
# since that's what actually produced this model bundle's training population
# ═══════════════════════════════════════════════════════════════════════════

def load_pipeline_module(path):
    spec = importlib.util.spec_from_file_location("pipeline_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real_join_transient(df_raw, pipe, transient_path):
    """The pipeline's ACTUAL join_transient(), run verbatim against the raw
    transient-demand file — see build_funnel_adr_website.py for the full
    rationale (same function, copied here so this script has no import
    dependency on that one)."""
    df = df_raw.copy()

    pre_existing_tr = [c for c in df.columns if c.lower().startswith("tr_")]
    if pre_existing_tr:
        print(f"  Dropping {len(pre_existing_tr)} pre-existing (partial) tr_* columns "
              f"before re-joining from the real raw file: {pre_existing_tr}")
        df = df.drop(columns=pre_existing_tr)

    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"] if "Arrival_Date" in df.columns else df["arrival_date"])
    df["Departure_Date"] = pd.to_datetime(df["Departure_Date"])

    pipe.TRANSIENT_FILE = transient_path
    df_joined = pipe.join_transient(df)

    new_tr = [c for c in df_joined.columns if c.lower().startswith("tr_") or c in pipe.TR_CAT_MAPS]
    print(f"  join_transient() produced {len(new_tr)} tr_* columns from the real raw file.")
    return df_joined


def engineer_features_full(df, pipe):
    """Same function as in build_funnel_adr_website.py — reproduced here so
    this script has no import dependency on that one. Reuses
    pipe.IDENTITY_COLS so the skip-list matches exactly."""
    df = df.copy()

    if "group_size_tier" in df.columns:
        df["is_meeting_only"] = (df["group_size_tier"] == "meeting_only").astype(int)
        size_order = {"small": 0, "medium": 1, "large": 2, "very_large": 3}
        df["group_size_tier_clean_enc"] = df["group_size_tier"].map(size_order).fillna(-1).astype(int)

    if "market_segment" in df.columns:
        ohe = pd.get_dummies(df["market_segment"], prefix="market_seg").astype(int)
        df = pd.concat([df, ohe], axis=1)

    skip_encode = pipe.IDENTITY_COLS | {"group_size_tier", "market_segment"}
    for c in df.select_dtypes(include="object").columns:
        if c in skip_encode:
            continue
        df[f"{c}_enc"] = pd.Categorical(df[c]).codes

    if "total_room_nights" not in df.columns:
        df["total_room_nights"] = df.get("room_block", df.get("total_room_nights_requested", np.nan))

    return df


def rebuild_scored_population(data_path, pipeline_path, transient_path, do_transient_join):
    pipe = load_pipeline_module(pipeline_path)

    print("Loading raw data...")
    df_raw = pd.read_csv(data_path)
    df_raw["rfp_id"] = df_raw["rfp_id"].astype(str)

    if do_transient_join:
        print(f"Joining REAL transient-demand data from {transient_path} "
              f"(pipe.join_transient(), run verbatim)...")
        df_raw = real_join_transient(df_raw, pipe, transient_path)
    else:
        print("Skipping the transient join (--no-transient-join): using whatever "
              "tr_* columns --data already has, as-is.")

    print("Engineering features on the full population...")
    df_eng = engineer_features_full(df_raw, pipe)

    print(f"Excluding {len(EXCLUDE_RFP_IDS)} flagged rows: {EXCLUDE_RFP_IDS}")
    ev = df_eng[~df_eng["rfp_id"].isin(EXCLUDE_RFP_IDS)].reset_index(drop=True)
    ev = ev[ev["room_block"] > 0].copy().reset_index(drop=True)
    print(f"  {len(ev)} rows in the scored population "
          f"(3 flagged rows excluded, room_block>0 filter applied)")

    return pipe, ev


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def tag_cluster(compression_pct, discount_pct, demand_tier):
    """Reverse-engineered against the 10 clusters in the ORIGINAL page (see
    adr_shap_umap_build.py's docstring) — not recovered from original source,
    a plausible rule kept unchanged here. demand_tier will be 0.0 (and this
    rule correspondingly less informative) if tr_demand_tier isn't in the
    scored population's columns — see the caller below."""
    if compression_pct >= 50:
        return "Compression dates"
    if discount_pct >= 15:
        return "Deep discount"
    if discount_pct <= 5 and demand_tier < 0.5:
        return "Low demand tier"
    if discount_pct <= 1 and demand_tier >= 0.95 and compression_pct < 5:
        return "High demand, list price"
    return "Mixed"


def knn_r2(Z, y, random_state, k=5):
    cv = KFold(5, shuffle=True, random_state=random_state)
    scores = []
    for tr, te in cv.split(Z):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(Z[tr], y[tr])
        scores.append(r2_score(y[te], knn.predict(Z[te])))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="adr_model_rebuilt.pkl (or _v2.pkl, etc.)")
    ap.add_argument("--data", required=True, help="rfp_training_data_complete_v3_with_transient.csv")
    ap.add_argument("--pipeline", required=True, help="rfp_adr_pipeline_filtered_first_final.py")
    ap.add_argument("--transient-data", default=None, help="Nexus_Transient_Demand_v2.csv")
    ap.add_argument("--no-transient-join", action="store_true",
                     help="Skip join_transient() entirely (use for a bundle whose "
                          "not_reconstructable_features is non-empty, e.g. an older "
                          "_v2 bundle built before the real transient join)")
    ap.add_argument("--out-js", default="adr_shap_umap_data.js")
    ap.add_argument("--out-png", default="shap_beeswarm.png")
    ap.add_argument("--beeswarm-n", type=int, default=18)
    args = ap.parse_args()

    do_join = args.transient_data is not None and not args.no_transient_join

    print("Loading ADR model bundle...")
    with open(args.model, "rb") as fh:
        bundle = pickle.load(fh)
    model = bundle["model"]
    features = bundle["features"]
    random_state = bundle.get("best_params", {}).get("random_state", 42)
    print(f"  {len(features)} features, target={bundle.get('target')}, "
          f"trained_at={bundle.get('trained_at_utc')}")

    pipe, ev = rebuild_scored_population(args.data, args.pipeline, args.transient_data, do_join)

    missing_feats = [f for f in features if f not in ev.columns]
    if missing_feats:
        raise SystemExit(
            f"{len(missing_feats)} of this bundle's features aren't in the reconstructed "
            f"population: {missing_feats}\nThis usually means --transient-data was omitted "
            f"(or --no-transient-join passed) for a bundle that needs the real join, or "
            f"vice versa. Check bundle['not_reconstructable_features'] to see what this "
            f"bundle expects.")

    X = pipe.prep_X(ev, features)
    actual_adr = ev["quoted_adr"].astype(float).values
    if "is_won" in ev.columns:
        won = ev["is_won"].astype(int).values
    elif "did_convert" in ev.columns:
        won = ev["did_convert"].astype(int).values
    else:
        print("  NOTE: neither is_won nor did_convert found — 'won' tooltip field will be 0 for every row.")
        won = np.zeros(len(ev), dtype=int)
    pred_adr = model.predict(X)

    print("Computing SHAP values (TreeExplainer, ADR units)...")
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_values = explainer.shap_values(X)
    recon = explainer.expected_value + shap_values.sum(axis=1)
    print(f"  max abs reconstruction error vs model.predict: {np.max(np.abs(recon - pred_adr)):.6f}")

    print("Running UMAP on all SHAP dimensions -> 3D...")
    reducer = umap.UMAP(n_components=3, random_state=random_state, n_neighbors=15, min_dist=0.1)
    emb = reducer.fit_transform(shap_values)
    emb_scaled = emb / (np.abs(emb).std(axis=0).mean() * 3)

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

    cluster_mean_adr = {c: actual_adr[raw_labels == c].mean() for c in unique_clusters}
    ordered = sorted(unique_clusters, key=lambda c: cluster_mean_adr[c])
    remap = {old: new for new, old in enumerate(ordered)}
    cluster_labels = np.array([remap[c] for c in raw_labels])

    has_demand_tier = "tr_demand_tier" in ev.columns
    if not has_demand_tier:
        print("  NOTE: tr_demand_tier not in the scored population — tag_cluster() will "
              "treat demand_tier as 0.0 for every cluster (less informative tagging).")

    cluster_info = []
    for new_id in range(len(ordered)):
        mask = cluster_labels == new_id
        compression_pct = float(ev.loc[mask, "Is_Compression_Date"].mean() * 100) if "Is_Compression_Date" in ev else 0.0
        discount_pct = float(ev.loc[mask, "rate_discount_pct"].mean()) if "rate_discount_pct" in ev else 0.0
        demand_tier = float(ev.loc[mask, "tr_demand_tier"].mean()) if has_demand_tier else 0.0
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
            "acct": str(ev["account_name"].iloc[i]) if "account_name" in ev.columns else "",
            "x": round(float(emb_scaled[i, 0]), 4),
            "y": round(float(emb_scaled[i, 1]), 4),
            "z": round(float(emb_scaled[i, 2]), 4),
            "won": int(won[i]),
            "actual": round(float(actual_adr[i]), 2),
            "pred": round(float(pred_adr[i]), 2),
            "contribs": contribs,
            "cluster": int(cluster_labels[i]),
        })

    print("Computing comparison metrics (5-fold k-NN R²)...")
    Xf = X.values.astype(float)
    Xstd = (Xf - Xf.mean(axis=0)) / (Xf.std(axis=0) + 1e-9)

    pca3 = PCA(n_components=3, random_state=random_state).fit_transform(Xstd)
    auc_pca3 = knn_r2(pca3, actual_adr, random_state)

    umap_raw3 = umap.UMAP(n_components=3, random_state=random_state).fit_transform(Xstd)
    auc_umap_raw3 = knn_r2(umap_raw3, actual_adr, random_state)

    top3_feats = list(pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False).head(3).index)
    top3_idx = [features.index(f) for f in top3_feats]
    auc_shap_top3 = knn_r2(Xstd[:, top3_idx], actual_adr, random_state)

    auc_umap_shap3 = knn_r2(emb_scaled, actual_adr, random_state)
    auc_shap_full = knn_r2(shap_values, actual_adr, random_state)

    recon_mae = float(mean_absolute_error(actual_adr, pred_adr))
    recon_r2 = float(r2_score(actual_adr, pred_adr))

    r2 = {
        "pca3": round(auc_pca3, 3),
        "umap_raw3": round(auc_umap_raw3, 3),
        "shap_top3": round(auc_shap_top3, 3),
        "umap_shap3": round(auc_umap_shap3, 3),
        f"shap_full{len(features)}": round(auc_shap_full, 3),
        "recon_r2": recon_r2,
        "recon_mae": recon_mae,
        "real_test_r2": bundle["metrics"]["test_r2"],
        "real_test_mae": bundle["metrics"]["test_mae"],
    }

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs_shap)

    # Which features are "derived" rather than raw CSV columns — computed
    # dynamically per-bundle instead of hardcoded, since it differs between
    # model versions (see module docstring).
    reconstructed_cols = {
        f for f in features
        if f.startswith("tr_") or f.startswith("market_seg_") or f.endswith("_enc")
    }

    ranking = [
        {
            "name": features[j],
            "mean_abs_shap": round(float(mean_abs_shap[j]), 3),
            "rank": rank + 1,
            "reconstructed": features[j] in reconstructed_cols,
        }
        for rank, j in enumerate(order)
    ]

    print("Extracting interactive beeswarm data...")
    beeswarm_idx = order[:args.beeswarm_n]
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

    print("Rendering SHAP beeswarm summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=features, max_display=18, show=False)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=130)
    plt.close()

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
        # Exported so the HTML page can state its own clustering parameters instead of a
        # hand-typed number going stale on the next retrain (see the "10 groups"/"70-87%"
        # bug this fixed).
        "dbscan_eps": float(best_eps),
        "dbscan_noise_points": int(best_n_noise),
    }

    with open(args.out_js, "w") as fh:
        fh.write("// Generated by adr_shap_umap_build_rebuilt.py - do not hand-edit.\n")
        fh.write("window.ADR_VIZ_DATA = ")
        json.dump(data, fh, separators=(",", ":"))
        fh.write(";\n")

    print(f"\nDONE. Wrote {args.out_js} and {args.out_png} "
          f"- put both next to adr_shap_umap_3d1_fixed.html and reopen it.")


if __name__ == "__main__":
    main()
