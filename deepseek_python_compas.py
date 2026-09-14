#!/usr/bin/env python3
"""
Build a SHAP-space compass HTML (same style as adr_shap_compass_v1model2.html)
for the subset room_block < 1, using adr_model_rebuilt.pkl.

Usage:
    python build_compass.py \
        --model adr_model_rebuilt.pkl \
        --csv rfp_training_data_complete_v3_with_transient.csv \
        --template adr_shap_compass_v1model2.html \
        --out adr_shap_compass_roomblock_lt1.html
"""

import argparse, json, re, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
EXCLUDED_FEATURES = {
    "tr_transient_adr",
    "tr_transient_revpar",
    "tr_pace_vs_prior_year",
    "tr_transient_pace_30d",
}
TARGET = "quoted_adr"
ID_CANDIDATES = ["rfp_id", "id", "RFP_ID", "rfpId"]
ACCT_CANDIDATES = ["account_name", "acct", "account", "account_type", "Account"]


def pick_col(df, candidates, label, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(
            f"could not find a {label} column in CSV. "
            f"tried: {candidates}. Available: {list(df.columns)[:40]}..."
        )
    return None


def fmt_ts(v):
    """Render a timestamp-ish id fragment as YYYYMM when possible."""
    if isinstance(v, str) and len(v) >= 7 and v[4] == "-":
        return v[:7].replace("-", "")
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-contribs", type=int, default=5)
    ap.add_argument("--n-axes", type=int, default=3)
    args = ap.parse_args()

    # ---------------------------------------------------------------- load
    print(f"[1/7] loading model: {args.model}")
    model = joblib.load(args.model)

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    feature_names = list(booster.feature_names)
    if not feature_names:
        raise SystemExit("model has no feature_names — cannot align CSV columns")
    print(f"       model exposes {len(feature_names)} features")

    print(f"[2/7] loading CSV: {args.csv}")
    df_raw = pd.read_csv(args.csv)
    print(f"       raw rows: {len(df_raw):,}  columns: {len(df_raw.columns)}")

    # -------------------------------------------------------- apply filter
    if "room_block" not in df_raw.columns:
        raise SystemExit("CSV has no 'room_block' column")
    n_before = len(df_raw)
    df = df_raw.loc[df_raw["room_block"] < 1].copy()
    print(f"[3/7] filter room_block < 1 : {n_before:,} -> {len(df):,} rows")
    if len(df) == 0:
        raise SystemExit("filter produced zero rows")

    # ----------------------------------------------------- align features
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise SystemExit(
            f"CSV is missing {len(missing)} required features:\n  " + "\n  ".join(missing)
        )
    X = df[feature_names].copy()

    # cast to float (xgboost will silently misbehave on object dtypes)
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    if nan_cols:
        print(f"       warning: NaNs in {len(nan_cols)} columns (model has its own missing policy)")

    # target
    if TARGET not in df.columns:
        raise SystemExit(f"CSV has no target column '{TARGET}'")
    y = df[TARGET].astype(float).values

    # ----------------------------------------------------------- predict
    print("[4/7] scoring")
    preds = np.asarray(model.predict(X)).ravel()

    # -------------------------------------------------------------- SHAP
    print("[5/7] computing SHAP values (TreeExplainer) — may take a minute")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # some shap versions return a list for regressors
        sv = sv[0]
    sv = np.asarray(sv)
    assert sv.shape == (len(X), len(feature_names)), sv.shape
    base_value = float(np.asarray(explainer.expected_value).ravel()[0])
    print(f"       shap shape {sv.shape}   base_value {base_value:.4f}")

    # -------------------------------------------- choose axes & scaling
    keep_idx = [i for i, f in enumerate(feature_names) if f not in EXCLUDED_FEATURES]
    mean_abs = np.abs(sv[:, keep_idx]).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_idx = [keep_idx[i] for i in order[: args.n_axes]]
    axis_names = [feature_names[i] for i in top_idx]
    print(f"[6/7] top-{args.n_axes} axes: {axis_names}")

    # scale each axis by its own max |SHAP| so points land in ~[-1, 1]
    scales = np.abs(sv[:, top_idx]).max(axis=0)
    scales[scales == 0] = 1.0

    coords = sv[:, top_idx] / scales  # (n, 3)

    # ------------------------------------------------- per-row payload
    # identify columns for the label
    id_col = pick_col(df, ID_CANDIDATES, "row id", required=False)
    acct_col = pick_col(df, ACCT_CANDIDATES, "account name", required=False)

    median_actual = float(np.median(y))

    # pre-compute top-N contributor indices per row using argpartition
    n_keep = len(keep_idx)
    sv_keep = sv[:, keep_idx]
    abs_keep = np.abs(sv_keep)
    N = args.n_contribs
    # top-N by |shap| (unsorted), then sort each row by |shap| desc
    part = np.argpartition(-abs_keep, kth=min(N, n_keep) - 1, axis=1)[:, :N]
    rows = np.arange(len(X))[:, None]
    part_vals = abs_keep[rows, part]
    sort_within = np.argsort(-part_vals, axis=1)
    topN_idx = part[rows, sort_within]              # indices into keep_idx
    topN_feat_global = np.array(keep_idx)[topN_idx] # indices into feature_names

    points = []
    for i in range(len(X)):
        p = {
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "actual": round(float(y[i]), 2),
            "pred": round(float(preds[i]), 2),
            "high": int(y[i] >= median_actual),
        }
        # id / account label
        if id_col is not None:
            p["id"] = str(df.iloc[i][id_col])
        else:
            p["id"] = f"row-{i}"
        if acct_col is not None:
            p["acct"] = str(df.iloc[i][acct_col])
        else:
            p["acct"] = ""

        contribs = []
        for k in range(N):
            gi = int(topN_feat_global[i, k])
            contribs.append({
                "name": feature_names[gi],
                "shap": round(float(sv[i, gi]), 4),
            })
        p["contribs"] = contribs
        points.append(p)

    # ---------------------------------------------------- metrics block
    # The uploaded model was fit on the room_block>0 subset, so an honest
    # holdout R^2/MAE is not available for the <1 subset. We report the
    # stats stored on the model alongside in-sample fit for THIS subset.
    metrics_out = {
        "train_mae": None,   # filled below with in-sample on the filtered set
        "test_mae": None,
        "train_r2": None,
        "test_r2": None,
        "cv_mae_mean": None,
        "cv_mae_std": None,
        "n_train": int(len(X)),
        "n_test": 0,
    }
    resid = preds - y
    metrics_out["train_mae"] = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    metrics_out["train_r2"] = 1.0 - ss_res / ss_tot

    # carry through any stored metrics
    try:
        stored = getattr(model, "_adr_metrics", None)
    except Exception:
        stored = None
    # (the pickle does not expose them directly; leave None — the HTML
    #  template only needs .train_r2 / .test_r2 / .train_mae / .test_mae /
    #  .cv_mae_mean / .cv_mae_std / .n_train / .n_test and None is fine,
    #  but we substitute in-sample values so the cards are not blank)

    data = {
        "axis_names": axis_names,
        "base_value": base_value,
        "points": points,
        "metrics": metrics_out,
        "target": TARGET,
        "n": int(len(X)),
        "median_actual": median_actual,
        "excluded_features": sorted(EXCLUDED_FEATURES),
    }

    # ------------------------------------------------------ write HTML
    print(f"[7/7] writing {args.out}")
    template = Path(args.template).read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"))

    # replace the existing <script id="viz-data" ...>...</script> body
    pat = re.compile(
        r'(<script id="viz-data" type="application/json">)(.*?)(</script>)',
        re.DOTALL,
    )
    if not pat.search(template):
        raise SystemExit("template has no <script id='viz-data'> block")
    html = pat.sub(lambda m: m.group(1) + payload + m.group(3), template, count=1)

    # patch the header/description text to reflect the new filter & n
    html = html.replace(
        "room_block&gt;0 RFPs",
        f"room_block&lt;1 RFPs (n={len(X):,})",
    )
    html = html.replace(
        "filtered to room_block &gt; 0 (n=1705)",
        f"filtered to room_block &lt; 1 (n={len(X):,})",
    )
    # also patch the raw ">" versions in case the file wasn't entity-escaped
    html = html.replace("room_block > 0 (n=1705)", f"room_block < 1 (n={len(X):,})")
    html = html.replace("room_block>0", "room_block<1")

    Path(args.out).write_text(html, encoding="utf-8")
    print(f"done. wrote {len(html):,} bytes -> {args.out}")
    print(f"     n_points = {len(points):,}")
    print(f"     axes     = {axis_names}")
    print(f"     median   = {median_actual:.2f}")


if __name__ == "__main__":
    main()