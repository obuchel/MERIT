"""
rebuild_adr_pipeline.py

Re-runs the REAL feature-selection + training pipeline (candidate pool ->
correlation pruning -> quick-XGBoost importance screening -> VIF pruning ->
leaky-feature exclusion -> Optuna-tuned XGBoost training), using the actual
verbatim functions from the user's own rfp_adr_pipeline_filtered_first_final.py
(drop_correlated, prune_vif with all 4 determinism/correctness fixes,
prune_features, exclude_leaky, train_model, build_candidate_pool) — imported
directly from that file, not reimplemented — with ONE deliberate addition:
the 3 rows identified as genuine data-quality problems
(B-202309-62288, B-202310-34967, B-202511-72086; see the earlier flagged-rows
analysis) are excluded BEFORE the candidate pool is built, so every pruning
decision (correlation, importance ranking, VIF) is made on the exact
population the final model is trained on — the same "filter first" principle
the original script already established for the room_block>0 filter.

What's genuinely reproduced vs. approximated
==============================================
The original pipeline's load_data()/join_transient() need two raw files this
script doesn't have (Enhanced_All_Events_v8_final.csv, Nexus_Transient_Demand_v2.csv)
— only the already-joined rfp_training_data_complete_v3_with_transient.csv is
available. That file already contains the output of engineer_features() for
almost every column EXCEPT: the market_segment one-hot dummies, and a few
label-encoded columns (length_of_stay_category_enc, event_format_enc). Those
are reconstructed directly below (verbatim logic to train_adr_model.py).
Four columns genuinely can't be reconstructed at all without the raw
transient-demand time series (tr_transient_adr, tr_transient_revpar,
tr_pace_vs_prior_year, tr_transient_pace_30d) — build_candidate_pool()'s own
>=50%-missing cutoff drops them automatically, exactly as it would if they
were degraded upstream; no special-casing needed. `tr_demand_tier` is kept as
a labeled APPROXIMATION (coded from the RFP's own Demand_Tier column, not the
true transient-market demand tier from the missing raw file) — flagged in the
output bundle.

Everything past that point — drop_correlated, the importance-screening
XGBoost, prune_vif (including the revision-3 SVD rank-deficiency check and
the revision-4 explicit standardization), exclude_leaky, and train_model's
Optuna search — runs UNCHANGED, imported directly from the user's own
pipeline file.

Usage:
    python rebuild_adr_pipeline.py \
        --data rfp_training_data_complete_v3_with_transient.csv \
        --pipeline rfp_adr_pipeline_filtered_first_final.py \
        --out-model adr_model_rebuilt_v2.pkl \
        --out-json funnel_payload_rebuilt.json
"""
import argparse
import importlib.util
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

EXCLUDE_RFP_IDS = ["B-202309-62288", "B-202310-34967", "B-202511-72086"]

# The original pipeline's candidate pool includes ~19 tr_*-prefixed columns from a
# raw transient-demand time series file we don't have. Rather than guess which ones
# survived into this already-joined CSV, NOT_RECONSTRUCTABLE below is computed after
# loading, from whichever expected tr_* columns are actually absent — nothing here
# is fabricated as a stand-in for a column this CSV doesn't genuinely have.
EXPECTED_TR_COLS = [
    "tr_transient_adr", "tr_market_adr", "tr_adr_index", "tr_transient_occ_of_available",
    "tr_total_occupancy_pct", "tr_rooms_to_capacity", "tr_transient_rooms_turned_away",
    "tr_transient_yield_pct", "tr_displacement_cost_per_room", "tr_pace_index",
    "tr_pace_vs_prior_year", "tr_market_occ_pct", "tr_transient_revpar",
    "tr_group_rooms_on_books", "tr_transient_pace_30d",
    "tr_demand_tier", "tr_demand_tier_3", "tr_rate_strategy", "tr_displacement_pressure",
]


def load_pipeline_module(path):
    spec = importlib.util.spec_from_file_location("pipeline_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def engineer_features_full(df, pipe):
    """As close to the real engineer_features() (stage 3 of the pipeline) as this CSV
    allows. Reuses pipe.IDENTITY_COLS so the object-column skip-list is guaranteed
    identical to the original, not re-typed by hand. Columns this CSV already carries
    from an earlier run of the real pipeline (arrival_month_sin, is_shoulder_season,
    budget_ratio, pricing_pressure_index, revenue_quality_score, ...) are left as-is.
    What's added here is only what's missing: the blanket per-object-column label
    encoding, the group_size_tier split, and the full market_segment one-hot set.
    NOTHING is fabricated for columns this CSV genuinely doesn't have — those are
    left NaN and drop out of the candidate pool on their own (see NOT_RECONSTRUCTABLE
    below, which is now computed, not guessed)."""
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


def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return None
    xm, ym = x[mask], y[mask]
    if xm.std() < 1e-12 or ym.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(xm, ym)[0, 1])


def spearman_rho(x, y):
    x = pd.Series(x).rank()
    y = pd.Series(y).rank()
    return pearson_r(x.values, y.values)


def jsonable(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if (np.isnan(v) or np.isinf(v)) else round(float(v), 6)
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rfp_training_data_complete_v3_with_transient.csv")
    ap.add_argument("--pipeline", default="rfp_adr_pipeline_filtered_first_final.py")
    ap.add_argument("--out-model", default="adr_model_rebuilt_v2.pkl")
    ap.add_argument("--out-json", default="funnel_payload_rebuilt.json")
    ap.add_argument("--scratch", default="pipeline_scratch")
    args = ap.parse_args()

    pipe = load_pipeline_module(args.pipeline)
    scratch = Path(args.scratch)
    scratch.mkdir(exist_ok=True)
    pipe.OUTPUT_DIR = scratch  # redirect its internal CSV writes away from cwd

    print("Loading raw data...")
    df_raw = pd.read_csv(args.data)
    df_raw["rfp_id"] = df_raw["rfp_id"].astype(str)

    print("Engineering features on the FULL population first (so categorical codes / "
          "one-hot columns are derived once, consistently, before any row is split off)...")
    df_eng = engineer_features_full(df_raw, pipe)
    not_reconstructable = [c for c in EXPECTED_TR_COLS if c not in df_eng.columns]
    print(f"  tr_* columns not reconstructable from this CSV ({len(not_reconstructable)}): {not_reconstructable}")

    print(f"Excluding {len(EXCLUDE_RFP_IDS)} flagged rows: {EXCLUDE_RFP_IDS}")
    missing = set(EXCLUDE_RFP_IDS) - set(df_eng["rfp_id"])
    if missing:
        print(f"  WARNING: not found: {sorted(missing)}")
    ev = df_eng[~df_eng["rfp_id"].isin(EXCLUDE_RFP_IDS)].reset_index(drop=True)
    excl_df = df_eng[df_eng["rfp_id"].isin(EXCLUDE_RFP_IDS)].reset_index(drop=True)
    rows_before_filter = len(ev)

    print(">>> Filtering room_block > 0 BEFORE pool/pruning (filter-first, matches the original script's own design).")
    ev_adr = ev[ev["room_block"] > 0].copy().reset_index(drop=True)
    rows_after_filter = len(ev_adr)
    rows_excluded = rows_before_filter - rows_after_filter
    print(f"  {rows_before_filter} -> {rows_after_filter} rows ({rows_excluded} meeting-only rows excluded)")

    print("\n[Stage: candidate pool]")
    pool = pipe.build_candidate_pool(ev_adr)

    print("\n[Stage: correlation pruning + importance screening + VIF pruning]")
    base = pipe.prune_features(ev_adr, pool, tag="rebuilt_")

    print("\n[Stage: leaky-feature exclusion]")
    adr_feats = pipe.exclude_leaky(base, "quoted_adr")
    dropped_leaky = sorted(set(base) - set(adr_feats))
    print(f"  Leaky-excluded: {dropped_leaky if dropped_leaky else '(none)'}")
    print(f"  Final feature list: {len(adr_feats)} features")

    print("\n[Stage: train final model — Optuna search]")
    result = pipe.train_model("Quoted ADR (rebuilt, outliers excluded)", ev, adr_feats,
                               "quoted_adr", filter_rooms=True)

    # ---- read back the intermediate stage artifacts prune_features() wrote to scratch ----
    corr_log = pd.read_csv(scratch / "rebuilt_multicollinearity_correlation_pairs.csv")
    imp_screen = pd.read_csv(scratch / "rebuilt_importance_pool_screening.csv")
    imp_screen.columns = ["feature", "importance"]
    vif_final = pd.read_csv(scratch / "rebuilt_multicollinearity_vif_final.csv")
    vif_removed_path = scratch / "rebuilt_multicollinearity_vif_removed.csv"
    vif_removed = pd.read_csv(vif_removed_path) if vif_removed_path.exists() else pd.DataFrame(columns=["feature", "vif"])

    kept_corr = sorted([c for c in pool if c not in set(corr_log["dropped"])])
    top60 = imp_screen.sort_values("importance", ascending=False).head(60)["feature"].tolist()

    y_all = ev_adr["quoted_adr"].astype(float).values
    stats = {}
    points = {"quoted_adr": [jsonable(v) for v in y_all]}
    for f in pool:
        xv = pd.to_numeric(ev_adr[f], errors="coerce").values.astype(float)
        stats[f] = {
            "pearson_r": pearson_r(xv, y_all),
            "spearman_rho": spearman_rho(xv, y_all),
            "nunique": int(pd.Series(xv).nunique()),
            "is_binary": bool(pd.Series(xv).nunique() <= 2),
        }
        points[f] = [jsonable(v) for v in xv]

    # extra context fields already used by the page's tooltip, if present
    for f in ["rate_discount_pct", "lead_time_days", "account_avg_revenue", "account_win_rate",
              "segment_win_rate", "attendees", "room_block", "nights"]:
        if f in ev_adr.columns and f not in points:
            points[f] = [jsonable(v) for v in ev_adr[f]]

    # excluded-from-training marker: none of ev_adr's rows are excluded (we removed them
    # upstream, before the pool was even built) — but the 3 excluded rows still deserve to
    # be SHOWN on the page, out-of-sample, per your request. Score them with the trained
    # model and append them to every points array so they render as distinct markers.
    # excl_df was already engineered together with ev (from the same df_eng call above),
    # so its categorical codes / one-hot columns are guaranteed consistent with adr_feats.
    is_excluded_flags = [False] * len(ev_adr)
    if len(excl_df):
        X_excl = pipe.prep_X(excl_df, adr_feats)
        pred_excl = result["model"].predict(X_excl)
        points["quoted_adr"] += [jsonable(v) for v in excl_df["quoted_adr"].astype(float)]
        for f in pool:
            if f in excl_df.columns:
                xv = pd.to_numeric(excl_df[f], errors="coerce").values.astype(float)
            else:
                xv = np.full(len(excl_df), np.nan)
            points[f] += [jsonable(v) for v in xv]
        for f in ["rate_discount_pct", "lead_time_days", "account_avg_revenue", "account_win_rate",
                  "segment_win_rate", "attendees", "room_block", "nights"]:
            if f in points:
                if f in excl_df.columns:
                    points[f] += [jsonable(v) for v in excl_df[f]]
                else:
                    points[f] += [None] * len(excl_df)
        points["predicted_adr_excluded"] = [None] * len(ev_adr) + [jsonable(v) for v in pred_excl]
        points["rfp_id"] = [None] * len(ev_adr) + list(excl_df["rfp_id"])
        is_excluded_flags += [True] * len(excl_df)
    points["is_excluded_from_training"] = is_excluded_flags

    final_importance = result["importance"][["feature", "importance", "importance_pct", "rank"]].to_dict("records")

    funnel = {
        "rows_before_filter": int(rows_before_filter),
        "rows_after_filter": int(rows_after_filter),
        "rows_excluded": int(rows_excluded),
        "excluded_rfp_ids": EXCLUDE_RFP_IDS,
        "pool": pool,
        "pool_size": len(pool),
        "corr_pairs": corr_log.to_dict("records"),
        "kept_corr": kept_corr,
        "kept_corr_size": len(kept_corr),
        "importance_screening": imp_screen.sort_values("importance", ascending=False).to_dict("records"),
        "top60": top60,
        "vif_removed": [
            {"feature": r["feature"], "vif": ("inf" if (isinstance(r["vif"], str) or np.isinf(r["vif"])) else round(float(r["vif"]), 2))}
            for r in vif_removed.to_dict("records")
        ],
        "vif_final": vif_final.to_dict("records"),
        "vif_final_size": len(vif_final),
        "leaky_dropped": dropped_leaky,
        "adr_feats": adr_feats,
        "adr_feats_size": len(adr_feats),
        "final_importance": final_importance,
        "metrics": result["metrics"],
        "best_params": result["best_params"],
    }

    payload = {
        "funnel": funnel,
        "stats": stats,
        "adr_mean": float(np.mean(y_all)),
        "adr_std": float(np.std(y_all)),
        "points": points,
    }
    with open(args.out_json, "w") as f:
        json.dump(payload, f)
    print(f"\nWrote {args.out_json}")

    bundle = {
        "model": result["model"],
        "features": result["features"],
        "target": "quoted_adr",
        "best_params": result["best_params"],
        "metrics": result["metrics"],
        "candidate_pool_size": len(pool),
        "base_pool_size": len(base),
        "leaky_dropped": dropped_leaky,
        "excluded_rfp_ids": EXCLUDE_RFP_IDS,
        "not_reconstructable_features": NOT_RECONSTRUCTABLE,
        "tr_demand_tier_is_approximation": True,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": {"xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(args.out_model, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Wrote {args.out_model}")
    print(f"\nFinal: {len(adr_feats)} features | test_r2={result['metrics']['test_r2']:.4f} | "
          f"test_mae=${result['metrics']['test_mae']:.2f}")


if __name__ == "__main__":
    main()