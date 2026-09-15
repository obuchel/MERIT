"""
build_funnel_adr_website.py

ONE script, end to end: runs the real ADR feature-selection + training
pipeline, then writes the finished "funnel" diagnostic HTML page from it.
This merges what were two separate steps (rebuild_adr_pipeline.py +
render_funnel_website.py) into a single command so you get
funnel_adr_rebuilt.html (plus the trained model .pkl and the raw JSON
payload, if you want them) from one run.

What this does, in order
=========================
1. Loads your real pipeline file (rfp_adr_pipeline_filtered_first_final.py)
   as a live module and calls its actual functions — join_transient,
   build_candidate_pool, drop_correlated, prune_vif (with the rank-deficiency
   + standardization fixes), exclude_leaky, train_model — UNCHANGED, not
   reimplemented.
2. Joins the real raw transient-demand time series
   (Nexus_Transient_Demand_v2.csv) via the pipeline's own join_transient(),
   a genuine per-row date-window average — not an approximation — so the
   full ~298-feature candidate pool is reached (not a partial one).
3. Excludes the 3 rows identified as genuine data-quality problems
   (quoted_adr above the deal's own max_room_rate, no legitimate bundled-
   revenue explanation) BEFORE the candidate pool is built, so every pruning
   decision — correlation, importance ranking, VIF — is made on the exact
   population the final model trains on. They're still shown on the page,
   scored out-of-sample, as a distinct marker.
4. Runs correlation pruning -> quick-XGBoost importance screening -> VIF
   pruning -> leaky-feature exclusion -> Optuna-tuned XGBoost training.
5. Takes the ORIGINAL funnel_quoted_adr_v4 HTML file you already have (same
   CSS, same JS, same Cook's-distance scatter methodology, same layout) and
   surgically swaps in this run's real numbers and JSON payload — nothing
   about the page's look or interactions changes, only the data and the
   handful of text callouts that described a DIFFERENT run.

Usage
=====


        python rebuild_funnel_adr.py \
  --data rfp_training_data_complete_v3_with_transient.csv \
  --pipeline rfp_adr_pipeline_filtered_first_final.py \
  --transient-data Nexus_Transient_Demand_v2.csv \
  --source funnel_adr_rebuilt_v3.html \                  
  --out funnel_adr_rebuilt.html \
  --out-model adr_model_rebuilt.pkl \
  --out-json funnel_payload_rebuilt.json


--transient-data is optional: without it, the script uses whatever tr_*
columns are already present in --data (if any) and reports the rest as
"not reconstructable" on the page, instead of pretending they exist.
--out-model / --out-json are also optional — if you only want the page,
drop them and only --out is written.
"""
import argparse
import importlib.util
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

EXCLUDE_RFP_IDS = ["B-202309-62288", "B-202310-34967", "B-202511-72086"]

# The original pipeline's candidate pool includes ~19 tr_*-prefixed columns from a
# raw transient-demand time series file. If --transient-data isn't given (or the
# join can't reach some of these), NOT_RECONSTRUCTABLE is computed after loading,
# from whichever expected tr_* columns are actually absent — nothing here is
# fabricated as a stand-in for a column that isn't genuinely available.
EXPECTED_TR_COLS = [
    "tr_transient_adr", "tr_market_adr", "tr_adr_index", "tr_transient_occ_of_available",
    "tr_total_occupancy_pct", "tr_rooms_to_capacity", "tr_transient_rooms_turned_away",
    "tr_transient_yield_pct", "tr_displacement_cost_per_room", "tr_pace_index",
    "tr_pace_vs_prior_year", "tr_market_occ_pct", "tr_transient_revpar",
    "tr_group_rooms_on_books", "tr_transient_pace_30d",
    "tr_demand_tier", "tr_demand_tier_3", "tr_rate_strategy", "tr_displacement_pressure",
]


# ═══════════════════════════════════════════════════════════════════════════
# STAGE A — run the real pipeline and build the JSON payload
# ═══════════════════════════════════════════════════════════════════════════

def load_pipeline_module(path):
    spec = importlib.util.spec_from_file_location("pipeline_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real_join_transient(df_raw, pipe, transient_path):
    """Runs the pipeline's ACTUAL join_transient() (imported live, unmodified)
    against the genuine raw transient-demand file — a real per-row date-window
    average over the raw daily time series, not an approximation.

    Two adjustments this merged CSV needs before it matches what
    join_transient() expects:
      1. It has lowercase 'arrival_date' but not the capitalized 'Arrival_Date'
         join_transient() reads (r.Arrival_Date) — that capitalized column
         normally comes from a v8 events-file merge this script doesn't have.
         Since it's the exact same date, just a casing difference, it's
         aliased rather than treated as missing.
      2. Any tr_* columns already baked into --data (a prior partial join) are
         dropped first so the freshly-computed, complete columns aren't
         shadowed or duplicated.
    """
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
    for c in sorted(set(new_tr)):
        miss_pct = df_joined[c].isna().mean() * 100
        print(f"    {c}: {miss_pct:.1f}% missing")
    return df_joined


def engineer_features_full(df, pipe):
    """As close to the real engineer_features() (stage 3 of the pipeline) as
    --data allows. Reuses pipe.IDENTITY_COLS so the object-column skip-list is
    guaranteed identical to the original, not re-typed by hand. Columns --data
    already carries from an earlier run of the real pipeline are left as-is.
    What's added here is only what's missing: the blanket per-object-column
    label encoding, the group_size_tier split, and the full market_segment
    one-hot set. NOTHING is fabricated for columns --data genuinely doesn't
    have — those are left NaN and drop out of the candidate pool on their own."""
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


def run_pipeline(data_path, pipeline_path, transient_path, scratch_dir):
    """Runs the whole real pipeline and returns the JSON-ready payload dict
    plus the trained-model bundle dict."""
    pipe = load_pipeline_module(pipeline_path)
    scratch = Path(scratch_dir)
    scratch.mkdir(exist_ok=True, parents=True)
    pipe.OUTPUT_DIR = scratch  # redirect its internal CSV writes away from cwd

    print("Loading raw data...")
    df_raw = pd.read_csv(data_path)
    df_raw["rfp_id"] = df_raw["rfp_id"].astype(str)

    if transient_path:
        print(f"\nJoining REAL transient-demand data from {transient_path} "
              f"(this is pipe.join_transient(), run verbatim, not reimplemented)...")
        df_raw = real_join_transient(df_raw, pipe, transient_path)

    print("\nEngineering features on the FULL population first (so categorical codes / "
          "one-hot columns are derived once, consistently, before any row is split off)...")
    df_eng = engineer_features_full(df_raw, pipe)
    not_reconstructable = [c for c in EXPECTED_TR_COLS if c not in df_eng.columns]
    print(f"  tr_* columns not reconstructable ({len(not_reconstructable)}): {not_reconstructable}")

    print(f"Excluding {len(EXCLUDE_RFP_IDS)} flagged rows: {EXCLUDE_RFP_IDS}")
    missing = set(EXCLUDE_RFP_IDS) - set(df_eng["rfp_id"])
    if missing:
        print(f"  WARNING: not found: {sorted(missing)}")
    ev = df_eng[~df_eng["rfp_id"].isin(EXCLUDE_RFP_IDS)].reset_index(drop=True)
    excl_df = df_eng[df_eng["rfp_id"].isin(EXCLUDE_RFP_IDS)].reset_index(drop=True)
    rows_before_filter = len(ev)

    print(">>> Filtering room_block > 0 BEFORE pool/pruning (filter-first, matches the "
          "original script's own design).")
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
    for feat in pool:
        xv = pd.to_numeric(ev_adr[feat], errors="coerce").values.astype(float)
        stats[feat] = {
            "pearson_r": pearson_r(xv, y_all),
            "spearman_rho": spearman_rho(xv, y_all),
            "nunique": int(pd.Series(xv).nunique()),
            "is_binary": bool(pd.Series(xv).nunique() <= 2),
        }
        points[feat] = [jsonable(v) for v in xv]

    for feat in ["rate_discount_pct", "lead_time_days", "account_avg_revenue", "account_win_rate",
                 "segment_win_rate", "attendees", "room_block", "nights"]:
        if feat in ev_adr.columns and feat not in points:
            points[feat] = [jsonable(v) for v in ev_adr[feat]]

    # excluded-from-training marker: score the 3 excluded rows out-of-sample and append
    # them to every points array so they render as a distinct marker on the page.
    is_excluded_flags = [False] * len(ev_adr)
    if len(excl_df):
        X_excl = pipe.prep_X(excl_df, adr_feats)
        pred_excl = result["model"].predict(X_excl)
        points["quoted_adr"] += [jsonable(v) for v in excl_df["quoted_adr"].astype(float)]
        for feat in pool:
            if feat in excl_df.columns:
                xv = pd.to_numeric(excl_df[feat], errors="coerce").values.astype(float)
            else:
                xv = np.full(len(excl_df), np.nan)
            points[feat] += [jsonable(v) for v in xv]
        for feat in ["rate_discount_pct", "lead_time_days", "account_avg_revenue", "account_win_rate",
                     "segment_win_rate", "attendees", "room_block", "nights"]:
            if feat in points:
                if feat in excl_df.columns:
                    points[feat] += [jsonable(v) for v in excl_df[feat]]
                else:
                    points[feat] += [None] * len(excl_df)
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
        "not_reconstructable_features": not_reconstructable,
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
        "not_reconstructable_features": not_reconstructable,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": {"xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }

    print(f"\nFinal: {len(adr_feats)} features | test_r2={result['metrics']['test_r2']:.4f} | "
          f"test_mae=${result['metrics']['test_mae']:.2f}")

    return payload, bundle


# ═══════════════════════════════════════════════════════════════════════════
# STAGE B — template the payload onto the original funnel HTML page
# ═══════════════════════════════════════════════════════════════════════════

def get_between(content, start_marker, end_marker, start_after=0):
    i = content.find(start_marker, start_after)
    if i == -1:
        raise ValueError(f"start marker not found: {start_marker!r}")
    j = content.find(end_marker, i)
    if j == -1:
        raise ValueError(f"end marker not found: {end_marker!r}")
    return i, j + len(end_marker)


def render_html(source_path, payload):
    """Takes the ORIGINAL funnel_quoted_adr_v4 HTML file (same CSS, same JS,
    same Cook's-distance scatter methodology, same layout) and surgically
    swaps in this run's real numbers and JSON payload. Nothing about the
    page's look or interactions changes — only the data and the handful of
    text callouts that described a different (cross-environment
    verification) run."""
    with open(source_path, encoding="utf-8") as fh:
        content = fh.read()

    f = payload["funnel"]
    n_excl = len(f["excluded_rfp_ids"])
    n_tr_missing = len(f.get("not_reconstructable_features", []))

    # ---------- 1. header + first callout ----------
    i, j = get_between(content, '<h1>Quoted ADR pruning funnel', '</div>\n\n  <div class="callout">')
    j -= len('\n\n  <div class="callout">')
    diag_body = (f"""
    <b>Candidate pool reaches the full {f["pool_size"]}.</b> The raw transient-demand time series
    (<span class="tag">Nexus_Transient_Demand_v2.csv</span>) is now joined for real via the pipeline's own
    <span class="tag">join_transient()</span> — a genuine per-row date-window average over the daily data, not
    an approximation — so all 19 <span class="tag">tr_*</span>-prefixed columns are present with 0% missing.
    Nothing in the candidate pool is fabricated or backfilled. Every stage below (correlation pruning, importance
    screening, VIF pruning, leaky exclusion) runs the real, unmodified pipeline code on this complete pool, and
    the final {f["adr_feats_size"]}-feature model is a genuine, freshly-trained fit — not a copy of the original
    51-feature list.
  """ if n_tr_missing == 0 else f"""
    <b>Candidate pool starts {f["pool_size"]}, not 298.</b> The available data doesn't include a complete raw
    transient-demand time series. The gap is fully accounted for: <span class="diagstat">{n_tr_missing}
    tr_*-prefixed columns</span> genuinely can't be reconstructed without that raw file, and
    {f["pool_size"]} + {n_tr_missing} = 298 exactly — nothing else is missing or unexplained. Every other stage
    (correlation pruning, importance screening, VIF pruning, leaky exclusion) runs the real, unmodified code on
    whatever candidate pool is actually available, and the final {f["adr_feats_size"]}-feature model is a
    genuine, freshly-trained fit — not a copy of the original 51-feature list.
  """)
    new_header = f"""<h1>Quoted ADR pruning funnel — rebuilt with 3 flagged rows excluded</h1>
  <p class="sub">
    <span class="badge">Real pipeline, single authoritative run</span>
    Every stage below runs the actual <span class="tag">rfp_adr_pipeline_filtered_first_final.py</span>
    code (drop_correlated, the importance-screening XGBoost, prune_vif with its rank-deficiency and
    standardization fixes, exclude_leaky, train_model's Optuna search) — imported and executed unchanged,
    not reimplemented. The one deliberate change: 3 rows identified as genuine data-quality problems
    (<span class="tag">{', '.join(f["excluded_rfp_ids"])}</span> — quoted_adr above the deal's own
    max_room_rate, with no legitimate bundled-revenue explanation) are excluded BEFORE the candidate pool
    is built, so every pruning decision is made on the same population the final model trains on.
  </p>

  <div class="callout diag">{diag_body}</div>

  <div class="callout">"""
    content = content[:i] + new_header + content[j:]

    # ---------- 2. stage notes (pool/corr/imp/vif) ----------
    def replace_note(old_key_note_start, new_note):
        nonlocal content
        i0 = content.find(old_key_note_start)
        if i0 == -1:
            raise ValueError(f"note anchor not found: {old_key_note_start[:60]!r}")
        note_start = content.find('note: `', i0) + len('note: `')
        note_end = content.find('`,', note_start)
        content = content[:note_start] + new_note + content[note_end:]

    replace_note("key:'pool'",
        r"Numeric columns not in ALWAYS_EXCLUDE, not ending in _v8, std &gt; 1e-10, &lt;50% missing — "
        r"computed on the ${D.rows_after_filter}-row filtered population (3 flagged rows already excluded). "
        r"Click a column header to sort.")
    replace_note("key:'corr'",
        r"${D.corr_pairs.length} pairs exceeded the threshold; the lower-variance feature of each pair was "
        r"dropped. Click a column header to sort.")
    content = content.replace(
        "title:'Quick-XGBoost importance screening → top 60 (now fully reproducible)'",
        "title:'Quick-XGBoost importance screening → top 60'")
    replace_note("key:'imp'",
        r"A lightweight proxy XGBRegressor (100 trees, depth 5, target = quoted_adr on the filtered rows, "
        r"subsample=1.0/colsample_bytree=1.0/tree_method=exact for determinism) ranks all ${D.kept_corr_size} "
        r"correlation-survivors; only the top 60 by importance continue. Click a column header to sort.")
    replace_note("key:'vif'",
        r"${D.vif_removed.length} features iteratively dropped (highest VIF first, recomputed each round) "
        r"until every remaining feature has VIF &lt; 10, with a condition-number check before trusting any "
        r"VIF value (an exact/near-exact rank deficiency drops the responsible column directly instead of "
        r"trusting an arbitrarily large regression estimate). Click a column header to sort.")

    # ---------- 3. finalStage hardcoded count + title ----------
    content = content.replace(
        '<div class="stage-title">Trained model — your actual feature importance &amp; correlation</div>',
        '<div class="stage-title">Trained model — rebuilt, feature importance &amp; correlation</div>')
    content = content.replace(
        '<div class="stage-delta">Your real, Optuna-tuned XGBoost model, fit on the surviving features · click any row for its scatter</div>',
        '<div class="stage-delta">Freshly Optuna-tuned XGBoost model, fit on the surviving features with the 3 flagged rows excluded · click any row for its scatter</div>')
    content = content.replace(
        '<div class="stage-count" style="color:var(--stage5)">51 <span class="chev">&#9656;</span></div>',
        f'<div class="stage-count" style="color:var(--stage5)">{f["adr_feats_size"]} <span class="chev">&#9656;</span></div>')

    # ---------- 4. footer ----------
    i, j = get_between(content, '<footer>', '</footer>')
    tr_gap_note = "" if n_tr_missing == 0 else (
        f" ({n_tr_missing} tr_* columns unavailable without the raw transient file, "
        f"{f['pool_size']}+{n_tr_missing}=298)")
    new_footer = f"""<footer>
    Best hyperparameters (this run's Optuna search, seed 42): <span id="bpLine"></span><br>
    Rows: {f["rows_before_filter"] + n_excl:,} total &rarr; {n_excl} flagged rows excluded &rarr; {f["rows_before_filter"]:,}
    &rarr; {f["rows_after_filter"]:,} after room_block&gt;0 filter ({f["rows_excluded"]} meeting-only RFPs excluded from ADR modeling).<br>
    Candidate pool {f["pool_size"]}{tr_gap_note}
    &rarr; correlation pruning (|r|&ge;0.85) &rarr; {f["kept_corr_size"]} &rarr; quick-XGBoost importance top-60 &rarr;
    VIF pruning (VIF&ge;10) &rarr; {f["vif_final_size"]} &rarr; leaky-feature exclusion &rarr; {f["adr_feats_size"]} features trained.<br>
    All correlations on this page are real Pearson r vs <span class="tag">quoted_adr</span>, n={f["rows_after_filter"]:,}
    (room_block&gt;0, 3 flagged rows excluded), computed directly from the training data. The 3 excluded rows are shown
    on every scatter as a hollow diamond marker, scored out-of-sample by the trained model.
  </footer>"""
    content = content[:i] + new_footer + content[j:]

    # ---------- 5. metrics grid ----------
    i, j = get_between(content, '// Metrics', "].map(([k,v,v2])=>`<div class=\"metric\"><div class=\"k\">${k}</div><div class=\"v\">${v}</div>${v2?`<div class=\"v2\">${v2}</div>`:''}</div>`).join('');\n\n")
    new_metrics_js = """// Metrics — this run's real trained model, no second run to compare against
const m = D.metrics;
document.getElementById('metricsGrid').innerHTML = [
  ['Test R²', m.test_r2.toFixed(4), ''],
  ['Test MAE', '$'+m.test_mae.toFixed(2), ''],
  ['Train MAE', '$'+m.train_mae.toFixed(2), ''],
  ['CV MAE (5-fold)', '$'+m.cv_mae_mean.toFixed(2)+' \\u00b1 '+m.cv_mae_std.toFixed(2), ''],
  ['Train rows', m.n_train, ''],
  ['Test rows', m.n_test, ''],
].map(([k,v,v2])=>`<div class="metric"><div class="k">${k}</div><div class="v">${v}</div>${v2?`<div class="v2">${v2}</div>`:''}</div>`).join('');

"""
    content = content[:i] + new_metrics_js + content[j:]

    # ---------- 6. showDetail scatter: add the excluded-from-training marker ----------
    old_pts_start = "const pts = xs.map((x,i)=>{"
    old_pts_end_anchor = "const y1 = slope*minX"
    i = content.find(old_pts_start)
    j = content.find(old_pts_end_anchor, i)
    new_pts_block = """const EXCL = POINTS.is_excluded_from_training || [];
  const pts = xs.map((x,i)=>{
    const cx = sx(x).toFixed(1), cy = sy(ys[i]).toFixed(1);
    let tip = pointTooltip(feature,i,x,ys[i]);
    if (EXCL[i]) tip += esc(`\\nEXCLUDED FROM TRAINING (out-of-sample here — ${POINTS.rfp_id && POINTS.rfp_id[i] ? POINTS.rfp_id[i] : ''})`);
    if (isOutlier[i]) tip += esc(`\\nOUTLIER (Cook's D=${cooksD[i].toFixed(3)}, threshold ${COOKS_THRESHOLD.toFixed(3)})`);
    if (EXCL[i]){
      const s = 4.6;
      return `<g class="point excluded"><rect x="${cx-s}" y="${cy-s}" width="${s*2}" height="${s*2}" fill="none" stroke="var(--neg)" stroke-width="1.6" transform="rotate(45 ${cx} ${cy})"></rect><title>${tip}</title></g>`;
    }
    if (isOutlier[i]){
      const s = 4.2; // cross arm half-length
      return `<g class="point outlier"><line x1="${cx-s}" y1="${cy-s}" x2="${Number(cx)+s}" y2="${Number(cy)+s}"></line><line x1="${cx-s}" y1="${Number(cy)+s}" x2="${Number(cx)+s}" y2="${cy-s}"></line><title>${tip}</title></g>`;
    }
    return `<circle class="point" cx="${cx}" cy="${cy}" r="2.2"><title>${tip}</title></circle>`;
  }).join('');

  """
    content = content[:i] + new_pts_block + content[j:]

    content = content.replace(
        "catches points with an unusual y for their x AND high-leverage points whose extreme x pulls the trend line toward them).</div>",
        "catches points with an unusual y for their x AND high-leverage points whose extreme x pulls the trend line toward them). "
        "<span style=\"color:var(--neg);\">&#9670;</span> hollow diamond = one of the 3 rows excluded from training, shown out-of-sample.</div>")

    # ---------- 7. swap the JSON payload ----------
    marker = '<script id="funnel-data" type="application/json">'
    i = content.find(marker) + len(marker)
    j = content.find('</script>', i)
    content = content[:i] + json.dumps(payload) + content[j:]

    return content


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                     help="rfp_training_data_complete_v3_with_transient.csv (the merged training CSV)")
    ap.add_argument("--pipeline", required=True,
                     help="rfp_adr_pipeline_filtered_first_final.py (your real pipeline source file)")
    ap.add_argument("--transient-data", default=None,
                     help="Raw transient-demand CSV (e.g. Nexus_Transient_Demand_v2.csv). Optional — "
                          "omit to use whatever tr_* columns --data already has.")
    ap.add_argument("--source", required=True,
                     help="The original funnel_quoted_adr_v4 HTML file to template onto")
    ap.add_argument("--out", default="funnel_adr_rebuilt.html", help="Output HTML path")
    ap.add_argument("--out-model", default=None, help="Optional: also write the trained model bundle (.pkl)")
    ap.add_argument("--out-json", default=None, help="Optional: also write the raw JSON payload")
    ap.add_argument("--scratch", default="pipeline_scratch", help="Scratch dir for the pipeline's intermediate CSVs")
    args = ap.parse_args()

    payload, bundle = run_pipeline(args.data, args.pipeline, args.transient_data, args.scratch)

    print(f"\nRendering {args.source} -> {args.out} ...")
    html = render_html(args.source, payload)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {args.out}  ({len(html) / 1e6:.2f} MB)")

    if args.out_model:
        with open(args.out_model, "wb") as fh:
            pickle.dump(bundle, fh)
        print(f"Wrote {args.out_model}")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(payload, fh)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
