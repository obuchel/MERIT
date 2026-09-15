"""
rfp_adr_pipeline_filtered_first.py
═══════════════════════════════════════════════════════════════════════════════
Quoted-ADR-only variant of rfp_xgboost_pipeline_v7.py, with ONE deliberate
methodological change requested by the user:

    ORIGINAL v7 ORDER (main() in rfp_xgboost_pipeline_v7.py):
        engineer_features(ev)                       # full 1,899 rows
        pool = build_candidate_pool(ev)              # full 1,899 rows
        base = prune_features(ev, pool)              # correlation + importance + VIF, full 1,899 rows
        adr_feats = exclude_leaky(base, "quoted_adr")
        train_model(..., filter_rooms=True)          # room_block>0 filter applied ONLY HERE, at training

    THIS SCRIPT'S ORDER:
        engineer_features(ev)                        # full 1,899 rows (unaffected by the ADR filter)
        ev_adr = ev[ev.room_block > 0]                # <-- FILTER MOVED HERE: 1,899 -> 1,705 rows
        pool = build_candidate_pool(ev_adr)           # candidate pool computed on the FILTERED population
        base = prune_features(ev_adr, pool)           # correlation + importance + VIF, all on the FILTERED population
        adr_feats = exclude_leaky(base, "quoted_adr")
        train_model(..., filter_rooms=True)           # re-applies the same filter (idempotent) for training

Why this matters: 194 "meeting-only" RFPs have room_block == 0 and a
structurally-fixed quoted_adr of $0. Several rate-related features
(ADR_Discount_Pct, rate_discount_pct, revenue_intensity, ...) are pinned at
degenerate/constant values for exactly those 194 rows. Under the ORIGINAL v7
order, those degenerate rows are still present when drop_correlated() and
prune_vif() run, which can distort correlation and VIF estimates for those
features. Filtering FIRST means every pruning decision — correlation,
proxy-importance ranking, VIF — is made on the same population the final
model is actually trained on.

This is a real, verified DIVERGENCE from the shipped v7 pipeline, not a bug
fix to it — v7's own author may have had reasons for the later filter (e.g.
keeping the shared 36-feature base pool identical across all 4 models). Both
orderings are defensible; this script implements the "filter first" ordering
end-to-end and reports exactly what changes.

All function bodies below (load_data, join_transient, engineer_features,
build_candidate_pool, drop_correlated, prune_vif, prep_X, optuna_search,
train_model) are copied VERBATIM from the user's uploaded
rfp_xgboost_pipeline_v7.py — nothing about the algorithms themselves was
changed, only the ORDER in which build_candidate_pool/prune_features vs. the
room_block filter are called (see run_adr_pipeline() at the bottom).

── CHANGE LOG (this revision) ──────────────────────────────────────────────
Cross-environment testing showed that two runs of THIS pipeline, on identical
data, with identical library versions and the same RANDOM_STATE=42, can still
land on materially different top-60 feature sets at the "quick-XGBoost
importance screening" step inside prune_features(). Root cause, verified by
direct experiment (shuffling only the column order of the same 191 features
and refitting, holding everything else fixed, reproduced the same ~35-42/60
overlap seen between two real independent runs): the quick screening model
uses colsample_bytree=0.8, which samples per-tree column subsets by INDEX
POSITION, not by name. So the physical left-to-right order of the kept_corr
columns feeding X_proxy — which is not itself pinned to anything meaningful,
just whatever order drop_correlated() happens to emit — silently determines
the result, even though it has no relationship to feature identity or
importance.

Two changes fix this (both applied below, in prune_features()):
  1. `kept_corr` is sorted alphabetically before building X_proxy, so the
     column order is a fixed, reproducible function of the feature NAMES,
     not of incidental upstream ordering.
  2. `colsample_bytree` for the quick screening model is raised from 0.8 to
     1.0, so every tree sees all 191 (kept-corr) features and the fit no
     longer depends on column order (or on the sampling RNG) at all. This is
     a proxy ranking model discarded immediately after use — it was never
     meant to generalize, only to rank — so removing its own internal
     feature-sampling regularization does not weaken the final trained
     model, which still uses its own separately-tuned colsample_bytree from
     the Optuna search in train_model().

── UPDATE (revision 2) ──────────────────────────────────────────────────
Revision 1 above (sort kept_corr + colsample_bytree=1.0) was verified to fix
the CROSS-environment case I could directly test (my xgboost 3.2.0 vs the
user's real xgboost 3.1.1, pinned in an isolated venv here: 60/60 top-60
match after the fix, vs 39/60 before). But the user then ran revision 1
TWICE on their own single machine and got two DIFFERENT results (31 features
one run, 32 the next) — i.e. it wasn't even reproducible run-to-run on the
SAME machine. I could not reproduce that same-machine non-determinism here
(4 repeated process runs of revision 1's exact code gave bit-identical
float32 importances every time on this Linux box), which means it's most
likely a platform/threading quirk specific to their environment (e.g. macOS
+ a conda-built xgboost, where some internal thread pool doesn't fully
honor n_jobs=1 the way it does on this Linux build) — I can't verify that
specific mechanism without access to their machine.

Rather than guess further at what's Mac-specific, revision 2 strips every
remaining source of randomness and approximation from the quick screening
model, so there is nothing left for platform-level nondeterminism to act on:
  3. `subsample` for the quick model is raised from 0.8 to 1.0 (revision 1
     left this at 0.8 — row-level subsampling was still active and is the
     most likely remaining suspect, since it's the one RNG-driven step left).
  4. `tree_method` is pinned to "exact" instead of the (default, in modern
     xgboost) "hist" — exact greedy split-finding has no histogram/quantile-
     sketch approximation step, which is where multi-threaded builds can
     introduce tiny floating-point summation-order differences even at
     n_jobs=1 on some platforms/builds.

With subsample=1.0, colsample_bytree=1.0, and tree_method="exact", the quick
model now sees the exact same 191x1,705 input on every tree, every run, with
no approximation — I verified this is bit-identical (down to raw float32
importances) across 4 separate process invocations here. Fit time was ~1.4s,
so the "exact" method is not a practical cost at this data size (this proxy
model is only there to rank features for a top-60 cut, not to generalize).

I have NOT been able to test revision 2 on a Mac/conda environment like the
user's — if platform-specific non-determinism persists even after this,
that's a real, deeper issue with xgboost's determinism guarantees on that
platform+build, worth reporting upstream, not something I can rule out from
here. The user should re-run this revision at least twice and confirm the
funnel counts and top features are identical before trusting it.

Together these make the importance-screening step, and therefore the whole
funnel downstream of it (VIF pruning, leaky exclusion, final feature count),
deterministic and reproducible, as long as the input data and kept_corr set
are identical — pending the user's own confirmation of revision 2 above.
NOTE: because this changes the quick model's behavior, re-running this
script will very likely change the exact top-60 (and therefore final)
feature set from earlier runs, including the runs behind
funnel_adr_real.json / funnel_adr_theirs.json and the revision-1 run
reported earlier — that is the whole point (removing an unintended source
of run-to-run variation), but it means past reported feature lists/metrics
are not directly comparable to a fresh run of this revision without
re-running it.

── UPDATE (revision 3) ──────────────────────────────────────────────────
Even after revisions 1-2, the user still got different final feature counts
(33-34 vs. others) from repeated runs of the SAME script. Root cause: the
pairwise correlation pruning (|r| >= 0.85) can't see dependencies among
THREE OR MORE columns, and the real top-60 matrix turned out to have exactly
one such dependency — avg_occupancy_7d / occupancy_velocity_7_30d /
avg_occupancy_30d are exactly linearly related (verified via SVD: numerical
rank 59, not 60). When variance_inflation_factor() is handed an exactly
rank-deficient matrix, the true VIF for the involved columns is
mathematically infinite, and the regression-based estimate of "how large"
becomes numerically arbitrary — this is what was flipping the pruning
outcome. Fix: prune_vif() now checks the matrix's condition number via SVD
before trusting any VIF value; on exact/near-exact rank deficiency it
deterministically drops the column most responsible for the dependency
instead. Verified via 8-trial noise-injection stress test (all converged
identically) and repeated full-pipeline runs (bit-identical).

── UPDATE (revision 4) ──────────────────────────────────────────────────
Revision 3 fixed the ONE exact singularity — confirmed correct on the user's
own machine too (their real vif_removed.csv logs avg_occupancy_7d with
vif=inf, exactly matching mine). But their run then removed 25 MORE features
with large-but-finite VIFs (up to 2777.85) that never showed up in my runs
(only 5 more, all VIF<35), landing at 34 kept features vs. my 54, on
PROVABLY IDENTICAL input data (I obtained the user's real output files and
directly diffed them against mine: the top-60 set, the full 191-feature
kept_corr set, and even the raw importance VALUES for all 191 features
matched to 0.0 difference — so this was not a data problem).

Root cause, found by regenerating the user's exact real top-60 matrix here
and testing it directly: prune_vif() was calling
variance_inflation_factor() on the RAW, unscaled design matrix (only the
revision-3 rank check used a standardized copy). The real top-60 columns
span roughly a 1.5-million-to-1 range of raw scales (e.g.
revenue_competition_30d has std~23,236 vs. displacement_velocity_30_90d at
std~0.016). It turns out my installed statsmodels (0.15.0) silently added an
internal standardize=True default inside variance_inflation_factor() itself
— undocumented behavior this script never explicitly asked for, but was
accidentally protected by. The user's real statsmodels, confirmed via their
own `pip show` output, is 0.14.5 — which has NO such parameter and computes
the VIF regression directly on whatever scale it's given. I installed
statsmodels==0.14.5 in an isolated venv and fed it this script's real raw
post-pruning matrix: it returned 2777.849300641239 for avg_occupancy_30d —
matching the user's real reported 2777.85 to the precision they logged,
exactly. Standardizing that same data myself first and re-running through
the SAME 0.14.5 install returned 5.75, matching my own result on 0.15.0 to
full precision. I then ran this file's actual prune_vif() function, verbatim
after the fix below, against the user's exact real top-60 data with
statsmodels 0.14.5 installed: 60 -> 54 (6 removed), bit-identical removal
order and VIF values to my own environment on statsmodels 0.15.0. This is
airtight, cross-version-verified: VIF is mathematically scale-invariant, but
was being computed on wildly different raw feature scales without explicit
standardization, so the result silently depended on which statsmodels
version happened to be installed. Fix: prune_vif() now standardizes the
matrix itself before every call to variance_inflation_factor(), both in the
pruning loop and in the final report, so the result no longer depends on the
installed statsmodels version.

CAVEAT: I have not been able to test anything on the user's actual machine —
everything above was verified here, using the user's self-reported package
versions and their real uploaded output files as ground truth. If their
environment differs from what they reported (e.g. a different statsmodels
somehow shadowing the one `pip show` reported), this fix could still leave
a gap; the user should re-run and confirm 60 -> 54 before trusting it.

Inputs (same as v7, place under these relative paths or edit the constants
below):
    outputs/rfp_training_data_complete_v3.csv
    outputs/Enhanced_All_Events_v8_final.csv
    filesData/outputs/Nexus_Transient_Demand_v2.csv

Requires: pandas, numpy, xgboost, scikit-learn, scipy, statsmodels.
Optional: optuna (30-trial hyperparameter search). Falls back to fixed
default hyperparameters if optuna is not installed — this only affects the
FINAL model's tuning, not the pruning funnel (correlation/importance/VIF),
which never uses optuna.

Usage:
    python rfp_adr_pipeline_filtered_first.py
"""

import warnings; warnings.filterwarnings("ignore")
import time
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import norm as scipy_norm  # noqa: F401 (kept for parity with v7 imports)
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed — using default hyperparameters. "
          "Install with: pip install optuna")

# ── Config (identical to v7) ────────────────────────────────────────────────
V3_FILE        = "outputs/rfp_training_data_complete_v3.csv"
EVENTS_FILE    = "outputs/Enhanced_All_Events_v8_final.csv"
TRANSIENT_FILE = "filesData/outputs/Nexus_Transient_Demand_v2.csv"
OUTPUT_DIR     = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CORR_THRESHOLD = 0.85
VIF_THRESHOLD  = 10.0
TOP_K          = 60
CV_FOLDS       = 5
RANDOM_STATE   = 42
N_OPTUNA       = 30
PEAK_MONTH     = 6
TRAINING_WINDOW_MONTHS = None

# Columns always excluded (identity, dates, targets, leaky actuals) — verbatim from v7
ALWAYS_EXCLUDE = {
    "rfp_id", "inquiry_date", "arrival_date", "Arrival_Date", "Departure_Date",
    "account_name", "event_name", "status", "lost_reason", "Arrival_DOW_Name",
    "Displacement_Risk_Level", "Demand_Tier", "arrival_dow_name",
    "displacement_risk_level", "demand_tier", "_month", "arrival_week",
    "quoted_adr", "actual_pickup_rate", "actual_fnb_per_person",
    "actual_total_revenue", "is_won", "did_convert", "revenue_realization_pct",
    "Actual_FnB_Revenue", "Actual_FnB_Revenue_v8",
    "Proposed_FnB_Revenue", "Proposed_FnB_Revenue_v8",
    "Arrival_Date_v8", "Departure_Date_v8", "Account_Win_Rate_APIT",
    "fnb_ratio",
    "group_size_tier_enc",
    "market_segment_enc",
}

LEAKY = {
    "quoted_adr": {
        "actual_pickup_rate", "is_won", "did_convert", "win_streak",
        "win_rate_7d", "win_rate_14d", "win_rate_30d", "win_rate_60d", "win_rate_90d",
        "win_rate_ema_30d", "win_rate_ema_7d", "win_rate_ema_90d",
        "win_rate_volatility_30d", "win_rate_volatility_60d", "win_rate_volatility_90d",
        "win_rate_acceleration", "win_rate_momentum",
        "win_rate_velocity_30_90d", "win_rate_velocity_7_30d",
    },
    "pickup":     {"is_won", "did_convert"},
    "conversion": {"actual_pickup_rate"},
    "fnb":        {"actual_pickup_rate", "is_won", "did_convert"},
}

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD & MERGE  (verbatim from v7)
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    print("[1] Loading data...")
    ev = pd.read_csv(V3_FILE)
    print(f"    v3: {ev.shape[0]:,} rows x {ev.shape[1]} cols")

    v8_cols = ["RFP_ID", "Proposed_FnB_Revenue", "Actual_FnB_Revenue",
               "Departure_Date", "Arrival_Date"]
    v8_avail = [c for c in v8_cols if c in pd.read_csv(EVENTS_FILE, nrows=0).columns]
    v8 = pd.read_csv(EVENTS_FILE, usecols=v8_avail).rename(columns={"RFP_ID": "rfp_id"})
    ev["rfp_id"] = ev["rfp_id"].astype(str)
    v8["rfp_id"] = v8["rfp_id"].astype(str)
    ev = ev.merge(v8, on="rfp_id", how="left", suffixes=("", "_v8"))
    print(f"    After v8 gap-fill: {ev.shape[1]} cols")

    ev["Arrival_Date"]   = pd.to_datetime(ev["Arrival_Date"])
    ev["Departure_Date"] = pd.to_datetime(ev["Departure_Date"])
    ev["inquiry_date"]   = pd.to_datetime(ev.get("inquiry_date", ev["arrival_date"]))
    ev = ev.sort_values("inquiry_date").reset_index(drop=True)

    if TRAINING_WINDOW_MONTHS is not None:
        cutoff = ev["inquiry_date"].max() - pd.DateOffset(months=TRAINING_WINDOW_MONTHS)
        before = len(ev)
        ev = ev[ev["inquiry_date"] >= cutoff].reset_index(drop=True)
        print(f"    Rolling window ({TRAINING_WINDOW_MONTHS}m): {before:,} -> {len(ev):,} rows")

    return ev


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — TRANSIENT JOIN  (verbatim from v7)
# ═══════════════════════════════════════════════════════════════════════════
TR_NUMERIC = [
    "Transient_ADR", "Market_ADR", "ADR_Index", "Transient_Occ_Of_Available",
    "Total_Occupancy_Pct", "Rooms_to_Capacity", "Transient_Rooms_Turned_Away",
    "Transient_Yield_Pct", "Displacement_Cost_Per_Room", "Pace_Index",
    "Pace_vs_Prior_Year", "Market_Occ_Pct", "Transient_RevPAR",
    "Group_Rooms_On_Books", "Transient_Pace_30d",
]
TR_CAT_MAPS = {
    "tr_demand_tier":           {"Soft": 0, "Low": 1, "Moderate": 2, "High": 3, "Peak": 4},
    "tr_demand_tier_3":         {"Low": 0, "Shoulder": 1, "Peak": 2},
    "tr_rate_strategy":         {"Promotional": 0, "Standard": 1, "Rack": 2,
                                  "Premium_Event": 3, "Holiday_Reduced": -1},
    "tr_displacement_pressure": {"Low": 0, "Medium": 1, "High": 2, "Sold Out": 3},
}

def join_transient(ev):
    print("\n[2] Joining transient features over stay window...")
    tr = pd.read_csv(TRANSIENT_FILE)
    tr["Date"] = pd.to_datetime(tr["Date"])
    tr_idx = tr.set_index("Date")
    print(f"    Transient: {len(tr):,} rows x {tr.shape[1]} cols")

    def _row(arrival, departure):
        dep = departure if pd.notna(departure) else arrival + pd.Timedelta(days=1)
        dates = pd.date_range(arrival, dep - pd.Timedelta(days=1), freq="D")
        sub = tr_idx.reindex(dates).dropna(how="all")
        row = {}
        for c in TR_NUMERIC:
            row[f"tr_{c.lower()}"] = sub[c].mean() if (len(sub) and c in sub.columns) else np.nan
        for col, mapping in TR_CAT_MAPS.items():
            cat_col = col.replace("tr_", "").replace("_", " ").title().replace(" ", "_")
            match = [c for c in (tr.columns if len(sub) == 0 else sub.columns)
                     if c.lower() == cat_col.lower()]
            if match and len(sub):
                mode = sub[match[0]].dropna().mode()
                row[col] = mode.iloc[0] if len(mode) else np.nan
            else:
                row[col] = np.nan
        return row

    t0 = time.time()
    tr_rows = [_row(r.Arrival_Date, r.Departure_Date) for _, r in ev.iterrows()]
    tr_df = pd.DataFrame(tr_rows, index=ev.index)
    print(f"    Join complete in {time.time()-t0:.1f}s - {tr_df.shape[1]} transient cols")

    for col, mapping in TR_CAT_MAPS.items():
        if col in tr_df.columns:
            tr_df[col] = tr_df[col].map(mapping).fillna(0).astype(int)
    for c in tr_df.select_dtypes("float").columns:
        tr_df[c] = tr_df[c].fillna(tr_df[c].median())

    return pd.concat([ev, tr_df], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — ENCODE & DERIVED FEATURES  (verbatim from v7)
# ═══════════════════════════════════════════════════════════════════════════
IDENTITY_COLS = {
    "rfp_id", "inquiry_date", "arrival_date", "account_name", "event_name",
    "status", "lost_reason", "Arrival_Date", "Departure_Date", "Arrival_DOW_Name",
    "Displacement_Risk_Level", "Demand_Tier", "arrival_dow_name",
    "displacement_risk_level", "demand_tier",
}

def engineer_features(ev):
    print("\n[3] Encoding categoricals and engineering features...")

    # FIX 2: Extract is_meeting_only BEFORE encoding group_size_tier
    if "group_size_tier" in ev.columns:
        ev["is_meeting_only"] = (ev["group_size_tier"] == "meeting_only").astype(int)
        size_order = {"small": 0, "medium": 1, "large": 2, "very_large": 3}
        ev["group_size_tier_clean_enc"] = (
            ev["group_size_tier"].map(size_order).fillna(-1).astype(int)
        )
        n_mo = ev["is_meeting_only"].sum()
        print(f"    is_meeting_only: {n_mo} records extracted ({n_mo/len(ev)*100:.1f}%)")
        print(f"    group_size_tier_clean_enc: 4 ordered tiers (small=0 ... very_large=3)")

    # FIX 3: One-hot encode market_segment (13 unordered categories)
    ohe_cols = []
    if "market_segment" in ev.columns:
        ohe = pd.get_dummies(ev["market_segment"], prefix="market_seg", drop_first=False)
        ohe = ohe.astype(int)
        ev = pd.concat([ev, ohe], axis=1)
        ohe_cols = ohe.columns.tolist()
        print(f"    market_segment one-hot: {len(ohe_cols)} columns")
    ev.attrs["market_seg_ohe_cols"] = ohe_cols

    # All other string columns: label encode as before
    SKIP_ENCODE = IDENTITY_COLS | {"group_size_tier", "market_segment"}
    for c in ev.select_dtypes(include="object").columns:
        if c in SKIP_ENCODE:
            continue
        ev[f"{c}_enc"] = pd.Categorical(ev[c]).codes
        print(f"    Encoded: {c} ({ev[c].nunique()} cats)")

    # Recompute FnB per person from v8 actuals
    ev["actual_fnb_per_person"] = np.where(
        (ev["attendees"] > 0) & ev["Actual_FnB_Revenue"].notna(),
        ev["Actual_FnB_Revenue"] / ev["attendees"], np.nan,
    )

    # Budget ratio
    ev["budget_ratio"] = (
        ev["budget_amount"] / ev["proposed_total_revenue"].replace(0, np.nan)
    ).fillna(0)

    # Pricing pressure
    ev["pricing_pressure_index"] = (
        (ev["baseline_adr"] - ev["quoted_adr"]) /
        ev["baseline_adr"].replace(0, np.nan)
    ).fillna(0) * (ev["forecasted_occupancy"] / 100)

    # Cyclical month encoding
    ev["_month"]              = ev["Arrival_Date"].dt.month
    ev["arrival_month_sin"]   = np.sin(2 * np.pi * ev["_month"] / 12)
    ev["arrival_month_cos"]   = np.cos(2 * np.pi * ev["_month"] / 12)
    ev["arrival_quarter_sin"] = np.sin(2 * np.pi * (((ev["_month"] - 1) // 3) + 1) / 4)
    ev["is_shoulder_season"]  = ev["_month"].isin([3, 4, 9, 10, 11]).astype(int)

    def _days_to_peak(d):
        p = d.replace(month=PEAK_MONTH, day=1)
        return int(min(abs((d - p).days), abs((d - p.replace(year=d.year + 1)).days)))

    ev["days_to_peak_season"] = ev["Arrival_Date"].apply(_days_to_peak)
    ev["did_convert"]         = (ev["status"] == "Booked").astype(int)

    if "total_room_nights" not in ev.columns:
        ev["total_room_nights"] = ev.get(
            "room_block", ev.get("total_room_nights_requested", np.nan)
        )

    # FIX 4a: log1p(revenue_quality_score)
    if "revenue_quality_score" in ev.columns:
        ev["revenue_quality_score"] = np.log1p(ev["revenue_quality_score"])
        print(f"    log1p(revenue_quality_score): skew after = "
              f"{ev['revenue_quality_score'].skew():.2f}")

    # FIX 4b: Winsorise occupancy_acceleration at p1/p99
    if "occupancy_acceleration" in ev.columns:
        p1  = ev["occupancy_acceleration"].quantile(0.01)
        p99 = ev["occupancy_acceleration"].quantile(0.99)
        ev["occupancy_acceleration"] = ev["occupancy_acceleration"].clip(p1, p99)
        print(f"    occupancy_acceleration winsorised: [{p1:.3f}, {p99:.3f}]")

    print(f"    Dataset after engineering: {ev.shape[1]} cols")
    return ev


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — CANDIDATE POOL  (verbatim from v7)
# ═══════════════════════════════════════════════════════════════════════════
def build_candidate_pool(ev):
    print("\n[4] Building candidate feature pool...")
    numeric = ev.select_dtypes(include=[np.number]).columns.tolist()
    pool = [
        c for c in numeric
        if c not in ALWAYS_EXCLUDE
        and not c.endswith("_v8")
        and ev[c].std() > 1e-10
        and ev[c].isna().mean() < 0.50
    ]
    for c in pool:
        med = ev[c].median()
        ev[c] = ev[c].fillna(med if not np.isnan(med) else 0)
        ev[c] = ev[c].replace([np.inf, -np.inf], med if not np.isnan(med) else 0)
    print(f"    Candidate pool: {len(pool)} features")
    return pool


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — MULTICOLLINEARITY PRUNING
# drop_correlated() is verbatim from v7. prune_features() carries the
# xgboost-determinism fixes described in revisions 1-2 of the CHANGE LOG
# above. prune_vif() carries an ADDITIONAL, revision-3 fix described just
# above it below — everything else in both functions is unchanged from v7.
# ═══════════════════════════════════════════════════════════════════════════
def drop_correlated(X, threshold=CORR_THRESHOLD):
    corr = X.corr().abs()
    removed, log = set(), []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            c1, c2 = corr.columns[i], corr.columns[j]
            if c1 in removed or c2 in removed:
                continue
            r = corr.iloc[i, j]
            if r >= threshold:
                drop = c2 if X[c1].std() >= X[c2].std() else c1
                keep = c1 if drop == c2 else c2
                removed.add(drop)
                log.append({"kept": keep, "dropped": drop, "correlation": round(float(r), 4)})
    return [c for c in X.columns if c not in removed], pd.DataFrame(log)


# ── REVISION 3 (this fix) ───────────────────────────────────────────────────
# Even after revisions 1-2 made the XGBoost importance-screening step fully
# deterministic (verified bit-identical here), the user still got different
# results (33-34 vs other counts) from repeated runs of the identical script.
# Root cause, found by directly testing prune_vif() for sensitivity to
# floating-point-scale noise: the correlation-pruning step (|r| >= 0.85,
# PAIRWISE only) does not catch dependencies among THREE OR MORE columns.
# Checking the real top-60 matrix directly: its numerical rank is 59, not 60
# — condition number ~2.2e15, smallest singular value ~5e-14 (machine
# epsilon). There is a genuine EXACT linear identity among
# avg_occupancy_7d / occupancy_velocity_7_30d / avg_occupancy_30d (confirmed
# via SVD) that no pairwise correlation check can see. When the design
# matrix passed to variance_inflation_factor() is exactly rank-deficient like
# this, the true VIF for the involved columns is mathematically infinite,
# and statsmodels' regression-based estimate of it becomes numerically
# ARBITRARY — governed by whatever floating-point rounding the local
# BLAS/LAPACK backend produces for a singular OLS solve (this is exactly the
# SingularMatrixWarning seen throughout every run of this pipeline). That
# arbitrary "very large number" is what was flipping which feature got
# picked as "worst VIF" first, and because pruning is greedy/iterative, one
# early flip cascades into a very different final feature set. I verified
# this directly: injecting synthetic noise of ~1e-9 relative magnitude (far
# below any real signal, comparable to cross-platform floating-point
# roundoff) into the SAME top-60 data flipped the final kept-feature count
# between 53 and 54 across repeated trials with the ORIGINAL prune_vif() —
# reproducing the same class of instability the user hit. Rounding the VIF
# value before comparison does NOT fix this (also tested) — the computed
# VIF itself swings by orders of magnitude near an exact singularity, not
# just in its last few digits, so rounding a meaningless number doesn't
# help.
#
# The fix: before computing any VIF in a given iteration, check the
# matrix's condition number via SVD. When it indicates exact/near-exact
# rank deficiency (smallest singular value below rank_eps * largest),
# skip the numerically-unstable VIF computation entirely and instead
# deterministically identify the column most responsible for the
# dependency — the one with the largest loading on the smallest singular
# vector — and drop it directly (logged with vif=inf, since that's the
# mathematically honest value). This is a well-defined, stable computation
# (SVD of a real matrix with one small singular value has an essentially
# unique — up to sign — associated singular vector), unlike the ill-
# conditioned regression VIF was trying to compute for the same columns.
# Verified: 8 repeated trials (no noise + 7 different ~1e-9-scale noise
# perturbations) now converge to the IDENTICAL 54-feature set every time,
# where the original code gave 53 or 54 unpredictably.
#
# ── REVISION 4 (this fix) ───────────────────────────────────────────────────
# Revision 3 fixed the ONE exact singularity (avg_occupancy_7d) — confirmed:
# the user's real vif_removed.csv logs that exact feature with vif=inf,
# identically to mine. But their run then removed 25 MORE features with
# large-but-finite VIFs (up to 2777.85) that my run never saw (I removed only
# 5 more, all VIF<35). I got the user's real output files (screening CSV,
# vif_removed.csv, vif_final.csv) and directly compared them byte-for-byte
# against mine. That ruled out the data itself: the top-60 AND kept_corr
# (191) feature sets are 100% identical between their environment and mine,
# same order, and the full 191-row importance VALUES match to 0.0 difference
# — so the design matrix feeding prune_vif() is provably identical on both
# machines. The divergence had to be in the computation, not the data.
#
# Root cause, found by regenerating the user's exact real top-60 matrix here
# and computing its condition number AFTER removing avg_occupancy_7d: on the
# STANDARDIZED matrix it's a well-conditioned 22.9 — nowhere near unstable.
# But this code was calling variance_inflation_factor() on the RAW, UNSCALED
# Xm (only the rank-deficiency check above used the standardized Xstd) — and
# the raw top-60 columns span a ~1.5-million-to-1 range of scales (e.g.
# revenue_competition_30d has std~23,236 vs. displacement_velocity_30_90d at
# std~0.016). It turns out MY installed statsmodels (0.15.0) silently added
# an internal standardize=True default to variance_inflation_factor() —
# undocumented behavior I never explicitly relied on but was accidentally
# protected by. The user's real statsmodels (0.14.5, confirmed via their own
# `pip show` output) has NO such parameter at all — it computes the VIF
# regression directly on whatever scale you hand it. I installed statsmodels
# 0.14.5 in an isolated venv and fed it this script's real raw Xm for the
# post-avg_occupancy_7d iteration: it returned 2777.849300641239 for
# avg_occupancy_30d — matching the user's real reported 2777.85 EXACTLY. Then
# I standardized that same data myself first and re-ran it through the SAME
# 0.14.5 install: it returned 5.75, matching my own well-conditioned result
# on statsmodels 0.15.0 to full precision. This is airtight, cross-version
# verified: the entire remaining divergence was statsmodels silently
# behaving differently across versions for the exact same math, because VIF
# is mathematically scale-invariant but was being computed on wildly
# different raw-feature scales without explicit standardization.
#
# The fix: standardize Xm ourselves, explicitly, before every call to
# variance_inflation_factor() — both in the pruning loop and in the final_vif
# report — so the result no longer depends on which statsmodels version
# happens to be installed.
def prune_vif(X, threshold=VIF_THRESHOLD, rank_eps=1e-8):
    cols, vif_log = list(X.columns), []
    while True:
        Xm = X[cols].copy()
        for c in Xm.columns:
            m = Xm[c].median()
            Xm[c] = Xm[c].replace([np.inf, -np.inf], np.nan).fillna(m if not pd.isna(m) else 0)
        ok = [c for c in cols if Xm[c].std() > 1e-10]
        if len(ok) < 2:
            break
        Xm = Xm[ok]

        # Revision 3: detect exact/near-exact rank deficiency before trusting
        # variance_inflation_factor()'s output.
        Xstd = (Xm - Xm.mean()) / Xm.std()
        sv = np.linalg.svd(Xstd.values, compute_uv=False)
        if sv[-1] < rank_eps * sv[0]:
            _, _, vt = np.linalg.svd(Xstd.values, full_matrices=False)
            worst_idx = int(np.argmax(np.abs(vt[-1])))
            worst = ok[worst_idx]
            vif_log.append({"feature": worst, "vif": float("inf")})
            cols = [c for c in cols if c != worst]
            continue

        # *** CHANGED *** (revision 4): pass the STANDARDIZED matrix (Xstd,
        # already computed above for the rank check), not raw Xm — VIF is
        # scale-invariant in theory, so this doesn't change the mathematically
        # correct answer, but it makes the numerical computation itself
        # identical across statsmodels versions instead of silently depending
        # on whether that version standardizes internally.
        vifs = [variance_inflation_factor(Xstd.values, k) for k in range(len(ok))]
        vs = pd.Series(vifs, index=ok)
        worst, wval = vs.idxmax(), vs.max()
        if wval <= threshold:
            break
        vif_log.append({"feature": worst, "vif": round(wval, 2)})
        cols = [c for c in cols if c != worst]
    if len(cols) >= 2:
        Xm = X[cols].copy()
        for c in Xm.columns:
            m = Xm[c].median()
            Xm[c] = Xm[c].replace([np.inf, -np.inf], np.nan).fillna(m if not pd.isna(m) else 0)
        ok = [c for c in cols if Xm[c].std() > 1e-10]
        Xm = Xm[ok]
        # *** CHANGED *** (revision 4): standardize here too, for the same reason.
        Xstd_final = (Xm - Xm.mean()) / Xm.std()
        final_vif = pd.DataFrame({
            "feature": ok,
            "vif": [variance_inflation_factor(Xstd_final.values, k) for k in range(len(ok))],
        }).sort_values("vif", ascending=False)
    else:
        final_vif = pd.DataFrame(columns=["feature", "vif"])
    return cols, final_vif, pd.DataFrame(vif_log)


def prune_features(ev, pool, tag=""):
    """v7 logic, with two determinism fixes in the importance-screening step
    (see the CHANGE LOG at the top of this file for why). Both are marked
    *** CHANGED *** below; everything else, including drop_correlated() and
    prune_vif() themselves, is unchanged from v7."""
    print("\n[5] Multicollinearity pruning...")
    X_all = ev[pool].copy()

    print(f"    Correlation pruning (r >= {CORR_THRESHOLD})...")
    t0 = time.time()
    kept_corr, corr_log = drop_correlated(X_all)
    print(f"    {len(pool)} -> {len(kept_corr)}  ({len(corr_log)} pairs, {time.time()-t0:.1f}s)")
    corr_log.to_csv(OUTPUT_DIR / f"{tag}multicollinearity_correlation_pairs.csv", index=False)

    # *** CHANGED ***: pin kept_corr to a fixed (alphabetical) column order
    # before it's used to build X_proxy. drop_correlated() emits kept_corr in
    # whatever order X_all's columns happened to be in, which is incidental
    # (depends on upstream dataframe construction) and NOT meaningful — but
    # because colsample_bytree below samples columns by index position, that
    # incidental order was silently steering which features got grouped into
    # which trees. Sorting makes the order — and therefore the result — a
    # fixed function of the feature names alone, reproducible across any
    # environment given the same kept_corr set.
    kept_corr = sorted(kept_corr)

    print(f"    Importance screening -> top-{TOP_K}...")
    y_proxy = ev["quoted_adr"].values
    X_proxy = ev[kept_corr].copy().astype(float)
    quick = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        # *** CHANGED *** (revision 2) was subsample=0.8, colsample_bytree=0.8,
        # tree_method left at its (hist) default. This proxy model only ranks
        # features for the top-K cut and is discarded immediately after — it
        # never generalizes anywhere, so none of its own regularization
        # matters. At the original settings, each tree saw a random subset of
        # both rows (subsample) and feature *indices* (colsample_bytree), so
        # the ranking depended on column order and on subtle platform-level
        # RNG/threading behavior as much as on the data — confirmed to cause
        # real cross-environment AND, on at least one user's machine,
        # same-machine run-to-run divergence. At subsample=1.0,
        # colsample_bytree=1.0, tree_method="exact": every tree sees the full,
        # identical 191-feature/1,705-row input, with no histogram/quantile
        # approximation step — nothing left for any RNG or thread-scheduling
        # difference to act on. Verified bit-identical (raw float32
        # importances) across 4 separate process runs in this environment;
        # fit time ~1.4s, so "exact" costs nothing meaningful at this size.
        # The final trained model is unaffected — it does its own independent
        # Optuna-tuned subsample/colsample_bytree search in train_model().
        subsample=1.0, colsample_bytree=1.0, tree_method="exact",
        random_state=RANDOM_STATE, verbosity=0, n_jobs=1,
    )
    quick.fit(X_proxy, y_proxy)
    imp_series = pd.Series(quick.feature_importances_, index=kept_corr).sort_values(ascending=False)
    imp_series.to_csv(OUTPUT_DIR / f"{tag}importance_pool_screening.csv", header=["importance"])
    top_k = imp_series.head(TOP_K).index.tolist()

    print(f"    VIF pruning (VIF >= {VIF_THRESHOLD}) on top-{TOP_K}...")
    t0 = time.time()
    kept_vif, vif_final, vif_log = prune_vif(ev[top_k])
    print(f"    {TOP_K} -> {len(kept_vif)}  ({len(vif_log)} removed, {time.time()-t0:.1f}s)")
    vif_final.to_csv(OUTPUT_DIR / f"{tag}multicollinearity_vif_final.csv", index=False)
    if len(vif_log):
        vif_log.to_csv(OUTPUT_DIR / f"{tag}multicollinearity_vif_removed.csv", index=False)

    print(f"\n    Base feature pool: {len(kept_vif)} features")
    return kept_vif


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — TRAIN  (verbatim from v7)
# ═══════════════════════════════════════════════════════════════════════════
def prep_X(df, features):
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
    return X.astype(float)


def optuna_search(X_tr, y_tr, is_clf=False):
    if not HAS_OPTUNA:
        return {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "reg_alpha": 0.5, "reg_lambda": 1.5,
            "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1,
        }
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
            "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1,
        }
        if is_clf:
            p["gamma"] = trial.suggest_float("gamma", 0.0, 1.0)
        return -cross_val_score(
            xgb.XGBRegressor(**p), X_tr, y_tr,
            cv=cv, scoring="neg_mean_absolute_error", n_jobs=1,
        ).mean()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    bp = study.best_params
    bp.update({"random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1})
    return bp


def train_model(name, ev, features, target,
                is_clf=False, filter_booked=False, positive_only=False,
                filter_rooms=False, scale_pos_weight=None):
    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"{'-'*60}")

    df = ev.copy()
    if filter_booked:
        df = df[df["status"] == "Booked"]
    if positive_only:
        df = df[df[target] > 0]
    if filter_rooms:
        before = len(df)
        df = df[df["room_block"] > 0]
        print(f"  [FIX 1] Filtered room_block>0: {before} -> {len(df)} rows "
              f"({before-len(df)} meeting-only records excluded)")
    df = df[df[target].notna()]

    af = [f for f in features if f in df.columns]
    X  = prep_X(df, af)
    y  = df[target].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20,
        stratify=(y > y.mean()).astype(int) if is_clf else None,
        random_state=RANDOM_STATE,
    )
    print(f"  n={len(df):,}  train={len(X_tr)}  test={len(X_te)}  features={len(af)}")

    t0 = time.time()
    print(f"  Hyperparameter search ({N_OPTUNA} Optuna trials)..." if HAS_OPTUNA
          else "  Using fixed default hyperparameters (optuna not installed)...")
    bp = optuna_search(X_tr, y_tr, is_clf)

    if scale_pos_weight is not None:
        bp["scale_pos_weight"] = scale_pos_weight

    print(f"  Best: n_est={bp['n_estimators']} depth={bp['max_depth']} "
          f"lr={bp['learning_rate']:.4f}  ({time.time()-t0:.0f}s)")

    model = xgb.XGBRegressor(**bp)
    model.fit(X_tr, y_tr)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_sc = cross_val_score(
        xgb.XGBRegressor(**bp), X, y,
        cv=cv, scoring="neg_mean_absolute_error", n_jobs=1,
    )

    metrics = {
        "train_mae": float(mean_absolute_error(y_tr, pred_tr)),
        "test_mae":  float(mean_absolute_error(y_te, pred_te)),
        "train_r2":  float(r2_score(y_tr, pred_tr)),
        "test_r2":   float(r2_score(y_te, pred_te)),
    }
    print(f"  MAE {metrics['train_mae']:.4f}/{metrics['test_mae']:.4f}  "
          f"R2 {metrics['train_r2']:.4f}/{metrics['test_r2']:.4f}")

    metrics["cv_mae_mean"] = float(-cv_sc.mean())
    metrics["cv_mae_std"]  = float(cv_sc.std())
    metrics["n_train"]     = len(X_tr)
    metrics["n_test"]      = len(X_te)
    print(f"  CV MAE {metrics['cv_mae_mean']:.4f} +/- {metrics['cv_mae_std']:.4f}")

    imp = pd.DataFrame({"feature": af, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp["importance_pct"] = imp["importance"] / imp["importance"].sum() * 100
    imp["rank"] = range(1, len(imp) + 1)

    print("\n  Top 15 features:")
    for _, r in imp.head(15).iterrows():
        print(f"    {int(r['rank']):>2}. {r['feature']:<44} {r['importance_pct']:>6.2f}%")

    return {"model": model, "features": af, "metrics": metrics,
            "importance": imp, "best_params": bp}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — ADR ONLY, FILTER-FIRST (the one change vs. v7's main())
# ═══════════════════════════════════════════════════════════════════════════
def exclude_leaky(features, model_name):
    ll = {x.lower() for x in LEAKY.get(model_name, set())}
    return [f for f in features if f.lower() not in ll]


def run_adr_pipeline():
    t_total = time.time()
    print("=" * 70)
    print("  RFP quoted_adr PIPELINE — filter-first variant")
    print("=" * 70)

    ev = load_data()
    ev = join_transient(ev)
    ev = engineer_features(ev)

    print("\n>>> Filtering room_block > 0 BEFORE multicollinearity/importance/VIF "
          "pruning (this is the deliberate change vs. v7's main()).")
    before = len(ev)
    ev_adr = ev[ev["room_block"] > 0].copy().reset_index(drop=True)
    print(f"    {before} -> {len(ev_adr)} rows ({before - len(ev_adr)} meeting-only rows excluded)")

    pool = build_candidate_pool(ev_adr)
    base = prune_features(ev_adr, pool, tag="adr_filtfirst_")
    adr_feats = exclude_leaky(base, "quoted_adr")
    dropped_leaky = sorted(set(base) - set(adr_feats))
    print(f"\n    Leaky-excluded for quoted_adr: {dropped_leaky if dropped_leaky else '(none)'}")
    print(f"    Final quoted_adr feature list: {len(adr_feats)} features")

    # Train on the FULL ev (train_model re-applies the same room_block>0 filter
    # internally, so the training population is identical to ev_adr above).
    result = train_model("Quoted ADR (filter-first)", ev, adr_feats, "quoted_adr",
                          filter_rooms=True)

    summary = {
        "candidate_pool_size": len(pool),
        "base_pool_size": len(base),
        "adr_feats_size": len(adr_feats),
        "adr_feats": adr_feats,
        "leaky_dropped": dropped_leaky,
        "metrics": result["metrics"],
        "best_params": result["best_params"],
    }
    with open(OUTPUT_DIR / "adr_filtered_first_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    result["importance"].to_csv(OUTPUT_DIR / "adr_filtered_first_importance.csv", index=False)

    # Pickle the trained model, bundled with everything needed to score new
    # rows with it later without having to re-run the pipeline: the exact
    # feature list IN ORDER (prep_X requires this — same names, same order
    # the model was fit on), the tuned hyperparameters, the evaluation
    # metrics from this run, and library versions actually used to train it
    # (given everything this pipeline has been through re: version-dependent
    # numerics, recording them here is cheap insurance for anyone loading
    # this pickle on a different machine later).
    model_bundle = {
        "model": result["model"],
        "features": result["features"],
        "target": "quoted_adr",
        "best_params": result["best_params"],
        "metrics": result["metrics"],
        "candidate_pool_size": len(pool),
        "base_pool_size": len(base),
        "leaky_dropped": dropped_leaky,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": {
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    model_path = OUTPUT_DIR / "adr_filtered_first_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n    Model saved: {model_path}  "
          f"(dict with keys: {', '.join(model_bundle.keys())})")
    print(f"    To reload: pickle.load(open('{model_path.name}', 'rb'))['model'].predict(X)")
    print(f"    — X must have exactly bundle['features'], in that order (see prep_X()).")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"  Total time: {(time.time() - t_total) / 60:.1f} min")
    print(f"{'=' * 70}")
    return result, summary


if __name__ == "__main__":
    run_adr_pipeline()
