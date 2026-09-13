"""
rfp_conversion_pipeline_filtered_first.py
═══════════════════════════════════════════════════════════════════════════════
did_convert (win/loss) model, re-run on the SAME filtered population as the
ADR model (rfp_adr_pipeline_filtered_first_final.py): room_block > 0 only,
1,899 -> 1,705 rows. This is a deliberate divergence from
rfp_conversion_pipeline.py, which trains on all 1,899 rows (meeting-only RFPs
included) because did_convert isn't structurally degenerate for room_block==0
rows the way quoted_adr is.

Nothing about that reasoning changed. This script exists because the user
asked, separately, to also see the conversion model run on the *same 1,705
rows* the ADR model uses, so the two can be compared apples-to-apples on one
population. Both conversion variants are legitimate; they answer different
questions ("does the RFP convert, over the full pipeline" vs. "does the RFP
convert, restricted to the population that also gets an ADR quote").

── WHY THIS SCRIPT LOADS DATA DIFFERENTLY FROM BOTH ORIGINAL SCRIPTS ─────────
Only one input file was provided for this run:
    rfp_training_data_complete_v3_with_transient.csv  (1,899 rows x 289 cols)
Neither outputs/Enhanced_All_Events_v8_final.csv nor
filesData/outputs/Nexus_Transient_Demand_v2.csv was available, so this script
does not call the original load_data()/join_transient(). Instead it reads the
supplied CSV directly, which already carries the transient join (tr_* columns
present) — i.e. it's the state right after STAGE 2 in the original scripts.

Checked directly against what STAGE 3 (engineer_features) normally produces:
this CSV already has arrival_month_sin/cos, arrival_quarter_sin,
days_to_peak_season, budget_ratio, pricing_pressure_index, and
actual_fnb_per_person under those exact names (so those are NOT recomputed
here — recomputing days_to_peak_season etc. would need a capitalized
"Arrival_Date" column that this CSV doesn't have; the lowercase "arrival_date"
string column is what's present instead). But it's missing did_convert,
is_meeting_only, group_size_tier_clean_enc, the market_segment one-hot
columns, and _enc codes for the remaining object columns — and
revenue_quality_score/occupancy_acceleration are still in their raw,
untransformed state (skew=6.49 on revenue_quality_score, confirmed directly;
log1p hasn't been applied). engineer_features_filtered() below fills in
exactly those gaps, verbatim to the logic in rfp_conversion_pipeline.py's
engineer_features(), and leaves every already-present column untouched.

── FILTER-FIRST ORDERING (same change as the ADR script, same reason) ────────
    ev = engineer_features_filtered(ev)              # full 1,899 rows
    ev = ev[ev.room_block > 0]                        # 1,899 -> 1,705 rows
    pool = build_candidate_pool(ev)                   # on the FILTERED population
    base = prune_features(ev, pool, ...)              # corr + importance + VIF, FILTERED
    conv_feats = exclude_leaky(base, LEAKY_CONVERSION)
    train_model_clf(..., ev)                          # trained + evaluated on the FILTERED population
scale_pos_weight is likewise computed on the filtered 1,705 rows, not the
full 1,899 — so it (and every pruning decision) reflects the actual training
population, exactly as in the ADR script.

── WHAT IS REUSED VERBATIM FROM rfp_conversion_pipeline.py ───────────────────
build_candidate_pool(), drop_correlated(), prune_vif(), prune_features(),
exclude_leaky(), prep_X(), optuna_search_clf(), train_model_clf() — none of
these look at how the population was assembled, so filtering earlier doesn't
require touching their internals. LEAK_EXCLUDE (win_streak +
win_rate_ema_7d/30d/90d, the four columns proven leaky by single-feature
ROC-AUC in the original script) and LEAKY_CONVERSION ({"actual_pickup_rate"})
are unchanged for the same reason — those are properties of the columns
themselves, not of which rows are in the population.

── OUTPUT NAMING (so old vs. new is unambiguous) ─────────────────────────────
Everything from this run is written with a "conversion_filtered_first_"
prefix — conversion_filtered_first_summary.json,
conversion_filtered_first_importance.csv, conversion_filtered_first_model.pkl
— so it never collides with rfp_conversion_pipeline.py's plain
"conversion_*" (1,899-row) outputs, and it mirrors the ADR script's own
"adr_filtered_first_*" naming for the same methodology.

Input (place under this relative path or edit the constant below):
    rfp_training_data_complete_v3_with_transient.csv

Requires: pandas, numpy, xgboost, scikit-learn, scipy, statsmodels.
Optional: optuna (30-trial hyperparameter search). Falls back to fixed
default hyperparameters if optuna is not installed.

Usage:
    python rfp_conversion_pipeline_filtered_first.py
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed — using default hyperparameters. "
          "Install with: pip install optuna")

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_FILE     = "rfp_training_data_complete_v3_with_transient.csv"
OUTPUT_DIR     = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET         = "did_convert"   # == is_won; created below if not already present

LEAK_EXCLUDE = {"win_streak", "win_rate_ema_7d", "win_rate_ema_30d", "win_rate_ema_90d"}
CORR_THRESHOLD = 0.85
VIF_THRESHOLD  = 10.0
TOP_K          = 60
CV_FOLDS       = 5
RANDOM_STATE   = 42
N_OPTUNA       = 30

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

LEAKY_CONVERSION = {"actual_pickup_rate"}

IDENTITY_COLS = {
    "rfp_id", "inquiry_date", "arrival_date", "account_name", "event_name",
    "status", "lost_reason", "Arrival_Date", "Departure_Date", "Arrival_DOW_Name",
    "Displacement_Risk_Level", "Demand_Tier", "arrival_dow_name",
    "displacement_risk_level", "demand_tier",
}


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1/2 — LOAD (reads the pre-merged CSV directly; no v8/transient join
# needed since this file already carries the tr_* transient columns)
# ═══════════════════════════════════════════════════════════════════════════
def load_data_filtered():
    print("[1] Loading pre-merged data...")
    ev = pd.read_csv(INPUT_FILE)
    print(f"    {ev.shape[0]:,} rows x {ev.shape[1]} cols")
    ev["inquiry_date"] = pd.to_datetime(ev["inquiry_date"])
    ev = ev.sort_values("inquiry_date").reset_index(drop=True)
    return ev


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — FILL IN ONLY THE ENGINEERED COLUMNS THIS CSV IS MISSING
# (see module docstring for exactly what's already present vs. not)
# ═══════════════════════════════════════════════════════════════════════════
def engineer_features_filtered(ev):
    print("\n[2] Encoding categoricals and engineering the missing features...")

    if "group_size_tier" in ev.columns and "is_meeting_only" not in ev.columns:
        ev["is_meeting_only"] = (ev["group_size_tier"] == "meeting_only").astype(int)
        size_order = {"small": 0, "medium": 1, "large": 2, "very_large": 3}
        ev["group_size_tier_clean_enc"] = (
            ev["group_size_tier"].map(size_order).fillna(-1).astype(int)
        )
        n_mo = ev["is_meeting_only"].sum()
        print(f"    is_meeting_only: {n_mo} records extracted ({n_mo/len(ev)*100:.1f}%)")

    ohe_cols = []
    if "market_segment" in ev.columns:
        existing_ohe = [c for c in ev.columns if c.startswith("market_seg_")]
        if not existing_ohe:
            ohe = pd.get_dummies(ev["market_segment"], prefix="market_seg", drop_first=False)
            ohe = ohe.astype(int)
            ev = pd.concat([ev, ohe], axis=1)
            ohe_cols = ohe.columns.tolist()
            print(f"    market_segment one-hot: {len(ohe_cols)} columns")
    ev.attrs["market_seg_ohe_cols"] = ohe_cols

    SKIP_ENCODE = IDENTITY_COLS | {"group_size_tier", "market_segment"}
    for c in ev.select_dtypes(include="object").columns:
        if c in SKIP_ENCODE:
            continue
        enc_col = f"{c}_enc"
        if enc_col in ev.columns:
            continue
        ev[enc_col] = pd.Categorical(ev[c]).codes
        print(f"    Encoded: {c} ({ev[c].nunique()} cats)")

    if "revenue_quality_score" in ev.columns:
        skew_before = ev["revenue_quality_score"].skew()
        if skew_before > 2.0:  # not yet log-transformed
            ev["revenue_quality_score"] = np.log1p(ev["revenue_quality_score"])
            print(f"    log1p(revenue_quality_score): skew {skew_before:.2f} -> "
                  f"{ev['revenue_quality_score'].skew():.2f}")

    if "occupancy_acceleration" in ev.columns:
        p1, p99 = ev["occupancy_acceleration"].quantile([0.01, 0.99])
        if ev["occupancy_acceleration"].min() < p1 or ev["occupancy_acceleration"].max() > p99:
            ev["occupancy_acceleration"] = ev["occupancy_acceleration"].clip(p1, p99)
            print(f"    occupancy_acceleration winsorised: [{p1:.3f}, {p99:.3f}]")

    if "did_convert" not in ev.columns:
        if "is_won" in ev.columns:
            ev["did_convert"] = ev["is_won"].astype(int)
        else:
            ev["did_convert"] = (ev["status"] == "Booked").astype(int)
        print(f"    did_convert created: {ev['did_convert'].sum()} won / "
              f"{(1 - ev['did_convert']).sum()} lost")

    if "total_room_nights" not in ev.columns:
        ev["total_room_nights"] = ev.get(
            "room_block", ev.get("total_room_nights_requested", np.nan)
        )

    print(f"    Dataset after engineering: {ev.shape[1]} cols")
    return ev


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — CANDIDATE POOL (verbatim from rfp_conversion_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════
def build_candidate_pool(ev):
    print("\n[3] Building candidate feature pool...")
    numeric = ev.select_dtypes(include=[np.number]).columns.tolist()
    pool = [
        c for c in numeric
        if c not in ALWAYS_EXCLUDE
        and c not in LEAK_EXCLUDE
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
# STAGE 5 — MULTICOLLINEARITY PRUNING (verbatim from rfp_conversion_pipeline.py,
# which itself carries the ADR script's revision 1-4 fixes)
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

        Xstd = (Xm - Xm.mean()) / Xm.std()
        sv = np.linalg.svd(Xstd.values, compute_uv=False)
        if sv[-1] < rank_eps * sv[0]:
            _, _, vt = np.linalg.svd(Xstd.values, full_matrices=False)
            worst_idx = int(np.argmax(np.abs(vt[-1])))
            worst = ok[worst_idx]
            vif_log.append({"feature": worst, "vif": float("inf")})
            cols = [c for c in cols if c != worst]
            continue

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
        Xstd_final = (Xm - Xm.mean()) / Xm.std()
        final_vif = pd.DataFrame({
            "feature": ok,
            "vif": [variance_inflation_factor(Xstd_final.values, k) for k in range(len(ok))],
        }).sort_values("vif", ascending=False)
    else:
        final_vif = pd.DataFrame(columns=["feature", "vif"])
    return cols, final_vif, pd.DataFrame(vif_log)


def prune_features(ev, pool, target, is_clf, scale_pos_weight=None, tag=""):
    print("\n[4] Multicollinearity pruning...")
    X_all = ev[pool].copy()

    print(f"    Correlation pruning (r >= {CORR_THRESHOLD})...")
    t0 = time.time()
    kept_corr, corr_log = drop_correlated(X_all)
    print(f"    {len(pool)} -> {len(kept_corr)}  ({len(corr_log)} pairs, {time.time()-t0:.1f}s)")
    corr_log.to_csv(OUTPUT_DIR / f"{tag}multicollinearity_correlation_pairs.csv", index=False)

    kept_corr = sorted(kept_corr)

    print(f"    Importance screening -> top-{TOP_K}...")
    y_proxy = ev[target].values
    X_proxy = ev[kept_corr].copy().astype(float)
    if is_clf:
        quick = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=1.0, colsample_bytree=1.0, tree_method="exact",
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=1,
        )
    else:
        quick = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
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


def exclude_leaky(features, leaky_set):
    ll = {x.lower() for x in leaky_set}
    return [f for f in features if f.lower() not in ll]


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — TRAIN (classifier) — verbatim from rfp_conversion_pipeline.py
# ═══════════════════════════════════════════════════════════════════════════
def prep_X(df, features):
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
    return X.astype(float)


def optuna_search_clf(X_tr, y_tr, scale_pos_weight):
    if not HAS_OPTUNA:
        return {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "reg_alpha": 0.5, "reg_lambda": 1.5, "gamma": 0.0,
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1,
        }
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

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
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1,
        }
        scores = cross_val_score(
            xgb.XGBClassifier(**p), X_tr, y_tr,
            cv=cv, scoring="roc_auc", n_jobs=1,
        )
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    bp = study.best_params
    bp.update({
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1,
    })
    return bp


def train_model_clf(name, ev, features, target):
    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"{'-'*60}")

    df = ev.copy()
    df = df[df[target].notna()]

    af = [f for f in features if f in df.columns]
    X  = prep_X(df, af)
    y  = df[target].astype(int).values

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"  n={len(df):,}  won={n_pos} ({n_pos/len(df):.1%})  lost={n_neg} ({n_neg/len(df):.1%})"
          f"  scale_pos_weight={scale_pos_weight:.4f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"  train={len(X_tr)}  test={len(X_te)}  features={len(af)}")

    t0 = time.time()
    print(f"  Hyperparameter search ({N_OPTUNA} Optuna trials, maximizing CV ROC-AUC)..." if HAS_OPTUNA
          else "  Using fixed default hyperparameters (optuna not installed)...")
    bp = optuna_search_clf(X_tr, y_tr, scale_pos_weight)
    print(f"  Best: n_est={bp['n_estimators']} depth={bp['max_depth']} "
          f"lr={bp['learning_rate']:.4f}  ({time.time()-t0:.0f}s)")

    model = xgb.XGBClassifier(**bp)
    model.fit(X_tr, y_tr)

    proba_tr, proba_te = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_te)[:, 1]
    pred_tr, pred_te = (proba_tr >= 0.5).astype(int), (proba_te >= 0.5).astype(int)

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_sc = cross_val_score(xgb.XGBClassifier(**bp), X, y, cv=cv, scoring="roc_auc", n_jobs=1)

    metrics = {
        "train_accuracy":  float(accuracy_score(y_tr, pred_tr)),
        "test_accuracy":   float(accuracy_score(y_te, pred_te)),
        "train_precision": float(precision_score(y_tr, pred_tr)),
        "test_precision":  float(precision_score(y_te, pred_te)),
        "train_recall":    float(recall_score(y_tr, pred_tr)),
        "test_recall":     float(recall_score(y_te, pred_te)),
        "train_f1":        float(f1_score(y_tr, pred_tr)),
        "test_f1":         float(f1_score(y_te, pred_te)),
        "train_roc_auc":   float(roc_auc_score(y_tr, proba_tr)),
        "test_roc_auc":    float(roc_auc_score(y_te, proba_te)),
        "train_log_loss":  float(log_loss(y_tr, proba_tr)),
        "test_log_loss":   float(log_loss(y_te, proba_te)),
        "cv_roc_auc_mean": float(cv_sc.mean()),
        "cv_roc_auc_std":  float(cv_sc.std()),
        "n_train": len(X_tr), "n_test": len(X_te),
        "scale_pos_weight": scale_pos_weight,
    }
    print(f"  Test:  accuracy={metrics['test_accuracy']:.4f}  precision={metrics['test_precision']:.4f}  "
          f"recall={metrics['test_recall']:.4f}  F1={metrics['test_f1']:.4f}")
    print(f"  ROC-AUC {metrics['train_roc_auc']:.4f}/{metrics['test_roc_auc']:.4f}  "
          f"log-loss {metrics['train_log_loss']:.4f}/{metrics['test_log_loss']:.4f}")
    print(f"  CV ROC-AUC {metrics['cv_roc_auc_mean']:.4f} +/- {metrics['cv_roc_auc_std']:.4f}")

    cm_test = confusion_matrix(y_te, pred_te)
    print(f"  Test confusion matrix [ [TN FP] [FN TP] ]:\n{cm_test}")

    imp = pd.DataFrame({"feature": af, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp["importance_pct"] = imp["importance"] / imp["importance"].sum() * 100
    imp["rank"] = range(1, len(imp) + 1)

    print("\n  Top 15 features:")
    for _, r in imp.head(15).iterrows():
        print(f"    {int(r['rank']):>2}. {r['feature']:<44} {r['importance_pct']:>6.2f}%")

    return {
        "model": model, "features": af, "metrics": metrics,
        "importance": imp, "best_params": bp,
        "confusion_matrix_test": cm_test.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def run_conversion_pipeline_filtered_first():
    t_total = time.time()
    print("=" * 70)
    print("  RFP conversion (did_convert) PIPELINE — filter-first, room_block>0")
    print("=" * 70)

    ev = load_data_filtered()
    ev = engineer_features_filtered(ev)

    before = len(ev)
    ev = ev[ev["room_block"] > 0].copy().reset_index(drop=True)
    print(f"\n>>> Filtered room_block > 0 BEFORE multicollinearity/importance/VIF "
          f"pruning (same population as the ADR model): {before} -> {len(ev)} rows "
          f"({before - len(ev)} meeting-only rows excluded).")

    n_pos_full = int(ev[TARGET].sum())
    n_neg_full = len(ev) - n_pos_full
    scale_pos_weight = n_neg_full / n_pos_full

    pool = build_candidate_pool(ev)
    base = prune_features(ev, pool, target=TARGET, is_clf=True,
                           scale_pos_weight=scale_pos_weight, tag="conversion_filtered_first_")
    conv_feats = exclude_leaky(base, LEAKY_CONVERSION)
    dropped_leaky = sorted(set(base) - set(conv_feats))
    print(f"\n    Leaky-excluded for conversion: {dropped_leaky if dropped_leaky else '(none)'}")
    print(f"    Final conversion feature list: {len(conv_feats)} features")

    result = train_model_clf("Conversion (did_convert) — filtered-first, n=1,705", ev, conv_feats, TARGET)

    summary = {
        "target": TARGET,
        "population": "room_block > 0 (filtered-first, matches ADR model population)",
        "n_rows": len(ev),
        "candidate_pool_size": len(pool),
        "base_pool_size": len(base),
        "conversion_feats_size": len(conv_feats),
        "conversion_feats": conv_feats,
        "leaky_dropped": dropped_leaky,
        "metrics": result["metrics"],
        "best_params": result["best_params"],
        "confusion_matrix_test": result["confusion_matrix_test"],
    }
    with open(OUTPUT_DIR / "conversion_filtered_first_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    result["importance"].to_csv(OUTPUT_DIR / "conversion_filtered_first_importance.csv", index=False)

    model_bundle = {
        "model": result["model"],
        "features": result["features"],
        "target": TARGET,
        "population": "room_block > 0 (n=1,705)",
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
    model_path = OUTPUT_DIR / "conversion_filtered_first_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n    Model saved: {model_path}  (dict with keys: {', '.join(model_bundle.keys())})")
    print(f"    To reload: pickle.load(open('{model_path.name}', 'rb'))['model'].predict_proba(X)[:, 1]")
    print(f"    — X must have exactly bundle['features'], in that order (see prep_X()).")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"  Total time: {(time.time() - t_total) / 60:.1f} min")
    print(f"{'=' * 70}")
    return result, summary


if __name__ == "__main__":
    run_conversion_pipeline_filtered_first()
