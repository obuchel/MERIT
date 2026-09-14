"""
meeting_revenue_pipeline.py
═══════════════════════════════════════════════════════════════════════════
"ADR-equivalent" model for the meeting-only complement (room_block == 0,
n=194). quoted_adr cannot be used here -- it is structurally $0 for every
meeting-only RFP (no rooms means no room rate to quote), so there is no
willingness-to-pay-per-room target the way there is for the room-based
population.

Substitute target: proposed_total_revenue -- the deal-size analog. Instead
of "what room rate will this RFP support", this model answers "how large a
deal (in dollars) is this meeting-only RFP", which is the meaningful
value-prediction question when there's no room rate to predict.

Leakage note: because proposed_total_revenue is now the TARGET (previously
it was just a feature), several columns that are exact or near-exact
functions of it must be excluded, on top of the model's own past outcome
columns:
  - revenue_intensity            -- verified identical (r=1.0) to the target
                                     for this subset (see analysis)
  - revenue_quality_score        -- also r=1.0 (log-transform of the same
                                     underlying revenue figure for this pop.)
  - risk_adjusted_revenue,
    expected_value               -- r~0.98, built directly from proposed
                                     revenue x a probability factor
  - revenue_intensity_deviation,
    priority_revenue_component   -- built from revenue_intensity /
                                     proposed_total_revenue directly
  - budget_ratio                 -- = budget_amount / proposed_total_revenue,
                                     leaky by construction once revenue IS
                                     the target
  - actual_total_revenue, actual_fnb_per_person, revenue_realization_pct,
    actual_pickup_rate, is_won, did_convert -- post-outcome, already
    excluded in the base pipeline for the same reason quoted_adr excludes
    them

TOP_K reduced to 20, same tiny-N safeguard as the meeting_conversion model.
"""
import warnings; warnings.filterwarnings("ignore")
import time, json, pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

INPUT_FILE = "/mnt/user-data/uploads/rfp_training_data_complete_v3_with_transient.csv"
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "proposed_total_revenue"
CORR_THRESHOLD = 0.85
VIF_THRESHOLD = 10.0
TOP_K = 20
CV_FOLDS = 5
RANDOM_STATE = 42
N_OPTUNA = 30

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
    "fnb_ratio", "group_size_tier_enc", "market_segment_enc",
    TARGET,  # the target itself
    # leaky-by-construction w.r.t. THIS target (see module docstring)
    "revenue_intensity", "revenue_quality_score", "risk_adjusted_revenue",
    "expected_value", "revenue_intensity_deviation", "priority_revenue_component",
    "budget_ratio",
}
IDENTITY_COLS = {
    "rfp_id", "inquiry_date", "arrival_date", "account_name", "event_name",
    "status", "lost_reason", "Arrival_Date", "Departure_Date", "Arrival_DOW_Name",
    "Displacement_Risk_Level", "Demand_Tier", "arrival_dow_name",
    "displacement_risk_level", "demand_tier",
}


def load_data_filtered():
    ev = pd.read_csv(INPUT_FILE)
    ev["inquiry_date"] = pd.to_datetime(ev["inquiry_date"])
    return ev.sort_values("inquiry_date").reset_index(drop=True)


def engineer_features_filtered(ev):
    if "group_size_tier" in ev.columns and "is_meeting_only" not in ev.columns:
        ev["is_meeting_only"] = (ev["group_size_tier"] == "meeting_only").astype(int)
        size_order = {"small": 0, "medium": 1, "large": 2, "very_large": 3}
        ev["group_size_tier_clean_enc"] = ev["group_size_tier"].map(size_order).fillna(-1).astype(int)

    if "market_segment" in ev.columns and not [c for c in ev.columns if c.startswith("market_seg_")]:
        ohe = pd.get_dummies(ev["market_segment"], prefix="market_seg", drop_first=False).astype(int)
        ev = pd.concat([ev, ohe], axis=1)

    SKIP_ENCODE = IDENTITY_COLS | {"group_size_tier", "market_segment"}
    for c in ev.select_dtypes(include="object").columns:
        if c in SKIP_ENCODE:
            continue
        enc_col = f"{c}_enc"
        if enc_col not in ev.columns:
            ev[enc_col] = pd.Categorical(ev[c]).codes

    if "occupancy_acceleration" in ev.columns:
        p1, p99 = ev["occupancy_acceleration"].quantile([0.01, 0.99])
        ev["occupancy_acceleration"] = ev["occupancy_acceleration"].clip(p1, p99)

    if "total_room_nights" not in ev.columns:
        ev["total_room_nights"] = ev.get("room_block", ev.get("total_room_nights_requested", np.nan))

    return ev


def build_candidate_pool(ev):
    numeric = ev.select_dtypes(include=[np.number]).columns.tolist()
    pool = [c for c in numeric if c not in ALWAYS_EXCLUDE
            and not c.endswith("_v8") and ev[c].std() > 1e-10 and ev[c].isna().mean() < 0.50]
    for c in pool:
        med = ev[c].median()
        ev[c] = ev[c].fillna(med if not np.isnan(med) else 0)
        ev[c] = ev[c].replace([np.inf, -np.inf], med if not np.isnan(med) else 0)
    return pool


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
        final_vif = pd.DataFrame({"feature": ok,
                                   "vif": [variance_inflation_factor(Xstd_final.values, k) for k in range(len(ok))]
                                   }).sort_values("vif", ascending=False)
    else:
        final_vif = pd.DataFrame(columns=["feature", "vif"])
    return cols, final_vif, pd.DataFrame(vif_log)


def prune_features(ev, pool, target, tag=""):
    X_all = ev[pool].copy()
    kept_corr, corr_log = drop_correlated(X_all)
    corr_log.to_csv(OUTPUT_DIR / f"{tag}multicollinearity_correlation_pairs.csv", index=False)
    kept_corr = sorted(kept_corr)

    y_proxy = ev[target].values
    X_proxy = ev[kept_corr].copy().astype(float)
    quick = xgb.XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0, tree_method="exact",
        random_state=RANDOM_STATE, verbosity=0, n_jobs=1,
    )
    quick.fit(X_proxy, y_proxy)
    imp_series = pd.Series(quick.feature_importances_, index=kept_corr).sort_values(ascending=False)
    imp_series.to_csv(OUTPUT_DIR / f"{tag}importance_pool_screening.csv", header=["importance"])
    top_k = imp_series.head(TOP_K).index.tolist()

    kept_vif, vif_final, vif_log = prune_vif(ev[top_k])
    vif_final.to_csv(OUTPUT_DIR / f"{tag}multicollinearity_vif_final.csv", index=False)
    if len(vif_log):
        vif_log.to_csv(OUTPUT_DIR / f"{tag}multicollinearity_vif_removed.csv", index=False)
    return kept_vif


def prep_X(df, features):
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
    return X.astype(float)


def optuna_search(X_tr, y_tr):
    if not HAS_OPTUNA:
        return {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.5,
                "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1}
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1,
        }
        return -cross_val_score(xgb.XGBRegressor(**p), X_tr, y_tr, cv=cv,
                                 scoring="neg_mean_absolute_error", n_jobs=1).mean()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    bp = study.best_params
    bp.update({"random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1})
    return bp


def train_model(name, ev, features, target):
    df = ev[ev[target].notna()].copy()
    af = [f for f in features if f in df.columns]
    X = prep_X(df, af)
    y = df[target].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    print(f"  n={len(df)}  train={len(X_tr)} test={len(X_te)}  features={len(af)}")

    bp = optuna_search(X_tr, y_tr)
    model = xgb.XGBRegressor(**bp)
    model.fit(X_tr, y_tr)

    pred_tr, pred_te = model.predict(X_tr), model.predict(X_te)
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_sc = cross_val_score(xgb.XGBRegressor(**bp), X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)

    metrics = {
        "train_mae": float(mean_absolute_error(y_tr, pred_tr)), "test_mae": float(mean_absolute_error(y_te, pred_te)),
        "train_r2": float(r2_score(y_tr, pred_tr)), "test_r2": float(r2_score(y_te, pred_te)),
        "cv_mae_mean": float(-cv_sc.mean()), "cv_mae_std": float(cv_sc.std()),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    imp = pd.DataFrame({"feature": af, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    imp["importance_pct"] = imp["importance"] / imp["importance"].sum() * 100
    imp["rank"] = range(1, len(imp) + 1)
    return {"model": model, "features": af, "metrics": metrics, "importance": imp, "best_params": bp}


def run():
    print("=" * 70)
    print("  MEETING-ONLY proposed_total_revenue (ADR analog) — room_block == 0")
    print("=" * 70)
    ev = load_data_filtered()
    ev = engineer_features_filtered(ev)

    before = len(ev)
    ev = ev[ev["room_block"] == 0].copy().reset_index(drop=True)
    print(f"  Filtered room_block == 0: {before} -> {len(ev)} rows")

    pool = build_candidate_pool(ev)
    print(f"  Candidate pool: {len(pool)} features")
    base = prune_features(ev, pool, target=TARGET, tag="meeting_revenue_")
    print(f"  Final feature list: {len(base)} features")

    result = train_model("Meeting-only proposed_total_revenue", ev, base, TARGET)
    print(f"\n  Test R2: {result['metrics']['test_r2']:.4f}   Test MAE: ${result['metrics']['test_mae']:.2f}")
    print("  Top 10 features:")
    for _, r in result["importance"].head(10).iterrows():
        print(f"    {int(r['rank']):>2}. {r['feature']:<40} {r['importance_pct']:>6.2f}%")

    summary = {
        "target": TARGET, "population": "room_block == 0 (meeting-only, n=194)",
        "n_rows": len(ev), "top_k_used": TOP_K, "candidate_pool_size": len(pool),
        "final_feats_size": len(base), "final_feats": base,
        "metrics": result["metrics"], "best_params": result["best_params"],
    }
    with open(OUTPUT_DIR / "meeting_revenue_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    result["importance"].to_csv(OUTPUT_DIR / "meeting_revenue_importance.csv", index=False)

    model_bundle = {"model": result["model"], "features": result["features"], "target": TARGET,
                     "population": "room_block == 0 (n=194)", "best_params": result["best_params"],
                     "metrics": result["metrics"], "trained_at_utc": datetime.now(timezone.utc).isoformat()}
    with open(OUTPUT_DIR / "meeting_revenue_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result, summary


if __name__ == "__main__":
    run()
