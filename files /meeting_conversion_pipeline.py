"""
meeting_conversion_pipeline.py
═══════════════════════════════════════════════════════════════════════════
Conversion (did_convert / is_won) model for the COMPLEMENT population that
both filtered-first models (ADR and conversion-filtered-first) excluded:
room_block == 0 ("meeting-only" RFPs -- no rooms requested, just meeting
space / F&B). n = 194 (156 won / 38 lost), vs. 1,705 for the room-based
models.

This is a direct analog of rfp_conversion_pipeline_filtered_first.py, with
the filter flipped (room_block == 0 instead of > 0). did_convert is NOT
structurally degenerate for these rows (unlike quoted_adr, which is
literally $0 for every meeting-only RFP), so this model is meaningful.

Everything else -- build_candidate_pool, drop_correlated, prune_vif,
prep_X, optuna_search_clf, train_model_clf -- is reused verbatim from the
uploaded conversion pipeline. The one deliberate addition: TOP_K is reduced
from 60 to 20 for this run. With only ~155 training rows after the 80/20
split, keeping a 60-feature ceiling (a >1:2.5 feature-to-row ratio) would
invite severe overfitting; 20 keeps the ceiling closer to a defensible
~1:8 ratio. This is flagged explicitly in the summary output.
"""
import warnings; warnings.filterwarnings("ignore")
import time, json, pickle
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

INPUT_FILE = "/mnt/user-data/uploads/rfp_training_data_complete_v3_with_transient.csv"
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "did_convert"
LEAK_EXCLUDE = {"win_streak", "win_rate_ema_7d", "win_rate_ema_30d", "win_rate_ema_90d"}
CORR_THRESHOLD = 0.85
VIF_THRESHOLD = 10.0
TOP_K = 20                 # reduced from 60 -- see module docstring
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
}
LEAKY_CONVERSION = {"actual_pickup_rate"}
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

    if "revenue_quality_score" in ev.columns and ev["revenue_quality_score"].skew() > 2.0:
        ev["revenue_quality_score"] = np.log1p(ev["revenue_quality_score"])

    if "occupancy_acceleration" in ev.columns:
        p1, p99 = ev["occupancy_acceleration"].quantile([0.01, 0.99])
        ev["occupancy_acceleration"] = ev["occupancy_acceleration"].clip(p1, p99)

    if "did_convert" not in ev.columns:
        ev["did_convert"] = ev["is_won"].astype(int) if "is_won" in ev.columns else (ev["status"] == "Booked").astype(int)

    if "total_room_nights" not in ev.columns:
        ev["total_room_nights"] = ev.get("room_block", ev.get("total_room_nights_requested", np.nan))

    return ev


def build_candidate_pool(ev):
    numeric = ev.select_dtypes(include=[np.number]).columns.tolist()
    pool = [c for c in numeric if c not in ALWAYS_EXCLUDE and c not in LEAK_EXCLUDE
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


def prune_features(ev, pool, target, is_clf, scale_pos_weight=None, tag=""):
    X_all = ev[pool].copy()
    kept_corr, corr_log = drop_correlated(X_all)
    corr_log.to_csv(OUTPUT_DIR / f"{tag}multicollinearity_correlation_pairs.csv", index=False)
    kept_corr = sorted(kept_corr)

    y_proxy = ev[target].values
    X_proxy = ev[kept_corr].copy().astype(float)
    quick = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0, tree_method="exact",
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
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


def exclude_leaky(features, leaky_set):
    ll = {x.lower() for x in leaky_set}
    return [f for f in features if f.lower() not in ll]


def prep_X(df, features):
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
    return X.astype(float)


def optuna_search_clf(X_tr, y_tr, scale_pos_weight):
    if not HAS_OPTUNA:
        return {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
                "reg_alpha": 0.5, "reg_lambda": 1.5, "gamma": 0.0,
                "scale_pos_weight": scale_pos_weight,
                "objective": "binary:logistic", "eval_metric": "logloss",
                "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1}
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 5),   # capped lower: tiny-N safeguard
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1,
        }
        return cross_val_score(xgb.XGBClassifier(**p), X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA, show_progress_bar=False)
    bp = study.best_params
    bp.update({"scale_pos_weight": scale_pos_weight, "objective": "binary:logistic",
               "eval_metric": "logloss", "random_state": RANDOM_STATE, "verbosity": 0, "n_jobs": 1})
    return bp


def train_model_clf(name, ev, features, target):
    df = ev[ev[target].notna()].copy()
    af = [f for f in features if f in df.columns]
    X = prep_X(df, af)
    y = df[target].astype(int).values
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    scale_pos_weight = n_neg / n_pos

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"  n={len(df)}  won={n_pos} lost={n_neg}  train={len(X_tr)} test={len(X_te)}  features={len(af)}")

    bp = optuna_search_clf(X_tr, y_tr, scale_pos_weight)
    model = xgb.XGBClassifier(**bp)
    model.fit(X_tr, y_tr)

    proba_tr, proba_te = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_te)[:, 1]
    pred_tr, pred_te = (proba_tr >= 0.5).astype(int), (proba_te >= 0.5).astype(int)

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_sc = cross_val_score(xgb.XGBClassifier(**bp), X, y, cv=cv, scoring="roc_auc", n_jobs=1)

    metrics = {
        "train_accuracy": float(accuracy_score(y_tr, pred_tr)), "test_accuracy": float(accuracy_score(y_te, pred_te)),
        "train_precision": float(precision_score(y_tr, pred_tr, zero_division=0)), "test_precision": float(precision_score(y_te, pred_te, zero_division=0)),
        "train_recall": float(recall_score(y_tr, pred_tr)), "test_recall": float(recall_score(y_te, pred_te)),
        "train_f1": float(f1_score(y_tr, pred_tr)), "test_f1": float(f1_score(y_te, pred_te)),
        "train_roc_auc": float(roc_auc_score(y_tr, proba_tr)), "test_roc_auc": float(roc_auc_score(y_te, proba_te)),
        "train_log_loss": float(log_loss(y_tr, proba_tr)), "test_log_loss": float(log_loss(y_te, proba_te)),
        "cv_roc_auc_mean": float(cv_sc.mean()), "cv_roc_auc_std": float(cv_sc.std()),
        "n_train": len(X_tr), "n_test": len(X_te), "scale_pos_weight": scale_pos_weight,
    }
    cm_test = confusion_matrix(y_te, pred_te)
    imp = pd.DataFrame({"feature": af, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    imp["importance_pct"] = imp["importance"] / imp["importance"].sum() * 100
    imp["rank"] = range(1, len(imp) + 1)
    return {"model": model, "features": af, "metrics": metrics, "importance": imp,
            "best_params": bp, "confusion_matrix_test": cm_test.tolist()}


def run():
    print("=" * 70)
    print("  MEETING-ONLY conversion (did_convert) — room_block == 0")
    print("=" * 70)
    ev = load_data_filtered()
    ev = engineer_features_filtered(ev)

    before = len(ev)
    ev = ev[ev["room_block"] == 0].copy().reset_index(drop=True)
    print(f"  Filtered room_block == 0: {before} -> {len(ev)} rows "
          f"({before - len(ev)} room-based rows excluded)")

    n_pos_full = int(ev[TARGET].sum())
    n_neg_full = len(ev) - n_pos_full
    scale_pos_weight = n_neg_full / n_pos_full

    pool = build_candidate_pool(ev)
    print(f"  Candidate pool: {len(pool)} features")
    base = prune_features(ev, pool, target=TARGET, is_clf=True,
                           scale_pos_weight=scale_pos_weight, tag="meeting_conversion_")
    conv_feats = exclude_leaky(base, LEAKY_CONVERSION)
    print(f"  Base pool after corr+importance+VIF pruning: {len(base)} -> final: {len(conv_feats)}")

    result = train_model_clf("Meeting-only conversion", ev, conv_feats, TARGET)
    print(f"\n  Test ROC-AUC: {result['metrics']['test_roc_auc']:.4f}   "
          f"Test accuracy: {result['metrics']['test_accuracy']:.4f}")
    print("  Top 10 features:")
    for _, r in result["importance"].head(10).iterrows():
        print(f"    {int(r['rank']):>2}. {r['feature']:<40} {r['importance_pct']:>6.2f}%")

    summary = {
        "target": TARGET, "population": "room_block == 0 (meeting-only, n=194)",
        "n_rows": len(ev), "top_k_used": TOP_K,
        "candidate_pool_size": len(pool), "base_pool_size": len(base),
        "conversion_feats_size": len(conv_feats), "conversion_feats": conv_feats,
        "metrics": result["metrics"], "best_params": result["best_params"],
        "confusion_matrix_test": result["confusion_matrix_test"],
    }
    with open(OUTPUT_DIR / "meeting_conversion_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    result["importance"].to_csv(OUTPUT_DIR / "meeting_conversion_importance.csv", index=False)

    model_bundle = {"model": result["model"], "features": result["features"], "target": TARGET,
                     "population": "room_block == 0 (n=194)", "best_params": result["best_params"],
                     "metrics": result["metrics"], "trained_at_utc": datetime.now(timezone.utc).isoformat()}
    with open(OUTPUT_DIR / "meeting_conversion_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result, summary


if __name__ == "__main__":
    run()
