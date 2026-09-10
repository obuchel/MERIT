"""
Follow-up to test_negative_cluster_significance.py.

That script found Moran's I of 0.97-0.99 (p=0.0001) for circulation's
negative cluster AND its positive cluster AND a totally unrelated
single-year check -- suspiciously extreme across the board. The likely
reason: MGWR estimates each district's coefficient from a geographically-
weighted average of its neighbors (the same bandwidth machinery Section 3.5
describes), so neighboring districts' coefficients are not independent
draws to begin with -- the estimator smooths them together before a
permutation test ever sees the data. A random-label permutation test
implicitly assumes independence, so it may just be detecting "this is
MGWR output" rather than "this is a genuine regional effect."

The cheap way to check without re-fitting MGWR (which is slow -- see the
timing warning in refit_2012_2015_no_alaska.py) is a benchmark already
sitting in the same file: program attendance (prog_per_child). Section 3.1
already established program attendance has NO reliable national pattern --
its sign flips year to year with no coherent region. It went through the
exact same MGWR fitting procedure as circulation. So:

  - If program attendance's negative/positive classification ALSO comes
    back at Moran's I ~ 0.97-0.99, that proves the test cannot distinguish
    a real regional effect from mechanical MGWR smoothing, and the
    circulation result should NOT be reported as evidence of anything.
  - If program attendance comes back substantially lower, that is a real,
    cheap, credible benchmark: it shows circulation's spatial coherence is
    distinctive relative to a variable known to have no coherent pattern,
    run through the identical estimator -- and THAT comparison (not the
    raw circulation p-value alone) is the number worth reporting.

Same method as before: k-nearest-neighbor graph (k=8) on real district
lat/lon, permutation-based Moran's I (9999 shuffles), only numpy required.

Requires (same directory): mgwr_cross_sectional_g6.json
Output: negative_cluster_benchmark_results.json
"""
import json
import numpy as np

GRADE = 6
K_NEIGHBORS = 8
N_PERM = 9999
SEED = 0
POST_YEARS = ['2022', '2023', '2024']
SIG_YEAR = '2022'
VARS = [('kidcirc_per_child', 'Circulation'), ('prog_per_child', 'Programs')]
MI_PER_DEG_LAT = 69.0


def log(msg):
    print(f"[{__name__}] {msg}", flush=True)


def project(lat, lon, lat0, lon0):
    x = (lon - lon0) * MI_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (lat - lat0) * MI_PER_DEG_LAT
    return x, y


def build_knn_weights(x, y, k):
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    return np.argpartition(d2, k, axis=1)[:, :k]


def morans_i(vals, nn_idx):
    xbar = vals.mean()
    dx = vals - xbar
    neighbor_mean_dx = dx[nn_idx].mean(axis=1)
    denom = (dx ** 2).sum()
    return (dx * neighbor_mean_dx).sum() / denom if denom > 0 else np.nan


def permutation_test(vals, nn_idx, n_perm, seed):
    obs = morans_i(vals, nn_idx)
    rng = np.random.default_rng(seed)
    v = vals.copy()
    count = 0
    for _ in range(n_perm):
        rng.shuffle(v)
        if morans_i(v, nn_idx) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    path = f'mgwr_cross_sectional_g{GRADE}.json'
    log(f"loading {path} ...")
    data = json.load(open(path))
    names = data['names']

    # --- district set present in all of 2022-2024 (same for every variable) ---
    by_geoid = {}
    latlon = {}
    for yr in POST_YEARS:
        w = data['years'][yr]
        for gid, p, t, lat, lon in zip(w['geoid'], w['params'], w['tvalues'], w['lat'], w['lon']):
            by_geoid.setdefault(gid, {})[yr] = (p, t)
            latlon[gid] = (lat, lon)
    geoids = sorted(g for g, d in by_geoid.items() if all(y in d for y in POST_YEARS))
    lat = np.array([latlon[g][0] for g in geoids])
    lon = np.array([latlon[g][1] for g in geoids])
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    x, y = project(lat, lon, lat0, lon0)
    log(f"n={len(geoids)} districts present in all of {POST_YEARS}")

    log(f"building k={K_NEIGHBORS} nearest-neighbor spatial weight matrix (shared across variables) ...")
    nn_idx = build_knn_weights(x, y, K_NEIGHBORS)

    # --- 2022-alone district set (same for every variable) ---
    yr_data = data['years'][SIG_YEAR]
    geoids_yr = yr_data['geoid']
    lat_yr = np.array(yr_data['lat']); lon_yr = np.array(yr_data['lon'])
    lat0y, lon0y = float(lat_yr.mean()), float(lon_yr.mean())
    xy, yy = project(lat_yr, lon_yr, lat0y, lon0y)
    nn_idx_yr = build_knn_weights(xy, yy, K_NEIGHBORS)

    results = {}
    for varname, label in VARS:
        vi = names.index(varname)
        log(f"--- {label} ({varname}) ---")

        signs = np.array([[np.sign(by_geoid[g][yr][0][vi]) for yr in POST_YEARS] for g in geoids])
        consistently_negative = (signs == -1).all(axis=1).astype(float)
        consistently_positive = (signs == 1).all(axis=1).astype(float)
        mean_sign_post = signs.mean(axis=1)

        pct_neg = 100 * consistently_negative.mean()
        pct_pos = 100 * consistently_positive.mean()
        log(f"  consistently NEGATIVE {pct_neg:.1f}% | consistently POSITIVE {pct_pos:.1f}%")

        I_neg, p_neg = permutation_test(consistently_negative, nn_idx, N_PERM, SEED)
        log(f"  Moran's I (consistently negative) = {I_neg:.4f}, p = {p_neg:.5f}")

        I_pos, p_pos = permutation_test(consistently_positive, nn_idx, N_PERM, SEED)
        log(f"  Moran's I (consistently positive) = {I_pos:.4f}, p = {p_pos:.5f}")

        I_cont, p_cont = permutation_test(mean_sign_post, nn_idx, N_PERM, SEED)
        log(f"  Moran's I (continuous mean-sign score) = {I_cont:.4f}, p = {p_cont:.5f}")

        tvals_yr = np.array([t[vi] for t in yr_data['tvalues']])
        sig_negative = (tvals_yr < -1.96).astype(float)
        pct_sig_neg = 100 * sig_negative.mean()
        I_sig, p_sig = permutation_test(sig_negative, nn_idx_yr, N_PERM, SEED)
        log(f"  {SIG_YEAR} significantly negative: {pct_sig_neg:.1f}% | Moran's I = {I_sig:.4f}, p = {p_sig:.5f}")

        results[varname] = {
            'label': label,
            'pct_consistently_negative': pct_neg, 'moran_I_consistently_negative': I_neg, 'p_consistently_negative': p_neg,
            'pct_consistently_positive': pct_pos, 'moran_I_consistently_positive': I_pos, 'p_consistently_positive': p_pos,
            'moran_I_continuous': I_cont, 'p_continuous': p_cont,
            'pct_sig_year_negative': pct_sig_neg, 'moran_I_sig_year_negative': I_sig, 'p_sig_year_negative': p_sig,
        }

    json.dump({'grade': GRADE, 'k_neighbors': K_NEIGHBORS, 'n_perm': N_PERM,
               'n_districts_post_complete': len(geoids), 'results': results},
              open('negative_cluster_benchmark_results.json', 'w'), indent=2)
    log("saved negative_cluster_benchmark_results.json")

    log("")
    log("=== SIDE-BY-SIDE: circulation vs. programs (same districts, same k-NN graph, same test) ===")
    c = results['kidcirc_per_child']; p_ = results['prog_per_child']
    log(f"  Moran's I, consistently-negative cluster:  circulation {c['moran_I_consistently_negative']:.4f}  vs.  programs {p_['moran_I_consistently_negative']:.4f}")
    log(f"  Moran's I, continuous post-pandemic score:  circulation {c['moran_I_continuous']:.4f}  vs.  programs {p_['moran_I_continuous']:.4f}")
    log("If these two numbers are close together, the Moran's I test is picking up MGWR's own spatial")
    log("smoothing rather than a real difference between the two variables, and should not be reported")
    log("as evidence for Section 3.3's negative cluster. If circulation is noticeably higher than")
    log("programs despite going through the identical estimator, that gap is the credible finding.")


if __name__ == '__main__':
    main()
