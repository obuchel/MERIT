"""
Is the grade-6 "negative circulation cluster" (Section 3.3) a genuine spatial
cluster, or could a region this size and this consistent arise by chance if
the negative districts were actually scattered randomly across the map?

Section 3.3 currently supports "geographically coherent" with a map a reader
can eyeball (Figure 3, Supp. Figs S3/S12) plus the fact that the pattern
recurs across years. That is suggestive but not a formal test. This script
adds one: global Moran's I on the negative-cluster indicator, with a
permutation-based p-value computed from the ACTUAL district coordinates
(not a state-level approximation) -- the standard test in spatial statistics
for "is this pattern more clustered than random spatial assignment would
produce."

Two versions of "negative" are tested, matching the two claims already in
the manuscript text:

  1. "Consistently negative post-pandemic" -- circulation coefficient sign is
     negative in EVERY one of 2022, 2023, 2024, among districts present in
     all three years (Section 3.3's ~21% figure).
  2. "Significantly negative in 2022 alone" -- t-value < -1.96 in that single
     year (Section 3.3's ~15% figure), independent of the multi-year check.

For context/comparison, the same test also runs on the *positive*-cluster
indicator (consistently positive post-pandemic), since a reviewer's natural
question is "is the positive region actually more/less spatially clustered
than the negative one, or are they comparably real?"

Method
------
Global Moran's I with a k-nearest-neighbor spatial weight matrix (k=8 by
default -- edit K_NEIGHBORS below to check robustness to that choice), built
from real district lat/lon using the same locally-flat (planar) projection
already used elsewhere in this project (build_gwr_dataset.py, make_gwr_map.py,
compute_bandwidth_miles.py) -- accurate enough for a k-nearest-neighbor
graph at continental-US scale, and avoids adding a new dependency (no
libpysal/esda required, pure numpy).

Significance is assessed by permutation (not the classical normal
approximation, which can be unreliable for a 0/1 indicator): shuffle the
indicator across districts N_PERM times, recompute Moran's I each time, and
see how often a random shuffle produces an I at least as extreme as the one
actually observed. This directly answers "how often would random spatial
assignment produce clustering this strong" without any distributional
assumptions.

A local (LISA) version is also computed per district, flagging districts
that are themselves negative AND whose k nearest neighbors are also mostly
negative, more than random assignment would predict -- these are the
specific "hot spot" districts driving the global result, useful for mapping
or for a supplementary table. Per-district LISA p-values are NOT corrected
for multiple testing here (there are ~7,000+ of them); if you want to report
individual significant districts rather than just the map pattern, apply a
Benjamini-Hochberg FDR correction to the printed p-values first.

Requires (same directory): mgwr_cross_sectional_g6.json (and g3, if you
change GRADE below) -- the same canonical per-year MGWR output file used
throughout Section 3.5. Only numpy is required beyond the standard library.

Output:
    negative_cluster_significance_results.json  -- summary numbers
    negative_cluster_lisa_by_district_g{GRADE}.csv -- per-district LISA table
"""
import json
import numpy as np

GRADE = 6                 # Section 3.3's negative cluster is reported for grade 6
K_NEIGHBORS = 8            # try 5 and 15 too, to check the result isn't an artifact of this choice
N_PERM = 9999
SEED = 0
PRE_YEARS = [str(y) for y in range(2012, 2020)]   # 2012-2019
POST_YEARS = ['2022', '2023', '2024']
SIG_YEAR = '2022'
VARNAME = 'kidcirc_per_child'
MI_PER_DEG_LAT = 69.0


def log(msg):
    print(f"[{__name__}] {msg}", flush=True)


def project(lat, lon, lat0, lon0):
    x = (lon - lon0) * MI_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (lat - lat0) * MI_PER_DEG_LAT
    return x, y


def build_knn_weights(x, y, k):
    """Row-standardized k-NN spatial weight matrix, returned as a
    (n, k) array of neighbor indices (W is implicit: each row's k
    neighbors get weight 1/k)."""
    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, k, axis=1)[:, :k]
    return nn_idx


def morans_i(vals, nn_idx):
    """Global Moran's I for a row-standardized k-NN weight matrix, given
    as a (n, k) neighbor-index array (implicit weight 1/k per neighbor).
    The classic formula is I = (n/S0) * sum_ij(w_ij dx_i dx_j) / sum_i(dx_i^2);
    for a row-standardized matrix S0 = n (every row sums to 1), so the n/S0
    factor is exactly 1 and drops out below.
    """
    xbar = vals.mean()
    dx = vals - xbar
    neighbor_mean_dx = dx[nn_idx].mean(axis=1)   # (1/k) * sum_j w_ij * dx_j, since w_ij = 1/k
    numerator = (dx * neighbor_mean_dx).sum()
    denominator = (dx ** 2).sum()
    return numerator / denominator if denominator > 0 else np.nan


def permutation_test(vals, nn_idx, n_perm, seed, tail='greater'):
    obs = morans_i(vals, nn_idx)
    rng = np.random.default_rng(seed)
    v = vals.copy()
    count = 0
    for _ in range(n_perm):
        rng.shuffle(v)
        I_perm = morans_i(v, nn_idx)
        if tail == 'greater':
            if I_perm >= obs:
                count += 1
        else:
            if I_perm <= obs:
                count += 1
    p = (count + 1) / (n_perm + 1)
    return obs, p


def local_morans_i(vals, nn_idx):
    """Local Moran's I_i for every district, plus a simple quadrant label
    (HH/LL = clustered with similar neighbors, HL/LH = spatial outlier),
    and a permutation p-value per district (conditional permutation: keep
    district i fixed, shuffle everyone else, same n_perm as the global
    test but done once per district would be extremely slow at n~8000 --
    instead this uses the standard analytic z-score approximation for LISA,
    which is the conventional shortcut here; treat p-values as approximate
    screening, not as strict per-district significance without further
    correction."""
    n = len(vals)
    xbar = vals.mean()
    var = vals.var()
    dx = vals - xbar
    neighbor_mean_dx = dx[nn_idx].mean(axis=1)
    Ii = (dx / var) * neighbor_mean_dx * (n - 1) / n if var > 0 else np.zeros(n)
    z = dx / np.sqrt(var) if var > 0 else np.zeros(n)
    neighbor_z_mean = z[nn_idx].mean(axis=1)
    quadrant = np.full(n, 'NS', dtype=object)
    quadrant[(z > 0) & (neighbor_z_mean > 0)] = 'HH'
    quadrant[(z < 0) & (neighbor_z_mean < 0)] = 'LL'
    quadrant[(z > 0) & (neighbor_z_mean < 0)] = 'HL'
    quadrant[(z < 0) & (neighbor_z_mean > 0)] = 'LH'
    # approximate two-sided p-value from Ii's z-score under randomization variance
    k = nn_idx.shape[1]
    EI = -1.0 / (n - 1)
    b2 = (((vals - xbar) ** 4).mean()) / (var ** 2) if var > 0 else 0
    VarIi = ((k / (n - 1) ** 2) * ((n - b2) / (n - 2)) if n > 2 else np.nan)  # rough approximation
    z_score = (Ii - EI) / np.sqrt(VarIi) if VarIi and VarIi > 0 else np.zeros(n)
    from math import erf, sqrt
    p_approx = np.array([2 * (1 - 0.5 * (1 + erf(abs(zz) / sqrt(2)))) for zz in z_score])
    return Ii, quadrant, p_approx


def main():
    path = f'mgwr_cross_sectional_g{GRADE}.json'
    log(f"loading {path} ...")
    data = json.load(open(path))
    names = data['names']
    vi = names.index(VARNAME)

    # --- build per-district year -> (coef, tval) lookup ---
    by_geoid = {}
    latlon = {}
    for yr, w in data['years'].items():
        for gid, p, t, lat, lon in zip(w['geoid'], w['params'], w['tvalues'], w['lat'], w['lon']):
            by_geoid.setdefault(gid, {})[yr] = (p[vi], t[vi])
            latlon[gid] = (lat, lon)

    # --- claim 1: consistently negative (or positive) across 2022-2024 ---
    complete_post = {g: d for g, d in by_geoid.items() if all(y in d for y in POST_YEARS)}
    geoids = sorted(complete_post.keys())
    lat = np.array([latlon[g][0] for g in geoids])
    lon = np.array([latlon[g][1] for g in geoids])
    lat0, lon0 = float(lat.mean()), float(lon.mean())
    x, y = project(lat, lon, lat0, lon0)

    signs = np.array([[np.sign(complete_post[g][yr][0]) for yr in POST_YEARS] for g in geoids])
    consistently_negative = (signs == -1).all(axis=1).astype(float)
    consistently_positive = (signs == 1).all(axis=1).astype(float)
    mean_sign_post = signs.mean(axis=1)   # continuous version, -1..+1

    pct_neg = 100 * consistently_negative.mean()
    pct_pos = 100 * consistently_positive.mean()
    log(f"grade {GRADE}, n={len(geoids)} districts present in all of {POST_YEARS}: "
        f"consistently NEGATIVE {pct_neg:.1f}% | consistently POSITIVE {pct_pos:.1f}% "
        f"(sanity check against the manuscript's stated ~21% / ~[positive figure])")

    log(f"building k={K_NEIGHBORS} nearest-neighbor spatial weight matrix over {len(geoids)} districts ...")
    nn_idx = build_knn_weights(x, y, K_NEIGHBORS)

    log("running permutation test: consistently-NEGATIVE indicator ...")
    I_neg, p_neg = permutation_test(consistently_negative, nn_idx, N_PERM, SEED)
    log(f"  Moran's I = {I_neg:.4f}, permutation p = {p_neg:.5f} ({N_PERM} shuffles)")

    log("running permutation test: consistently-POSITIVE indicator (for comparison) ...")
    I_pos, p_pos = permutation_test(consistently_positive, nn_idx, N_PERM, SEED)
    log(f"  Moran's I = {I_pos:.4f}, permutation p = {p_pos:.5f} ({N_PERM} shuffles)")

    log("running permutation test: continuous post-pandemic mean-sign score ...")
    I_cont, p_cont = permutation_test(mean_sign_post, nn_idx, N_PERM, SEED)
    log(f"  Moran's I = {I_cont:.4f}, permutation p = {p_cont:.5f} ({N_PERM} shuffles)")

    # --- claim 2: significantly negative in 2022 alone (t < -1.96) ---
    yr_data = data['years'][SIG_YEAR]
    geoids_yr = yr_data['geoid']
    tvals_yr = np.array([t[vi] for t in yr_data['tvalues']])
    lat_yr = np.array(yr_data['lat']); lon_yr = np.array(yr_data['lon'])
    lat0y, lon0y = float(lat_yr.mean()), float(lon_yr.mean())
    xy, yy = project(lat_yr, lon_yr, lat0y, lon0y)
    sig_negative = (tvals_yr < -1.96).astype(float)
    pct_sig_neg = 100 * sig_negative.mean()
    log(f"grade {GRADE}, {SIG_YEAR} alone, n={len(geoids_yr)}: significantly NEGATIVE "
        f"(t<-1.96) {pct_sig_neg:.1f}% (sanity check against the manuscript's stated ~15%)")

    nn_idx_yr = build_knn_weights(xy, yy, K_NEIGHBORS)
    log(f"running permutation test: {SIG_YEAR} significantly-negative indicator ...")
    I_sig, p_sig = permutation_test(sig_negative, nn_idx_yr, N_PERM, SEED)
    log(f"  Moran's I = {I_sig:.4f}, permutation p = {p_sig:.5f} ({N_PERM} shuffles)")

    # --- LISA (local) on the post-pandemic continuous score, for mapping ---
    log("computing local Moran's I (LISA) per district on the continuous post-pandemic score ...")
    Ii, quadrant, p_approx = local_morans_i(mean_sign_post, nn_idx)

    out = {
        'grade': GRADE, 'k_neighbors': K_NEIGHBORS, 'n_perm': N_PERM,
        'post_years': POST_YEARS, 'sig_year': SIG_YEAR,
        'n_districts_post_complete': len(geoids),
        'pct_consistently_negative_post': pct_neg,
        'pct_consistently_positive_post': pct_pos,
        'moran_I_consistently_negative': I_neg, 'p_consistently_negative': p_neg,
        'moran_I_consistently_positive': I_pos, 'p_consistently_positive': p_pos,
        'moran_I_continuous_post_score': I_cont, 'p_continuous_post_score': p_cont,
        'n_districts_sig_year': len(geoids_yr),
        'pct_significantly_negative_sig_year': pct_sig_neg,
        'moran_I_sig_year_negative': I_sig, 'p_sig_year_negative': p_sig,
    }
    json.dump(out, open('negative_cluster_significance_results.json', 'w'), indent=2)
    log("saved negative_cluster_significance_results.json")

    with open(f'negative_cluster_lisa_by_district_g{GRADE}.csv', 'w') as f:
        f.write('geoid,lat,lon,mean_sign_post,local_moran_I,quadrant,p_approx\n')
        for i, g in enumerate(geoids):
            f.write(f'{g},{lat[i]:.5f},{lon[i]:.5f},{mean_sign_post[i]:.3f},'
                    f'{Ii[i]:.4f},{quadrant[i]},{p_approx[i]:.4f}\n')
    log(f"saved negative_cluster_lisa_by_district_g{GRADE}.csv ({len(geoids)} rows)")
    log("DONE. Interpretation: a small p-value (e.g. < 0.01) on the "
        "consistently-negative indicator means a cluster this spatially "
        "concentrated would be very unlikely under random spatial "
        "assignment -- i.e. it supports 'geographically coherent' as a "
        "formal claim, not just a visual impression from the map.")


if __name__ == '__main__':
    main()
