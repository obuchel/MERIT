"""
fit_mgwr_yourself.py

One script, run locally, that goes straight from your four raw files to the
final MGWR district-temporal fit for both grades. No separate pickle-build
step, so there is nothing that can drift out of sync with itself the way
the pickle and the old per-year JSON cache did last time.

Required input files (same folder as this script):
    district_year_library_gis_v6.csv
    seda_filtered_2025.csv
    poverty_income_panel.csv
    district_centroids_full.csv
    child_population_panel.csv (GEOID, checkpoint_year, child_population --
        built by build_child_population_panel.py from
        pull_census_income_poverty_children_by_year.py's per-year output)

Usage:
    python3 fit_mgwr_yourself.py       # both grades, 3 then 6
    python3 fit_mgwr_yourself.py 3     # just grade 3
    python3 fit_mgwr_yourself.py 6     # just grade 6

Before fitting a grade, this DELETES any existing
mgwr_district_temporal_g{grade}_*.json/.geojson and the combined
mgwr_district_temporal_g{grade}.json/.geojson for that grade, then writes
them fresh. The original run_mgwr_district_temporal_v6.py skips a year
whenever its output file already exists -- that is exactly what let stale
files from an old pickle silently survive a rebuild last time. This version
never skips, so every run is a clean, honest fit against today's four
input files.

Same expanding-window definition throughout the project: 11 checkpoints
(2012-2019, 2022-2024; 2009-2011 mathematically impossible -- see below;
2020/2021 skipped -- COVID disrupted SEDA testing nationally), each
district's trend computed as an OLS slope over every year through that
checkpoint (minimum 4 points), same MGWR algorithm as
run_mgwr_district_temporal_v6.py: Sel_BW search on the final (2024) window,
then every earlier window refit with bandwidths rescaled to the same
fraction of that window's size.

Covariates are now kids-population-normalized rather than total-population-
normalized: kidcirc_per_child_slope and prog_per_child_slope replace the
earlier circ_pc_slope/prog_pc_slope (which were both / total district
population, all ages, via the pre-computed circ_pc/prog_pc columns).

Per your call to recalculate the denominator for each checkpoint rather
than reuse one static snapshot (matching how poverty/income already work):
this computes the RAW, unnormalized OLS slope of kidcirc and kidatten over
year through each checkpoint, then divides that slope by THAT CHECKPOINT'S
OWN child_population value (from child_population_panel.csv, one ACS-vintage
figure per checkpoint, same B09001_001E "population under 18" variable as
the pull script). That's mathematically the same as dividing kidcirc/kidatten
by a constant before taking the slope (a fixed denominator scales a slope
linearly), so each checkpoint's trend is deflated by its own contemporary
child population rather than a single borrowed-from-elsewhere figure.
"""
import glob
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR

CHECKPOINTS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]
# 2009/2010/2011 were requested but are mathematically impossible: every
# checkpoint's circ/prog/reading trend needs >=4 years of data through the
# cutoff, and the library/SEDA series itself only starts in 2009 -- so
# 2012 (exactly 4 years: 2009-2012) is the earliest checkpoint that can
# produce a slope at all. Requires poverty_income_panel.csv to have rows
# for every year in this list -- 2012/2014/2016/2018 aren't in it yet as
# of this writing (see build_poverty_panel.py).
MIN_PTS = 4
NAMES = ['const', 'kidcirc_per_child_slope', 'prog_per_child_slope', 'poverty', 'income']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------
# Step A: build the expanding-window tables from the 4 raw files
# (same vectorized OLS-slope logic as build_expanding_window_v7.py)
# ---------------------------------------------------------------------

def slopes_through(df, id_col, year_col, val_col, cutoff_year, min_pts=MIN_PTS):
    sub = df.loc[(df[year_col] <= cutoff_year) & df[val_col].notna(), [id_col, year_col, val_col]].copy()
    sub['xy'] = sub[year_col] * sub[val_col]
    sub['xx'] = sub[year_col].astype(float) ** 2
    agg = sub.groupby(id_col).agg(n=(year_col, 'count'), sx=(year_col, 'sum'),
                                   sy=(val_col, 'sum'), sxy=('xy', 'sum'), sxx=('xx', 'sum'))
    denom = agg['n'] * agg['sxx'] - agg['sx'] ** 2
    slope = (agg['n'] * agg['sxy'] - agg['sx'] * agg['sy']) / denom
    slope = slope.where((agg['n'] >= min_pts) & (denom != 0))
    return slope.rename('slope').reset_index()


def build_windows():
    lib = pd.read_csv('district_year_library_gis_v6.csv', dtype={'district_geoid': str})
    lib['district_geoid'] = lib['district_geoid'].astype(str).str.zfill(7)

    seda = pd.read_csv('seda_filtered_2025.csv', dtype={'sedaadmin': str})
    seda['district_geoid'] = seda['sedaadmin'].str.zfill(7)

    poverty_panel = pd.read_csv('poverty_income_panel.csv', dtype={'GEOID': str})
    poverty_panel['GEOID'] = poverty_panel['GEOID'].astype(str).str.zfill(7)
    poverty_panel = poverty_panel.rename(columns={
        'GEOID': 'district_geoid', 'median_income': 'income', 'child_poverty_pct': 'poverty',
    })[['district_geoid', 'checkpoint_year', 'poverty', 'income']]

    child_pop_panel = pd.read_csv('child_population_panel.csv', dtype={'GEOID': str})
    child_pop_panel['GEOID'] = child_pop_panel['GEOID'].astype(str).str.zfill(7)
    child_pop_panel = child_pop_panel.rename(columns={'GEOID': 'district_geoid'})[
        ['district_geoid', 'checkpoint_year', 'child_population']]

    centroids = pd.read_csv('district_centroids_full.csv', dtype={'GEOID': str})
    centroids['GEOID'] = centroids['GEOID'].astype(str).str.zfill(7)
    centroids = centroids.rename(columns={'GEOID': 'district_geoid'})[['district_geoid', 'lat', 'lon']]

    # raw (unnormalized) OLS slopes of kidcirc/kidatten over year -- the
    # per-child normalization happens AFTER, dividing by that checkpoint's
    # own child_population (a constant denominator scales a slope linearly,
    # so this is equivalent to normalizing every year first, but only needs
    # one child-population figure per checkpoint instead of one per year).
    kidcirc_raw_by_year, prog_raw_by_year = {}, {}
    for year in CHECKPOINTS:
        kidcirc_raw_by_year[year] = slopes_through(lib, 'district_geoid', 'year', 'kidcirc', year).rename(columns={'slope': 'kidcirc_slope_raw'})
        prog_raw_by_year[year] = slopes_through(lib, 'district_geoid', 'year', 'kidatten', year).rename(columns={'slope': 'kidatten_slope_raw'})

    out = {}
    for grade in [3, 6]:
        seda_g = seda[seda['grade'] == grade][['district_geoid', 'year', 'gcs_mn_all']]
        windows = []
        for year in CHECKPOINTS:
            reading = slopes_through(seda_g, 'district_geoid', 'year', 'gcs_mn_all', year).rename(columns={'slope': 'reading_slope'})
            pov = poverty_panel[poverty_panel['checkpoint_year'] == year][['district_geoid', 'poverty', 'income']]
            child_pop = child_pop_panel[child_pop_panel['checkpoint_year'] == year][['district_geoid', 'child_population']]

            df = kidcirc_raw_by_year[year].merge(prog_raw_by_year[year], on='district_geoid').merge(reading, on='district_geoid')
            df = df.merge(pov, on='district_geoid', how='inner')
            df = df.merge(child_pop, on='district_geoid', how='inner')
            df = df.merge(centroids, on='district_geoid', how='inner')
            df['kidcirc_per_child_slope'] = df['kidcirc_slope_raw'] / df['child_population']
            df['prog_per_child_slope'] = df['kidatten_slope_raw'] / df['child_population']
            df = df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=['kidcirc_per_child_slope', 'prog_per_child_slope', 'reading_slope', 'poverty', 'income', 'lat', 'lon'])
            windows.append((year, df))
            log(f"grade {grade} window {year}: {len(df)} districts with full data")
        out[grade] = windows
    return out


# ---------------------------------------------------------------------
# Step B: the MGWR fit -- identical algorithm to run_mgwr_district_temporal_v6.py
# ---------------------------------------------------------------------

def build_xyz(df):
    geoid = df['district_geoid'].values
    lat = df['lat'].values.astype(float)
    lon = df['lon'].values.astype(float)
    y = df['reading_slope'].values.reshape(-1, 1).astype(float)
    Xraw = df[['kidcirc_per_child_slope', 'prog_per_child_slope', 'poverty', 'income']].values.astype(float)
    Xs = (Xraw - Xraw.mean(axis=0)) / Xraw.std(axis=0)
    LAT0, LON0 = 38.5, -96.0
    x_km = (lon - LON0) * 111.32 * np.cos(np.radians(LAT0))
    y_km = (lat - LAT0) * 110.54
    coords = np.column_stack([x_km, y_km])
    return geoid, coords, y, Xs


def bw_for_window(fixed_bws, final_n, nw):
    out = []
    for b in fixed_bws:
        frac = b / final_n
        scaled = round(frac * nw)
        scaled = max(40, min(nw - 1, scaled))
        out.append(float(scaled))
    return out


def write_year_files(grade, year, names, fixed_bws, final_year, final_n, geoid, lat, lon, params, tvalues, n, bws_used=None):
    rec = {"grade": grade, "names": names, "fixed_bws": fixed_bws, "final_year": final_year,
           "final_n": final_n, "year": year, "geoid": list(geoid), "lat": list(lat), "lon": list(lon),
           "params": params, "tvalues": tvalues, "n": n}
    if bws_used is not None:
        rec["bws_used"] = bws_used
    json.dump(rec, open(f'mgwr_district_temporal_g{grade}_{year}.json', 'w'))

    features = []
    for i in range(n):
        props = {'district_geoid': rec['geoid'][i], 'year': year, 'n_window': n}
        for j, name in enumerate(names):
            props[f'coef_{name}'] = params[i][j]
            props[f'tval_{name}'] = tvalues[i][j]
            props[f'sig_{name}'] = bool(abs(tvalues[i][j]) > 1.96)
        features.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [rec['lon'][i], rec['lat'][i]]},
                          "properties": props})
    geojson = {"type": "FeatureCollection", "features": features,
               "metadata": {"grade": grade, "names": names, "fixed_bws": dict(zip(names, fixed_bws)),
                            "final_year": final_year, "year": year}}
    json.dump(geojson, open(f'mgwr_district_temporal_g{grade}_{year}.geojson', 'w'))
    log(f"saved mgwr_district_temporal_g{grade}_{year}.json/.geojson ({n} districts)")


def rebuild_combined(grade, years_present):
    windows = {}
    names = fixed_bws = final_year = final_n = None
    all_features = []
    for year in years_present:
        p = f'mgwr_district_temporal_g{grade}_{year}.json'
        if not os.path.exists(p):
            continue
        rec = json.load(open(p))
        names, fixed_bws, final_year, final_n = rec['names'], rec['fixed_bws'], rec['final_year'], rec['final_n']
        windows[str(year)] = {"geoid": rec['geoid'], "lat": rec['lat'], "lon": rec['lon'],
                               "params": rec['params'], "tvalues": rec['tvalues'], "n": rec['n']}
        gj = json.load(open(f'mgwr_district_temporal_g{grade}_{year}.geojson'))
        all_features.extend(gj['features'])
    if not windows:
        return
    json.dump({"grade": grade, "names": names, "fixed_bws": fixed_bws, "final_year": final_year,
               "final_n": final_n, "windows": windows}, open(f'mgwr_district_temporal_g{grade}.json', 'w'))
    json.dump({"type": "FeatureCollection", "features": all_features,
               "metadata": {"grade": grade, "names": names, "fixed_bws": dict(zip(names, fixed_bws)),
                            "final_year": final_year}}, open(f'mgwr_district_temporal_g{grade}.geojson', 'w'))
    log(f"rebuilt combined mgwr_district_temporal_g{grade}.json/.geojson ({len(windows)} of {len(years_present)} years)")


def clear_old_outputs(grade):
    patterns = [f'mgwr_district_temporal_g{grade}_*.json', f'mgwr_district_temporal_g{grade}_*.geojson',
                f'mgwr_district_temporal_g{grade}.json', f'mgwr_district_temporal_g{grade}.geojson']
    removed = 0
    for pat in patterns:
        for f in glob.glob(pat):
            os.remove(f)
            removed += 1
    if removed:
        log(f"grade {grade}: removed {removed} old output file(s) so this run starts clean")


MIN_WINDOW_N = 80  # twice the minimum bandwidth (40) used in the search bounds below


def fit_grade(grade, windows_data):
    clear_old_outputs(grade)
    skipped = [(y, len(df)) for y, df in windows_data if len(df) < MIN_WINDOW_N]
    windows_data = [(y, df) for y, df in windows_data if len(df) >= MIN_WINDOW_N]
    if skipped:
        log(f"grade {grade}: skipping {len(skipped)} window(s) with too few districts to fit "
            f"(need >={MIN_WINDOW_N}): {skipped} -- almost always means a source panel "
            f"(poverty_income_panel.csv or child_population_panel.csv) doesn't have that "
            f"checkpoint year yet, not a real absence of data")
    if not windows_data:
        log(f"grade {grade}: no window has enough districts to fit anything -- stopping")
        return
    all_years = [w[0] for w in windows_data]
    log(f"grade {grade}: windows = {all_years}")
    final_year, final_df = windows_data[-1]

    geoid, coords, y, Xs = build_xyz(final_df)
    final_n = len(y)
    log(f"final window = {final_year}, n = {final_n}. Starting MGWR bandwidth search...")
    t0 = time.time()
    sel = Sel_BW(coords, y, Xs, multi=True, constant=True, n_jobs=1)
    bws = sel.search(multi_bw_min=[40], multi_bw_max=[final_n], verbose=True)
    log(f"bandwidth search done in {time.time()-t0:.1f}s. bws = {dict(zip(NAMES, bws))}")

    t0 = time.time()
    model = MGWR(coords, y, Xs, sel, constant=True, n_jobs=1)
    final_results = model.fit()
    log(f"final-window fit done in {time.time()-t0:.1f}s. R2={final_results.R2:.4f} AICc={final_results.aicc:.2f}")

    fixed_bws = [float(b) for b in bws]
    write_year_files(grade, final_year, NAMES, fixed_bws, final_year, final_n,
                      geoid, final_df['lat'].astype(float).values, final_df['lon'].astype(float).values,
                      final_results.params.tolist(), final_results.tvalues.tolist(), final_n)
    rebuild_combined(grade, all_years)

    for year, df in windows_data[:-1]:
        geoid_w, coords_w, y_w, Xs_w = build_xyz(df)
        nw = len(y_w)
        bws_w_fixed = bw_for_window(fixed_bws, final_n, nw)
        log(f"window {year}: n = {nw}. Refitting with rescaled bandwidths {dict(zip(NAMES, bws_w_fixed))}...")
        t0 = time.time()
        sel_w = Sel_BW(coords_w, y_w, Xs_w, multi=True, constant=True, n_jobs=1)
        sel_w.search(multi_bw_min=bws_w_fixed, multi_bw_max=bws_w_fixed)
        model_w = MGWR(coords_w, y_w, Xs_w, sel_w, constant=True, n_jobs=1)
        results_w = model_w.fit()
        log(f"window {year} done in {time.time()-t0:.1f}s. R2={results_w.R2:.4f} AICc={results_w.aicc:.2f}")
        write_year_files(grade, year, NAMES, fixed_bws, final_year, final_n,
                          geoid_w, df['lat'].astype(float).values, df['lon'].astype(float).values,
                          results_w.params.tolist(), results_w.tvalues.tolist(), nw, bws_used=bws_w_fixed)
        rebuild_combined(grade, all_years)

    log(f"grade {grade}: ALL DONE")


if __name__ == '__main__':
    grades = [int(sys.argv[1])] if len(sys.argv) > 1 else [3, 6]
    log("building expanding-window tables from your 4 CSV files...")
    data = build_windows()
    for grade in grades:
        fit_grade(grade, data[grade])
