"""
fit_mgwr_cross_sectional.py

A cross-sectional companion to fit_mgwr_yourself.py -- same input files,
same grades, same 11 checkpoint years, but a genuinely different question.

fit_mgwr_yourself.py asks a TREND question: does a district's own CHANGE
in per-child library use over time associate with that same district's
own CHANGE in reading scores over time? Every covariate there is an OLS
slope through a checkpoint year, and the whole design exists specifically
to cancel out anything about a place that doesn't change much year to
year (its wealth, its demographics) -- see Section 2.4/3.4 of the paper.

This script asks the plainer, LEVEL question instead: in one given year,
across different districts, does having more circulation and program
attendance per child associate with a HIGHER reading score that same
year, after controlling for that year's child poverty rate and income --
and does that association vary by where a district is? This is the same
kind of question Sections 3.1-3.2 and 3.6 already ask at the county
level, repeated here at the district level, independently, at each of
11 points in time, so you can see whether the cross-sectional pattern
holds steady or drifts.

Each of the 11 checkpoints is a FULLY INDEPENDENT MGWR fit: its own real
Sel_BW(multi=True).search() on that year's own districts, with its own
five AICc-optimal bandwidths. There is no rescaled-bandwidth trick
carried over from one year to the next, because unlike the trend design
there is no single largest window whose bandwidths substantively apply
to the others -- every year here is its own, equally legitimate,
equally-sized cross-section.

That independence has a real runtime cost worth knowing before you
start this running: fit_mgwr_yourself.py pays for exactly ONE real
bandwidth search (on its largest window) and reuses it, rescaled,
everywhere else. This script pays for a REAL bandwidth search on EVERY
ONE of the 11 checkpoints, each against ~9,000-9,900 districts and 5
covariates. Expect the total wall-clock time to be several times
fit_mgwr_yourself.py's, even though any one year's fit is conceptually
simpler.

It's also worth being explicit about what this design can and can't
tell you, since it's easy to blur the two scripts together. A
cross-sectional result carries the same confounding risk the paper
flags in Sections 3.1 and 4.3: comparing different places to each
other in the same year lets any fixed trait that drives both better
library funding and better reading scores (community wealth, for
instance) masquerade as a library effect, in a way the trend design is
specifically built to screen out. Running both and comparing what
survives in each is the actual point -- this script is not a
replacement for fit_mgwr_yourself.py, it's a second, independent lens
on the same 11 years of data.

Required input files (same folder as this script -- identical list to
fit_mgwr_yourself.py):
    district_year_library_gis_v6.csv
    seda_filtered_2025.csv
    child_demographics_panel.csv (GEOID, checkpoint_year, poverty, income,
        child_population -- built by build_child_demographics_panel.py;
        replaces the earlier poverty_income_panel.csv + child_population_panel.csv
        split)
    district_centroids_full.csv

Usage:
    python3 fit_mgwr_cross_sectional.py       # both grades, 3 then 6
    python3 fit_mgwr_cross_sectional.py 3     # just grade 3
    python3 fit_mgwr_cross_sectional.py 6     # just grade 6

Before fitting a grade, this DELETES any existing
mgwr_cross_sectional_g{grade}_*.json/.geojson and the combined
mgwr_cross_sectional_g{grade}.json/.geojson for that grade, then writes
them fresh -- same "never skip, always rebuild clean" policy as
fit_mgwr_yourself.py, and a different filename prefix so the two
designs' output files can never collide or overwrite each other.
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
# Same 11 years as fit_mgwr_yourself.py, for the same reason: these are
# exactly the years poverty_income_panel.csv and child_population_panel.csv
# have Census data for. Unlike the trend design, this script does NOT need
# >=4 years of history to use a checkpoint (a level needs only that one
# year's data), so this list is a deliberate choice to keep the two
# designs directly comparable -- not a constraint this script itself
# imposes. Feel free to add other years SEDA/library data covers (e.g.
# any year 2009-2019 or 2022-2024) if you pull matching poverty/income/
# child-population files for it first.
NAMES = ['const', 'kidcirc_per_child', 'prog_per_child', 'poverty', 'income']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------
# Step A: build one cross-sectional table per checkpoint year -- LEVELS,
# not slopes. Every column below is that single year's own value.
# ---------------------------------------------------------------------

def build_cross_sectional_years():
    lib = pd.read_csv('district_year_library_gis_v6.csv', dtype={'district_geoid': str})
    lib['district_geoid'] = lib['district_geoid'].astype(str).str.zfill(7)

    seda = pd.read_csv('seda_filtered_2025.csv', dtype={'sedaadmin': str})
    seda['district_geoid'] = seda['sedaadmin'].str.zfill(7)

    demo_panel = pd.read_csv('child_demographics_panel.csv', dtype={'GEOID': str})
    demo_panel['GEOID'] = demo_panel['GEOID'].astype(str).str.zfill(7)
    demo_panel = demo_panel.rename(columns={'GEOID': 'district_geoid'})[
        ['district_geoid', 'checkpoint_year', 'poverty', 'income', 'child_population']]

    centroids = pd.read_csv('district_centroids_full.csv', dtype={'GEOID': str})
    centroids['GEOID'] = centroids['GEOID'].astype(str).str.zfill(7)
    centroids = centroids.rename(columns={'GEOID': 'district_geoid'})[['district_geoid', 'lat', 'lon']]

    # that single year's own kidcirc/kidatten LEVEL -- no OLS slope, no
    # multi-year history required, just that year's row from the library file
    lib_by_year = {
        year: lib.loc[lib['year'] == year, ['district_geoid', 'kidcirc', 'kidatten']]
        for year in CHECKPOINTS
    }

    out = {}
    for grade in [3, 6]:
        seda_g = seda[(seda['grade'] == grade) & (seda['year'].isin(CHECKPOINTS))][
            ['district_geoid', 'year', 'gcs_mn_all']]
        years = []
        for year in CHECKPOINTS:
            reading = seda_g.loc[seda_g['year'] == year, ['district_geoid', 'gcs_mn_all']].rename(
                columns={'gcs_mn_all': 'reading_level'})
            demo = demo_panel[demo_panel['checkpoint_year'] == year][['district_geoid', 'poverty', 'income', 'child_population']]

            df = lib_by_year[year].merge(reading, on='district_geoid', how='inner')
            df = df.merge(demo, on='district_geoid', how='inner')
            df = df.merge(centroids, on='district_geoid', how='inner')
            df['kidcirc_per_child'] = df['kidcirc'] / df['child_population']
            df['prog_per_child'] = df['kidatten'] / df['child_population']
            df = df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=['kidcirc_per_child', 'prog_per_child', 'reading_level', 'poverty', 'income', 'lat', 'lon'])
            years.append((year, df))
            log(f"grade {grade} year {year}: {len(df)} districts with full cross-sectional data")
        out[grade] = years
    return out


# ---------------------------------------------------------------------
# Step B: the MGWR fit -- one independent real bandwidth search per year
# ---------------------------------------------------------------------

def build_xyz(df):
    geoid = df['district_geoid'].values
    lat = df['lat'].values.astype(float)
    lon = df['lon'].values.astype(float)
    y = df['reading_level'].values.reshape(-1, 1).astype(float)
    Xraw = df[['kidcirc_per_child', 'prog_per_child', 'poverty', 'income']].values.astype(float)
    Xs = (Xraw - Xraw.mean(axis=0)) / Xraw.std(axis=0)
    LAT0, LON0 = 38.5, -96.0
    x_km = (lon - LON0) * 111.32 * np.cos(np.radians(LAT0))
    y_km = (lat - LAT0) * 110.54
    coords = np.column_stack([x_km, y_km])
    return geoid, coords, y, Xs


def write_year_files(grade, year, names, bws, geoid, lat, lon, params, tvalues, n, r2, aicc):
    rec = {"grade": grade, "names": names, "year": year, "bws": bws, "n": n,
           "r2": r2, "aicc": aicc, "geoid": list(geoid), "lat": list(lat), "lon": list(lon),
           "params": params, "tvalues": tvalues}
    json.dump(rec, open(f'mgwr_cross_sectional_g{grade}_{year}.json', 'w'))

    features = []
    for i in range(n):
        props = {'district_geoid': rec['geoid'][i], 'year': year, 'n_year': n}
        for j, name in enumerate(names):
            props[f'coef_{name}'] = params[i][j]
            props[f'tval_{name}'] = tvalues[i][j]
            props[f'sig_{name}'] = bool(abs(tvalues[i][j]) > 1.96)
        features.append({"type": "Feature",
                          "geometry": {"type": "Point", "coordinates": [rec['lon'][i], rec['lat'][i]]},
                          "properties": props})
    geojson = {"type": "FeatureCollection", "features": features,
               "metadata": {"grade": grade, "names": names, "bws": dict(zip(names, bws)),
                            "year": year, "r2": r2, "aicc": aicc}}
    json.dump(geojson, open(f'mgwr_cross_sectional_g{grade}_{year}.geojson', 'w'))
    log(f"saved mgwr_cross_sectional_g{grade}_{year}.json/.geojson ({n} districts)")


def rebuild_combined(grade, years_present):
    years = {}
    names = None
    all_features = []
    for year in years_present:
        p = f'mgwr_cross_sectional_g{grade}_{year}.json'
        if not os.path.exists(p):
            continue
        rec = json.load(open(p))
        names = rec['names']
        years[str(year)] = {"geoid": rec['geoid'], "lat": rec['lat'], "lon": rec['lon'],
                             "params": rec['params'], "tvalues": rec['tvalues'], "n": rec['n'],
                             "bws": rec['bws'], "r2": rec['r2'], "aicc": rec['aicc']}
        gj = json.load(open(f'mgwr_cross_sectional_g{grade}_{year}.geojson'))
        all_features.extend(gj['features'])
    if not years:
        return
    json.dump({"grade": grade, "names": names, "years": years},
               open(f'mgwr_cross_sectional_g{grade}.json', 'w'))
    json.dump({"type": "FeatureCollection", "features": all_features,
               "metadata": {"grade": grade, "names": names}},
              open(f'mgwr_cross_sectional_g{grade}.geojson', 'w'))
    log(f"rebuilt combined mgwr_cross_sectional_g{grade}.json/.geojson ({len(years)} of {len(years_present)} years)")


def clear_old_outputs(grade):
    patterns = [f'mgwr_cross_sectional_g{grade}_*.json', f'mgwr_cross_sectional_g{grade}_*.geojson',
                f'mgwr_cross_sectional_g{grade}.json', f'mgwr_cross_sectional_g{grade}.geojson']
    removed = 0
    for pat in patterns:
        for f in glob.glob(pat):
            os.remove(f)
            removed += 1
    if removed:
        log(f"grade {grade}: removed {removed} old output file(s) so this run starts clean")


MIN_YEAR_N = 80  # twice the minimum bandwidth (40) used in the search bounds below


def fit_grade(grade, years_data):
    clear_old_outputs(grade)
    skipped = [(y, len(df)) for y, df in years_data if len(df) < MIN_YEAR_N]
    years_data = [(y, df) for y, df in years_data if len(df) >= MIN_YEAR_N]
    if skipped:
        log(f"grade {grade}: skipping {len(skipped)} year(s) with too few districts to fit "
            f"(need >={MIN_YEAR_N}): {skipped}")
    if not years_data:
        log(f"grade {grade}: no year has enough districts to fit anything -- stopping")
        return
    all_years = [y for y, _ in years_data]
    log(f"grade {grade}: years = {all_years}")

    for year, df in years_data:
        geoid, coords, y_arr, Xs = build_xyz(df)
        n = len(y_arr)
        log(f"year {year}: n = {n}. Starting independent MGWR bandwidth search...")
        t0 = time.time()
        sel = Sel_BW(coords, y_arr, Xs, multi=True, constant=True, n_jobs=1)
        bws = sel.search(multi_bw_min=[40], multi_bw_max=[n], verbose=True)
        log(f"year {year}: bandwidth search done in {time.time()-t0:.1f}s. bws = {dict(zip(NAMES, bws))}")

        t0 = time.time()
        model = MGWR(coords, y_arr, Xs, sel, constant=True, n_jobs=1)
        results = model.fit()
        log(f"year {year}: fit done in {time.time()-t0:.1f}s. R2={results.R2:.4f} AICc={results.aicc:.2f}")

        write_year_files(grade, year, NAMES, [float(b) for b in bws], geoid,
                          df['lat'].astype(float).values, df['lon'].astype(float).values,
                          results.params.tolist(), results.tvalues.tolist(), n,
                          float(results.R2), float(results.aicc))
        rebuild_combined(grade, all_years)

    log(f"grade {grade}: ALL DONE")


if __name__ == '__main__':
    grades = [int(sys.argv[1])] if len(sys.argv) > 1 else [3, 6]
    log("building cross-sectional per-year tables from your 5 CSV files...")
    data = build_cross_sectional_years()
    for grade in grades:
        fit_grade(grade, data[grade])
