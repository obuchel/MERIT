"""
Rebuild expanding_window_data_with_programming_v6.pkl (filename kept as-is
so run_mgwr_district_temporal_v6.py needs no changes) from the CURRENT,
verified project data -- this replaces two things build_expanding_window_v6.py
still had wrong:

  1. Circulation/programming source: district_year_library_gis_v5.csv ->
     district_year_library_gis_v6.csv (the joint-reporting fix for
     kidcirc_share, which also affects nothing else in THIS pickle since
     circ_pc/prog_pc were already correct in v5 -- but v6 is the current,
     user-verified file, so that's what every other output now uses).

  2. Poverty/income source: a single time-invariant CHR-2025 county-level
     snapshot (matched by fuzzy county-name string matching) -> the
     checkpoint-specific, district-level poverty_income_panel.csv built
     from the raw Census API (B19125_002E income, S1701_C03_002E child
     poverty), joined on BOTH district_geoid AND checkpoint_year. Poverty
     and income are not actually constant over 2013-2024 -- the old
     pipeline used one fixed value in every window because that's what a
     single CHR snapshot could offer; now that a real 7-checkpoint panel
     exists (97.7-99.4% match rate against the library districts, vs.
     88-90% for the old EDGE/CDP03 mix this eventually replaced), each
     checkpoint uses ITS OWN poverty/income reading rather than reusing
     one 2025 figure everywhere. This is a real methodology correction:
     any coefficient trend over checkpoints in the old pickle was partly
     an artifact of poverty/income never moving while everything else did.

SEDA reading scores now read directly from seda_filtered_2025.csv (district_geoid
derived here from sedaadmin, zero-padded to 7 digits) rather than requiring a
separate pre-built seda_with_geoid.csv -- one less file to keep track of.
Centroids (district_centroids_full.csv) are unchanged from v6.

Same expanding-window slope methodology as v2 through v6, unchanged: for
each of the 7 checkpoints, a district's circ_pc/prog_pc/reading trend uses
every year of that series through the checkpoint (minimum 4 usable points
required for a slope -- fewer than that and np.polyfit's line has no real
meaning), fit as a simple ordinary-least-squares slope over year.

Run this from a folder containing all four inputs: district_year_library_gis_v6.csv,
seda_filtered_2025.csv, poverty_income_panel.csv, district_centroids_full.csv.

Usage:
    python3 build_expanding_window_v7.py

Output: expanding_window_data_with_programming_v6.pkl
"""
import pickle
import numpy as np
import pandas as pd

CHECKPOINTS = [2013, 2015, 2017, 2019, 2022, 2023, 2024]
MIN_PTS = 4

def slopes_through(df, id_col, year_col, val_col, cutoff_year, min_pts=MIN_PTS):
    """
    Same definition as a per-district np.polyfit(year, val, 1) slope through
    cutoff_year, but computed for every district in df at once via the
    closed-form OLS slope (sums, not a per-group Python callback) --
    groupby().apply() with a Python function was taking minutes here across
    ~12,000 districts x 7 checkpoints x several metrics; this is the same
    math, vectorized, and finishes in under a second per call.
    """
    sub = df.loc[(df[year_col] <= cutoff_year) & df[val_col].notna(), [id_col, year_col, val_col]].copy()
    sub['xy'] = sub[year_col] * sub[val_col]
    sub['xx'] = sub[year_col].astype(float) ** 2
    agg = sub.groupby(id_col).agg(n=(year_col, 'count'), sx=(year_col, 'sum'),
                                   sy=(val_col, 'sum'), sxy=('xy', 'sum'), sxx=('xx', 'sum'))
    denom = agg['n'] * agg['sxx'] - agg['sx'] ** 2
    slope = (agg['n'] * agg['sxy'] - agg['sx'] * agg['sy']) / denom
    slope = slope.where((agg['n'] >= min_pts) & (denom != 0))
    return slope.rename('slope').reset_index()

# --- GIS-apportioned circulation + programming per district-year (v6: joint-reporting fix applied) ---
lib = pd.read_csv('district_year_library_gis_v6.csv', dtype={'district_geoid': str})
lib['district_geoid'] = lib['district_geoid'].astype(str).str.zfill(7)

# --- SEDA reading scores, long format -- district_geoid derived from sedaadmin here
# rather than requiring a separate pre-built file, since sedaadmin loses its
# leading zero for state FIPS 01-09 (e.g. Alabama's 100005 -> 0100005) ---
seda = pd.read_csv('seda_filtered_2025.csv', dtype={'sedaadmin': str})
seda['district_geoid'] = seda['sedaadmin'].str.zfill(7)

# --- checkpoint-specific poverty/income, from the raw Census API panel (replaces the old static CHR lookup) ---
poverty_panel = pd.read_csv('poverty_income_panel.csv', dtype={'GEOID': str})
poverty_panel['GEOID'] = poverty_panel['GEOID'].astype(str).str.zfill(7)
poverty_panel = poverty_panel.rename(columns={
    'GEOID': 'district_geoid',
    'median_income': 'income',
    'child_poverty_pct': 'poverty',
})[['district_geoid', 'checkpoint_year', 'poverty', 'income']]
income_defs = pd.read_csv('poverty_income_panel.csv', usecols=['checkpoint_year', 'income_definition']).drop_duplicates()
print("poverty/income source by checkpoint:")
print(income_defs.to_string(index=False))

# --- centroids, full 13,368-district set (Census's own internal points) ---
centroids = pd.read_csv('district_centroids_full.csv', dtype={'GEOID': str})
centroids['GEOID'] = centroids['GEOID'].astype(str).str.zfill(7)
centroids = centroids.rename(columns={'GEOID': 'district_geoid'})[['district_geoid', 'lat', 'lon']]

# circ_pc_slope/prog_pc_slope don't depend on grade -- compute each once per
# checkpoint rather than redundantly inside the grade loop (this alone was
# roughly half the runtime before vectorizing slopes_through).
circ_slopes_by_year = {}
prog_slopes_by_year = {}
for year in CHECKPOINTS:
    circ_slopes_by_year[year] = slopes_through(lib, 'district_geoid', 'year', 'circ_pc', year).rename(columns={'slope': 'circ_pc_slope'})
    prog_slopes_by_year[year] = slopes_through(lib, 'district_geoid', 'year', 'prog_pc', year).rename(columns={'slope': 'prog_pc_slope'})
    print(f"checkpoint {year}: circ_pc_slope computed for {circ_slopes_by_year[year]['circ_pc_slope'].notna().sum()} districts, "
          f"prog_pc_slope for {prog_slopes_by_year[year]['prog_pc_slope'].notna().sum()}")

out = {}
for grade in [3, 6]:
    seda_g = seda[seda['grade'] == grade][['district_geoid', 'year', 'gcs_mn_all']]
    windows = []
    for year in CHECKPOINTS:
        reading_slopes = slopes_through(seda_g, 'district_geoid', 'year', 'gcs_mn_all', year).rename(columns={'slope': 'reading_slope'})

        # poverty/income AT THIS CHECKPOINT specifically -- not a fixed lookup
        pov_this_year = poverty_panel[poverty_panel['checkpoint_year'] == year][['district_geoid', 'poverty', 'income']]

        df = circ_slopes_by_year[year].merge(prog_slopes_by_year[year], on='district_geoid').merge(reading_slopes, on='district_geoid')
        df = df.merge(pov_this_year, on='district_geoid', how='inner')
        df = df.merge(centroids, on='district_geoid', how='inner')
        n_before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['circ_pc_slope', 'prog_pc_slope', 'reading_slope', 'poverty', 'income', 'lat', 'lon'])
        print(f"grade {grade} window {year}: {n_before} candidates -> {len(df)} districts with full data")
        windows.append((year, df))
    out[grade] = windows

pickle.dump(out, open('expanding_window_data_with_programming_v6.pkl', 'wb'))
print("\nSaved expanding_window_data_with_programming_v6.pkl")

print("\n--- Coverage comparison, 2024 checkpoint ---")
for grade in [3, 6]:
    n_2024 = [n for y, n in out[grade] if y == 2024][0]
    print(f"grade {grade}: {len(n_2024)} districts")
