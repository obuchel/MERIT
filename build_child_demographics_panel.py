"""
Stitch the 11 checkpoint-year census_income_poverty_children_<year>_combined.csv
files into ONE panel: one row per (GEOID, checkpoint_year), carrying poverty,
income, AND child_population together.

This replaces the earlier two-file split (poverty_income_panel.csv +
child_population_panel.csv, built separately by build_poverty_panel.py and
build_child_population_panel.py). That split was a historical artifact, not
a real design choice: poverty_income_panel.csv used to source its early
checkpoints (2013/2015/2017/2019) from older CDP03/EDGE exports, a
genuinely different provenance than child_population_panel.csv's brand-new
Census pull. Once build_poverty_panel.py was updated to pull every
checkpoint from the same census_income_poverty_children_<year>_combined.csv
files child_population_panel.csv already used, that distinction disappeared
-- both panels ended up reading the identical 11 source files and merging
the result back together downstream. This script does that read once.

(The two old panels weren't even byte-identical in row count, for what it's
worth -- child_population_panel.csv dropped rows with missing/zero
child_population while poverty_income_panel.csv didn't apply the same
filter to poverty/income, a ~35-50-district-per-year discrepancy that had
no effect on the final fitted sample, since both fit scripts inner-merge
against both panels and then dropna on everything anyway. This script
applies that one filter consistently instead of inconsistently across two
files.)

Run this from the folder containing either form of the source data below --
whichever you actually have. No need to build the combined form yourself
first; this reads straight from the unzipped per-state folders.

Usage:
    python3 build_child_demographics_panel.py [folder]

For each year in CHECKPOINTS, looks for the data in either of two shapes,
in this order:
  1. census_income_poverty_children_<year>_combined.csv -- one already-
     concatenated file (what my sandbox happened to have built).
  2. census_income_poverty_children_<year>/*.csv -- a folder of per-state
     files (01.csv, 02.csv, ... one per state FIPS code), each with columns
     GEOID, district_name, district_type, median_family_income_with_children,
     child_poverty_pct, child_population. This is what you get from
     unzipping census_income_poverty_children_<year>.zip directly -- the
     normal case, and the one this script now handles without any
     intermediate combine step.
Any missing year is skipped (printed as MISSING), so this runs fine before
every checkpoint's data has arrived.

Output: child_demographics_panel.csv with columns
    GEOID, checkpoint_year, poverty, income, child_population, income_definition
"""
import sys
import os
import glob
import pandas as pd

CHECKPOINTS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]

INCOME_DEFINITION = 'median family income, families with own children under 18 (B19125_002E)'


def drop_remainder_rows(df, year):
    # Census assigns a "remainder of state" placeholder GEOID (5-digit
    # district code 99999) once per district type -- elementary, secondary,
    # unified -- meaning "territory not covered by any single operating
    # school district," not a real school system. Because it's assigned
    # per type rather than being one code, the SAME numeric GEOID can
    # legitimately appear 2-3 times with wildly different values (one
    # state-scale, one tiny) -- that's a real collision if left in, since
    # a later merge on (district_geoid, checkpoint_year) would multiply
    # rows for anything that matched it. None of these can ever match a
    # real library system, so they're dropped outright, not deduplicated.
    is_remainder = df['GEOID'].str[-5:] == '99999'
    if is_remainder.any():
        print(f"{year}: dropped {is_remainder.sum()} 'remainder of state' placeholder rows (GEOID ending 99999)")
    df = df[~is_remainder].copy()
    dup = df[df.duplicated(subset=['GEOID'], keep=False)]
    if len(dup):
        print(f"{year}: WARNING -- {len(dup)} rows still share a GEOID after dropping remainders -- inspect these")
    return df


def load_year(folder, year):
    combined_path = os.path.join(folder, f'census_income_poverty_children_{year}_combined.csv')
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path, dtype={'GEOID': str})
        return drop_remainder_rows(df, year)

    per_state_dir = os.path.join(folder, f'census_income_poverty_children_{year}')
    if os.path.isdir(per_state_dir):
        # per-state files are named by 2-digit FIPS code (01.csv, 02.csv, ...);
        # a zip's __MACOSX resource-fork folder (if present alongside, not
        # inside, this directory) is never touched since we only glob here
        state_files = sorted(glob.glob(os.path.join(per_state_dir, '*.csv')))
        if state_files:
            df = pd.concat((pd.read_csv(f, dtype={'GEOID': str}) for f in state_files), ignore_index=True)
            return drop_remainder_rows(df, year)

    return None


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    frames = []
    for year in CHECKPOINTS:
        df = load_year(folder, year)
        if df is None:
            print(f"MISSING: no census_income_poverty_children_{year}_combined.csv or "
                  f"census_income_poverty_children_{year}/ folder found -- skipping checkpoint {year}")
            continue
        df['income'] = pd.to_numeric(df['median_family_income_with_children'], errors='coerce')
        df['poverty'] = pd.to_numeric(df['child_poverty_pct'], errors='coerce')
        df['child_population'] = pd.to_numeric(df['child_population'], errors='coerce')
        df['checkpoint_year'] = year
        df['income_definition'] = INCOME_DEFINITION
        # one consistent filter, applied once: child_population is a
        # denominator downstream, so a missing or non-positive value there
        # is definitionally unusable regardless of whether poverty/income
        # happen to be present for that row. Poverty/income are left as
        # NaN when missing rather than dropped here -- the fit scripts'
        # own dropna() at merge time is where "needs every field" actually
        # gets enforced, same as before.
        df = df[df['child_population'] > 0]
        df = df[['GEOID', 'checkpoint_year', 'poverty', 'income', 'child_population', 'income_definition']]
        print(f"{year}: {len(df)} districts, {df['poverty'].notna().sum()} with poverty, "
              f"{df['income'].notna().sum()} with income, {df['child_population'].notna().sum()} with child_population")
        frames.append(df)

    if not frames:
        print("\nNo input files found yet -- nothing to save.")
        return

    panel = pd.concat(frames, ignore_index=True)
    dup = panel[panel.duplicated(subset=['GEOID', 'checkpoint_year'], keep=False)]
    if len(dup):
        print(f"\nWARNING: {len(dup)} rows share a (GEOID, checkpoint_year) pair -- inspect before merging:")
        print(dup.sort_values(['GEOID', 'checkpoint_year']).to_string(index=False))

    panel.to_csv('child_demographics_panel.csv', index=False)
    print(f"\nSaved child_demographics_panel.csv: {len(panel)} rows across "
          f"{panel['checkpoint_year'].nunique()} checkpoint years, "
          f"{panel['GEOID'].nunique()} distinct districts")


if __name__ == '__main__':
    main()
