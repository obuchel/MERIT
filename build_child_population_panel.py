"""
Stitch per-checkpoint child_population columns (from
pull_census_income_poverty_children_by_year.py + combine_census_income_poverty_2024.py)
into one panel: one row per (GEOID, checkpoint_year), the same shape as
poverty_income_panel.csv, ready to merge onto the library district-year
panel. Only the child_population column is used here -- that pull script's
income/poverty columns are redundant with poverty_income_panel.csv and are
ignored, so it's fine if the two panels were built from separately-run
pulls.

Run this from the folder containing the combined files.

Usage:
    python3 build_child_population_panel.py [folder]

Looks for: census_income_poverty_children_<year>_combined.csv for every
year in CHECKPOINTS. Any missing year is skipped (printed as MISSING), so
this runs fine before every checkpoint's file has arrived.

Output: child_population_panel.csv with columns GEOID, checkpoint_year, child_population
"""
import sys
import os
import pandas as pd

CHECKPOINTS = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    frames = []
    for year in CHECKPOINTS:
        path = os.path.join(folder, f'census_income_poverty_children_{year}_combined.csv')
        if not os.path.exists(path):
            print(f"MISSING: {path} -- skipping checkpoint {year}")
            continue
        df = pd.read_csv(path, dtype={'GEOID': str})
        df['checkpoint_year'] = year
        df['child_population'] = pd.to_numeric(df['child_population'], errors='coerce')
        df = df[['GEOID', 'checkpoint_year', 'child_population']].dropna(subset=['child_population'])
        df = df[df['child_population'] > 0]
        print(f"{year}: {len(df)} districts with a valid child_population")
        frames.append(df)

    if not frames:
        print("\nNo input files found yet -- nothing to save.")
        return

    panel = pd.concat(frames, ignore_index=True)
    dup = panel[panel.duplicated(subset=['GEOID', 'checkpoint_year'], keep=False)]
    if len(dup):
        print(f"\nWARNING: {len(dup)} rows share a (GEOID, checkpoint_year) pair -- inspect before merging:")
        print(dup.sort_values(['GEOID', 'checkpoint_year']).to_string(index=False))

    panel.to_csv('child_population_panel.csv', index=False)
    print(f"\nSaved child_population_panel.csv: {len(panel)} rows across "
          f"{panel['checkpoint_year'].nunique()} checkpoint years, "
          f"{panel['GEOID'].nunique()} distinct districts")

if __name__ == '__main__':
    main()
