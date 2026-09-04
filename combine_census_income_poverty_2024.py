"""
Combine the per-state 2024 income/poverty CSVs (from
pull_census_income_poverty_2024_v2.py) into one national file.

Also drops Census's "remainder of state" placeholder rows -- any GEOID
whose 5-digit district code is 99999. That code means "territory not
covered by any single operating school district," not a real school
system, and it's assigned once per district type (elementary/secondary/
unified), so the SAME numeric code can legitimately appear 2-3 times with
different district_type values. That's what the per-state "duplicate
GEOID" warnings from the pull script were flagging -- it isn't the same
district counted twice, it's three different non-districts sharing a
code. None of them can ever match a real library system, so they're
dropped here rather than "deduplicated."

Usage:
    python3 combine_census_income_poverty_2024.py <state_csv_folder> <output.csv>

Example:
    python3 combine_census_income_poverty_2024.py census_income_poverty_2024 census_income_poverty_2024_combined.csv
"""
import sys
import glob
import os
import pandas as pd

def main():
    in_dir, out_path = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(in_dir, '*.csv')))
    if not files:
        print(f"No CSV files found in {in_dir}")
        sys.exit(1)

    dfs = [pd.read_csv(f, dtype={'GEOID': str}) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} state files, {len(combined)} rows total")

    is_remainder = combined['GEOID'].str[-5:] == '99999'
    n_remainder = is_remainder.sum()
    combined = combined[~is_remainder].copy()
    print(f"Dropped {n_remainder} 'remainder of state' placeholder rows (GEOID ending 99999)")

    # Sanity check: after dropping remainders, GEOID should be unique on its
    # own (a real district shouldn't be double-classified as two types).
    dup = combined[combined.duplicated(subset=['GEOID'], keep=False)].sort_values('GEOID')
    if len(dup):
        print(f"\nWARNING: {len(dup)} rows still share a GEOID after dropping remainders -- inspect these:")
        print(dup[['GEOID', 'district_name', 'district_type']].to_string(index=False))
    else:
        print("GEOID is unique across the remaining rows -- good.")

    combined.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}: {len(combined)} districts, "
          f"{combined['median_family_income_with_children'].notna().sum()} with income, "
          f"{combined['child_poverty_pct'].notna().sum()} with poverty rate")

if __name__ == '__main__':
    main()
