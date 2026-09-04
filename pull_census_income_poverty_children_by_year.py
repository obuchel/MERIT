"""
Same as pull_census_income_poverty_by_year.py (B19125_002E income +
S1701_C03_002E child poverty), plus a third variable: B09001_001E, total
population under 18 years, from the same ACS 5-year detail-table endpoint
as income (not the subject-table endpoint poverty uses). This is the
per-checkpoint child-population denominator for kidcirc_per_child_slope /
prog_per_child_slope -- recalculated fresh for each checkpoint's own ACS
vintage, rather than reusing one snapshot everywhere.

Remember Census's own naming: the ACS 5-year vintage ending in year Y
covers Y-4 through Y, so checkpoint 2022 -> vintage year 2022 (covers
2018-2022), checkpoint 2013 -> vintage year 2013 (covers 2009-2013), etc.
-- same convention as the income/poverty pull.

Usage:
    python3 pull_census_income_poverty_children_by_year.py <vintage_year>

Example:
    python3 pull_census_income_poverty_children_by_year.py 2012
    python3 pull_census_income_poverty_children_by_year.py 2014

Output: census_income_poverty_children_<vintage_year>/ST.csv for each state
FIPS code ST, each with columns GEOID, district_name, district_type,
median_family_income_with_children, child_poverty_pct, child_population.
Then combine with combine_census_income_poverty_2024.py exactly as before
(it just concatenates whatever columns are present, so the extra
child_population column comes along for free).

NOTE: this cloud workspace has no network route to api.census.gov, so run
this locally (same environment as the original income/poverty pull).
"""
import sys
import requests
import pandas as pd
import time
import os

API_KEY = "90cd40b6aee28d2cd0cb17ddd0e43f45778f2418"

STATE_FIPS = [
    '01','02','04','05','06','08','09','10','11','12','13','15','16','17','18','19','20',
    '21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37',
    '38','39','40','41','42','44','45','46','47','48','49','50','51','53','54','55','56',
    '60','66','69','72','78',
]

DISTRICT_TYPES = {
    'elementary': 'school district (elementary)',
    'secondary': 'school district (secondary)',
    'unified': 'school district (unified)',
}

def fetch(base_url, get_vars, state_fips, district_type_label, max_retries=3):
    params = {
        'get': f'NAME,{get_vars}',
        'for': f'{district_type_label}:*',
        'in': f'state:{state_fips}',
        'key': API_KEY,
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(base_url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                cols = data[0]
                df = pd.DataFrame(data[1:], columns=cols)
                geo_col = [c for c in df.columns if c.startswith('school district')]
                if geo_col:
                    df = df.drop_duplicates(subset=['state'] + geo_col)
                return df
            elif r.status_code == 204:
                return pd.DataFrame()
            else:
                print(f"  [{state_fips}/{district_type_label}] HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(1)
        except Exception as e:
            print(f"  [{state_fips}/{district_type_label}] error: {e}")
            time.sleep(1)
    return pd.DataFrame()

def geo_col(df):
    for c in ['school district (elementary)', 'school district (secondary)', 'school district (unified)']:
        if c in df.columns:
            return c
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 pull_census_income_poverty_children_by_year.py <vintage_year>")
        sys.exit(1)
    year = sys.argv[1]
    income_url = f'https://api.census.gov/data/{year}/acs/acs5'
    poverty_url = f'https://api.census.gov/data/{year}/acs/acs5/subject'
    out_dir = f'census_income_poverty_children_{year}'
    os.makedirs(out_dir, exist_ok=True)

    for st in STATE_FIPS:
        state_frames = []
        for dtype_key, dtype_label in DISTRICT_TYPES.items():
            inc = fetch(income_url, 'B19125_002E', st, dtype_label)
            pov = fetch(poverty_url, 'S1701_C03_002E', st, dtype_label)
            chd = fetch(income_url, 'B09001_001E', st, dtype_label)

            if inc.empty and pov.empty and chd.empty:
                time.sleep(0.2)
                continue

            gc_inc = geo_col(inc) if not inc.empty else None
            gc_pov = geo_col(pov) if not pov.empty else None
            gc_chd = geo_col(chd) if not chd.empty else None

            if not inc.empty:
                inc = inc[['NAME', gc_inc, 'state', 'B19125_002E']].rename(columns={gc_inc: 'district_code'})
            if not pov.empty:
                pov = pov[[gc_pov, 'state', 'S1701_C03_002E']].rename(columns={gc_pov: 'district_code'})
            if not chd.empty:
                chd = chd[[gc_chd, 'state', 'B09001_001E']].rename(columns={gc_chd: 'district_code'})

            merged = None
            for part in (inc, pov, chd):
                if part.empty:
                    continue
                merged = part if merged is None else merged.merge(part, on=['state', 'district_code'], how='outer')
            if merged is None:
                time.sleep(0.2)
                continue
            for col in ['NAME', 'B19125_002E', 'S1701_C03_002E', 'B09001_001E']:
                if col not in merged.columns:
                    merged[col] = pd.NA

            merged['district_type'] = dtype_key
            state_frames.append(merged)
            time.sleep(0.2)

        if not state_frames:
            print(f"state {st}: no districts returned at all -- check manually")
            continue

        state_df = pd.concat(state_frames, ignore_index=True)
        state_df['GEOID'] = state_df['state'] + state_df['district_code']
        state_df = state_df.rename(columns={
            'NAME': 'district_name',
            'B19125_002E': 'median_family_income_with_children',
            'S1701_C03_002E': 'child_poverty_pct',
            'B09001_001E': 'child_population',
        })[['GEOID', 'district_name', 'district_type', 'median_family_income_with_children',
            'child_poverty_pct', 'child_population']]

        for col in ['median_family_income_with_children', 'child_poverty_pct', 'child_population']:
            state_df[col] = pd.to_numeric(state_df[col], errors='coerce')
            state_df.loc[state_df[col] < 0, col] = pd.NA

        dup_count = state_df['GEOID'].duplicated().sum()
        state_df.to_csv(f'{out_dir}/{st}.csv', index=False)
        print(f"state {st}: {len(state_df)} districts saved to {out_dir}/{st}.csv"
              + (f"  [WARNING: {dup_count} duplicate GEOIDs]" if dup_count else ""))

    print(f"\nAll done. {len(os.listdir(out_dir))} state files in {out_dir}/")
    print(f"Then run: python3 combine_census_income_poverty_2024.py {out_dir} census_income_poverty_children_{year}_combined.csv")
    print("(that combine script works for any year/table set -- it just concatenates whatever columns are present)")

if __name__ == '__main__':
    main()
