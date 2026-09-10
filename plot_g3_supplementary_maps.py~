"""
Regenerates the five remaining grade-6 supplementary map figures that still
have the same "missing 2024" gap already fixed this session for Figure 3
(fig_g6_stability.png) and Figure S12 (supp_g6_direction_by_year.png):

    supp_g6_circulation_own.png     (Figure S3)
    supp_g6_program_shared.png      (Figure S6)
    supp_g6_program_own.png         (Figure S7)
    supp_g6_poverty_shared.png      (Figure S10)
    supp_g6_poverty_own.png         (Figure S11)

Why 2024 belongs in these too: the manuscript's own text (just above Figure
S1) says grade-6 year-by-year maps were "available through 2023 rather than
2024" because "the grade 6 2024 fit ... is included in Table 1 but is not
shown in a map here." But mgwr_cross_sectional_g6.json's 2024 entry has full
geoid/lat/lon/params/tvalues data, identical in structure to every other
year -- there is no missing-data reason for 2024 to be map-less. This matches
what was already found for the Figure 3 and S12 bug: the original figure
pipeline no longer exists on disk, and the "through 2023" wording is most
likely describing that lost pipeline's limitation, not a deliberate
methodological choice. Grade 3's own supplementary maps already run the full
2012-2024, eleven-year span; this brings grade 6 to the same eleven years
(2012-2019, 2022-2024) for a like-for-like comparison.

If you decide instead to keep grade 6 capped at 2023 for some reason not
captured above, just drop '2024' from YEARS_G6 below and nothing else needs
to change in this script.

Two manuscript-text edits go along with running this:
  1. The paragraph just above Figure S1 should drop the "through 2023 rather
     than 2024" sentence (or rephrase it, since 2024 is now included).
  2. Captions for S3, S6, S7, S10, S11 currently say "2012-2023" and need to
     become "2012-2024".
  3. Figure S3's caption also still has the stale "21%" post-pandemic figure
     -- that was already corrected to "20%" (20.4%) in Section 3.3 this
     session and should be updated here too for consistency.

Requires (same directory): mgwr_cross_sectional_g6.json, us_borders.py,
state_borders.json, state_labels.json
Run this yourself and it writes the five PNGs into the current directory.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from us_borders import draw_state_borders, draw_state_labels

IN = "mgwr_cross_sectional_g6.json"
data = json.load(open(IN))
names = data['names']
YEARS_G6 = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2022', '2023', '2024']
print("years used:", YEARS_G6, f"({len(YEARS_G6)} panels)")

POS, NEG, MID = "#9c3b2e", "#2f5f8a", "#eeece4"
INK, MUTED = "#22262a", "#6d716d"
DIVERGE = LinearSegmentedColormap.from_list("posneg", [NEG, MID, POS])

LAT0, LON0 = 38.5, -96.0
def project(lat, lon):
    x = (lon - LON0) * 111.32 * np.cos(np.radians(LAT0))
    y = (lat - LAT0) * 110.54
    return x, y

CONUS_LON, CONUS_LAT = (-125, -66), (24, 50)
X0, Y0 = project(CONUS_LAT[0], CONUS_LON[0])
X1, Y1 = project(CONUS_LAT[1], CONUS_LON[1])

# 11 panels -> 3 columns x 4 rows (last slot blank)
NCOLS, NROWS = 3, 4


def year_arrays(varname):
    """Returns {year: (xs, ys, vals)} for one predictor across YEARS_G6."""
    vi = names.index(varname)
    out = {}
    for yr in YEARS_G6:
        w = data['years'][yr]
        lat = np.array(w['lat']); lon = np.array(w['lon'])
        vals = np.array([p[vi] for p in w['params']])
        xs, ys = project(lat, lon)
        out[yr] = (xs, ys, vals)
    return out


def make_grid(varname, label, scale, outfile):
    """
    scale: 'shared' -> one vmin/vmax (symmetric, based on max |coef| across
           all YEARS_G6) with a single colorbar for the whole figure.
           'own'    -> each panel scaled to its own max |coef|, with its own
           small colorbar underneath.
    """
    arrays = year_arrays(varname)

    if scale == 'shared':
        vmax_shared = max(np.abs(v).max() for _, _, v in arrays.values())

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(12.5, 15.5), dpi=150,
                              layout='constrained')
    axes_flat = axes.flatten()

    for i, yr in enumerate(YEARS_G6):
        ax = axes_flat[i]
        xs, ys, vals = arrays[yr]
        vmax = vmax_shared if scale == 'shared' else np.abs(vals).max()
        draw_state_borders(ax)
        draw_state_labels(ax, fontsize=4.5)
        order = np.argsort(np.abs(vals))
        sc = ax.scatter(xs[order], ys[order], c=vals[order], cmap=DIVERGE,
                         vmin=-vmax, vmax=vmax, s=4.2, linewidths=0, zorder=2)
        ax.set_title(yr, fontsize=12, fontweight='bold', color=INK, pad=4)
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if scale == 'own':
            cb = fig.colorbar(sc, ax=ax, orientation='horizontal',
                               fraction=0.045, pad=0.02, shrink=0.9)
            cb.ax.tick_params(labelsize=6, colors=MUTED)
            cb.outline.set_visible(False)

    # blank remaining slot(s)
    for j in range(len(YEARS_G6), NROWS * NCOLS):
        axes_flat[j].axis('off')

    if scale == 'shared':
        sc_for_cb = axes_flat[0].collections[0]
        cb = fig.colorbar(sc_for_cb, ax=axes_flat.tolist(), orientation='horizontal',
                           fraction=0.025, pad=0.01, shrink=0.5, aspect=40)
        cb.ax.tick_params(labelsize=8.5, colors=MUTED)
        cb.outline.set_visible(False)
        cb.set_label(f"{label}: local MGWR coefficient (shared scale, all years)",
                     fontsize=9, color=MUTED)

    scale_note = "one color scale shared across all years" if scale == 'shared' \
        else "each year scaled to its own range"
    # Headline / subtitle pair, same two-tier convention as the other grade-6
    # figures (e.g. plot_g6_direction_by_year.py): a short bold claim, and a
    # smaller muted line underneath carrying the methodological detail --
    # rather than one long line competing with itself for emphasis.
    fig.suptitle(f"{label}, grade 6", fontsize=21, fontweight='bold',
                 color=INK, x=0.5, ha='center', y=1.035)
    fig.text(0.5, 1.012,
             f"Local MGWR coefficient by district, 2012–2024 — {scale_note}",
             ha='center', va='top', fontsize=12, color=MUTED)

    plt.savefig(outfile, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {outfile}")


if __name__ == '__main__':
    make_grid('kidcirc_per_child', "Children's circulation per child", 'own',
               'supp_g6_circulation_own.png')
    make_grid('prog_per_child', "Children's program attendance per child", 'shared',
               'supp_g6_program_shared.png')
    make_grid('prog_per_child', "Children's program attendance per child", 'own',
               'supp_g6_program_own.png')
    make_grid('poverty', "Child poverty", 'shared',
               'supp_g6_poverty_shared.png')
    make_grid('poverty', "Child poverty", 'own',
               'supp_g6_poverty_own.png')
