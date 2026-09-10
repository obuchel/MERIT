"""
Superimposed "which pattern is significant, and how consistently" map for
grade 6, split by period rather than pooled across all 11 years -- pre-
pandemic (2012-2019, 8 years) and post-pandemic (2022-2024, 3 years) each get
their own row, all three predictors as columns. This is the significance-
filtered companion to plot_g6_period_comparison.py's Figure 3, which uses
sign-consistency only; this version restricts to statistically significant
years, answering the sharper question "where does a significant effect hold,
not just where does the sign happen to agree."

Method, per period, per predictor, per district (restricted to the district
panel present in every year of THAT period -- 8 years for pre-pandemic, 3 for
post -- computed independently per period, not across all 11 years at once):

  n_pos_sig = number of years in the period that t-value > 1.96
  n_neg_sig = number of years in the period that t-value < -1.96
  net_score = (n_pos_sig - n_neg_sig) / n_years_in_period      (in [-1, +1])

net_score = +1 means significant positive in every year of that period; -1
means significant negative in every year of that period. Districts NEVER
significant in either direction during that period are drawn as small pale
grey background dots rather than colored, so the colored layer shows only
"significant patterns," per the original ask. A district that is significant
in both directions within the same period (rare with only 3 or 8 years, but
possible) nets toward 0 and reads pale, same as a never-significant district
-- an intentional simplification: a coefficient that reverses sign under
significance within one period is not a stable pattern for that period
either.

Requires (same directory): mgwr_cross_sectional_g6.json, us_borders.py,
state_borders.json, state_labels.json
Run this yourself; it writes one PNG into the current directory.
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
T_SIG = 1.96

PERIODS = [
    ("2012–2019 (pre-pandemic)", ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']),
    ("2022–2024 (post-pandemic)", ['2022', '2023', '2024']),
]

POS, NEG, MID = "#9c3b2e", "#2f5f8a", "#eeece4"
GREY = "#d8dad6"
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

VARS = [
    ("kidcirc_per_child", "Circulation"),
    ("prog_per_child",    "Programming"),
    ("poverty",           "Poverty"),
]

fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=150, layout='compressed')
fig.get_layout_engine().set(hspace=0.02, wspace=0.03, h_pad=0.02, w_pad=0.02)

for row, (period_label, years) in enumerate(PERIODS):
    n_years = len(years)
    for col, (key, label) in enumerate(VARS):
        ax = axes[row, col]
        vi = names.index(key)

        by_geoid, latlon = {}, {}
        for yr in years:
            w = data['years'][yr]
            for gid, t, lat, lon in zip(w['geoid'], w['tvalues'], w['lat'], w['lon']):
                by_geoid.setdefault(gid, {})[yr] = t[vi]
                latlon[gid] = (lat, lon)
        complete = {g: d for g, d in by_geoid.items() if all(y in d for y in years)}

        net, xs, ys, ever_sig = [], [], [], []
        for g, d in complete.items():
            tvals = np.array([d[y] for y in years])
            n_pos = int((tvals > T_SIG).sum())
            n_neg = int((tvals < -T_SIG).sum())
            net.append((n_pos - n_neg) / n_years)
            ever_sig.append(n_pos > 0 or n_neg > 0)
            lat, lon = latlon[g]
            x, y = project(lat, lon)
            xs.append(x); ys.append(y)
        net = np.array(net); xs = np.array(xs); ys = np.array(ys)
        ever_sig = np.array(ever_sig)

        n_total = len(net)
        n_ever = ever_sig.sum()
        n_always_pos = (net > 1 - 1e-9).sum()
        n_always_neg = (net < -1 + 1e-9).sum()

        draw_state_borders(ax)
        draw_state_labels(ax, fontsize=5.5)

        ax.scatter(xs[~ever_sig], ys[~ever_sig], color=GREY, s=3.2, linewidths=0, zorder=1)
        order = np.argsort(np.abs(net[ever_sig]))
        sc = ax.scatter(xs[ever_sig][order], ys[ever_sig][order], c=net[ever_sig][order],
                         cmap=DIVERGE, vmin=-1, vmax=1, s=5.5, linewidths=0, zorder=2)

        if row == 1:
            cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.045, pad=0.02, shrink=0.85)
            cb.set_ticks([-1, 0, 1])
            cb.set_ticklabels(["sig. neg.\nevery yr", "never sig. /\nflips sign", "sig. pos.\nevery yr"])
            cb.ax.tick_params(labelsize=7.3, colors=INK)
            cb.outline.set_visible(False)

        if row == 0:
            ax.text(0.5, 1.10, label, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color=INK)
        ax.text(0.5, 1.015,
                f"sig. ≥1 yr: {100*n_ever/n_total:.0f}% · always sig+: {100*n_always_pos/n_total:.0f}% · "
                f"always sig–: {100*n_always_neg/n_total:.0f}% · n={n_total:,}",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=8.0, color=MUTED)

        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        print(f"[{period_label}] {label}: n={n_total} | sig ≥1yr {100*n_ever/n_total:.1f}% | "
              f"always sig+ {100*n_always_pos/n_total:.1f}% | always sig- {100*n_always_neg/n_total:.1f}%")

    axes[row, 0].text(-0.08, 0.5, period_label, transform=axes[row, 0].transAxes,
                       ha='right', va='center', fontsize=11.5, fontweight='bold', color=INK, rotation=90)

fig.suptitle("Grade 6: where does a significant pattern hold, before vs. after the pandemic?",
             fontsize=19, fontweight='bold', color=INK, x=0.5, ha='center', y=1.05)
fig.text(0.5, 1.018,
         "Each period's years superimposed per district (8 years pre-pandemic, 3 post). Grey = never "
         "statistically significant (|t| < 1.96) in that period. Colored districts were significant at "
         "least once in that period; color is the net share of that period's years significant in the "
         "dominant direction.",
         ha='center', va='top', fontsize=9.4, color=MUTED)

plt.savefig('g6_significant_pattern_by_period.png', bbox_inches='tight', facecolor='white')
print("Saved g6_significant_pattern_by_period.png")
