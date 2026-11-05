"""
generate_manuscript_figures_revised.py
Standalone figure script for the C and N allocation manuscript.

Reads: CN Manuscript/EGU BGS/C_N Allocation Supplementary II.xlsx (sheet S1),
using the final N columns (the sheet's legacy headers still read
"δ15N (‰) corrected", "TN (%) corrected", "C/N corrected"; they are exported
to the combined workbook under plain names — see the "S1" sheet below).
Fig 3 uses the climate/soil columns in S1: Precipitation GSP (GSP),
Surface Temperature GST (GST), Soil Moisture (SM, GLDAS, kg m^-2) and
Soil Microbial N (SMN). SM enters the δ15N model; δ13C uses SMN only.

Fig 2 calls cluster_mycorrhizal_association.py (keep it in the same folder).
It produces the selected cluster plot and a supplementary figure comparing
the tested predictor combinations and clustering scores. The main figure
uses the selected K-means solution; association labels are shown by point
colour and predicted clusters by pale background regions.

Outputs (saved to CN Manuscript/EGU BGS/figures/):
    Fig1b_1c_vertical.png                   (stacked main-text panels b and c)
    Fig2_mycorrhizal_clustering.png          (main-text Fig 2: selected K-means solution)
    FigS_clustering_comparison.png           (mycorrhizal clustering scores and tests)
    Fig_S1_climatological_distinction.png
    Fig_comparison_original_vs_corrected.png
        (OFF by default: editorial record only — the manuscript contains no
         correction statement. Set MAKE_CORRECTION_COMPARISON = True to
         regenerate it, e.g. for the journal letter.)
    FigS2_pearson_heatmap.png                (Fig S2: inter-variable AND driver×target
                                              Pearson heatmaps; tables in workbook)
    FigS6_elevation_coverage.png            (Fig S6: (a) elevation coverage by genus and
                                              tissue type, (b) within-transect foliar
                                              δ15N–elevation LOESS profiles, span 0.75)
    Fig3_isoscape_validation.png
    Supplementary_Tables_combined.xlsx      (ONE workbook with every table:
        S1 (the analysis dataset, plain column names, no "corrected"),
        TableS1_pairwise_kmeans, TableS2_inter-variable, FigS2_driver-target,
        TableS3_within_genus, TableS4_descriptive_statistics,
        Fig3_model_statistics, FigS6_d15N_elevation)
    Additional statistics and each figure's caption are printed to the
    console; all tables go to the combined workbook (no separate CSVs).

Figure 3 reconstructs the manuscript analysis:
    - delta15N: linear multiple regression using GSP, GST, SM, and SMN.
    - delta13C: ridge regression using GSP, GST, elevation, SM, SMN, and C4 status.
"""

from pathlib import Path
import warnings
import colorsys
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.text import Text
import re
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import (gaussian_kde, kendalltau, kruskal, norm, pearsonr,
                         rankdata, shapiro)
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (adjusted_rand_score, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fig 2 clustering: the exhaustive predictor × model search lives in this
# companion module (keep cluster_mycorrhizal_association.py next to this file).
try:
    import cluster_mycorrhizal_association as myco_clustering
except ImportError as _exc:  # pragma: no cover
    raise SystemExit(
        "Fig 2 needs cluster_mycorrhizal_association.py in the same folder "
        f"as this script (import failed: {_exc})"
    )

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
})
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
XLSX_PATH   = PROJECT_DIR / "CN Manuscript" / "EGU BGS" / "C_N Allocation Supplementary II.xlsx"
SHEET       = "S1"
OUTPUT_DIR  = PROJECT_DIR / "CN Manuscript" / "EGU BGS" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Registry of every tabular output of this script.  All tables are collected
# here and written to ONE combined Excel workbook at the end of main()
# (see _write_combined_workbook); no separate CSVs are produced.
_TABLES: "dict[str, pd.DataFrame]" = {}

# Isoscape predictors, following the manuscript Results §3.4 equations:
#   delta15N ~ GSP + GST + SM + SMN               (linear regression)
#   delta13C ~ GSP + GST + Elevation + SMN + C4    (ridge regression; no SM)
# Dung samples are excluded from both models (plants only, n ≈ 135).
# True (default): include the soil layers SM and SMN as in §3.4.
ISOSCAPE_SOIL_COVARIATES = True

# The original-vs-corrected comparison figure documents the TN recalibration
# for the editorial record ONLY — the manuscript itself contains no
# correction statement.  It is therefore OFF by default; set True to
# regenerate it (e.g., for the journal letter).
MAKE_CORRECTION_COMPARISON = False

# ----------------------------------------------------------------------------
# Captions — printed to the console after the matching figure is saved, so
# every generated figure ships with its caption (plain unicode for consoles).
# ----------------------------------------------------------------------------
FIGURE_CAPTIONS = {
    "fig1b_1c": (
        "Figure 1: Study area map showing sampled locations with blue markers (a); distribution of δ13C and TOC (b); and distribution of δ15N values and TN (c) across the three transects. Dashed lines mark reference values (−24.5 ‰ and 0 ‰)."
    ),
    "fig_comparison": (
        "Figure S1 (comparison) — TN (%) and δ15N (‰, AIR): original vs "
        "corrected values by genus and transect. Paired half-violins per "
        "genus; left half = original, right half = corrected, white dot = mean."
    ),
    "fig_climatological": (
        "Figure S1. Climatological distinction among the Western (Karnali), Central (Annapurna) and Eastern (Kosi) Himalayan transects. Points show sample-level growing-season mean precipitation (GSP) and growing-season mean surface temperature (GST); shaded polygons show each transect's convex-hull range."
    ),
    "fig2": (
        "Figure 2 — Mycorrhizal association clustering. K-means solution with "
        "the selected predictor combination (elevation, δ13C, δ15N and C/N; "
        "ARI = 0.227, k = 4). Points are coloured by genus-based mycorrhizal "
        "association; pale background regions show the predicted clusters. "
        "All other tested predictor combinations and their silhouette and "
        "adjusted Rand scores are reported in Fig S (clustering comparison)."
    ),
    "figS_clustering_comparison": (
        "Figure S (clustering comparison) — Predictor combinations tested in "
        "the exhaustive K-means search with their silhouette and adjusted "
        "Rand scores, and the position of the selected solution."
    ),
    "figS2": (
        "Figure S2 — Pearson correlation heatmaps (plants only, dung "
        "excluded). Top: inter-variable correlations among δ13C, TOC, δ15N "
        "and TN per transect and for all plants (lower triangle shown). "
        "Bottom: climate/soil drivers (GSP, GST, MAT, MAP, ST, SM, Elevation) "
        "vs foliar targets (δ15N, δ13C, TOC, TN, C/N). Pairs involving TN or "
        "C/N use positive values only (exclusion codes removed)."
    ),
    "figS6": (
        "Figure S6 — (a) Elevation coverage by genus and tissue type (n = 141). "
        "Point colour indicates transect (red: Western; blue: Central; "
        "green: Eastern); black ticks mark genus mean elevations. All grass "
        "and dung samples are from the Western transect; no Pinus was sampled "
        "in the Eastern transect. (b) Within-transect foliar δ15N–elevation "
        "profiles with LOESS smoothers (span = 0.75)."
    ),
    "fig3": (
        "Figure 3 — Isoscape validation (plants only, dung excluded). "
        "(a) Observed vs modelled δ15N (linear multiple regression on GSP, "
        "GST, SM, SMN); (b) observed vs modelled δ13C (ridge regression on "
        "GSP, GST, Elevation, SM, SMN and C4 status); (c, d) model residuals "
        "by transect. In-sample, held-out (80:20) and within-transect r "
        "statistics are given in the annotation boxes."
    ),
}


def print_caption(key):
    """Print the caption belonging to the figure that was just saved."""
    print(f"  Caption — {FIGURE_CAPTIONS[key]}")
    print()

# ============================================================================
# LOAD
# ============================================================================

df = pd.read_excel(XLSX_PATH, sheet_name=SHEET)

# Normalise the header so the final column names match exactly: strip stray
# leading/trailing whitespace and collapse runs of whitespace (including
# non-breaking spaces) to single spaces.  No column is renamed or invented.
df.columns = [" ".join(str(c).split()) for c in df.columns]

# The FINAL workbook column names (revised header).  The script reads ONLY
# these from the sheet; everything else it uses is an alias derived below.
# If any of them is missing, stop immediately and show what the sheet really
# contains, instead of failing later with a KeyError inside a figure function.
REQUIRED_FINAL_COLUMNS = [
    "Transect", "Genus", "Domain", "Mycorrhizal association",
    "δ13C (‰)", "TOC (%)", "δ15N (‰)", "TN (%)",
    "δ15N (‰) corrected", "TN (%) corrected", "C/N corrected",
    "Elevation (m)",
    "Precipitation GSP (mm/m)", "Surface Temperature GST (K)",
    "Soil Moisture (m3/m3)", "Soil Temperature (K)",
    "Soil Microbial Niotrogen (gN/m2)",
    "Precipitation Annual (mm/m)", "Surface Temperature Annual (K)",
]
_missing = [c for c in REQUIRED_FINAL_COLUMNS if c not in df.columns]
if _missing:
    raise SystemExit(
        f"Missing final columns in sheet '{SHEET}': {_missing}\n"
        f"Columns actually present: {list(df.columns)}"
    )

# Aliases used throughout. "original" = uncorrected values as in the file;
# the corrected columns are aliased to d15N_corrected / TN_pct_corrected /
# CN_corrected and used as supplied.  (In the exported "S1" sheet these
# appear under plain names — see _S1_EXPORT_COLUMNS.)
df = df.rename(columns={
    "Transect":                             "Transect_full",
    "δ13C (‰)":                             "d13C",
    "TOC (%)":                              "TOC_pct",
    "δ15N (‰)":                             "d15N_original",
    "TN (%)":                               "TN_pct_original",
    "δ15N (‰) corrected":                   "d15N_corrected",
    "TN (%) corrected":                     "TN_pct_corrected",
    "C/N corrected":                        "CN_corrected",
    "Elevation (m)":                        "Elevation_m",
    "Precipitation GSP (mm/m)":             "GSP_mm",
    "Surface Temperature GST (K)":          "GST_K",
})
df["Transect"] = df["Transect_full"]

# Grass C3 / Grass C4 -> Grass (Domain keeps the C3/C4 information)
df["Genus"] = df["Genus"].where(~df["Genus"].str.startswith("Grass"), "Grass")
df["is_C4"] = df["Domain"].eq("Grass C4").astype(int)

# Mycorrhizal association is taken from the file (Dung is labelled "Dung" and
# is therefore excluded from the four-class clustering in Fig 2).
# Labels are genus-based: Pinus/Abies (ectomycorrhizal), Rhododendron (ericoid),
# grasses (arbuscular) and Juniperus (Cupressaceae; arbuscular, kept as its own genus class).
_MYCO_LABELS = {"En" + "domycorrhizal": "Juniperus (arbuscular)",
                "Ecto" + "mycorrhizal": "Pinus, Abies (ectomycorrhizal)",
                "Arbuscular" + " mycorrhiza": "Grasses (arbuscular)",
                "Eri" + "coid": "Rhododendron (ericoid)"}
df["Mycorrhizal"] = df["Mycorrhizal association"].replace(_MYCO_LABELS)

# Extra drivers for Fig S2 (MAT, MAP, soil temperature), final column names
df["MAT_K"]  = df["Surface Temperature Annual (K)"]
df["MAP_mm"] = df["Precipitation Annual (mm/m)"]
df["ST_K"]   = df["Soil Temperature (K)"]

# SM / SMN for Fig 3 (final sheet header spellings).
# Soil moisture unit note: the S1 sheet column is headed "Soil Moisture
# (m3/m3)", but the product actually extracted is the GLDAS Catchment Land
# Surface Model (GLDAS_CLSM025_DA1_D v2.2), which reports soil moisture in
# kg m^-2 (Li et al., 2020; doi:10.5067/TXBMLX370XX8), and the manuscript
# Section 3.4 equation uses SM in kg m^-2.  The values are therefore used
# AS-IS, with no x1000 conversion.  If a future sheet really holds
# volumetric (m3/m3) data instead, set SM_IS_VOLUMETRIC = True to apply the
# legacy x1000 conversion (resulting unit then kg m^-3, not kg m^-2).
SM_IS_VOLUMETRIC = False
SM_COL  = "Soil Moisture (m3/m3)"             # legacy header; values are GLDAS kg m^-2
SMN_COL = "Soil Microbial Niotrogen (gN/m2)"  # "Niotrogen" typo kept: must match sheet
df["SM_kg_m2"]  = df[SM_COL] * 1000.0 if SM_IS_VOLUMETRIC else df[SM_COL]
df["SMN_gN_m2"] = df[SMN_COL]
_sm_check = df["SM_kg_m2"].dropna()
if len(_sm_check):
    print(f"Soil moisture used by the models: min = {_sm_check.min():.2f}, "
          f"median = {_sm_check.median():.2f}, max = {_sm_check.max():.2f} "
          f"({'kg m^-3' if SM_IS_VOLUMETRIC else 'kg m^-2 (GLDAS)'})")
    if not SM_IS_VOLUMETRIC and _sm_check.max() < 5.0:
        print("  WARNING: every SM value is < 5, which looks like volumetric "
              "m3/m3 data rather than GLDAS kg m^-2. Check the source "
              "extraction, or set SM_IS_VOLUMETRIC = True.")

# Exclusion codes: the sheet encodes samples excluded from TN statistics with
# NEGATIVE codes (-10.00 / -5.00) instead of blanks, so dropna() alone does
# not catch them.  They are NOT data, so they are replaced with NaN here,
# GLOBALLY: every figure, table, statistical test and model below (Fig 1c
# bars, Table S4, the Kruskal-Wallis tests, the LDA features, the exported
# "S1" sheet) then works from the same cleaned columns.  Fig S2 / Table S2
# keep their own positive-value filter as a second line of defence.
_n_codes = {}
for _col in ("TN_pct_corrected", "CN_corrected"):
    _bad = df[_col].notna() & df[_col].le(0)
    _n_codes[_col] = int(_bad.sum())
    df.loc[_bad, _col] = np.nan
df["_tn_ok"] = df["TN_pct_corrected"].gt(0) & df["CN_corrected"].gt(0)
if any(_n_codes.values()):
    print(f"NOTE: non-positive TN/C/N exclusion codes replaced with blanks — "
          f"TN: {_n_codes['TN_pct_corrected']} row(s), "
          f"C/N: {_n_codes['CN_corrected']} row(s).")

# ============================================================================
# CONSTANTS
# ============================================================================

TRANSECTS = [
    "Western Himalaya (Karnali)",
    "Central Himalaya (Annapurna)",
    "Eastern Himalaya (Kosi)",
]

GENERA_MAIN = ["Rhododendron", "Pinus", "Abies", "Juniperus"]

GENUS_COLORS = {
    "Rhododendron": "#3B7EA1",
    "Pinus":        "#FDB462",
    "Abies":        "#8FBC8F",
    "Juniperus":    "#B03060",
    "Grass":        "#808080",
    "Dung":         "#A52A2A",
    "Other":        "#D3D3D3",
}

TRANSECT_COLORS = {
    "Western Himalaya (Karnali)":   "#32CD32",
    "Central Himalaya (Annapurna)": "#4169E1",
    "Eastern Himalaya (Kosi)":      "#DC143C",
}

TRANSECT_LABELS = {
    "Western Himalaya (Karnali)":   "Karnali",
    "Central Himalaya (Annapurna)": "Annapurna",
    "Eastern Himalaya (Kosi)":      "Kosi",
}

MYCO_COLORS = {
    "Juniperus (arbuscular)":       "#1f77b4",
    "Pinus, Abies (ectomycorrhizal)":       "#2ca02c",
    "Grasses (arbuscular)": "#17becf",
    "Rhododendron (ericoid)":               "#ffbb00",
}

def _desaturate(hex_color, factor=0.38):
    """Return a paler version of hex_color for 'original' half-violins."""
    r, g, b = [int(hex_color.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s * factor, min(v * 1.3, 1.0))
    return (r2, g2, b2)

GENUS_COLORS_ORIG = {g: _desaturate(c) for g, c in GENUS_COLORS.items()}

# ============================================================================
# HELPERS
# ============================================================================

def style_figure(fig):
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontfamily("DejaVu Sans")
        text_artist.set_fontsize(9.5)


def violin_or_point(ax, values, position, color, width=0.30, alpha=0.75):
    """Full violin (both sides) or scatter point for n < 4."""
    vals = np.asarray(pd.Series(values).dropna())
    if len(vals) == 0:
        return
    if len(vals) < 4:
        ax.scatter([position], [vals.mean()], s=50, color=color,
                   alpha=alpha, edgecolors="black", linewidth=0.5, zorder=4)
        return
    kde    = gaussian_kde(vals, bw_method="scott")
    y_grid = np.linspace(vals.min(), vals.max(), 200)
    dens   = kde(y_grid)
    dens   = dens / dens.max() * width
    ax.fill_betweenx(y_grid, position - dens, position + dens,
                     color=color, alpha=alpha, linewidth=0)
    ax.scatter([position], [vals.mean()], s=20, color="white",
               edgecolors=color, linewidth=1.0, zorder=5)


def half_violin(ax, values, position, side, color, width=0.40, alpha=0.82):
    """
    Left half  (side='l') or right half (side='r') violin.
    A thin white centre line separates the two halves.
    """
    vals = np.asarray(pd.Series(values).dropna())
    if len(vals) == 0:
        return
    if len(vals) < 4:
        offset = -0.12 if side == "l" else 0.12
        ax.scatter([position + offset], [vals.mean()], s=50, color=color,
                   alpha=alpha, edgecolors="black", linewidth=0.5, zorder=4)
        return
    kde    = gaussian_kde(vals, bw_method="scott")
    y_grid = np.linspace(vals.min(), vals.max(), 200)
    dens   = kde(y_grid)
    dens   = dens / dens.max() * width
    if side == "l":
        ax.fill_betweenx(y_grid, position - dens, position,
                         color=color, alpha=alpha, linewidth=0)
    else:
        ax.fill_betweenx(y_grid, position, position + dens,
                         color=color, alpha=alpha, linewidth=0)
    # mean dot
    ax.scatter([position], [vals.mean()], s=22, color="white",
               edgecolors=color, linewidth=1.2, zorder=5)


def _legend_patches(genera, label_suffix=""):
    return [mpatches.Patch(facecolor=GENUS_COLORS[g], edgecolor="none",
                           label=g + label_suffix)
            for g in genera if g in GENUS_COLORS]


# ============================================================================
# FIG 1B  — δ13C (violin) + TOC (bar)
# ============================================================================

def make_fig1b_1c():
    print("Fig 1b–1c …")
    x_pos = np.arange(len(TRANSECTS))
    width = 0.15
    offset = np.array([-1, 0, 1, 2]) * width - 0.5 * width
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    panels = [
        ("d13C", "TOC_pct", r"$\delta^{13}$C (‰, VPDB)$_{\mathrm{violin}}$",
         "TOC (%)$_{\mathrm{bar}}$", (-34, -18), (0, 60), -24.5,
         r"(b)  $\delta^{13}$C and TOC across the three transects"),
        ("d15N_corrected", "TN_pct_corrected",
         r"$\delta^{15}$N (‰, AIR)$_{\mathrm{violin}}$",
         "TN (%)$_{\mathrm{bar}}$", (-6, 15), (0, 8), 0,
         r"(c)  $\delta^{15}$N and TN across the three transects"),
    ]

    for ax_left, (isotope_col, bar_col, left_label, right_label,
                  left_ylim, right_ylim, reference, title) in zip(axes, panels):
        ax_right = ax_left.twinx()
        for i, transect in enumerate(TRANSECTS):
            for j, genus in enumerate(GENERA_MAIN):
                subset = df[
                    (df["Transect_full"] == transect) & (df["Genus"] == genus)
                ]
                position = x_pos[i] + offset[j]
                violin_or_point(
                    ax_left, subset[isotope_col], position, GENUS_COLORS[genus],
                    width=width * 0.85,
                )
                mean_value = subset[bar_col].mean()
                if pd.notna(mean_value):
                    ax_right.bar(
                        position,
                        mean_value,
                        width=width * 0.75,
                        color=GENUS_COLORS[genus],
                        alpha=0.28,
                        edgecolor="black",
                        linewidth=0.4,
                    )

        ax_left.set_ylabel(left_label, fontweight="bold")
        ax_right.set_ylabel(right_label, fontweight="bold")
        ax_right.spines["right"].set_visible(True)
        ax_right.tick_params(axis="y", right=True, labelright=True)
        ax_left.set_ylim(left_ylim)
        ax_right.set_ylim(right_ylim)
        ax_left.axhline(reference, color="grey", lw=0.6, ls="--", alpha=0.5)
        ax_left.grid(False)
        ax_right.grid(False)
        if ax_left == axes[-1]:
            ax_left.legend(
                handles=_legend_patches(GENERA_MAIN),
                title="Genus",
                loc="upper right",
                frameon=True,
            )
        ax_left.set_title(title, loc="left", fontweight="bold")

    axes[-1].set_xlabel("Transect", fontweight="bold")
    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels(TRANSECTS)
    fig.tight_layout(h_pad=1.5)
    style_figure(fig)
    fig.savefig(
        OUTPUT_DIR / "Fig1b_1c_vertical.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  ✓ saved")
    print_caption("fig1b_1c")


# ============================================================================
# FIG COMPARISON  — paired half-violins, original vs corrected
# ============================================================================

def _comparison_panel(ax, orig_col, corr_col, genera_here, t_df, ylabel, ylim):
    """Draw one panel: paired half-violins per genus."""
    tick_pos, tick_lab = [], []
    spacing = 1.3

    for g_idx, genus in enumerate(genera_here):
        g_df   = t_df[t_df["Genus"] == genus]
        centre = g_idx * spacing
        tick_pos.append(centre)
        tick_lab.append(genus)

        orig_vals = g_df[orig_col].dropna().values
        corr_vals = g_df[corr_col].dropna().values

        half_violin(ax, orig_vals, centre, "l", GENUS_COLORS_ORIG[genus])
        half_violin(ax, corr_vals, centre, "r", GENUS_COLORS[genus])

        # mean labels
        if len(orig_vals):
            ax.text(centre - 0.44, np.mean(orig_vals), f"{np.mean(orig_vals):.1f}",
                    ha="right", va="center", color="dimgrey")
        if len(corr_vals):
            ax.text(centre + 0.44, np.mean(corr_vals), f"{np.mean(corr_vals):.1f}",
                    ha="left",  va="center", color=GENUS_COLORS[genus])

        # centre divider
        ax.axvline(centre, color="white", lw=1.2, zorder=6)
        ax.axvline(centre, color="grey",  lw=0.5, ls=":", alpha=0.5, zorder=5)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=28, ha="right")
    ax.set_ylim(ylim)
    ax.set_xlim(-0.75, max(tick_pos, default=0) + 0.75)
    ax.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.45)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(axis="y", alpha=0.22, lw=0.6)


def make_fig_comparison():
    print("Comparison figure …")

    TRANSECT_ORDER = [
        ("Western Himalaya (Karnali)",   "Western\n(Karnali)"),
        ("Central Himalaya (Annapurna)", "Central\n(Annapurna)"),
        ("Eastern Himalaya (Kosi)",      "Eastern\n(Kosi)"),
    ]

    row_cfg = [
        # orig_col            corr_col             ylabel         ylim
        ("TN_pct_original",  "TN_pct_corrected",  "TN (%)",       (-0.5, 16)),
        ("d15N_original", "d15N_corrected", r"$\delta^{15}$N (‰, AIR)", (-8, 15)),
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 10),
        sharey="row", sharex=False,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.06)

    for row_idx, (orig_col, corr_col, ylabel, ylim) in enumerate(row_cfg):
        for col_idx, (t_full, t_short) in enumerate(TRANSECT_ORDER):
            ax    = axes[row_idx, col_idx]
            t_df  = df[df["Transect_full"] == t_full]
            genera_here = [g for g in GENERA_MAIN if g in t_df["Genus"].values]

            _comparison_panel(ax, orig_col, corr_col,
                              genera_here, t_df, ylabel, ylim)

            if col_idx > 0:
                ax.set_ylabel("")
            if row_idx == 0:
                ax.set_title(t_short, fontweight="bold")

    # ── shared legend ──────────────────────────────────────────────────────
    legend_handles = []
    for genus in GENERA_MAIN:
        legend_handles += [
            mpatches.Patch(facecolor=GENUS_COLORS_ORIG[genus], edgecolor="none",
                           label=f"{genus} — original"),
            mpatches.Patch(facecolor=GENUS_COLORS[genus],      edgecolor="none",
                           label=f"{genus} — corrected"),
        ]

    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        frameon=True, edgecolor="lightgrey",
        title="Left half = original   ·   Right half = corrected   ·   white dot = mean",
    )
    fig.suptitle(
        "TN% and " + r"$\delta^{15}$N" + ": original vs corrected values by genus and transect",
        fontweight="bold", y=1.01,
    )

    style_figure(fig)
    fig.savefig(
        OUTPUT_DIR / "Fig_comparison_original_vs_corrected.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  ✓ saved")
    print_caption("fig_comparison")


def make_fig_climatological_distinction():
    print("Climatological distinction figure …")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    legend_handles = []

    for transect in TRANSECTS:
        subset = df.loc[
            df["Transect_full"] == transect, ["GSP_mm", "GST_K"]
        ].dropna()
        points = subset.to_numpy(dtype=float)
        if not len(points):
            raise ValueError(f"No complete GSP/GST observations for transect {transect!r}")
        color = TRANSECT_COLORS[transect]
        label = TRANSECT_LABELS[transect]

        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
            except QhullError:
                pass
            else:
                polygon = mpatches.Polygon(
                    points[hull.vertices],
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.12,
                    linestyle="--",
                    linewidth=1.5,
                    zorder=1,
                )
                ax.add_patch(polygon)

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=62,
            color=color,
            edgecolors="#333333",
            linewidth=0.55,
            label=label,
            zorder=2,
        )
        legend_handles.append(
            mpatches.Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
                label=(
                    f"{label} ({subset['GSP_mm'].min():.0f}–"
                    f"{subset['GSP_mm'].max():.0f} mm, "
                    f"{subset['GST_K'].min():.0f}–{subset['GST_K'].max():.0f} K)"
                ),
            )
        )

    ax.set_xlabel("Growing-season precipitation (mm month⁻¹)", fontweight="bold")
    ax.set_ylabel("Growing-season surface temperature (K)", fontweight="bold")
    ax.set_title(
        "Climatological distinction between the three Himalayan transects",
        fontweight="bold",
    )
    ax.grid(True, color="#000000", alpha=0.1, zorder=0)
    ax.legend(
        handles=legend_handles,
        title="Transect (GSP and GST range)",
        loc="upper right",
        frameon=True,
        edgecolor="grey",
    )
    fig.tight_layout()
    style_figure(fig)
    fig.savefig(
        OUTPUT_DIR / "Fig_S1_climatological_distinction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  ✓ saved")
    print_caption("fig_climatological")


# ============================================================================
# LDA + K-MEANS MACHINERY  —  shared data-prep used by the values report
#                              (LDA projection of the six
#                              leaf-chemistry features, K-means in LDA space)
# ============================================================================

def _lda_kmeans_data():
    """
    Shared data-prep used by the values report.
    Returns a dict with everything needed so the LDA is only fitted once.
    """
    MYCO_ORDER = ["Juniperus (arbuscular)", "Pinus, Abies (ectomycorrhizal)",
                  "Grasses (arbuscular)", "Rhododendron (ericoid)"]
    FEAT_COLS  = ["d13C", "TN_pct_corrected", "d15N_corrected",
                  "TOC_pct", "CN_corrected", "Elevation_m"]
    FEAT_LABELS = {
        "d13C":             r"$\delta^{13}$C",
        "TN_pct_corrected": "TN (%)",
        "d15N_corrected":   r"$\delta^{15}$N",
        "TOC_pct":          "TOC (%)",
        "CN_corrected":     "C/N",
        "Elevation_m":      "Elevation",
    }

    clean = df[FEAT_COLS + ["Mycorrhizal"]].dropna().copy()
    clean = clean[clean["Mycorrhizal"].isin(MYCO_ORDER)].reset_index(drop=True)
    # (dung carries the label "Dung" and drops out here; the TN/C/N exclusion
    #  codes were already replaced with NaN at load time)

    X_raw = StandardScaler().fit_transform(clean[FEAT_COLS])
    le    = LabelEncoder()
    y     = le.fit_transform(clean["Mycorrhizal"])

    lda     = LinearDiscriminantAnalysis(n_components=3)
    X_lda   = lda.fit_transform(X_raw, y)
    ev      = lda.explained_variance_ratio_

    # Loadings scaled for the biplot arrow length
    scalings_12 = lda.scalings_[:, :2]
    arrow_scale = 3.2
    loadings    = scalings_12 * arrow_scale / np.abs(scalings_12).max()

    # K-means in LDA space
    km      = KMeans(n_clusters=4, random_state=0, n_init=30)
    cl_km   = km.fit_predict(X_lda)
    sil_km  = silhouette_score(X_lda, cl_km)
    ari_km  = adjusted_rand_score(y, cl_km)

    # Map each cluster → colour of its dominant true group
    classes  = le.classes_
    dom_cls  = {ci: classes[np.bincount(y[cl_km == ci], minlength=len(classes)).argmax()]
                for ci in range(4)}
    cl_color = {ci: MYCO_COLORS[dom_cls[ci]] for ci in range(4)}

    # ── validation of the supervised LDA step ────────────────────────────────
    # (1) 5-fold cross-validated LDA predictions (LDA fitted on training folds
    #     only), repeated with 20 different fold assignments.
    accs, aris = [], []
    for seed in range(20):
        pred = cross_val_predict(LinearDiscriminantAnalysis(), X_raw, y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        accs.append((pred == y).mean())
        aris.append(adjusted_rand_score(y, pred))
    # (3) cluster composition and one-to-one matching accuracy
    ct = pd.crosstab(pd.Series(classes[y], name="class"), pd.Series(cl_km, name="cluster"))
    ct = ct.reindex(columns=range(4), fill_value=0)
    r_i, c_i = linear_sum_assignment(-ct.values)
    match_acc = ct.values[r_i, c_i].sum() / len(y)
    recall = {ct.index[r]: ct.values[r, c] / ct.values[r].sum() for r, c in zip(r_i, c_i)}
    modal = {cls: (int(ct.loc[cls].max()), int(ct.loc[cls].sum()), int(ct.loc[cls].values.argmax()))
             for cls in ct.index}
    in_own = {cls: int(sum(ct.loc[cls, ci] for ci in range(4) if dom_cls[ci] == cls))
              for cls in ct.index}
    cv = dict(acc=float(np.mean(accs)), ari=float(np.mean(aris)),
              base=float(np.bincount(y).max() / len(y)),
              crosstab=ct, match_acc=float(match_acc), recall=recall,
              modal=modal, in_own=in_own, n_by_class=ct.sum(axis=1).to_dict())

    return dict(
        clean=clean, X_raw=X_raw, X_lda=X_lda, y=y,
        le=le, classes=classes, ev=ev,
        loadings=loadings, feat_cols=FEAT_COLS, feat_labels=FEAT_LABELS,
        km=km, cl_km=cl_km, sil_km=sil_km, ari_km=ari_km,
        dom_cls=dom_cls, cl_color=cl_color, cv=cv,
    )


# ============================================================================
# FIG 2  — best clustering from the exhaustive predictor × model search
# ============================================================================
#
# cluster_mycorrhizal_association.py (companion script, same folder) clusters
# every combination of 2-8 candidate predictors with K-means and Gaussian
# mixture models (full, tied and diagonal covariance) and scores each
# partition by ARI against the known mycorrhizal associations.  The
# 2-variable solution is kept unless a model with more variables gains at
# least ARI_GAIN_REQUIRED (0.05) ARI on both the reported fit and an
# 8-seed stability average.  Only the winning solution is plotted here,
# as main-text Fig 2.
#
# Outputs: Fig2_mycorrhizal_clustering.png and FigS_clustering_comparison.png

_FIG2_SEARCH = {}


def make_fig2():
    """Run clustering and create only the main and supplementary figures."""
    print("Fig 2 (mycorrhizal clustering) …")
    _FIG2_SEARCH.update(myco_clustering.run_analysis(XLSX_PATH, OUTPUT_DIR))
    print("  ✓ saved Fig2_mycorrhizal_clustering.png and FigS_clustering_comparison.png")
    print_caption("fig2")
    print_caption("figS_clustering_comparison")


# ============================================================================
# FIG 3  — isoscape validation (observed vs modelled d15N, d13C)
# ============================================================================

def _model_stats(obs, mod):
    obs, mod = np.asarray(obs, float), np.asarray(mod, float)
    return {
        "n":    len(obs),
        "r":    np.corrcoef(obs, mod)[0, 1],
        "r2":   r2_score(obs, mod),
        "rmse": mean_squared_error(obs, mod) ** 0.5,
        "mae":  mean_absolute_error(obs, mod),
        "bias": float(np.mean(mod - obs)),
    }


def _stats_box(ax, st, extra=""):
    txt = (f"In-sample, n = {st['n']}\nR² = {st['r2']:.3f}\nr = {st['r']:.3f}\n"
           f"RMSE = {st['rmse']:.2f}‰\nMAE = {st['mae']:.2f}‰")
    if extra:
        txt += "\n" + extra
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55))


def _equation(target, intercept, predictors, coefs):
    out = f"{target}_modelled = {intercept:.6g}"
    for p, c in zip(predictors, coefs):
        out += f" {'+' if c >= 0 else '-'} {abs(c):.6g}*{p}"
    return out


_ISO_CACHE = {}


def make_fig3():
    print("Fig 3 …")

    have_soil = not (df["SM_kg_m2"].isna().all() or df["SMN_gN_m2"].isna().all())
    if ISOSCAPE_SOIL_COVARIATES and not have_soil:
        print("  ✗ skipped: soil layers requested but missing or empty in sheet S1.")
        return

    P15_BASE = ["GST_K", "GSP_mm"]
    P13_BASE = ["GST_K", "GSP_mm", "Elevation_m", "is_C4"]
    # δ15N: GST, GSP, SM, SMN (linear).  δ13C: GST, GSP, elevation, SMN and
    # the C3/C4 indicator (ridge) — NO SM, matching the manuscript §3.4.
    P15_SOIL = ["GST_K", "GSP_mm", "SM_kg_m2", "SMN_gN_m2"]
    P13_SOIL = ["GST_K", "GSP_mm", "Elevation_m", "SMN_gN_m2", "is_C4"]
    d15_pred = P15_SOIL if ISOSCAPE_SOIL_COVARIATES else P15_BASE
    d13_pred = P13_SOIL if ISOSCAPE_SOIL_COVARIATES else P13_BASE

    # δ15N uses the final values; dung is excluded from both models
    # (manuscript §3.4: plants only, n ≈ 135)
    plants3 = df["Genus"] != "Dung"
    d15 = df.loc[plants3, ["Transect", "d15N_corrected"] + d15_pred].dropna().copy()
    d13 = df.loc[plants3, ["Transect", "d13C"] + d13_pred].dropna().copy()
    print(f"  plants only (dung excluded): d15N n = {len(d15)}, d13C n = {len(d13)}")

    # δ15N: linear multiple regression
    m15 = LinearRegression().fit(d15[d15_pred], d15["d15N_corrected"])
    d15["mod"] = m15.predict(d15[d15_pred])
    d15["res"] = d15["d15N_corrected"] - d15["mod"]
    eq15 = _equation("d15N", m15.intercept_, d15_pred, m15.coef_)

    # δ13C: ridge regression (as per manuscript)
    m13 = Ridge(alpha=1.0).fit(d13[d13_pred], d13["d13C"])
    d13["mod"] = m13.predict(d13[d13_pred])
    d13["res"] = d13["d13C"] - d13["mod"]
    eq13 = _equation("d13C", m13.intercept_, d13_pred, m13.coef_)

    st15 = _model_stats(d15["d15N_corrected"], d15["mod"])
    st13 = _model_stats(d13["d13C"],           d13["mod"])

    # Held-out skill (80:20 split stratified by transect, seed 42) and
    # within-transect agreement of the fitted values with the observations
    def _holdout(d, col, preds, mk):
        tr, te = train_test_split(d, test_size=0.2, stratify=d["Transect"], random_state=42)
        mt = mk().fit(tr[preds], tr[col])
        return _model_stats(te[col], mt.predict(te[preds]))
    ho15 = _holdout(d15, "d15N_corrected", d15_pred, LinearRegression)
    ho13 = _holdout(d13, "d13C", d13_pred, lambda: Ridge(alpha=1.0))
    wr15 = {TRANSECT_LABELS[t]: pearsonr(g["d15N_corrected"], g["mod"])[0]
            for t, g in d15.groupby("Transect")}
    wr13 = {TRANSECT_LABELS[t]: pearsonr(g["d13C"], g["mod"])[0]
            for t, g in d13.groupby("Transect")}
    _ISO_CACHE.update(dict(ho15=ho15, ho13=ho13, wr15=wr15, wr13=wr13, st15=st15, st13=st13))
    ex15 = (f"Held-out (n = {ho15['n']}): R² = {ho15['r2']:.2f}, RMSE = {ho15['rmse']:.2f}‰\n"
            "Within-transect r: " + ", ".join(f"{k} {v:.2f}" for k, v in wr15.items()))
    ex13 = (f"Held-out (n = {ho13['n']}): R² = {ho13['r2']:.2f}, RMSE = {ho13['rmse']:.2f}‰\n"
            "Within-transect r: " + ", ".join(f"{k} {v:.2f}" for k, v in wr13.items()))

    # Sensitivity: the alternative predictor set (with / without SM and SMN)
    alt = []
    if have_soil:
        for nm, col, pr, mk in (
                ("delta15N", "d15N_corrected", P15_BASE if ISOSCAPE_SOIL_COVARIATES else P15_SOIL, LinearRegression),
                ("delta13C", "d13C", P13_BASE if ISOSCAPE_SOIL_COVARIATES else P13_SOIL, lambda: Ridge(alpha=1.0))):
            dd = df.loc[plants3, [col, "Transect"] + pr].dropna()
            mm = mk().fit(dd[pr], dd[col])
            alt.append({**_model_stats(dd[col], mm.predict(dd[pr])),
                        "model": f"{nm} sensitivity, plants ("
                                 + ("without" if ISOSCAPE_SOIL_COVARIATES else "with") + " SM and SMN)",
                        "equation": _equation(nm, mm.intercept_, pr, mm.coef_)})
            alt.append({**_holdout(dd, col, pr, mk),
                        "model": f"{nm} sensitivity, held-out ("
                                 + ("without" if ISOSCAPE_SOIL_COVARIATES else "with") + " SM and SMN)",
                        "equation": ""})
    _TABLES["Fig3_model_statistics"] = pd.DataFrame([
        {**st15,  "model": "delta15N fitted (plants, dung excluded)",  "equation": eq15},
        {**st13,  "model": "delta13C fitted (plants, dung excluded)",  "equation": eq13},
        {**ho15,  "model": "delta15N held-out (80:20, seed 42)", "equation": ""},
        {**ho13,  "model": "delta13C held-out (80:20, seed 42)", "equation": ""},
    ] + alt)

    print(f"  d15N fitted  R2 = {st15['r2']:.3f}, RMSE = {st15['rmse']:.2f}‰  (held-out R2 = {ho15['r2']:.3f})")
    print(f"  d13C fitted  R2 = {st13['r2']:.3f}, RMSE = {st13['rmse']:.2f}‰  (held-out R2 = {ho13['r2']:.3f})")
    _ISO_CACHE.update(dict(alt=alt, eq15=eq15, eq13=eq13))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def obs_vs_mod(ax, d, obs_col, st, label, title, extra=""):
        for t, c in TRANSECT_COLORS.items():
            sub = d[d["Transect"] == t]
            ax.scatter(sub[obs_col], sub["mod"], c=c, label=TRANSECT_LABELS[t],
                       s=60, alpha=0.65, edgecolors="black", linewidth=0.5)
        lo = min(d[obs_col].min(), d["mod"].min())
        hi = max(d[obs_col].max(), d["mod"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="1:1 line")
        _stats_box(ax, st, extra)
        ax.set_xlabel(f"Observed {label} (‰)", fontweight="bold")
        ax.set_ylabel(f"Modelled {label} (‰)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(title="Transect", loc="lower right")
        ax.grid(True, alpha=0.3)

    def residuals(ax, d, label, title):
        for t, c in TRANSECT_COLORS.items():
            ax.hist(d.loc[d["Transect"] == t, "res"], bins=15, alpha=0.5,
                    color=c, label=TRANSECT_LABELS[t],
                    edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", ls="--", lw=2)
        ax.set_xlabel(f"Residual {label} (observed − modelled, ‰)",
                      fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    d15_label = r"$\delta^{15}$N (‰, AIR)"
    d13_label = r"$\delta^{13}$C (‰, VPDB)"
    obs_vs_mod(axes[0, 0], d15, "d15N_corrected", st15, d15_label,
               r"(a) $\delta^{15}$N: linear multi-regression", ex15)
    obs_vs_mod(axes[0, 1], d13, "d13C", st13, d13_label,
               r"(b) $\delta^{13}$C: ridge regression", ex13)
    residuals(axes[1, 0], d15, d15_label, r"(c) $\delta^{15}$N model residuals")
    residuals(axes[1, 1], d13, d13_label, r"(d) $\delta^{13}$C model residuals")

    handles = [plt.Rectangle((0, 0), 1, 1, color=TRANSECT_COLORS[t], alpha=0.6)
               for t in TRANSECTS]
    fig.legend(handles, [TRANSECT_LABELS[t] for t in TRANSECTS],
               title="Transect", loc="lower center", ncol=3)
    plt.tight_layout()
    style_figure(fig)
    fig.savefig(
        OUTPUT_DIR / "Fig3_isoscape_validation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  ✓ saved")
    print_caption("fig3")


# ============================================================================
# SHARED STATISTICS  (used by the report, Table S3 and Fig S2, so that every
# number in the text comes from one place)
# ============================================================================

def _dunn_bonferroni(groups):
    """Dunn's post hoc test (tie-corrected) with Bonferroni adjustment.
    groups: dict name -> 1-D array.  Returns {(a, b): adjusted p}."""
    names = list(groups)
    vals = np.concatenate([np.asarray(groups[k], float) for k in names])
    N = len(vals)
    ranks = rankdata(vals)
    mean_rank, n_i, i0 = {}, {}, 0
    for k in names:
        n = len(groups[k]); mean_rank[k] = ranks[i0:i0 + n].mean(); n_i[k] = n; i0 += n
    _, cnt = np.unique(vals, return_counts=True)
    sigma2 = N * (N + 1) / 12 - (cnt ** 3 - cnt).sum() / (12 * (N - 1))
    m = len(names) * (len(names) - 1) // 2
    out = {}
    for a, b in combinations(names, 2):
        z = (mean_rank[a] - mean_rank[b]) / np.sqrt(sigma2 * (1 / n_i[a] + 1 / n_i[b]))
        out[(a, b)] = min(1.0, 2 * norm.sf(abs(z)) * m)
    return out


def _pair_p(dunn, a, b):
    return dunn[(a, b)] if (a, b) in dunn else dunn[(b, a)]


def _stats_bundle():
    TNc, N15 = "TN_pct_corrected", "d15N_corrected"
    GENERA = ["Rhododendron", "Pinus", "Abies", "Juniperus", "Grass"]
    S = {}

    def kw(col, groups):
        g = {k: v[col].dropna().values for k, v in groups.items()}
        H, p = kruskal(*g.values())
        return dict(H=H, p=p, n={k: len(v) for k, v in g.items()},
                    mean={k: v.mean() for k, v in g.items()},
                    dunn=_dunn_bonferroni(g))

    tr = {TRANSECT_LABELS[t]: df[df["Transect_full"] == t] for t in TRANSECTS}
    S["tn_transect"]   = kw(TNc, tr)
    S["d15n_transect"] = kw(N15, tr)
    S["tn_genus"]      = kw(TNc, {g: df[df["Genus"] == g] for g in GENERA})

    # within-transect and all-sample Pearson r (GSP/GST vs TN and d15N)
    S["wt"] = {}
    for lab, g in list(tr.items()) + [("ALL", df)]:
        for drv in ("GSP_mm", "GST_K"):
            for tg, col in (("TN", TNc), ("d15N", N15)):
                s = g[[drv, col]].dropna()
                S["wt"][(lab, drv, tg)] = pearsonr(s[drv], s[col])[0]

    # d13C vs elevation within transect (C3 plants, dung excluded)
    c3 = df[(df["Domain"] != "Grass C4") & (df["Genus"] != "Dung")]
    S["d13c_elev"] = {}
    for t in TRANSECTS:
        s = c3[c3["Transect_full"] == t][["Elevation_m", "d13C"]].dropna()
        r, p = pearsonr(s["Elevation_m"], s["d13C"])
        S["d13c_elev"][TRANSECT_LABELS[t]] = (r, p, len(s))
    S["d13c_c3_mean"] = c3["d13C"].mean()
    S["d13c_c3_by_transect"] = {TRANSECT_LABELS[t]: c3.loc[c3["Transect_full"] == t, "d13C"].mean()
                                for t in TRANSECTS}

    # Shapiro-Wilk
    S["shapiro"] = {
        "TN": shapiro(df[TNc].dropna()), "d15N": shapiro(df[N15].dropna()),
        "d13C (all)": shapiro(df["d13C"].dropna()), "d13C (C3)": shapiro(c3["d13C"].dropna()),
        "TOC": shapiro(df["TOC_pct"].dropna()), "Elevation": shapiro(df["Elevation_m"].dropna()),
    }

    # negative d15N
    neg = df[df[N15] < 0]
    S["neg"] = dict(n=len(neg), share=len(neg) / len(df),
                    genus=neg["Genus"].value_counts().to_dict(),
                    transect=neg["Transect_full"].map(TRANSECT_LABELS).value_counts().to_dict(),
                    elev_mean=neg["Elevation_m"].mean(), elev_min=neg["Elevation_m"].min(),
                    elev_max=neg["Elevation_m"].max())

    # elevation-band means (same bands as Section 4.4)
    def band(g, lo, hi):
        s = g[g["Elevation_m"].between(lo, hi)]
        return dict(n=len(s), d15N=s[N15].mean(), d15N_min=s[N15].min(), d15N_max=s[N15].max(),
                    d13C=s["d13C"].mean(), TOC=s["TOC_pct"].mean())
    wk, ap, ks = TRANSECTS
    S["bands"] = {
        "West 2000-3200 m (C3, no dung)": band(c3[c3["Transect_full"] == wk], 2000, 3200),
        "Central 2500-3500 m":            band(df[df["Transect_full"] == ap], 2500, 3500),
        "East 3000-4500 m":               band(df[df["Transect_full"] == ks], 3000, 4500),
    }
    # d15N by elevation bin, Central and Eastern
    edges = [0, 2800, 3400, 4000, 4600, 6000]
    S["d15n_bins"] = {}
    for t in (ap, ks):
        g = df[df["Transect_full"] == t]
        b = pd.cut(g["Elevation_m"], edges)
        S["d15n_bins"][TRANSECT_LABELS[t]] = g.groupby(b, observed=True)[N15].agg(["mean", "count"])

    S["genus_elev"] = df.groupby("Genus")["Elevation_m"].agg(["min", "mean", "max"])
    S["lda"] = _lda_kmeans_data()
    S["tn_by_transect"] = {TRANSECT_LABELS[t]: (df.loc[df["Transect_full"] == t, TNc].mean(),
                                                df.loc[df["Transect_full"] == t, TNc].std(),
                                                int(df.loc[df["Transect_full"] == t, TNc].notna().sum()))
                           for t in TRANSECTS}
    return S


def _table_s4():
    """Genus and transect descriptive statistics (mean ± SD) and Kendall tau
    (d15N vs TOC) within genus."""
    TNc, N15, CNc = "TN_pct_corrected", "d15N_corrected", "CN_corrected"
    rows = []
    def add(name, g, tau=True):
        r = dict(Group=name, n=len(g), n_TN=int(g[TNc].notna().sum()))
        for lab, col in (("TN", TNc), ("TOC", "TOC_pct"), ("d13C", "d13C"), ("d15N", N15), ("C/N", CNc)):
            r[lab + "_mean"] = g[col].mean(); r[lab + "_sd"] = g[col].std()
        if tau and len(g) >= 5:
            s = g[[N15, "TOC_pct"]].dropna(); k = kendalltau(s[N15], s["TOC_pct"])
            r["tau_d15N_TOC"], r["p_tau"] = k[0], k[1]
        rows.append(r)
    for gname in ["Rhododendron", "Pinus", "Abies", "Juniperus", "Grass", "Dung"]:
        add(gname, df[df["Genus"] == gname])
        if gname == "Grass":
            add("  of which C3 grasses", df[(df["Genus"] == "Grass") & (df["Domain"] == "Grass C3")], tau=False)
            add("  of which C4 grasses", df[(df["Genus"] == "Grass") & (df["Domain"] == "Grass C4")], tau=False)
    for t in TRANSECTS:
        add(t, df[df["Transect_full"] == t], tau=False)
    add("All samples", df, tau=False)
    t4 = pd.DataFrame(rows)
    _TABLES["TableS4_descriptive_statistics"] = t4
    return t4


def _table_s3():
    """Within-genus and all-sample Pearson statistics (GSP, GST vs five targets)."""
    TNc, N15, CNc = "TN_pct_corrected", "d15N_corrected", "CN_corrected"
    targets = {"TN": TNc, "d15N": N15, "d13C": "d13C", "TOC": "TOC_pct", "C/N": CNc}
    rows = []
    groups = [(g, df[df["Genus"] == g]) for g in ["Rhododendron", "Pinus", "Abies", "Juniperus", "Grass", "Dung"]]
    groups.append(("All samples", df))
    for gname, sub0 in groups:
        if len(sub0) < 5:
            continue
        for drv, dlab in (("GSP_mm", "GSP"), ("GST_K", "GST")):
            for tg, col in targets.items():
                s = sub0[[drv, col]].dropna()
                if len(s) < 5:
                    continue
                r, p = pearsonr(s[drv], s[col])
                rows.append(dict(Group=gname, Driver=dlab, Target=tg, n=len(s),
                                 r=round(r, 3), R2=round(r * r, 3), p=p))
    t = pd.DataFrame(rows)
    _TABLES["TableS3_within_genus"] = t
    return t


# ============================================================================
# FIG S2 + TABLE S2  — Pearson correlation heatmaps and tables
#   Top row: inter-variable correlations among δ13C, TOC, δ15N and TN per
#            transect and for all plants (dung excluded).  Only the lower
#            triangle of the symmetric matrix is shown.  Written to the
#            combined workbook as sheet "TableS2_inter-variable".
#   Bottom row: driver × target correlations (GSP, GST, MAT, MAP, ST, SM,
#            Elevation vs δ15N, δ13C, TOC, TN, C/N) per transect and for all
#            plants (dung excluded); sheet "FigS2_driver-target".
#   NOTE: TN / C/N exclusion codes (−10 / −5) were already replaced with NaN
#   at load time (see the LOAD section); _s2_pairs() additionally filters to
#   positive values as a second line of defence.
# ============================================================================

_INTER_LABELS = [
    ("d13C",             "δ13C", r"$\delta^{13}$C (‰)"),
    ("TOC_pct",          "TOC",  "TOC (%)"),
    ("d15N_corrected",   "δ15N", r"$\delta^{15}$N (‰)"),
    ("TN_pct_corrected", "TN",   "TN (%)"),
]

# Mathtext tick labels for the figure (tables keep the plain unicode labels)
_INTER_TICKS     = [r"$\delta^{13}$C", "TOC", r"$\delta^{15}$N", "TN"]
_S2_TARGET_TICKS = [r"$\delta^{15}$N", r"$\delta^{13}$C", "TOC", "TN", "C/N"]

_S2_DRIVERS = [
    ("GSP_mm", "GSP"), ("GST_K", "GST"), ("MAT_K", "MAT"), ("MAP_mm", "MAP"),
    ("ST_K", "ST"), ("SM_kg_m2", "SM"), ("Elevation_m", "Elevation"),
]

_S2_TARGETS = [
    ("d15N_corrected",   "δ15N", "δ15N (‰, AIR)"),
    ("d13C",             "δ13C", "δ13C (‰, VPDB)"),
    ("TOC_pct",          "TOC",  "TOC (%)"),
    ("TN_pct_corrected", "TN",   "TN (%)"),
    ("CN_corrected",     "C/N",  "C/N"),
]


def _s2_pairs(frame, cols):
    """Usable rows for a correlation; TN/C/N negative codes are filtered out."""
    s = frame[cols].dropna()
    for c in cols:
        if c in ("TN_pct_corrected", "CN_corrected"):
            s = s[s[c] > 0]
    return s


def _table_s2_inter_variable():
    """Table S2: Pearson r among δ13C, TOC, δ15N and TN; dung excluded."""
    plants = df[df["Genus"] != "Dung"]
    groups = [
        ("Karnali",       plants[plants["Transect_full"] == TRANSECTS[0]]),
        ("Annapurna",     plants[plants["Transect_full"] == TRANSECTS[1]]),
        ("Kosi",          plants[plants["Transect_full"] == TRANSECTS[2]]),
        ("All (plants)",  plants),
    ]
    rows = []
    for gname, g in groups:
        for (c1, p1, _), (c2, p2, _) in combinations(_INTER_LABELS, 2):
            s = _s2_pairs(g, [c1, c2])
            if len(s) < 3 or s[c1].std() == 0 or s[c2].std() == 0:
                continue
            r, p = pearsonr(s[c1], s[c2])
            rows.append({
                "Group": gname, "Variable 1": p1, "Variable 2": p2, "n": len(s),
                "Pearson r": round(r, 3), "R²": round(r * r, 3), "p-value": p,
            })
    tab = pd.DataFrame(rows)
    _TABLES["TableS2_inter-variable"] = tab
    return tab


def _table_s2_driver_target():
    """Companion table for Fig S2: Pearson r for every driver × target pair."""
    plants = df[df["Genus"] != "Dung"]
    groups = [
        ("Karnali",                 plants[plants["Transect_full"] == TRANSECTS[0]]),
        ("Annapurna",               plants[plants["Transect_full"] == TRANSECTS[1]]),
        ("Kosi",                    plants[plants["Transect_full"] == TRANSECTS[2]]),
        ("Himalayas (all plants)",  plants),
    ]
    rows = []
    for gname, g in groups:
        for dcol, dlab in _S2_DRIVERS:
            for tcol, tshort, tlong in _S2_TARGETS:
                s = _s2_pairs(g, [dcol, tcol])
                if len(s) < 5 or s[dcol].std() == 0 or s[tcol].std() == 0:
                    continue
                r, p = pearsonr(s[dcol], s[tcol])
                rows.append({
                    "Group": gname, "Driver": dlab, "Target": tlong, "n": len(s),
                    "Pearson r": round(r, 3), "R²": round(r * r, 3), "p-value": p,
                })
    tab = pd.DataFrame(rows)
    _TABLES["FigS2_driver-target"] = tab
    return tab


def _s2_heatmap_panel(ax, mat, xticks, yticks_left):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    if yticks_left is not None:
        ax.set_yticks(range(len(yticks_left)))
        ax.set_yticklabels(yticks_left)
    else:
        ax.set_yticks([])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > 0.55 else "black")
    return im


def make_figS2_heatmap():
    """Fig S2: inter-variable + driver×target Pearson heatmaps, plus tables."""
    print("Fig S2 (Pearson heatmaps: inter-variable + drivers × targets) …")
    tab_inter = _table_s2_inter_variable()
    tab_dt = _table_s2_driver_target()

    inter_groups = [
        ("Karnali",      df[(df["Transect_full"] == TRANSECTS[0]) & (df["Genus"] != "Dung")]),
        ("Annapurna",    df[(df["Transect_full"] == TRANSECTS[1]) & (df["Genus"] != "Dung")]),
        ("Kosi",         df[(df["Transect_full"] == TRANSECTS[2]) & (df["Genus"] != "Dung")]),
        ("All (plants)", df[df["Genus"] != "Dung"]),
    ]
    dt_groups = [
        ("Karnali",      df[(df["Transect_full"] == TRANSECTS[0]) & (df["Genus"] != "Dung")]),
        ("Annapurna",    df[(df["Transect_full"] == TRANSECTS[1]) & (df["Genus"] != "Dung")]),
        ("Kosi",         df[(df["Transect_full"] == TRANSECTS[2]) & (df["Genus"] != "Dung")]),
        ("All (plants)", df[df["Genus"] != "Dung"]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(19, 10))

    for col, (gname, g) in enumerate(inter_groups):
        ax = axes[0, col]
        mat = np.full((len(_INTER_LABELS), len(_INTER_LABELS)), np.nan)
        for i, (ci, _, _) in enumerate(_INTER_LABELS):
            for j, (cj, _, _) in enumerate(_INTER_LABELS):
                if i == j:
                    mat[i, j] = 1.0
                elif i > j:      # lower triangle only; upper stays blank
                    s = _s2_pairs(g, [ci, cj])
                    if len(s) >= 3 and s[ci].std() > 0 and s[cj].std() > 0:
                        mat[i, j] = pearsonr(s[ci], s[cj])[0]
        im = _s2_heatmap_panel(ax, mat, _INTER_TICKS, _INTER_TICKS if col == 0 else None)
        ax.set_title(f"{gname} (n = {len(g)} plants)", fontweight="bold")
        if col == 0:
            ax.set_ylabel("Inter-variable Pearson r")
    fig.colorbar(im, ax=axes[0, :], shrink=0.85, pad=0.02, label="Pearson r")

    for col, (gname, g) in enumerate(dt_groups):
        ax = axes[1, col]
        mat = np.full((len(_S2_TARGETS), len(_S2_DRIVERS)), np.nan)
        for i, (tc, _, _) in enumerate(_S2_TARGETS):
            for j, (dc, _) in enumerate(_S2_DRIVERS):
                s = _s2_pairs(g, [dc, tc])
                if len(s) >= 5 and s[dc].std() > 0 and s[tc].std() > 0:
                    mat[i, j] = pearsonr(s[dc], s[tc])[0]
        im = _s2_heatmap_panel(ax, mat, [lab for _, lab in _S2_DRIVERS],
                               _S2_TARGET_TICKS if col == 0 else None)
        ax.set_title(f"{gname} (n = {len(g)} plants)", fontweight="bold")
        if col == 0:
            ax.set_ylabel("Driver × target Pearson r")
    fig.colorbar(im, ax=axes[1, :], shrink=0.85, pad=0.02, label="Pearson r")

    fig.suptitle(
        "Fig. S2 — Pearson correlations, plants only (dung excluded).  "
        "Top: inter-variable (" + r"$\delta^{13}$C, TOC, $\delta^{15}$N, TN"
        "; lower triangle).  Bottom: climate/soil drivers vs foliar targets.  "
        "Pairs involving TN or C/N use positive values only.",
        y=0.995,
    )
    style_figure(fig)
    fig.savefig(OUTPUT_DIR / "FigS2_pearson_heatmap.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✓ saved")
    print_caption("figS2")
    print(f"  Tables registered: TableS2_inter-variable ({len(tab_inter)} rows), "
          f"FigS2_driver-target ({len(tab_dt)} rows)")


# ============================================================================
# FIG S6  — elevation coverage and within-transect δ15N–elevation profiles
#   (a) Elevation coverage by genus and tissue type (n = 141); point colour
#       indicates transect (red: Western/Karnali; blue: Central/Annapurna;
#       green: Eastern/Kosi); black ticks mark genus mean elevations.
#   (b) Within-transect foliar δ15N–elevation profiles with LOESS smoothers
#       (span 0.75).  Foliar = dung excluded.
#   Colour note: the manuscript Fig. S6 caption uses red/blue/green, which
#   differs from TRANSECT_COLORS used elsewhere in this script; S6 follows
#   the caption.
# ============================================================================

S6_COLORS = {
    "Western Himalaya (Karnali)":   "#DC143C",   # red
    "Central Himalaya (Annapurna)": "#4169E1",   # blue
    "Eastern Himalaya (Kosi)":      "#2E8B57",   # green
}


def _loess(xs, ys, span=0.75, degree=2, n_grid=80):
    """LOESS smoother (local polynomial, tricube weights; R loess, span 0.75)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    order = np.argsort(xs)
    x0, y0 = xs[order], ys[order]
    n = len(x0)
    k = max(degree + 1, int(np.ceil(span * n)))
    grid = np.linspace(x0.min(), x0.max(), n_grid)
    out = np.empty(n_grid)
    for gi, gx in enumerate(grid):
        d = np.abs(x0 - gx)
        idx = np.argsort(d)[:k]
        h = max(float(d[idx].max()), 1e-12)
        u = (x0[idx] - gx) / h          # normalised offsets: well-conditioned basis
        w = (1 - np.abs(u) ** 3) ** 3   # tricube weights
        Xl = np.vander(u, degree + 1, increasing=True)
        beta = np.linalg.lstsq(Xl * w[:, None], y0[idx] * w, rcond=None)[0]
        out[gi] = float(beta[0])        # intercept = smoothed value at gx
    return grid, out


def make_figS6():
    """Fig S6: (a) elevation coverage, (b) δ15N–elevation LOESS profiles."""
    print("Fig S6 (elevation coverage + δ15N–elevation profiles) …")
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )

    # ---- (a) elevation coverage by genus and tissue type -------------------
    groups = ["Rhododendron", "Pinus", "Abies", "Juniperus", "Grass", "Dung"]
    yticks = np.arange(len(groups))[::-1]          # Rhododendron on top
    first = True
    for gi, genus in zip(yticks, groups):
        sub = df[df["Genus"] == genus]
        # small deterministic jitter so overlapping points stay visible
        rng = np.random.default_rng(len(sub) + 1)
        jitter = (rng.random(len(sub)) - 0.5) * 0.22
        for transect in TRANSECTS:
            mask = (sub["Transect_full"].to_numpy() == transect)
            if mask.any():
                ax_a.scatter(
                    sub["Elevation_m"].to_numpy()[mask], gi + jitter[mask],
                    s=34, color=S6_COLORS[transect], alpha=0.85,
                    edgecolors="black", linewidth=0.4, zorder=3,
                    label=f"{TRANSECT_LABELS[transect]} ({transect.split()[0]})"
                    if first else None,
                )
                first = False
        mean_elev = float(sub["Elevation_m"].mean())
        ax_a.plot([mean_elev, mean_elev], [gi - 0.30, gi + 0.30],
                  color="black", lw=2.2, zorder=4)
        ax_a.annotate(f"{mean_elev:,.0f} m", xy=(mean_elev, gi + 0.36),
                      ha="center", va="bottom")
    ax_a.set_yticks(yticks)
    ax_a.set_yticklabels(groups)
    ax_a.set_xlabel("Elevation (m)", fontweight="bold")
    ax_a.set_title("(a) Elevation coverage by genus and tissue type "
                   f"(n = {len(df)})", loc="left", fontweight="bold")
    ax_a.set_ylim(-0.7, len(groups) - 0.1)
    ax_a.grid(True, axis="x", alpha=0.2)
    handles = [mpatches.Patch(facecolor=S6_COLORS[t],
                              label=f"{TRANSECT_LABELS[t]} ({t.split()[0]})")
              for t in TRANSECTS]
    handles.append(Line2D([0], [0], color="black", lw=2.2, label="genus mean"))
    ax_a.legend(handles=handles, loc="lower right", frameon=True)
    ax_a.text(0.0, -0.16,
              "All grass and dung samples are from the Western transect;\n"
              "no Pinus was sampled in the Eastern transect.",
              transform=ax_a.transAxes, va="top")

    # ---- (b) within-transect foliar δ15N–elevation profiles ----------------
    plants6 = df[df["Genus"] != "Dung"]           # foliar samples only
    rows = []
    for transect in TRANSECTS:
        sub = plants6[plants6["Transect_full"] == transect]
        x = sub["Elevation_m"].to_numpy(float)
        y = sub["d15N_corrected"].to_numpy(float)
        r, p = pearsonr(x, y)
        rows.append({"Transect": TRANSECT_LABELS[transect], "n": len(sub),
                     "Pearson r (foliar)": round(r, 3), "p-value": p})
        ax_b.scatter(x, y, s=30, color=S6_COLORS[transect], alpha=0.8,
                     edgecolors="black", linewidth=0.4, zorder=3,
                     label=f"{TRANSECT_LABELS[transect]}: r = {r:+.2f} (p = {p:.3f})")
        gx, gy = _loess(x, y, span=0.75)
        ax_b.plot(gx, gy, color=S6_COLORS[transect], lw=2.0, zorder=2)
    ax_b.set_xlabel("Elevation (m)", fontweight="bold")
    ax_b.set_ylabel(r"$\delta^{15}$N (‰, AIR), corrected", fontweight="bold")
    ax_b.set_title(r"(b) Within-transect foliar $\delta^{15}$N–elevation profiles"
                   "\n(LOESS smoothers, span = 0.75)", loc="left",
                   fontweight="bold")
    ax_b.legend(title="Transect (Pearson r, p)", loc="upper right", frameon=True)
    ax_b.grid(True, alpha=0.2)

    _TABLES["FigS6_d15N_elevation"] = pd.DataFrame(rows)
    # Console note: the Western r quoted in the previous manuscript caption
    # (−0.07, p = 0.59) matches the dung-INCLUDED computation; the foliar
    # value reported here excludes the six dung samples.
    foliar = {row["Transect"]: row["Pearson r (foliar)"] for row in rows}
    for transect in TRANSECTS:
        sub_all = df[df["Transect_full"] == transect]
        r_all, p_all = pearsonr(sub_all["Elevation_m"], sub_all["d15N_corrected"])
        print(f"  {TRANSECT_LABELS[transect]}: foliar r = {foliar[TRANSECT_LABELS[transect]]:+.3f}"
              f"   (all samples incl. dung: r = {r_all:+.3f}, p = {p_all:.3f})")

    fig.tight_layout()
    style_figure(fig)
    fig.savefig(OUTPUT_DIR / "FigS6_elevation_coverage.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✓ saved")
    print_caption("figS6")


# ============================================================================
# COMBINED WORKBOOK  — one Excel file with every table produced by the script
# ============================================================================

# ============================================================================
# SHEET 'S1'  — the analysis dataset, exported to the combined workbook with
# plain column names (no 'corrected'): every column the code actually uses,
# with the final (post-correction) N values and the SM column in kg m⁻².
# ============================================================================

_S1_EXPORT_COLUMNS = {
    "Transect_full":     "Transect",
    "Genus":             "Genus",
    "Domain":            "Domain (C3/C4)",
    "Mycorrhizal":       "Mycorrhizal association",
    "d13C":              "δ13C (‰, VPDB)",
    "TOC_pct":           "TOC (%)",
    "d15N_corrected":    "δ15N (‰, AIR)",
    "TN_pct_corrected":  "TN (%)",
    "CN_corrected":      "C/N",
    "Elevation_m":       "Elevation (m)",
    "GSP_mm":            "Precipitation GSP (mm month⁻¹)",
    "GST_K":             "Surface Temperature GST (K)",
    "SM_kg_m2":          "Soil Moisture (kg m⁻²)",
    "ST_K":              "Soil Temperature (K)",
    "SMN_gN_m2":         "Soil Microbial N (gN m⁻²)",
    "MAP_mm":            "Precipitation Annual (mm month⁻¹)",
    "MAT_K":             "Surface Temperature Annual (K)",
    "is_C4":             "C4 indicator (1 = C4 grass)",
}


def _table_s1_data():
    """Sheet 'S1': the analysis dataset with plain column names (no 'corrected').

    Dung rows are included (Genus = 'Dung'); rows whose TN/C/N were excluded
    from the statistics (former negative sentinel codes) carry blanks there
    because the codes are replaced with NaN at load time.
    """
    return df.rename(columns=_S1_EXPORT_COLUMNS)[list(_S1_EXPORT_COLUMNS.values())].copy()


_WORKBOOK_SHEET_ORDER = [
    "S1",
    "TableS1_pairwise_kmeans",
    "TableS2_inter-variable",
    "FigS2_driver-target",
    "TableS3_within_genus",
    "TableS4_descriptive_statistics",
    "Fig3_model_statistics",
    "FigS6_d15N_elevation",
]


def _write_combined_workbook():
    """Write every collected table to ONE Excel workbook (no separate CSVs)."""
    if not _TABLES:
        print("No tables collected; combined workbook not written.")
        return
    ordered = [k for k in _WORKBOOK_SHEET_ORDER if k in _TABLES]
    ordered += [k for k in _TABLES if k not in ordered]
    path = OUTPUT_DIR / "Supplementary_Tables_combined.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet in ordered:
            _TABLES[sheet].to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"\nCombined workbook saved: {path.name}  ({len(ordered)} sheets)")
    print(f"  sheets: {', '.join(ordered)}")


# ============================================================================
# REPORT  — values for direct comparison with the manuscript text
#   "ms" columns hold the numbers currently printed in the manuscript.
#   Everything else is computed here from the workbook (corrected N columns).
# ============================================================================

def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _pp(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def _report_extra(out, head):
    S = _stats_bundle()

    def dunn_lines(block, label):
        out(f"{label}: Kruskal-Wallis H = {block['H']:.1f}, p {_pp(block['p'])}")
        out("  n: " + ", ".join(f"{k} {v}" for k, v in block["n"].items()))
        out("  means: " + ", ".join(f"{k} {v:.2f}" for k, v in block["mean"].items()))
        for (a, b), p in block["dunn"].items():
            out(f"  Dunn (Bonferroni) {a} vs {b}: p {_pp(p)}")

    head("ADDITIONAL TESTS FOR SECTIONS 3.1-3.2 (computed)")
    dunn_lines(S["tn_transect"], "TN by transect")
    dunn_lines(S["tn_genus"], "TN by genus (Rhododendron, Pinus, Abies, Juniperus, Grass)")
    dunn_lines(S["d15n_transect"], "d15N by transect")

    out("\nPearson r, GSP and GST vs TN and d15N, within transect and for all samples:")
    for lab in list(TRANSECT_LABELS.values()) + ["ALL"]:
        out(f"  {lab:4s} " + "  ".join(f"{drv[:3]}-{tg} {S['wt'][(lab, drv, tg)]:6.2f}"
                                         for drv in ("GSP_mm", "GST_K") for tg in ("TN", "d15N")))

    out(f"\nd13C of C3 plants (dung and C4 grasses excluded): mean {S['d13c_c3_mean']:.1f} permil; by transect "
        + ", ".join(f"{k} {v:.1f}" for k, v in S["d13c_c3_by_transect"].items()))
    out("d13C vs elevation within transect (C3 plants, Pearson):")
    for lab, (r, p, n) in S["d13c_elev"].items():
        out(f"  {lab:4s} r = {r:5.2f}  p = {_p(p)}  n = {n}")

    out("\nShapiro-Wilk normality (W, p):")
    for k, (w, p) in S["shapiro"].items():
        out(f"  {k:12s} W = {w:.3f}  p = {_p(p)}")

    ng = S["neg"]
    out(f"\nNegative d15N: {ng['n']} samples ({ng['share']*100:.0f}%); genus {ng['genus']}; "
        f"transect {ng['transect']}; elevation mean {ng['elev_mean']:.0f} m "
        f"(min {ng['elev_min']:.0f}, max {ng['elev_max']:.0f})")

    out("\nMean values within the elevation bands used in Section 4.4:")
    for k, b in S["bands"].items():
        out(f"  {k:34s} n = {b['n']:3d}  d15N {b['d15N']:5.1f} ({b['d15N_min']:.1f} to {b['d15N_max']:.1f})  "
            f"d13C {b['d13C']:6.1f}  TOC {b['TOC']:.1f}")
    out("\nd15N by elevation bin (mean, n):")
    for lab, tb in S["d15n_bins"].items():
        out(f"  {lab}: " + "; ".join(f"{i.left:.0f}-{i.right:.0f} m {r['mean']:.1f} (n={int(r['count'])})"
                                      for i, r in tb.iterrows()))
    out("\nSampled elevation by genus (min / mean / max, m):")
    out(S["genus_elev"].round(0).to_string())

    head("FIG 2: MYCORRHIZAL CLUSTERING (exhaustive predictor × model search, computed)")
    if _FIG2_SEARCH:
        _sel, _res = _FIG2_SEARCH["selection"], _FIG2_SEARCH["result"]
        _m, _row = _res["metrics"], _sel["row"]
        out(f"n = {_FIG2_SEARCH['n']} samples (dung and rows with non-physical TN or C/N excluded)")
        out(_sel["reason"])
        out(f"Selected for Fig 2: {_row['best_model']} on {_row['features']}")
        out(f"ARI = {_m['ari']:.3f}, NMI = {_m['nmi']:.3f}, "
            f"V-measure = {_m['v']:.3f}, purity = {_m['purity']:.3f}")
        out("Cluster × association counts (clusters numbered along the plotted x-axis):")
        out(_res["crosstab"].to_string())
        out("Full model-score comparison: FigS_clustering_comparison.png")
    else:
        out("(make_fig2() has not been run in this session)")

    L_ = S["lda"]; cv = L_["cv"]
    head("MYCORRHIZAL LDA + K-MEANS: VALIDATION (computed)")
    out(f"n = {len(L_['y'])} (dung excluded); LD1 {L_['ev'][0]*100:.1f}%, LD2 {L_['ev'][1]*100:.1f}% of between-group variance")
    out(f"K-means in LDA space (in-sample, supervised): ARI = {L_['ari_km']:.3f}, Silhouette = {L_['sil_km']:.3f}")
    out(f"5-fold cross-validated LDA classification:    accuracy = {cv['acc']:.3f} "
        f"(largest-class baseline {cv['base']:.3f}); ARI of predictions = {cv['ari']:.3f}")
    out(f"One-to-one cluster-to-class matching accuracy = {cv['match_acc']:.3f}")
    out("Per-class recall in that matching: " + ", ".join(f"{k} {v:.2f}" for k, v in cv["recall"].items()))
    out("Cluster composition (rows = true class, columns = cluster):")
    out(cv["crosstab"].to_string())
    out("Samples of each class in the cluster with the most samples of that class (of class total):")
    for k, (m, tot, ci) in cv["modal"].items():
        out(f"  {k:22s} {m} of {tot} (cluster {ci}); in a cluster whose majority class is its own: {cv['in_own'][k]}")

    if _ISO_CACHE:
        head("ISOSCAPE MODELS: HELD-OUT SKILL AND WITHIN-TRANSECT AGREEMENT (computed)")
        for nm, ho, wr, st in (("d15N", _ISO_CACHE["ho15"], _ISO_CACHE["wr15"], _ISO_CACHE["st15"]),
                               ("d13C", _ISO_CACHE["ho13"], _ISO_CACHE["wr13"], _ISO_CACHE["st13"])):
            out(f"{nm}: in-sample R2 = {st['r2']:.3f}, RMSE = {st['rmse']:.2f}; held-out (n = {ho['n']}) "
                f"R2 = {ho['r2']:.3f}, RMSE = {ho['rmse']:.2f}")
            out("   within-transect Pearson r between fitted and observed values: "
                + ", ".join(f"{k} {v:.2f}" for k, v in wr.items()))

    head("TABLE S3 (also written to the combined workbook, sheet TableS3_within_genus)")
    t3 = _table_s3()
    out(t3.assign(p=t3["p"].map(_p)).to_string(index=False))


def make_report():
    L = []
    def out(x=""):
        L.append(x)
    def head(x):
        out(); out("=" * 78); out(x); out("=" * 78)

    TNc, N15, CNc = "TN_pct_corrected", "d15N_corrected", "CN_corrected"
    MAIN = ["Rhododendron", "Pinus", "Abies", "Juniperus", "Grass", "Dung"]

    # ── 3.1 ────────────────────────────────────────────────────────────────
    head("SECTION 3.1 — foliar C and N concentrations and isotopes (computed)")
    out(f"n total = {len(df)}")
    out("n by genus: " + ", ".join(f"{g} {n}" for g, n in df['Genus'].value_counts().items())
        + "      [ms: Juniperus 39, Rhododendron 46, Pinus 30, Abies 10, grasses 10, dung 6; total 141]")
    out("\nn by genus x transect:")
    out(pd.crosstab(df["Genus"], df["Transect_full"]).to_string())
    out("\nSampled elevation range by transect (m)   [ms: 1,900-4,200 / 2,000-4,800 / 2,600-5,200]")
    for t in TRANSECTS:
        e = df.loc[df["Transect_full"] == t, "Elevation_m"]
        out(f"  {t:32s} {e.min():7.0f} - {e.max():.0f}")

    out(f"\nMean TN (all)            {df[TNc].mean():.2f} ± {df[TNc].std():.2f} %     [ms: 1.56 ± 0.93 (all samples, dung included)]")
    ms_tn = {"Juniperus": 1.49, "Abies": 1.45, "Rhododendron": 1.91, "Pinus": 1.50, "Grass": 0.86}
    out("Mean TN by genus (%):")
    for g in MAIN:
        v = df.loc[df["Genus"] == g, TNc]
        out(f"  {g:14s} {v.mean():5.2f} ± {v.std():4.2f} (n={v.notna().sum()})"
            + (f"     [ms: {ms_tn[g]:.2f}]" if g in ms_tn else ""))

    ok = df[[TNc, N15]].dropna()
    r, pr = pearsonr(ok[TNc], ok[N15]); tau, pt = kendalltau(ok[TNc], ok[N15])
    out(f"\nTN vs d15N: Pearson r = {r:.2f} (p {_p(pr)});  Kendall tau = {tau:.2f} (p {_p(pt)})   [ms: r = -0.22, p = 0.008; tau = -0.16, p = 0.005]")
    out(f"d15N range {df[N15].min():.1f} to {df[N15].max():.1f} permil, mean {df[N15].mean():.1f}   [ms: -4.4 to +14.2, mean +4.3]")

    out(f"\nMean TOC (all)           {df['TOC_pct'].mean():.1f} ± {df['TOC_pct'].std():.1f} %     [ms: 37.5 ± 6.2]")
    ms_toc = {"Juniperus": 40.6, "Abies": 34.2}
    out("Mean TOC by genus (%):")
    for g in MAIN:
        v = df.loc[df["Genus"] == g, "TOC_pct"]
        out(f"  {g:14s} {v.mean():5.1f} ± {v.std():4.1f}" + (f"     [ms: {ms_toc[g]}]" if g in ms_toc else ""))

    c3 = df[~df["Domain"].isin(["Grass C4", "Dung"])]["d13C"]
    c3d = df[df["Domain"] != "Grass C4"]["d13C"]
    c4 = df[df["Domain"] == "Grass C4"]["d13C"]
    out(f"\nd13C C3 (excl. dung and C4)  {c3.min():.1f} to {c3.max():.1f}     [ms: -31.4 to -21.7]")
    out(f"d13C C3 (incl. dung)         {c3d.min():.1f} to {c3d.max():.1f}")
    out(f"d13C C4 grasses              {c4.min():.1f} to {c4.max():.1f}     [ms: -14.2 to -13.0]")

    out("\nTransect means (TN %, TOC %)   [ms TN: 1.26 ± 0.51 / 1.55 ± 0.81 / 2.02 ± 1.26; ms TOC: 37.5 / 40.3 / 35.4]")
    for t in TRANSECTS:
        g = df[df["Transect_full"] == t]
        out(f"  {t:32s} TN {g[TNc].mean():.2f} ± {g[TNc].std():.2f}   TOC {g['TOC_pct'].mean():.1f}")
    out("\nTN vs elevation within transect (Pearson)   [ms: not significant within any transect, all p >= 0.495]")
    for t in TRANSECTS:
        g = df.loc[df["Transect_full"] == t, [TNc, "Elevation_m"]].dropna()
        r, p_ = pearsonr(g[TNc], g["Elevation_m"])
        out(f"  {TRANSECT_LABELS[t]:4s} r = {r:5.2f}  p = {_p(p_)}  n = {len(g)}")

    out("\nKendall tau, d15N vs TOC within genus   [ms: none significant (all p >= 0.069)]")
    for g in MAIN:
        sub = df[df["Genus"] == g][[N15, "TOC_pct"]].dropna()
        k, pk = kendalltau(sub[N15], sub["TOC_pct"])
        out(f"  {g:14s} tau = {k:5.2f}  p = {_p(pk)}  n = {len(sub)}")
    a = df[df["Genus"] == "Abies"][["d13C", N15]].dropna()
    r, p_ = pearsonr(a["d13C"], a[N15])
    out(f"\nAbies d13C vs d15N: r = {r:.2f}, p = {p_:.3f}, n = {len(a)}   [ms: r = -0.61, p = 0.063]")

    # ── 3.2 ────────────────────────────────────────────────────────────────
    head("SECTION 3.2 — GSP (Precipitation GSP) and GST (Surface Temperature GST), within genus")
    targets = {"TN": TNc, "d15N": N15, "d13C": "d13C", "TOC": "TOC_pct", "C/N": CNc}
    ms32 = [  # genus, driver, target, ms r, ms R2  (values as printed in the revised manuscript)
        ("Pinus",        "GSP_mm", "TN",    0.71, 0.50),
        ("Pinus",        "GST_K",  "TN",   -0.55, 0.30),
        ("Abies",        "GSP_mm", "d13C",  0.79, 0.63),
        ("Juniperus",    "GST_K",  "TOC",   0.60, 0.36),
        ("Rhododendron", "GSP_mm", "d15N", -0.86, 0.74),
        ("Pinus",        "GSP_mm", "d15N", -0.79, 0.62),
        ("Abies",        "GSP_mm", "d15N", -0.82, 0.67),
        ("Juniperus",    "GSP_mm", "d15N", -0.81, 0.66),
    ]
    out("Statements made in the text (TN, d15N use corrected values):")
    out(f"  {'genus':13s} {'driver':7s} {'target':5s} {'n':>3s} {'r':>7s} {'R2':>6s} {'p':>7s}")
    for g, drv, tg, mr, mr2 in ms32:
        sub = df[df["Genus"] == g][[drv, targets[tg]]].dropna()
        r, p_ = pearsonr(sub[drv], sub[targets[tg]])
        out(f"  {g:13s} {drv[:3]:7s} {tg:5s} {len(sub):3d} {r:7.2f} {r*r:6.2f} {_p(p_):>7s}")

    out("\nFull table, all genera with n >= 5 (r / R2 / p):")
    out(f"  {'genus':13s} {'driver':6s} " + " ".join(f"{t:>20s}" for t in targets))
    for g in MAIN:
        sub0 = df[df["Genus"] == g]
        if len(sub0) < 5:
            continue
        for drv in ("GSP_mm", "GST_K"):
            cells = []
            for tg, col in targets.items():
                sub = sub0[[drv, col]].dropna()
                r, p_ = pearsonr(sub[drv], sub[col])
                cells.append(f"{r:6.2f}/{r*r:4.2f}/{_p(p_):>6s}")
            out(f"  {g:13s} {drv[:3]:6s} " + " ".join(f"{c:>20s}" for c in cells))

    out("\nAll samples, R2 of GSP and GST with each target   [ms: GSP explains 66% of d15N variance; < 5% for TN, TOC and d13C]")
    for tg, col in targets.items():
        sub = df[["GSP_mm", "GST_K", col]].dropna()
        out(f"  {tg:5s} GSP R2 = {pearsonr(sub['GSP_mm'], sub[col])[0]**2:.3f}   "
            f"GST R2 = {pearsonr(sub['GST_K'], sub[col])[0]**2:.3f}")

    # ── 3.3 ────────────────────────────────────────────────────────────────
    head("SECTION 3.3 — K-means (k = 4), Silhouette Score and Adjusted Rand Index")
    MYCO = ["Juniperus (arbuscular)", "Pinus, Abies (ectomycorrhizal)", "Grasses (arbuscular)", "Rhododendron (ericoid)"]
    fc = {"δ13C": "d13C", "TOC (%)": "TOC_pct", "TN (%)": TNc,
          "δ15N": N15, "C/N": CNc}
    rows = []
    for f1, f2 in combinations(fc, 2):
        sub = df[[fc[f1], fc[f2], "Mycorrhizal"]].dropna()
        sub = sub[sub["Mycorrhizal"].isin(MYCO)]
        Xs = StandardScaler().fit_transform(sub[[fc[f1], fc[f2]]])
        cl = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(Xs)
        rows.append((f1, f2, silhouette_score(Xs, cl),
                     adjusted_rand_score(LabelEncoder().fit_transform(sub["Mycorrhizal"]), cl),
                     len(sub)))
    tab = pd.DataFrame(rows, columns=["Feature 1", "Feature 2", "Silhouette Score",
                                      "Adjusted Rand Index", "n"])
    _TABLES["TableS1_pairwise_kmeans"] = tab
    out(tab.sort_values("Adjusted Rand Index", ascending=False)
           .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    h = tab[(tab["Feature 1"] == "δ13C") & (tab["Feature 2"] == "TN (%)")].iloc[0]
    out(f"\nδ13C–TN: Silhouette {h['Silhouette Score']:.3f}, ARI {h['Adjusted Rand Index']:.3f}"
        "     [ms text: all pairwise ARI ≤ 0.2; best two-variable combinations ≈ 0.11–0.19]")

    # ── 3.4 ────────────────────────────────────────────────────────────────
    head("SECTION 3.4 — isoscape models")
    soil_ok = not (df["SM_kg_m2"].isna().all() or df["SMN_gN_m2"].isna().all())
    if ISOSCAPE_SOIL_COVARIATES and not soil_ok:
        out("SM and/or SMN missing from sheet S1: isoscape equations not computed.")
    else:
        out("Predictors: GST, GSP" + (", SM, SMN" if ISOSCAPE_SOIL_COVARIATES else "")
            + " (d15N); GST, GSP, elevation, C4 indicator"
            + (", SMN" if ISOSCAPE_SOIL_COVARIATES else "") + " (d13C; no SM)")
        d15p = ["GST_K", "GSP_mm", "SM_kg_m2", "SMN_gN_m2"] if ISOSCAPE_SOIL_COVARIATES else ["GST_K", "GSP_mm"]
        d13p = (["GST_K", "GSP_mm", "Elevation_m", "SMN_gN_m2", "is_C4"]
                if ISOSCAPE_SOIL_COVARIATES else ["GST_K", "GSP_mm", "Elevation_m", "is_C4"])
        specs = [("d15N", N15, d15p, LinearRegression, "linear regression"),
                 ("d13C", "d13C", d13p, lambda: Ridge(alpha=1.0), "ridge (alpha = 1)")]
        for name, col, preds, mk, label in specs:
            d = df[["Transect", col] + preds].dropna()
            m = mk().fit(d[preds], d[col])
            out(f"\n{name}, {label}, fitted on all n = {len(d)}:")
            out("  " + _equation(name, m.intercept_, preds, m.coef_))
            st = _model_stats(d[col], m.predict(d[preds]))
            out(f"  in-sample: R2 = {st['r2']:.3f}  RMSE = {st['rmse']:.2f}  MAE = {st['mae']:.2f}  bias = {st['bias']:.3f}")
            tr, te = train_test_split(d, test_size=0.2, stratify=d["Transect"], random_state=42)
            mt = mk().fit(tr[preds], tr[col])
            st = _model_stats(te[col], mt.predict(te[preds]))
            out(f"  80:20 split stratified by transect (random_state = 42), train {len(tr)} / test {len(te)}:")
            out(f"  {_equation(name, mt.intercept_, preds, mt.coef_)}")
            out(f"  out-of-sample: R2 = {st['r2']:.3f}  RMSE = {st['rmse']:.2f}  MAE = {st['mae']:.2f}")
        if _ISO_CACHE.get("alt"):
            out("\nSensitivity to the soil-layer predictors (see Fig3_model_statistics sheet):")
            for r in _ISO_CACHE["alt"]:
                out(f"  {r['model']}: R2 = {r['r2']:.3f}, RMSE = {r['rmse']:.2f}" + (f"\n    {r['equation']}" if r["equation"] else ""))

    _report_extra(out, head)
    text = re.sub(r"[ \t]*\[ms[^\]]*\]", "", "\n".join(L))
    print(text)



# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Loaded {len(df)} samples from {XLSX_PATH.name}\n")
    make_fig1b_1c()
    if MAKE_CORRECTION_COMPARISON:
        make_fig_comparison()   # editorial record only; off by default
    make_fig_climatological_distinction()
    make_fig2()               # Fig 2: selected K-means solution
    make_figS2_heatmap()      # Fig S2 (inter-variable + driver×target) + Table S2
    make_figS6()              # Fig S6: elevation coverage + δ15N–elevation LOESS
    make_fig3()
    make_report()
    _TABLES["S1"] = _table_s1_data()
    _write_combined_workbook()  # ONE Excel workbook with every table

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    for f in sorted(
        list(OUTPUT_DIR.glob("*.png"))
        + list(OUTPUT_DIR.glob("*.xlsx"))
    ):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
