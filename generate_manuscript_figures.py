"""
Generate manuscript figures for the C and N allocation manuscript.

This script creates:
    - Fig1b_d13C_TOC_by_transect.png
    - Fig1c_d15N_TN_by_transect.png
    - Fig2_mycorrhizal_clustering.png
    - Fig3_isoscape_validation.png
    - Fig3_model_statistics.csv

Figure 3 reconstructs the manuscript analysis:
    - delta15N: linear multiple regression using GSP, GST, SM, and SMN.
    - delta13C: ridge regression using GSP, GST, elevation, SM, SMN, and C4 status.

The printed manuscript equations are not used as fixed calculators for Figure 3,
because applying the printed delta13C equation directly to the sample table gives a
much poorer fit than the reported 64%. Re-fitting the stated model structure to the
available data reproduces the manuscript-level accuracies.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import adjusted_rand_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
#plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10


# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = PROJECT_DIR / "CN Manuscript"

FULL_DATA_PATH = PROJECT_DIR / "Plant Bulk Isotope Composition.xlsx"
SUPPLEMENTARY_PATH = MANUSCRIPT_DIR / "C_N Allocation Supplementary Table.xlsx"

OUTPUT_DIR = PROJECT_DIR / "manuscript_figure_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

def load_data():
    full = pd.read_excel(FULL_DATA_PATH, sheet_name="Sheet1")
    supplementary = pd.read_excel(SUPPLEMENTARY_PATH, sheet_name="S1")

    # Supplementary S1 stores the manuscript's growing-season drivers and
    # rerun d15N/TN values, so no separate TN rerun workbook is needed here.
    supplementary_cols = [
        "ID",
        "Growing Season Precipitation (mm/m)",
        "Growing Season Temperature (K)",
        "d15 rerun",
        "TN% Orig rerun",
    ]
    df = full.merge(supplementary[supplementary_cols], on="ID", how="left")

    df["δ15N_original (‰)"] = df["δ15N (‰)"]
    df["TN_original (%)"] = df["TN (%)"]
    df["d13C"] = df["δ13C (‰)"]
    df["d15N"] = df["d15 rerun"].fillna(df["δ15N (‰)"])
    df["TOC_%"] = df["TOC (%)"]
    df["TN%_Final"] = df["TN% Orig rerun"].fillna(df["TN (%)"])

    # Make the manuscript-style column names use the rerun data for all
    # downstream analyses that still refer to these labels directly.
    df["δ15N (‰)"] = df["d15N"]
    df["TN (%)"] = df["TN%_Final"]

    df["GSP_mm"] = df["Growing Season Precipitation (mm/m)"]
    df["GST_K"] = df["Growing Season Temperature (K)"]
    df["SM_kg_m3"] = df["Soil Moisture (m3/m3)"] * 1000.0
    df["SMN_gN_m2"] = df["Soil Microbial Niotrogen (gN/m2)"]
    df["Elevation_m"] = df["Elevation (m)"]
    df["is_C4"] = df["Domain"].eq("Grass C4").astype(int)

    df["Genus"] = df.apply(normalize_genus, axis=1)
    return df


def normalize_genus(row):
    genus = row.get("Genus")
    species = row.get("Species")

    if pd.notna(genus):
        genus_str = str(genus)
        if genus_str.startswith("Grass"):
            return "Grass"
        return genus_str

    if pd.isna(species):
        return "Unknown"

    species_str = str(species)
    for name in ["Rhododendron", "Pinus", "Abies", "Juniperus", "Grass", "Dung"]:
        if name in species_str:
            return name
    return "Unidentified"


df = load_data()

TRANSECTS = [
    "Western Himalaya (Karnali)",
    "Central Himalaya (Annapurna)",
    "Eastern Himalaya (Kosi)",
]

TRANSECT_LABELS = {
    "Western Himalaya (Karnali)": "K",
    "Central Himalaya (Annapurna)": "AP",
    "Eastern Himalaya (Kosi)": "DKB",
}

TRANSECT_COLORS = {
    "Western Himalaya (Karnali)": "#32CD32",
    "Central Himalaya (Annapurna)": "#4169E1",
    "Eastern Himalaya (Kosi)": "#DC143C",
}

GENERA_TO_PLOT = ["Rhododendron", "Pinus", "Abies", "Juniperus"]

GENUS_COLORS = {
    "Rhododendron": "#3B7EA1",
    "Pinus": "#FDB462",
    "Abies": "#8FBC8F",
    "Juniperus": "#B03060",
    "Grass": "#808080",
    "Dung": "#A52A2A",
    "Unidentified": "#D3D3D3",
    "Unknown": "#D3D3D3",
}


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_violin_or_point(ax, values, position, color, width):
    values = pd.Series(values).dropna()
    if values.empty:
        return

    if len(values) >= 2:
        parts = ax.violinplot(
            [values],
            positions=[position],
            widths=width * 0.9,
            showmeans=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.70)
            pc.set_edgecolor("black")
            pc.set_linewidth(0.5)
    else:
        ax.scatter(
            [position],
            values,
            s=45,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )


def model_stats(observed, modelled):
    observed = np.asarray(observed, dtype=float)
    modelled = np.asarray(modelled, dtype=float)
    return {
        "n": len(observed),
        "r": np.corrcoef(observed, modelled)[0, 1],
        "r2": r2_score(observed, modelled),
        "rmse": mean_squared_error(observed, modelled) ** 0.5,
        "mae": mean_absolute_error(observed, modelled),
        "bias": np.mean(modelled - observed),
    }


def add_stats_box(ax, stats):
    ax.text(
        0.05,
        0.95,
        (
            f"n = {stats['n']}\n"
            f"R² = {stats['r2']:.3f}\n"
            f"r = {stats['r']:.3f}\n"
            f"RMSE = {stats['rmse']:.2f}‰\n"
            f"MAE = {stats['mae']:.2f}‰"
        ),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )


def format_model_equation(target, intercept, predictors, coefficients):
    terms = [f"{target}_modelled = {intercept:.6g}"]
    for predictor, coefficient in zip(predictors, coefficients):
        sign = "+" if coefficient >= 0 else "-"
        terms.append(f" {sign} {abs(coefficient):.6g}*{predictor}")
    return "".join(terms)


# ============================================================================
# FIGURE 1B AND 1C
# ============================================================================

def make_figure_1_panels():
    print("Generating Figure 1b and Figure 1c...")

    x_positions = np.arange(len(TRANSECTS))
    width = 0.15
    legend_elements = [
        Patch(facecolor=GENUS_COLORS[g], edgecolor="black", label=g, alpha=0.7)
        for g in GENERA_TO_PLOT
    ]

    # Figure 1b: d13C and TOC
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    for i, transect in enumerate(TRANSECTS):
        for j, genus in enumerate(GENERA_TO_PLOT):
            subset = df[(df["Transect"] == transect) & (df["Genus"] == genus)]
            position = x_positions[i] - 0.3 + j * width
            plot_violin_or_point(ax1, subset["d13C"], position, GENUS_COLORS[genus], width)

    ax2 = ax1.twinx()
    for i, transect in enumerate(TRANSECTS):
        for j, genus in enumerate(GENERA_TO_PLOT):
            subset = df[(df["Transect"] == transect) & (df["Genus"] == genus)]
            if len(subset) > 0:
                position = x_positions[i] - 0.3 + j * width
                ax2.bar(
                    position,
                    subset["TOC_%"].mean(),
                    width=width * 0.8,
                    alpha=0.3,
                    color=GENUS_COLORS[genus],
                    edgecolor="black",
                    linewidth=0.5,
                )

    ax1.set_ylabel("δ¹³C (‰, VPDB)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Transect", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(TRANSECTS)
    ax1.set_ylim(-34, -18)
    ax1.axhline(y=-24.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Mean TOC (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 60)
    ax1.grid(False)
    ax2.grid(False)
    ax1.legend(handles=legend_elements, loc="upper left", frameon=True, title="Genus", fontsize=9)
    plt.title("Distribution of δ¹³C values and TOC across the three transects", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig1b_d13C_TOC_by_transect.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 1c: d15N and TN
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    for i, transect in enumerate(TRANSECTS):
        for j, genus in enumerate(GENERA_TO_PLOT):
            subset = df[(df["Transect"] == transect) & (df["Genus"] == genus)]
            position = x_positions[i] - 0.3 + j * width
            plot_violin_or_point(ax1, subset["d15N"], position, GENUS_COLORS[genus], width)

    ax2 = ax1.twinx()
    for i, transect in enumerate(TRANSECTS):
        for j, genus in enumerate(GENERA_TO_PLOT):
            subset = df[(df["Transect"] == transect) & (df["Genus"] == genus)]
            if len(subset) > 0:
                position = x_positions[i] - 0.3 + j * width
                ax2.bar(
                    position,
                    subset["TN%_Final"].mean(),
                    width=width * 0.8,
                    alpha=0.3,
                    color=GENUS_COLORS[genus],
                    edgecolor="black",
                    linewidth=0.5,
                )

    ax1.set_ylabel("δ¹⁵N (‰, AIR)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Transect", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(TRANSECTS)
    ax1.set_ylim(-5, 15)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Mean TN (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 15)
    ax1.grid(False)
    ax2.grid(False)
    ax1.legend(handles=legend_elements, loc="upper left", frameon=True, title="Genus", fontsize=9)
    plt.title("Distribution of δ¹⁵N values and TN across the three transects", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig1c_d15N_TN_by_transect.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# FIGURE 2
# ============================================================================

def make_figure_2():
    print("Generating Figure 2...")

    options = ["Endomycorrhizal", "Ectomycorrhizal", "Arbuscular mycorrhiza", "Ericoid"]
    df_filtered = df.loc[df["Mycorrhizal association"].isin(options)].copy()

    # Match the notebook analysis: cluster on original delta13C and TN columns.
    feature_cols = ["δ13C (‰)", "TN (%)"]
    df_clean = df_filtered[feature_cols + ["Mycorrhizal association"]].dropna().copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_clean[feature_cols])

    kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
    df_clean["Cluster"] = kmeans.fit_predict(x_scaled)

    silhouette_avg = silhouette_score(x_scaled, df_clean["Cluster"])

    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(df_clean["Mycorrhizal association"])
    ari = adjusted_rand_score(true_labels, df_clean["Cluster"])

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Adjusted Rand Index: {ari}")

    # Evaluate all dual-feature combinations from the notebook and save them.
    domain_options = ["Gymnosperm", "Angiosperm", "Grass C3", "Grass C4"]
    df_domain = df.loc[df["Domain"].isin(domain_options)].copy()
    feature_columns = ["TOC (%)", "δ13C (‰)", "TN (%)", "δ15N (‰)", "C/N"]
    results = []

    for comb in combinations(feature_columns, 2):
        combo_data = df_domain[list(comb) + ["Mycorrhizal association"]].dropna().copy()
        if len(combo_data) < 4:
            continue

        combo_scaled = StandardScaler().fit_transform(combo_data[list(comb)])
        combo_clusters = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(combo_scaled)
        combo_true_labels = LabelEncoder().fit_transform(combo_data["Mycorrhizal association"])

        results.append(
            {
                "Features": ", ".join(comb),
                "Silhouette Score": silhouette_score(combo_scaled, combo_clusters),
                "Adjusted Rand Index": adjusted_rand_score(combo_true_labels, combo_clusters),
            }
        )

    output_df = pd.DataFrame(results).sort_values(
        ["Silhouette Score", "Adjusted Rand Index"],
        ascending=False,
    )
    output_df.to_csv(OUTPUT_DIR / "Fig2_feature_combination_scores.csv", index=False)
    print(output_df)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scatter = ax.scatter(
        x_scaled[:, 0],
        x_scaled[:, 1],
        c=df_clean["Cluster"],
        cmap="viridis",
        s=80,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.text(
        0.05,
        0.95,
        f"Silhouette Score = {silhouette_avg:.2f}\nAdjusted Rand Index = {ari:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax.set_xlabel("δ¹³C (Standardized)", fontsize=13, fontweight="bold")
    ax.set_ylabel("TN (%) (Standardized)", fontsize=13, fontweight="bold")
    ax.set_title("Clustering Results with Mycorrhizal Association Labels", fontsize=13, fontweight="bold")
    handles, _ = scatter.legend_elements()
    ax.legend(
        handles,
        options[: len(handles)],
        title="Mycorrhizal association",
        fontsize=10,
        title_fontsize=11,
        loc="upper right",
        frameon=True,
        edgecolor="black",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig2_mycorrhizal_clustering.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================================
# FIGURE 3
# ============================================================================

def make_figure_3():
    print("Generating Figure 3 from reconstructed manuscript models...")

    d15_predictors = ["GST_K", "GSP_mm", "SM_kg_m3", "SMN_gN_m2"]
    d13_predictors = ["GST_K", "GSP_mm", "Elevation_m", "SMN_gN_m2", "SM_kg_m3", "is_C4"]

    d15_data = df[["Transect", "d15N"] + d15_predictors].dropna().copy()
    d13_data = df[["Transect", "d13C"] + d13_predictors].dropna().copy()

    d15_model = LinearRegression()
    d15_model.fit(d15_data[d15_predictors], d15_data["d15N"])
    d15_data["d15N_modelled"] = d15_model.predict(d15_data[d15_predictors])
    d15_data["d15N_residual"] = d15_data["d15N"] - d15_data["d15N_modelled"]
    d15_equation = format_model_equation(
        "d15N",
        d15_model.intercept_,
        d15_predictors,
        d15_model.coef_,
    )

    # Ridge regression follows the manuscript's stated model type for delta13C.
    d13_model = Ridge(alpha=1.0)
    d13_model.fit(d13_data[d13_predictors], d13_data["d13C"])
    d13_data["d13C_modelled"] = d13_model.predict(d13_data[d13_predictors])
    d13_data["d13C_residual"] = d13_data["d13C"] - d13_data["d13C_modelled"]
    d13_equation = format_model_equation(
        "d13C",
        d13_model.intercept_,
        d13_predictors,
        d13_model.coef_,
    )

    print("Fitted model equations used for Figure 3:")
    print(f"  {d15_equation}")
    print(f"  {d13_equation}")

    stats_rows = []
    d15_stats = model_stats(d15_data["d15N"], d15_data["d15N_modelled"])
    d15_stats.update({"model": "delta15N linear regression", "target": "d15N", "equation": d15_equation})
    stats_rows.append(d15_stats)

    d13_stats = model_stats(d13_data["d13C"], d13_data["d13C_modelled"])
    d13_stats.update({"model": "delta13C ridge regression", "target": "d13C", "equation": d13_equation})
    stats_rows.append(d13_stats)

    # Also record the fixed printed equations as a diagnostic, since they do not
    # reproduce the manuscript's delta13C accuracy on the available table.
    d15_data["d15N_printed_equation"] = (
        133.46
        + 0.12 * d15_data["GST_K"]
        - 0.19 * d15_data["GSP_mm"]
        - 0.006 * d15_data["SM_kg_m3"]
        - 1.41 * d15_data["SMN_gN_m2"]
    )
    printed_d15_stats = model_stats(d15_data["d15N"], d15_data["d15N_printed_equation"])
    printed_d15_stats.update({
        "model": "delta15N printed equation",
        "target": "d15N",
        "equation": "d15N_modelled = 133.46 + 0.12*GST_K - 0.19*GSP_mm - 0.006*SM_kg_m3 - 1.41*SMN_gN_m2",
    })
    stats_rows.append(printed_d15_stats)

    d13_data["d13C_printed_equation"] = (
        -210.97
        + 0.16 * d13_data["GST_K"]
        - 0.0003 * d13_data["GSP_mm"]
        - 0.001 * d13_data["Elevation_m"]
        + 1.29 * d13_data["SMN_gN_m2"]
        + 6.27 * d13_data["is_C4"]
    )
    printed_d13_stats = model_stats(d13_data["d13C"], d13_data["d13C_printed_equation"])
    printed_d13_stats.update({
        "model": "delta13C printed equation, C4=1 else 0",
        "target": "d13C",
        "equation": "d13C_modelled = -210.97 + 0.16*GST_K - 0.0003*GSP_mm - 0.001*Elevation_m + 1.29*SMN_gN_m2 + 6.27*is_C4",
    })
    stats_rows.append(printed_d13_stats)

    pd.DataFrame(stats_rows).to_csv(OUTPUT_DIR / "Fig3_model_statistics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    for transect, color in TRANSECT_COLORS.items():
        subset = d15_data[d15_data["Transect"] == transect]
        ax.scatter(
            subset["d15N"],
            subset["d15N_modelled"],
            c=color,
            label=TRANSECT_LABELS[transect],
            s=60,
            alpha=0.65,
            edgecolors="black",
            linewidth=0.5,
        )
    min_val = min(d15_data["d15N"].min(), d15_data["d15N_modelled"].min())
    max_val = max(d15_data["d15N"].max(), d15_data["d15N_modelled"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="1:1 line")
    add_stats_box(ax, d15_stats)
    ax.set_xlabel("Observed δ¹⁵N (‰)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Modelled δ¹⁵N (‰)", fontsize=12, fontweight="bold")
    ax.set_title("(a) δ¹⁵N: linear multi-regression", fontsize=12, fontweight="bold")
    ax.legend(title="Transect", fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for transect, color in TRANSECT_COLORS.items():
        subset = d13_data[d13_data["Transect"] == transect]
        ax.scatter(
            subset["d13C"],
            subset["d13C_modelled"],
            c=color,
            label=TRANSECT_LABELS[transect],
            s=60,
            alpha=0.65,
            edgecolors="black",
            linewidth=0.5,
        )
    min_val = min(d13_data["d13C"].min(), d13_data["d13C_modelled"].min())
    max_val = max(d13_data["d13C"].max(), d13_data["d13C_modelled"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="1:1 line")
    add_stats_box(ax, d13_stats)
    ax.set_xlabel("Observed δ¹³C (‰)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Modelled δ¹³C (‰)", fontsize=12, fontweight="bold")
    ax.set_title("(b) δ¹³C: ridge regression", fontsize=12, fontweight="bold")
    ax.legend(title="Transect", fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for transect, color in TRANSECT_COLORS.items():
        subset = d15_data[d15_data["Transect"] == transect]
        ax.hist(
            subset["d15N_residual"],
            bins=15,
            alpha=0.50,
            color=color,
            label=TRANSECT_LABELS[transect],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual δ¹⁵N (observed - modelled, ‰)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("(c) δ¹⁵N model residuals", fontsize=12, fontweight="bold")
    #ax.legend(title="Transect", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    for transect, color in TRANSECT_COLORS.items():
        subset = d13_data[d13_data["Transect"] == transect]
        ax.hist(
            subset["d13C_residual"],
            bins=15,
            alpha=0.50,
            color=color,
            label=TRANSECT_LABELS[transect],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual δ¹³C (observed - modelled, ‰)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title("(d) δ¹³C model residuals", fontsize=12, fontweight="bold")
    #ax.legend(title="Transect", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig3_isoscape_validation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Figure 3 model summary:")
    print(f"  delta15N linear regression R2 = {d15_stats['r2']:.3f}, RMSE = {d15_stats['rmse']:.2f} per mil")
    print(f"  delta13C ridge regression R2 = {d13_stats['r2']:.3f}, RMSE = {d13_stats['rmse']:.2f} per mil")


# ============================================================================
# MAIN
# ============================================================================

def main():
    make_figure_1_panels()
    make_figure_2()
    make_figure_3()

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Output folder: {OUTPUT_DIR}")
    print("Files:")
    for filename in [
        "Fig1b_d13C_TOC_by_transect.png",
        "Fig1c_d15N_TN_by_transect.png",
        "Fig2_mycorrhizal_clustering.png",
        "Fig3_isoscape_validation.png",
        "Fig3_model_statistics.csv",
    ]:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
