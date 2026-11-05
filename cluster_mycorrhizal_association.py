#!/usr/bin/env python3
"""Cluster foliar samples to recover their mycorrhizal association (Fig. 2).

Every combination of the candidate predictors (2 to 8 variables) is clustered
with K-means and with Gaussian mixture models (full, tied, and diagonal
covariance). Labels are used only to score the partitions, not to fit them.

Binary (2-variable) distribution clustering is kept unless a model with more
variables improves the adjusted Rand index by at least ARI_GAIN_REQUIRED on
both the reported fit and a seed-averaged stability check.

Input : CN Manuscript/EGU BGS/C_N Allocation Supplementary II.xlsx
        (sheet S1, falling back to the first sheet if S1 is absent).
        Column names follow the revised sheet header (see FEATURES). All
        nitrogen predictors use the corrected columns; the 87FLA row was
        already fixed in the sheet, so no values are patched here.

Outputs (saved to CN Manuscript/EGU BGS/figures/):
    Fig2_mycorrhizal_clustering.png   selected clustering solution
    FigS_clustering_comparison.png    scores and comparisons for the tested models

Dung rows are excluded (not a plant mycorrhizal association), as are rows
with non-physical TN or C/N (<= 0), so every model sees the same samples.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
})

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = PROJECT_DIR / "CN Manuscript" / "EGU BGS" / "C_N Allocation Supplementary II.xlsx"
DEFAULT_OUT = PROJECT_DIR / "CN Manuscript" / "EGU BGS" / "figures"

FEATURES = [
    "Elevation (m)",
    "δ13C (‰)",
    "TOC (%)",
    "δ15N (‰) corrected",
    "TN (%) corrected",
    "C/N corrected",
    "Precipitation GSP (mm/m)",
    "Surface Temperature GST (K)",
]

SHORT = {
    "Elevation (m)": "Elevation",
    "δ13C (‰)": "δ¹³C",
    "TOC (%)": "TOC",
    "δ15N (‰) corrected": "δ¹⁵N corr.",
    "TN (%) corrected": "TN corr.",
    "C/N corrected": "C/N corr.",
    "Precipitation GSP (mm/m)": "Precip GSP",
    "Surface Temperature GST (K)": "Temp GST",
}

ASSOC_ORDER = [
    "Arbuscular mycorrhiza",
    "Endomycorrhizal",
    "Ectomycorrhizal",
    "Ericoid",
]

ASSOC_COLOR = {
    "Arbuscular mycorrhiza": "#CC79A7",
    "Endomycorrhizal": "#E69F00",
    "Ectomycorrhizal": "#0072B2",
    "Ericoid": "#009E73",
}

ASSOC_MARKER = {
    "Arbuscular mycorrhiza": "D",
    "Endomycorrhizal": "s",
    "Ectomycorrhizal": "o",
    "Ericoid": "^",
}

# Pale region fills, deliberately different hues from the marker palette.
CLUSTER_RGBA = [
    (0.22, 0.42, 0.72, 0.32),
    (0.62, 0.40, 0.18, 0.28),
    (0.36, 0.58, 0.32, 0.28),
    (0.52, 0.34, 0.62, 0.28),
    (0.72, 0.32, 0.38, 0.30),
    (0.28, 0.52, 0.56, 0.28),
    (0.62, 0.56, 0.22, 0.28),
]

# A higher-dimensional model must beat the best 2-variable model by this
# much (adjusted Rand) on the reported fit AND on the seed-averaged fit.
ARI_GAIN_REQUIRED = 0.05
N_STABILITY_SEEDS = 8

MODEL_ARI_COLS = [
    ("GMM-diag", "gmm_diag_ari"),
    ("GMM-tied", "gmm_tied_ari"),
    ("GMM-full", "gmm_full_ari"),
    ("KMeans", "kmeans_ari"),
]


def short_join(names: list[str]) -> str:
    return " + ".join(SHORT[n] for n in names)


def style_figure(fig) -> None:
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontfamily("DejaVu Sans")
        text_artist.set_fontsize(9.5)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        if not np.any(mask):
            continue
        _, counts = np.unique(y_true[mask], return_counts=True)
        total += counts.max()
    return float(total / len(y_true))


def agreement(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "v": float(v_measure_score(y_true, y_pred)),
        "purity": purity_score(y_true, y_pred),
    }


def preview_columns(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Columns to show in console previews; never crash on an absent column."""
    cols = [c for c in wanted if c in df.columns]
    return cols if cols else list(df.columns[: min(4, len(df.columns))])


def load_samples(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the supplementary sheet and split clean / non-physical / dung rows."""
    xls = pd.ExcelFile(path)
    sheet = "S1" if "S1" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    missing = [c for c in FEATURES + ["Mycorrhizal association", "Sample_Code", "Domain"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in sheet '{sheet}': {missing}")
    # Dung is not a plant mycorrhizal association.
    dung_mask = (
        df["Mycorrhizal association"].astype(str).str.strip().eq("Dung")
        | df["Domain"].astype(str).str.strip().eq("Dung")
    )
    dung = df.loc[dung_mask].copy()
    df = df.loc[~dung_mask].copy()
    # Anything still non-physical is dropped so every model sees the same rows.
    bad = (df["TN (%) corrected"] <= 0) | (df["C/N corrected"] <= 0)
    dropped = df.loc[bad].copy()
    clean = df.loc[~bad].copy()
    clean["Mycorrhizal association"] = clean["Mycorrhizal association"].astype(str)
    return clean, dropped, dung


def fit_predict(kind: str, X: np.ndarray, k: int, seed: int):
    if kind == "KMeans":
        model = KMeans(n_clusters=k, n_init=25, random_state=seed)
        labels = model.fit_predict(X)
        return model, labels
    cov = kind.split("-", 1)[1]
    model = GaussianMixture(
        n_components=k,
        covariance_type=cov,
        n_init=5,
        random_state=seed,
        reg_covar=1e-5,
        max_iter=300,
    )
    labels = model.fit_predict(X)
    return model, labels


def evaluate_all(X_raw: np.ndarray, y: np.ndarray, k: int) -> pd.DataFrame:
    rows = []
    index_sets = []
    for n_features in range(2, len(FEATURES) + 1):
        for comb in itertools.combinations(range(len(FEATURES)), n_features):
            names = [FEATURES[i] for i in comb]
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw[:, comb])
            row: dict = {
                "n_features": n_features,
                "features": short_join(names),
                "feature_idx": ",".join(str(i) for i in comb),
            }
            for kind, col in MODEL_ARI_COLS:
                try:
                    _, labels = fit_predict(kind, X, k, seed=0)
                    scores = agreement(y, labels)
                except Exception:
                    scores = {"ari": np.nan, "nmi": np.nan, "v": np.nan, "purity": np.nan}
                prefix = col.replace("_ari", "")
                row[f"{prefix}_ari"] = scores["ari"]
                row[f"{prefix}_nmi"] = scores["nmi"]
                row[f"{prefix}_purity"] = scores["purity"]
            ari_vals = [(kind, row[col]) for kind, col in MODEL_ARI_COLS]
            ari_vals = [(kind, val) for kind, val in ari_vals if np.isfinite(val)]
            if not ari_vals:
                row["best_model"] = "none"
                row["best_ari"] = np.nan
            else:
                # Simpler distribution models win exact numerical ties.
                best_ari = max(val for _, val in ari_vals)
                candidates = [kind for kind, val in ari_vals if abs(val - best_ari) < 1e-9]
                preference = [knd for knd, _ in MODEL_ARI_COLS]
                candidates.sort(key=lambda kind: preference.index(kind))
                row["best_model"] = candidates[0]
                row["best_ari"] = best_ari
            rows.append(row)
            index_sets.append(comb)
    scores = pd.DataFrame(rows)
    scores.attrs["index_sets"] = index_sets
    return scores


def stability(kind: str, X: np.ndarray, y: np.ndarray, k: int) -> tuple[float, float]:
    vals = []
    for seed in range(N_STABILITY_SEEDS):
        try:
            _, labels = fit_predict(kind, X, k, seed=seed)
        except Exception:
            continue
        vals.append(adjusted_rand_score(y, labels))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def subset_matrix(X_raw: np.ndarray, feature_idx: str) -> tuple[np.ndarray, StandardScaler, list[str]]:
    comb = tuple(int(i) for i in feature_idx.split(","))
    names = [FEATURES[i] for i in comb]
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw[:, comb])
    return X, scaler, names


def choose_model(scores: pd.DataFrame, X_raw: np.ndarray, y: np.ndarray, k: int) -> dict:
    two = scores[scores["n_features"] == 2]
    multi = scores[scores["n_features"] > 2]
    best2 = two.loc[two["best_ari"].idxmax()]
    best_m = multi.loc[multi["best_ari"].idxmax()]
    point_gain = float(best_m["best_ari"] - best2["best_ari"])

    X2, _, names2 = subset_matrix(X_raw, best2["feature_idx"])
    Xm, _, names_m = subset_matrix(X_raw, best_m["feature_idx"])
    mean2, std2 = stability(best2["best_model"], X2, y, k)
    mean_m, std_m = stability(best_m["best_model"], Xm, y, k)
    mean_gain = float(mean_m - mean2)

    use_multi = point_gain >= ARI_GAIN_REQUIRED and mean_gain >= ARI_GAIN_REQUIRED
    chosen = best_m if use_multi else best2
    reason = (
        f"Best model with 3 or more variables ({best_m['best_model']}: {best_m['features']}) "
        f"has ARI {best_m['best_ari']:.3f} versus {best2['best_ari']:.3f} for the best "
        f"2-variable model ({best2['best_model']}: {best2['features']}). "
        f"Point gain {point_gain:+.3f}; seed-mean gain {mean_gain:+.3f} "
        f"(2-var mean {mean2:.3f} ± {std2:.3f}, multi mean {mean_m:.3f} ± {std_m:.3f}). "
        f"Required gain is {ARI_GAIN_REQUIRED:.2f} on both. "
        + (
            "The higher-dimensional model is significantly better, so it is used."
            if use_multi
            else "The gain is not significant and stable, so the 2-variable distribution cluster is kept."
        )
    )
    return {
        "row": chosen,
        "best2": best2,
        "best_multi": best_m,
        "use_multi": use_multi,
        "point_gain": point_gain,
        "mean_gain": mean_gain,
        "mean2": mean2,
        "std2": std2,
        "mean_multi": mean_m,
        "std_multi": std_m,
        "reason": reason,
        "names2": names2,
        "names_multi": names_m,
    }


def order_clusters(labels: np.ndarray, x_coord: np.ndarray, k: int) -> tuple[np.ndarray, dict[int, int]]:
    """Renumber clusters 0..k-1 by increasing mean of the plotted x coordinate."""
    occupied = []
    empty = []
    for cluster in range(k):
        mask = labels == cluster
        if np.any(mask):
            occupied.append((float(np.mean(x_coord[mask])), cluster))
        else:
            empty.append(cluster)
    occupied.sort()
    ordered_ids = [cluster for _, cluster in occupied] + empty
    remap = {old: new for new, old in enumerate(ordered_ids)}
    ordered = np.array([remap[int(c)] for c in labels])
    return ordered, remap


def draw_main_figure(
    clean: pd.DataFrame,
    selection: dict,
    X_raw: np.ndarray,
    y: np.ndarray,
    k: int,
    out_path: Path,
) -> dict:
    row = selection["row"]
    kind = str(row["best_model"])
    X, _, names = subset_matrix(X_raw, row["feature_idx"])
    model, raw_labels = fit_predict(kind, X, k, seed=0)
    metrics = agreement(y, raw_labels)

    use_pca = X.shape[1] > 2
    if use_pca:
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(X)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    else:
        xy = X
        x_label, y_label = names

    labels, remap = order_clusters(raw_labels, xy[:, 0], k)
    x_pad = max(float(np.ptp(xy[:, 0])) * 0.08, 0.1)
    y_pad = max(float(np.ptp(xy[:, 1])) * 0.08, 0.1)
    xx, yy = np.meshgrid(
        np.linspace(float(xy[:, 0].min()) - x_pad, float(xy[:, 0].max()) + x_pad, 240),
        np.linspace(float(xy[:, 1].min()) - y_pad, float(xy[:, 1].max()) + y_pad, 240),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    model_grid = pca.inverse_transform(grid) if use_pca else grid
    regions = model.predict(model_grid)
    regions = np.array([remap[int(cluster)] for cluster in regions]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.88)
    cluster_colors = [CLUSTER_RGBA[i % len(CLUSTER_RGBA)] for i in range(k)]
    ax.pcolormesh(
        xx,
        yy,
        regions,
        cmap=ListedColormap(cluster_colors),
        vmin=-0.5,
        vmax=k - 0.5,
        shading="nearest",
        zorder=0,
    )
    associations = [assoc for assoc in ASSOC_ORDER if assoc in set(y)]
    for assoc in associations:
        mask = y == assoc
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=45,
            color=ASSOC_COLOR[assoc],
            marker=ASSOC_MARKER[assoc],
            edgecolors="#333333",
            linewidth=0.5,
            label=assoc,
            zorder=2,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        "Mycorrhizal clustering from C/N predictors using K-means\n"
        f"(ARI = {metrics['ari']:.3f}, n = {len(y)}; isotopes: "
        r"$\delta^{13}$C (‰, VPDB) and $\delta^{15}$N (‰, AIR))",
        fontweight="bold",
    )
    ax.set_xlim(float(xy[:, 0].min()) - x_pad, float(xy[:, 0].max()) + x_pad)
    ax.set_ylim(float(xy[:, 1].min()) - y_pad, float(xy[:, 1].max()) + y_pad)
    ax.grid(True, color="#000000", alpha=0.08, zorder=1)
    association_handles = [
        Line2D(
            [0], [0], marker=ASSOC_MARKER[assoc], linestyle="none",
            markerfacecolor=ASSOC_COLOR[assoc], markeredgecolor="#333333",
            markersize=7, label=assoc,
        )
        for assoc in associations
    ]
    cluster_handles = [
        Patch(
            facecolor=CLUSTER_RGBA[i % len(CLUSTER_RGBA)],
            edgecolor=CLUSTER_RGBA[i % len(CLUSTER_RGBA)][:3],
            label=f"Cluster {i + 1}",
        )
        for i in range(k)
    ]
    # ONE combined legend inside the axes (loc="best"): association markers
    # plus cluster-region shading patches.
    ax.legend(
        handles=association_handles + cluster_handles,
        title="Mycorrhizal association (markers) · cluster regions (shading)",
        loc="best",
        frameon=True,
    )
    style_figure(fig)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    crosstab = pd.crosstab(
        pd.Series(y, name="Mycorrhizal association"),
        pd.Series(labels, name="Cluster").map(lambda c: f"C{c + 1}"),
    )
    crosstab = crosstab.reindex(
        index=[a for a in ASSOC_ORDER if a in crosstab.index],
        columns=[f"C{j + 1}" for j in range(k)],
    )
    return {
        "metrics": metrics,
        "crosstab": crosstab,
        "labels": labels,
        "names": names,
        "kind": kind,
    }


def draw_supplementary_figure(
    scores: pd.DataFrame,
    selection: dict,
    out_path: Path,
) -> None:
    model_cols = [
        ("K-means", "kmeans_ari", "#E09F3E"),
        ("GMM diagonal", "gmm_diag_ari", "#3D5A80"),
        ("GMM tied", "gmm_tied_ari", "#81B29A"),
        ("GMM full", "gmm_full_ari", "#C8553D"),
    ]
    grouped = scores.groupby("n_features")[
        [column for _, column, _ in model_cols]
    ].max()

    selected_idx = str(selection["row"]["feature_idx"])
    ranked = scores.nlargest(10, "best_ari").copy()
    ranked = ranked[ranked["feature_idx"].astype(str) != selected_idx]
    selected_row = scores[scores["feature_idx"].astype(str) == selected_idx].copy()
    selected_row["best_model"] = "KMeans"
    selected_row["best_ari"] = selected_row["kmeans_ari"]
    ranked = pd.concat([ranked, selected_row], ignore_index=True)
    ranked = ranked.sort_values("best_ari")
    colors = {
        "KMeans": "#E09F3E",
        "GMM-diag": "#3D5A80",
        "GMM-tied": "#81B29A",
        "GMM-full": "#C8553D",
    }

    fig, (ax_models, ax_features) = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.0, 1.9]}
    )
    dimensions = grouped.index.to_numpy(dtype=int)
    bar_width = 0.19
    offsets = (np.arange(len(model_cols)) - (len(model_cols) - 1) / 2) * bar_width
    for offset, (label, column, color) in zip(offsets, model_cols):
        bars = ax_models.bar(
            dimensions + offset,
            grouped[column].to_numpy(),
            width=bar_width,
            color=color,
            label=label,
        )
        ax_models.bar_label(bars, fmt="%.2f", padding=1, rotation=90)
    ax_models.set_xticks(dimensions)
    ax_models.set_xlabel("Number of predictors")
    ax_models.set_ylabel("Best ARI among tested combinations")
    ax_models.set_title("Best score by model and dimensionality", loc="left")
    ax_models.set_ylim(top=float(grouped.to_numpy().max()) + 0.07)
    ax_models.legend(loc="best", frameon=False)
    ax_models.grid(axis="y", alpha=0.2)

    y_positions = np.arange(len(ranked))
    bar_colors = [colors[str(name)] for name in ranked["best_model"]]
    bars = ax_features.barh(
        y_positions,
        ranked["best_ari"],
        color=bar_colors,
        edgecolor="black",
        linewidth=[
            1.5 if str(feature_idx) == selected_idx else 0
            for feature_idx in ranked["feature_idx"]
        ],
    )
    ax_features.bar_label(bars, fmt="%.3f", padding=3)
    ax_features.set_yticks(y_positions)
    ax_features.set_yticklabels(ranked["features"])
    ax_features.set_xlabel("Adjusted Rand Index (ARI)")
    ax_features.set_title("Highest-scoring predictor combinations", loc="left")
    ax_features.grid(axis="x", alpha=0.2)
    ax_features.set_xlim(left=0)
    ax_features.text(
        0.02,
        0.02,
        "Black outline = selected model",
        transform=ax_features.transAxes,
        va="bottom",
    )
    ax_features.set_xlim(right=float(ranked["best_ari"].max()) + 0.12)

    chosen = selection["row"]
    fig.suptitle(
        "Mycorrhizal clustering comparison\n"
        f"All predictor combinations (2–{len(FEATURES)} features), "
        "K-means and three Gaussian-mixture models; "
        f"selected: {chosen['best_model']} on {chosen['features']}",
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "ARI is calculated against known association labels for scoring only. "
        f"A model with >2 predictors is selected only when both gains are ≥{ARI_GAIN_REQUIRED:.2f} "
        f"for the fitted score and the {N_STABILITY_SEEDS}-seed mean. "
        r"Isotopes: $\delta^{13}$C (‰, VPDB); corrected $\delta^{15}$N (‰, AIR).",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.91))
    style_figure(fig)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_table(scores: pd.DataFrame) -> str:
    show = scores.sort_values(["n_features", "best_ari"], ascending=[True, False]).copy()
    show = show[
        [
            "n_features",
            "features",
            "kmeans_ari",
            "gmm_full_ari",
            "gmm_tied_ari",
            "gmm_diag_ari",
            "best_model",
            "best_ari",
        ]
    ]
    show = show.rename(
        columns={
            "n_features": "n",
            "kmeans_ari": "KMeans",
            "gmm_full_ari": "GMM_full",
            "gmm_tied_ari": "GMM_tied",
            "gmm_diag_ari": "GMM_diag",
            "best_model": "best_model",
            "best_ari": "best_ARI",
        }
    )
    return show.to_string(index=False, float_format=lambda v: f"{v:.3f}")


def run_analysis(xlsx: Path = DEFAULT_XLSX, out: Path = DEFAULT_OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    clean, dropped, dung = load_samples(xlsx)
    y = clean["Mycorrhizal association"].to_numpy()
    classes = [a for a in ASSOC_ORDER if a in set(y)]
    extra = sorted(set(y) - set(classes))
    classes.extend(extra)
    k = len(classes)
    X_raw = clean[FEATURES].to_numpy(dtype=float)

    print("Mycorrhizal-association clustering")
    print(f"Samples used: {len(clean)}   clusters k = {k} (one per recorded association)")
    print("Class counts:")
    for assoc, count in clean["Mycorrhizal association"].value_counts().reindex(classes).items():
        print(f"  {assoc:<24} {int(count)}")
    if len(dung):
        print(f"Excluded {len(dung)} dung sample(s); they are not plants:")
        cols = preview_columns(dung, ["Sample_Code", "Species", "Mycorrhizal association"])
        print(dung[cols].to_string(index=False))
    if len(dropped):
        print(f"Excluded {len(dropped)} row(s) with non-physical TN or C/N:")
        cols = preview_columns(dropped, ["Sample_Code", "Species", "Mycorrhizal association", "TN (%) corrected", "C/N corrected"])
        print(dropped[cols].to_string(index=False))
    print()
    print(
        "Predictors: "
        + ", ".join(FEATURES)
    )
    print(
        f"Searched all {2 ** len(FEATURES) - len(FEATURES) - 1} combinations of 2 or more predictors "
        "(feature order does not matter, so these are combinations rather than ordered permutations)."
    )
    print("Models: K-means; Gaussian mixture with full, tied, and diagonal covariance.")
    print("Features are standardised within each combination. Labels score the clusters; they are not used to fit them.")
    print(f"Primary score: adjusted Rand index. A model with 3+ variables replaces the best 2-variable")
    print(f"model only if its ARI is at least {ARI_GAIN_REQUIRED:.2f} higher on the reported fit and on an")
    print(f"{N_STABILITY_SEEDS}-seed average.")
    print()

    scores = evaluate_all(X_raw, y, k)
    selection_scores = scores.copy()
    selection_scores["best_model"] = "KMeans"
    selection_scores["best_ari"] = selection_scores["kmeans_ari"]
    selection = choose_model(selection_scores, X_raw, y, k)
    figure_path = out / "Fig2_mycorrhizal_clustering.png"
    supplementary_path = out / "FigS_clustering_comparison.png"
    result = draw_main_figure(clean, selection, X_raw, y, k, figure_path)
    draw_supplementary_figure(scores, selection, supplementary_path)

    print("Decision")
    print(selection["reason"])
    chosen = selection["row"]
    metrics = result["metrics"]
    print()
    print("Selected model")
    print(f"  Algorithm : {chosen['best_model']}")
    print(f"  Variables : {', '.join(result['names'])}")
    print(f"  ARI       : {metrics['ari']:.3f}")
    print(f"  NMI       : {metrics['nmi']:.3f}")
    print(f"  V-measure : {metrics['v']:.3f}")
    print(f"  Purity    : {metrics['purity']:.3f}")
    majority = clean["Mycorrhizal association"].value_counts().max() / len(clean)
    print(f"  Majority-class purity baseline: {majority:.3f}")
    print()
    print("Cluster × association counts (clusters numbered along the plotted x-axis)")
    print(result["crosstab"].to_string())
    print()
    print(f"Figure: {figure_path}")
    print(f"Supplementary comparison: {supplementary_path}")
    return {
        "selection": selection,
        "result": result,
        "scores": scores,
        "n": len(clean),
        "k": k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx", nargs="?", default=str(DEFAULT_XLSX))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run_analysis(Path(args.xlsx), Path(args.out))


if __name__ == "__main__":
    main()