"""
README — Full Calibration Analysis Pipeline for brGDGTs & 3-OH Fatty Acids
===========================================================================

Overview
--------
This script performs a comprehensive calibration and model evaluation workflow
for two biomarker proxy systems:
    1) brGDGTs (branched glycerol dialkyl glycerol tetraethers)
    2) 3-OH fatty acids

Both proxies are analyzed at two spatial scales:
    • Regional datasets
    • Global datasets

The goal is to quantify and compare their ability to reconstruct two key
environmental variables:
    • MAAT (Mean Annual Air Temperature)
    • pH

The pipeline integrates statistical modeling, machine learning, feature
importance analysis, and visualization to produce a complete, publication-
quality calibration assessment.


Core Methodology
----------------
For each proxy × scale × target combination (total = 8 cases), the script:

1. Model Training & Validation
   • Multiple Linear Regression (MLR)
   • Random Forest Regression (RF)
   • Cross-validation strategy:
        - Leave-One-Out (LOO-CV) for small datasets (n < 30)
        - k-fold CV (default k=10) for larger datasets

2. Performance Metrics
   • RMSE (Root Mean Squared Error)
   • R² (coefficient of determination)
   • Both cross-validated (CV) and in-sample (training) metrics are reported

3. Feature Importance
   • MLR:
        - Raw coefficients
        - Standardized coefficients (for comparability)
   • Random Forest:
        - Permutation importance (mean decrease in model performance)
        - Importance uncertainty (standard deviation across permutations)

4. Driver Selection Analysis
   • Features ranked by RF permutation importance
   • Incremental model building:
        - Add features one-by-one in order of importance
        - Track CV-RMSE and CV-R² at each step
   • Identifies “minimum driver set” and performance saturation point

5. Residual Diagnostics
   • Predicted vs observed plots
   • Residual vs predicted trends (with smoothing)
   • Residual distributions with Gaussian fit
   • Used to assess bias, heteroscedasticity, and model adequacy


Unified (Pooled) Calibration
----------------------------
To bridge regional and global models:

• Top drivers (features) are extracted separately from regional and global RF rankings
• Shared top features are identified (intersection of top-N lists)
• A unified MLR model is trained on pooled data (regional + global)
• Outputs:
    - Cross-validated performance
    - Final regression equation
    - Standardized coefficient interpretation

This provides a robust, transferable calibration across spatial scales.


Figures Generated
-----------------
Figure 1 — Feature Importance (4 rows × 4 panels): For every proxy/target/scale combination, side-by-side MLR standardised coefficients (|β_std|, coloured by sign: red=positive, blue=negative) and RF permutation importance with error bars. This directly shows which compounds drive each model.
Figure 2 — Driver Cutoff Curves (4 rows × 2 columns): Features are added one at a time in RF importance rank order, and CV-RMSE + CV-R² are tracked on twin axes for both MLR and RF. An automated "elbow" marker identifies the minimum number of features needed before further additions stop improving performance meaningfully. The elbow feature name is annotated on the plot.
Figure 3 — Residual Diagnostics (one figure per proxy/target pair, regional and global side by side): Predicted vs observed, residuals vs predicted with smoothed trend line, and residual histogram with normal overlay — for MLR and RF separately.
Figure 4 — Regional vs Global Comparison (4 rows × 3 columns): Overlay scatter of all predicted vs observed colour-coded by regional/global, plus bar charts of CV-RMSE and CV-R² for every model/scale combination.
Figure 5 — Unified (Pooled) MLR (4 rows × 2 columns): The regional and global datasets are pooled, and a single MLR is fit on the features appearing in the top-10 of both regional and global RF importance rankings (the intersection). Equations, CV metrics, and standardised coefficient plots are printed to console and visualised.
Figure 6 — Driver Overlap Heatmap: A rank matrix showing which top-10 features are shared vs exclusive between regional and global datasets, with shared drivers highlighted in gold — useful for justifying which compounds enter the unified equation.


Data Handling
-------------
• Input: Excel workbook with multiple sheets
• Automatic:
    - Column cleaning and renaming
    - Conversion to numeric types
    - Removal of missing values in required variables
• Separate loaders for:
    - Regional brGDGTs
    - Regional 3-OH FAs
    - Global brGDGTs
    - Global 3-OH FAs


Key Design Features
-------------------
• Fully reproducible (fixed random seed)
• Adaptive cross-validation strategy
• Modular function design for:
    - Model fitting
    - Feature ranking
    - Driver selection
    - Plot generation
• Publication-ready visual outputs
• Consistent color palette across figures
• Scalable to additional proxies or datasets

Author: Bibhasvata Dasgupta
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FILE_PATH  = '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Projects/CN/Supplementary Datasets.xlsx'
OUT_PATH   = '/Users/Dasgu004/Library/CloudStorage/OneDrive-UniversiteitUtrecht(2)/PostDoc/Projects/CN'
K_FOLDS    = 10
RS         = 42

BRGDGT_FEATS = ['IIIa', 'IIIa_prime', 'IIIb', 'IIIb_prime',
                'IIIc', 'IIIc_prime', 'IIa', 'IIa_prime',
                'IIb', 'IIb_prime', 'IIc', 'IIc_prime', 'Ia', 'Ib', 'Ic']

OHFA_FEATS   = ['C10', 'i-C11', 'a-C11', 'C11', 'i-C12', 'C12',
                'i-C13', 'a-C13', 'C13', 'i-C14', 'C14',
                'i-C15', 'a-C15', 'C15', 'i-C16', 'C16',
                'i-C17', 'a-C17', 'C17', 'C18']

PALETTE = {'regional': '#E64B35', 'global': '#4DBBD5',
           'mlr': '#FF9800',      'rf': '#4CAF50',
           'si':  '#2196F3'}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_clean(path, sheet, header, rename=None):
    """
    Load a sheet, normalise column names, replace 'n.a.' with NaN,
    apply optional column rename, then coerce everything to numeric.
    Does NOT dropna — that happens per-target inside run(), so each
    target retains the maximum available sample size.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=header)
    df.columns = [str(c).strip().replace("'", "_prime") for c in df.columns]
    df.replace('n.a.', np.nan, inplace=True)
    if rename:
        df.rename(columns=rename, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

rename_reg = {'iC15':'i-C15','aC15':'a-C15','iC16':'i-C16','iC17':'i-C17','aC17':'a-C17'}

# Regional datasets
df_br_reg  = load_clean(FILE_PATH, 'brGDGTs', 0)
df_fa_reg  = load_clean(FILE_PATH, '3-OH FAs', 0, rename=rename_reg)

# Global datasets — loaded as full base frames; dropna happens per-target in run()
# so each target uses the maximum available samples.
# brGDGTs global: header=0  → 1938 raw rows (MAAT complete for all; feature/pH NaNs vary)
# 3-OH FAs global: header=1 → 343 raw rows; MAAT column is labelled 'MAAT (°C)'
df_br_glob = load_clean(FILE_PATH, 'brGDGTs global data set', 0)
df_fa_glob = load_clean(FILE_PATH, '3-OH FAs global data', 1,
                        rename={'MAAT (℃)': 'MAAT'})


# ─────────────────────────────────────────────────────────────────────────────
# CV HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_cv(n):
    if n < 30:
        return LeaveOneOut(), 'LOO-CV'
    return KFold(n_splits=K_FOLDS, shuffle=True, random_state=RS), f'{K_FOLDS}-fold CV'

def cv_pred(X, y, model, cv):
    return cross_val_predict(model, X, y, cv=cv)

def metrics(y, yp):
    return np.sqrt(mean_squared_error(y, yp)), r2_score(y, yp)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def feature_importance(X, y, feat_names):
    """
    Returns two DataFrames (mlr_imp, rf_imp) with |coef| and permutation importance.
    """
    rf = RandomForestRegressor(250, random_state=RS)
    rf.fit(X, y)
    pi = permutation_importance(rf, X, y, n_repeats=30, random_state=RS)
    rf_imp = pd.DataFrame({'feature': feat_names,
                           'importance': pi.importances_mean,
                           'std': pi.importances_std}).sort_values('importance', ascending=False)

    lr = LinearRegression().fit(X, y)
    sc = StandardScaler().fit_transform(X)
    lr_sc = LinearRegression().fit(sc, y)
    mlr_imp = pd.DataFrame({'feature': feat_names,
                            'coef': lr.coef_,
                            'std_coef': np.abs(lr_sc.coef_)}).sort_values('std_coef', ascending=False)
    return mlr_imp, rf_imp

# ─────────────────────────────────────────────────────────────────────────────
# DRIVER CUTOFF CURVE
# ─────────────────────────────────────────────────────────────────────────────
def driver_cutoff(X, y, feat_names, cv):
    """
    Incrementally add features (by RF permutation importance rank) and record
    CV-RMSE and CV-R² for MLR and RF at each step.
    """
    rf_full = RandomForestRegressor(250, random_state=RS)
    rf_full.fit(X, y)
    pi = permutation_importance(rf_full, X, y, n_repeats=20, random_state=RS)
    order = np.argsort(pi.importances_mean)[::-1]
    ranked = [feat_names[i] for i in order]

    results = []
    for k in range(1, len(ranked)+1):
        sel = ranked[:k]
        Xk  = X[sel]
        yp_lr = cv_pred(Xk, y, LinearRegression(), cv)
        yp_rf = cv_pred(Xk, y, RandomForestRegressor(250, random_state=RS), cv)
        rm_lr, r2_lr = metrics(y, yp_lr)
        rm_rf, r2_rf = metrics(y, yp_rf)
        results.append({'k': k, 'feature': sel[-1],
                        'rmse_lr': rm_lr, 'r2_lr': r2_lr,
                        'rmse_rf': rm_rf, 'r2_rf': r2_rf})
    return pd.DataFrame(results), ranked

# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED (POOLED) MLR ON SHARED TOP DRIVERS
# ─────────────────────────────────────────────────────────────────────────────
def unified_mlr(df_reg, df_glob, shared_feats, target):
    """Pool regional+global, fit MLR on shared features, report full CV metrics."""
    df_pool = pd.concat([df_reg[shared_feats + [target]],
                         df_glob[shared_feats + [target]]], ignore_index=True)
    df_pool.dropna(inplace=True)
    X = df_pool[shared_feats]
    y = df_pool[target]
    cv, cv_lbl = get_cv(len(y))
    lr = LinearRegression().fit(X, y)
    yp_cv = cv_pred(X, y, LinearRegression(), cv)
    rm, r2 = metrics(y, yp_cv)
    eq_parts = ' + '.join([f'{c:.4f}*{n}' for n,c in zip(shared_feats, lr.coef_)])
    eq = f'{target} = {eq_parts} + {lr.intercept_:.4f}'
    return {'eq': eq, 'cv_rmse': rm, 'cv_r2': r2, 'cv_label': cv_lbl,
            'n': len(y), 'y': y.values, 'yp': yp_cv, 'features': shared_feats}

# ─────────────────────────────────────────────────────────────────────────────
# ── FULL ANALYSIS RUNNER ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def run(df, feats, target, label):
    # Per-target dropna: only drop rows missing THIS target or any feature,
    # so each target uses the maximum available samples independently.
    df_clean = df[feats + [target]].dropna()
    X   = df_clean[feats]
    y   = df_clean[target].values
    n   = len(y)
    cv, cv_lbl = get_cv(n)
    feat_arr = np.array(feats)

    lr = LinearRegression()
    rf = RandomForestRegressor(250, random_state=RS)
    yp_lr = cv_pred(X, y, lr, cv)
    yp_rf = cv_pred(X, y, rf, cv)
    rm_lr, r2_lr = metrics(y, yp_lr)
    rm_rf, r2_rf = metrics(y, yp_rf)

    lr.fit(X, y); rf.fit(X, y)
    r2_lr_in = r2_score(y, lr.predict(X))
    r2_rf_in = r2_score(y, rf.predict(X))

    mlr_imp, rf_imp = feature_importance(X, y, feats)
    cutoff_df, ranked = driver_cutoff(X, pd.Series(y), feats, cv)

    return {
        'label': label, 'target': target, 'n': n, 'cv_label': cv_lbl,
        'feats': feats, 'ranked': ranked,
        'y': y, 'yp_lr': yp_lr, 'yp_rf': yp_rf,
        'rm_lr': rm_lr, 'r2_lr': r2_lr, 'r2_lr_in': r2_lr_in,
        'rm_rf': rm_rf, 'r2_rf': r2_rf, 'r2_rf_in': r2_rf_in,
        'lr_coef': lr.coef_, 'lr_int': lr.intercept_,
        'mlr_imp': mlr_imp, 'rf_imp': rf_imp,
        'cutoff': cutoff_df, 'lr_model': lr, 'rf_model': rf,
        'X': X,
    }

# ─────────────────────────────────────────────────────────────────────────────
# ── RUN ALL 8 CASES ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
results = {}
cases = [
    (df_br_reg,  BRGDGT_FEATS, 'brGDGTs',  'regional'),
    (df_br_glob, BRGDGT_FEATS, 'brGDGTs',  'global'),
    (df_fa_reg,  OHFA_FEATS,   '3-OH FAs', 'regional'),
    (df_fa_glob, OHFA_FEATS,   '3-OH FAs', 'global'),
]
for df, feats, proxy, scale in cases:
    for tgt in ['MAAT', 'pH']:
        key = f'{proxy}_{scale}_{tgt}'
        print(f'Running: {key}  (n={len(df)})')
        results[key] = run(df, feats, tgt, f'{proxy} {scale.capitalize()}')
        results[key]['proxy'] = proxy
        results[key]['scale'] = scale

# ─────────────────────────────────────────────────────────────────────────────
# ── PRINT CONSOLE SUMMARY ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*75)
print(f"{'Dataset':<30} {'Tgt':<5} {'N':>5}  {'CV':>10}  "
      f"{'MLR-RMSE':>9} {'MLR-R²CV':>9} {'MLR-R²in':>9}  "
      f"{'RF-RMSE':>8} {'RF-R²CV':>8} {'RF-R²in':>8}")
print('-'*75)
for key, r in results.items():
    print(f"{r['label']:<30} {r['target']:<5} {r['n']:>5}  {r['cv_label']:>10}  "
          f"{r['rm_lr']:>9.3f} {r['r2_lr']:>9.3f} {r['r2_lr_in']:>9.3f}  "
          f"{r['rm_rf']:>8.3f} {r['r2_rf']:>8.3f} {r['r2_rf_in']:>8.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# ── PRINT MLR EQUATIONS & RF IMPORTANCES ─────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def print_equations_and_importances(results, top_n_rf=8):
    VALIDITY_NOTE = {
        'brGDGTs_regional_pH':    '  *** CV-R² < 0: no valid predictive skill (severe overfitting) ***',
        '3-OH FAs_regional_MAAT': '  *** CV-R² < 0: no valid predictive skill (severe overfitting) ***',
        '3-OH FAs_regional_pH':   '  *** CV-R² < 0: no valid predictive skill (severe overfitting) ***',
    }

    print('\n' + '='*80)
    print('  MLR EQUATIONS & RF FEATURE IMPORTANCES — ALL MODELS')
    print('='*80)

    order = [
        ('brGDGTs',  'regional', 'MAAT'),
        ('brGDGTs',  'regional', 'pH'),
        ('brGDGTs',  'global',   'MAAT'),
        ('brGDGTs',  'global',   'pH'),
        ('3-OH FAs', 'regional', 'MAAT'),
        ('3-OH FAs', 'regional', 'pH'),
        ('3-OH FAs', 'global',   'MAAT'),
        ('3-OH FAs', 'global',   'pH'),
    ]

    for proxy, scale, tgt in order:
        key = f'{proxy}_{scale}_{tgt}'
        if key not in results:
            continue
        r = results[key]

        # ── header ────────────────────────────────────────────────────────────
        print(f'\n{"─"*80}')
        print(f'  {proxy.upper()}  |  {scale.upper()}  |  Target: {tgt}')
        print(f'  n={r["n"]}   CV: {r["cv_label"]}')
        print(f'  MLR  →  CV-RMSE={r["rm_lr"]:.3f}   CV-R²={r["r2_lr"]:.3f}   in-sample R²={r["r2_lr_in"]:.3f}')
        print(f'  RF   →  CV-RMSE={r["rm_rf"]:.3f}   CV-R²={r["r2_rf"]:.3f}   in-sample R²={r["r2_rf_in"]:.3f}')
        if key in VALIDITY_NOTE:
            print(VALIDITY_NOTE[key])
        print(f'{"─"*80}')

        # ── MLR equation with line-wrapping ───────────────────────────────────
        feats     = r['feats']
        coefs     = r['lr_coef']
        intercept = r['lr_int']
        terms     = [f'{c:+.4f}·{f}' for f, c in zip(feats, coefs)]
        prefix    = f'  {tgt} = '
        pad       = ' ' * len(prefix)
        row       = prefix
        for i, t in enumerate(terms):
            sep = ' + ' if i < len(terms) - 1 else ''
            candidate = row + t + sep
            if len(candidate) > 112 and i > 0:
                print(row.rstrip(' +'))
                row = pad + t + sep
            else:
                row = candidate
        print(row + f'{intercept:+.4f}')

        # ── RF permutation importances ────────────────────────────────────────
        rf_imp = r['rf_imp'].head(top_n_rf).reset_index(drop=True)
        max_imp = rf_imp['importance'].max()
        print(f'\n  RF Permutation Importance (top {top_n_rf}, mean ΔR²):')
        for _, row_imp in rf_imp.iterrows():
            bar_len = int(round(row_imp['importance'] / max_imp * 30)) if max_imp > 0 else 0
            bar = '█' * bar_len
            print(f'    {row_imp["feature"]:>15s}  {row_imp["importance"]:+.4f} ± {row_imp["std"]:.4f}  {bar}')

    print('\n' + '='*80)

print_equations_and_importances(results)


# ─────────────────────────────────────────────────────────────────────────────
# ── FIGURE 1: Feature Importance  (2 proxy × 2 target = 4 panels × 2 models)
# ─────────────────────────────────────────────────────────────────────────────
def plot_importance(results, out):
    proxy_target_pairs = [('brGDGTs','MAAT'),('brGDGTs','pH'),
                          ('3-OH FAs','MAAT'),('3-OH FAs','pH')]
    scales = ['regional','global']
    fig, axes = plt.subplots(4, 4, figsize=(22, 22))
    fig.suptitle('Feature Importance: MLR Standardised Coefficients & RF Permutation Importance',
                 fontsize=14, fontweight='bold', y=1.01)

    for row_i, (proxy, tgt) in enumerate(proxy_target_pairs):
        for col_i, scale in enumerate(scales):
            key = f'{proxy}_{scale}_{tgt}'
            if key not in results:
                continue
            r = results[key]
            # MLR importance (standardised coef)
            ax_m = axes[row_i, col_i*2]
            mi = r['mlr_imp'].head(15)
            colors_m = ['#E64B35' if v >= 0 else '#4DBBD5' for v in mi['coef']]
            ax_m.barh(mi['feature'], mi['std_coef'], color=colors_m, edgecolor='k', linewidth=0.4)
            ax_m.set_title(f'{proxy} {scale.capitalize()} {tgt}\nMLR |std coef|', fontsize=9)
            ax_m.set_xlabel('|Standardised coefficient|', fontsize=8)
            ax_m.invert_yaxis()
            ax_m.grid(axis='x', alpha=0.3)

            # RF permutation importance
            ax_r = axes[row_i, col_i*2+1]
            ri = r['rf_imp'].head(15)
            ax_r.barh(ri['feature'], ri['importance'],
                      xerr=ri['std'], color='#4CAF50', edgecolor='k', linewidth=0.4,
                      error_kw={'elinewidth':0.8,'capsize':3})
            ax_r.set_title(f'{proxy} {scale.capitalize()} {tgt}\nRF Permutation Importance', fontsize=9)
            ax_r.set_xlabel('Mean decrease in R²', fontsize=8)
            ax_r.invert_yaxis()
            ax_r.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{out}/Fig1_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] Fig1_feature_importance.png')

plot_importance(results, OUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# ── FIGURE 2: Driver Cutoff Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_cutoff(results, out):
    proxy_target_pairs = [('brGDGTs','MAAT'),('brGDGTs','pH'),
                          ('3-OH FAs','MAAT'),('3-OH FAs','pH')]
    scales = ['regional','global']
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle('Minimum Driver Cutoff: CV Performance vs Number of Features',
                 fontsize=13, fontweight='bold')

    for row_i, (proxy, tgt) in enumerate(proxy_target_pairs):
        ax = axes[row_i, 0] if scales[0] else None  # reuse ax slots differently
        for col_i, scale in enumerate(scales):
            key = f'{proxy}_{scale}_{tgt}'
            if key not in results:
                continue
            r = results[key]
            c = r['cutoff']
            ax = axes[row_i, col_i]

            ax2 = ax.twinx()
            lns = []
            lns += ax.plot(c['k'], c['rmse_lr'], 'o-', color=PALETTE['mlr'],
                           ms=4, lw=1.5, label='MLR RMSE')
            lns += ax.plot(c['k'], c['rmse_rf'], 's-', color=PALETTE['rf'],
                           ms=4, lw=1.5, label='RF RMSE')
            lns += ax2.plot(c['k'], c['r2_lr'], 'o--', color=PALETTE['mlr'],
                            ms=4, lw=1.2, alpha=0.7, label='MLR R²')
            lns += ax2.plot(c['k'], c['r2_rf'], 's--', color=PALETTE['rf'],
                            ms=4, lw=1.2, alpha=0.7, label='RF R²')

            # Mark elbow: biggest improvement in RF RMSE
            diff = np.abs(np.diff(c['rmse_rf'].values))
            elbow = int(np.argmax(diff)) + 1  # 0-indexed → 1-indexed feature count
            ax.axvline(elbow, color='grey', lw=1.2, ls=':', alpha=0.8)
            ax.text(elbow + 0.2, ax.get_ylim()[1]*0.95,
                    f'elbow k={elbow}\n({c.iloc[elbow-1]["feature"]})',
                    fontsize=7, color='grey', va='top')

            ax.set_xlabel('Number of features (ranked by RF importance)', fontsize=8)
            ax.set_ylabel('CV-RMSE', fontsize=8)
            ax2.set_ylabel('CV-R²', fontsize=8, color='grey')
            ax.set_title(f'{proxy} {scale.capitalize()} — {tgt}  (n={r["n"]})', fontsize=9)
            ax.grid(alpha=0.3)
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(f'{out}/Fig2_driver_cutoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] Fig2_driver_cutoff.png')

plot_cutoff(results, OUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# ── FIGURE 3: Residual Diagnostics (predicted vs observed + residuals + hist)
# ─────────────────────────────────────────────────────────────────────────────
def plot_residuals(results, out):
    keys_ordered = [
        ('brGDGTs_regional_MAAT',  'brGDGTs_global_MAAT'),
        ('brGDGTs_regional_pH',    'brGDGTs_global_pH'),
        ('3-OH FAs_regional_MAAT', '3-OH FAs_global_MAAT'),
        ('3-OH FAs_regional_pH',   '3-OH FAs_global_pH'),
    ]
    for k_reg, k_glob in keys_ordered:
        fig, axes = plt.subplots(2, 6, figsize=(26, 9))
        for row_idx, key in enumerate([k_reg, k_glob]):
            if key not in results:
                continue
            r = results[key]
            y, yp_lr, yp_rf = r['y'], r['yp_lr'], r['yp_rf']
            res_lr = y - yp_lr
            res_rf = y - yp_rf
            scale_lbl = r['scale'].capitalize()

            for col_base, (yp, res, lbl, col) in enumerate([
                (yp_lr, res_lr, 'MLR', PALETTE['mlr']),
                (yp_rf, res_rf, 'RF',  PALETTE['rf'])
            ]):
                c0 = col_base * 3
                ax_obs  = axes[row_idx, c0]
                ax_res  = axes[row_idx, c0+1]
                ax_hist = axes[row_idx, c0+2]
                rm, r2  = metrics(y, yp)

                # Predicted vs observed
                lims = (min(y.min(), yp.min())-1, max(y.max(), yp.max())+1)
                ax_obs.scatter(y, yp, s=20, alpha=0.5, color=col, edgecolors='k', lw=0.3)
                ax_obs.plot(lims, lims, 'k--', lw=1)
                ax_obs.set_xlim(lims); ax_obs.set_ylim(lims)
                ax_obs.set_xlabel(f'Observed {r["target"]}', fontsize=8)
                ax_obs.set_ylabel(f'Predicted {r["target"]}', fontsize=8)
                ax_obs.set_title(f'{scale_lbl} {lbl}\nRMSE={rm:.3f}  CV-R²={r2:.3f}', fontsize=8)
                ax_obs.grid(alpha=0.3)

                # Residuals vs predicted
                ax_res.scatter(yp, res, s=20, alpha=0.5, color=col, edgecolors='k', lw=0.3)
                ax_res.axhline(0, color='k', lw=1, ls='--')
                si = np.argsort(yp)
                win = max(3, len(yp)//8)
                roll = pd.Series(res[si]).rolling(win, center=True, min_periods=1).mean().values
                ax_res.plot(yp[si], roll, color='red', lw=1.5, label='Trend')
                ax_res.set_xlabel(f'Predicted {r["target"]}', fontsize=8)
                ax_res.set_ylabel('Residual', fontsize=8)
                ax_res.set_title('Residuals vs Predicted', fontsize=8)
                ax_res.legend(fontsize=7); ax_res.grid(alpha=0.3)

                # Histogram
                ax_hist.hist(res, bins=20, color=col, edgecolor='k', alpha=0.7, density=True)
                mu, sd = res.mean(), res.std()
                xn = np.linspace(res.min(), res.max(), 100)
                ax_hist.plot(xn, norm.pdf(xn, mu, sd), 'k-', lw=1.5)
                ax_hist.set_xlabel('Residual', fontsize=8)
                ax_hist.set_ylabel('Density', fontsize=8)
                ax_hist.set_title(f'μ={mu:.2f}  σ={sd:.2f}', fontsize=8)
                ax_hist.grid(alpha=0.3)

        proxy_lbl = results[k_reg]['proxy'].replace(' ','_').replace('-','')
        tgt_lbl   = results[k_reg]['target']
        fig.suptitle(f'Residual Diagnostics — {results[k_reg]["proxy"]} — {tgt_lbl}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        fname = f'{out}/Fig3_residuals_{proxy_lbl}_{tgt_lbl}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  [Saved] {fname}')

plot_residuals(results, OUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# ── FIGURE 4: Regional vs Global Comparison Panels
# ─────────────────────────────────────────────────────────────────────────────
def plot_reg_glob_comparison(results, out):
    """
    Side-by-side: predicted vs observed, colour-coded regional/global,
    plus bar chart of RMSE/R² for MLR and RF.
    """
    combos = [
        ('brGDGTs',  'MAAT'), ('brGDGTs',  'pH'),
        ('3-OH FAs', 'MAAT'), ('3-OH FAs', 'pH'),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.suptitle('Regional vs Global Model Comparison', fontsize=14, fontweight='bold')

    for row_i, (proxy, tgt) in enumerate(combos):
        ax_obs = axes[row_i, 0]   # predicted vs observed overlay
        ax_bar = axes[row_i, 1]   # RMSE bars
        ax_r2  = axes[row_i, 2]   # R² bars

        bar_labels, rmse_vals, r2_vals, bar_colors = [], [], [], []

        for scale in ['regional', 'global']:
            key = f'{proxy}_{scale}_{tgt}'
            if key not in results:
                continue
            r = results[key]
            col = PALETTE[scale]
            marker = 'o' if scale == 'regional' else 's'

            for lbl, yp, mod_col, ls in [
                ('MLR', r['yp_lr'], PALETTE['mlr'], '-'),
                ('RF',  r['yp_rf'], PALETTE['rf'],  '--'),
            ]:
                rm, r2 = metrics(r['y'], yp)
                lims = (min(r['y'].min(), yp.min())-1, max(r['y'].max(), yp.max())+1)
                ax_obs.scatter(r['y'], yp, s=15, alpha=0.45, color=col,
                               marker=marker, edgecolors='none')
                bar_labels.append(f'{scale[:3].upper()}\n{lbl}')
                rmse_vals.append(rm)
                r2_vals.append(max(r2, -0.5))
                bar_colors.append(col if lbl == 'MLR' else mod_col)

        ax_obs.plot([ax_obs.get_xlim()[0], ax_obs.get_xlim()[1]],
                   [ax_obs.get_xlim()[0], ax_obs.get_xlim()[1]], 'k--', lw=1)
        ax_obs.set_xlabel(f'Observed {tgt}', fontsize=9)
        ax_obs.set_ylabel(f'Predicted {tgt}', fontsize=9)
        ax_obs.set_title(f'{proxy} — {tgt}\nPredicted vs Observed', fontsize=9)
        ax_obs.grid(alpha=0.3)
        legend_els = [Line2D([0],[0], marker='o', color='w', markerfacecolor=PALETTE['regional'],
                              label='Regional', ms=8),
                      Line2D([0],[0], marker='s', color='w', markerfacecolor=PALETTE['global'],
                              label='Global', ms=8)]
        ax_obs.legend(handles=legend_els, fontsize=8)

        xs = np.arange(len(bar_labels))
        ax_bar.bar(xs, rmse_vals, color=bar_colors, edgecolor='k', linewidth=0.5)
        ax_bar.set_xticks(xs); ax_bar.set_xticklabels(bar_labels, fontsize=8)
        ax_bar.set_ylabel('CV-RMSE', fontsize=9)
        ax_bar.set_title('CV-RMSE by Model', fontsize=9)
        ax_bar.grid(axis='y', alpha=0.3)

        ax_r2.bar(xs, r2_vals, color=bar_colors, edgecolor='k', linewidth=0.5)
        ax_r2.axhline(0, color='k', lw=0.8)
        ax_r2.set_xticks(xs); ax_r2.set_xticklabels(bar_labels, fontsize=8)
        ax_r2.set_ylabel('CV-R²', fontsize=9)
        ax_r2.set_title('CV-R² by Model', fontsize=9)
        ax_r2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{out}/Fig4_regional_global_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] Fig4_regional_global_comparison.png')

plot_reg_glob_comparison(results, OUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# ── FIGURE 5: Unified MLR on shared top-N drivers
# ─────────────────────────────────────────────────────────────────────────────
def find_shared_top_drivers(results, proxy, tgt, top_n=10):
    """
    Get top_n RF-importance ranked features from both regional and global,
    return their intersection (in order of global rank).
    """
    key_reg  = f'{proxy}_regional_{tgt}'
    key_glob = f'{proxy}_global_{tgt}'
    top_reg  = results[key_reg]['ranked'][:top_n]
    top_glob = results[key_glob]['ranked'][:top_n]
    shared   = [f for f in top_glob if f in top_reg]
    return shared

def plot_unified(results, out):
    combos = [
        ('brGDGTs',  'MAAT', df_br_reg,  df_br_glob,  BRGDGT_FEATS),
        ('brGDGTs',  'pH',   df_br_reg,  df_br_glob,  BRGDGT_FEATS),
        ('3-OH FAs', 'MAAT', df_fa_reg,  df_fa_glob,  OHFA_FEATS),
        ('3-OH FAs', 'pH',   df_fa_reg,  df_fa_glob,  OHFA_FEATS),
    ]

    print('\n' + '='*75)
    print('UNIFIED MLR EQUATIONS (pooled regional + global, shared top drivers)')
    print('='*75)

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle('Unified (Pooled) MLR: Shared Top Drivers\nPredicted vs Observed & Coefficients',
                 fontsize=12, fontweight='bold')

    for row_i, (proxy, tgt, df_reg, df_glob, all_feats) in enumerate(combos):
        shared = find_shared_top_drivers(results, proxy, tgt, top_n=10)
        if len(shared) < 2:
            shared = all_feats  # fallback: use all

        uni = unified_mlr(df_reg, df_glob, shared, tgt)

        print(f'\n{proxy} — {tgt}  (n_pool={uni["n"]}, {uni["cv_label"]})')
        print(f'  Shared features ({len(shared)}): {shared}')
        print(f'  CV-RMSE = {uni["cv_rmse"]:.3f}  |  CV-R² = {uni["cv_r2"]:.3f}')
        print(f'  Equation: {uni["eq"]}')

        # Predicted vs observed
        ax_obs = axes[row_i, 0]
        y, yp  = uni['y'], uni['yp']
        lims   = (min(y.min(), yp.min())-1, max(y.max(), yp.max())+1)
        ax_obs.scatter(y, yp, s=20, alpha=0.5, color='#7E57C2', edgecolors='k', lw=0.3)
        ax_obs.plot(lims, lims, 'k--', lw=1)
        ax_obs.set_xlim(lims); ax_obs.set_ylim(lims)
        ax_obs.set_xlabel(f'Observed {tgt}', fontsize=9)
        ax_obs.set_ylabel(f'Predicted {tgt}', fontsize=9)
        ax_obs.set_title(f'{proxy} — {tgt}\nUnified MLR  CV-RMSE={uni["cv_rmse"]:.3f}  CV-R²={uni["cv_r2"]:.3f}',
                         fontsize=9)
        ax_obs.grid(alpha=0.3)

        # Coefficient plot
        ax_coef = axes[row_i, 1]
        lr_tmp  = LinearRegression()
        X_pool  = pd.concat([df_reg[shared], df_glob[shared]], ignore_index=True)
        y_pool  = pd.concat([df_reg[tgt],    df_glob[tgt]],    ignore_index=True)
        mask    = X_pool.notna().all(axis=1) & y_pool.notna()
        lr_tmp.fit(X_pool[mask], y_pool[mask])
        sc_tmp  = StandardScaler().fit_transform(X_pool[mask])
        lr_sc   = LinearRegression().fit(sc_tmp, y_pool[mask])
        coef_df = pd.DataFrame({'feature': shared, 'std_coef': lr_sc.coef_,
                                'coef': lr_tmp.coef_}).sort_values('std_coef')
        colors  = ['#E64B35' if v > 0 else '#4DBBD5' for v in coef_df['std_coef']]
        ax_coef.barh(coef_df['feature'], coef_df['std_coef'], color=colors,
                     edgecolor='k', linewidth=0.4)
        ax_coef.axvline(0, color='k', lw=0.8)
        ax_coef.set_xlabel('Standardised coefficient', fontsize=9)
        ax_coef.set_title('Unified MLR Coefficients', fontsize=9)
        ax_coef.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{out}/Fig5_unified_mlr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  [Saved] Fig5_unified_mlr.png')

plot_unified(results, OUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# ── FIGURE 6: Top-driver overlap heatmap (regional vs global, both proxies)
# ─────────────────────────────────────────────────────────────────────────────
def plot_driver_overlap(results, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Top-10 RF Driver Overlap: Regional vs Global Importance Rank',
                 fontsize=12, fontweight='bold')

    combos = [('brGDGTs','MAAT'),('brGDGTs','pH'),('3-OH FAs','MAAT'),('3-OH FAs','pH')]
    for ax, (proxy, tgt) in zip(axes.flatten(), combos):
        key_r = f'{proxy}_regional_{tgt}'
        key_g = f'{proxy}_global_{tgt}'
        if key_r not in results or key_g not in results:
            continue
        top_r = results[key_r]['ranked'][:10]
        top_g = results[key_g]['ranked'][:10]
        all_f = list(dict.fromkeys(top_r + top_g))

        mat = np.zeros((len(all_f), 2))
        for i, f in enumerate(all_f):
            mat[i, 0] = (top_r.index(f) + 1) if f in top_r else 0
            mat[i, 1] = (top_g.index(f) + 1) if f in top_g else 0

        shared_mask = [(f in top_r and f in top_g) for f in all_f]
        colors_row  = ['#FFB300' if s else '#BDBDBD' for s in shared_mask]

        im = ax.imshow(mat.T, aspect='auto', cmap='YlOrRd_r',
                       vmin=0, vmax=10, interpolation='nearest')
        ax.set_xticks(range(len(all_f)))
        ax.set_xticklabels(all_f, rotation=45, ha='right', fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Regional', 'Global'], fontsize=9)
        ax.set_title(f'{proxy} — {tgt}\n(gold = shared top-10)', fontsize=10)
        for i, col in enumerate(colors_row):
            if col == '#FFB300':
                ax.add_patch(plt.Rectangle((i-0.5, -0.5), 1, 2.0,
                                           fill=False, edgecolor='#FFB300', lw=2.5))
        plt.colorbar(im, ax=ax, label='Rank (0=not in top-10)', shrink=0.7)

    plt.tight_layout()
    fig.savefig(f'{out}/Fig6_driver_overlap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  [Saved] Fig6_driver_overlap.png')

plot_driver_overlap(results, OUT_PATH)

print('\n✓ All figures and equations written to:', OUT_PATH)