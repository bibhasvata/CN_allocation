generate_manuscript_figures.py
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
