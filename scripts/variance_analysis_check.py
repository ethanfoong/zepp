import os
import sys
import pickle

# Ensure project root is importable when this file is executed directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.average_anomaly_pipeline import (
    plot_variance_comparison_pdf_with_tvar_land_stations,
    plot_variance_histograms_by_coast_with_tvar_land_stations,
    plot_variance_with_tvar_land_stations,
    plot_skewness_comparison_pdf_with_tskew_land_stations,
    plot_skewness_histograms_by_coast_with_tskew_land_stations,
    plot_skewness_with_tskew_land_stations,
)

with open(os.path.join('cache', 'average_anomaly_analysis.pkl'), 'rb') as f:
    results = pickle.load(f)

pdf_out = plot_variance_comparison_pdf_with_tvar_land_stations(
    results,
    output_name='variance_pdf_buoy_vs_tvar_land.png',
    west_coast_only=False,
)

hist_by_coast_out = plot_variance_histograms_by_coast_with_tvar_land_stations(
    results,
    output_name='variance_histograms_by_coast_buoy_vs_tvar_land.png',
    bins=28,
)

map_out = plot_variance_with_tvar_land_stations(
    results,
    output_name='variance_with_TVAR_land_stations.png',
    west_coast_only=False,
)

print('variance pdf output:', pdf_out)
print('variance hist by coast output:', hist_by_coast_out)
print('variance map output:', map_out)

# TSKEW Analysis
skew_pdf_out = plot_skewness_comparison_pdf_with_tskew_land_stations(
    results,
    output_name='skewness_pdf_buoy_vs_tskew_land.png',
    west_coast_only=False,
)

skew_hist_out = plot_skewness_histograms_by_coast_with_tskew_land_stations(
    results,
    output_name='skewness_histograms_by_coast_buoy_vs_tskew_land.png',
    bins=28,
)

skew_map_out = plot_skewness_with_tskew_land_stations(
    results,
    output_name='skewness_with_TSKEW_land_stations.png',
    west_coast_only=False,
)

print('skewness pdf output:', skew_pdf_out)
print('skewness hist by coast output:', skew_hist_out)
print('skewness map output:', skew_map_out)
