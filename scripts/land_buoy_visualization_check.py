import os
import sys
import pickle

# Ensure project root is importable when this file is executed directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.average_anomaly_pipeline import (
    TX_LAND_STATION_FILE,
    plot_min_anomaly_histograms_by_coast_with_tmin_land_stations,
    plot_min_anomaly_pdf_with_tmin_land_stations,
    plot_min_anomaly_with_tmin_land_stations,
    plot_extreme_anomaly_with_land_stations,
)

with open(os.path.join('cache', 'average_anomaly_analysis.pkl'), 'rb') as f:
    results = pickle.load(f)

min_out = plot_min_anomaly_with_tmin_land_stations(
    results,
    output_name='min_anomaly_with_TMIN_land_stations.png',
    west_coast_only=False,
)

min_tx_out = plot_extreme_anomaly_with_land_stations(
    results,
    anomaly_type='min',
    land_station_nc_path=TX_LAND_STATION_FILE,
    output_name='min_anomaly_with_TX_land_stations.png',
    west_coast_only=False,
    land_station_label='TX land stations',
)

pdf_out = plot_min_anomaly_pdf_with_tmin_land_stations(
    results,
    output_name='min_anomaly_pdf_buoy_vs_tmin_land.png',
    west_coast_only=False,
)

hist_by_coast_out = plot_min_anomaly_histograms_by_coast_with_tmin_land_stations(
    results,
    output_name='min_anomaly_histograms_by_coast_buoy_vs_tmin_land.png',
    bins=28,
)

print('min output:', min_out)
print('min + TX output:', min_tx_out)
print('pdf output:', pdf_out)
print('hist by coast output:', hist_by_coast_out)
