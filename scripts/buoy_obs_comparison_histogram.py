import argparse
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


DEFAULT_TX_FILE = os.path.join('nc', 'TX_Land_Station.nc')
DEFAULT_BUOY_RESULTS = os.path.join('cache', 'average_anomaly_analysis.pkl')
DEFAULT_OUTPUT = os.path.join('figures', 'buoy_obs_comparison_histogram.png')

WEST_BOUNDS = {
    'lat_min': 30.0,
    'lat_max': 50.0,
    'lon_min': -130.0,
    'lon_max': -115.0,
}

EAST_BOUNDS = {
    'lat_min': 25.0,
    'lat_max': 45.0,
    'lon_min': -85.0,
    'lon_max': -65.0,
}


def load_land_tx_values(nc_path, coast='all'):
    """Load land-station Tx values (variable D) and optionally coast-filter them."""
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f'TX NetCDF file not found: {nc_path}')

    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        d_vals = np.asarray(ds['D'].values, dtype=float)
        lat = np.asarray(ds['lat'].values, dtype=float)
        lon = np.asarray(ds['lon'].values, dtype=float)
        ds.close()
    except Exception:
        from netCDF4 import Dataset
        nc = Dataset(nc_path, 'r')
        d_vals = np.asarray(nc['D'][:], dtype=float)
        lat = np.asarray(nc['lat'][:], dtype=float)
        lon = np.asarray(nc['lon'][:], dtype=float)
        nc.close()

    mask = np.isfinite(d_vals) & np.isfinite(lat) & np.isfinite(lon)

    if coast == 'west':
        mask &= (
            (lat >= WEST_BOUNDS['lat_min']) & (lat <= WEST_BOUNDS['lat_max'])
            & (lon >= WEST_BOUNDS['lon_min']) & (lon <= WEST_BOUNDS['lon_max'])
        )
    elif coast == 'east':
        mask &= (
            (lat >= EAST_BOUNDS['lat_min']) & (lat <= EAST_BOUNDS['lat_max'])
            & (lon >= EAST_BOUNDS['lon_min']) & (lon <= EAST_BOUNDS['lon_max'])
        )

    return d_vals[mask]


def load_buoy_avg_max_tx_values(results_path, coast='all'):
    """Load buoy average max Tx values from cached pipeline results."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f'Buoy results file not found: {results_path}')

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    values = []
    for record in results:
        if record.get('anomaly_type') != 'max':
            continue

        if coast in ('west', 'east') and record.get('coast') != coast:
            continue

        avg_tx = record.get('average_anomaly', np.nan)
        if np.isfinite(avg_tx):
            values.append(float(avg_tx))

    return np.asarray(values, dtype=float)


def plot_single_histogram(ax, land_vals, buoy_vals, bins, coast):
    """Plot a single histogram with KDE/PDF overlays on the given axes."""
    all_vals = np.concatenate([land_vals, buoy_vals])
    x_min = float(np.nanmin(all_vals))
    x_max = float(np.nanmax(all_vals))

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError('Histogram bounds are not finite.')
    if x_min == x_max:
        x_max = x_min + 1e-6

    bin_edges = np.linspace(x_min, x_max, bins + 1)

    ax.hist(
        land_vals,
        bins=bin_edges,
        density=True,
        alpha=0.55,
        color='#2a9d8f',
        edgecolor='white',
        linewidth=0.8,
        label=f'Land observations (n={land_vals.size})',
    )
    ax.hist(
        buoy_vals,
        bins=bin_edges,
        density=True,
        alpha=0.55,
        color='#e76f51',
        edgecolor='white',
        linewidth=0.8,
        label=f'Buoy stations (n={buoy_vals.size})',
    )

    # KDE/PDF trend overlays in TSKEW-style format.
    x_grid = np.linspace(x_min, x_max, 600)

    try:
        land_pdf = gaussian_kde(land_vals)(x_grid)
        ax.plot(x_grid, land_pdf, color='#1f6f66', linewidth=2.6, label='Land PDF trend', zorder=8)
    except Exception:
        pass

    try:
        buoy_pdf = gaussian_kde(buoy_vals)(x_grid)
        ax.plot(x_grid, buoy_pdf, color='#b74a2f', linewidth=2.6, label='Buoy PDF trend', zorder=8)
    except Exception:
        pass

    land_mean = float(np.nanmean(land_vals))
    buoy_mean = float(np.nanmean(buoy_vals))
    ax.axvline(land_mean, color='#2a9d8f', linestyle='--', linewidth=1.5, alpha=0.95)
    ax.axvline(buoy_mean, color='#e76f51', linestyle='--', linewidth=1.5, alpha=0.95)

    coast_label = {'west': 'West Coast', 'east': 'East Coast'}[coast]
    ax.set_title(f'{coast_label}', fontsize=13, weight='bold')
    ax.set_xlabel('Average Max Tx (degC)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(framealpha=0.9, fontsize=10)


def plot_normalized_histogram(land_vals_east, buoy_vals_east, land_vals_west, buoy_vals_west, output_path, bins=25):
    """Plot east and west coast histograms side by side for comparison."""
    if land_vals_east.size == 0:
        raise ValueError('No land-station Tx values available for east coast.')
    if buoy_vals_east.size == 0:
        raise ValueError('No buoy average max Tx values available for east coast.')
    if land_vals_west.size == 0:
        raise ValueError('No land-station Tx values available for west coast.')
    if buoy_vals_west.size == 0:
        raise ValueError('No buoy average max Tx values available for west coast.')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax_west, ax_east) = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    # West coast histogram (left)
    plot_single_histogram(ax_west, land_vals_west, buoy_vals_west, bins, 'west')

    # East coast histogram (right)
    plot_single_histogram(ax_east, land_vals_east, buoy_vals_east, bins, 'east')

    fig.suptitle(
        'Histogram PDFs of Average Max Tx\nLand Observations vs Buoy Stations by Coast',
        fontsize=14,
        weight='bold',
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create side-by-side histograms with PDF overlays comparing east and west coast land Tx and buoy average max Tx distributions.'
    )
    parser.add_argument('--tx-file', default=DEFAULT_TX_FILE, help='Path to TX_Land_Station NetCDF file.')
    parser.add_argument('--buoy-results', default=DEFAULT_BUOY_RESULTS, help='Path to buoy results pickle file.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output histogram image path.')
    parser.add_argument('--bins', type=int, default=25, help='Number of histogram bins.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data for east coast
    land_vals_east = load_land_tx_values(args.tx_file, coast='east')
    buoy_vals_east = load_buoy_avg_max_tx_values(args.buoy_results, coast='east')

    # Load data for west coast
    land_vals_west = load_land_tx_values(args.tx_file, coast='west')
    buoy_vals_west = load_buoy_avg_max_tx_values(args.buoy_results, coast='west')

    plot_normalized_histogram(
        land_vals_east=land_vals_east,
        buoy_vals_east=buoy_vals_east,
        land_vals_west=land_vals_west,
        buoy_vals_west=buoy_vals_west,
        output_path=args.output,
        bins=args.bins,
    )

    print(f'[OK] Saved histogram: {args.output}')
    print(f'     East Coast - Land sample size: {land_vals_east.size}, Buoy sample size: {buoy_vals_east.size}')
    print(f'     West Coast - Land sample size: {land_vals_west.size}, Buoy sample size: {buoy_vals_west.size}')


if __name__ == '__main__':
    main()
