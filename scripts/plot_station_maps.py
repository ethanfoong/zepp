"""
Plot skewness and variance from NetCDF files on a world map using cartopy.

Run: conda activate zepp; python scripts\plot_station_maps.py
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import skew

# Station locations (lat, lon) - NDBC buoy positions
STATION_LOCATIONS = {
    '46001': (56.3, -148.1),   # 150 NM West of Cape St. James, AK
    '46014': (39.2, -123.9),   # Pt Arena - 60 NM West of San Francisco, CA  
    '46025': (33.8, -119.1),   # 33 NM West of Santa Monica, CA
    '46278': (37.5, -122.6),   # GGWSC San Francisco West Coast
}

NC_DIR = Path("nc")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)


def calculate_statistics_from_netcdf(nc_path):
    """
    skewness and variance calculation from a NetCDF file.
    """

    ds = xr.open_dataset(nc_path)
    anomalies = ds['anomalies'].values.flatten()
    
    # NaN removal 
    anomalies = anomalies[~np.isnan(anomalies)]
    
    if len(anomalies) == 0:
        return np.nan, np.nan
    

    variance = np.var(anomalies)
    skewness = skew(anomalies)
    
    return skewness, variance


def plot_map(lons, lats, values, title, cmap, vmin=None, vmax=None, 
             output_path=None, cbar_label='Value'):
    """
    cartopy mapping
    
    params:
        lons: list of longitudes
        lats: list of latitudes
        values: list of values to plot
        title: plot title
        cmap: colormap name
        vmin, vmax: color scale limits
        output_path: path to save figure
        cbar_label: colorbar label
    """
    fig = plt.figure(figsize=(14, 8))
    
    # platcarree projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                 linewidth=0.5, alpha=0.5, linestyle='--')
    
    # north specific granularity
    ax.set_extent([-180, -100, 25, 65], crs=ccrs.PlateCarree())
    
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    
    scatter = ax.scatter(lons, lats, c=values, s=200, 
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors='black', linewidths=2,
                        transform=ccrs.PlateCarree(), zorder=5)
    
    #station labels
    for lon, lat, val, in zip(lons, lats, values):
        ax.text(lon, lat + 1.5, f'{val:.2f}', 
               ha='center', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               transform=ccrs.PlateCarree())
    
    # colors
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


def main():
    """Main function to generate both skewness and variance maps."""
    
    # finding all netcdf files 
    nc_files = sorted(NC_DIR.glob("*_warm_anomalies_*d.nc"))
    
    if not nc_files:
        print(f"No NetCDF files found in {NC_DIR}")
        return
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    # extract statistics for each station
    stations = []
    lons = []
    lats = []
    skewness_values = []
    variance_values = []
    
    for nc_file in nc_files:
        # Extract station ID from filename
        station_id = nc_file.stem.split('_')[0]
        
        if station_id not in STATION_LOCATIONS:
            print(f"Warning: Station {station_id} location not found, skipping")
            continue
        
        # Calculate statistics
        skewness, variance = calculate_statistics_from_netcdf(nc_file)
        
        if np.isnan(skewness) or np.isnan(variance):
            print(f"Warning: Station {station_id} has no valid data, skipping")
            continue
        
        # Get location
        lat, lon = STATION_LOCATIONS[station_id]
        
        stations.append(station_id)
        lons.append(lon)
        lats.append(lat)
        skewness_values.append(skewness)
        variance_values.append(variance)
        
        print(f"Station {station_id}: skewness={skewness:.3f}, variance={variance:.3f}")
    
    if not stations:
        print("No valid stations to plot")
        return
    
    print(f"\nPlotting {len(stations)} stations: {', '.join(stations)}")
    
    # Plot 1: Skewness map
    fig1, ax1 = plot_map(
        lons, lats, skewness_values,
        title='Warm Season Temperature Anomaly Skewness by NDBC Station',
        cmap='RdBu_r',  # Red for positive skew (right tail), blue for negative
        vmin=-1.0, vmax=1.0,
        output_path=OUT_DIR / 'station_skewness_map.png',
        cbar_label='Skewness'
    )
    
    # Plot 2: Variance map
    fig2, ax2 = plot_map(
        lons, lats, variance_values,
        title='Warm Season Temperature Anomaly Variance by NDBC Station',
        cmap='YlOrRd',  # Yellow to red for increasing variance
        output_path=OUT_DIR / 'station_variance_map.png',
        cbar_label='Variance (°C²)'
    )
    
    # Create a summary table
    print("\n" + "="*60)
    print("Summary Statistics by Station")
    print("="*60)
    print(f"{'Station':<10} {'Latitude':<10} {'Longitude':<12} {'Skewness':<12} {'Variance':<12}")
    print("-"*60)
    for sid, lat, lon, sk, var in zip(stations, lats, lons, skewness_values, variance_values):
        print(f"{sid:<10} {lat:<10.2f} {lon:<12.2f} {sk:<12.3f} {var:<12.3f}")
    print("="*60)
    
    plt.show()


if __name__ == "__main__":
    main()
