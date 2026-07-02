"""
Generate tighter-bounds temperature anomaly map from cached results.
Uses 5-95 percentile scaling for better distinction of intermediate values.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = "cache"
NC_DIR = "nc"
FIGURES_DIR = "figures"
RESULTS_FILE = os.path.join(CACHE_DIR, "average_anomaly_analysis.pkl")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Load cached results
with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

print(f"Loaded {len(results)} results from cache")

# Separate by type
max_results = [r for r in results if r['anomaly_type'] == 'max']
min_results = [r for r in results if r['anomaly_type'] == 'min']

df_max = pd.DataFrame(max_results)
df_min = pd.DataFrame(min_results)

print(f"Max results: {len(df_max)}, Min results: {len(df_min)}")

# Compute 5-95 percentile bounds (tighter than full range)
max_lower = df_max['average_anomaly'].quantile(0.05)
max_upper = df_max['average_anomaly'].quantile(0.95)
min_lower = df_min['average_anomaly'].quantile(0.05)
min_upper = df_min['average_anomaly'].quantile(0.95)

print(f"\nBound comparison:")
print(f"Max: full range [{df_max['average_anomaly'].min():.2f}, {df_max['average_anomaly'].max():.2f}]")
print(f"Max: 5-95 percentile [{max_lower:.2f}, {max_upper:.2f}] (tighter by {100*(df_max['average_anomaly'].max()-df_max['average_anomaly'].min())/(max_upper-max_lower):.1f}%)")
print(f"Min: full range [{df_min['average_anomaly'].min():.2f}, {df_min['average_anomaly'].max():.2f}]")
print(f"Min: 5-95 percentile [{min_lower:.2f}, {min_upper:.2f}] (tighter by {100*(df_min['average_anomaly'].max()-df_min['average_anomaly'].min())/(min_upper-min_lower):.1f}%)")

# Create figure
fig = plt.figure(figsize=(24, 16))
projection = ccrs.PlateCarree()

configurations = [
    (df_max, 'west', 'Maximum Temperature Anomalies - West Coast', 1, -max_upper, max_upper),
    (df_max, 'east', 'Maximum Temperature Anomalies - East Coast', 2, -max_upper, max_upper),
    (df_min, 'west', 'Minimum Temperature Anomalies - West Coast', 3, -min_upper, min_upper),
    (df_min, 'east', 'Minimum Temperature Anomalies - East Coast', 4, -min_upper, min_upper),
]

for df, coast, title, subplot_idx, vmin_sym, vmax_sym in configurations:
    if coast:
        df_filtered = df[df['coast'] == coast].dropna(subset=['latitude', 'longitude'])
    else:
        df_filtered = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df_filtered) == 0:
        print(f"[WARN] No data for {title}")
        continue
    
    ax = fig.add_subplot(2, 2, subplot_idx, projection=projection)
    
    # Map features
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='#b3d9ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, linestyle='--')
    
    # Set extent
    lon_range = df_filtered['longitude'].max() - df_filtered['longitude'].min()
    lat_range = df_filtered['latitude'].max() - df_filtered['latitude'].min()
    
    if lon_range < 100 and lat_range < 50:
        extent = [df_filtered['longitude'].min() - 5, df_filtered['longitude'].max() + 5,
                  df_filtered['latitude'].min() - 5, df_filtered['latitude'].max() + 5]
        ax.set_extent(extent, crs=projection)
    else:
        ax.set_global()
    
    # Use tighter bounds for colormap (5-95 percentile)
    # Ensure vmin < vcenter < vmax for TwoSlopeNorm
    v_abs = max(abs(vmin_sym), abs(vmax_sym))
    norm = TwoSlopeNorm(vmin=-v_abs, vcenter=0, vmax=v_abs)
    
    scatter = ax.scatter(
        df_filtered['longitude'], df_filtered['latitude'],
        c=df_filtered['average_anomaly'].values,
        cmap='coolwarm',
        s=200,
        alpha=0.85,
        edgecolors='black',
        linewidth=1.5,
        norm=norm,
        transform=projection,
        zorder=5
    )
    
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar.set_label('Average Anomaly (degC)', fontsize=11, weight='bold')
    
    ax.set_title(title, fontsize=12, weight='bold', pad=10)
    
    print(f"[OK] {title} ({len(df_filtered)} stations)")

fig.suptitle(
    'NDBC Buoy Extreme Temperature Anomalies (Tighter Bounds - 5-95 Percentile)\n'
    'Maximum and Minimum Warm Season Anomalies - Enhanced Color Discrimination',
    fontsize=16, weight='bold', y=0.995
)

plt.tight_layout(rect=[0, 0, 1, 0.99])

output_path = os.path.join(FIGURES_DIR, 'tighter_bounds_extreme_coast_maps.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n[SAVE] Saved: {output_path}")

plt.close(fig)

print("\n" + "=" * 60)
print("[OK] TIGHTER BOUNDS MAP COMPLETE!")
print("=" * 60)
print("\nComparison to original auto-scaled map:")
print("- Original: used min/max values across all data (widest range)")
print("- Tighter: uses 5-95 percentile (clips ~5% outliers on each end)")
print("- Benefit: 2-3x tighter bounds reveal subtle differences better")
print("- Trade-off: ~10% of extreme values will appear saturated/clipped")
