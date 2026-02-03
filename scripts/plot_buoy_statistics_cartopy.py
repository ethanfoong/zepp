"""
Plot NDBC Buoy Statistics on World Maps using Cartopy
======================================================
Visualizes skewness and variance statistics from all processed buoys
on world maps using Cartopy projection.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = "cache"
FIGURES_DIR = "figures"


def load_buoy_results(input_file="buoy_statistics.pkl"):
    """Load processed buoy statistics."""
    input_path = os.path.join(CACHE_DIR, input_file)
    
    if not os.path.exists(input_path):
        print(f"❌ No results found at {input_path}")
        print("   Run process_all_buoys.py first to generate statistics.")
        return None
    
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    
    df = pd.DataFrame(results)
    df = df.dropna(subset=['latitude', 'longitude'])
    
    print(f"📂 Loaded {len(df)} buoy stations with location data")
    return df


def create_cartopy_maps(df, figsize=(20, 10), save=True):
    """
    Create side-by-side world maps showing variance and skewness.
    
    Parameters:
        df: DataFrame with columns ['latitude', 'longitude', 'mean_variance', 'mean_skewness']
        figsize: Figure size tuple
        save: Whether to save the figure
    """
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    
    # Define projection (PlateCarree for global view)
    projection = ccrs.PlateCarree()
    
    # Determine extent based on data
    # If mostly Pacific coast, zoom in; otherwise show global
    lat_range = df['latitude'].max() - df['latitude'].min()
    lon_range = df['longitude'].max() - df['longitude'].min()
    
    if lat_range < 50 and lon_range < 100:
        # Regional view (e.g., US West Coast)
        extent = [df['longitude'].min() - 5, df['longitude'].max() + 5,
                  df['latitude'].min() - 5, df['latitude'].max() + 5]
    else:
        # Global view
        extent = None
    
    # ============ VARIANCE MAP ============
    ax1 = fig.add_subplot(1, 2, 1, projection=projection)
    
    # Add map features
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='-', alpha=0.5)
    ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    
    if extent:
        ax1.set_extent(extent, crs=projection)
    else:
        ax1.set_global()
    
    # Normalize variance for color mapping
    variance = df['mean_variance'].values
    vmin, vmax = variance.min(), variance.max()
    norm_var = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot buoys colored by variance
    scatter1 = ax1.scatter(
        df['longitude'], df['latitude'],
        c=variance,
        cmap='YlOrRd',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5,
        norm=norm_var,
        transform=projection,
        zorder=5
    )
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', 
                        pad=0.05, shrink=0.8, aspect=30)
    cbar1.set_label('Mean Warm Season Variance [°C²]', fontsize=12, weight='bold')
    
    ax1.set_title('NDBC Buoy Warm Season Temperature Variance', 
                  fontsize=14, weight='bold', pad=10)
    
    # ============ SKEWNESS MAP ============
    ax2 = fig.add_subplot(1, 2, 2, projection=projection)
    
    # Add map features
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax2.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle='-', alpha=0.5)
    ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    
    if extent:
        ax2.set_extent(extent, crs=projection)
    else:
        ax2.set_global()
    
    # Normalize skewness with center at 0 (diverging colormap)
    skewness = df['mean_skewness'].values
    abs_max = max(abs(skewness.min()), abs(skewness.max()))
    norm_skew = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    
    # Plot buoys colored by skewness
    scatter2 = ax2.scatter(
        df['longitude'], df['latitude'],
        c=skewness,
        cmap='coolwarm',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5,
        norm=norm_skew,
        transform=projection,
        zorder=5
    )
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal',
                        pad=0.05, shrink=0.8, aspect=30)
    cbar2.set_label('Mean Warm Season Skewness', fontsize=12, weight='bold')
    
    ax2.set_title('NDBC Buoy Warm Season Temperature Skewness',
                  fontsize=14, weight='bold', pad=10)
    
    # Add overall title
    n_buoys = len(df)
    year_range = f"{int(df['year_start'].min())}-{int(df['year_end'].max())}"
    fig.suptitle(
        f'NDBC Buoy Temperature Statistics: {n_buoys} Stations ({year_range})\n'
        f'Warm Season Analysis (±100 days from climatological peak)',
        fontsize=16, weight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        output_path = os.path.join(FIGURES_DIR, 'buoy_statistics_cartopy_maps.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Figure saved to {output_path}")
    
    plt.show()
    
    return fig


def create_detailed_regional_maps(df, regions=None, figsize=(20, 12), save=True):
    """
    Create detailed regional maps for different ocean basins.
    
    Parameters:
        df: DataFrame with buoy statistics
        regions: Dict of region definitions {name: (lon_min, lon_max, lat_min, lat_max)}
        figsize: Figure size
        save: Whether to save figures
    """
    
    if regions is None:
        # Default regions
        regions = {
            'North Pacific': (-180, -100, 20, 70),
            'Northeast Pacific': (-140, -110, 30, 60),
            'California Coast': (-130, -115, 30, 50),
            'Alaska Coast': (-180, -120, 45, 70),
        }
    
    for region_name, (lon_min, lon_max, lat_min, lat_max) in regions.items():
        # Filter data for region
        df_region = df[
            (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) &
            (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)
        ]
        
        if len(df_region) < 3:
            print(f"⏭️  Skipping {region_name}: only {len(df_region)} buoys")
            continue
        
        print(f"\n📍 Creating map for {region_name} ({len(df_region)} buoys)")
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        projection = ccrs.PlateCarree()
        
        # Variance subplot
        ax1 = fig.add_subplot(1, 2, 1, projection=projection)
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
        
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        
        variance = df_region['mean_variance'].values
        norm_var = Normalize(vmin=variance.min(), vmax=variance.max())
        
        scatter1 = ax1.scatter(
            df_region['longitude'], df_region['latitude'],
            c=variance, cmap='YlOrRd', s=150, alpha=0.8,
            edgecolors='black', linewidth=1, norm=norm_var,
            transform=projection, zorder=5
        )
        
        # Add station labels
        for _, row in df_region.iterrows():
            ax1.text(row['longitude'] + 0.5, row['latitude'] + 0.5,
                    row['station_id'].upper(), fontsize=8,
                    transform=projection, zorder=6)
        
        cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal',
                           pad=0.05, shrink=0.8)
        cbar1.set_label('Variance [°C²]', fontsize=11, weight='bold')
        ax1.set_title('Warm Season Variance', fontsize=13, weight='bold')
        
        # Skewness subplot
        ax2 = fig.add_subplot(1, 2, 2, projection=projection)
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)
        
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax2.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        
        skewness = df_region['mean_skewness'].values
        abs_max = max(abs(skewness.min()), abs(skewness.max()))
        norm_skew = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        
        scatter2 = ax2.scatter(
            df_region['longitude'], df_region['latitude'],
            c=skewness, cmap='coolwarm', s=150, alpha=0.8,
            edgecolors='black', linewidth=1, norm=norm_skew,
            transform=projection, zorder=5
        )
        
        # Add station labels
        for _, row in df_region.iterrows():
            ax2.text(row['longitude'] + 0.5, row['latitude'] + 0.5,
                    row['station_id'].upper(), fontsize=8,
                    transform=projection, zorder=6)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal',
                           pad=0.05, shrink=0.8)
        cbar2.set_label('Skewness', fontsize=11, weight='bold')
        ax2.set_title('Warm Season Skewness', fontsize=13, weight='bold')
        
        fig.suptitle(f'{region_name}: NDBC Buoy Temperature Statistics',
                    fontsize=15, weight='bold')
        
        plt.tight_layout()
        
        if save:
            os.makedirs(FIGURES_DIR, exist_ok=True)
            filename = f"buoy_stats_{region_name.lower().replace(' ', '_')}.png"
            output_path = os.path.join(FIGURES_DIR, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {output_path}")
        
        plt.show()


def print_statistics_summary(df):
    """Print summary statistics of the results."""
    
    print("\n" + "="*70)
    print("📊 BUOY STATISTICS SUMMARY")
    print("="*70)
    
    print(f"\nTotal buoys analyzed: {len(df)}")
    print(f"Year range: {int(df['year_start'].min())} - {int(df['year_end'].max())}")
    print(f"Mean years per buoy: {df['num_years'].mean():.1f}")
    
    print("\n--- VARIANCE STATISTICS ---")
    print(f"Mean: {df['mean_variance'].mean():.4f} °C²")
    print(f"Std Dev: {df['mean_variance'].std():.4f} °C²")
    print(f"Range: {df['mean_variance'].min():.4f} - {df['mean_variance'].max():.4f} °C²")
    
    print("\n--- SKEWNESS STATISTICS ---")
    print(f"Mean: {df['mean_skewness'].mean():.4f}")
    print(f"Std Dev: {df['mean_skewness'].std():.4f}")
    print(f"Range: {df['mean_skewness'].min():.4f} - {df['mean_skewness'].max():.4f}")
    
    print("\n--- TOP 10 BY VARIANCE ---")
    top_var = df.nlargest(10, 'mean_variance')[
        ['station_id', 'latitude', 'longitude', 'num_years', 'mean_variance', 'mean_skewness']
    ]
    print(top_var.to_string(index=False))
    
    print("\n--- TOP 10 BY SKEWNESS (most positive) ---")
    top_skew = df.nlargest(10, 'mean_skewness')[
        ['station_id', 'latitude', 'longitude', 'num_years', 'mean_variance', 'mean_skewness']
    ]
    print(top_skew.to_string(index=False))
    
    print("\n--- BOTTOM 10 BY SKEWNESS (most negative) ---")
    bottom_skew = df.nsmallest(10, 'mean_skewness')[
        ['station_id', 'latitude', 'longitude', 'num_years', 'mean_variance', 'mean_skewness']
    ]
    print(bottom_skew.to_string(index=False))
    
    print("\n" + "="*70)


def main():
    """Main visualization pipeline."""
    
    # Load results
    df = load_buoy_results()
    
    if df is None or len(df) == 0:
        return
    
    # Print statistics summary
    print_statistics_summary(df)
    
    # Create global maps
    print("\n🗺️  Creating global Cartopy maps...")
    create_cartopy_maps(df, figsize=(20, 10), save=True)
    
    # Create regional maps
    print("\n🗺️  Creating regional maps...")
    create_detailed_regional_maps(df, save=True)
    
    print("\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()
