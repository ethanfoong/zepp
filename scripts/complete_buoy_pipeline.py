import os
import sys
import re
import time
import pandas as pd
import numpy as np
import pickle
import cartopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import skew
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_buoy_helpers import list_station_files, list_all_station_ids, load_station, parse_station_table, load_station_locations
from stat_buoy_helpers import write_warm_season_netcdf, get_warm_season_data, compute_warm_season_anomalies, read_netcdf_statistics

'''average biggest/ and smallest anomaly in buoy time series



'''


BASE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
CACHE_DIR = "cache"
NC_DIR = "nc"
FIGURES_DIR = "figures"
RESULTS_FILE = os.path.join(CACHE_DIR, "complete_buoy_analysis.pkl")
TARGET_VALID_BUOYS = 120
LOCATION_CACHE_FILE = os.path.join(CACHE_DIR, "buoy_locations.json")

for d in [CACHE_DIR, NC_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Helper functions (defined first, before initialization)
def _get_location_from_dict(station_id, locations_dict):
    """Internal: Get location with case-insensitive lookup"""
    station_lower = station_id.lower()
    station_upper = station_id.upper()
    
    if station_lower in locations_dict:
        return locations_dict[station_lower]
    elif station_upper in locations_dict:
        return locations_dict[station_upper]
    elif station_id in locations_dict:
        return locations_dict[station_id]
    return None


# Initialize station locations from NDBC station table
def initialize_station_locations():
    """Load or fetch all NDBC station locations (only downloads if cache missing)"""
    # Check if cache exists - if so, just load it (no download)
    if os.path.exists(LOCATION_CACHE_FILE):
        locations = load_station_locations(LOCATION_CACHE_FILE)
        if locations:
            print(f"[CACHE] Loaded {len(locations)} station locations from cache")
            return locations
        else:
            print("[WARN] Cache file exists but is empty/corrupted, re-downloading...")
    
    # Only reaches here if cache doesn't exist or is corrupted
    print("[FETCH] Downloading station table from NDBC (first run only)...")
    locations = parse_station_table(output_json=LOCATION_CACHE_FILE)
    print(f"[CACHE] Saved {len(locations)} locations for future runs")
    return locations

# Load persistent location cache if it exists
def load_location_cache():
    """Load buoy locations from cached JSON file (legacy function for compatibility)"""
    return load_station_locations(LOCATION_CACHE_FILE) if os.path.exists(LOCATION_CACHE_FILE) else {}

def save_location_cache(locations):
    """Save buoy locations to JSON file for future runs"""
    try:
        import json
        with open(LOCATION_CACHE_FILE, 'w') as f:
            json.dump(locations, f, indent=2)
    except:
        pass

# Hardcoded locations (fallback for commonly used buoys)
BUOY_LOCATIONS = {
    '46001': (56.300, -148.020), '46002': (42.566, -130.487), '46003': (51.333, -155.978),
    '46005': (46.089, -131.018), '46006': (40.776, -137.475), '46011': (34.883, -120.862),
    '46012': (37.363, -122.881), '46013': (38.228, -123.307), '46014': (39.233, -123.968),
    '46022': (40.733, -124.525), '46025': (33.749, -119.053), '46026': (37.754, -122.839),
    '46027': (41.849, -124.382), '46028': (35.741, -121.884), '46029': (46.144, -124.509),
    '46030': (43.560, -124.530), '46035': (57.027, -177.738), '46036': (48.333, -133.867),
    '46041': (47.353, -124.731), '46042': (36.787, -122.398), '46047': (32.432, -119.533),
    '46050': (44.658, -124.530), '46051': (23.481, -162.206), '46053': (34.247, -119.849),
    '46054': (34.266, -120.478), '46059': (38.049, -129.969), '46060': (40.975, -127.006),
    '46061': (35.782, -121.905), '46062': (36.762, -122.031), '46063': (36.927, -122.030),
    '46069': (33.665, -120.212), '46073': (37.750, -122.670), '46078': (48.860, -125.774),
    '46080': (39.822, -121.898), '46082': (59.658, -143.399), '46084': (37.779, -122.465),
    '46085': (55.318, -134.671), '46086': (32.492, -118.031), '46087': (33.616, -118.683),
    '46088': (48.333, -123.167), '46089': (45.867, -125.768), '46090': (45.640, -124.970),
    '46091': (39.776, -123.714), '46092': (36.750, -122.028), '46093': (34.709, -120.867),
    '46094': (57.486, -153.859), '46097': (47.208, -124.731), '46098': (45.138, -124.712),
}

# Load all station locations at startup (merge with hardcoded fallbacks)
ALL_STATION_LOCATIONS = initialize_station_locations()
BUOY_LOCATIONS.update(ALL_STATION_LOCATIONS)  # Merge: cached locations override hardcoded ones

# Public API functions (use global BUOY_LOCATIONS)
def get_buoy_location(station_id):
    """Get buoy location from pre-loaded dictionary"""
    return _get_location_from_dict(station_id, BUOY_LOCATIONS)


def is_west_coast_buoy(station_id):
    """Check if buoy is on US West Coast (CA, OR, WA)"""
    location = get_buoy_location(station_id)
    if not location:
        return False
    
    lat, lon = location
    
    # US West Coast bounds:
    # Latitude: 30°N to 50°N (Southern CA to WA/Canada border)
    # Longitude: -130°W to -115°W (Pacific coast)
    return (30.0 <= lat <= 50.0) and (-130.0 <= lon <= -115.0)

# Count West Coast buoys
west_coast_count = sum(1 for sid in BUOY_LOCATIONS if is_west_coast_buoy(sid))
print(f"[OK] Total buoy locations available: {len(BUOY_LOCATIONS)}")
print(f"[OK] US West Coast buoys available: {west_coast_count}\n")

print("""
=======================================================================
         NDBC Buoy Comprehensive Analysis Pipeline
                                                                   
  Utilizing all buoys with 5+ years of ATMP data          
  Creating NetCDF files and Cartopy visualizations                 
=======================================================================
""")


def discover_all_buoys_robust(west_coast_only=True):
    print("\nP1: DISCOVERING AVAILABLE BUOYS")
    print("=" * 60)

    try:
        all_buoys = list_all_station_ids(BASE_URL)
        print("\n Discovered {} unique NDBC buoy stations".format(len(all_buoys)))
        
        if west_coast_only:
            print("\n Filtering for US West Coast buoys (30-50°N, 130-115°W)...")
            west_coast_buoys = [b for b in all_buoys if is_west_coast_buoy(b)]
            print(" Found {} West Coast buoys ({}% of total)".format(
                len(west_coast_buoys), 
                int(100 * len(west_coast_buoys) / len(all_buoys)) if all_buoys else 0
            ))
            return west_coast_buoys
        
        return all_buoys
    except Exception as e:
        print("\n error occurred in discovering buoys: {}".format(e))
        return []


def count_buoy_years(station_id, cached_urls=None):
    """available years for a buoy (uses cached URL list if provided)"""
    try:
        if cached_urls is None:
            urls = list_station_files(station_id)
        else:
            # Filter cached URLs for this station instead of re-scraping
            station_prefix = station_id.lower() + "h"
            urls = [u for u in cached_urls if station_prefix in u.lower()]
        
        year_pattern = re.compile(r'(\d{4})\.txt\.gz')
        years = set()
        
        for url in urls:
            match = year_pattern.search(url)
            if match:
                years.add(int(match.group(1)))
        
        return station_id, len(years)
    except:
        return station_id, 0


def filter_buoys_by_years(all_buoys, min_years=5, target_count=None, max_workers=10):
    """filter buoys to those with min_years or more of data"""
    print("\n P2: FILTERING BUOYS BY DATA AVAILABILITY")
    print("=" * 60)
    if target_count is None:
        print("Checking buoys for {}+ years of data (processing all matches)...\n".format(min_years))
    else:
        print("Checking buoys for {}+ years of data (target: {} buoys)...\n".format(min_years, target_count))
    print("Caching NDBC directory listing (downloading once)...")
    
    # Download the directory listing ONCE instead of for each buoy
    import requests
    from bs4 import BeautifulSoup
    try:
        r = requests.get(BASE_URL, timeout=120)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        cached_urls = []
        for a in soup.find_all("a", href=True):
            h = a["href"].lower()
            if h.endswith(".txt.gz"):
                from urllib.parse import urljoin
                cached_urls.append(urljoin(BASE_URL, h))
        print("[OK] Cached {} total NDBC files\n".format(len(cached_urls)))
    except Exception as e:
        print("[WARN] Failed to cache directory: {}\n".format(e))
        cached_urls = None
    
    valid_buoys = {}
    processed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(count_buoy_years, buoy, cached_urls): buoy for buoy in all_buoys}
        
        for future in as_completed(futures):
            station_id, num_years = future.result()
            processed += 1
            
            if num_years >= min_years:
                valid_buoys[station_id] = num_years
                print("[{:4d}/{}] ACCEPTED: {} ({} years of data)".format(processed, len(all_buoys), station_id, num_years))
            else:
                print("[{:4d}/{}] SKIPPED:  {} (only {} years - need {}+)".format(processed, len(all_buoys), station_id, num_years, min_years))
            
            if target_count is not None and len(valid_buoys) >= target_count:
                print("\n[OK] Found {} buoys with {}+ years of data. Stopping search.".format(target_count, min_years))
                break
    
    print()  # newline after progress bar
    
    print("\nFound {} buoys with {}+ years of data".format(len(valid_buoys), min_years))
    
    if valid_buoys:
        years_list = sorted(valid_buoys.values(), reverse=True)
        print("\nYear distribution:")
        print("  Max: {} years".format(years_list[0]))
        print("  Min: {} years".format(years_list[-1]))
        print("  Mean: {:.1f} years".format(np.mean(years_list)))
    
    return valid_buoys


def select_first_buoys(valid_buoys, first_n=10):
    if not valid_buoys:
        return {}
    first_items = list(valid_buoys.items())[:first_n]
    return dict(first_items)


def process_single_buoy_to_netcdf(station_id, window_size=100):
    """Process a single buoy to NetCDF"""
    try:
        print("\n Processing {}...".format(station_id), end=" ")
        sys.stdout.flush()
        
        # Check location FIRST before processing data
        location = get_buoy_location(station_id)
        if not location:
            print("FAIL: Location not found for '{}' (ID format: lowercase='{}', check cache)".format(
                station_id, station_id.lower()))
            return None
        
        df_filled, completeness = load_station(station_id, workers=4, cache_dir=CACHE_DIR)
        
        if df_filled is None or df_filled.empty:
            print("FAIL: No data loaded")
            return None
        
        year_start = int(df_filled['year'].min())
        year_end = int(df_filled['year'].max())
        num_years = year_end - year_start + 1
        
        nc_path = write_warm_season_netcdf(
            df_filled, station_id,
            out_dir=NC_DIR,
            window_size=window_size,
            target_days=None  # Keep all warm season days (no truncation)
        )
        
        stats = read_netcdf_statistics(nc_path, station_id)
        mean_variance = stats['mean_variance']
        mean_skewness = stats['mean_skewness']
        
        result = {
            'station_id': station_id,
            'latitude': location[0],
            'longitude': location[1],
            'num_years': num_years,
            'year_start': year_start,
            'year_end': year_end,
            'mean_variance': mean_variance,
            'mean_skewness': mean_skewness,
            'netcdf_path': nc_path,
        }
        
        print("NetCDF written and statistics extracted from file")
        return result
    
    except Exception as e:
        print("[FAIL] Error: {}".format(str(e)[:50]))
        return None


def process_all_buoys(valid_buoys, window_size=100):
    """Process all valid buoys"""
    print("\n P3: PROCESSING BUOYS TO NETCDF")
    print("=" * 60)
    
    results = []
    buoy_list = sorted(valid_buoys.keys())
    
    print("Processing {} buoys (sequential):\n".format(len(buoy_list)))
    
    for i, station_id in enumerate(buoy_list, 1):
        print("[{}/{}]".format(i, len(buoy_list)), end=" ")
        result = process_single_buoy_to_netcdf(station_id, window_size=window_size)
        
        if result:
            results.append(result)
        
        if i % 5 == 0:
            time.sleep(1)
    
    print("\n[OK] Successfully processed {}/{} buoys".format(len(results), len(buoy_list)))
    return results


def save_results(results):
    """Save processing results"""
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)
    print("[SAVE] Results saved to {}".format(RESULTS_FILE))


def plot_cartopy_maps(results, window_size=100):
    """Create Cartopy maps"""
    print("\n P4: CREATING CARTOPY WORLD MAPS")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    df = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df) == 0:
        print("No data to plot")
        return None
    
    print("Plotting {} stations...".format(len(df)))
    
    fig = plt.figure(figsize=(22, 10))
    projection = ccrs.PlateCarree()
    
    ax1 = fig.add_subplot(1, 2, 1, projection=projection)
    
    ax1.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
    ax1.add_feature(cfeature.OCEAN, facecolor='#b3d9ff')
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
    ax1.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, linestyle='--')
    
    lon_range = df['longitude'].max() - df['longitude'].min()
    lat_range = df['latitude'].max() - df['latitude'].min()
    
    if lon_range < 100 and lat_range < 50:
        extent = [df['longitude'].min() - 5, df['longitude'].max() + 5,
                  df['latitude'].min() - 5, df['latitude'].max() + 5]
        ax1.set_extent(extent, crs=projection)
    else:
        ax1.set_global()
    
    variance = df['mean_variance'].values
    norm_var = Normalize(vmin=variance.min(), vmax=variance.max())
    
    scatter1 = ax1.scatter(
        df['longitude'], df['latitude'],
        c=variance,
        cmap='YlOrRd',
        s=150,
        alpha=0.85,
        edgecolors='black',
        linewidth=1,
        norm=norm_var,
        transform=projection,
        zorder=5
    )
    
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar1.set_label('Mean Warm Season Variance (C^2)', fontsize=12, weight='bold')
    
    ax1.set_title('Warm Season Temperature Variance', fontsize=14, weight='bold', pad=15)
    
    ax2 = fig.add_subplot(1, 2, 2, projection=projection)
    
    ax2.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
    ax2.add_feature(cfeature.OCEAN, facecolor='#b3d9ff')
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
    ax2.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, linestyle='--')
    
    if lon_range < 100 and lat_range < 50:
        ax2.set_extent(extent, crs=projection)
    else:
        ax2.set_global()
    
    skewness = df['mean_skewness'].values
    abs_max = max(abs(skewness.min()), abs(skewness.max()))
    norm_skew = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    
    scatter2 = ax2.scatter(
        df['longitude'], df['latitude'],
        c=skewness,
        cmap='coolwarm',
        s=150,
        alpha=0.85,
        edgecolors='black',
        linewidth=1,
        norm=norm_skew,
        transform=projection,
        zorder=5
    )
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.8)
    cbar2.set_label('Mean Warm Season Skewness', fontsize=12, weight='bold')
    
    ax2.set_title('Warm Season Temperature Skewness', fontsize=14, weight='bold', pad=15)
    
    n_buoys = len(df)
    year_range = "{}-{}".format(int(df['year_start'].min()), int(df['year_end'].max()))
    fig.suptitle(
        'NDBC Buoy Warm Season Temperature Statistics\n'
        '{} Stations | Years: {} | ±{} day window (centered on peak temp)'.format(n_buoys, year_range, window_size),
        fontsize=16, weight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(FIGURES_DIR, 'West_Coast_cartopy_variance_skewness_maps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("[SAVE] Saved: {}".format(output_path))
    
    plt.close(fig)
    
    return df


def print_summary_statistics(df):
    """Print summary statistics"""
    print("\nP5: SUMMARY STATISTICS")
    print("=" * 60)
    
    print("\nTotal stations plotted: {}".format(len(df)))
    print("Year range: {} - {}".format(int(df['year_start'].min()), int(df['year_end'].max())))
    print("Mean coverage: {:.1f} years per station".format(df['num_years'].mean()))
    
    print("\n--- VARIANCE ---")
    print("Range: {:.4f} - {:.4f} C^2".format(df['mean_variance'].min(), df['mean_variance'].max()))
    print("Mean: {:.4f} C^2".format(df['mean_variance'].mean()))
    print("Std Dev: {:.4f} C^2".format(df['mean_variance'].std()))
    
    print("\n--- SKEWNESS ---")
    print("Range: {:.4f} - {:.4f}".format(df['mean_skewness'].min(), df['mean_skewness'].max()))
    print("Mean: {:.4f}".format(df['mean_skewness'].mean()))
    print("Std Dev: {:.4f}".format(df['mean_skewness'].std()))
    
    print("\n--- TOP 10 HIGHEST VARIANCE ---")
    top_var = df.nlargest(10, 'mean_variance')[['station_id', 'num_years', 'mean_variance', 'mean_skewness']]
    print(top_var.to_string(index=False))
    
    print("\n--- TOP 10 MOST POSITIVE SKEWNESS ---")
    top_skew = df.nlargest(10, 'mean_skewness')[['station_id', 'num_years', 'mean_variance', 'mean_skewness']]
    print(top_skew.to_string(index=False))
    
    print("\n--- TOP 10 MOST NEGATIVE SKEWNESS ---")
    neg_skew = df.nsmallest(10, 'mean_skewness')[['station_id', 'num_years', 'mean_variance', 'mean_skewness']]
    print(neg_skew.to_string(index=False))


def main():
    """Execute complete pipeline"""
    
    start_time = datetime.now()
    
    try:
        # Set to True to filter for US West Coast only
        WEST_COAST_ONLY = True
        
        all_buoys = discover_all_buoys_robust(west_coast_only=WEST_COAST_ONLY)
        if not all_buoys:
            print("[ERROR] Failed to discover buoys. Exiting.")
            return
        
        valid_buoys = filter_buoys_by_years(all_buoys, min_years=5, target_count=TARGET_VALID_BUOYS, max_workers=15)
        if not valid_buoys:
            print("[ERROR] No buoys with sufficient data. Exiting.")
            return

        results = process_all_buoys(valid_buoys, window_size=100)
        if not results:
            print("[ERROR] No successful buoy processing. Exiting.")
            return

        save_results(results)

        df = plot_cartopy_maps(results, window_size=100)

        print_summary_statistics(df)

        elapsed = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("[OK] PIPELINE COMPLETE!")
        print("=" * 60)
        print("\nProcessed: {} buoys".format(len(results)))
        print("Output: {}/  (NetCDF files)".format(NC_DIR))
        print("Output: {}/  (Cartopy maps)".format(FIGURES_DIR))
        print("Elapsed time: {}".format(elapsed))
        
    except KeyboardInterrupt:
        print("\n\n[WARN] Pipeline interrupted by user")
    except Exception as e:
        print("\n[ERROR] Unexpected error: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
