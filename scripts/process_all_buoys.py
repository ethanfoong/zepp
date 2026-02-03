"""
Comprehensive NDBC Buoy Analysis
=================================
Scrapes all NDBC buoys with 10+ years of data, computes warm-season statistics
(skewness and variance), and visualizes them on world maps using Cartopy.

Follows the methodology from nbdc statistical analyses.ipynb
"""

import os
import sys
import re
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import skew
import pickle
from tqdm import tqdm

# Add parent directory to path to import helper modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_buoy_helpers import list_station_files, load_station
from stat_buoy_helpers import get_warm_season_data, compute_warm_season_anomalies

BASE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
CACHE_DIR = "cache"
RESULTS_DIR = "nc"
METADATA_CACHE = os.path.join(CACHE_DIR, "buoy_metadata.pkl")

# Known buoy locations (can be expanded with NDBC station metadata)
BUOY_LOCATIONS = {
    '46001': (56.300, -148.020),
    '46002': (42.566, -130.487),
    '46003': (51.333, -155.978),
    '46005': (46.089, -131.018),
    '46006': (40.776, -137.475),
    '46011': (34.883, -120.862),
    '46012': (37.363, -122.881),
    '46013': (38.228, -123.307),
    '46014': (39.233, -123.968),
    '46022': (40.733, -124.525),
    '46025': (33.749, -119.053),
    '46026': (37.754, -122.839),
    '46027': (41.849, -124.382),
    '46028': (35.741, -121.884),
    '46029': (46.144, -124.509),
    '46030': (43.560, -124.530),
    '46035': (57.027, -177.738),
    '46036': (48.333, -133.867),
    '46041': (47.353, -124.731),
    '46042': (36.787, -122.398),
    '46047': (32.432, -119.533),
    '46050': (44.658, -124.530),
    '46051': (23.481, -162.206),
    '46053': (34.247, -119.849),
    '46054': (34.266, -120.478),
    '46059': (38.049, -129.969),
    '46060': (40.975, -127.006),
    '46061': (35.782, -121.905),
    '46062': (36.762, -122.031),
    '46063': (36.927, -122.030),
    '46069': (33.665, -120.212),
    '46073': (37.750, -122.670),
    '46078': (48.860, -125.774),
    '46080': (39.822, -121.898),
    '46082': (59.658, -143.399),
    '46084': (37.779, -122.465),
    '46085': (55.318, -134.671),
    '46086': (32.492, -118.031),
    '46087': (33.616, -118.683),
    '46088': (48.333, -123.167),
    '46089': (45.867, -125.768),
    '46090': (45.640, -124.970),
    '46091': (39.776, -123.714),
    '46092': (36.750, -122.028),
    '46093': (34.709, -120.867),
    '46094': (57.486, -153.859),
    '46097': (47.208, -124.731),
    '46098': (45.138, -124.712),
    '46106': (32.465, -117.168),
    '46107': (33.221, -117.476),
    '46108': (34.165, -120.005),
    '46109': (40.304, -124.349),
    '46114': (43.756, -124.554),
    '46116': (40.229, -124.773),
    '46120': (34.208, -119.877),
    '46121': (33.919, -120.216),
    '46131': (43.571, -124.555),
    '46132': (40.294, -124.363),
    '46212': (37.168, -122.457),
    '46213': (37.760, -122.638),
    '46214': (37.777, -122.465),
    '46215': (37.771, -122.626),
    '46216': (37.782, -122.466),
    '46218': (37.788, -122.465),
    '46219': (37.929, -122.599),
    '46221': (37.856, -122.465),
    '46222': (37.947, -122.511),
    '46224': (37.507, -122.854),
    '46225': (36.836, -121.887),
    '46226': (36.771, -121.903),
    '46229': (34.893, -120.802),
    '46232': (37.494, -122.495),
    '46235': (37.802, -122.465),
    '46236': (37.586, -122.530),
    '46237': (37.367, -122.974),
    '46239': (37.577, -122.977),
    '46240': (37.287, -122.303),
    '46242': (36.914, -121.918),
    '46243': (36.932, -122.014),
    '46244': (37.014, -122.107),
    '46245': (37.034, -122.302),
    '46246': (36.217, -121.847),
    '46250': (34.762, -121.468),
    '46251': (33.856, -118.634),
    '46253': (32.931, -117.391),
    '46254': (32.579, -117.169),
    '46255': (32.931, -117.391),
    '46256': (33.021, -117.466),
    '46258': (33.583, -117.867),
    '46259': (33.676, -118.012),
    '46266': (33.212, -119.881),
    '46268': (33.139, -117.494),
    '46269': (33.468, -117.785),
    '46270': (34.453, -120.007),
    '46271': (34.151, -119.878),
    '46272': (33.186, -117.479),
    '46273': (33.557, -118.185),
    '46274': (33.826, -118.317),
    '46275': (33.646, -118.014),
    '46276': (33.467, -117.785),
    '46277': (33.548, -118.184),
    '46278': (37.403, -122.537),
}


def discover_all_buoys():
    """
    Scrape NDBC historical stdmet directory to find all available buoy stations.
    Returns a list of unique station IDs.
    """
    print("🔍 Discovering all available NDBC buoys...")
    try:
        r = requests.get(BASE_URL, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        
        buoy_pattern = re.compile(r'^(\d{5}|[a-z0-9]{5})h\d{4}\.txt\.gz$', re.IGNORECASE)
        buoys = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            match = buoy_pattern.match(href)
            if match:
                buoys.add(match.group(1).lower())
        
        print(f"✅ Found {len(buoys)} unique buoy stations")
        return sorted(buoys)
    
    except Exception as e:
        print(f"❌ Error discovering buoys: {e}")
        return []


def get_buoy_years(station_id):
    """
    Count the number of years of data available for a given buoy.
    Returns (station_id, num_years)
    """
    try:
        urls = list_station_files(station_id)
        year_pattern = re.compile(r'(\d{4})\.txt\.gz')
        years = set()
        
        for url in urls:
            match = year_pattern.search(url)
            if match:
                years.add(int(match.group(1)))
        
        return station_id, len(years)
    
    except Exception as e:
        return station_id, 0


def filter_buoys_by_years(buoys, min_years=10, max_workers=10):
    """
    Filter buoys to only those with at least min_years of data.
    Returns dict {station_id: num_years}
    """
    print(f"\n📊 Checking data availability for {len(buoys)} buoys (min {min_years} years)...")
    
    valid_buoys = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_buoy_years, buoy): buoy for buoy in buoys}
        
        with tqdm(total=len(buoys), desc="Checking buoys") as pbar:
            for future in as_completed(futures):
                station_id, num_years = future.result()
                if num_years >= min_years:
                    valid_buoys[station_id] = num_years
                pbar.update(1)
    
    print(f"✅ Found {len(valid_buoys)} buoys with {min_years}+ years of data")
    return valid_buoys


def fetch_buoy_location(station_id):
    """
    Attempt to fetch buoy location from NDBC station page.
    Returns (lat, lon) or None if not found.
    """
    # First check our hardcoded locations
    if station_id in BUOY_LOCATIONS:
        return BUOY_LOCATIONS[station_id]
    
    # Try to scrape from NDBC station page
    try:
        url = f"https://www.ndbc.noaa.gov/station_page.php?station={station_id}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        # Look for lat/lon in the page
        lat_match = re.search(r'(\d+\.\d+)\s*°?\s*[NS]', r.text)
        lon_match = re.search(r'(\d+\.\d+)\s*°?\s*[EW]', r.text)
        
        if lat_match and lon_match:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))
            
            # Check hemisphere
            if 'S' in lat_match.group(0):
                lat = -lat
            if 'W' in lon_match.group(0):
                lon = -lon
            
            return (lat, lon)
    
    except Exception as e:
        pass
    
    return None


def process_single_buoy(station_id, window_size=100):
    """
    Process a single buoy: load data, compute warm season statistics.
    Returns dict with station metadata and statistics, or None if processing fails.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing Station: {station_id.upper()}")
        print(f"{'='*60}")
        
        # Load station data
        df_filled, completeness = load_station(station_id, workers=6, cache_dir=CACHE_DIR)
        
        if df_filled is None or df_filled.empty:
            print(f"❌ No data loaded for {station_id}")
            return None
        
        # Get warm season data and compute anomalies
        warm_df, (warm_start, warm_center, warm_end) = get_warm_season_data(df_filled, window_size=window_size)
        anomalies, mean_cycle, _ = compute_warm_season_anomalies(df_filled, window_size=window_size)
        
        # Compute statistics
        # Variance: across all days in warm season, per year
        yearly_var = anomalies.var(axis=0)
        
        # Skewness: across all days in warm season, per year
        yearly_skew = anomalies.apply(lambda col: skew(col.dropna()) if col.notna().sum() > 3 else np.nan, axis=0)
        
        # Get overall statistics (mean across all years)
        mean_variance = yearly_var.mean()
        mean_skewness = yearly_skew.mean()
        
        # Get location
        location = fetch_buoy_location(station_id)
        
        # Year range
        years = anomalies.columns.tolist()
        year_start = int(min(years))
        year_end = int(max(years))
        num_years = len(years)
        
        result = {
            'station_id': station_id,
            'latitude': location[0] if location else np.nan,
            'longitude': location[1] if location else np.nan,
            'num_years': num_years,
            'year_start': year_start,
            'year_end': year_end,
            'warm_start': warm_start,
            'warm_center': warm_center,
            'warm_end': warm_end,
            'mean_variance': mean_variance,
            'mean_skewness': mean_skewness,
            'yearly_variance': yearly_var.to_dict(),
            'yearly_skewness': yearly_skew.to_dict(),
        }
        
        print(f"✅ {station_id}: {num_years} years ({year_start}-{year_end})")
        print(f"   Warm season: days {warm_start}-{warm_end} (center: {warm_center})")
        print(f"   Mean variance: {mean_variance:.4f}")
        print(f"   Mean skewness: {mean_skewness:.4f}")
        
        return result
    
    except Exception as e:
        print(f"❌ Error processing {station_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_all_buoys(buoy_dict, window_size=100, max_workers=3):
    """
    Process all buoys in parallel.
    
    Parameters:
        buoy_dict: {station_id: num_years}
        window_size: days around peak temperature for warm season
        max_workers: number of concurrent processing threads
    
    Returns:
        List of result dictionaries
    """
    results = []
    buoy_list = list(buoy_dict.keys())
    
    print(f"\n🚀 Processing {len(buoy_list)} buoys with {max_workers} workers...")
    
    # Process sequentially for better error tracking and progress monitoring
    with tqdm(total=len(buoy_list), desc="Processing buoys") as pbar:
        for station_id in buoy_list:
            result = process_single_buoy(station_id, window_size=window_size)
            if result is not None:
                results.append(result)
            pbar.update(1)
    
    print(f"\n✅ Successfully processed {len(results)}/{len(buoy_list)} buoys")
    return results


def save_results(results, output_file="buoy_statistics.pkl"):
    """Save processing results to pickle file."""
    output_path = os.path.join(CACHE_DIR, output_file)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"💾 Results saved to {output_path}")


def load_results(input_file="buoy_statistics.pkl"):
    """Load previously saved results."""
    input_path = os.path.join(CACHE_DIR, input_file)
    
    if not os.path.exists(input_path):
        print(f"❌ No saved results found at {input_path}")
        return None
    
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    
    print(f"📂 Loaded {len(results)} buoy results from {input_path}")
    return results


def main():
    """Main processing pipeline."""
    
    # Step 1: Discover all buoys
    all_buoys = discover_all_buoys()
    
    if not all_buoys:
        print("❌ No buoys discovered. Exiting.")
        return
    
    # Step 2: Filter buoys with 10+ years of data
    valid_buoys = filter_buoys_by_years(all_buoys, min_years=10, max_workers=20)
    
    if not valid_buoys:
        print("❌ No buoys with sufficient data found. Exiting.")
        return
    
    # Step 3: Process all valid buoys
    results = process_all_buoys(valid_buoys, window_size=100, max_workers=1)
    
    # Step 4: Save results
    save_results(results)
    
    # Step 5: Create summary DataFrame
    df_summary = pd.DataFrame(results)
    df_summary = df_summary.dropna(subset=['latitude', 'longitude'])
    
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    print(f"Total buoys processed: {len(results)}")
    print(f"Buoys with location data: {len(df_summary)}")
    print(f"\nVariance range: {df_summary['mean_variance'].min():.4f} - {df_summary['mean_variance'].max():.4f}")
    print(f"Skewness range: {df_summary['mean_skewness'].min():.4f} - {df_summary['mean_skewness'].max():.4f}")
    print("\nTop 5 by variance:")
    print(df_summary.nlargest(5, 'mean_variance')[['station_id', 'num_years', 'mean_variance', 'mean_skewness']])
    print("\nTop 5 by skewness:")
    print(df_summary.nlargest(5, 'mean_skewness')[['station_id', 'num_years', 'mean_variance', 'mean_skewness']])
    
    return results, df_summary


if __name__ == "__main__":
    results, df_summary = main()
