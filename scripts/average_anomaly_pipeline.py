import os
import sys
import re
import time
import glob
import pandas as pd
import numpy as np
import pickle
import cartopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import skew, gaussian_kde
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_buoy_helpers import load_station_locations

"""
Average Anomaly Pipeline
========================
Extends complete_buoy_pipeline to compute highest/lowest temperature anomalies
for each buoy across its daily time series.

For each buoy:
- Extract daily max anomalies during warm season
- Find the highest and lowest anomaly per year
- Average these yearly max/min anomalies across entire time series
- Segment buoys by coast (west/east)
- Create NETCDF files and visualizations for high/low anomaly metrics
- Parallelize high/low processing simultaneously
"""

BASE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
CACHE_DIR = "cache"
NC_DIR = "nc"
FIGURES_DIR = "figures"
RESULTS_FILE = os.path.join(CACHE_DIR, "average_anomaly_analysis.pkl")
TX_LAND_STATION_FILE = os.path.join(NC_DIR, "TX_Land_Station.nc")
TMIN_LAND_STATION_FILE = os.path.join(NC_DIR, "TMIN_Land_Station.nc")
TVAR_LAND_STATION_FILE = os.path.join(NC_DIR, "TVAR_Land_Station.nc")
TSKEW_LAND_STATION_FILE = os.path.join(NC_DIR, "TSKEW_Land_Station.nc")
TARGET_VALID_BUOYS = 120
LOCATION_CACHE_FILE = os.path.join(CACHE_DIR, "buoy_locations.json")

for d in [CACHE_DIR, NC_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Helper functions
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
    """Load station locations from local cache only (offline mode)."""
    if os.path.exists(LOCATION_CACHE_FILE):
        locations = load_station_locations(LOCATION_CACHE_FILE)
        if locations:
            print(f"[CACHE] Loaded {len(locations)} station locations from cache")
            return locations

    print("[WARN] Location cache missing/corrupted; using hardcoded fallback locations only.")
    return {}

# Load persistent location cache if it exists
def load_location_cache():
    """Load buoy locations from cached JSON file"""
    return load_station_locations(LOCATION_CACHE_FILE) if os.path.exists(LOCATION_CACHE_FILE) else {}

def save_location_cache(locations):
    """Save buoy locations to JSON file for future runs"""
    try:
        import json
        with open(LOCATION_CACHE_FILE, 'w') as f:
            json.dump(locations, f, indent=2)
    except:
        pass

# Hardcoded locations (fallback)
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

# Load all station locations at startup
ALL_STATION_LOCATIONS = initialize_station_locations()
BUOY_LOCATIONS.update(ALL_STATION_LOCATIONS)

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
    # Latitude: 30N to 50N (Southern CA to WA/Canada border)
    # Longitude: -130W to -115W (Pacific coast)
    return (30.0 <= lat <= 50.0) and (-130.0 <= lon <= -115.0)


def is_east_coast_buoy(station_id):
    """Check if buoy is on US East Coast (Atlantic)"""
    location = get_buoy_location(station_id)
    if not location:
        return False
    
    lat, lon = location
    
    # US East Coast bounds:
    # Latitude: 25N to 45N (Florida to Maine)
    # Longitude: -85W to -65W (Atlantic coast)
    return (25.0 <= lat <= 45.0) and (-85.0 <= lon <= -65.0)


def get_coast_region(station_id):
    """Return coast region: 'west', 'east', or None"""
    if is_west_coast_buoy(station_id):
        return 'west'
    elif is_east_coast_buoy(station_id):
        return 'east'
    return None


# Count coast buoys
west_coast_count = sum(1 for sid in BUOY_LOCATIONS if is_west_coast_buoy(sid))
east_coast_count = sum(1 for sid in BUOY_LOCATIONS if is_east_coast_buoy(sid))
print(f"[OK] Total buoy locations available: {len(BUOY_LOCATIONS)}")
print(f"[OK] US West Coast buoys available: {west_coast_count}")
print(f"[OK] US East Coast buoys available: {east_coast_count}\n")

print("""
=======================================================================
      NDBC Buoy Average Anomaly Analysis Pipeline
                                                                   
  Compute max/min warm-season anomalies per buoy
  Segment by coast and parallelize high/low processing
  Creating NetCDF files and Cartopy visualizations
=======================================================================
""")


def _parse_warm_nc_name(path):
    """Extract station id and year count from a warm-anomaly NetCDF filename."""
    name = os.path.basename(path)
    m = re.match(r'^([a-z0-9]+)_warm_anomalies_(\d+)y_\d+d\.nc$', name, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def discover_cached_buoys_from_warm_netcdf(west_coast_only=False, min_years=5, target_count=None):
    """Discover buoys from existing local warm-anomaly NetCDF files only."""
    print("\nP1: DISCOVERING CACHED BUOYS (OFFLINE)")
    print("=" * 60)

    warm_files = glob.glob(os.path.join(NC_DIR, "*_warm_anomalies_*y_*d.nc"))
    if not warm_files:
        print("[ERROR] No local warm-anomaly NetCDF files found in nc/.")
        return {}

    best_by_station = {}
    for path in warm_files:
        station_id, year_count = _parse_warm_nc_name(path)
        if not station_id:
            continue
        prev = best_by_station.get(station_id)
        if prev is None or year_count > prev['years']:
            best_by_station[station_id] = {'years': year_count, 'warm_netcdf_path': path}

    valid = {
        sid: info for sid, info in best_by_station.items()
        if info['years'] >= min_years
    }

    if west_coast_only:
        valid = {sid: info for sid, info in valid.items() if is_west_coast_buoy(sid)}

    if target_count is not None and len(valid) > target_count:
        valid_items = sorted(valid.items(), key=lambda kv: kv[1]['years'], reverse=True)[:target_count]
        valid = dict(valid_items)

    print(f"Found {len(warm_files)} warm NetCDF files total")
    print(f"Found {len(best_by_station)} unique stations with warm NetCDF")
    print(f"Selected {len(valid)} stations with {min_years}+ years from local cache")
    return valid


def _find_existing_extreme_netcdf(station_id, anomaly_type):
    """Return a previously generated extreme-anomaly NetCDF path if available."""
    pattern = os.path.join(NC_DIR, f"{station_id}_{anomaly_type}_anomaly_*y.nc")
    matches = glob.glob(pattern)
    if not matches:
        return None

    def year_key(path):
        m = re.search(r'_(\d+)y\.nc$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    return sorted(matches, key=year_key, reverse=True)[0]


def _read_extreme_anomaly_netcdf(path):
    """Read an extreme-anomaly NetCDF and return years plus yearly anomaly values."""
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        years = ds['year'].values.astype(int)
        yearly = ds['yearly_anomalies'].values
        attrs = dict(ds.attrs)
        ds.close()
    except Exception:
        from netCDF4 import Dataset
        nc = Dataset(path, 'r')
        years = nc['year'][:].astype(int)
        yearly = nc['yearly_anomalies'][:]
        attrs = {
            'station_id': getattr(nc, 'station_id', ''),
            'anomaly_type': getattr(nc, 'anomaly_type', ''),
            'average_anomaly': getattr(nc, 'average_anomaly', np.nan),
        }
        nc.close()

    yearly = np.asarray(yearly)
    if yearly.ndim == 2:
        yearly = yearly[0, :]
    return years, yearly.astype(float), attrs


def _read_warm_anomaly_netcdf(path):
    """Read warm-season anomaly NetCDF and return per-year anomaly matrix."""
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        years = ds['year'].values.astype(int)
        anomalies = ds['anomalies'].values
        ds.close()
    except Exception:
        from netCDF4 import Dataset
        nc = Dataset(path, 'r')
        years = nc['year'][:].astype(int)
        anomalies = nc['anomalies'][:]
        nc.close()

    anomalies = np.asarray(anomalies)
    if anomalies.ndim != 2:
        raise ValueError(f"Expected 2D anomalies in {path}, got shape {anomalies.shape}")
    return years, anomalies.astype(float)


def compute_yearly_max_min_from_warm_netcdf(warm_netcdf_path):
    """Compute yearly max/min anomalies directly from existing warm-anomaly NetCDF."""
    years, anomalies = _read_warm_anomaly_netcdf(warm_netcdf_path)
    yearly_max_anomalies = np.nanmax(anomalies, axis=1)
    yearly_min_anomalies = np.nanmin(anomalies, axis=1)
    return years, yearly_max_anomalies, yearly_min_anomalies


def write_extreme_anomaly_netcdf(years, yearly_anomalies, station_id, anomaly_type='max', out_dir="nc"):
    """
    Create a NetCDF file of yearly extreme anomalies (max or min).
    
    Parameters:
    -----------
    years : 1D array of years
    yearly_anomalies : 1D array with one value per year
    station_id : str
    anomaly_type : 'max' or 'min'
    out_dir : output directory
    out_dir : output directory
    
    Returns:
    --------
    Tuple: (filepath, avg_anomaly, yearly_anomalies, years)
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    # Average anomaly across entire time series
    avg_anomaly = np.nanmean(yearly_anomalies)
    
    # Build NetCDF file
    years_coord = years.astype(int)
    anom_data = yearly_anomalies.reshape(1, -1)  # Shape (1, n_years)
    
    fname = os.path.join(out_dir, 
                         f"{station_id}_{anomaly_type}_anomaly_{len(years)}y.nc")
    
    try:
        import xarray as xr
        ds = xr.Dataset(
            {
                'yearly_anomalies': (('instance', 'year'), anom_data),
            },
            coords={
                'year': years_coord,
                'instance': [0],
            },
            attrs={
                'station_id': station_id,
                'anomaly_type': anomaly_type,
                'average_anomaly': float(avg_anomaly),
                'n_years': len(years),
            }
        )
        ds.to_netcdf(fname)
    except Exception:
        # fallback using netCDF4
        try:
            from netCDF4 import Dataset
            nc = Dataset(fname, 'w')
            nc.createDimension('instance', 1)
            nc.createDimension('year', len(years))
            year_var = nc.createVariable('year', 'i4', ('year',))
            anom_var = nc.createVariable('yearly_anomalies', 'f4', ('instance', 'year'), fill_value=np.nan)
            year_var[:] = years_coord
            anom_var[0, :] = yearly_anomalies
            nc.station_id = station_id
            nc.anomaly_type = anomaly_type
            nc.average_anomaly = float(avg_anomaly)
            nc.n_years = len(years)
            nc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to write NetCDF file: {e}")
    
    return fname, avg_anomaly, yearly_anomalies, years


def process_single_buoy_anomalies(station_id, warm_netcdf_path, anomaly_type='max'):
    """
    Process a single buoy to extract extreme anomalies.
    
    Parameters:
    -----------
    station_id : str
    anomaly_type : 'max' or 'min'
    warm_netcdf_path : path to source warm-anomaly NetCDF
    
    Returns:
    --------
    dict with results, or None if failed
    """
    try:
        print(f"\n [Processing {station_id} ({anomaly_type})...]", end=" ", flush=True)
        
        # Check location FIRST before processing data
        location = get_buoy_location(station_id)
        if not location:
            print(f"FAIL: Location not found")
            return None
        
        coast = get_coast_region(station_id)

        existing_path = _find_existing_extreme_netcdf(station_id, anomaly_type)
        if existing_path:
            years, yearly_anomalies, attrs = _read_extreme_anomaly_netcdf(existing_path)
            avg_anomaly = float(np.nanmean(yearly_anomalies))
            nc_path = existing_path
            print(f"REUSED {os.path.basename(existing_path)}")
        else:
            years, yearly_max, yearly_min = compute_yearly_max_min_from_warm_netcdf(warm_netcdf_path)
            yearly_anomalies = yearly_max if anomaly_type == 'max' else yearly_min
            nc_path, avg_anomaly, _, _ = write_extreme_anomaly_netcdf(
                years,
                yearly_anomalies,
                station_id,
                anomaly_type=anomaly_type,
                out_dir=NC_DIR,
            )
            print(f"CREATED {os.path.basename(nc_path)}")

        year_start = int(np.nanmin(years))
        year_end = int(np.nanmax(years))
        num_years = len(years)
        
        result = {
            'station_id': station_id,
            'anomaly_type': anomaly_type,
            'latitude': location[0],
            'longitude': location[1],
            'coast': coast,
            'num_years': num_years,
            'year_start': year_start,
            'year_end': year_end,
            'average_anomaly': avg_anomaly,
            'max_yearly_anomaly': np.nanmax(yearly_anomalies),
            'min_yearly_anomaly': np.nanmin(yearly_anomalies),
            'std_yearly_anomaly': np.nanstd(yearly_anomalies),
            'netcdf_path': nc_path,
        }
        
        print(f"OK (avg={avg_anomaly:.3f} degC)")
        return result
    
    except Exception as e:
        print(f"[FAIL] {str(e)[:50]}")
        return None


def process_all_buoys_parallel(valid_buoys, max_workers=4):
    """
    Process all valid buoys for both max and min anomalies in parallel.
    Parallelizes both high/low computation AND buoy processing.
    """
    print("\n P3: PROCESSING BUOYS FOR EXTREME ANOMALIES")
    print("=" * 60)
    
    buoy_list = sorted(valid_buoys.keys())
    total_tasks = len(buoy_list) * 2  # Each buoy gets max AND min processing
    
    print(f"Processing {len(buoy_list)} buoys x 2 anomaly types = {total_tasks} tasks\n")
    
    results = []
    completed = 0
    
    # Submit all tasks to executor at once
    futures_to_info = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all max and min tasks
        for station_id in buoy_list:
            warm_netcdf_path = valid_buoys[station_id]['warm_netcdf_path']
            fut_max = executor.submit(process_single_buoy_anomalies, station_id, warm_netcdf_path, 'max')
            fut_min = executor.submit(process_single_buoy_anomalies, station_id, warm_netcdf_path, 'min')
            futures_to_info[fut_max] = (station_id, 'max')
            futures_to_info[fut_min] = (station_id, 'min')
        
        # Collect results as they complete
        for future in as_completed(futures_to_info):
            completed += 1
            station_id, anom_type = futures_to_info[future]
            
            result = future.result()
            if result:
                results.append(result)
                print(f"[{completed}/{total_tasks}] [OK] {station_id} ({anom_type})")
            else:
                print(f"[{completed}/{total_tasks}] [FAIL] {station_id} ({anom_type})")
    
    print(f"\n[OK] Successfully processed {len(results)}/{total_tasks} tasks")
    return results


def save_results(results):
    """Save processing results"""
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)
    print(f"[SAVE] Results saved to {RESULTS_FILE}")


def load_land_station_points(nc_path, preferred_var='D'):
    """Load land-station coordinates and anomaly values from NetCDF."""
    if not os.path.exists(nc_path):
        print(f"[WARN] Land-station file not found: {nc_path}")
        return np.array([]), np.array([]), np.array([])

    def _select_data_var(var_names):
        if preferred_var in var_names:
            return preferred_var
        if 'D' in var_names:
            return 'D'
        if not var_names:
            raise KeyError("No data variables found in land-station NetCDF.")
        return var_names[0]

    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        lat = np.asarray(ds['lat'].values, dtype=float)
        lon = np.asarray(ds['lon'].values, dtype=float)
        data_var = _select_data_var(list(ds.data_vars))
        d_val = np.asarray(ds[data_var].values, dtype=float)
        ds.close()
    except Exception:
        try:
            from netCDF4 import Dataset
            nc = Dataset(nc_path, 'r')
            lat = np.asarray(nc['lat'][:], dtype=float)
            lon = np.asarray(nc['lon'][:], dtype=float)
            candidate_vars = [
                name for name, var in nc.variables.items()
                if getattr(var, 'dimensions', ()) != ('lat',) and getattr(var, 'dimensions', ()) != ('lon',)
                and name not in {'lat', 'lon'}
            ]
            data_var = preferred_var if preferred_var in nc.variables else (
                'D' if 'D' in nc.variables else candidate_vars[0]
            )
            d_val = np.asarray(nc[data_var][:], dtype=float)
            nc.close()
        except Exception as e:
            print(f"[WARN] Failed to load land-station file {nc_path}: {e}")
            return np.array([]), np.array([]), np.array([])

    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(d_val)
    return lat[mask], lon[mask], d_val[mask]


def load_tx_land_station_points(nc_path=TX_LAND_STATION_FILE):
    """Load TX land-station coordinates and D values from NetCDF."""
    return load_land_station_points(nc_path, preferred_var='D')


def load_tmin_land_station_points(nc_path=TMIN_LAND_STATION_FILE):
    """Load TMIN land-station coordinates and anomaly values from NetCDF."""
    return load_land_station_points(nc_path, preferred_var='D')


def load_tvar_land_station_points(nc_path=TVAR_LAND_STATION_FILE):
    """Load TVAR land-station coordinates and variance values from NetCDF."""
    return load_land_station_points(nc_path, preferred_var='D')


def load_tskew_land_station_points(nc_path=TSKEW_LAND_STATION_FILE):
    """Load TSKEW land-station coordinates and skewness values from NetCDF."""
    return load_land_station_points(nc_path, preferred_var='D')


def filter_points_west_coast(lat, lon, values=None):
    """Filter point arrays to the same west-coast bounds used for buoy filtering."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    west_mask = (lat >= 30.0) & (lat <= 50.0) & (lon >= -130.0) & (lon <= -115.0)
    if values is None:
        return lat[west_mask], lon[west_mask]
    values = np.asarray(values, dtype=float)
    return lat[west_mask], lon[west_mask], values[west_mask]


def filter_points_east_coast(lat, lon, values=None):
    """Filter point arrays to the same east-coast bounds used for buoy filtering."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    east_mask = (lat >= 25.0) & (lat <= 45.0) & (lon >= -85.0) & (lon <= -65.0)
    if values is None:
        return lat[east_mask], lon[east_mask]
    values = np.asarray(values, dtype=float)
    return lat[east_mask], lon[east_mask], values[east_mask]


def plot_extreme_anomaly_with_land_stations(
    results,
    anomaly_type='max',
    land_station_nc_path=None,
    output_name=None,
    west_coast_only=False,
    land_station_label=None,
):
    """Create coast maps for a buoy anomaly type with matching land-station overlays."""
    if anomaly_type not in {'max', 'min'}:
        raise ValueError(f"Unsupported anomaly_type: {anomaly_type}")

    if land_station_nc_path is None:
        land_station_nc_path = (
            TX_LAND_STATION_FILE if anomaly_type == 'max' else TMIN_LAND_STATION_FILE
        )
    if output_name is None:
        output_name = (
            'max_anomaly_with_TX_land_stations.png'
            if anomaly_type == 'max'
            else 'min_anomaly_with_TMIN_land_stations.png'
        )
    if land_station_label is None:
        land_station_label = (
            'TX land stations' if anomaly_type == 'max' else 'TMIN land stations'
        )

    anomaly_descriptor = 'Maximum' if anomaly_type == 'max' else 'Minimum'
    anomaly_short = 'Max' if anomaly_type == 'max' else 'Min'

    print(f"\n P4C: CREATING {anomaly_descriptor.upper()}-ANOMALY MAP WITH {land_station_label.upper()}")
    print("=" * 60)

    filtered_results = [r for r in results if r['anomaly_type'] == anomaly_type]
    df_extreme = pd.DataFrame(filtered_results)
    if df_extreme.empty:
        print(f"[WARN] No {anomaly_type}-anomaly data available for land-station overlay map")
        return None

    land_lat_all, land_lon_all, land_d_all = load_land_station_points(land_station_nc_path)
    land_w_lat, land_w_lon, land_w_d = filter_points_west_coast(land_lat_all, land_lon_all, land_d_all)
    land_e_lat, land_e_lon, land_e_d = filter_points_east_coast(land_lat_all, land_lon_all, land_d_all)
    if west_coast_only:
        print(
            f"Loaded {len(land_lat_all)} {land_station_label}; "
            f"{len(land_w_lat)} within west-coast bounds (30-50N, -130 to -115)"
        )
    else:
        print(
            f"Loaded {len(land_lat_all)} {land_station_label}; "
            f"west-bounds={len(land_w_lat)}, east-bounds={len(land_e_lat)}"
        )

    fig = plt.figure(figsize=(22, 9))
    projection = ccrs.PlateCarree()

    for subplot_idx, coast, title in [
        (1, 'west', f'{anomaly_descriptor} Temperature Anomalies - West Coast + {land_station_label}'),
        (2, 'east', f'{anomaly_descriptor} Temperature Anomalies - East Coast + {land_station_label}'),
    ]:
        df_filtered = df_extreme[df_extreme['coast'] == coast].dropna(subset=['latitude', 'longitude'])
        if len(df_filtered) == 0:
            print(f"[WARN] No {anomaly_type} data for {coast} coast")
            continue

        ax = fig.add_subplot(1, 2, subplot_idx, projection=projection)
        ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='#b3d9ff')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, linestyle='--')

        lon_range = df_filtered['longitude'].max() - df_filtered['longitude'].min()
        lat_range = df_filtered['latitude'].max() - df_filtered['latitude'].min()
        if lon_range < 100 and lat_range < 50:
            extent = [
                df_filtered['longitude'].min() - 6,
                df_filtered['longitude'].max() + 6,
                df_filtered['latitude'].min() - 6,
                df_filtered['latitude'].max() + 6,
            ]
            ax.set_extent(extent, crs=projection)
        else:
            ax.set_global()

        if coast == 'west':
            land_lat_panel, land_lon_panel, land_d_panel = land_w_lat, land_w_lon, land_w_d
        else:
            land_lat_panel, land_lon_panel, land_d_panel = land_e_lat, land_e_lon, land_e_d

        if west_coast_only and coast == 'east':
            land_lat_panel = np.array([])
            land_lon_panel = np.array([])
            land_d_panel = np.array([])

        anomalies = np.asarray(df_filtered['average_anomaly'].values, dtype=float)
        all_values = anomalies
        if len(land_d_panel) > 0:
            all_values = np.concatenate([all_values, np.asarray(land_d_panel, dtype=float)])
        abs_max = np.nanmax(np.abs(all_values))
        if not np.isfinite(abs_max) or abs_max == 0:
            abs_max = 1e-6
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        buoy_scatter = ax.scatter(
            df_filtered['longitude'], df_filtered['latitude'],
            c=anomalies,
            cmap='coolwarm',
            s=250,
            alpha=0.9,
            edgecolors='black',
            linewidth=1.3,
            norm=norm,
            transform=projection,
            zorder=6,
        )

        # Overlay land-station points with same colormap as buoys and triangle markers.
        if len(land_lat_panel) > 0:
            ax.scatter(
                land_lon_panel,
                land_lat_panel,
                c=land_d_panel,
                cmap='coolwarm',
                norm=norm,
                s=75,
                alpha=0.95,
                marker='^',
                edgecolors='white',
                linewidths=0.8,
                transform=projection,
                zorder=8,
            )

        cbar = plt.colorbar(buoy_scatter, ax=ax, orientation='horizontal', pad=0.08, shrink=0.85)
        cbar.set_label(f'Average {anomaly_short} Anomaly (degC)', fontsize=11, weight='bold')
        ax.set_title(title, fontsize=12, weight='bold', pad=10)

        if len(land_lat_panel) > 0:
            legend_handles = [
                Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
                       markeredgecolor='black', markersize=8, label='Buoy station'),
                Line2D([0], [0], marker='^', color='none', markerfacecolor='gray',
                       markeredgecolor='white', markersize=9, label=f'Land station ({land_station_label})'),
            ]
            ax.legend(handles=legend_handles, loc='lower left', framealpha=0.8, fontsize=9)

        print(f"[OK] {title} ({len(df_filtered)} buoys)")

    fig.suptitle(
        f'NDBC Buoy {anomaly_descriptor} Temperature Anomalies with {land_station_label} Overlay',
        fontsize=15, weight='bold', y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] Saved: {output_path}")
    return output_path


def plot_max_anomaly_with_tx_land_stations(
    results,
    output_name='max_anomaly_with_TX_land_stations.png',
    west_coast_only=False,
):
    """Create max-anomaly coast maps and overlay TX land-station points."""
    return plot_extreme_anomaly_with_land_stations(
        results,
        anomaly_type='max',
        land_station_nc_path=TX_LAND_STATION_FILE,
        output_name=output_name,
        west_coast_only=west_coast_only,
        land_station_label='TX land stations',
    )


def plot_min_anomaly_with_tmin_land_stations(
    results,
    output_name='min_anomaly_with_TMIN_land_stations.png',
    west_coast_only=False,
):
    """Create min-anomaly coast maps and overlay TMIN land-station points."""
    return plot_extreme_anomaly_with_land_stations(
        results,
        anomaly_type='min',
        land_station_nc_path=TMIN_LAND_STATION_FILE,
        output_name=output_name,
        west_coast_only=west_coast_only,
        land_station_label='TMIN land stations',
    )


def plot_min_anomaly_pdf_with_tmin_land_stations(
    results,
    output_name='min_anomaly_pdf_buoy_vs_tmin_land.png',
    west_coast_only=False,
):
    """Plot overlaid PDFs of minimum anomalies for buoys and TMIN land stations."""
    print("\n P4D: CREATING PDF OVERLAY (MIN BUOYS VS TMIN LAND STATIONS)")
    print("=" * 60)

    min_results = [r for r in results if r['anomaly_type'] == 'min']
    df_min = pd.DataFrame(min_results)
    if df_min.empty:
        print("[WARN] No minimum-anomaly buoy data available for PDF plot")
        return None

    if west_coast_only:
        df_min = df_min[df_min['coast'] == 'west']

    buoy_vals = pd.to_numeric(df_min['average_anomaly'], errors='coerce').to_numpy(dtype=float)
    buoy_vals = buoy_vals[np.isfinite(buoy_vals)]

    land_lat, land_lon, land_vals = load_tmin_land_station_points(TMIN_LAND_STATION_FILE)
    if west_coast_only:
        _, _, land_vals = filter_points_west_coast(land_lat, land_lon, land_vals)

    land_vals = np.asarray(land_vals, dtype=float)
    land_vals = land_vals[np.isfinite(land_vals)]

    if buoy_vals.size < 2 or land_vals.size < 2:
        print("[WARN] Insufficient data to compute KDE PDFs")
        return None

    combined = np.concatenate([buoy_vals, land_vals])
    x_min = float(np.nanmin(combined))
    x_max = float(np.nanmax(combined))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        print("[WARN] Degenerate anomaly range; skipping PDF plot")
        return None

    pad = 0.05 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    fig, ax = plt.subplots(figsize=(12, 7))

    # KDE can fail on near-constant arrays; use normalized histograms as fallback.
    try:
        buoy_pdf = gaussian_kde(buoy_vals)(x_grid)
        ax.plot(x_grid, buoy_pdf, color='#1f77b4', linewidth=2.8, label=f'Buoy min anomalies (n={buoy_vals.size})')
    except Exception:
        ax.hist(
            buoy_vals,
            bins=35,
            density=True,
            histtype='step',
            color='#1f77b4',
            linewidth=2.0,
            label=f'Buoy min anomalies (hist, n={buoy_vals.size})',
        )

    try:
        land_pdf = gaussian_kde(land_vals)(x_grid)
        ax.plot(x_grid, land_pdf, color='#d62728', linewidth=2.8, label=f'TMIN land observations (n={land_vals.size})')
    except Exception:
        ax.hist(
            land_vals,
            bins=35,
            density=True,
            histtype='step',
            color='#d62728',
            linewidth=2.0,
            label=f'TMIN land observations (hist, n={land_vals.size})',
        )

    buoy_mean = float(np.nanmean(buoy_vals))
    land_mean = float(np.nanmean(land_vals))
    ax.axvline(buoy_mean, color='#1f77b4', linestyle='--', linewidth=1.4, alpha=0.9)
    ax.axvline(land_mean, color='#d62728', linestyle='--', linewidth=1.4, alpha=0.9)

    coast_label = 'West Coast only' if west_coast_only else 'West + East coasts'
    ax.set_title(
        f'PDF of Minimum Temperature Anomalies\nBuoy Stations vs TMIN Land Observations ({coast_label})',
        fontsize=13,
        weight='bold',
    )
    ax.set_xlabel('Minimum Temperature Anomaly (degC)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(
        f"[OK] PDF created with buoy n={buoy_vals.size}, land n={land_vals.size}; "
        f"mean_buoy={buoy_mean:.3f}, mean_land={land_mean:.3f}"
    )
    print(f"[SAVE] Saved: {output_path}")
    return output_path


def compute_buoy_variance_from_anomalies(results, anomaly_type='min'):
    """
    Compute variance of anomalies for each buoy from the anomaly data in results.
    Uses std_yearly_anomaly where variance = std^2.
    Returns dict with station_id as key and variance as value.
    """
    variance_dict = {}
    for result in results:
        if result.get('anomaly_type') != anomaly_type:
            continue
        station_id = result.get('station_id')
        std_anom = result.get('std_yearly_anomaly')
        if station_id and std_anom is not None and np.isfinite(std_anom):
            variance_dict[station_id] = float(std_anom ** 2)
    return variance_dict


def compute_buoy_skewness_from_anomalies(results, anomaly_type='min'):
    """
    Compute skewness of anomalies for each buoy by reading from the stored NetCDF files.
    Returns dict with station_id as key and skewness as value.
    """
    skewness_dict = {}
    for result in results:
        if result.get('anomaly_type') != anomaly_type:
            continue
        station_id = result.get('station_id')
        nc_path = result.get('netcdf_path')
        if station_id and nc_path:
            try:
                years, yearly_anom, _ = _read_extreme_anomaly_netcdf(nc_path)
                yearly_anom = np.asarray(yearly_anom, dtype=float)
                yearly_anom = yearly_anom[np.isfinite(yearly_anom)]
                if yearly_anom.size > 2:
                    skew_val = float(skew(yearly_anom))
                    if np.isfinite(skew_val):
                        skewness_dict[station_id] = skew_val
            except Exception as e:
                pass  # Skip stations where NetCDF read fails
    return skewness_dict


def plot_min_anomaly_histograms_by_coast_with_tmin_land_stations(
    results,
    output_name='min_anomaly_histograms_by_coast_buoy_vs_tmin_land.png',
    bins=28,
):
    """Plot bar histograms with PDF trend lines for buoy vs TMIN minimum anomalies by coast."""
    print("\n P4E: CREATING HISTOGRAM + PDF OVERLAYS BY COAST (MIN BUOYS VS TMIN LAND)")
    print("=" * 60)

    min_results = [r for r in results if r['anomaly_type'] == 'min']
    df_min = pd.DataFrame(min_results)
    if df_min.empty:
        print("[WARN] No minimum-anomaly buoy data available for histogram plot")
        return None

    land_lat_all, land_lon_all, land_vals_all = load_tmin_land_station_points(TMIN_LAND_STATION_FILE)
    west_land_lat, west_land_lon, west_land_vals = filter_points_west_coast(
        land_lat_all, land_lon_all, land_vals_all
    )
    east_land_lat, east_land_lon, east_land_vals = filter_points_east_coast(
        land_lat_all, land_lon_all, land_vals_all
    )

    coast_data = [
        (
            'west',
            'West Coast',
            pd.to_numeric(
                df_min.loc[df_min['coast'] == 'west', 'average_anomaly'], errors='coerce'
            ).to_numpy(dtype=float),
            np.asarray(west_land_vals, dtype=float),
        ),
        (
            'east',
            'East Coast',
            pd.to_numeric(
                df_min.loc[df_min['coast'] == 'east', 'average_anomaly'], errors='coerce'
            ).to_numpy(dtype=float),
            np.asarray(east_land_vals, dtype=float),
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    for ax, (_coast, coast_label, buoy_vals, land_vals) in zip(axes, coast_data):
        buoy_vals = buoy_vals[np.isfinite(buoy_vals)]
        land_vals = land_vals[np.isfinite(land_vals)]

        if buoy_vals.size < 2 or land_vals.size < 2:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            ax.set_title(f'{coast_label}: Minimum Anomaly Distributions')
            ax.grid(alpha=0.2, linestyle='--')
            continue

        combined = np.concatenate([buoy_vals, land_vals])
        x_min = float(np.nanmin(combined))
        x_max = float(np.nanmax(combined))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            ax.text(0.5, 0.5, 'Degenerate range', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            ax.set_title(f'{coast_label}: Minimum Anomaly Distributions')
            ax.grid(alpha=0.2, linestyle='--')
            continue

        pad = 0.05 * (x_max - x_min)
        bin_edges = np.linspace(x_min - pad, x_max + pad, int(bins) + 1)
        x_grid = np.linspace(x_min - pad, x_max + pad, 600)

        ax.hist(
            buoy_vals,
            bins=bin_edges,
            density=True,
            alpha=0.35,
            color='#1f77b4',
            edgecolor='#1f77b4',
            linewidth=0.9,
            label=f'Buoy histogram (n={buoy_vals.size})',
        )
        ax.hist(
            land_vals,
            bins=bin_edges,
            density=True,
            alpha=0.35,
            color='#d62728',
            edgecolor='#d62728',
            linewidth=0.9,
            label=f'TMIN land histogram (n={land_vals.size})',
        )

        # Draw smooth PDF trend lines over the histogram bars.
        try:
            buoy_pdf = gaussian_kde(buoy_vals)(x_grid)
            ax.plot(
                x_grid,
                buoy_pdf,
                color='#0d4f8b',
                linewidth=2.6,
                label='Buoy PDF trend',
                zorder=8,
            )
        except Exception:
            pass

        try:
            land_pdf = gaussian_kde(land_vals)(x_grid)
            ax.plot(
                x_grid,
                land_pdf,
                color='#8f1a1a',
                linewidth=2.6,
                label='TMIN land PDF trend',
                zorder=8,
            )
        except Exception:
            pass

        buoy_mean = float(np.nanmean(buoy_vals))
        land_mean = float(np.nanmean(land_vals))
        ax.axvline(buoy_mean, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.axvline(land_mean, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.9)

        ax.set_title(f'{coast_label}: Minimum Anomaly Distributions', fontsize=12, weight='bold')
        ax.set_xlabel('Minimum Temperature Anomaly (degC)', fontsize=10)
        ax.grid(alpha=0.25, linestyle='--')
        ax.legend(framealpha=0.9, fontsize=9)

        print(
            f"[OK] {coast_label} histogram: buoy n={buoy_vals.size}, land n={land_vals.size}, "
            f"mean_buoy={buoy_mean:.3f}, mean_land={land_mean:.3f}"
        )

    axes[0].set_ylabel('Probability Density', fontsize=10)
    fig.suptitle(
        'Histogram PDFs of Minimum Temperature Anomalies\n'
        'Buoy Stations vs TMIN Land Observations by Coast',
        fontsize=14,
        weight='bold',
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVE] Saved: {output_path}")
    return output_path


def plot_variance_comparison_pdf_with_tvar_land_stations(
    results,
    output_name='variance_pdf_buoy_vs_tvar_land.png',
    west_coast_only=False,
):
    """Plot overlaid PDFs of buoy anomaly variance vs TVAR land-station variance."""
    print("\n P6A: CREATING VARIANCE PDF OVERLAY (BUOYS VS TVAR LAND STATIONS)")
    print("=" * 60)

    # Compute variance from min anomalies for buoys
    variance_dict = compute_buoy_variance_from_anomalies(results, anomaly_type='min')
    min_results = [r for r in results if r['anomaly_type'] == 'min']
    df_min = pd.DataFrame(min_results)
    
    # Merge variance into dataframe
    df_min['variance'] = df_min['station_id'].map(variance_dict)
    df_min = df_min.dropna(subset=['variance'])
    
    if df_min.empty:
        print("[WARN] No buoy variance data available for PDF plot")
        return None

    if west_coast_only:
        df_min = df_min[df_min['coast'] == 'west']

    buoy_vals = pd.to_numeric(df_min['variance'], errors='coerce').to_numpy(dtype=float)
    buoy_vals = buoy_vals[np.isfinite(buoy_vals)]

    land_lat, land_lon, land_vals = load_tvar_land_station_points(TVAR_LAND_STATION_FILE)
    if west_coast_only:
        _, _, land_vals = filter_points_west_coast(land_lat, land_lon, land_vals)

    land_vals = np.asarray(land_vals, dtype=float)
    land_vals = land_vals[np.isfinite(land_vals)]

    if buoy_vals.size < 2 or land_vals.size < 2:
        print("[WARN] Insufficient data to compute KDE PDFs for variance")
        return None

    combined = np.concatenate([buoy_vals, land_vals])
    x_min = float(np.nanmin(combined))
    x_max = float(np.nanmax(combined))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        print("[WARN] Degenerate variance range; skipping PDF plot")
        return None

    pad = 0.05 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    fig, ax = plt.subplots(figsize=(12, 7))

    try:
        buoy_pdf = gaussian_kde(buoy_vals)(x_grid)
        ax.plot(x_grid, buoy_pdf, color='#1f77b4', linewidth=2.8, label=f'Buoy variance (n={buoy_vals.size})')
    except Exception:
        ax.hist(
            buoy_vals,
            bins=35,
            density=True,
            histtype='step',
            color='#1f77b4',
            linewidth=2.0,
            label=f'Buoy variance (hist, n={buoy_vals.size})',
        )

    try:
        land_pdf = gaussian_kde(land_vals)(x_grid)
        ax.plot(x_grid, land_pdf, color='#d62728', linewidth=2.8, label=f'TVAR land observations (n={land_vals.size})')
    except Exception:
        ax.hist(
            land_vals,
            bins=35,
            density=True,
            histtype='step',
            color='#d62728',
            linewidth=2.0,
            label=f'TVAR land observations (hist, n={land_vals.size})',
        )

    buoy_mean = float(np.nanmean(buoy_vals))
    land_mean = float(np.nanmean(land_vals))
    ax.axvline(buoy_mean, color='#1f77b4', linestyle='--', linewidth=1.4, alpha=0.9)
    ax.axvline(land_mean, color='#d62728', linestyle='--', linewidth=1.4, alpha=0.9)

    coast_label = 'West Coast only' if west_coast_only else 'West + East coasts'
    ax.set_title(
        f'PDF of Temperature Variance\nBuoy Stations vs TVAR Land Observations ({coast_label})',
        fontsize=13,
        weight='bold',
    )
    ax.set_xlabel('Temperature Variance (degC²)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(
        f"[OK] Variance PDF created with buoy n={buoy_vals.size}, land n={land_vals.size}; "
        f"mean_buoy={buoy_mean:.3f}, mean_land={land_mean:.3f}"
    )
    print(f"[SAVE] Saved: {output_path}")
    return output_path


def plot_variance_histograms_by_coast_with_tvar_land_stations(
    results,
    output_name='variance_histograms_by_coast_buoy_vs_tvar_land.png',
    bins=28,
):
    """Plot bar histograms with PDF trend lines for buoy variance vs TVAR by coast."""
    print("\n P6B: CREATING VARIANCE HISTOGRAM + PDF OVERLAYS BY COAST (BUOYS VS TVAR LAND)")
    print("=" * 60)

    # Compute variance from min anomalies for buoys
    variance_dict = compute_buoy_variance_from_anomalies(results, anomaly_type='min')
    min_results = [r for r in results if r['anomaly_type'] == 'min']
    df_min = pd.DataFrame(min_results)
    df_min['variance'] = df_min['station_id'].map(variance_dict)
    df_min = df_min.dropna(subset=['variance'])
    
    if df_min.empty:
        print("[WARN] No buoy variance data available for histogram plot")
        return None

    land_lat_all, land_lon_all, land_vals_all = load_tvar_land_station_points(TVAR_LAND_STATION_FILE)
    west_land_lat, west_land_lon, west_land_vals = filter_points_west_coast(
        land_lat_all, land_lon_all, land_vals_all
    )
    east_land_lat, east_land_lon, east_land_vals = filter_points_east_coast(
        land_lat_all, land_lon_all, land_vals_all
    )

    coast_data = [
        (
            'west',
            'West Coast',
            pd.to_numeric(
                df_min.loc[df_min['coast'] == 'west', 'variance'], errors='coerce'
            ).to_numpy(dtype=float),
            np.asarray(west_land_vals, dtype=float),
        ),
        (
            'east',
            'East Coast',
            pd.to_numeric(
                df_min.loc[df_min['coast'] == 'east', 'variance'], errors='coerce'
            ).to_numpy(dtype=float),
            np.asarray(east_land_vals, dtype=float),
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    for ax, (_coast, coast_label, buoy_vals, land_vals) in zip(axes, coast_data):
        buoy_vals = buoy_vals[np.isfinite(buoy_vals)]
        land_vals = land_vals[np.isfinite(land_vals)]

        if buoy_vals.size < 2 or land_vals.size < 2:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            ax.set_title(f'{coast_label}: Variance Distributions')
            ax.grid(alpha=0.2, linestyle='--')
            continue

        combined = np.concatenate([buoy_vals, land_vals])
        x_min = float(np.nanmin(combined))
        x_max = float(np.nanmax(combined))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            ax.text(0.5, 0.5, 'Degenerate range', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11)
            ax.set_title(f'{coast_label}: Variance Distributions')
            ax.grid(alpha=0.2, linestyle='--')
            continue

        pad = 0.05 * (x_max - x_min)
        bin_edges = np.linspace(x_min - pad, x_max + pad, int(bins) + 1)
        x_grid = np.linspace(x_min - pad, x_max + pad, 600)

        ax.hist(
            buoy_vals,
            bins=bin_edges,
            density=True,
            alpha=0.35,
            color='#1f77b4',
            edgecolor='#1f77b4',
            linewidth=0.9,
            label=f'Buoy variance histogram (n={buoy_vals.size})',
        )
        ax.hist(
            land_vals,
            bins=bin_edges,
            density=True,
            alpha=0.35,
            color='#d62728',
            edgecolor='#d62728',
            linewidth=0.9,
            label=f'TVAR land histogram (n={land_vals.size})',
        )

        # Draw smooth PDF trend lines over the histogram bars.
        try:
            buoy_pdf = gaussian_kde(buoy_vals)(x_grid)
            ax.plot(
                x_grid,
                buoy_pdf,
                color='#0d4f8b',
                linewidth=2.6,
                label='Buoy PDF trend',
                zorder=8,
            )
        except Exception:
            pass

        try:
            land_pdf = gaussian_kde(land_vals)(x_grid)
            ax.plot(
                x_grid,
                land_pdf,
                color='#8f1a1a',
                linewidth=2.6,
                label='TVAR land PDF trend',
                zorder=8,
            )
        except Exception:
            pass

        buoy_mean = float(np.nanmean(buoy_vals))
        land_mean = float(np.nanmean(land_vals))
        ax.axvline(buoy_mean, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.axvline(land_mean, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.9)

        ax.set_title(f'{coast_label}: Variance Distributions', fontsize=12, weight='bold')
        ax.set_xlabel('Temperature Variance (degC²)', fontsize=10)
        ax.grid(alpha=0.25, linestyle='--')
        ax.legend(framealpha=0.9, fontsize=9)

        print(
            f"[OK] {coast_label} variance histogram: buoy n={buoy_vals.size}, land n={land_vals.size}, "
            f"mean_buoy={buoy_mean:.3f}, mean_land={land_mean:.3f}"
        )

    axes[0].set_ylabel('Probability Density', fontsize=10)
    fig.suptitle(
        'Histogram PDFs of Temperature Variance\n'
        'Buoy Stations vs TVAR Land Observations by Coast',
        fontsize=14,
        weight='bold',
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVE] Saved: {output_path}")
    return output_path


def plot_skewness_comparison_pdf_with_tskew_land_stations(
    results,
    output_name='skewness_pdf_buoy_vs_tskew_land.png',
    west_coast_only=False,
):
    """Plot overlaid PDFs of buoy anomaly skewness vs TSKEW land-station skewness."""
    print("\n P7A: CREATING SKEWNESS PDF OVERLAY (BUOYS VS TSKEW LAND STATIONS)")
    print("=" * 60)

    skewness_dict = compute_buoy_skewness_from_anomalies(results, anomaly_type='min')
    min_results = [r for r in results if r['anomaly_type'] == 'min']
    df_min = pd.DataFrame(min_results)
    df_min['skewness'] = df_min['station_id'].map(skewness_dict)
    df_min = df_min.dropna(subset=['skewness'])
    
    if df_min.empty:
        print("[WARN] No buoy skewness data available for PDF plot")
        return None

    if west_coast_only:
        df_min = df_min[df_min['coast'] == 'west']

    buoy_vals = pd.to_numeric(df_min['skewness'], errors='coerce').to_numpy(dtype=float)
    buoy_vals = buoy_vals[np.isfinite(buoy_vals)]

    land_lat, land_lon, land_vals = load_tskew_land_station_points(TSKEW_LAND_STATION_FILE)
    if west_coast_only:
        _, _, land_vals = filter_points_west_coast(land_lat, land_lon, land_vals)

    land_vals = np.asarray(land_vals, dtype=float)
    land_vals = land_vals[np.isfinite(land_vals)]

    if buoy_vals.size < 2 or land_vals.size < 2:
        print("[WARN] Insufficient data to compute KDE PDFs for skewness")
        return None

    combined = np.concatenate([buoy_vals, land_vals])
    x_min = float(np.nanmin(combined))
    x_max = float(np.nanmax(combined))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        print("[WARN] Degenerate skewness range; skipping PDF plot")
        return None

    pad = 0.05 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 600)

    fig, ax = plt.subplots(figsize=(12, 7))

    try:
        buoy_pdf = gaussian_kde(buoy_vals)(x_grid)
        ax.plot(x_grid, buoy_pdf, color='#1f77b4', linewidth=2.8, label=f'Buoy skewness (n={buoy_vals.size})')
    except Exception:
        ax.hist(buoy_vals, bins=35, density=True, histtype='step', color='#1f77b4', linewidth=2.0,
                label=f'Buoy skewness (hist, n={buoy_vals.size})')

    try:
        land_pdf = gaussian_kde(land_vals)(x_grid)
        ax.plot(x_grid, land_pdf, color='#d62728', linewidth=2.8, label=f'TSKEW land observations (n={land_vals.size})')
    except Exception:
        ax.hist(land_vals, bins=35, density=True, histtype='step', color='#d62728', linewidth=2.0,
                label=f'TSKEW land observations (hist, n={land_vals.size})')

    buoy_mean = float(np.nanmean(buoy_vals))
    land_mean = float(np.nanmean(land_vals))
    ax.axvline(buoy_mean, color='#1f77b4', linestyle='--', linewidth=1.4, alpha=0.9)
    ax.axvline(land_mean, color='#d62728', linestyle='--', linewidth=1.4, alpha=0.9)

    coast_label = 'West Coast only' if west_coast_only else 'West + East coasts'
    ax.set_title(f'PDF of Temperature Skewness\nBuoy Stations vs TSKEW Land Observations ({coast_label})',
                 fontsize=13, weight='bold')
    ax.set_xlabel('Temperature Skewness', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[OK] Skewness PDF created with buoy n={buoy_vals.size}, land n={land_vals.size}; "
          f"mean_buoy={buoy_mean:.3f}, mean_land={land_mean:.3f}")
    print(f"[SAVE] Saved: {output_path}")
    return output_path


def plot_skewness_histograms_by_coast_with_tskew_land_stations(
    results,
    output_name='skewness_histograms_by_coast_buoy_vs_tskew_land.png',
    bins=28,
):
    """Plot bar histograms with PDF trend lines for buoy skewness vs TSKEW by coast."""
    print("\n P7B: CREATING SKEWNESS HISTOGRAM + PDF OVERLAYS BY COAST (BUOYS VS TSKEW LAND)")
    print("=" * 60)

    skewness_dict = compute_buoy_skewness_from_anomalies(results, anomaly_type='min')
    min_results = [r for r in results if r['anomaly_type'] == 'min']
    df_min = pd.DataFrame(min_results)
    df_min['skewness'] = df_min['station_id'].map(skewness_dict)
    df_min = df_min.dropna(subset=['skewness'])
    
    if df_min.empty:
        print("[WARN] No buoy skewness data available for histogram plot")
        return None

    land_lat_all, land_lon_all, land_vals_all = load_tskew_land_station_points(TSKEW_LAND_STATION_FILE)
    west_land_lat, west_land_lon, west_land_vals = filter_points_west_coast(
        land_lat_all, land_lon_all, land_vals_all
    )
    east_land_lat, east_land_lon, east_land_vals = filter_points_east_coast(
        land_lat_all, land_lon_all, land_vals_all
    )

    coast_data = [
        ('west', 'West Coast', pd.to_numeric(df_min.loc[df_min['coast'] == 'west', 'skewness'], errors='coerce').to_numpy(dtype=float),
         np.asarray(west_land_vals, dtype=float)),
        ('east', 'East Coast', pd.to_numeric(df_min.loc[df_min['coast'] == 'east', 'skewness'], errors='coerce').to_numpy(dtype=float),
         np.asarray(east_land_vals, dtype=float)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    for ax, (_coast, coast_label, buoy_vals, land_vals) in zip(axes, coast_data):
        buoy_vals = buoy_vals[np.isfinite(buoy_vals)]
        land_vals = land_vals[np.isfinite(land_vals)]

        if buoy_vals.size < 2 or land_vals.size < 2:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center', va='center', fontsize=11)
            ax.set_title(f'{coast_label}: Skewness Distributions')
            ax.grid(alpha=0.2, linestyle='--')
            continue

        combined = np.concatenate([buoy_vals, land_vals])
        x_min, x_max = float(np.nanmin(combined)), float(np.nanmax(combined))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            ax.text(0.5, 0.5, 'Degenerate range', transform=ax.transAxes, ha='center', va='center', fontsize=11)
            ax.set_title(f'{coast_label}: Skewness Distributions')
            ax.grid(alpha=0.2, linestyle='--')
            continue

        pad = 0.05 * (x_max - x_min)
        bin_edges = np.linspace(x_min - pad, x_max + pad, int(bins) + 1)
        x_grid = np.linspace(x_min - pad, x_max + pad, 600)

        ax.hist(buoy_vals, bins=bin_edges, density=True, alpha=0.35, color='#1f77b4', edgecolor='#1f77b4',
                linewidth=0.9, label=f'Buoy skewness histogram (n={buoy_vals.size})')
        ax.hist(land_vals, bins=bin_edges, density=True, alpha=0.35, color='#d62728', edgecolor='#d62728',
                linewidth=0.9, label=f'TSKEW land histogram (n={land_vals.size})')

        try:
            buoy_pdf = gaussian_kde(buoy_vals)(x_grid)
            ax.plot(x_grid, buoy_pdf, color='#0d4f8b', linewidth=2.6, label='Buoy PDF trend', zorder=8)
        except Exception:
            pass

        try:
            land_pdf = gaussian_kde(land_vals)(x_grid)
            ax.plot(x_grid, land_pdf, color='#8f1a1a', linewidth=2.6, label='TSKEW land PDF trend', zorder=8)
        except Exception:
            pass

        buoy_mean, land_mean = float(np.nanmean(buoy_vals)), float(np.nanmean(land_vals))
        ax.axvline(buoy_mean, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.axvline(land_mean, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.9)

        ax.set_title(f'{coast_label}: Skewness Distributions', fontsize=12, weight='bold')
        ax.set_xlabel('Temperature Skewness', fontsize=10)
        ax.grid(alpha=0.25, linestyle='--')
        ax.legend(framealpha=0.9, fontsize=9)

        print(f"[OK] {coast_label} skewness histogram: buoy n={buoy_vals.size}, land n={land_vals.size}, "
              f"mean_buoy={buoy_mean:.3f}, mean_land={land_mean:.3f}")

    axes[0].set_ylabel('Probability Density', fontsize=10)
    fig.suptitle('Histogram PDFs of Temperature Skewness\nBuoy Stations vs TSKEW Land Observations by Coast',
                 fontsize=14, weight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVE] Saved: {output_path}")
    return output_path


def plot_skewness_with_tskew_land_stations(
    results,
    output_name='skewness_with_TSKEW_land_stations.png',
    west_coast_only=False,
):
    """Create coast maps for skewness with matching TSKEW land-station overlays."""
    print("\n P7C: CREATING SKEWNESS MAP WITH TSKEW LAND STATIONS")
    print("=" * 60)

    skewness_dict = compute_buoy_skewness_from_anomalies(results, anomaly_type='min')
    min_results = [r for r in results if r['anomaly_type'] == 'min']
    df_min = pd.DataFrame(min_results)
    df_min['skewness'] = df_min['station_id'].map(skewness_dict)
    df_min = df_min.dropna(subset=['skewness'])
    
    if df_min.empty:
        print("[WARN] No skewness data available for map plot")
        return None

    land_lat_all, land_lon_all, land_d_all = load_tskew_land_station_points(TSKEW_LAND_STATION_FILE)
    land_w_lat, land_w_lon, land_w_d = filter_points_west_coast(land_lat_all, land_lon_all, land_d_all)
    land_e_lat, land_e_lon, land_e_d = filter_points_east_coast(land_lat_all, land_lon_all, land_d_all)
    
    print(f"Loaded {len(land_lat_all)} TSKEW land stations; west-bounds={len(land_w_lat)}, east-bounds={len(land_e_lat)}")

    fig = plt.figure(figsize=(22, 9))
    projection = ccrs.PlateCarree()

    for subplot_idx, coast, title in [(1, 'west', f'Temperature Skewness - West Coast + TSKEW Land Stations'),
                                       (2, 'east', f'Temperature Skewness - East Coast + TSKEW Land Stations')]:
        df_filtered = df_min[df_min['coast'] == coast].dropna(subset=['latitude', 'longitude'])
        if len(df_filtered) == 0:
            print(f"[WARN] No skewness data for {coast} coast")
            continue

        ax = fig.add_subplot(1, 2, subplot_idx, projection=projection)
        ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='#b3d9ff')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, linestyle='--')

        lon_range = df_filtered['longitude'].max() - df_filtered['longitude'].min()
        lat_range = df_filtered['latitude'].max() - df_filtered['latitude'].min()
        if lon_range < 100 and lat_range < 50:
            extent = [df_filtered['longitude'].min() - 6, df_filtered['longitude'].max() + 6,
                      df_filtered['latitude'].min() - 6, df_filtered['latitude'].max() + 6]
            ax.set_extent(extent, crs=projection)
        else:
            ax.set_global()

        land_lat_panel = land_w_lat if coast == 'west' else land_e_lat
        land_lon_panel = land_w_lon if coast == 'west' else land_e_lon
        land_d_panel = land_w_d if coast == 'west' else land_e_d

        skewnesses = np.asarray(df_filtered['skewness'].values, dtype=float)
        all_values = skewnesses
        if len(land_d_panel) > 0:
            all_values = np.concatenate([all_values, np.asarray(land_d_panel, dtype=float)])
        
        abs_max = np.nanmax(np.abs(all_values))
        if not np.isfinite(abs_max) or abs_max == 0:
            abs_max = 1e-6
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        buoy_scatter = ax.scatter(df_filtered['longitude'], df_filtered['latitude'], c=skewnesses,
                                  cmap='coolwarm', s=250, alpha=0.9, edgecolors='black', linewidth=1.3,
                                  norm=norm, transform=projection, zorder=6)

        if len(land_lat_panel) > 0:
            ax.scatter(land_lon_panel, land_lat_panel, c=land_d_panel, cmap='coolwarm', norm=norm,
                      s=75, alpha=0.95, marker='^', edgecolors='white', linewidths=0.8,
                      transform=projection, zorder=8)

        cbar = plt.colorbar(buoy_scatter, ax=ax, orientation='horizontal', pad=0.08, shrink=0.85)
        cbar.set_label('Temperature Skewness', fontsize=11, weight='bold')
        ax.set_title(title, fontsize=12, weight='bold', pad=10)

        if len(land_lat_panel) > 0:
            legend_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
                                    markeredgecolor='black', markersize=8, label='Buoy station'),
                             Line2D([0], [0], marker='^', color='none', markerfacecolor='gray',
                                    markeredgecolor='white', markersize=9, label='TSKEW land station')]
            ax.legend(handles=legend_handles, loc='lower left', framealpha=0.8, fontsize=9)

        print(f"[OK] {title} ({len(df_filtered)} buoys)")

    fig.suptitle(f'NDBC Buoy Temperature Skewness with TSKEW Land Stations Overlay',
                 fontsize=15, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] Saved: {output_path}")
    return output_path


def plot_variance_with_tvar_land_stations(
    results,
    output_name='variance_with_TVAR_land_stations.png',
    west_coast_only=False,
):
    """Create coast maps for variance with matching TVAR land-station overlays."""
    print("\n P6C: CREATING VARIANCE MAP WITH TVAR LAND STATIONS")
    print("=" * 60)

    # Compute variance from min anomalies for buoys
    variance_dict = compute_buoy_variance_from_anomalies(results, anomaly_type='min')
    min_results = [r for r in results if r['anomaly_type'] == 'min']
    df_min = pd.DataFrame(min_results)
    df_min['variance'] = df_min['station_id'].map(variance_dict)
    df_min = df_min.dropna(subset=['variance'])
    
    if df_min.empty:
        print("[WARN] No variance data available for map plot")
        return None

    land_lat_all, land_lon_all, land_d_all = load_tvar_land_station_points(TVAR_LAND_STATION_FILE)
    land_w_lat, land_w_lon, land_w_d = filter_points_west_coast(land_lat_all, land_lon_all, land_d_all)
    land_e_lat, land_e_lon, land_e_d = filter_points_east_coast(land_lat_all, land_lon_all, land_d_all)
    
    if west_coast_only:
        print(
            f"Loaded {len(land_lat_all)} TVAR land stations; "
            f"{len(land_w_lat)} within west-coast bounds (30-50N, -130 to -115)"
        )
    else:
        print(
            f"Loaded {len(land_lat_all)} TVAR land stations; "
            f"west-bounds={len(land_w_lat)}, east-bounds={len(land_e_lat)}"
        )

    fig = plt.figure(figsize=(22, 9))
    projection = ccrs.PlateCarree()

    for subplot_idx, coast, title in [
        (1, 'west', f'Temperature Variance - West Coast + TVAR Land Stations'),
        (2, 'east', f'Temperature Variance - East Coast + TVAR Land Stations'),
    ]:
        df_filtered = df_min[df_min['coast'] == coast].dropna(subset=['latitude', 'longitude'])
        if len(df_filtered) == 0:
            print(f"[WARN] No variance data for {coast} coast")
            continue

        ax = fig.add_subplot(1, 2, subplot_idx, projection=projection)
        ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='#b3d9ff')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':', alpha=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, linestyle='--')

        lon_range = df_filtered['longitude'].max() - df_filtered['longitude'].min()
        lat_range = df_filtered['latitude'].max() - df_filtered['latitude'].min()
        if lon_range < 100 and lat_range < 50:
            extent = [
                df_filtered['longitude'].min() - 6,
                df_filtered['longitude'].max() + 6,
                df_filtered['latitude'].min() - 6,
                df_filtered['latitude'].max() + 6,
            ]
            ax.set_extent(extent, crs=projection)
        else:
            ax.set_global()

        if coast == 'west':
            land_lat_panel, land_lon_panel, land_d_panel = land_w_lat, land_w_lon, land_w_d
        else:
            land_lat_panel, land_lon_panel, land_d_panel = land_e_lat, land_e_lon, land_e_d

        if west_coast_only and coast == 'east':
            land_lat_panel = np.array([])
            land_lon_panel = np.array([])
            land_d_panel = np.array([])

        variances = np.asarray(df_filtered['variance'].values, dtype=float)
        all_values = variances
        if len(land_d_panel) > 0:
            all_values = np.concatenate([all_values, np.asarray(land_d_panel, dtype=float)])
        
        v_min = np.nanmin(all_values)
        v_max = np.nanmax(all_values)
        if not np.isfinite(v_min) or not np.isfinite(v_max) or v_min == v_max:
            v_min = 0
            v_max = 1e-6

        norm = plt.Normalize(vmin=v_min, vmax=v_max)

        buoy_scatter = ax.scatter(
            df_filtered['longitude'], df_filtered['latitude'],
            c=variances,
            cmap='viridis',
            s=250,
            alpha=0.9,
            edgecolors='black',
            linewidth=1.3,
            norm=norm,
            transform=projection,
            zorder=6,
        )

        # Overlay land-station points with same colormap as buoys and triangle markers.
        if len(land_lat_panel) > 0:
            ax.scatter(
                land_lon_panel,
                land_lat_panel,
                c=land_d_panel,
                cmap='viridis',
                norm=norm,
                s=75,
                alpha=0.95,
                marker='^',
                edgecolors='white',
                linewidths=0.8,
                transform=projection,
                zorder=8,
            )

        cbar = plt.colorbar(buoy_scatter, ax=ax, orientation='horizontal', pad=0.08, shrink=0.85)
        cbar.set_label('Temperature Variance (degC²)', fontsize=11, weight='bold')
        ax.set_title(title, fontsize=12, weight='bold', pad=10)

        if len(land_lat_panel) > 0:
            legend_handles = [
                Line2D([0], [0], marker='o', color='none', markerfacecolor='gray',
                       markeredgecolor='black', markersize=8, label='Buoy station'),
                Line2D([0], [0], marker='^', color='none', markerfacecolor='gray',
                       markeredgecolor='white', markersize=9, label='TVAR land station'),
            ]
            ax.legend(handles=legend_handles, loc='lower left', framealpha=0.8, fontsize=9)

        print(f"[OK] {title} ({len(df_filtered)} buoys)")

    fig.suptitle(
        f'NDBC Buoy Temperature Variance with TVAR Land Stations Overlay',
        fontsize=15, weight='bold', y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] Saved: {output_path}")
    return output_path


def plot_cartopy_maps_by_coast(results, window_size=100):
    """
    Create Cartopy maps for max and min anomalies, separated by coast.
    """
    print("\n P4: CREATING CARTOPY MAPS")
    print("=" * 60)
    
    # Separate max and min results
    max_results = [r for r in results if r['anomaly_type'] == 'max']
    min_results = [r for r in results if r['anomaly_type'] == 'min']
    
    df_max = pd.DataFrame(max_results)
    df_min = pd.DataFrame(min_results)
    
    print(f"Plotting {len(df_max)} stations for max anomalies")
    print(f"Plotting {len(df_min)} stations for min anomalies\n")
    
    # Create figure with 4 subplots (2x2: west/east, max/min)
    fig = plt.figure(figsize=(24, 16))
    projection = ccrs.PlateCarree()
    
    # Process each combination
    configurations = [
        (df_max, 'west', 'Maximum Temperature Anomalies - West Coast', 1),
        (df_max, 'east', 'Maximum Temperature Anomalies - East Coast', 2),
        (df_min, 'west', 'Minimum Temperature Anomalies - West Coast', 3),
        (df_min, 'east', 'Minimum Temperature Anomalies - East Coast', 4),
    ]
    
    for df, coast, title, subplot_idx in configurations:
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
        
        # Set extent to focus on data
        lon_range = df_filtered['longitude'].max() - df_filtered['longitude'].min()
        lat_range = df_filtered['latitude'].max() - df_filtered['latitude'].min()
        
        if lon_range < 100 and lat_range < 50:
            extent = [df_filtered['longitude'].min() - 5, df_filtered['longitude'].max() + 5,
                      df_filtered['latitude'].min() - 5, df_filtered['latitude'].max() + 5]
            ax.set_extent(extent, crs=projection)
        else:
            ax.set_global()
        
        # Plot data
        anomalies = df_filtered['average_anomaly'].values
        abs_max = max(abs(anomalies.min()), abs(anomalies.max()))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        
        scatter = ax.scatter(
            df_filtered['longitude'], df_filtered['latitude'],
            c=anomalies,
            cmap='coolwarm',
            s=250,
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
        'NDBC Buoy Extreme Temperature Anomalies\n'
        'Maximum and Minimum Warm Season Anomalies by Coast Region',
        fontsize=16, weight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(FIGURES_DIR, 'extreme_anomalies_coast_maps.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVE] Saved: {output_path}")
    
    plt.close(fig)
    
    return df_max, df_min


def plot_tighter_bounds_maps(results, output_name='tighter_bounds_extreme_coast_maps.png'):
    """
    Create Cartopy maps with tighter (5-95 percentile) temperature bounds.
    Helps distinguish subtle differences in temperature anomalies.
    """
    print("\n P4B: CREATING TIGHTER-BOUNDS CARTOPY MAPS")
    print("=" * 60)
    
    # Separate max and min results
    max_results = [r for r in results if r['anomaly_type'] == 'max']
    min_results = [r for r in results if r['anomaly_type'] == 'min']
    
    df_max = pd.DataFrame(max_results)
    df_min = pd.DataFrame(min_results)
    
    # Compute 5-95 percentile bounds for tighter color scale
    max_lower = df_max['average_anomaly'].quantile(0.05)
    max_upper = df_max['average_anomaly'].quantile(0.95)
    min_lower = df_min['average_anomaly'].quantile(0.05)
    min_upper = df_min['average_anomaly'].quantile(0.95)
    
    print(f"Max anomaly bounds (5-95%): {max_lower:.2f} to {max_upper:.2f} degC")
    print(f"Min anomaly bounds (5-95%): {min_lower:.2f} to {min_upper:.2f} degC")
    print(f"Original auto-scale max: {df_max['average_anomaly'].min():.2f} to {df_max['average_anomaly'].max():.2f}")
    print(f"Original auto-scale min: {df_min['average_anomaly'].min():.2f} to {df_min['average_anomaly'].max():.2f}\n")
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(24, 16))
    projection = ccrs.PlateCarree()
    
    # Configuration: (dataframe, coast, title, subplot, vmin, vmax)
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
        
        # Use tighter bounds for colormap
        norm = TwoSlopeNorm(vmin=vmin_sym, vcenter=0, vmax=vmax_sym)
        
        scatter = ax.scatter(
            df_filtered['longitude'], df_filtered['latitude'],
            c=df_filtered['average_anomaly'].values,
            cmap='coolwarm',
            s=250,
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
        'NDBC Buoy Extreme Temperature Anomalies (Tighter Bounds)\n'
        'Maximum and Minimum Warm Season Anomalies - 5-95 Percentile Scaling',
        fontsize=16, weight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(FIGURES_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVE] Saved: {output_path}")
    
    plt.close(fig)
    
    return df_max, df_min


def print_summary_statistics(df_max, df_min):
    """Print summary statistics"""
    print("\nP5: SUMMARY STATISTICS")
    print("=" * 60)
    
    print("\n--- MAXIMUM ANOMALIES ---")
    print(f"Total stations: {len(df_max)}")
    if len(df_max) > 0:
        west = len(df_max[df_max['coast'] == 'west'])
        east = len(df_max[df_max['coast'] == 'east'])
        print(f"  West Coast: {west}")
        print(f"  East Coast: {east}")
        print(f"Range: {df_max['average_anomaly'].min():.4f} degC - {df_max['average_anomaly'].max():.4f} degC")
        print(f"Mean: {df_max['average_anomaly'].mean():.4f} degC")
        print(f"Std Dev: {df_max['average_anomaly'].std():.4f} degC")
        
        print("\n  TOP 10 HIGHEST MAX ANOMALIES:")
        top_max = df_max.nlargest(10, 'average_anomaly')[['station_id', 'coast', 'average_anomaly', 'max_yearly_anomaly']]
        print(top_max.to_string(index=False))
    
    print("\n--- MINIMUM ANOMALIES ---")
    print(f"Total stations: {len(df_min)}")
    if len(df_min) > 0:
        west = len(df_min[df_min['coast'] == 'west'])
        east = len(df_min[df_min['coast'] == 'east'])
        print(f"  West Coast: {west}")
        print(f"  East Coast: {east}")
        print(f"Range: {df_min['average_anomaly'].min():.4f} degC - {df_min['average_anomaly'].max():.4f} degC")
        print(f"Mean: {df_min['average_anomaly'].mean():.4f} degC")
        print(f"Std Dev: {df_min['average_anomaly'].std():.4f} degC")
        
        print("\n  TOP 10 MOST NEGATIVE MIN ANOMALIES:")
        top_min = df_min.nsmallest(10, 'average_anomaly')[['station_id', 'coast', 'average_anomaly', 'min_yearly_anomaly']]
        print(top_min.to_string(index=False))


def main():
    """Execute complete average anomaly pipeline"""
    
    start_time = datetime.now()
    
    try:
        # Strict local mode: no scraping/network buoy discovery
        WEST_COAST_ONLY = False

        valid_buoys = discover_cached_buoys_from_warm_netcdf(
            west_coast_only=WEST_COAST_ONLY,
            min_years=5,
            target_count=TARGET_VALID_BUOYS,
        )
        if not valid_buoys:
            print("[ERROR] No cached buoys with sufficient local data. Exiting.")
            return

        # Process all buoys with parallelization for max/min anomalies
        results = process_all_buoys_parallel(valid_buoys, max_workers=8)
        if not results:
            print("[ERROR] No successful buoy processing. Exiting.")
            return

        save_results(results)

        df_max, df_min = plot_cartopy_maps_by_coast(results, window_size=100)
        
        plot_max_anomaly_with_tx_land_stations(
            results,
            output_name='max_anomaly_with_TX_land_stations.png',
            west_coast_only=False,
        )
        plot_min_anomaly_with_tmin_land_stations(
            results,
            output_name='min_anomaly_with_TMIN_land_stations.png',
            west_coast_only=False,
        )

        # Create tighter-bounds version for better distinction
        df_max_tight, df_min_tight = plot_tighter_bounds_maps(results, output_name='tighter_bounds_extreme_coast_maps.png')

        print_summary_statistics(df_max, df_min)

        elapsed = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("[OK] PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nProcessed: {len(results)} total results (buoys x anomaly types)")
        print(f"Output: {NC_DIR}/  (NetCDF files)")
        print(f"Output: {FIGURES_DIR}/  (Cartopy maps)")
        print(f"Elapsed time: {elapsed}")
        
    except KeyboardInterrupt:
        print("\n\n[WARN] Pipeline interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
