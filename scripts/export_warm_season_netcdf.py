"""
Export warm-season anomaly NetCDFs for each station detected in the `cache/` directory
or for a provided station list. Produces files in `nc/` and a summary variance PNG.

Run: conda activate zepp; python scripts\export_warm_season_netcdf.py
"""
from pathlib import Path
import re
import os
import numpy as np
from data_buoy_helpers import load_station
from stat_buoy_helpers import write_warm_season_netcdf

OUT_DIR = Path("nc")
OUT_DIR.mkdir(exist_ok=True)

# Build station list from cache
cache_dir = Path("cache")
station_ids = set()
if cache_dir.exists():
    for f in cache_dir.iterdir():
        m = re.match(r"(\d+)h", f.name)
        if m:
            station_ids.add(m.group(1))

if not station_ids:
    # As a small fallback, allow manual list here
    station_ids = {"46001"}

created = []
for sid in sorted(station_ids):
    print(f"Loading station {sid}")
    try:
        df_filled, completeness = load_station(sid)
    except Exception as e:
        print(f"Failed to load {sid}: {e}")
        continue
    if df_filled is None:
        print(f"No data for {sid}; skipping")
        continue
    try:
        path = write_warm_season_netcdf(df_filled, sid, out_dir=str(OUT_DIR), window_size=50, target_days=100)
        print("Wrote:", path)
        created.append(path)
    except Exception as e:
        print(f"Failed to write NetCDF for {sid}: {e}")

# Summary plotting (requires xarray)
try:
    import xarray as xr
    import matplotlib.pyplot as plt
    files = list(OUT_DIR.glob("*_warm_anomalies_*d.nc"))
    if files:
        datasets = []
        stations = []
        for p in files:
            try:
                ds = xr.open_dataset(p)
                datasets.append(ds)
                stations.append(p.stem.split("_")[0])
            except Exception as e:
                print("Failed to open", p, e)
        if datasets:
            var_stack = np.vstack([ds['anomalies'].to_numpy().var(axis=0) for ds in datasets])
            fig, ax = plt.subplots(1,1, figsize=(10, max(3, 0.3*len(stations))))
            im = ax.imshow(var_stack, aspect='auto', interpolation='nearest')
            ax.set_yticks(np.arange(len(stations)))
            ax.set_yticklabels(stations)
            ax.set_xlabel('day (0..{})'.format(var_stack.shape[1]-1))
            ax.set_title('Per-day variance of warm-season anomalies (stations x days)')
            fig.colorbar(im, ax=ax, label='variance')
            plt.tight_layout()
            out_png = OUT_DIR / 'summary_variance.png'
            fig.savefig(out_png, dpi=150)
            print('Saved summary plot:', out_png)
    else:
        print('No NetCDF files to summarize.')
except Exception as e:
    print('Skipping summary plotting — xarray/matplotlib not available:', e)

print('Created files:', created)
