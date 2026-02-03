# NDBC Buoy Comprehensive Analysis

## Overview

This pipeline processes all NDBC (National Data Buoy Center) buoys with 10+ years of historical data, computes warm-season temperature statistics (variance and skewness), and visualizes them on world maps using Cartopy.

## Methodology

Following the approach from `nbdc statistical analyses.ipynb`:

1. **Data Collection**: Scrape NDBC historical stdmet directory to discover all available buoys
2. **Filtering**: Select buoys with ≥10 years of continuous data
3. **Warm Season Identification**: For each buoy, identify the warm season as ±100 days around the climatological temperature maximum
4. **Anomaly Computation**: Calculate daily temperature anomalies relative to the seasonal cycle
5. **Statistics**: Compute variance and skewness of warm-season anomalies
6. **Visualization**: Plot results on Cartopy world maps

## Scripts

### 1. `process_all_buoys.py`

**Purpose**: Discover and process all NDBC buoys with sufficient data

**Key Features**:
- Web scraping to discover all available NDBC stations
- Parallel processing for efficient data collection
- Automatic warm season detection for each buoy
- Computation of variance and skewness statistics
- Caching of results for reuse

**Usage**:
```python
python scripts/process_all_buoys.py
```

**Output**:
- `cache/buoy_statistics.pkl`: Pickle file containing processed statistics for all buoys

### 2. `plot_buoy_statistics_cartopy.py`

**Purpose**: Visualize buoy statistics on world maps using Cartopy

**Key Features**:
- Global maps showing variance and skewness
- Automatic projection and extent selection
- Regional zoom maps for detailed analysis
- Color-coded scatter plots with labeled stations
- Publication-quality figures

**Usage**:
```python
python scripts/plot_buoy_statistics_cartopy.py
```

**Output**:
- `figures/buoy_statistics_cartopy_maps.png`: Global variance and skewness maps
- `figures/buoy_stats_*.png`: Regional detail maps

## Requirements

Install dependencies:
```bash
pip install numpy pandas scipy matplotlib cartopy xarray requests beautifulsoup4 tqdm
```

## Workflow

### Complete Pipeline

```bash
# Step 1: Process all buoys (takes several hours)
python scripts/process_all_buoys.py

# Step 2: Generate visualizations
python scripts/plot_buoy_statistics_cartopy.py
```

### Quick Start (Using Cached Data)

If results are already cached in `cache/buoy_statistics.pkl`, you can skip directly to visualization:

```bash
python scripts/plot_buoy_statistics_cartopy.py
```

## Output Description

### Statistics Computed

For each buoy with ≥10 years of data:

- **Warm Season Window**: Days around climatological temperature maximum (±100 days)
- **Mean Variance**: Average variance of daily temperature anomalies during warm season
- **Mean Skewness**: Average skewness of daily temperature anomalies during warm season
- **Yearly Time Series**: Annual variance and skewness values

### Map Interpretation

**Variance Map** (Yellow-Orange-Red colormap):
- Shows variability in warm-season temperatures
- Higher values indicate greater temperature variability
- Useful for identifying regions with unstable warm-season conditions

**Skewness Map** (Blue-White-Red diverging colormap):
- Shows asymmetry in temperature distribution
- Positive (red): More extreme warm anomalies than cold
- Negative (blue): More extreme cold anomalies than warm
- Zero (white): Symmetric distribution

## Data Sources

- **NDBC Historical Data**: https://www.ndbc.noaa.gov/data/historical/stdmet/
- **Buoy Locations**: NDBC station metadata and hardcoded coordinates

## File Structure

```
Zeppetello Research/
├── scripts/
│   ├── process_all_buoys.py          # Main processing script
│   ├── plot_buoy_statistics_cartopy.py  # Visualization script
│   └── plot_station_maps.py          # Original 4-buoy plotting script
├── data_buoy_helpers.py              # Data loading utilities
├── stat_buoy_helpers.py              # Statistical analysis utilities
├── cache/
│   ├── buoy_statistics.pkl           # Processed statistics
│   └── *.txt.gz                      # Downloaded NDBC data files
├── figures/
│   └── buoy_statistics_cartopy_maps.png
└── nc/
    └── *_warm_anomalies_*.nc         # Individual buoy NetCDF files
```

## Performance Notes

- **Processing Time**: ~2-5 hours for 100+ buoys (depends on network speed and CPU)
- **Disk Space**: ~500 MB for cached data files
- **Memory**: ~2-4 GB RAM recommended for parallel processing
- **Network**: Stable internet connection required for data download

## Customization

### Adjust Warm Season Window

Edit `window_size` parameter in `process_all_buoys.py`:

```python
results = process_all_buoys(valid_buoys, window_size=50, max_workers=1)
```

### Change Minimum Years Threshold

Edit `min_years` in `process_all_buoys.py`:

```python
valid_buoys = filter_buoys_by_years(all_buoys, min_years=15, max_workers=20)
```

### Add Custom Regions

Edit `regions` parameter in `plot_buoy_statistics_cartopy.py`:

```python
regions = {
    'Custom Region': (lon_min, lon_max, lat_min, lat_max),
}
create_detailed_regional_maps(df, regions=regions)
```

## Troubleshooting

### Missing Dependencies

```bash
# Install cartopy (may require conda on Windows)
conda install -c conda-forge cartopy

# Or pip (may have GEOS/PROJ issues)
pip install cartopy
```

### Memory Issues

Reduce `max_workers` in `process_all_buoys()`:

```python
results = process_all_buoys(valid_buoys, max_workers=1)  # Sequential processing
```

### Network Timeouts

Increase timeout in `discover_all_buoys()`:

```python
r = requests.get(BASE_URL, timeout=60)  # Increase from 30 to 60 seconds
```

## Citation

Based on methodology from:
- NDBC buoy temperature analysis (Zeppetello Research)
- Statistical analysis of warm season temperature extremes

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.
