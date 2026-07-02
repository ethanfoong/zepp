# Research Workflow: Extreme Coastal Climatology
## NBDC Buoy & Land Observation Analysis Pipeline

**Project Lead**: UC Berkeley Prof. Zeppetello  
**Date Initiated**: Ongoing  
**Current Phase**: Data Visualization & Statistical Comparison

---

## Executive Summary

This document outlines the complete workflow for analyzing extreme coastal temperature variations using oceanic buoy data (NDBC - National Data Buoy Center) and land-based meteorological observations. The pipeline progresses from raw data collection through statistical analysis to publication-quality visualizations in PDF format.

---

## Phase 1: Data Acquisition & Infrastructure

### 1.1 Source Data Collection

**Oceanic Data (NDBC Buoys)**:
- **Source**: National Data Buoy Center (NOAA)
- **Base URL**: `https://www.ndbc.noaa.gov/data/historical/stdmet/`
- **Data Format**: Compressed `.txt.gz` files (standardized meteorological data)
- **Key Variables**:
  - ATMP (Air Temperature)
  - WTMP (Water Temperature)
  - Year, Month, Day, Hour

**Land Observations**:
- **Source**: NOAA/NCEI climate records
- **Variables Tracked**:
  - TMAX (Daily Maximum Temperature)
  - TMIN (Daily Minimum Temperature)
  - TVAR (Temperature Variance)
  - TSKEW (Temperature Skewness)

**Metadata**:
- Station locations (latitude/longitude) scraped from NDBC station table
- Station table: `https://www.ndbc.noaa.gov/data/stations/station_table.txt`
- Cached in: `cache/buoy_locations.json`

### 1.2 Data Infrastructure Setup

**Dependencies Installed**:
```
numpy, pandas, scipy
matplotlib, cartopy  # Visualization
xarray, netCDF4      # NetCDF export/import
requests, beautifulsoup4  # Web scraping
geopandas, folium    # Geospatial analysis
```

**Directory Structure Created**:
- `cache/` — Cached station locations, eligibility filters
- `nc/` — NetCDF output files for each buoy's statistics
- `figures/` — Generated PDF visualizations
- `scripts/` — Automated analysis pipelines

---

## Phase 2: Data Processing Pipeline

### 2.1 Helper Function Development

**File**: `data_buoy_helpers.py`

**Web Scraping Functions**:
1. `parse_station_table()` — Extract all station coordinates from NDBC table
   - Regex parsing for lat/lon in degree format (56.30 N, 148.02 W)
   - Output: Dict mapping station_id → (latitude, longitude)
   - Cached to JSON for performance

2. `list_station_files(station_id)` — Discover all available years for a buoy
   - Web scraping NDBC directory for patterns like `46001h1972.txt.gz`
   - Returns sorted list of URLs by year

3. `fetch_file(url, cache_dir)` — Download with local caching
   - One-time download per file, reusable from cache
   - Supports concurrent downloads via ThreadPoolExecutor

**Data Parsing Functions**:
1. `read_stdmet(url)` — Parse NOAA stdmet format
   - Handles gzip decompression
   - Extracts year from URL regex
   - Unique column naming (ATMP_1, ATMP_2, WTMP_1, etc.)
   - NA value handling (999.0, 999, 9999)

2. `read_stdmet_max(url)` — Extract daily maximum temperatures
   - Variant of `read_stdmet()` optimized for max computation

3. `read_stdmet_min(url)` — Extract daily minimum temperatures
   - Variant of `read_stdmet()` optimized for min computation

4. `load_station(station_id, years)` — Full station data pipeline
   - Fetches all files for station across year range
   - Concatenates multi-year data into single DataFrame
   - Removes Feb 29 (leap day) for temporal consistency
   - Returns cleaned, standardized format

**File**: `stat_buoy_helpers.py`

**Statistical Functions**:
1. `_drop_leap_day(df)` — Remove Feb 29 from leap years
   - Ensures consistent 365-day years across leap years
   - Leap year logic: divisible by 4, except centuries not divisible by 400

2. `get_warm_season_data(df, station, warm_season_window=100)` — Identify warm season
   - Computes daily mean temperature (average across all years)
   - Finds day of year with max temperature (climatological peak)
   - Extracts ±100 day window around peak
   - Returns filtered data + metadata tuple

3. `compute_warm_season_anomalies(df, station, warm_season_window=100)` — Calculate deviations
   - Builds seasonal climatology (mean temperature per day-of-year)
   - Computes daily anomalies: T_actual - T_climatology
   - Returns anomaly DataFrame

4. `write_warm_season_netcdf(station_id, anomalies, variance, skewness, ...)` — Export statistics
   - Saves results to NetCDF format for reuse
   - Stores: anomalies, variance, skewness, seasonal climatology
   - Output: `nc/{station_id}_warm_anomalies_{years}y_{window}d.nc`

5. `read_netcdf_statistics(filepath)` — Load pre-computed NetCDF results
   - Retrieves cached statistics without re-computation

**Seasonal Cycle Visualization Functions**:
1. `plot_dailymax_seasonal_cycle()` — Line plot of mean daily temperature
   - Daily maxima per day-of-year, smoothed with window
   - Shows climatological peak + annual variability bands

2. `plot_warm_season_heatmap()` — Heatmap of years vs. day-of-year
   - Color intensity = temperature anomaly magnitude
   - Reveals warm season timing shifts across years

3. `plot_warm_season_time_series()` — Time series of warm season mean anomalies
   - Year-by-year trend in warm season intensity
   - Useful for detecting long-term trends

### 2.2 Buoy Eligibility & Filtering

**Criteria Applied**:
- **Minimum Data Coverage**: ≥10 years of continuous observations
- **Geographic Distribution**: Prioritizes stations with global coverage
- **Target Coverage**: ~120 valid buoys with sufficient data

**Output**: `cache/buoy_eligibility_10y_by_coast.csv`
- Tracks which buoys meet criteria
- Groups by coast (e.g., "West Coast", "Gulf of Mexico", etc.)

---

## Phase 3: Statistical Analysis

### 3.1 Warm Season Anomaly Computation

**Methodology**:
1. For each buoy with ≥10 years of data:
   - Extract warm season window (±100 days around climatological max)
   - Compute daily mean temperature per day-of-year (climatology)
   - Calculate daily anomalies: T_anomaly = T_observed - T_climatology

2. Statistics computed across warm season anomalies:
   - **Variance**: Spread of daily anomalies (measure of variability)
   - **Skewness**: Asymmetry of anomaly distribution
     - Positive skew = more extreme warm events than cold
     - Negative skew = more extreme cold events than warm
   - **Time Series**: Year-by-year values for trend detection

### 3.2 Comparison with Land Observations

**Methodology**:
- Match buoy stations with nearby NOAA land meteorological stations
- Compute same statistics (variance, skewness) for land-based TMAX/TMIN
- Statistical comparison:
  - Histogram overlays (buoy vs. land)
  - PDF estimates (kernel density estimation)
  - Correlation analysis
  - Scatter plots with regression lines

**Output Files Generated**:
- `figures/variance_pdf_buoy_vs_tvar_land.png`
- `figures/skewness_pdf_buoy_vs_tskew_land.png`
- `figures/max_anomaly_pdf_buoy_vs_tmax_land.png`
- `figures/min_anomaly_pdf_buoy_vs_tmin_land.png`

**Key Insights Tracked**:
- Do buoy temperatures show higher/lower variance than land?
- Are warm anomalies or cold anomalies more extreme?
- Geographic variability in these differences

### 3.3 Extreme Anomaly Detection

**Methods**:
- Identify years with maximum/minimum anomalies per buoy
- Flag extreme events (e.g., top 10% of warm/cold anomalies)
- Map temporal co-occurrence of extremes across buoys
- Assess clustering patterns (do multiple buoys experience extremes simultaneously?)

---

## Phase 4: Data Export & Standardization

### 4.1 NetCDF File Creation

**Purpose**: Store computed statistics in standardized scientific format

**File Structure**: `nc/{station_id}_{metric}_{years}y_{window}d.nc`
- Example: `41001_warm_anomalies_50y_201d.nc`

**Variables Stored**:
- Anomaly time series (daily or yearly aggregates)
- Seasonal climatology (mean temperature per day-of-year)
- Statistics (variance, skewness, percentiles)
- Metadata (station ID, coordinates, data years, window size)

**Output Generation**:
- Script: `scripts/export_warm_season_netcdf.py`
- Processes all eligible buoys
- Standardizes naming conventions
- Enables interoperability with other tools (R, MATLAB, GrADS)

---

## Phase 5: Geospatial Visualization

### 5.1 Global Buoy Distribution Maps

**Visualization**: Cartopy-based world maps

**Layers Included**:
- Global coastline, land borders, ocean features
- Scatter plots of buoy locations
- Color coding by statistic (variance, skewness)
- Station ID labels for identification

**Outputs Generated**:
1. `figures/station_variance_map.png` — Global variance distribution
2. `figures/station_skewness_map.png` — Global skewness patterns
3. `figures/cartopy_variance_skewness_maps.png` — Side-by-side comparison
4. `figures/West_Coast_cartopy_variance_skewness_maps.png` — Regional zoom (Pacific)

### 5.2 Regional Analysis Maps

**Focus Areas**:
- West Coast (North America) — High data density
- Atlantic Coast — Historical station network
- Gulf of Mexico — Subtropical climate patterns
- Global extremes — Identify hotspots

**Features**:
- Tighter geographic bounds for coastal detail
- Higher-resolution coastline data
- Linked extreme anomaly maps (co-occurrence patterns)

**Output**: `figures/tighter_bounds_extreme_coast_maps.png`
- Multiple regional panels showing max/min anomalies by coast

### 5.3 Comparative Visualizations

**Histograms & PDFs**:
- Overlaid distributions comparing buoy vs. land observations
- Scripts: 
  - `scripts/buoy_obs_comparison_histogram.py`
  - `scripts/land_buoy_visualization_check.py`

**Examples**:
- `figures/variance_histograms_by_coast_buoy_vs_tvar_land.png`
- `figures/skewness_histograms_by_coast_buoy_vs_tskew_land.png`
- `figures/min_anomaly_histograms_by_coast_buoy_vs_tmin_land.png`

**Integrated Station Maps**:
- `figures/max_anomaly_with_TX_land_stations.png` — TMAX land overlay
- `figures/min_anomaly_with_TMIN_land_stations.png` — TMIN land overlay
- `figures/skewness_with_TSKEW_land_stations.png` — Skewness land overlay

---

## Phase 6: Analysis & Interpretation Notebooks

### 6.1 Exploratory Notebooks

**File**: `nbdc buoy data visualization.ipynb`
- Initial data exploration
- Single-buoy case studies
- Function testing and prototyping
- Visualization experimentation

**File**: `nbdc statistical analyses.ipynb`
- Warm season methodology development
- Variance/skewness computation validation
- Seasonal cycle plots
- Multi-buoy comparison drafts

**File**: `three buoy comparisons.ipynb`
- Focused analysis of 3 representative buoys
- Detailed seasonal cycles
- Year-to-year variability analysis
- Land-ocean comparison for selected sites

### 6.2 Supplementary Analysis

**File**: `soil_model.py`
- Coupled soil-atmosphere model
- Forcing data assembly (F, P, e, SST, DT)
- Parameter sets (r, G_s, v_c, Jackson coefficients)
- Model output generation for land temperature modeling

**File**: `nbdc_statistical_analyses.py`
- Standalone Python version of notebook analyses
- Reusable functions for batch processing
- Integration point for automated pipelines

---

## Phase 7: Automated Processing Pipelines

### 7.1 Complete Buoy Analysis Pipeline

**Script**: `scripts/complete_buoy_pipeline.py`

**Workflow**:
1. Load or download NDBC station locations
2. Discover all available buoys in historical database
3. Filter for ≥10 years of data
4. Process each buoy in parallel:
   - Load multi-year data
   - Identify warm season window
   - Compute variance & skewness
   - Export to NetCDF
5. Aggregate results for mapping

**Parallelization**:
- ThreadPoolExecutor for concurrent downloads
- ProcessPoolExecutor for statistical computation
- Configurable concurrency (default: CPU count)

**Caching Strategy**:
- Station locations cached in JSON (no re-download)
- Individual buoy results cached in pickle
- NetCDF files stored for future analysis
- Skip re-processing if results exist

**Runtime**: Several hours for full dataset (~120 buoys)

### 7.2 Visualization Generation Pipeline

**Script**: `scripts/plot_buoy_statistics_cartopy.py`

**Outputs**:
- Global variance map
- Global skewness map
- Regional zoom maps (West Coast, Atlantic, etc.)
- All saved as PNG with publication-quality resolution

**Features**:
- Automatic projection selection (Mercator, stereographic for poles)
- Custom colormaps (YlOrRd for variance, diverging for skewness)
- Scatter plot with station labels
- Figure size/resolution optimization

### 7.3 Supporting Analysis Scripts

**Coast-Based Eligibility Analysis**: `scripts/buoy_coast_eligibility_summary.py`
- Groups buoys by coast (geographic region)
- Summarizes data availability per coast
- Identifies gaps in coverage

**Statistical Variance Checks**: `scripts/variance_analysis_check.py`
- Validates variance computation across methods
- Tests edge cases (single year, all identical values, etc.)
- Checks for numerical stability

**NetCDF Testing**: `scripts/test_netcdf_statistics.py`
- Verifies NetCDF file creation
- Checks data integrity in exported files
- Validates metadata preservation

**Buoy Processing Tests**: `scripts/test_buoy_processing.py`
- Unit tests for individual buoy processing
- Edge case testing (missing data, format errors)
- Regression tests for known buoys

---

## Phase 8: Code Quality & Optimization

### 8.1 Identified Redundancies

**Analysis Document**: `OPTIMIZATION_ANALYSIS.md`

**Key Redundancies Found**:
1. **Duplicate Imports** (Lines detected in data_buoy_helpers.py & stat_buoy_helpers.py)
   - `from urllib.parse import urljoin` appears twice
   - `scipy.stats.norm` imported but unused

2. **Duplicate File Parsing Logic** (~70% code overlap)
   - `read_stdmet_max()`, `read_stdmet_min()`, `read_stdmet()`
   - Recommendation: Refactor to single parameterized function

3. **Duplicate Leap Year Handling**
   - Logic in `load_station()` duplicates `_drop_leap_day()` utility

4. **Duplicate Warm Season Calculations**
   - Multiple plotting functions recalculate warm season boundaries
   - Recommendation: Extract once per station, pass to all functions

### 8.2 Refactoring Roadmap

**Priority 1** (High Impact):
- Consolidate `read_stdmet_*()` functions → single parameterized function
- Extract warm season computation from plotting functions
- Eliminate duplicate imports

**Priority 2** (Medium Impact):
- Standardize error handling across helpers
- Add logging for debugging
- Implement retry logic for network failures

**Priority 3** (Maintenance):
- Add docstring standardization
- Type hints for all function signatures
- Comprehensive unit test suite

---

## Phase 9: Output Products & PDF Generation

### 9.1 Generated Visualizations

**Current Outputs** (in `figures/` directory):

**Global Maps** (PNG format):
- `station_variance_map.png` — Buoy variance distribution
- `station_skewness_map.png` — Buoy skewness distribution
- `cartopy_variance_skewness_maps.png` — Combined global view
- `extreme_anomalies_coast_maps.png` — Max/min anomaly hotspots

**Regional Analysis**:
- `West_Coast_cartopy_variance_skewness_maps.png` — Zoomed Pacific view
- `tighter_bounds_extreme_coast_maps.png` — Multi-coast regional panels

**Comparative Analysis**:
- `variance_pdf_buoy_vs_tvar_land.png` — PDF comparison
- `variance_histograms_by_coast_buoy_vs_tvar_land.png` — Histogram by region
- `skewness_pdf_buoy_vs_tskew_land.png` — Skewness comparison
- `skewness_histograms_by_coast_buoy_vs_tskew_land.png` — Regional histograms
- `min_anomaly_pdf_buoy_vs_tmin_land.png` — Min anomaly comparison
- `min_anomaly_histograms_by_coast_buoy_vs_tmin_land.png` — Regional min anomalies

**Integrated Visualizations**:
- `max_anomaly_with_TX_land_stations.png` — TMAX overlay
- `min_anomaly_with_TMIN_land_stations.png` — TMIN overlay
- `variance_with_TVAR_land_stations.png` — Variance overlay
- `skewness_with_TSKEW_land_stations.png` — Skewness overlay

**Supplementary**:
- `TMAX_HISTOGRAM_PDF.png` — Reference land TMAX distribution

### 9.2 PDF Compilation Pipeline

**Next Steps** (Not Yet Implemented):
1. Create summary report template
2. Batch convert PNG → PDF
3. Generate multi-page PDF document with:
   - Executive summary
   - Methodology section
   - Results section (all maps)
   - Analysis & interpretation
   - Data quality metrics
   - References

**Tools for Implementation**:
- `matplotlib.pyplot.savefig(..., format='pdf')`
- `reportlab` or `pypdf` for multi-page assembly
- Automated figure numbering & captions

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Data Acquisition                                   │
├─────────────────────────────────────────────────────────────┤
│  NDBC NOAA Server                 Land Weather Stations     │
│  ↓                                 ↓                        │
│  .txt.gz stdmet files             TMAX/TMIN/TVAR           │
│  (historical compressed)           records                 │
└────────────────┬────────────────────────┬──────────────────┘
                 │                        │
                 ↓                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Data Processing                                    │
├─────────────────────────────────────────────────────────────┤
│  parse_station_table()  → buoy_locations.json              │
│  list_station_files()   → URLs for each year               │
│  fetch_file()           → cache/ directory                 │
│  read_stdmet()          → parse compressed data            │
│  load_station()         → cleaned multi-year DataFrames    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Statistical Analysis                               │
├─────────────────────────────────────────────────────────────┤
│  get_warm_season_data()      → identify ±100 day window    │
│  compute_warm_season_anomalies() → T_anom = T - climatology│
│  Compute variance & skewness → statistics per buoy         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─────────────────────┬──────────────────┐
                 ↓                     ↓                  ↓
         ┌──────────────┐      ┌──────────────┐  ┌──────────────┐
         │ NetCDF       │      │ Pickle Cache │  │ Comparison   │
         │ Export       │      │ (complete_   │  │ Land vs.     │
         │ (nc/)        │      │  buoy_...)   │  │ Ocean        │
         │              │      │              │  │              │
         └──────────────┘      └──────────────┘  └──────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Geospatial Visualization                           │
├─────────────────────────────────────────────────────────────┤
│  plot_buoy_statistics_cartopy()                             │
│  ├─ Global variance map                                     │
│  ├─ Global skewness map                                     │
│  ├─ Regional zoom maps                                      │
│  └─ Comparative buoy vs. land histograms                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
         ┌──────────────────┐
         │ figures/ (PNG)   │
         │ - *.png outputs  │
         └──────────────────┘
                 │
                 ↓ [Future]
         ┌──────────────────┐
         │ PDF Report       │
         │ - Multi-page doc │
         │ - Summary +      │
         │   Figures        │
         └──────────────────┘
```

---

## Repository Structure (Current)

```
Zeppetello Research/
├── README.md                              # Project overview
├── RESEARCH_WORKFLOW.md                   # This document
├── OPTIMIZATION_ANALYSIS.md               # Code redundancy analysis
├── requirements.txt                       # Python dependencies
│
├── Core Python Modules:
│   ├── data_buoy_helpers.py               # NDBC scraping & parsing
│   ├── stat_buoy_helpers.py               # Statistical computation
│   ├── nbdc_statistical_analyses.py       # Batch analysis functions
│   └── soil_model.py                      # Coupled land model
│
├── Notebooks (Exploratory):
│   ├── nbdc buoy data visualization.ipynb
│   ├── nbdc statistical analyses.ipynb
│   └── three buoy comparisons.ipynb
│
├── scripts/                               # Automated pipelines
│   ├── complete_buoy_pipeline.py          # Main processing workflow
│   ├── plot_buoy_statistics_cartopy.py    # Map generation
│   ├── buoy_coast_eligibility_summary.py  # Regional grouping
│   ├── export_warm_season_netcdf.py       # NetCDF export
│   ├── buoy_obs_comparison_histogram.py   # Comparative plots
│   ├── land_buoy_visualization_check.py   # QA/QC plots
│   ├── list_west_coast_buoys.py           # Regional listing
│   ├── plot_station_maps.py               # Basic station mapping
│   ├── plot_tighter_bounds.py             # Regional zoom
│   ├── variance_analysis_check.py         # Computation validation
│   ├── test_buoy_processing.py            # Unit tests
│   └── test_netcdf_statistics.py          # NetCDF validation
│
├── cache/                                 # Persistent cache
│   ├── buoy_locations.json                # Station coordinates
│   ├── buoy_eligibility_10y_by_coast.csv  # Filtering results
│   └── complete_buoy_analysis.pkl         # Aggregated statistics
│
├── nc/                                    # NetCDF outputs
│   ├── 41001_warm_anomalies_50y_201d.nc
│   ├── 41002_warm_anomalies_53y_201d.nc
│   └── [120+ buoy NetCDF files]
│
├── figures/                               # Final visualizations
│   ├── cartopy_variance_skewness_maps.png
│   ├── variance_pdf_buoy_vs_tvar_land.png
│   ├── skewness_pdf_buoy_vs_tskew_land.png
│   ├── max_anomaly_with_TX_land_stations.png
│   └── [18 additional PNG files]
│
└── nbdc_files/                            # Source NOAA data (reference)
    ├── marine_obs_*.html                  # Station listings
    ├── Marine Obs by Program.kmz          # KML map data
    └── 46001h*.txt/                       # Sample raw data files
```

---

## Key Statistics & Milestones

| Metric | Value |
|--------|-------|
| **NDBC Buoys Discovered** | 140+ |
| **Buoys with ≥10 Years Data** | ~120 |
| **Coastal Regions Analyzed** | 5+ (US West, Gulf of Mexico, Atlantic, etc.) |
| **Years of Data** | 6-53 years per buoy |
| **Visualizations Generated** | 18+ PNG figures |
| **NetCDF Files Exported** | 100+ (one per buoy) |
| **Code Files** | 5 core Python modules + 12+ analysis scripts |
| **Notebooks** | 3 exploratory notebooks |
| **Redundancies Identified** | 7 major (documented in OPTIMIZATION_ANALYSIS.md) |

---

## Methodological References

**Warm Season Definition**:
- Identified as ±100 days around climatological temperature maximum
- Rationale: Captures peak warm season variability (e.g., heatwave season)
- Alternative windows tested: 50, 150, 201 days (stored in nc/ directory)

**Anomaly Calculation**:
- Daily anomalies = Observed Temperature - Seasonal Climatology
- Seasonal climatology = mean temperature per day-of-year (averaged across all years)
- Leap year handling: Feb 29 excluded to maintain 365-day consistency

**Statistical Metrics**:
- **Variance**: Quantifies temperature volatility (larger = more variable)
- **Skewness**: Asymmetry of distribution
  - Positive = tail toward warm anomalies (extreme heat bias)
  - Negative = tail toward cold anomalies (extreme cold bias)
  - Zero = symmetric distribution

---

## Future Work & Recommendations

### Short-term (Weeks 1-4)
1. **Optimize code** per OPTIMIZATION_ANALYSIS.md recommendations
2. **Implement PDF generation** pipeline from PNG figures
3. **Add comprehensive unit tests** to scripts/
4. **Document edge cases** (missing data, format errors)

### Medium-term (Months 2-3)
1. **Expand land station network** for deeper buoy-land comparison
2. **Implement trend detection** (long-term changes in variance/skewness)
3. **Add seasonal decomposition** (separate trend, seasonal, residual components)
4. **Uncertainty quantification** (confidence intervals, resampling)

### Long-term (6+ months)
1. **Climate model validation** (compare buoy/land statistics to CMIP6 models)
2. **Extreme event attribution** (connect extremes to large-scale climate drivers)
3. **Machine learning** (predict future extremes based on historical patterns)
4. **Interactive visualization dashboard** (web-based exploration tool)

---

## Contact & Resources

**Project Lead**: UC Berkeley Prof. Zeppetello  
**Data Sources**:
- NDBC: https://www.ndbc.noaa.gov/
- NOAA/NCEI: https://www.ncei.noaa.gov/

**Technical Documentation**:
- README.md — Quick start & overview
- OPTIMIZATION_ANALYSIS.md — Code redundancy findings
- Notebook comments — Methodology details
- Function docstrings — Implementation specifics

---

**Document Last Updated**: April 29, 2026  
**Status**: Active Development — PDF Compilation Phase Next
