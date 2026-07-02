import os, re, io, gzip, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm, skew
from urllib.parse import urljoin


BASE = "https://www.ndbc.noaa.gov/data/historical/stdmet/"

"""tackling leap years 

if lat > 0: northern hemisphere 

if year %4 == 0 and (year != 1900):
    is_leap = True
    delete T[59] #remove Feb 29

if lat < 0: southern hemisphere
    if year %4 == 0 and year != 1900:
    delete T[59 + 182] #halfway through the year 

#simpler solution get rid of july 1st 

high resolution cmip6 

kilometer scale climate models

"""


def _drop_leap_day(df):
    """Remove Feb 29 (day 60) only for leap years (divisible by 4, except centuries not divisible by 400)."""
    if "day_of_year" in df.columns and "year" in df.columns:
        is_leap_year = (df['year'] % 4 == 0) & ((df['year'] % 100 != 0) | (df['year'] % 400 == 0))
        return df[~((df['day_of_year'] == 60) & is_leap_year)]
    return df


#################### seasonal cycle helpers ########################

def plot_dailymax_seasonal_cycle(df, station, warm_season_window=50):
    """
    Plot seasonal temperature cycle using daily maximum ATMP.
    """

    df = _drop_leap_day(df)
    # --- Compute daily maxima per year ---
    daily_max = df.groupby(["year", "day_of_year"])["ATMP"].max().reset_index()

    # Pivot to shape (365 x n_years)
    pivot = daily_max.pivot(index="day_of_year", columns="year", values="ATMP")

    mean_cycle = pivot.mean(axis=1)   
    std_cycle = pivot.std(axis=1)     

    warm_center = int(mean_cycle.idxmax())   # day-of-year of climatological max
    warm_start = max(warm_center - warm_season_window, 1)
    warm_end = min(warm_center + warm_season_window, 365)

    print(f" Warm season for {station}: Days {warm_start}–{warm_end} (centered at {warm_center})")

    plt.figure(figsize=(10, 6))

    for y in pivot.columns:
        plt.plot(pivot.index, pivot[y], color="black", alpha=0.25, linewidth=0.6)

    # climatological daily max mean (red line)
    plt.plot(mean_cycle.index, mean_cycle, color="red", linewidth=2.5, label="Mean Seasonal Cycle")

    # warm season highlights
    plt.axvspan(warm_start, warm_end, color="lightblue", alpha=0.15, label="Warm Season")
    plt.axvline(warm_start, color="blue", linestyle="--", alpha=0.6)
    plt.axvline(warm_end, color="blue", linestyle="--", alpha=0.6)
    plt.text(warm_center, plt.ylim()[0] + 1, "Warm Season", color="blue",
             ha="center", va="bottom", fontsize=11)

    plt.title(f"NDBC {station}: Seasonal Cycle of Daily Max Air Temperature (ATMP)")
    plt.xlabel("Day of Year")
    plt.ylabel("Daily Maximum Air Temperature [°C]")
    plt.legend(["Yearly Daily Max", "Climatological Mean", "Warm Season"], loc="upper right")
    plt.tight_layout()
    plt.show()


#################### warm-season helpers ########################

def get_warm_season_data(df, window_size=50):
    """
    Identify and extract warm season data based on climatological maximum.
    """
    df = _drop_leap_day(df)
    # compute the multi-year daily mean using daily maxima
    daily_max = df.groupby(["year", "day_of_year"])['ATMP'].max().reset_index()
    daily_mean = daily_max.groupby('day_of_year')['ATMP'].mean()

    warm_center = int(daily_mean.idxmax())
    warm_start = max(warm_center - window_size, 1)
    warm_end = min(warm_center + window_size, 365)

    warm_season_mask = (df['day_of_year'] >= warm_start) & (df['day_of_year'] <= warm_end)
    warm_season_df = df[warm_season_mask].copy()

    return warm_season_df, (warm_start, warm_center, warm_end)


def compute_warm_season_anomalies(df, window_size=50):
    """
    Compute anomaly pivot table restricted to the warm season.
    """

    warm_df, (warm_start, warm_center, warm_end) = get_warm_season_data(df, window_size)
    daily_maxes = warm_df.groupby(['year', 'day_of_year'])['ATMP'].max().reset_index()
    pivot = daily_maxes.pivot(index='day_of_year', columns='year', values='ATMP')

    mean_cycle = pivot.mean(axis=1)
    anomalies = pivot.subtract(mean_cycle, axis=0)

    return anomalies, mean_cycle, (warm_start, warm_center, warm_end)


def plot_warm_season_heatmap(df, station, window_size=50):
    """Plot heatmap of temperature anomalies limited to warm season."""
    anomalies, mean_cycle, (warm_start, warm_center, warm_end) = compute_warm_season_anomalies(df, window_size)

    plt.figure(figsize=(12,6))
    plt.imshow(anomalies.T, aspect='auto', cmap='coolwarm',
               extent=[warm_start, warm_end, anomalies.columns.min(), anomalies.columns.max()],
               origin='lower')
    plt.colorbar(label="Temperature Anomaly [°C]")
    plt.axvline(warm_center, color='black', linestyle='--', alpha=0.5, label='Peak Temperature Day')
    plt.xlabel("Day of Year")
    plt.ylabel("Year")
    plt.title(f"NDBC {station}: Warm Season Temperature Anomalies\n(Days {warm_start}-{warm_end}, Peak: {warm_center})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_warm_season_time_series(df, station, window_size=50):
    """
    Plot a time series of mean warm-season temperature anomalies per year,
    with a KDE distribution plot showing skewness.
    """
    anomalies, mean_cycle, (warm_start, warm_center, warm_end) = compute_warm_season_anomalies(df, window_size)
    yearly_anom = anomalies.mean(axis=0)
    
    from scipy.stats import skew
    skewness = skew(yearly_anom.dropna())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #time series of anomalies
    ax1.plot(yearly_anom.index, yearly_anom.values, marker='o', color='darkorange', linewidth=2)
    ax1.axhline(0, color='black', lw=1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean Warm Season Temperature Anomaly [°C]")
    ax1.set_title(f"NDBC {station}: Warm Season Temperature Anomalies\n(Days {warm_start}-{warm_end}, Peak: {warm_center})")
    ax1.grid(True, alpha=0.3)
    
    # KDE distribution showing skewness
    ax2.hist(yearly_anom.dropna(), bins=15, density=True, alpha=0.6, color='lightcoral', edgecolor='black', label='Histogram')
    
    # KDE plot
    from scipy.stats import gaussian_kde
    data = yearly_anom.dropna()
    if len(data) > 1:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
        ax2.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
        
        # pinpointing mean and median
        mean_val = data.mean()
        median_val = data.median()
        ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}°C')
        ax2.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}°C')
    
    ax2.set_xlabel("Temperature Anomaly [°C]")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Distribution of Anomalies\nSkewness: {skewness:.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def _compute_yearly_var_skew_from_df(df):
    """
    compute yearly variance and skew of daily-max ATMP 
    based on pivot daily maxima to (day_of_year x year)
    then compute variance and skew per year.
    """
    df = _drop_leap_day(df)
    # pivot on daily maxima
    daily_maxes = df.groupby(['year', 'day_of_year'])['ATMP'].max().reset_index()
    pivot = daily_maxes.pivot(index='day_of_year', columns='year', values='ATMP')

    # all years 
    years = pivot.columns.tolist()
    if not years:
        return None, None, None

    yearly_var = pivot.var(axis=0)
    yearly_skew = pivot.apply(lambda col: skew(col.dropna()), axis=0)

    # reindex to continuous year range and interpolate missing skew
    all_years = np.arange(int(min(years)), int(max(years)) + 1)
    yearly_var = yearly_var.reindex(all_years)
    yearly_skew = yearly_skew.reindex(all_years).interpolate(limit_direction='both')

    return all_years, yearly_var, yearly_skew


def compare_stations_variance(station_data, stations=None, figsize_per_col=3.5):
    """
    Compare seasonal variance and skew across multiple stations side-by-side.
    """
    if stations is None:
        stations = list(station_data.keys())

    n = len(stations)
    if n == 0:
        raise ValueError("No stations provided for comparison")

    fig, axes = plt.subplots(2, n, figsize=(figsize_per_col * n, 6), sharex=False)
    if n == 1:
        axes = axes.reshape(2,1)

    for i, st in enumerate(stations):
        df = station_data.get(st)
        if df is None:
            # blank subplot with message
            axes[0, i].text(0.5, 0.5, f"No data for {st}", ha='center', va='center')
            axes[1, i].axis('off')
            axes[0, i].axis('off')
            continue

        all_years, yearly_var, yearly_skew = _compute_yearly_var_skew_from_df(df)
        if all_years is None:
            axes[0, i].text(0.5, 0.5, f"Insufficient data {st}", ha='center', va='center')
            axes[1, i].axis('off')
            axes[0, i].axis('off')
            continue

        # Top row: Variance
        axv = axes[0, i]
        axv.plot(all_years, yearly_var.values, marker='o', color='green')
        axv.set_title(f"{st}")
        if i == 0:
            axv.set_ylabel("Variance (°C²)")
        axv.grid(True, alpha=0.3)

        # Bottom row: Skewness
        axs = axes[1, i]
        axs.plot(all_years, yearly_skew.values, marker='o', color='purple')
        if i == 0:
            axs.set_ylabel("Skewness (γ₁)")
        axs.set_xlabel("Year")
        axs.grid(True, alpha=0.3)

    fig.suptitle("Comparison: Annual Variance (top) and Skewness (bottom) of Daily Max ATMP")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def write_warm_season_netcdf(df, station_id, out_dir="nc", window_size=50, target_days=None):
    """
    Create a NetCDF file of warm-season anomalies for a station.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    anomalies, mean_cycle, (warm_start, warm_center, warm_end) = compute_warm_season_anomalies(df, window_size)

    # anomalies: index = day_of_year (warm days), columns = years
    warm_days = anomalies.index.to_numpy()
    years = anomalies.columns.to_numpy()

    n_days = len(warm_days)
    if target_days is None:
        target_days = n_days

    # Build array shape (n_years, target_days)
    data = np.full((len(years), target_days), np.nan, dtype=np.float32)

    # Center the warm_days into the target_days window if sizes differ
    if n_days <= target_days:
        # place warm_days centered
        start = (target_days - n_days) // 2
        for i, yr in enumerate(years):
            row = anomalies[yr].reindex(warm_days).to_numpy()
            data[i, start:start + n_days] = row
        day_coords = np.arange(start, start + target_days)
    else:
        # truncate: take centered slice of warm_days to fit target_days
        start_idx = (n_days - target_days) // 2
        selected_days = warm_days[start_idx:start_idx + target_days]
        for i, yr in enumerate(years):
            row = anomalies[yr].reindex(selected_days).to_numpy()
            data[i, :] = row
        warm_days = selected_days

    # Build coordinates
    years_coord = years.astype(int)
    days_coord = np.arange(target_days)

    # Try using xarray for convenience, fall back to netCDF4
    fname = os.path.join(out_dir, f"{station_id}_warm_anomalies_{len(years)}y_{target_days}d.nc")
    try:
        import xarray as xr
        ds = xr.Dataset(
            {
                'anomalies': (('year', 'day'), (data)),
            },
            coords={
                'year': years_coord,
                'day': days_coord,
                'warm_day_of_year': (('day',), np.arange(warm_start, warm_start + target_days) if n_days <= target_days else warm_days),
            },
            attrs={
                'station_id': station_id,
                'warm_start': int(warm_start),
                'warm_center': int(warm_center),
                'warm_end': int(warm_end),
                'window_size': int(window_size),
            }
        )
        ds.to_netcdf(fname)
    except Exception:
        # fallback using netCDF4
        try:
            from netCDF4 import Dataset
            nc = Dataset(fname, 'w')
            nc.createDimension('year', data.shape[0])
            nc.createDimension('day', data.shape[1])
            years_var = nc.createVariable('year', 'i4', ('year',))
            day_var = nc.createVariable('day', 'i4', ('day',))
            anom_var = nc.createVariable('anomalies', 'f4', ('year', 'day'), fill_value=np.nan)
            years_var[:] = years_coord
            day_var[:] = days_coord
            anom_var[:, :] = data
            nc.station_id = station_id
            nc.warm_start = int(warm_start)
            nc.warm_center = int(warm_center)
            nc.warm_end = int(warm_end)
            nc.window_size = int(window_size)
            nc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to write NetCDF file: {e}")

    return fname


def read_netcdf_statistics(netcdf_path, station_id):
    """
    Read a NetCDF file and compute variance and skewness statistics.
    
    Parameters
    ----------
    netcdf_path : str
        Path to the NetCDF file
    station_id : str
        Station ID (for reference)
    
    Returns
    -------
    dict : Contains mean_variance and mean_skewness
    """
    # Load NetCDF data with dual library support
    try:
        import xarray as xr
        ds = xr.open_dataset(netcdf_path)
        anom_data = ds['anomalies'].values
        ds.close()
    except Exception:
        try:
            from netCDF4 import Dataset
            nc = Dataset(netcdf_path, 'r')
            anom_data = nc['anomalies'][:]
            nc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to read NetCDF file {netcdf_path}: {e}")
    
    # Validate data shape and content
    if anom_data.size == 0:
        raise ValueError(f"Empty anomalies array in {netcdf_path}")
    
    if anom_data.ndim != 2:
        raise ValueError(f"Expected 2D array (years, days), got shape {anom_data.shape}")
    
    n_years, n_days = anom_data.shape
    
    # Check that we have sufficient non-NaN data
    total_valid = np.sum(~np.isnan(anom_data))
    if total_valid == 0:
        raise ValueError(f"No valid (non-NaN) data in {netcdf_path}")
    
    # Compute yearly variance (across days, per year)
    # axis=1 means variance computed across columns (days) for each row (year)
    yearly_var = np.nanvar(anom_data, axis=1)
    mean_variance = np.nanmean(yearly_var)
    
    # Compute yearly skewness (across days, per year)
    # Require at least 10 valid data points for reliable skewness estimation
    yearly_skew = np.full(n_years, np.nan)
    for i in range(n_years):
        year_data = anom_data[i, :]
        valid_data = year_data[~np.isnan(year_data)]
        
        if len(valid_data) >= 10:  # Minimum 10 points for reliable skewness
            yearly_skew[i] = skew(valid_data, nan_policy='omit')
    
    mean_skewness = np.nanmean(yearly_skew)
    
    # Validation: ensure we got valid statistics
    if np.isnan(mean_variance) or np.isinf(mean_variance):
        raise ValueError(f"Invalid variance computed for {station_id}: {mean_variance}")
    
    if np.isnan(mean_skewness):
        # This is acceptable if no years had sufficient data, but warn
        import warnings
        warnings.warn(f"Could not compute skewness for {station_id} (insufficient data per year)")
        mean_skewness = 0.0  # Default to symmetric distribution
    
    return {
        'mean_variance': mean_variance,
        'mean_skewness': mean_skewness
    }



#################### "visualization helpers" ########################

# Add new visualization functions
def plot_heatmap(df, station):
    """
    Plot a heatmap of temperature anomalies.
    
    Parameters:
        df : DataFrame (from load_station_full)
        station : str, station ID
    """
    df = _drop_leap_day(df)
    # Calculate daily maxes and pivot
    daily_maxes = df.groupby(['year', 'day_of_year'])['ATMP'].max().reset_index()
    pivot = daily_maxes.pivot(index='day_of_year', columns='year', values='ATMP')
    
    # Compute climatological mean and anomalies
    mean_cycle = pivot.mean(axis=1)
    anomalies = pivot.subtract(mean_cycle, axis=0)
    
    # Create heatmap
    plt.figure(figsize=(12,6))
    plt.imshow(anomalies.T, aspect='auto', cmap='coolwarm',
               extent=[1, 365, anomalies.columns.min(), anomalies.columns.max()],
               origin='lower')
    plt.colorbar(label="Temperature Anomaly [°C]")
    plt.xlabel("Day of Year")
    plt.ylabel("Year")
    plt.title(f"NDBC {station}: Daily Air Temperature Anomalies")
    plt.tight_layout()
    plt.show()

def plot_time_series_anomalies(df, station):
    """
    Plot time series of annual temperature anomalies.
    
    Parameters:
        df : DataFrame (from load_station_full)
        station : str, station ID
    """
    df = _drop_leap_day(df)
    # Calculate daily maxes and pivot
    daily_maxes = df.groupby(['year', 'day_of_year'])['ATMP'].max().reset_index()
    pivot = daily_maxes.pivot(index='day_of_year', columns='year', values='ATMP')
    
    # Compute climatological mean and anomalies
    mean_cycle = pivot.mean(axis=1)
    anomalies = pivot.subtract(mean_cycle, axis=0)
    yearly_anom = anomalies.mean(axis=0)
    
    # Plot time series
    plt.figure(figsize=(10,5))
    plt.plot(yearly_anom.index, yearly_anom.values, marker='o', color='darkorange')
    plt.axhline(0, color='black', lw=1)
    plt.xlabel("Year")
    plt.ylabel("Mean Annual Temperature Anomaly [°C]")
    plt.title(f"NDBC {station}: Annual Temperature Anomalies")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_variance(df, station):
    """
    Plot seasonal variance in temperature.
    
    Parameters:
        df : DataFrame (from load_station_full)
        station : str, station ID
    """
    df = _drop_leap_day(df)
    # Calculate daily statistics
    daily_stats = df.groupby('day_of_year')['ATMP'].agg(['mean', 'std']).reset_index()
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    # Plot mean temperature
    color = 'tab:red'
    ax1.set_xlabel('Day of Year')
    ax1.set_ylabel('Mean Temperature [°C]', color=color)
    ax1.plot(daily_stats['day_of_year'], daily_stats['mean'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot standard deviation on secondary axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Standard Deviation [°C]', color=color)
    ax2.plot(daily_stats['day_of_year'], daily_stats['std'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"NDBC {station}: Seasonal Temperature Variance")
    fig.tight_layout()
    plt.show()

def plot_variance_skew(df, station):
    """
    Plot temperature distribution characteristics including skewness.
    
    Parameters:
        df : DataFrame (from load_station_full)
        station : str, station ID
    """
    df = _drop_leap_day(df)
    # Calculate monthly statistics
    df['month'] = pd.to_datetime(df['date']).dt.month
    monthly_stats = df.groupby('month')['ATMP'].agg(['mean', 'std', lambda x: skew(x.dropna())]).reset_index()
    monthly_stats = monthly_stats.rename(columns={'<lambda_0>': 'skewness'})
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot mean temperature
    ax1.plot(monthly_stats['month'], monthly_stats['mean'], 'o-', color='tab:red')
    ax1.set_ylabel('Mean Temperature [°C]')
    ax1.set_title(f"NDBC {station}: Monthly Temperature Characteristics")
    ax1.grid(True, alpha=0.3)
    
    # Plot standard deviation
    ax2.plot(monthly_stats['month'], monthly_stats['std'], 'o-', color='tab:blue')
    ax2.set_ylabel('Standard Deviation [°C]')
    ax2.grid(True, alpha=0.3)
    
    # Plot skewness
    ax3.plot(monthly_stats['month'], monthly_stats['skewness'], 'o-', color='tab:green')
    ax3.set_ylabel('Skewness')
    ax3.set_xlabel('Month')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

