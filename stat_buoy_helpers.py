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

#################### seasonal cycle helpers ########################

def plot_dailymax_seasonal_cycle(df, station, warm_season_window=50):
    """
    Plot seasonal temperature cycle using daily maximum ATMP.
    
    Parameters:
        df : DataFrame (from load_station_full)
            Must contain columns ['year', 'day_of_year', 'ATMP']
        station : str
            NDBC station ID
        warm_season_window : int, optional
            Days on each side of the climatological max to define 'warm season'
    """
    # --- Compute daily maxima per year ---
    daily_max = df.groupby(["year", "day_of_year"])["ATMP"].max().reset_index()

    # Pivot to shape (365 x n_years)
    pivot = daily_max.pivot(index="day_of_year", columns="year", values="ATMP")

    # --- Multi-year climatology (red line) ---
    mean_cycle = pivot.mean(axis=1)   # daily mean of yearly maxes
    std_cycle = pivot.std(axis=1)     # variability envelope

    # --- Determine dynamic warm-season window ---
    warm_center = int(mean_cycle.idxmax())   # day-of-year of climatological max
    warm_start = max(warm_center - warm_season_window, 1)
    warm_end = min(warm_center + warm_season_window, 365)

    print(f" Warm season for {station}: Days {warm_start}–{warm_end} (centered at {warm_center})")

    # --- Plot setup ---
    plt.figure(figsize=(10, 6))

    # Plot each year's daily max pattern (thin black lines)
    for y in pivot.columns:
        plt.plot(pivot.index, pivot[y], color="black", alpha=0.25, linewidth=0.6)

    # Plot climatological daily max mean (red line)
    plt.plot(mean_cycle.index, mean_cycle, color="red", linewidth=2.5, label="Mean Seasonal Cycle")

    # --- Highlight warm season dynamically ---
    plt.axvspan(warm_start, warm_end, color="lightblue", alpha=0.15, label="Warm Season")
    plt.axvline(warm_start, color="blue", linestyle="--", alpha=0.6)
    plt.axvline(warm_end, color="blue", linestyle="--", alpha=0.6)
    plt.text(warm_center, plt.ylim()[0] + 1, "Warm Season", color="blue",
             ha="center", va="bottom", fontsize=11)

    # --- Aesthetics ---
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

    Parameters
    ----------
    df : DataFrame
        Must contain columns ['year', 'day_of_year', 'ATMP']
    window_size : int
        Number of days on each side of the climatological max to include

    Returns
    -------
    warm_season_df : DataFrame
        Subset of the input dataframe with only warm season days
    warm_period : tuple
        (warm_start, warm_center, warm_end)
    """
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

    Returns (anomalies, mean_cycle, warm_period)
    - anomalies: DataFrame indexed by day_of_year, columns=years
    - mean_cycle: Series daily climatology for the warm season
    - warm_period: (start, center, end)
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
    
    # Calculate skewness
    from scipy.stats import skew
    skewness = skew(yearly_anom.dropna())

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Time series
    ax1.plot(yearly_anom.index, yearly_anom.values, marker='o', color='darkorange', linewidth=2)
    ax1.axhline(0, color='black', lw=1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean Warm Season Temperature Anomaly [°C]")
    ax1.set_title(f"NDBC {station}: Warm Season Temperature Anomalies\n(Days {warm_start}-{warm_end}, Peak: {warm_center})")
    ax1.grid(True, alpha=0.3)
    
    # Right plot: KDE distribution showing skewness
    ax2.hist(yearly_anom.dropna(), bins=15, density=True, alpha=0.6, color='lightcoral', edgecolor='black', label='Histogram')
    
    # KDE plot
    from scipy.stats import gaussian_kde
    data = yearly_anom.dropna()
    if len(data) > 1:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
        ax2.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
        
        # Mark mean and median to show skewness
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
    Helper: compute yearly variance and skew of daily-max ATMP following the
    template used in the notebook: pivot daily maxima to (day_of_year x year)
    then compute variance and skew per year.

    Returns: (all_years, yearly_var, yearly_skew)
    """
    # Pivot daily maxima
    daily_maxes = df.groupby(['year', 'day_of_year'])['ATMP'].max().reset_index()
    pivot = daily_maxes.pivot(index='day_of_year', columns='year', values='ATMP')

    # Years present
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

    Layout: 2 rows x N columns, where top row shows yearly variance and bottom row
    shows yearly skewness for each station. Columns correspond to stations.

    Parameters
    ----------
    station_data : dict
        Mapping station_id -> DataFrame (as returned by load_station_full)
    stations : list or None
        List of station IDs (keys of station_data) in the order to plot. If None,
        plot all keys in station_data.
    figsize_per_col : float
        Width in inches per column (controls overall figure width)
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



#################### "visualization helpers" ########################

# Add new visualization functions
def plot_heatmap(df, station):
    """
    Plot a heatmap of temperature anomalies.
    
    Parameters:
        df : DataFrame (from load_station_full)
        station : str, station ID
    """
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

