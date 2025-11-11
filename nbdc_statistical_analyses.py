import os, re, io, gzip, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm, skew

def get_warm_season_data(df, window_size=50):
    """
    Identify and extract warm season data based on climatological maximum.
    
    Parameters:
        df : DataFrame from load_station_full
            Must contain columns ['year', 'day_of_year', 'ATMP']
        window_size : int
            Number of days on each side of peak temperature to include
            
    Returns:
        tuple: (warm_season_df, warm_period)
            - warm_season_df: DataFrame containing only warm season data
            - warm_period: tuple of (start_day, center_day, end_day)
    """
    # Calculate daily maxes and find climatological maximum
    daily_maxes = df.groupby('day_of_year')['ATMP'].mean()
    warm_center = int(daily_maxes.idxmax())
    
    # Define warm season window
    warm_start = max(warm_center - window_size, 1)
    warm_end = min(warm_center + window_size, 365)
    
    # Filter data for warm season only
    warm_season_mask = (df['day_of_year'] >= warm_start) & (df['day_of_year'] <= warm_end)
    warm_season_df = df[warm_season_mask].copy()
    
    # Store warm season period information
    warm_period = (warm_start, warm_center, warm_end)
    
    return warm_season_df, warm_period

def plot_warm_season_heatmap(df, station, window_size=50):
    """
    Plot a heatmap of temperature anomalies during the warm season.
    
    Parameters:
        df : DataFrame (from load_station_full)
        station : str, station ID
        window_size : int, days on each side of peak temperature
    """
    # Get warm season data
    warm_season_df, (warm_start, warm_center, warm_end) = get_warm_season_data(df, window_size)
    
    # Calculate daily maxes and pivot
    daily_maxes = warm_season_df.groupby(['year', 'day_of_year'])['ATMP'].max().reset_index()
    pivot = daily_maxes.pivot(index='day_of_year', columns='year', values='ATMP')
    
    # Compute climatological mean and anomalies
    mean_cycle = pivot.mean(axis=1)
    anomalies = pivot.subtract(mean_cycle, axis=0)
    
    # Create heatmap
    plt.figure(figsize=(12,6))
    plt.imshow(anomalies.T, aspect='auto', cmap='coolwarm',
               extent=[warm_start, warm_end, anomalies.columns.min(), anomalies.columns.max()],
               origin='lower')
    plt.colorbar(label="Temperature Anomaly [°C]")
    
    # Add warm season markers
    plt.axvline(warm_center, color='black', linestyle='--', alpha=0.5, label='Peak Temperature Day')
    
    plt.xlabel("Day of Year")
    plt.ylabel("Year")
    plt.title(f"NDBC {station}: Warm Season Temperature Anomalies\n(Days {warm_start}-{warm_end}, Peak: {warm_center})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_warm_season_anomalies(df, station, window_size=50):
    """
    Plot time series of warm season temperature anomalies.
    
    Parameters:
        df : DataFrame (from load_station_full)
        station : str, station ID
        window_size : int, days on each side of peak temperature
    """
    # Get warm season data
    warm_season_df, (warm_start, warm_center, warm_end) = get_warm_season_data(df, window_size)
    
    # Calculate daily maxes and pivot
    daily_maxes = warm_season_df.groupby(['year', 'day_of_year'])['ATMP'].max().reset_index()
    pivot = daily_maxes.pivot(index='day_of_year', columns='year', values='ATMP')
    
    # Compute anomalies
    mean_cycle = pivot.mean(axis=1)
    anomalies = pivot.subtract(mean_cycle, axis=0)
    yearly_anom = anomalies.mean(axis=0)
    
    # Plot time series
    plt.figure(figsize=(10,5))
    plt.plot(yearly_anom.index, yearly_anom.values, marker='o', color='darkorange')
    plt.axhline(0, color='black', lw=1)
    plt.xlabel("Year")
    plt.ylabel("Mean Warm Season Temperature Anomaly [°C]")
    plt.title(f"NDBC {station}: Warm Season Temperature Anomalies\n(Days {warm_start}-{warm_end}, Peak: {warm_center})")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_warm_season_variance(df, station, window_size=50):
    """
    Plot variance analysis specifically for the warm season.
    
    Parameters:
        df : DataFrame (from load_station_full)
        station : str, station ID
        window_size : int, days on each side of peak temperature
    """
    # Get warm season data
    warm_season_df, (warm_start, warm_center, warm_end) = get_warm_season_data(df, window_size)
    
    # Calculate daily statistics within warm season
    daily_stats = warm_season_df.groupby('day_of_year')['ATMP'].agg(['mean', 'std', lambda x: skew(x.dropna())]).reset_index()
    daily_stats = daily_stats.rename(columns={'<lambda_0>': 'skewness'})
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot mean temperature
    ax1.plot(daily_stats['day_of_year'], daily_stats['mean'], color='tab:red')
    ax1.axvline(warm_center, color='black', linestyle='--', alpha=0.5, label='Peak Temperature Day')
    ax1.set_ylabel('Mean Temperature [°C]')
    ax1.set_title(f"NDBC {station}: Warm Season Temperature Characteristics\n(Days {warm_start}-{warm_end})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot standard deviation
    ax2.plot(daily_stats['day_of_year'], daily_stats['std'], color='tab:blue')
    ax2.axvline(warm_center, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Standard Deviation [°C]')
    ax2.grid(True, alpha=0.3)
    
    # Plot skewness
    ax3.plot(daily_stats['day_of_year'], daily_stats['skewness'], color='tab:green')
    ax3.axvline(warm_center, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Skewness')
    ax3.set_xlabel('Day of Year')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()