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

############ imports and web scraping ################

def list_station_files(station_id: str):
    "noaa scraper that returns .gz files"
    r = requests.get(BASE)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.startswith(f"{station_id}h") and h.endswith(".txt.gz"):
            urls.append(urljoin(BASE, h))
    return sorted(urls)

def fetch_file(url, cache_dir="cache"):
    "downloaded once and can be reused from cache"
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(fname):
        r = requests.get(url)
        r.raise_for_status()
        with open(fname, "wb") as f:
            f.write(r.content)
    return fname

def read_stdmet_max(url):
    "atmp data from noaa that returns (max atmp, year) if not returns none"
    m = re.search(r"(\d{4})\.txt\.gz", url)
    if not m:
        return None
    year = int(m.group(1))

    fname = fetch_file(url)
    with gzip.open(fname, "rt", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header = None
    for ln in reversed(lines):
        if ln.startswith("#"):
            toks = ln.lstrip("#").strip().split()
            if "MM" in toks and "DD" in toks:
                header = toks
                break
    if not header or "ATMP" not in [h.upper() for h in header]:
        return None

    colnames = [c.lstrip("#").upper() for c in header]

    seen, unique_cols = {}, []
    for c in colnames:
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        unique_cols.append(c)

    df = pd.read_csv(
        io.StringIO("".join(lines)),
        sep=r"\s+",
        comment="#",
        header=None,
        names=unique_cols,
        usecols=lambda c: c == "ATMP",
        na_values=["MM", "MM.MM", "99.0", "999.0", "9999.0"],
        engine="python",
    )

    if df["ATMP"].dropna().empty:
        return None

    return year, float(df["ATMP"].max())

def read_stdmet_min(url):
    "atmp data from noaa that returns (min atmp, year) if not returns none"
    m = re.search(r"(\d{4})\.txt\.gz", url)
    if not m:
        return None
    year = int(m.group(1))

    fname = fetch_file(url)
    with gzip.open(fname, "rt", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header = None
    for ln in reversed(lines):
        if ln.startswith("#"):
            toks = ln.lstrip("#").strip().split()
            if "MM" in toks and "DD" in toks:
                header = toks
                break
    if not header or "ATMP" not in [h.upper() for h in header]:
        return None

    colnames = [c.lstrip("#").upper() for c in header]

    seen, unique_cols = {}, []
    for c in colnames:
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        unique_cols.append(c)

    df = pd.read_csv(
        io.StringIO("".join(lines)),
        sep=r"\s+",
        comment="#",
        header=None,
        names=unique_cols,
        usecols=lambda c: c == "ATMP",
        na_values=["MM", "MM.MM", "99.0", "999.0", "9999.0"],
        engine="python",
    )

    if df["ATMP"].dropna().empty:
        return None

    return year, float(df["ATMP"].min())

def collect_station_max(station, workers=6):
    "process all files for a station in parallel, return series of annual maxima."
    urls = list_station_files(station)
    print(f"Found {len(urls)} files for station {station}")
    annual = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(read_stdmet_max, urls):
            if res is None:
                continue
            yr, val = res
            annual[yr] = val
    return pd.Series(annual).sort_index()

def collect_station_min(station, workers=6):
    "process all files for a station in parallel, return series of annual maxima."
    urls = list_station_files(station)
    print(f"Found {len(urls)} files for station {station}")
    annual = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(read_stdmet_min, urls):
            if res is None:
                continue
            yr, val = res
            annual[yr] = val
    return pd.Series(annual).sort_index()

def read_stdmet(url):
    """
    Parse NOAA NDBC stdmet files into DataFrame with date, year, day_of_year, ATMP.
    Handles both 'YY' and 'YYYY' header formats and variable whitespace.
    """
    import io, gzip, re
    from datetime import datetime
    import pandas as pd

    m = re.search(r"(\d{4})\.txt\.gz", url)
    if not m:
        print(f"⚠️ Could not extract year from {url}")
        return None
    year = int(m.group(1))
    fname = fetch_file(url)

    # --- Read file ---
    try:
        with gzip.open(fname, "rt", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠️ Could not read {year}: {e}")
        return None

    # --- Detect header line ---
    header = None
    for ln in lines:
        if (("YY" in ln) or ("YYYY" in ln)) and ("MM" in ln) and ("ATMP" in ln):
            header = ln.strip().replace("#", "").split()
            break

    if not header:
        print(f"⚠️ Skipping {year}: no valid header found.")
        return None

    # --- Normalize header ---
    header = [h.replace("YYYY", "YY") for h in header]
    seen = {}
    unique_cols = []
    for c in header:
        if c in seen:
            seen[c] += 1
            unique_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            unique_cols.append(c)

    # --- Parse data section ---
    try:
        df = pd.read_csv(
            io.StringIO("".join(lines)),
            sep=r"\s+",
            comment="#",
            names=unique_cols,
            header=None,
            engine="python",
            na_values=["MM", "MM.MM", "99.0", "99.00", "999.0", "9999.0"]
        )
    except Exception as e:
        print(f"⚠️ Failed to parse {year}: {e}")
        return None

    # --- Sanity check ---
    if "YY" not in df.columns or "MM" not in df.columns or "DD" not in df.columns or "ATMP" not in df.columns:
        print(f"⚠️ Skipping {year}: missing key columns.")
        return None

    # --- Force numeric conversions ---
    df["YY"] = pd.to_numeric(df["YY"], errors="coerce")
    df["MM"] = pd.to_numeric(df["MM"], errors="coerce")
    df["DD"] = pd.to_numeric(df["DD"], errors="coerce")
    df["ATMP"] = pd.to_numeric(df["ATMP"], errors="coerce")

    # --- Drop rows missing key info ---
    df = df.dropna(subset=["YY", "MM", "DD", "ATMP"])

    # --- Handle year logic (2-digit or 4-digit) ---
    if df["YY"].max() < 100:
        first_year = int(df["YY"].iloc[0])
        century_base = 1900 if first_year > 50 else 2000
        df["year"] = century_base + df["YY"].astype(int)
    else:
        df["year"] = df["YY"].astype(int)

    # --- Construct datetime and day_of_year ---
    try:
        df["date"] = pd.to_datetime(
            dict(year=df["year"], month=df["MM"].astype(int), day=df["DD"].astype(int)),
            errors="coerce"
        )
        df["day_of_year"] = df["date"].dt.dayofyear
    except Exception as e:
        print(f"⚠️ Failed to build dates for {year}: {e}")
        return None

    df = df.dropna(subset=["date", "ATMP"])
    df = df[df["day_of_year"] <= 365]

    if df.empty:
        print(f"⚠️ No valid ATMP data found for {year}.")
        return None

    print(f"✅ Parsed {year}: {len(df)} valid rows.")
    return df[["date", "year", "day_of_year", "ATMP"]]


def load_station(station, workers=6, cache_dir="cache"):
    """
    Load and merge full historical ATMP data for an NDBC station.
    Performs intelligent gap-filling (intra-year interpolation),
    and computes data completeness diagnostics.

    Returns:
        df_raw      - original data with NaNs intact
        df_filled   - gap-filled version for visualization
        completeness - DataFrame summarizing % valid ATMP data per year
    """
    urls = list_station_files(station)
    print(f"📡 Found {len(urls)} files for station {station}")
    if not urls:
        return None, None, None

    dfs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(read_stdmet, urls):
            if res is not None and not res.empty:
                dfs.append(res)

    if not dfs:
        print("❌ No valid data found.")
        return None, None, None

    # Merge all years together
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["day_of_year"] <= 365]  # Drop leap days
    #df_raw = df.copy(), can be used for more precise GEV PDF distributions

    # ---- Compute Data Completeness Diagnostics ----
    completeness = (
        df.groupby("year")["ATMP"]
        .apply(lambda s: 100 * s.notna().sum() / len(s))
        .reset_index(name="valid_percent")
    )

    # ---- Intelligent Gap-Filling ----
    # Interpolate within each year
    df_filled = (
        df.groupby("year", group_keys=False)
          .apply(lambda g: g.assign(ATMP=g["ATMP"].interpolate(limit_direction="both")))
    )

    # Fill remaining edge NaNs (start/end of year)
    df_filled["ATMP"] = df_filled["ATMP"].fillna(method="ffill").fillna(method="bfill")

    # ---- Reindex to include all years ----
    all_years = np.arange(df["year"].min(), df["year"].max() + 1)
    df_filled["year"] = df_filled["year"].astype(int)

    # Ensure unique multi-index before reindexing
    df_filled = df_filled.drop_duplicates(subset=["year", "day_of_year"])

    # Build a complete index of all (year, day) pairs
    year_day_index = pd.MultiIndex.from_product(
        [all_years, np.arange(1, 366)],
        names=["year", "day_of_year"]
    )
    df_filled = df_filled.set_index(["year", "day_of_year"]).reindex(year_day_index).reset_index()

    print(#f"✅ Loaded {df_raw['year'].nunique()} raw years; "
          f"continuous coverage from {all_years.min()}–{all_years.max()} "
          f"({len(all_years)} total years)")

    return df_filled, completeness

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
    """Plot a time series of mean warm-season temperature anomalies per year."""
    anomalies, mean_cycle, (warm_start, warm_center, warm_end) = compute_warm_season_anomalies(df, window_size)
    yearly_anom = anomalies.mean(axis=0)

    plt.figure(figsize=(10,5))
    plt.plot(yearly_anom.index, yearly_anom.values, marker='o', color='darkorange')
    plt.axhline(0, color='black', lw=1)
    plt.xlabel("Year")
    plt.ylabel("Mean Warm Season Temperature Anomaly [°C]")
    plt.title(f"NDBC {station}: Warm Season Temperature Anomalies\n(Days {warm_start}-{warm_end}, Peak: {warm_center})")
    plt.grid(True, alpha=0.3)
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

