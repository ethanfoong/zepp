import os, re, io, gzip, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm, skew
from urllib.parse import urljoin
import json


BASE = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
STATION_TABLE_URL = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"

############ imports and web scraping ################

def parse_station_table(url=STATION_TABLE_URL, output_json=None):
    """
    Parse NDBC station_table.txt to extract all station coordinates.
    
    The table format has columns with lat/lon in degrees + N/S/E/W format.
    Example: "46001 | ... | 56.30 N | 148.02 W | ..."
    
    Parameters:
    -----------
    url : str
        URL to the NDBC station table
    output_json : str, optional
        If provided, saves results to this JSON file
    
    Returns:
    --------
    dict : {station_id: (lat, lon), ...}
    """
    try:
        print(f"Downloading station table from {url}...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        
        station_locations = {}
        
        # Regex patterns for coordinates
        # Matches formats like: "56.30 N", "148.02 W", "12.345 S", "98.765 E"
        lat_pattern = re.compile(r'(\d+\.\d+)\s*([NS])')
        lon_pattern = re.compile(r'(\d+\.\d+)\s*([EW])')
        
        lines = r.text.split('\n')
        
        for line in lines:
            # Skip empty lines and headers
            if not line.strip() or line.startswith('#') or line.startswith('Station'):
                continue
            
            # Split by pipe delimiter (station table uses | as separator)
            parts = [p.strip() for p in line.split('|')]
            
            if len(parts) < 4:
                # Try space-delimited format (backup)
                parts = line.split()
            
            if len(parts) >= 4:
                # First column is typically station ID
                station_id = parts[0].strip().lower()
                
                # Find latitude and longitude in the line
                lat_match = lat_pattern.search(line)
                lon_match = lon_pattern.search(line)
                
                if lat_match and lon_match:
                    # Extract latitude
                    lat_value = float(lat_match.group(1))
                    lat_dir = lat_match.group(2)
                    lat = lat_value if lat_dir == 'N' else -lat_value
                    
                    # Extract longitude
                    lon_value = float(lon_match.group(1))
                    lon_dir = lon_match.group(2)
                    lon = -lon_value if lon_dir == 'W' else lon_value
                    
                    station_locations[station_id] = (lat, lon)
        
        print(f"✓ Parsed {len(station_locations)} station locations")
        
        # Save to JSON if requested
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(station_locations, f, indent=2)
            print(f"✓ Saved to {output_json}")
        
        return station_locations
        
    except Exception as e:
        print(f"✗ Failed to parse station table: {e}")
        return {}


def load_station_locations(json_file):
    """Load station locations from JSON cache file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Convert lists back to tuples (JSON stores tuples as lists)
            return {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    except:
        return {}

def list_station_files(station_id: str, base_url: str = BASE, session: requests.Session | None = None):
    """noaa scraper that returns .gz files for a station (case-insensitive)."""
    station_id = station_id.lower()
    sess = session or requests
    r = sess.get(base_url, timeout=120)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        h = a["href"]
        h_lower = h.lower()
        if h_lower.startswith(f"{station_id}h") and h_lower.endswith(".txt.gz"):
            urls.append(urljoin(base_url, h))
    return sorted(urls)


def list_all_station_ids(base_url: str = BASE, session: requests.Session | None = None):
    """Return all station IDs listed in the NDBC stdmet directory."""
    sess = session or requests
    r = sess.get(base_url, timeout=120)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    station_ids = set()
    pattern = re.compile(r"^([a-z0-9]{1,8})h(\d{4})\.txt\.gz$", re.IGNORECASE)
    for a in soup.find_all("a", href=True):
        h = a["href"].strip()
        m = pattern.match(h)
        if m:
            station_ids.add(m.group(1).lower())

    return sorted(station_ids)

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
    parsing NOAA NDBC stdmet files into DataFrame with date, year, day_of_year, ATMP.
    handles both situations of  'YY' and 'YYYY' header formats and variable whitespace.
    """
    import io, gzip, re
    from datetime import datetime
    import pandas as pd

    m = re.search(r"(\d{4})\.txt\.gz", url)
    if not m:
        print(f" failed to extract year from {url}")
        return None
    year = int(m.group(1))
    fname = fetch_file(url)

    try:
        with gzip.open(fname, "rt", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f" failed to read {year}: {e}")
        return None

    header = None
    for ln in lines:
        if (("YY" in ln) or ("YYYY" in ln)) and ("MM" in ln) and ("ATMP" in ln):
            header = ln.strip().replace("#", "").split()
            break

    if not header:
        return None

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
        print(f" failed to parse {year}: {e}")
        return None

    if "YY" not in df.columns or "MM" not in df.columns or "DD" not in df.columns or "ATMP" not in df.columns:
        print(f"⚠️ Skipping {year}: missing key columns.")
        return None

    # --- numeric conversions ---
    df["YY"] = pd.to_numeric(df["YY"], errors="coerce")
    df["MM"] = pd.to_numeric(df["MM"], errors="coerce")
    df["DD"] = pd.to_numeric(df["DD"], errors="coerce")
    df["ATMP"] = pd.to_numeric(df["ATMP"], errors="coerce")

    #rows that do not have these must be dropped
    df = df.dropna(subset=["YY", "MM", "DD", "ATMP"])

    # year logic
    if df["YY"].max() < 100:
        first_year = int(df["YY"].iloc[0])
        century_base = 1900 if first_year > 50 else 2000
        df["year"] = century_base + df["YY"].astype(int)
    else:
        df["year"] = df["YY"].astype(int)

    try:
        df["date"] = pd.to_datetime(
            dict(year=df["year"], month=df["MM"].astype(int), day=df["DD"].astype(int)),
            errors="coerce"
        )
        df["day_of_year"] = df["date"].dt.dayofyear
    except Exception as e:
        print(f" failed to build dates for {year}: {e}")
        return None

    df = df.dropna(subset=["date", "ATMP"])
    df = df[df["day_of_year"] <= 365]

    print(f" parsed {year}: {len(df)} rows")
    return df[["date", "year", "day_of_year", "ATMP"]]


def load_station(station, workers=6, cache_dir="cache"):
    """
    Load and gap-fill NOAA NDBC stdmet data for a given station.

    Returns:
        df_filled   - gap-filled DataFrame for analysis
        completeness - DataFrame summarizing % valid ATMP data per year
    """
    urls = list_station_files(station)
    print(f" Found {len(urls)} files for station {station}")
    if not urls:
        return None, None

    dfs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(read_stdmet, urls):
            if res is not None and not res.empty:
                dfs.append(res)

    if not dfs:
        print(" No valid data found.")
        return None, None

    df = pd.concat(dfs, ignore_index=True)
    # drop Feb 29 (day 60) only for leap years (divisible by 4, except centuries not divisible by 400)
    is_leap_year = (df['year'] % 4 == 0) & ((df['year'] % 100 != 0) | (df['year'] % 400 == 0))
    df = df[~((df['day_of_year'] == 60) & is_leap_year)]
    #df_raw = df.copy(), can be used for more precise GEV PDF distributions

    completeness = (
        df.groupby("year")["ATMP"]
        .apply(lambda s: 100 * s.notna().sum() / len(s))
        .reset_index(name="valid_percent")
    )

    # interpolate within each year
    df_filled = (
        df.groupby("year", group_keys=False)
          .apply(lambda g: g.assign(ATMP=g["ATMP"].interpolate(limit_direction="both")))
    )
    df_filled["ATMP"] = df_filled["ATMP"].ffill().bfill()

    all_years = np.arange(df["year"].min(), df["year"].max() + 1)
    df_filled["year"] = df_filled["year"].astype(int)

    df_filled = df_filled.drop_duplicates(subset=["year", "day_of_year"])

    year_day_index = pd.MultiIndex.from_product(
        [all_years, np.arange(1, 366)],
        names=["year", "day_of_year"]
    )
    df_filled = df_filled.set_index(["year", "day_of_year"]).reindex(year_day_index).reset_index()

    print(#f" {df_raw['year'].nunique()} raw years; "
          f"continuous coverage from {all_years.min()}–{all_years.max()} "
          f"({len(all_years)} total years)")

    return df_filled, completeness