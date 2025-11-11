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