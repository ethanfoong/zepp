import argparse
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from data_buoy_helpers import load_station_locations, parse_station_table

BASE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/"
LOCATION_CACHE_FILE = os.path.join("cache", "buoy_locations.json")

WEST_BOUNDS = {
    "lat_min": 30.0,
    "lat_max": 50.0,
    "lon_min": -130.0,
    "lon_max": -115.0,
}

EAST_BOUNDS = {
    "lat_min": 25.0,
    "lat_max": 45.0,
    "lon_min": -85.0,
    "lon_max": -65.0,
}


def classify_coast(lat, lon):
    if (
        WEST_BOUNDS["lat_min"] <= lat <= WEST_BOUNDS["lat_max"]
        and WEST_BOUNDS["lon_min"] <= lon <= WEST_BOUNDS["lon_max"]
    ):
        return "west"

    if (
        EAST_BOUNDS["lat_min"] <= lat <= EAST_BOUNDS["lat_max"]
        and EAST_BOUNDS["lon_min"] <= lon <= EAST_BOUNDS["lon_max"]
    ):
        return "east"

    return "other"


def get_station_locations(cache_path):
    if os.path.exists(cache_path):
        locations = load_station_locations(cache_path)
        if locations:
            return {str(k).upper(): tuple(v) for k, v in locations.items()}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    locations = parse_station_table(output_json=cache_path)
    return {str(k).upper(): tuple(v) for k, v in locations.items()}


def fetch_station_year_counts(base_url):
    response = requests.get(base_url, timeout=120)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Matches names like 41002h1973.txt.gz
    pattern = re.compile(r"^([a-z0-9]{1,8})h(\d{4})\.txt\.gz$", re.IGNORECASE)

    station_years = {}
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        match = pattern.match(href)
        if not match:
            continue

        station_id = match.group(1).upper()
        year = int(match.group(2))
        station_years.setdefault(station_id, set()).add(year)

    return {station_id: len(years) for station_id, years in station_years.items()}


def build_summary(min_years, base_url, cache_path):
    year_counts = fetch_station_year_counts(base_url)
    locations = get_station_locations(cache_path)

    rows = []
    for station_id, n_years in year_counts.items():
        if n_years < min_years:
            continue

        lat_lon = locations.get(station_id)
        if lat_lon is None:
            lat = None
            lon = None
            coast = "other"
        else:
            lat, lon = float(lat_lon[0]), float(lat_lon[1])
            coast = classify_coast(lat, lon)

        rows.append(
            {
                "station_id": station_id,
                "years_available": n_years,
                "coast": coast,
                "latitude": lat,
                "longitude": lon,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(["coast", "years_available", "station_id"], ascending=[True, False, True]).reset_index(drop=True)


def print_summary(df, min_years):
    total = int(len(df))
    west = int((df["coast"] == "west").sum()) if total else 0
    east = int((df["coast"] == "east").sum()) if total else 0
    other = int((df["coast"] == "other").sum()) if total else 0

    print("=== NDBC Buoy Eligibility Summary ===")
    print(f"Threshold: {min_years}+ years")
    print(f"Eligible stations: {total}")
    print(f"West Coast: {west}")
    print(f"East Coast: {east}")
    print(f"Other/Unclassified: {other}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize NDBC buoys with min-year availability and classify coast (east/west/other)."
    )
    parser.add_argument("--min-years", type=int, default=10, help="Minimum years available per station.")
    parser.add_argument("--base-url", default=BASE_URL, help="NDBC stdmet historical directory URL.")
    parser.add_argument("--location-cache", default=LOCATION_CACHE_FILE, help="Path to cached buoy location JSON.")
    parser.add_argument(
        "--output",
        default=os.path.join("cache", "buoy_eligibility_10y_by_coast.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = build_summary(
        min_years=args.min_years,
        base_url=args.base_url,
        cache_path=args.location_cache,
    )

    if df.empty:
        print("No stations matched the threshold.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print_summary(df, args.min_years)
    print(f"Saved CSV: {args.output}")


if __name__ == "__main__":
    main()
