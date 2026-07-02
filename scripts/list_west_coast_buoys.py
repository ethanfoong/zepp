"""Quick diagnostic to show which West Coast buoys are available"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.complete_buoy_pipeline import BUOY_LOCATIONS, is_west_coast_buoy

print("=" * 70)
print("US WEST COAST BUOY STATIONS")
print("=" * 70)
print("\nBounds: 30-50°N, 130-115°W (CA, OR, WA coasts)\n")

west_coast_buoys = {sid: loc for sid, loc in BUOY_LOCATIONS.items() if is_west_coast_buoy(sid)}

# Sort by latitude (north to south)
sorted_buoys = sorted(west_coast_buoys.items(), key=lambda x: x[1][0], reverse=True)

print(f"Found {len(sorted_buoys)} West Coast buoy stations:\n")
print(f"{'Station ID':<12} {'Latitude':<10} {'Longitude':<10} Region")
print("-" * 70)

for sid, (lat, lon) in sorted_buoys[:50]:  # Show first 50
    if lat >= 42:
        region = "WA/OR"
    elif lat >= 38:
        region = "N. California"
    elif lat >= 34:
        region = "C. California"
    else:
        region = "S. California"
    
    print(f"{sid:<12} {lat:>9.3f}  {lon:>9.3f}  {region}")

if len(sorted_buoys) > 50:
    print(f"\n... and {len(sorted_buoys) - 50} more")

print("\n" + "=" * 70)
print(f"Total West Coast stations with location data: {len(sorted_buoys)}")
print("=" * 70)
