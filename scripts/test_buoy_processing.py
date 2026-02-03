"""
TEST VERSION: Process a few sample buoys to verify the pipeline
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.process_all_buoys import (
    discover_all_buoys,
    filter_buoys_by_years,
    process_single_buoy,
    save_results
)
import pandas as pd

def test_pipeline():
    """Test the processing pipeline with a small sample of buoys."""
    
    print("="*60)
    print("TESTING BUOY PROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Discover buoys
    print("\n🔍 Discovering buoys...")
    all_buoys = discover_all_buoys()
    print(f"   Found {len(all_buoys)} total buoys")
    
    # Step 2: Filter for buoys with 10+ years (test with small sample)
    print("\n📊 Checking data availability (testing 20 buoys)...")
    test_buoys = all_buoys[:20]  # Just test first 20
    valid_buoys = filter_buoys_by_years(test_buoys, min_years=10, max_workers=5)
    
    if not valid_buoys:
        print("❌ No valid buoys found in test sample")
        return
    
    print(f"\n✅ Found {len(valid_buoys)} buoys with 10+ years:")
    for buoy_id, years in sorted(valid_buoys.items(), key=lambda x: -x[1])[:10]:
        print(f"   {buoy_id}: {years} years")
    
    # Step 3: Process just the top 3 buoys with most data
    print("\n🚀 Processing top 3 buoys...")
    top_buoys = sorted(valid_buoys.items(), key=lambda x: -x[1])[:3]
    
    results = []
    for buoy_id, num_years in top_buoys:
        print(f"\n--- Processing {buoy_id} ({num_years} years) ---")
        result = process_single_buoy(buoy_id, window_size=100)
        if result:
            results.append(result)
    
    # Step 4: Display results
    if results:
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        df = pd.DataFrame(results)
        print("\nProcessed buoys:")
        print(df[['station_id', 'num_years', 'latitude', 'longitude', 
                  'mean_variance', 'mean_skewness']])
        
        # Save test results
        save_results(results, output_file="buoy_statistics_TEST.pkl")
        
        print("\n✅ Test complete! Pipeline is working correctly.")
        print("   Ready to run full analysis with process_all_buoys.py")
    else:
        print("\n❌ No results generated. Check error messages above.")

if __name__ == "__main__":
    test_pipeline()
