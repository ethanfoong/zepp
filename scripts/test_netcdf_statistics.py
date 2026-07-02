"""
Quick validation script to verify NetCDF statistics computation is working correctly.
Tests the data flow: write → read → verify
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stat_buoy_helpers import compute_warm_season_anomalies, write_warm_season_netcdf, read_netcdf_statistics


def generate_synthetic_data():
    """Generate synthetic buoy data for testing"""
    np.random.seed(42)
    
    years = range(2000, 2010)  # 10 years
    data = []
    
    for year in years:
        for doy in range(1, 366):  # 365 days
            # Create synthetic temperature with seasonal cycle + noise
            seasonal = 15 + 10 * np.sin(2 * np.pi * (doy - 80) / 365)
            noise = np.random.normal(0, 2)
            temp = seasonal + noise
            
            # Add some positive skewness in summer (extreme heat events)
            if 150 < doy < 250:
                if np.random.random() < 0.1:  # 10% chance of heat spike
                    temp += np.random.exponential(3)
            
            data.append({'year': year, 'day_of_year': doy, 'ATMP': temp})
    
    df = pd.DataFrame(data)
    return df


def test_netcdf_statistics():
    """Test the complete pipeline"""
    print("="*70)
    print("TESTING NETCDF STATISTICS COMPUTATION")
    print("="*70)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic buoy data...")
    df = generate_synthetic_data()
    print(f"    Created {len(df)} observations across {df['year'].nunique()} years")
    
    # Compute warm season anomalies
    print("\n[2] Computing warm season anomalies...")
    anomalies, mean_cycle, (warm_start, warm_center, warm_end) = compute_warm_season_anomalies(df, window_size=50)
    print(f"    Warm season: days {warm_start}-{warm_end} (center: {warm_center})")
    print(f"    Anomalies shape: {anomalies.shape} (days={anomalies.shape[0]}, years={anomalies.shape[1]})")
    
    # Write to NetCDF
    print("\n[3] Writing to NetCDF...")
    test_nc = write_warm_season_netcdf(df, "TEST_STATION", out_dir="nc", window_size=50, target_days=None)
    print(f"    Written: {test_nc}")
    
    # Load NetCDF and verify dimensions
    print("\n[4] Loading NetCDF to verify structure...")
    try:
        import xarray as xr
        ds = xr.open_dataset(test_nc)
        anom_data = ds['anomalies'].values
        ds.close()
        print(f"    Loaded array shape: {anom_data.shape} (years={anom_data.shape[0]}, days={anom_data.shape[1]})")
        print(f"    Non-NaN values: {np.sum(~np.isnan(anom_data))} / {anom_data.size}")
        
        # Verify axis orientation
        print("\n[5] Verifying axis orientation...")
        print(f"    Year 0, first 5 days: {anom_data[0, :5]}")
        print(f"    Year 0, variance across days: {np.nanvar(anom_data[0, :]):.4f}")
        print(f"    Day 0, variance across years: {np.nanvar(anom_data[:, 0]):.4f}")
        
    except Exception as e:
        print(f"    Failed to load with xarray: {e}")
        return False
    
    # Read statistics
    print("\n[6] Computing statistics...")
    stats = read_netcdf_statistics(test_nc, "TEST_STATION")
    print(f"    Mean Variance: {stats['mean_variance']:.4f} °C²")
    print(f"    Mean Skewness: {stats['mean_skewness']:.4f}")
    
    # Manual verification
    print("\n[7] Manual verification of statistics...")
    yearly_var_manual = np.nanvar(anom_data, axis=1)  # variance along days
    yearly_skew_manual = []
    from scipy.stats import skew
    for i in range(anom_data.shape[0]):
        year_data = anom_data[i, :]
        valid_data = year_data[~np.isnan(year_data)]
        if len(valid_data) >= 10:
            yearly_skew_manual.append(skew(valid_data))
    
    mean_var_manual = np.nanmean(yearly_var_manual)
    mean_skew_manual = np.mean(yearly_skew_manual)
    
    print(f"    Manual Mean Variance: {mean_var_manual:.4f} °C²")
    print(f"    Manual Mean Skewness: {mean_skew_manual:.4f}")
    
    # Check if they match
    var_match = np.isclose(stats['mean_variance'], mean_var_manual)
    skew_match = np.isclose(stats['mean_skewness'], mean_skew_manual)
    
    print("\n[8] Validation Results:")
    print(f"    Variance matches: {'✓ PASS' if var_match else '✗ FAIL'}")
    print(f"    Skewness matches: {'✓ PASS' if skew_match else '✗ FAIL'}")
    
    # Cleanup
    if os.path.exists(test_nc):
        os.remove(test_nc)
        print(f"\n[9] Cleaned up test file: {test_nc}")
    
    print("\n" + "="*70)
    if var_match and skew_match:
        print("✓ ALL TESTS PASSED - NetCDF statistics computation is correct!")
    else:
        print("✗ TESTS FAILED - Check implementation")
    print("="*70)
    
    return var_match and skew_match


if __name__ == "__main__":
    success = test_netcdf_statistics()
    sys.exit(0 if success else 1)
