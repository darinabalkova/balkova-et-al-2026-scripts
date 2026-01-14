#!/usr/bin/env python3
"""
3. ERA5-Land vs Station Data Agreement Analysis

This script performs a comparison between ERA5-Land and Station data:
1. Overall agreement metrics (bias, RMSE, correlation) for T, RH, P
2. Rainy day frequency comparison
3. Analysis for full period and growing season (April-September)
4. Precipitation filtering threshold optimization

Input: CSV file with columns: Date, Location, T (°C), RH (%), Rain (mm), DATA
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input file path
# UPDATE: Replace with your actual input file path
INPUT_FILE = r"UPDATE_WITH_YOUR_INPUT_FILE_PATH"

# Output directory - will be set to input file directory in main() function
OUTPUT_DIR = None

# Date range
START_YEAR = 2008
END_YEAR = 2025

# Growing season months (April-October per latest req)
GROWING_SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]

# Precipitation filtering thresholds to test (mm)
PRECIP_THRESHOLDS = np.arange(1.0, 5.5, 0.5)  # 1.0, 1.5, 2.0, ..., 5.0

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

def load_data(file_path):
    """
    Load and prepare the merged CSV file.
    
    Expected columns:
    - Date (dd/mm/yyyy) or Date
    - Location
    - T (°C) or T
    - RH (%) or RH
    - Rain (mm) or P or Precip
    - DATA (source: 'ERA5-Land' or 'Station')
    """
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    try:
        df = pd.read_csv(file_path)
        print(f"  Initial shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Standardize date column - check for exact match or contains "Date"
        date_col = None
        for col in df.columns:
            if col == 'Date (dd/mm/yyyy)' or 'Date (dd/mm/yyyy)' in col or (col.lower().startswith('date') and 'dd/mm/yyyy' in col):
                date_col = col
                break
        
        # If not found, try simpler date column names
        if date_col is None:
            for col in df.columns:
                if col.lower() in ['date', 'date (dd/mm/yyyy)']:
                    date_col = col
                    break
        
        if date_col is None:
            raise ValueError(f"No date column found. Available columns: {df.columns.tolist()}")
        
        print(f"  Using date column: '{date_col}'")
        
        # Parse dates - try format first, then fallback
        df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        if df['Date'].isna().all() or df['Date'].isna().sum() > len(df) * 0.5:
            # Try alternative format or without format specification
            print(f"  Warning: Date parsing with format '%d/%m/%Y' failed, trying alternative...")
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df.dropna(subset=['Date'])
        
        # Filter date range
        df = df[(df['Date'].dt.year >= START_YEAR) & (df['Date'].dt.year <= END_YEAR)]
        
        # Standardize column names - be more flexible with matching
        rename_map = {}
        
        # Find temperature column
        temp_cols = [col for col in df.columns if 'T' in col and ('°C' in col or 'C' in col) or col.strip() == 'T']
        if temp_cols:
            rename_map[temp_cols[0]] = 'T'
        
        # Find RH column
        rh_cols = [col for col in df.columns if 'RH' in col or 'relative' in col.lower() or col.strip() == 'RH']
        if rh_cols:
            rename_map[rh_cols[0]] = 'RH'
        
        # Find precipitation column - check multiple possibilities
        precip_cols = [col for col in df.columns if 'Rain' in col or 'Precip' in col or 'P' in col or 'precip' in col.lower()]
        # Prioritize 'Rain (mm)' if it exists
        if 'Rain (mm)' in df.columns:
            rename_map['Rain (mm)'] = 'P'
        elif precip_cols:
            rename_map[precip_cols[0]] = 'P'
        
        print(f"  Column rename mapping: {rename_map}")
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        print(f"  Columns after renaming: {df.columns.tolist()}")
        
        # Check required columns
        required_cols = ['Date', 'Location', 'T', 'RH', 'P', 'DATA']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  WARNING: Missing required columns: {missing_cols}")
            print(f"  Available columns: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check if P column has data
        if 'P' in df.columns:
            print(f"  P column (precipitation) statistics:")
            print(f"    Total rows: {len(df)}")
            print(f"    Non-null P values: {df['P'].notna().sum()}")
            print(f"    Null P values: {df['P'].isna().sum()}")
            print(f"    P values by DATA source:")
            for source in df['DATA'].unique():
                source_df = df[df['DATA'] == source]
                print(f"      {source}: {source_df['P'].notna().sum()} non-null, sample values: {source_df['P'].dropna().head(3).tolist()}")
        
        # Get unique locations
        locations = sorted(df['Location'].unique())
        print(f"  Locations found: {len(locations)}")
        print(f"  Location names: {locations}")
        
        # Get data sources
        sources = sorted(df['DATA'].unique())
        print(f"  Data sources: {sources}")
        
        # Date range in data
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Total records: {len(df)}")
        
        print("Data loaded successfully\n")
        
        return df, locations
        
    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        raise

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_aligned_data(df, location, debug=False):
    """
    Prepare aligned data for a specific location.
    Pivots data to have separate columns for Station and ERA5-Land.
    """
    # Filter for location - try exact match first, then case-insensitive
    loc_df = df[df['Location'] == location].copy()
    
    if debug:
        print(f"      Initial filter for '{location}': {len(loc_df)} rows")
        if len(loc_df) == 0:
            print(f"      Checking all unique locations in data...")
            all_locs = df['Location'].unique()
            print(f"      All locations: {all_locs.tolist()}")
            # Check for exact matches with different case/spaces
            matches = [loc for loc in all_locs if location.strip().lower() == loc.strip().lower()]
            if matches:
                print(f"      Found case/whitespace matches: {matches}")
    
    # If no data found, try case-insensitive and strip whitespace
    if len(loc_df) == 0:
        loc_df = df[df['Location'].str.strip().str.lower() == location.strip().lower()].copy()
        if len(loc_df) > 0 and debug:
            print(f"      WARNING: Found {len(loc_df)} rows using case-insensitive match")
    
    # Also check for similar location names (handle spaces/underscores)
    if len(loc_df) == 0:
        # Try to find similar names
        all_locations = df['Location'].unique()
        similar = [loc for loc in all_locations 
                   if location.lower().replace(' ', '_').replace('-', '_') in loc.lower().replace(' ', '_').replace('-', '_')
                   or loc.lower().replace(' ', '_').replace('-', '_') in location.lower().replace(' ', '_').replace('-', '_')]
        if similar:
            if debug:
                print(f"      WARNING: No exact match, but found similar: {similar}")
                print(f"      Trying first similar: {similar[0]}")
            loc_df = df[df['Location'] == similar[0]].copy()
    
    if debug:
        print(f"\n    DEBUG for {location}:")
        print(f"      Rows in location data: {len(loc_df)}")
        print(f"      Available columns: {loc_df.columns.tolist()}")
        print(f"      DATA column values: {loc_df['DATA'].unique()}")
        print(f"      P column (precipitation) present: {'P' in loc_df.columns}")
        if 'P' in loc_df.columns:
            print(f"      P column statistics:")
            print(f"        Total rows: {len(loc_df)}")
            print(f"        Non-null count: {loc_df['P'].notna().sum()}")
            print(f"        Null count: {loc_df['P'].isna().sum()}")
            print(f"        Zero values: {(loc_df['P'] == 0).sum()}")
            print(f"        Non-zero values: {(loc_df['P'] != 0).sum()}")
            print(f"        Min value: {loc_df['P'].min()}")
            print(f"        Max value: {loc_df['P'].max()}")
            print(f"      P column sample values (first 5, including zeros):")
            station_p = loc_df[loc_df['DATA'] == 'Station']['P']
            era5_p = loc_df[loc_df['DATA'] == 'ERA5-Land']['P']
            print(f"        Station: {station_p.head(5).tolist()}")
            print(f"        ERA5-Land: {era5_p.head(5).tolist()}")
            print(f"      P column value counts (first 10):")
            print(f"        Station: {station_p.value_counts().head(10)}")
            print(f"        ERA5-Land: {era5_p.value_counts().head(10)}")
    
    # Check if required columns exist
    required_vars = ['T', 'RH', 'P']
    missing_vars = [v for v in required_vars if v not in loc_df.columns]
    if missing_vars:
        if debug:
            print(f"      WARNING: Missing variables: {missing_vars}")
        # Return empty dataframe with expected structure
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    
    # Pivot to separate Station and ERA5-Land
    # First, check what values are in DATA column - normalize them
    if len(loc_df) > 0:
        # Normalize DATA column values (strip whitespace, handle case)
        loc_df['DATA'] = loc_df['DATA'].astype(str).str.strip()
        
        # Check what values we actually have
        data_values = loc_df['DATA'].unique()
        if debug:
            print(f"      DATA column unique values (after strip): {data_values}")
            print(f"      DATA value counts:")
            print(loc_df['DATA'].value_counts())
        
        # Try to map to expected values
        data_mapping = {}
        for val in data_values:
            val_lower = val.lower()
            if 'station' in val_lower:
                data_mapping[val] = 'Station'
            elif 'era5' in val_lower or 'land' in val_lower:
                data_mapping[val] = 'ERA5-Land'
        
        if data_mapping and debug:
            print(f"      DATA value mapping: {data_mapping}")
        
        # Apply mapping if we found matches
        if data_mapping:
            loc_df['DATA'] = loc_df['DATA'].map(data_mapping).fillna(loc_df['DATA'])
    
    if debug:
        print(f"      Checking for required variables in columns: {loc_df.columns.tolist()}")
    
    # Check which variables actually exist
    available_vars = [v for v in ['T', 'RH', 'P'] if v in loc_df.columns]
    if debug:
        print(f"      Available variables for pivot: {available_vars}")
    
    if not available_vars:
        if debug:
            print(f"      WARNING: No variables available for pivot!")
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    
    try:
        pivot_df = loc_df.pivot_table(
            index='Date',
            columns='DATA',
            values=available_vars,  # Use only available variables
            aggfunc='first'
        )
    except Exception as e:
        if debug:
            print(f"      WARNING: Pivot error: {e}")
            print(f"      Attempting to debug pivot...")
            print(f"      Date column: {loc_df['Date'].dtype}")
            print(f"      DATA column values: {loc_df['DATA'].value_counts()}")
            print(f"      Sample data:")
            print(loc_df[['Date', 'DATA', 'T', 'RH', 'P']].head(10) if all(c in loc_df.columns for c in ['T', 'RH', 'P']) else loc_df.head(10))
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    
    if debug:
        print(f"      Pivot shape: {pivot_df.shape}")
        print(f"      Pivot columns (before flattening): {pivot_df.columns.tolist()}")
    
    # Flatten column names - handle multi-level columns
    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = [f'{var}_{source}' for var, source in pivot_df.columns]
    else:
        # Already flattened
        pass
    
    if debug:
        print(f"      Columns after flattening: {pivot_df.columns.tolist()}")
    
    # Rename for consistency - handle both 'ERA5-Land' and 'ERA5' in column names
    rename_cols = {}
    for col in pivot_df.columns:
        # Split by underscore to get variable and source
        parts = col.split('_')
        var = parts[0]  # T, RH, or P
        
        # Check source part (could be 'Station', 'ERA5-Land', 'ERA5', etc.)
        source_part = '_'.join(parts[1:]) if len(parts) > 1 else ''
        
        if 'Station' in source_part or 'station' in source_part.lower():
            rename_cols[col] = f'{var}_Station'
        elif 'ERA5' in source_part or 'era5' in source_part.lower():
            rename_cols[col] = f'{var}_ERA5'
    
    if debug:
        print(f"      Rename mapping: {rename_cols}")
    
    pivot_df.rename(columns=rename_cols, inplace=True)
    
    if debug:
        print(f"      Final columns: {pivot_df.columns.tolist()}")
        if 'P_Station' in pivot_df.columns:
            print(f"      P_Station statistics:")
            print(f"        Total rows: {len(pivot_df)}")
            print(f"        Non-null: {pivot_df['P_Station'].notna().sum()}")
            print(f"        Null: {pivot_df['P_Station'].isna().sum()}")
            print(f"        Zero values: {(pivot_df['P_Station'] == 0).sum()}")
            print(f"        Non-zero values: {(pivot_df['P_Station'] != 0).sum()}")
            print(f"        Sample values (first 5): {pivot_df['P_Station'].head(5).tolist()}")
        if 'P_ERA5' in pivot_df.columns:
            print(f"      P_ERA5 statistics:")
            print(f"        Total rows: {len(pivot_df)}")
            print(f"        Non-null: {pivot_df['P_ERA5'].notna().sum()}")
            print(f"        Null: {pivot_df['P_ERA5'].isna().sum()}")
            print(f"        Zero values: {(pivot_df['P_ERA5'] == 0).sum()}")
            print(f"        Non-zero values: {(pivot_df['P_ERA5'] != 0).sum()}")
            print(f"        Sample values (first 5): {pivot_df['P_ERA5'].head(5).tolist()}")
    
    return pivot_df

def filter_growing_season(df):
    """Filter dataframe to only include growing season months (April-September)."""
    return df[df.index.month.isin(GROWING_SEASON_MONTHS)]

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(aligned_data, variable='T'):
    """
    Calculate agreement metrics between Station and ERA5-Land.
    
    Returns: dict with bias, RMSE, correlation, and sample size
    """
    station_col = f'{variable}_Station'
    era5_col = f'{variable}_ERA5'
    
    # Check if columns exist
    if station_col not in aligned_data.columns or era5_col not in aligned_data.columns:
        return None
    
    # Get aligned values (both must be non-null)
    station_vals = aligned_data[station_col].dropna()
    era5_vals = aligned_data[era5_col].dropna()
    
    # Align by index
    common_idx = station_vals.index.intersection(era5_vals.index)
    if len(common_idx) == 0:
        return None
    
    station_aligned = station_vals.loc[common_idx]
    era5_aligned = era5_vals.loc[common_idx]
    
    # Remove any remaining NaN pairs
    mask = ~(station_aligned.isna() | era5_aligned.isna())
    station_aligned = station_aligned[mask]
    era5_aligned = era5_aligned[mask]
    
    if len(station_aligned) == 0:
        return None
    
    # Calculate metrics
    bias = np.mean(era5_aligned - station_aligned)
    rmse = np.sqrt(np.mean((era5_aligned - station_aligned)**2))
    
    # Correlation (handle case where all values are the same)
    if station_aligned.std() == 0 or era5_aligned.std() == 0:
        corr = np.nan
    else:
        corr, _ = stats.pearsonr(station_aligned, era5_aligned)
    
    n = len(station_aligned)
    
    return {
        'bias': bias,
        'rmse': rmse,
        'corr': corr,
        'n': n
    }

def calculate_rainy_days(aligned_data, precip_threshold=0.0):
    """
    Calculate number of rainy days for Station and ERA5-Land.
    
    Args:
        aligned_data: DataFrame with P_Station and P_ERA5 columns
        precip_threshold: Minimum precipitation to count as rainy day (mm)
    
    Returns: dict with station_rainy_days, era5_rainy_days, and total_days
    """
    station_col = 'P_Station'
    era5_col = 'P_ERA5'
    
    if station_col not in aligned_data.columns or era5_col not in aligned_data.columns:
        return None
    
    # Count rainy days
    station_rainy = (aligned_data[station_col] > precip_threshold).sum()
    era5_rainy = (aligned_data[era5_col] > precip_threshold).sum()
    total_days = len(aligned_data.dropna(subset=[station_col, era5_col]))
    
    return {
        'station_rainy_days': station_rainy,
        'era5_rainy_days': era5_rainy,
        'total_days': total_days,
        'station_rainy_freq': station_rainy / total_days * 100 if total_days > 0 else 0,
        'era5_rainy_freq': era5_rainy / total_days * 100 if total_days > 0 else 0,
        'rainy_days_diff': era5_rainy - station_rainy
    }

# ============================================================================
# PRECIPITATION FILTERING ANALYSIS
# ============================================================================

def test_precip_filtering(aligned_data, thresholds):
    """
    Test different precipitation thresholds to filter ERA5-Land data.
    Goal: Find threshold that gives similar number of rainy days as station.
    
    HOW THE FILTERING WORKS:
    ------------------------
    1. For each threshold (e.g., 1.0, 1.5, 2.0, ..., 5.0 mm):
       - Take the ERA5-Land precipitation values
       - Set all values BELOW the threshold to 0 (filter out small amounts)
       - Keep values >= threshold unchanged
    
    2. Count rainy days:
       - Station: Count days where precipitation > 0 mm (baseline)
       - ERA5-Land (filtered): Count days where filtered precipitation > 0 mm
       - Compare the difference in rainy day counts
    
    3. Find best threshold:
       - The threshold that minimizes the absolute difference between
         ERA5-Land rainy days and Station rainy days
    
    EXAMPLE:
    --------
    If threshold = 2.0 mm:
    - ERA5-Land values: [0.5, 1.5, 2.5, 3.0, 0.1] mm
    - After filtering:   [0.0, 0.0, 2.5, 3.0, 0.0] mm
    - Rainy days: 2 (instead of 5 before filtering)
    
    This helps reduce ERA5-Land's overestimation of rainy day frequency
    by removing very small precipitation amounts that may not be realistic.
    
    Returns: DataFrame with results for each threshold
    """
    results = []
    
    # Check if required columns exist
    if 'P_Station' not in aligned_data.columns or 'P_ERA5' not in aligned_data.columns:
        print(f"    WARNING: Missing precipitation columns. Available: {aligned_data.columns.tolist()}")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Get baseline (station rainy days with 0 threshold)
    station_baseline = (aligned_data['P_Station'] > 0).sum()
    
    for threshold in thresholds:
        # Filter ERA5-Land: set values below threshold to 0
        era5_filtered = aligned_data['P_ERA5'].copy()
        era5_filtered[era5_filtered < threshold] = 0
        
        # Count rainy days after filtering
        era5_rainy_filtered = (era5_filtered > 0).sum()
        
        # Calculate difference from station
        diff = era5_rainy_filtered - station_baseline
        abs_diff = abs(diff)
        
        # Also calculate metrics with filtered data
        station_vals = aligned_data['P_Station'].dropna()
        era5_filtered_vals = era5_filtered.dropna()
        
        common_idx = station_vals.index.intersection(era5_filtered_vals.index)
        if len(common_idx) > 0:
            station_aligned = station_vals.loc[common_idx]
            era5_aligned = era5_filtered_vals.loc[common_idx]
            mask = ~(station_aligned.isna() | era5_aligned.isna())
            station_aligned = station_aligned[mask]
            era5_aligned = era5_aligned[mask]
            
            if len(station_aligned) > 0:
                bias = np.mean(era5_aligned - station_aligned)
                rmse = np.sqrt(np.mean((era5_aligned - station_aligned)**2))
                if station_aligned.std() > 0 and era5_aligned.std() > 0:
                    corr, _ = stats.pearsonr(station_aligned, era5_aligned)
                else:
                    corr = np.nan
            else:
                bias, rmse, corr = np.nan, np.nan, np.nan
        else:
            bias, rmse, corr = np.nan, np.nan, np.nan
        
        results.append({
            'threshold_mm': threshold,
            'station_rainy_days': station_baseline,
            'era5_rainy_days_filtered': era5_rainy_filtered,
            'rainy_days_diff': diff,
            'abs_rainy_days_diff': abs_diff,
            'bias_mm': bias,
            'rmse_mm': rmse,
            'corr': corr
        })
    
    return pd.DataFrame(results)

# ============================================================================
# SUMMARY TABLE CREATION
# ============================================================================

def create_summary_table(df, locations, period_name="Full Period"):
    """
    Create comprehensive summary table for all locations and variables.
    Rainy day frequency is calculated for the specified period (Full Period or Growing Season).
    """
    print(f"\n{'='*80}")
    print(f"CREATING SUMMARY TABLE: {period_name}")
    print(f"{'='*80}")
    
    results = []
    
    for location in locations:
        print(f"  Processing {location}...")
        
        # Prepare aligned data with debug for first location
        debug_mode = (location == locations[0])
        aligned_data = prepare_aligned_data(df, location, debug=debug_mode)
        
        # Filter for period if needed
        if period_name == "Growing Season":
            aligned_data = filter_growing_season(aligned_data)
        
        # Calculate metrics for each variable
        for var in ['T', 'RH', 'P']:
            metrics = calculate_metrics(aligned_data, variable=var)
            
            if metrics is None:
                continue
            
            # Rainy days (only for precipitation) - calculated for the current period
            rainy_days_info = None
            if var == 'P':
                rainy_days_info = calculate_rainy_days(aligned_data, precip_threshold=0.0)
            
            row = {
                'Location': location,
                'Variable': var,
                'Period': period_name,
                'Bias': f"{metrics['bias']:.3f}",
                'RMSE': f"{metrics['rmse']:.3f}",
                'Correlation': f"{metrics['corr']:.3f}",
                'N': metrics['n']
            }
            
            if rainy_days_info:
                row.update({
                    'Station_RainyDays': rainy_days_info['station_rainy_days'],
                    'ERA5_RainyDays': rainy_days_info['era5_rainy_days'],
                    'RainyDays_Diff': rainy_days_info['rainy_days_diff'],
                    'Station_RainyFreq_%': f"{rainy_days_info['station_rainy_freq']:.2f}",
                    'ERA5_RainyFreq_%': f"{rainy_days_info['era5_rainy_freq']:.2f}"
                })
            
            results.append(row)
    
    summary_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = os.path.join(OUTPUT_DIR, f'Summary_Table_{period_name.replace(" ", "_")}.csv')
    summary_df.to_csv(output_file, index=False)
    
    print(f"\nSummary table saved: {output_file}")
    print(f"   Total rows: {len(summary_df)}")
    
    return summary_df

# ============================================================================
# PRECIPITATION FILTERING ANALYSIS FOR ALL LOCATIONS
# ============================================================================

def create_precip_filtering_analysis(df, locations):
    """
    Test precipitation filtering thresholds for all locations.
    """
    print(f"\n{'='*80}")
    print("PRECIPITATION FILTERING ANALYSIS")
    print(f"{'='*80}")
    
    all_results = []
    
    for location in locations:
        print(f"  Testing thresholds for {location}...")
        
        # Prepare aligned data with debug for first location
        debug_mode = (location == locations[0])
        aligned_data = prepare_aligned_data(df, location, debug=debug_mode)
        
        # Check if we have the required columns
        if aligned_data.empty:
            print(f"    WARNING: Skipping {location}: No aligned data (empty dataframe)")
            continue
        
        # Check what columns we have
        has_station = 'P_Station' in aligned_data.columns
        has_era5 = 'P_ERA5' in aligned_data.columns
        
        if not has_station and not has_era5:
            print(f"    WARNING: Skipping {location}: No precipitation data at all")
            print(f"       Available columns: {aligned_data.columns.tolist()}")
            continue
        
        # Final check - we need both for comparison
        if 'P_Station' not in aligned_data.columns or 'P_ERA5' not in aligned_data.columns:
            print(f"    WARNING: Still missing one source after merge attempt, skipping threshold analysis")
            print(f"       Available columns: {aligned_data.columns.tolist()}")
            continue
        
        # Test thresholds
        threshold_results = test_precip_filtering(aligned_data, PRECIP_THRESHOLDS)
        
        if threshold_results.empty:
            print(f"    WARNING: Skipping {location}: No results from threshold testing")
            continue
        
        threshold_results['Location'] = location
        all_results.append(threshold_results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Find best threshold for each location (minimum absolute difference in rainy days)
    best_thresholds = []
    for location in locations:
        loc_results = combined_results[combined_results['Location'] == location]
        if len(loc_results) > 0:
            best_idx = loc_results['abs_rainy_days_diff'].idxmin()
            best_row = loc_results.loc[best_idx]
            best_thresholds.append({
                'Location': location,
                'Best_Threshold_mm': best_row['threshold_mm'],
                'Station_RainyDays': best_row['station_rainy_days'],
                'ERA5_RainyDays_Filtered': best_row['era5_rainy_days_filtered'],
                'RainyDays_Diff': best_row['rainy_days_diff'],
                'Bias_mm': best_row['bias_mm'],
                'RMSE_mm': best_row['rmse_mm'],
                'Correlation': best_row['corr']
            })
    
    best_thresholds_df = pd.DataFrame(best_thresholds)
    
    # Save results
    output_file_all = os.path.join(OUTPUT_DIR, 'Precip_Filtering_All_Thresholds.csv')
    output_file_best = os.path.join(OUTPUT_DIR, 'Precip_Filtering_Best_Thresholds.csv')
    
    combined_results.to_csv(output_file_all, index=False)
    best_thresholds_df.to_csv(output_file_best, index=False)
    
    print(f"\nFiltering analysis complete:")
    print(f"   All thresholds: {output_file_all}")
    print(f"   Best thresholds: {output_file_best}")
    
    # Print summary
    print(f"\n--- Best Threshold Summary ---")
    print(best_thresholds_df.to_string(index=False))
    
    return combined_results, best_thresholds_df

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main analysis function."""
    print("="*80)
    print("ERA5-LAND vs STATION DATA AGREEMENT ANALYSIS")
    print("="*80)
    print(f"Date range: {START_YEAR}-{END_YEAR}")
    print(f"Growing season: Months {GROWING_SEASON_MONTHS}")
    print(f"Precipitation thresholds to test: {PRECIP_THRESHOLDS}")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found: {INPUT_FILE}")
        print("   Please check the file path in the script configuration.")
        return
    
    # Set output directory to input file directory
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.dirname(INPUT_FILE) if os.path.dirname(INPUT_FILE) else os.getcwd()
    
    df, locations = load_data(INPUT_FILE)
    
    # ========================================================================
    # STEP 2: Create summary tables
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: CREATING SUMMARY TABLES")
    print("="*80)
    
    # Full period analysis
    summary_full = create_summary_table(df, locations, period_name="Full Period")
    
    # Growing season analysis
    summary_growing = create_summary_table(df, locations, period_name="Growing Season")
    
    # ========================================================================
    # STEP 3: Precipitation filtering analysis (per location)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: PRECIPITATION FILTERING ANALYSIS (Per Location)")
    print("="*80)
    
    filtering_results, best_thresholds = create_precip_filtering_analysis(df, locations)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"All results saved in: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. Summary_Table_Full_Period.csv - Metrics for all months")
    print("  2. Summary_Table_Growing_Season.csv - Metrics for April-October only")
    print("  3. Precip_Filtering_All_Thresholds.csv - All tested thresholds (1.0-5.0 mm) per location")
    print("  4. Precip_Filtering_Best_Thresholds.csv - Optimal threshold per location")
    print("="*80)

if __name__ == "__main__":
    main()

