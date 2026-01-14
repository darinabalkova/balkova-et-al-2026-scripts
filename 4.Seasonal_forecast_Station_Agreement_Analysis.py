#!/usr/bin/env python3
"""
4. Seasonal Forecast vs Station Data Agreement Analysis

This script performs comprehensive comparison between Seasonal Forecast (ensemble mean) 
and Station data:
1. Overall agreement metrics (bias, RMSE, correlation, MAE, MB) for T, RH, P
2. Rainy day frequency comparison
3. Analysis for growing season only (April-September)
4. Precipitation filtering threshold optimization (1-10 mm by 0.5 increments)

Input: 
- Forecast CSV with columns: Date (dd/mm/yyyy), Ensemble_Member, Location, T (°C), RH (%), Rain (mm), LW (h)
- Station CSV with columns: Date (dd/mm/yyyy), Location, T (°C), RH (%), Rain (mm), DATA
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

# Input file paths
# Define forecast files with their initiation months
# Format: (file_path, initiation_month, growing_season_months)
# Growing season months: April=4, June=6, July=7, August=8, September=9
# UPDATE_WITH_YOUR_FORECAST_FILE_PATHS
FORECAST_FILES = [
    (r"UPDATE_WITH_YOUR_FORECAST_FILE_PATH\SF_042008-2025_daily.csv", 4, [4, 5, 6, 7, 8, 9]),  # April: April-September
    (r"UPDATE_WITH_YOUR_FORECAST_FILE_PATH\SF_062008-2025_daily.csv", 6, [6, 7, 8, 9]),  # June: June-September
    (r"UPDATE_WITH_YOUR_FORECAST_FILE_PATH\SF_072008-2025_daily.csv", 7, [7, 8, 9]),      # July: July-September
    (r"UPDATE_WITH_YOUR_FORECAST_FILE_PATH\SF_082008-2025_daily.csv", 8, [8, 9]),         # August: August-September
]

# UPDATE_WITH_YOUR_STATION_FILE_PATH
STATION_FILE = r"UPDATE_WITH_YOUR_STATION_FILE_PATH"

# Output directory - will be set to station file directory in main() function
OUTPUT_DIR = None

# NOTE on RH offset: When using ensemble means, the RH bias/offset can be large and systematic.
# Subtracting this offset may not be appropriate if:
# - The offset varies by location or conditions
# - The offset is due to systematic model biases rather than calibration issues
# The offset is reported here for information, but correction should be applied carefully.

# Helper function to get safe output path (fallback to current dir if path too long)
def get_safe_output_path(filename):
    """Get output path, using current directory if OUTPUT_DIR path is too long."""
    output_dir = os.path.abspath(OUTPUT_DIR)
    full_path = os.path.join(output_dir, filename)
    
    # Windows path length limit is 260 characters
    if len(full_path) > 250:
        # Use current working directory as fallback
        fallback_path = os.path.join(os.getcwd(), filename)
        print(f"  WARNING: Output path too long ({len(full_path)} chars), using current directory instead")
        print(f"  Original path: {full_path}")
        print(f"  Fallback path: {fallback_path}")
        return fallback_path
    return full_path

# Date range
START_YEAR = 2008
END_YEAR = 2025

# Growing season months (April-September)
GROWING_SEASON_MONTHS = [4, 5, 6, 7, 8, 9]

# Precipitation filtering thresholds to test (mm)
PRECIP_THRESHOLDS = np.arange(1.0, 10.5, 0.5)  # 1.0, 1.5, 2.0, ..., 10.0

# Station information with Seasonal Forecast grid mapping
# Grid distribution:
# - 45,9: Castel San Giovanni
# - 45,10: Caorso, Roncopascolo
# - 45,11: Luzzara, Mirandola, Castel Maggiore
# - 45,12: Guarda Ferrarese, Ostellato
# - 44,12: Forli, Medicina
STATION_INFO = {
    'Castel San Giovanni': {'ID': 1, 'Latitude': 45.06, 'Longitude': 9.43, 'Seasonal_forecast_grid': (45, 9)},
    'Caorso': {'ID': 2, 'Latitude': 45.05, 'Longitude': 9.87, 'Seasonal_forecast_grid': (45, 10)},
    'Roncopascolo': {'ID': 3, 'Latitude': 44.83, 'Longitude': 10.27, 'Seasonal_forecast_grid': (45, 10)},
    'Luzzara': {'ID': 4, 'Latitude': 44.95, 'Longitude': 10.68, 'Seasonal_forecast_grid': (45, 11)},
    'Mirandola': {'ID': 5, 'Latitude': 44.88, 'Longitude': 11.07, 'Seasonal_forecast_grid': (45, 11)},
    'Castel Maggiore': {'ID': 6, 'Latitude': 44.58, 'Longitude': 11.36, 'Seasonal_forecast_grid': (45, 11)},
    'Guarda Ferrarese': {'ID': 7, 'Latitude': 44.93, 'Longitude': 11.75, 'Seasonal_forecast_grid': (45, 12)},
    'Ostellato': {'ID': 8, 'Latitude': 44.75, 'Longitude': 11.94, 'Seasonal_forecast_grid': (45, 12)},
    'Medicina': {'ID': 9, 'Latitude': 44.47, 'Longitude': 11.63, 'Seasonal_forecast_grid': (44, 12)},
    'Forli': {'ID': 10, 'Latitude': 44.22, 'Longitude': 12.04, 'Seasonal_forecast_grid': (44, 12)},
}

# Create reverse mapping: forecast grid -> list of station names (multiple stations can share a grid)
FORECAST_GRID_TO_STATIONS = {}
for station_name, info in STATION_INFO.items():
    grid = info['Seasonal_forecast_grid']
    grid_str = f"{grid[0]},{grid[1]}"
    if grid_str not in FORECAST_GRID_TO_STATIONS:
        FORECAST_GRID_TO_STATIONS[grid_str] = []
    FORECAST_GRID_TO_STATIONS[grid_str].append(station_name)

# Identify grids with multiple stations
GRIDS_WITH_MULTIPLE_STATIONS = {grid: stations for grid, stations in FORECAST_GRID_TO_STATIONS.items() if len(stations) > 1}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_forecast_data(file_path):
    """
    Load forecast CSV and calculate ensemble mean.
    
    Expected columns:
    - Date (dd/mm/yyyy)
    - Ensemble_Member (0-50)
    - Location (format "45,9")
    - T (°C)
    - RH (%)
    - Rain (mm)
    - LW (h) - optional
    """
    print("="*80)
    print("LOADING FORECAST DATA")
    print("="*80)
    
    try:
        df = pd.read_csv(file_path)
        print(f"  Initial shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Standardize date column
        date_col = None
        for col in df.columns:
            if 'Date' in col and 'dd/mm/yyyy' in col:
                date_col = col
                break
        
        if date_col is None:
            for col in df.columns:
                if col.lower() in ['date', 'date (dd/mm/yyyy)']:
                    date_col = col
                    break
        
        if date_col is None:
            raise ValueError(f"No date column found. Available columns: {df.columns.tolist()}")
        
        print(f"  Using date column: '{date_col}'")
        
        # Parse dates
        df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        if df['Date'].isna().all() or df['Date'].isna().sum() > len(df) * 0.5:
            print(f"  Warning: Date parsing with format '%d/%m/%Y' failed, trying alternative...")
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df.dropna(subset=['Date'])
        
        # Filter date range
        df = df[(df['Date'].dt.year >= START_YEAR) & (df['Date'].dt.year <= END_YEAR)]
        
        # Standardize column names
        rename_map = {}
        for col in df.columns:
            if 'T (°C)' in col or (col == 'T' and 'T' not in rename_map.values()):
                rename_map[col] = 'T'
            elif 'RH (%)' in col or (col == 'RH' and 'RH' not in rename_map.values()):
                rename_map[col] = 'RH'
            elif 'Rain (mm)' in col or 'Precip' in col or (col == 'P' and 'P' not in rename_map.values()):
                rename_map[col] = 'P'
        
        df.rename(columns=rename_map, inplace=True)
        
        # Check required columns
        required_cols = ['Date', 'Ensemble_Member', 'Location', 'T', 'RH', 'P']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Unique locations: {sorted(df['Location'].unique())}")
        print(f"  Ensemble members: {sorted(df['Ensemble_Member'].unique())}")
        
        # Calculate ensemble mean for each date and location
        print("  Calculating ensemble mean...")
        ensemble_mean = df.groupby(['Date', 'Location'])[['T', 'RH', 'P']].mean().reset_index()
        
        print(f"  Ensemble mean shape: {ensemble_mean.shape}")
        print(f"  Unique locations in ensemble mean: {sorted(ensemble_mean['Location'].unique())}")
        
        return ensemble_mean
        
    except Exception as e:
        print(f"ERROR: Failed to load forecast data: {e}")
        raise


def load_station_data(file_path):
    """
    Load station data CSV.
    
    Expected columns:
    - Date (dd/mm/yyyy) or Date
    - Location
    - T (°C) or T
    - RH (%) or RH
    - Rain (mm) or P or Precip
    - DATA (source: 'Station')
    """
    print("\n" + "="*80)
    print("LOADING STATION DATA")
    print("="*80)
    
    try:
        df = pd.read_csv(file_path)
        print(f"  Initial shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Standardize date column
        date_col = None
        for col in df.columns:
            if 'Date' in col and 'dd/mm/yyyy' in col:
                date_col = col
                break
        
        if date_col is None:
            for col in df.columns:
                if col.lower() in ['date', 'date (dd/mm/yyyy)']:
                    date_col = col
                    break
        
        if date_col is None:
            raise ValueError(f"No date column found. Available columns: {df.columns.tolist()}")
        
        print(f"  Using date column: '{date_col}'")
        
        # Parse dates
        df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        if df['Date'].isna().all() or df['Date'].isna().sum() > len(df) * 0.5:
            print(f"  Warning: Date parsing with format '%d/%m/%Y' failed, trying alternative...")
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        df = df.dropna(subset=['Date'])
        
        # Filter date range
        df = df[(df['Date'].dt.year >= START_YEAR) & (df['Date'].dt.year <= END_YEAR)]
        
        # Standardize column names
        rename_map = {}
        for col in df.columns:
            if 'T (°C)' in col or (col == 'T' and 'T' not in rename_map.values()):
                rename_map[col] = 'T'
            elif 'RH (%)' in col or (col == 'RH' and 'RH' not in rename_map.values()):
                rename_map[col] = 'RH'
            elif 'Rain (mm)' in col or 'Precip' in col or (col == 'P' and 'P' not in rename_map.values()):
                rename_map[col] = 'P'
        
        df.rename(columns=rename_map, inplace=True)
        
        # Filter to Station data only
        if 'DATA' in df.columns:
            df = df[df['DATA'].astype(str).str.strip().str.lower() == 'station']
        
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  Unique locations: {sorted(df['Location'].unique())}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: Failed to load station data: {e}")
        raise


def find_matching_location(loc_name, station_info_keys):
    """Find matching location name, handling variations, spaces, and abbreviations."""
    loc_clean = loc_name.strip()
    
    # Try exact match first
    if loc_clean in station_info_keys:
        return loc_clean
    
    # Try case-insensitive
    for key in station_info_keys:
        if key.lower() == loc_clean.lower():
            return key
    
    # Try with normalized spaces
    loc_normalized = ' '.join(loc_clean.split())
    for key in station_info_keys:
        if ' '.join(key.split()).lower() == loc_normalized.lower():
            return key
    
    # Try handling abbreviations (e.g., "S." vs "San", "St." vs "Saint")
    def normalize_abbrev(text):
        text = text.replace('S.', 'San').replace('St.', 'Saint').replace('St ', 'Saint ')
        return text
    
    loc_normalized_abbrev = normalize_abbrev(loc_clean)
    for key in station_info_keys:
        key_normalized = normalize_abbrev(key)
        if key_normalized.lower() == loc_normalized_abbrev.lower():
            return key
    
    # Try partial matching (e.g., "Castel S.Giovanni" should match "Castel San Giovanni")
    # First normalize abbreviations, then remove punctuation and normalize
    def normalize_for_match(text):
        # First expand abbreviations
        text = normalize_abbrev(text)
        # Then remove punctuation and normalize
        text = text.replace('.', '').replace(',', '').replace('-', ' ').replace("'", '')
        text = ' '.join(text.split())  # Normalize spaces
        return text.lower()
    
    loc_normalized_match = normalize_for_match(loc_clean)
    for key in station_info_keys:
        key_normalized_match = normalize_for_match(key)
        if key_normalized_match == loc_normalized_match:
            return key
    
    return None


def prepare_aligned_data(forecast_df, station_df, location):
    """
    Align forecast and station data for a specific location.
    
    Args:
        forecast_df: Forecast data with ensemble mean
        station_df: Station data
        location: Station name (e.g., 'Castel San Giovanni')
    
    Returns:
        DataFrame with aligned data, columns: Date, T_Forecast, T_Station, RH_Forecast, RH_Station, P_Forecast, P_Station
    """
    # Get forecast grid location for this station
    if location not in STATION_INFO:
        print(f"  WARNING: Location '{location}' not found in STATION_INFO")
        return pd.DataFrame()
    
    forecast_grid = STATION_INFO[location]['Seasonal_forecast_grid']
    forecast_grid_str = f"{forecast_grid[0]},{forecast_grid[1]}"
    
    # Filter forecast data for this grid location
    forecast_loc = forecast_df[forecast_df['Location'] == forecast_grid_str].copy()
    
    # Filter station data for this location - try to find matching location name
    station_loc = station_df[station_df['Location'] == location].copy()
    
    # If not found, try to find matching location name
    if station_loc.empty:
        available_locations = station_df['Location'].unique()
        matched_location = find_matching_location(location, available_locations)
        if matched_location:
            station_loc = station_df[station_df['Location'] == matched_location].copy()
            if not station_loc.empty:
                print(f"    Matched '{location}' to '{matched_location}' in station data")
    
    if forecast_loc.empty:
        print(f"  WARNING: No forecast data found for grid {forecast_grid_str}")
        return pd.DataFrame()
    
    if station_loc.empty:
        print(f"  WARNING: No station data found for {location}")
        return pd.DataFrame()
    
    # Merge on Date
    merged = pd.merge(
        forecast_loc[['Date', 'T', 'RH', 'P']].rename(columns={'T': 'T_Forecast', 'RH': 'RH_Forecast', 'P': 'P_Forecast'}),
        station_loc[['Date', 'T', 'RH', 'P']].rename(columns={'T': 'T_Station', 'RH': 'RH_Station', 'P': 'P_Station'}),
        on='Date',
        how='inner'
    )
    
    merged.set_index('Date', inplace=True)
    merged.sort_index(inplace=True)
    
    return merged


def filter_growing_season(df, growing_season_months=None):
    """
    Filter DataFrame to growing season months.
    
    Args:
        df: DataFrame with datetime index
        growing_season_months: List of month numbers (default: GROWING_SEASON_MONTHS)
    
    Returns:
        Filtered DataFrame
    """
    if growing_season_months is None:
        growing_season_months = GROWING_SEASON_MONTHS
    return df[df.index.month.isin(growing_season_months)]


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(aligned_data, variable='T'):
    """
    Calculate agreement metrics between forecast and station data.
    
    Returns:
        dict with keys: bias, rmse, corr, mae, mb, n
    """
    forecast_col = f'{variable}_Forecast'
    station_col = f'{variable}_Station'
    
    if forecast_col not in aligned_data.columns or station_col not in aligned_data.columns:
        return None
    
    # Remove NaN values
    valid_mask = aligned_data[forecast_col].notna() & aligned_data[station_col].notna()
    forecast_vals = aligned_data.loc[valid_mask, forecast_col]
    station_vals = aligned_data.loc[valid_mask, station_col]
    
    if len(forecast_vals) == 0:
        return None
    
    # Calculate metrics
    bias = np.mean(forecast_vals - station_vals)  # Forecast - Station
    rmse = np.sqrt(np.mean((forecast_vals - station_vals)**2))
    mae = np.mean(np.abs(forecast_vals - station_vals))
    mb = np.mean(forecast_vals - station_vals)  # Mean Bias (same as bias)
    
    if forecast_vals.std() > 0 and station_vals.std() > 0:
        corr, _ = stats.pearsonr(forecast_vals, station_vals)
    else:
        corr = np.nan
    
    return {
        'bias': bias,
        'rmse': rmse,
        'corr': corr,
        'mae': mae,
        'mb': mb,
        'n': len(forecast_vals)
    }


def calculate_rainy_days(aligned_data, precip_threshold=0.0):
    """
    Calculate rainy day frequency for forecast and station data.
    
    Args:
        aligned_data: DataFrame with P_Forecast and P_Station columns
        precip_threshold: Threshold for defining rainy days (mm)
    
    Returns:
        dict with rainy day counts and frequencies
    """
    if 'P_Forecast' not in aligned_data.columns or 'P_Station' not in aligned_data.columns:
        return None
    
    # Station: days with precipitation > threshold
    station_rainy = (aligned_data['P_Station'] > precip_threshold).sum()
    
    # Forecast: days with precipitation > threshold
    forecast_rainy = (aligned_data['P_Forecast'] > precip_threshold).sum()
    
    total_days = len(aligned_data)
    
    station_freq = (station_rainy / total_days * 100) if total_days > 0 else 0
    forecast_freq = (forecast_rainy / total_days * 100) if total_days > 0 else 0
    
    return {
        'station_rainy_days': station_rainy,
        'forecast_rainy_days': forecast_rainy,
        'rainy_days_diff': forecast_rainy - station_rainy,
        'total_days': total_days,
        'station_rainy_freq': station_freq,
        'forecast_rainy_freq': forecast_freq
    }


def analyze_precipitation_thresholds(aligned_data):
    """
    Analyze different precipitation thresholds to find optimal value.
    
    Returns:
        DataFrame with results for each threshold
    """
    results = []
    
    for threshold in PRECIP_THRESHOLDS:
        # Station baseline (threshold = 0 mm)
        station_baseline = (aligned_data['P_Station'] > 0).sum()
        
        # Forecast with threshold filtering
        forecast_filtered = aligned_data['P_Forecast'].copy()
        forecast_filtered[forecast_filtered < threshold] = 0
        forecast_rainy_filtered = (forecast_filtered > 0).sum()
        
        diff = forecast_rainy_filtered - station_baseline
        abs_diff = abs(diff)
        
        # Calculate metrics with filtered data
        station_vals = aligned_data['P_Station'].dropna()
        forecast_filtered_vals = forecast_filtered.dropna()
        
        common_idx = station_vals.index.intersection(forecast_filtered_vals.index)
        if len(common_idx) > 0:
            station_aligned = station_vals.loc[common_idx]
            forecast_aligned = forecast_filtered_vals.loc[common_idx]
            mask = ~(station_aligned.isna() | forecast_aligned.isna())
            station_aligned = station_aligned[mask]
            forecast_aligned = forecast_aligned[mask]
            
            if len(station_aligned) > 0:
                bias = np.mean(forecast_aligned - station_aligned)
                rmse = np.sqrt(np.mean((forecast_aligned - station_aligned)**2))
                if station_aligned.std() > 0 and forecast_aligned.std() > 0:
                    corr, _ = stats.pearsonr(station_aligned, forecast_aligned)
                else:
                    corr = np.nan
            else:
                bias, rmse, corr = np.nan, np.nan, np.nan
        else:
            bias, rmse, corr = np.nan, np.nan, np.nan
        
        results.append({
            'threshold_mm': threshold,
            'station_rainy_days': station_baseline,
            'forecast_rainy_days_filtered': forecast_rainy_filtered,
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

def create_summary_table(forecast_df, station_df, locations, growing_season_months=None, initiation_month=None):
    """
    Create comprehensive summary table for all locations and variables.
    Only for specified growing season months.
    
    Args:
        forecast_df: Forecast data
        station_df: Station data
        locations: List of location names
        growing_season_months: List of month numbers for growing season (default: GROWING_SEASON_MONTHS)
        initiation_month: Month number when forecast was initiated (for labeling)
    """
    if growing_season_months is None:
        growing_season_months = GROWING_SEASON_MONTHS
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    season_label = f"{month_names[growing_season_months[0]]}-{month_names[growing_season_months[-1]]}"
    if initiation_month:
        init_label = f" (Initiated {month_names[initiation_month]})"
    else:
        init_label = ""
    
    print("\n" + "="*80)
    print(f"CREATING SUMMARY TABLE: Growing Season {season_label}{init_label}")
    print("="*80)
    
    results = []
    
    for location in locations:
        print(f"  Processing {location}...")
        
        # Prepare aligned data
        aligned_data = prepare_aligned_data(forecast_df, station_df, location)
        
        if aligned_data.empty:
            print(f"    WARNING: No aligned data for {location}")
            continue
        
        # Filter to growing season
        aligned_data = filter_growing_season(aligned_data, growing_season_months)
        
        if aligned_data.empty:
            print(f"    WARNING: No growing season data for {location}")
            continue
        
        # Get station info
        station_id = STATION_INFO[location]['ID']
        lat = STATION_INFO[location]['Latitude']
        lon = STATION_INFO[location]['Longitude']
        forecast_grid = STATION_INFO[location]['Seasonal_forecast_grid']
        forecast_grid_str = f"{forecast_grid[0]},{forecast_grid[1]}"
        
        # Check if multiple stations share this grid
        stations_in_grid = FORECAST_GRID_TO_STATIONS.get(forecast_grid_str, [])
        shared_grid_note = f"Shared with {len(stations_in_grid)-1} other station(s)" if len(stations_in_grid) > 1 else "Unique grid"
        
        # Calculate metrics for each variable
        for var in ['T', 'RH', 'P']:
            metrics = calculate_metrics(aligned_data, variable=var)
            
            if metrics is None:
                continue
            
            # Rainy days (only for precipitation)
            rainy_days_info = None
            if var == 'P':
                rainy_days_info = calculate_rainy_days(aligned_data, precip_threshold=0.0)
            
            row = {
                'Location': location,
                'ID': station_id,
                'Latitude': lat,
                'Longitude': lon,
                'Forecast_Grid': forecast_grid_str,
                'Grid_Sharing': shared_grid_note,
                'Variable': var,
                'Bias': round(metrics['bias'], 3),
                'RMSE': round(metrics['rmse'], 3),
                'Correlation': round(metrics['corr'], 3),
                'MAE': round(metrics['mae'], 3),
                'MB': round(metrics['mb'], 3),
                'N': metrics['n']
            }
            
            if rainy_days_info:
                row.update({
                    'Station_RainyDays': rainy_days_info['station_rainy_days'],
                    'Forecast_RainyDays': rainy_days_info['forecast_rainy_days'],
                    'RainyDays_Diff': rainy_days_info['rainy_days_diff'],
                    'Station_RainyFreq_%': round(rainy_days_info['station_rainy_freq'], 2),
                    'Forecast_RainyFreq_%': round(rainy_days_info['forecast_rainy_freq'], 2)
                })
            
            results.append(row)
    
    summary_df = pd.DataFrame(results)
    
    if summary_df.empty:
        print("  WARNING: No data available for summary table")
        return None
    
    # Calculate and print overall mean RH bias (offset)
    rh_data = summary_df[summary_df['Variable'] == 'RH']
    if not rh_data.empty:
        overall_rh_bias = rh_data['Bias'].mean()
        print(f"\n  Overall mean RH bias (offset): {overall_rh_bias:.3f} %")
        print(f"    (Forecast - Station, averaged across all locations)")
        print(f"    Number of locations: {len(rh_data)}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save summary table with initiation month in filename if provided
    if initiation_month:
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        suffix = f"_Init{month_names[initiation_month]}"
    else:
        suffix = ""
    
    output_path = get_safe_output_path(f'Summary_Table_Forecast_vs_Station_GrowingSeason{suffix}.csv')
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"\n  Summary table saved to: {output_path}")
    except Exception as e:
        print(f"  ERROR: Could not save summary table: {e}")
        print(f"  Path length: {len(output_path)} characters")
    
    # Print warning about shared grids
    if GRIDS_WITH_MULTIPLE_STATIONS:
        print("\n  NOTE: Multiple stations share the same forecast grid:")
        for grid, stations in GRIDS_WITH_MULTIPLE_STATIONS.items():
            print(f"    Grid {grid}: {', '.join(stations)}")
        print("    Each station is compared individually against the same forecast values.")
    
    return summary_df


def create_aggregated_comparison(forecast_df, station_df, growing_season_months=None, initiation_month=None):
    """
    Create aggregated comparison for grids with multiple stations.
    For each grid with multiple stations, average all station data and compare against forecast.
    
    Args:
        forecast_df: Forecast data
        station_df: Station data
        growing_season_months: List of month numbers for growing season (default: GROWING_SEASON_MONTHS)
        initiation_month: Month number when forecast was initiated (for labeling)
    """
    global OUTPUT_DIR
    
    if growing_season_months is None:
        growing_season_months = GROWING_SEASON_MONTHS
    
    if not GRIDS_WITH_MULTIPLE_STATIONS:
        print("\n  No grids with multiple stations - skipping aggregated comparison")
        return None
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    season_label = f"{month_names[growing_season_months[0]]}-{month_names[growing_season_months[-1]]}"
    if initiation_month:
        init_label = f" (Initiated {month_names[initiation_month]})"
    else:
        init_label = ""
    
    print("\n" + "="*80)
    print(f"CREATING AGGREGATED COMPARISON: Forecast vs Average of Stations in Shared Grids{init_label}")
    print(f"Growing Season: {season_label}")
    print("="*80)
    
    results = []
    
    for grid_str, stations in GRIDS_WITH_MULTIPLE_STATIONS.items():
        print(f"  Processing grid {grid_str} with {len(stations)} stations: {', '.join(stations)}")
        
        # Get forecast data for this grid
        forecast_grid_data = forecast_df[forecast_df['Location'] == grid_str].copy()
        
        if forecast_grid_data.empty:
            print(f"    WARNING: No forecast data for grid {grid_str}")
            continue
        
        # Get and average station data for all stations in this grid
        station_data_list = []
        for station in stations:
            station_data = station_df[station_df['Location'] == station].copy()
            
            # If not found, try to find matching location name
            if station_data.empty:
                available_locations = station_df['Location'].unique()
                matched_location = find_matching_location(station, available_locations)
                if matched_location:
                    station_data = station_df[station_df['Location'] == matched_location].copy()
                    if not station_data.empty:
                        print(f"      Matched '{station}' to '{matched_location}' in station data")
            
            if not station_data.empty:
                station_data_list.append(station_data[['Date', 'T', 'RH', 'P']].set_index('Date'))
        
        if not station_data_list:
            print(f"    WARNING: No station data for any station in grid {grid_str}")
            continue
        
        # Average all station data
        from functools import reduce
        station_combined = reduce(lambda x, y: pd.concat([x, y]), station_data_list)
        station_averaged = station_combined.groupby('Date')[['T', 'RH', 'P']].mean()
        
        # Merge forecast and averaged station data
        forecast_aligned = forecast_grid_data.set_index('Date')[['T', 'RH', 'P']]
        merged = pd.merge(
            forecast_aligned.rename(columns={'T': 'T_Forecast', 'RH': 'RH_Forecast', 'P': 'P_Forecast'}),
            station_averaged.rename(columns={'T': 'T_Station_Avg', 'RH': 'RH_Station_Avg', 'P': 'P_Station_Avg'}),
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Filter to growing season
        merged = filter_growing_season(merged, growing_season_months)
        
        if merged.empty:
            print(f"    WARNING: No growing season data for grid {grid_str}")
            continue
        
        # Calculate metrics for each variable
        for var in ['T', 'RH', 'P']:
            forecast_col = f'{var}_Forecast'
            station_col = f'{var}_Station_Avg'
            
            if forecast_col not in merged.columns or station_col not in merged.columns:
                continue
            
            valid_mask = merged[forecast_col].notna() & merged[station_col].notna()
            forecast_vals = merged.loc[valid_mask, forecast_col]
            station_vals = merged.loc[valid_mask, station_col]
            
            if len(forecast_vals) == 0:
                continue
            
            # Calculate metrics
            bias = np.mean(forecast_vals - station_vals)
            rmse = np.sqrt(np.mean((forecast_vals - station_vals)**2))
            mae = np.mean(np.abs(forecast_vals - station_vals))
            mb = np.mean(forecast_vals - station_vals)
            
            if forecast_vals.std() > 0 and station_vals.std() > 0:
                corr, _ = stats.pearsonr(forecast_vals, station_vals)
            else:
                corr = np.nan
            
            # Rainy days for precipitation
            rainy_days_info = None
            if var == 'P':
                station_rainy = (merged['P_Station_Avg'] > 0).sum()
                forecast_rainy = (merged['P_Forecast'] > 0).sum()
                total_days = len(merged)
                station_freq = (station_rainy / total_days * 100) if total_days > 0 else 0
                forecast_freq = (forecast_rainy / total_days * 100) if total_days > 0 else 0
                rainy_days_info = {
                    'station_rainy_days': station_rainy,
                    'forecast_rainy_days': forecast_rainy,
                    'rainy_days_diff': forecast_rainy - station_rainy,
                    'station_rainy_freq': station_freq,
                    'forecast_rainy_freq': forecast_freq
                }
            
            row = {
                'Forecast_Grid': grid_str,
                'Stations_in_Grid': ', '.join(stations),
                'N_Stations': len(stations),
                'Variable': var,
                'Bias': round(bias, 3),
                'RMSE': round(rmse, 3),
                'Correlation': round(corr, 3),
                'MAE': round(mae, 3),
                'MB': round(mb, 3),
                'N': len(forecast_vals)
            }
            
            if rainy_days_info:
                row.update({
                    'Station_Avg_RainyDays': rainy_days_info['station_rainy_days'],
                    'Forecast_RainyDays': rainy_days_info['forecast_rainy_days'],
                    'RainyDays_Diff': rainy_days_info['rainy_days_diff'],
                    'Station_Avg_RainyFreq_%': round(rainy_days_info['station_rainy_freq'], 2),
                    'Forecast_RainyFreq_%': round(rainy_days_info['forecast_rainy_freq'], 2)
                })
            
            results.append(row)
    
    if not results:
        print("  WARNING: No aggregated comparison data available")
        return None
    
    aggregated_df = pd.DataFrame(results)
    
    # Ensure output directory exists (use absolute path)
    output_dir = os.path.abspath(OUTPUT_DIR)
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(output_dir):
            raise OSError(f"Failed to create output directory: {output_dir}")
        # Test write access
        test_file = os.path.join(output_dir, '.test_write')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            raise OSError(f"Directory exists but cannot write to it: {e}")
    except Exception as e:
        print(f"  ERROR: Could not create/access output directory: {e}")
        print(f"  Attempted path: {output_dir}")
        return None
    
    # Save aggregated comparison
    output_path = get_safe_output_path('Aggregated_Comparison_Forecast_vs_Station_Avg_GrowingSeason.csv')
    try:
        # Ensure parent directory exists right before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        aggregated_df.to_csv(output_path, index=False)
        print(f"\n  Aggregated comparison saved to: {output_path}")
    except Exception as e:
        print(f"  ERROR: Could not save aggregated comparison: {e}")
        print(f"  Output path: {output_path}")
        print(f"  Path length: {len(output_path)} characters")
        return None
    
    return aggregated_df


# ============================================================================
# PRECIPITATION THRESHOLD ANALYSIS
# ============================================================================

def prepare_aligned_data_by_grid(forecast_df, station_df, grid_str):
    """
    Align forecast and aggregated station data for a specific grid.
    For grids with multiple stations, average the station data.
    
    Args:
        forecast_df: Forecast data with ensemble mean
        station_df: Station data
        grid_str: Forecast grid string (e.g., "45,11")
    
    Returns:
        DataFrame with aligned data, columns: Date, T_Forecast, T_Station_Avg, RH_Forecast, RH_Station_Avg, P_Forecast, P_Station_Avg
    """
    # Get stations in this grid
    stations = FORECAST_GRID_TO_STATIONS.get(grid_str, [])
    
    if not stations:
        return pd.DataFrame()
    
    # Get forecast data for this grid
    forecast_grid_data = forecast_df[forecast_df['Location'] == grid_str].copy()
    
    if forecast_grid_data.empty:
        return pd.DataFrame()
    
    # Get and average station data for all stations in this grid
    station_data_list = []
    for station in stations:
        station_data = station_df[station_df['Location'] == station].copy()
        if not station_data.empty:
            station_data_list.append(station_data[['Date', 'T', 'RH', 'P']].set_index('Date'))
    
    if not station_data_list:
        return pd.DataFrame()
    
    # Average all station data
    from functools import reduce
    station_combined = reduce(lambda x, y: pd.concat([x, y]), station_data_list)
    station_averaged = station_combined.groupby('Date')[['T', 'RH', 'P']].mean()
    
    # Merge forecast and averaged station data
    forecast_aligned = forecast_grid_data.set_index('Date')[['T', 'RH', 'P']]
    merged = pd.merge(
        forecast_aligned.rename(columns={'T': 'T_Forecast', 'RH': 'RH_Forecast', 'P': 'P_Forecast'}),
        station_averaged.rename(columns={'T': 'T_Station_Avg', 'RH': 'RH_Station_Avg', 'P': 'P_Station_Avg'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    merged.sort_index(inplace=True)
    return merged


def analyze_thresholds_all_locations(forecast_df, station_df, locations, growing_season_months=None, initiation_month=None):
    """
    Analyze precipitation thresholds for all GRIDS (not individual stations).
    For grids with multiple stations, use aggregated (averaged) station data.
    
    Args:
        forecast_df: Forecast data
        station_df: Station data
        locations: List of location names
        growing_season_months: List of month numbers for growing season (default: GROWING_SEASON_MONTHS)
        initiation_month: Month number when forecast was initiated (for labeling)
    """
    if growing_season_months is None:
        growing_season_months = GROWING_SEASON_MONTHS
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    season_label = f"{month_names[growing_season_months[0]]}-{month_names[growing_season_months[-1]]}"
    if initiation_month:
        init_label = f" (Initiated {month_names[initiation_month]})"
    else:
        init_label = ""
    
    print("\n" + "="*80)
    print(f"ANALYZING PRECIPITATION THRESHOLDS BY GRID{init_label}")
    print(f"Growing Season: {season_label}")
    print("="*80)
    
    all_results = []
    
    # Process each unique grid
    unique_grids = sorted(FORECAST_GRID_TO_STATIONS.keys())
    
    for grid_str in unique_grids:
        stations = FORECAST_GRID_TO_STATIONS[grid_str]
        print(f"  Processing grid {grid_str} with {len(stations)} station(s): {', '.join(stations)}")
        
        # Prepare aligned data (using aggregated station data for grids with multiple stations)
        aligned_data = prepare_aligned_data_by_grid(forecast_df, station_df, grid_str)
        
        if aligned_data.empty:
            print(f"    WARNING: No aligned data for grid {grid_str}")
            continue
        
        # Filter to growing season
        aligned_data = filter_growing_season(aligned_data, growing_season_months)
        
        if aligned_data.empty:
            print(f"    WARNING: No growing season data for grid {grid_str}")
            continue
        
        # Rename columns to match analyze_precipitation_thresholds function
        # It expects P_Forecast and P_Station
        aligned_data = aligned_data.rename(columns={'P_Station_Avg': 'P_Station'})
        
        # Analyze thresholds
        threshold_results = analyze_precipitation_thresholds(aligned_data)
        threshold_results['Forecast_Grid'] = grid_str
        threshold_results['Stations_in_Grid'] = ', '.join(stations)
        threshold_results['N_Stations'] = len(stations)
        
        all_results.append(threshold_results)
    
    if not all_results:
        print("  WARNING: No data available for threshold analysis")
        return None, None
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Find optimal threshold per GRID (minimum absolute difference in rainy days)
    optimal_by_grid = combined_results.loc[
        combined_results.groupby('Forecast_Grid')['abs_rainy_days_diff'].idxmin()
    ][['Forecast_Grid', 'Stations_in_Grid', 'N_Stations', 'threshold_mm', 'abs_rainy_days_diff', 'rainy_days_diff', 'bias_mm', 'rmse_mm', 'corr']].copy()
    optimal_by_grid.columns = ['Forecast_Grid', 'Stations_in_Grid', 'N_Stations', 'Optimal_Threshold_mm', 'Min_Abs_Diff', 'RainyDays_Diff', 'Bias_mm', 'RMSE_mm', 'Correlation']
    
    # Overall optimal threshold (across all grids) - minimize mean absolute difference
    overall_optimal = combined_results.groupby('threshold_mm')['abs_rainy_days_diff'].mean().idxmin()
    
    # Get statistics for overall optimal threshold
    overall_stats = combined_results[combined_results['threshold_mm'] == overall_optimal].agg({
        'abs_rainy_days_diff': 'mean',
        'rainy_days_diff': 'mean',
        'bias_mm': 'mean',
        'rmse_mm': 'mean',
        'corr': 'mean'
    })
    
    print(f"\n  Overall optimal threshold: {overall_optimal} mm")
    print(f"    Mean absolute difference: {overall_stats['abs_rainy_days_diff']:.2f} days")
    print(f"    Mean difference: {overall_stats['rainy_days_diff']:.2f} days")
    print(f"    Mean bias: {overall_stats['bias_mm']:.3f} mm")
    print(f"    Mean RMSE: {overall_stats['rmse_mm']:.3f} mm")
    print(f"    Mean correlation: {overall_stats['corr']:.3f}")
    print(f"\n  Optimal threshold by grid:")
    print(optimal_by_grid.to_string(index=False))
    
    # Ensure output directory exists (use absolute path)
    output_dir = os.path.abspath(OUTPUT_DIR)
    try:
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(output_dir):
            raise OSError(f"Failed to create output directory: {output_dir}")
    except Exception as e:
        print(f"  ERROR: Could not create output directory: {e}")
        return None, None
    
    # Save results
    threshold_output = get_safe_output_path('Precipitation_Threshold_Analysis_Forecast_vs_Station_ByGrid.csv')
    try:
        os.makedirs(os.path.dirname(threshold_output), exist_ok=True)
        combined_results.to_csv(threshold_output, index=False)
        print(f"\n  Threshold analysis saved to: {threshold_output}")
    except Exception as e:
        print(f"  ERROR: Could not save threshold analysis: {e}")
        print(f"  Path length: {len(threshold_output)} characters")
    
    optimal_output = get_safe_output_path('Optimal_Precipitation_Thresholds_ByGrid_Forecast_vs_Station.csv')
    try:
        os.makedirs(os.path.dirname(optimal_output), exist_ok=True)
        optimal_by_grid.to_csv(optimal_output, index=False)
        print(f"  Optimal thresholds by grid saved to: {optimal_output}")
    except Exception as e:
        print(f"  ERROR: Could not save optimal thresholds: {e}")
        print(f"  Path length: {len(optimal_output)} characters")
    
    # Create summary with overall optimal
    overall_summary = pd.DataFrame([{
        'Overall_Optimal_Threshold_mm': overall_optimal,
        'Mean_Abs_Diff_Days': round(overall_stats['abs_rainy_days_diff'], 2),
        'Mean_Diff_Days': round(overall_stats['rainy_days_diff'], 2),
        'Mean_Bias_mm': round(overall_stats['bias_mm'], 3),
        'Mean_RMSE_mm': round(overall_stats['rmse_mm'], 3),
        'Mean_Correlation': round(overall_stats['corr'], 3)
    }])
    
    overall_output = get_safe_output_path('Overall_Optimal_Precipitation_Threshold_Forecast_vs_Station.csv')
    try:
        os.makedirs(os.path.dirname(overall_output), exist_ok=True)
        overall_summary.to_csv(overall_output, index=False)
        print(f"  Overall optimal threshold saved to: {overall_output}")
    except Exception as e:
        print(f"  ERROR: Could not save overall optimal threshold: {e}")
        print(f"  Path length: {len(overall_output)} characters")
    
    return combined_results, optimal_by_grid


# ============================================================================
# SUPPLEMENTARY TABLE (WIDE FORMAT) - Similar to Figue1.py
# ============================================================================

def create_supplementary_table_forecast(forecast_df, station_df, growing_season_months):
    """
    Creates Supplementary Table in wide format for Seasonal Forecast evaluation.
    Similar to Supplementary_Table_Statistical_Comparison_GrowingSeason_Only from Figue1.py.
    
    Format: One row per forecast grid with metrics (r, RMSE, MAE, MBE) for T, RH, P.
    Shows wet day frequency changes after applying 3mm threshold.
    Uses aggregated station data for grids with multiple stations.
    
    Period: Growing season months (April-September) across whole period 2008-2025
    """
    print("\n" + "="*80)
    print("CREATING SUPPLEMENTARY TABLE (Forecast Evaluation - Growing Season Only)")
    print("="*80)
    print("Format: Pearson correlation (r), RMSE, MAE, MBE for T, RH, P")
    print("Period: Growing season months (April-September) across whole period 2008-2025")
    print("Note: ALL metrics calculated using ONLY growing season months")
    print("Includes: Wet day frequency delta (Station vs Forecast with 3mm threshold)")
    print("="*80)
    
    # Get all unique forecast grids
    unique_grids = sorted(FORECAST_GRID_TO_STATIONS.keys())
    print(f"\n  Processing {len(unique_grids)} forecast grids: {unique_grids}")
    
    reshaped_data = []
    PRECIP_THRESHOLD_MM = 3.0  # 3mm threshold as requested
    
    for grid_str in unique_grids:
        print(f"  Processing grid {grid_str}...")
        
        # Prepare aligned data using aggregated station data
        aligned_data = prepare_aligned_data_by_grid(forecast_df, station_df, grid_str)
        
        if aligned_data.empty:
            print(f"    WARNING: No aligned data for grid {grid_str}")
            # Add row with NaN values
            row = {
                'Forecast_Grid': grid_str,
                'Stations': ', '.join(FORECAST_GRID_TO_STATIONS.get(grid_str, []))
            }
            for var in ['T', 'RH', 'P']:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
            row['P_RainyDays_Station'] = np.nan
            row['P_RainyDays_Forecast_3mm'] = np.nan
            row['P_Delta_RainyDays'] = np.nan
            reshaped_data.append(row)
            continue
        
        # Filter to growing season months ONLY
        aligned_data = filter_growing_season(aligned_data, growing_season_months)
        
        # Filter to 2008-2025 (whole period, but only growing season months)
        aligned_data = aligned_data[(aligned_data.index.year >= START_YEAR) & 
                                    (aligned_data.index.year <= END_YEAR)]
        
        if aligned_data.empty:
            print(f"    WARNING: No growing season data for grid {grid_str}")
            # Add row with NaN values
            row = {
                'Forecast_Grid': grid_str,
                'Stations': ', '.join(FORECAST_GRID_TO_STATIONS.get(grid_str, []))
            }
            for var in ['T', 'RH', 'P']:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
            row['P_RainyDays_Station'] = np.nan
            row['P_RainyDays_Forecast_3mm'] = np.nan
            row['P_Delta_RainyDays'] = np.nan
            reshaped_data.append(row)
            continue
        
        # Get stations in this grid for display
        stations_in_grid = FORECAST_GRID_TO_STATIONS.get(grid_str, [])
        
        row = {
            'Forecast_Grid': grid_str,
            'Stations': ', '.join(stations_in_grid)
        }
        
        # Calculate metrics for each variable
        for var in ['T', 'RH', 'P']:
            forecast_col = f'{var}_Forecast'
            station_col = f'{var}_Station_Avg'
            
            if forecast_col not in aligned_data.columns or station_col not in aligned_data.columns:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
                continue
            
            # Remove NaN values for comparison
            valid_mask = aligned_data[forecast_col].notna() & aligned_data[station_col].notna()
            forecast_vals = aligned_data.loc[valid_mask, forecast_col]
            station_vals = aligned_data.loc[valid_mask, station_col]
            
            if len(forecast_vals) == 0:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
                continue
            
            # Calculate metrics
            try:
                r, _ = stats.pearsonr(station_vals, forecast_vals)
                rmse = np.sqrt(np.mean((forecast_vals - station_vals)**2))
                mae = np.mean(np.abs(forecast_vals - station_vals))
                mb = np.mean(forecast_vals - station_vals)  # Mean Bias (Forecast - Station)
                
                row[f'{var}_r'] = round(r, 2)
                row[f'{var}_RMSE'] = round(rmse, 2)
                row[f'{var}_MAE'] = round(mae, 2)
                row[f'{var}_MBE'] = round(mb, 2)
            except:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
        
        # Calculate rainy day frequency delta for precipitation (growing season only)
        if 'P_Forecast' in aligned_data.columns and 'P_Station_Avg' in aligned_data.columns:
            # Get aligned precipitation data
            valid_mask_p = aligned_data['P_Station_Avg'].notna() & aligned_data['P_Forecast'].notna()
            station_p = aligned_data.loc[valid_mask_p, 'P_Station_Avg']
            forecast_p = aligned_data.loc[valid_mask_p, 'P_Forecast']
            
            if len(station_p) > 0:
                # Station rainy days (threshold = 0mm)
                station_rainy = (station_p > 0).sum()
                
                # Forecast rainy days with 3mm threshold
                forecast_p_filtered = forecast_p.copy()
                forecast_p_filtered[forecast_p_filtered < PRECIP_THRESHOLD_MM] = 0
                forecast_rainy = (forecast_p_filtered > 0).sum()
                
                # Delta (difference: Forecast - Station)
                delta_rainy_days = forecast_rainy - station_rainy
                
                row['P_RainyDays_Station'] = station_rainy
                row['P_RainyDays_Forecast_3mm'] = forecast_rainy
                row['P_Delta_RainyDays'] = delta_rainy_days
            else:
                row['P_RainyDays_Station'] = np.nan
                row['P_RainyDays_Forecast_3mm'] = np.nan
                row['P_Delta_RainyDays'] = np.nan
        else:
            row['P_RainyDays_Station'] = np.nan
            row['P_RainyDays_Forecast_3mm'] = np.nan
            row['P_Delta_RainyDays'] = np.nan
        
        reshaped_data.append(row)
    
    if not reshaped_data:
        print("  WARNING: No data available for supplementary table")
        return None
    
    final_table = pd.DataFrame(reshaped_data)
    
    # Reorder columns
    col_order = ['Forecast_Grid', 'Stations']
    for var in ['T', 'RH', 'P']:
        col_order.extend([f'{var}_r', f'{var}_RMSE', f'{var}_MAE', f'{var}_MBE'])
    # Add precipitation rainy day columns after P metrics
    col_order.extend(['P_RainyDays_Station', 'P_RainyDays_Forecast_3mm', 'P_Delta_RainyDays'])
    
    final_table = final_table.reindex(columns=col_order)
    
    output_path = get_safe_output_path('Supplementary_Table_Forecast_Evaluation_GrowingSeason_Only.csv')
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_table.to_csv(output_path, index=False)
        print(f"\n  Supplementary table saved to: {output_path}")
        print("\n" + final_table.to_string(index=False))
    except Exception as e:
        print(f"  ERROR: Could not save supplementary table: {e}")
        print(f"  Path length: {len(output_path)} characters")
        return None
    
    return final_table


def analyze_overall_rainy_day_frequency(forecast_df, station_df, growing_season_months):
    """
    Analyze overall rainy day frequency across all grids and all years.
    Finds the best threshold that minimizes the difference between forecast and station rainy day frequencies.
    Prints results to terminal.
    
    Args:
        forecast_df: Forecast data with ensemble mean
        station_df: Station data
        growing_season_months: List of month numbers for growing season
    """
    print("\n" + "="*80)
    print("OVERALL RAINY DAY FREQUENCY ANALYSIS (All Grids, All Years)")
    print("="*80)
    print("Analyzing overall rainy day frequencies across all forecast grids")
    print("to find the best threshold that matches station frequency.")
    print("="*80)
    
    # Collect all aligned data from all grids
    all_aligned_data = []
    
    unique_grids = sorted(FORECAST_GRID_TO_STATIONS.keys())
    print(f"\n  Collecting data from {len(unique_grids)} forecast grids...")
    
    for grid_str in unique_grids:
        # Prepare aligned data using aggregated station data
        aligned_data = prepare_aligned_data_by_grid(forecast_df, station_df, grid_str)
        
        if aligned_data.empty:
            continue
        
        # Filter to growing season months ONLY
        aligned_data = filter_growing_season(aligned_data, growing_season_months)
        
        # Filter to 2008-2025 (whole period, but only growing season months)
        aligned_data = aligned_data[(aligned_data.index.year >= START_YEAR) & 
                                    (aligned_data.index.year <= END_YEAR)]
        
        if aligned_data.empty:
            continue
        
        # Rename P_Station_Avg to P_Station for consistency
        if 'P_Station_Avg' in aligned_data.columns:
            aligned_data = aligned_data.rename(columns={'P_Station_Avg': 'P_Station'})
        
        if 'P_Forecast' in aligned_data.columns and 'P_Station' in aligned_data.columns:
            all_aligned_data.append(aligned_data[['P_Forecast', 'P_Station']])
    
    if not all_aligned_data:
        print("  WARNING: No aligned data available for overall analysis")
        return
    
    # Combine all data from all grids
    combined_data = pd.concat(all_aligned_data, ignore_index=False)
    combined_data = combined_data.sort_index()
    
    # Remove rows with NaN in either column
    valid_mask = combined_data['P_Station'].notna() & combined_data['P_Forecast'].notna()
    combined_data = combined_data[valid_mask]
    
    if combined_data.empty:
        print("  WARNING: No valid precipitation data after filtering")
        return
    
    total_days = len(combined_data)
    print(f"\n  Total days analyzed: {total_days}")
    print(f"  Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    # Station baseline: rainy days with threshold = 0mm
    station_rainy_days = (combined_data['P_Station'] > 0).sum()
    station_freq = (station_rainy_days / total_days * 100) if total_days > 0 else 0
    
    print(f"\n  STATION DATA (threshold = 0mm):")
    print(f"    Rainy days: {station_rainy_days}")
    print(f"    Frequency: {station_freq:.2f}%")
    
    # Test different thresholds
    print(f"\n  FORECAST DATA (with different thresholds):")
    print(f"  {'Threshold (mm)':<15} {'Rainy Days':<15} {'Frequency (%)':<15} {'Difference (days)':<20} {'Abs Difference':<20}")
    print(f"  {'-'*85}")
    
    best_threshold = None
    best_abs_diff = float('inf')
    best_diff = None
    best_forecast_rainy = None
    best_forecast_freq = None
    
    results = []
    
    for threshold in PRECIP_THRESHOLDS:
        # Apply threshold to forecast
        forecast_filtered = combined_data['P_Forecast'].copy()
        forecast_filtered[forecast_filtered < threshold] = 0
        forecast_rainy_days = (forecast_filtered > 0).sum()
        forecast_freq = (forecast_rainy_days / total_days * 100) if total_days > 0 else 0
        
        diff = forecast_rainy_days - station_rainy_days
        abs_diff = abs(diff)
        
        results.append({
            'threshold_mm': threshold,
            'forecast_rainy_days': forecast_rainy_days,
            'forecast_freq_%': forecast_freq,
            'difference_days': diff,
            'abs_difference_days': abs_diff
        })
        
        # Check if this is the best threshold
        if abs_diff < best_abs_diff:
            best_abs_diff = abs_diff
            best_threshold = threshold
            best_diff = diff
            best_forecast_rainy = forecast_rainy_days
            best_forecast_freq = forecast_freq
        
        # Print all thresholds, highlight 3mm
        marker = " <-- 3mm threshold" if threshold == 3.0 else ""
        print(f"  {threshold:<15.1f} {forecast_rainy_days:<15} {forecast_freq:<15.2f} {diff:<20} {abs_diff:<20}{marker}")
    
    print(f"\n  {'='*85}")
    print(f"\n  BEST THRESHOLD: {best_threshold} mm")
    print(f"    Forecast rainy days: {best_forecast_rainy}")
    print(f"    Forecast frequency: {best_forecast_freq:.2f}%")
    print(f"    Difference from station: {best_diff} days")
    print(f"    Absolute difference: {best_abs_diff} days")
    
    # Check if 3mm is still optimal
    three_mm_result = next((r for r in results if r['threshold_mm'] == 3.0), None)
    if three_mm_result:
        three_mm_diff = three_mm_result['abs_difference_days']
        print(f"\n  3mm THRESHOLD ANALYSIS:")
        print(f"    Forecast rainy days: {three_mm_result['forecast_rainy_days']}")
        print(f"    Forecast frequency: {three_mm_result['forecast_freq_%']:.2f}%")
        print(f"    Difference from station: {three_mm_result['difference_days']} days")
        print(f"    Absolute difference: {three_mm_diff} days")
        
        if best_threshold == 3.0:
            print(f"\n  >>> 3mm IS the optimal threshold! <<<")
        else:
            print(f"\n  >>> 3mm is NOT the optimal threshold (best is {best_threshold}mm) <<<")
            print(f"    Difference: {three_mm_diff - best_abs_diff} days worse than optimal")
    
    print(f"\n  {'='*85}")
    print(f"\n  SUMMARY:")
    print(f"    Station rainy days: {station_rainy_days} ({station_freq:.2f}%)")
    print(f"    Best forecast threshold: {best_threshold}mm")
    print(f"    Best forecast rainy days: {best_forecast_rainy} ({best_forecast_freq:.2f}%)")
    print(f"    Best match difference: {best_diff} days ({best_abs_diff} days absolute)")
    print(f"  {'='*85}\n")


# ============================================================================
# RH BIAS SUMMARY BY INITIATION MONTH AND GRID
# ============================================================================

def build_rh_bias_summary(all_results):
    """
    Build a table of mean RH bias per forecast initiation month and grid.

    Uses both:
    - Aggregated comparison tables (for grids with multiple stations)
    - Summary tables (for all grids, including single-station grids)

    Output:
      * Prints a compact table to the terminal.
      * Saves a CSV: RH_Bias_By_InitMonth_And_Grid.csv

    This is useful if you want to apply a simple RH bias-offset correction:
      RH_corrected = RH_forecast - RH_bias (for a given init month & grid).
    """
    if not all_results:
        print("\nNo results available to build RH bias summary.")
        return

    print("\n" + "="*80)
    print("RH BIAS SUMMARY BY INITIATION MONTH AND GRID")
    print("="*80)

    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    rows = []
    # Track which grids we've seen (to avoid duplicates)
    seen_grids_per_init = {}

    for result in all_results:
        init_month = result.get('initiation_month')
        init_label = month_names.get(init_month, str(init_month))
        
        # First, try aggregated_df (for grids with multiple stations)
        agg = result.get('aggregated_df')
        if agg is not None and isinstance(agg, pd.DataFrame) and not agg.empty:
            try:
                rh_rows = agg[agg['Variable'] == 'RH']
                if not rh_rows.empty:
                    for _, row in rh_rows.iterrows():
                        grid = row.get('Forecast_Grid')
                        key = (grid, init_month)
                        if key not in seen_grids_per_init:
                            rows.append({
                                'Forecast_Grid': grid,
                                'Stations_in_Grid': row.get('Stations_in_Grid'),
                                'N_Stations': row.get('N_Stations'),
                                'Init_Month_Num': init_month,
                                'Init_Month': init_label,
                                'RH_Bias': row.get('Bias')
                            })
                            seen_grids_per_init[key] = True
            except Exception:
                pass
        
        # Also check summary_df for all grids (including single-station grids)
        summary = result.get('summary_df')
        if summary is not None and isinstance(summary, pd.DataFrame) and not summary.empty:
            try:
                rh_rows = summary[summary['Variable'] == 'RH']
                if not rh_rows.empty:
                    for _, row in rh_rows.iterrows():
                        grid = row.get('Forecast_Grid')
                        key = (grid, init_month)
                        # Only add if we haven't seen this grid+init_month combo yet
                        # (aggregated_df takes precedence for multi-station grids)
                        if key not in seen_grids_per_init:
                            # Get station name(s) for this grid
                            stations = FORECAST_GRID_TO_STATIONS.get(grid, [])
                            rows.append({
                                'Forecast_Grid': grid,
                                'Stations_in_Grid': ', '.join(stations) if stations else row.get('Location', ''),
                                'N_Stations': len(stations) if stations else 1,
                                'Init_Month_Num': init_month,
                                'Init_Month': init_label,
                                'RH_Bias': row.get('Bias')
                            })
                            seen_grids_per_init[key] = True
            except Exception:
                pass

    if not rows:
        print("  WARNING: No RH bias data found in aggregated results.")
        return

    bias_df = pd.DataFrame(rows)

    # Get all unique grids to ensure all are included (even if missing data for some init months)
    all_grids = sorted(FORECAST_GRID_TO_STATIONS.keys())
    
    # Pivot to wide format: one row per grid, one column per initiation month
    wide = bias_df.pivot_table(
        index=['Forecast_Grid', 'Stations_in_Grid', 'N_Stations'],
        columns='Init_Month',
        values='RH_Bias',
        aggfunc='mean'
    )

    # Order columns by calendar month if possible
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    existing_cols = [c for c in month_order if c in wide.columns]
    wide = wide[existing_cols]

    wide = wide.reset_index()
    
    # Ensure all grids are included (add missing grids with NaN values)
    existing_grids = set(wide['Forecast_Grid'].values)
    missing_grids = [g for g in all_grids if g not in existing_grids]
    
    if missing_grids:
        for grid in missing_grids:
            stations = FORECAST_GRID_TO_STATIONS.get(grid, [])
            new_row = {
                'Forecast_Grid': grid,
                'Stations_in_Grid': ', '.join(stations) if stations else '',
                'N_Stations': len(stations) if stations else 1
            }
            # Add NaN for each init month column
            for col in existing_cols:
                new_row[col] = np.nan
            wide = pd.concat([wide, pd.DataFrame([new_row])], ignore_index=True)
    
    # Sort by grid
    wide = wide.sort_values('Forecast_Grid')

    print("\nRH Bias (Forecast - Station, %RH) by initiation month and grid:")
    print(wide.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save to CSV
    output_path = get_safe_output_path('RH_Bias_By_InitMonth_And_Grid.csv')
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        wide.to_csv(output_path, index=False)
        print(f"\nRH bias summary table saved to: {output_path}")
    except Exception as e:
        print(f"\nERROR: Could not save RH bias summary table: {e}")
        print(f"Path length: {len(output_path)} characters")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def process_single_forecast(forecast_file_path, initiation_month, growing_season_months, station_df):
    """
    Process a single forecast file with specified initiation month and growing season.
    
    Args:
        forecast_file_path: Path to forecast CSV file
        initiation_month: Month number when forecast was initiated (for labeling)
        growing_season_months: List of month numbers for growing season
        station_df: Station data DataFrame (already loaded)
    
    Returns:
        tuple: (summary_df, aggregated_df, threshold_results, optimal_thresholds)
    """
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    season_label = f"{month_names[growing_season_months[0]]}-{month_names[growing_season_months[-1]]}"
    
    print("\n" + "="*80)
    print(f"PROCESSING FORECAST INITIATED IN {month_names[initiation_month].upper()}")
    print(f"Growing Season: {season_label}")
    print("="*80)
    
    # Load forecast data
    forecast_df = load_forecast_data(forecast_file_path)
    
    # Get list of locations to process
    locations = sorted(STATION_INFO.keys())
    print(f"\nProcessing {len(locations)} locations: {locations}")
    
    # Create summary table (individual station comparisons)
    summary_df = create_summary_table(forecast_df, station_df, locations, growing_season_months, initiation_month)
    
    # Create aggregated comparison for grids with multiple stations
    aggregated_df = create_aggregated_comparison(forecast_df, station_df, growing_season_months, initiation_month)
    
    # Analyze precipitation thresholds
    threshold_results, optimal_thresholds = analyze_thresholds_all_locations(
        forecast_df, station_df, locations, growing_season_months, initiation_month
    )
    
    # Create supplementary table (wide format) only for April forecast (full season)
    supplementary_table = None
    if initiation_month == 4:  # Only for April forecast
        supplementary_table = create_supplementary_table_forecast(
            forecast_df, station_df, growing_season_months
        )
        # Analyze overall rainy day frequency across all grids
        analyze_overall_rainy_day_frequency(forecast_df, station_df, growing_season_months)
    
    return summary_df, aggregated_df, threshold_results, optimal_thresholds, supplementary_table


def main():
    """Main execution function."""
    print("="*80)
    print("SEASONAL FORECAST vs STATION DATA AGREEMENT ANALYSIS")
    print("="*80)
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Precipitation thresholds: {PRECIP_THRESHOLDS[0]} to {PRECIP_THRESHOLDS[-1]} mm (step: 0.5 mm)")
    print("="*80)
    
    # Check if forecast files are defined
    if not FORECAST_FILES:
        print("\nERROR: Please define FORECAST_FILES list in the script configuration.")
        return
    
    if not STATION_FILE:
        print("\nERROR: Please set STATION_FILE path in the script configuration.")
        return
    
    # Set output directory to first forecast file directory (same as input directory)
    global OUTPUT_DIR
    first_forecast_file = FORECAST_FILES[0][0] if FORECAST_FILES else None
    if first_forecast_file and first_forecast_file != r"UPDATE_WITH_YOUR_FORECAST_FILE_PATH\SF_042008-2025_daily.csv":
        OUTPUT_DIR = os.path.dirname(first_forecast_file) if os.path.dirname(first_forecast_file) else os.getcwd()
    else:
        # Fallback to station file directory if forecast files not set
        OUTPUT_DIR = os.path.dirname(STATION_FILE) if os.path.dirname(STATION_FILE) else os.getcwd()
    
    # Ensure output directory exists
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(OUTPUT_DIR):
            raise OSError(f"Output directory does not exist and could not be created: {OUTPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Output directory exists: {os.path.exists(OUTPUT_DIR)}")
    except Exception as e:
        print(f"ERROR: Could not create output directory: {e}")
        print(f"Attempted path: {OUTPUT_DIR}")
        return
    
    # Load station data (once for all forecasts)
    station_df = load_station_data(STATION_FILE)
    
    # Process each forecast file
    all_results = []
    for forecast_file_path, initiation_month, growing_season_months in FORECAST_FILES:
        if not os.path.exists(forecast_file_path):
            print(f"\nWARNING: Forecast file not found: {forecast_file_path}")
            print("  Skipping this forecast file.")
            continue
        
        try:
            summary_df, aggregated_df, threshold_results, optimal_thresholds, supplementary_table = process_single_forecast(
                forecast_file_path, initiation_month, growing_season_months, station_df
            )
            all_results.append({
                'initiation_month': initiation_month,
                'growing_season_months': growing_season_months,
                'summary_df': summary_df,
                'aggregated_df': aggregated_df,
                'threshold_results': threshold_results,
                'optimal_thresholds': optimal_thresholds,
                'supplementary_table': supplementary_table
            })
        except Exception as e:
            print(f"\nERROR: Failed to process forecast file {forecast_file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary of all analyses
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nProcessed {len(all_results)} forecast file(s):")
    for result in all_results:
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        init_month = result['initiation_month']
        season_months = result['growing_season_months']
        season_label = f"{month_names[season_months[0]]}-{month_names[season_months[-1]]}"
        print(f"  - Initiated {month_names[init_month]}: Growing season {season_label}")
    
    # Build and save RH bias summary across all initiation months and grids
    build_rh_bias_summary(all_results)

    print("\nGenerated files (for each forecast):")
    print("  - Summary_Table_Forecast_vs_Station_GrowingSeason_Init[Month].csv")
    print("  - Aggregated_Comparison_Forecast_vs_Station_Avg_GrowingSeason_Init[Month].csv")
    print("  - Precipitation_Threshold_Analysis_Forecast_vs_Station_ByGrid_Init[Month].csv")
    print("  - Optimal_Precipitation_Thresholds_ByGrid_Forecast_vs_Station_Init[Month].csv")
    print("  - Overall_Optimal_Precipitation_Threshold_Forecast_vs_Station_Init[Month].csv")
    print("\nGenerated files (April forecast only):")
    print("  - Supplementary_Table_Forecast_Evaluation_GrowingSeason_Only.csv")
    print("\nGenerated files (all forecasts combined):")
    print("  - RH_Bias_By_InitMonth_And_Grid.csv")


if __name__ == "__main__":
    main()

