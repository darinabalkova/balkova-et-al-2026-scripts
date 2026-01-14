#!/usr/bin/env python3
"""
5. Figure 1 Analysis Script - Updated for 10 Sites (2008-2025)

This script creates:
- Full time series plots for all 10 sites (2008-2025)
- Supplementary Table: Statistical comparison (r, RMSE, MAE, MB) for all sites
- Grid localization table
- Figure 1: Three panels per variable (T, RH, Precip) with scatterplots, histograms, and seasonal cycles

Data sources: Station, ERA5-Land, ERA5-Land* (filtered)
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# --- CONFIGURATION ---
# Output directory
# UPDATE_WITH_YOUR_OUTPUT_DIRECTORY
OUTPUT_DIR = r"UPDATE_WITH_YOUR_OUTPUT_DIRECTORY"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thresholds
HOT_DAY_THRESHOLD = 25
HIGH_HUMIDITY_THRESHOLD = 90
PRECIP_THRESHOLD_MM = 2.0  # Threshold for ERA5-Land precipitation filtering

# Growing season months (April-September)
GROWING_SEASON_MONTHS = [4, 5, 6, 7, 8, 9]

# Color scheme
COLORS = {
    'Station': 'black',
    'ERA5': '#90EE90',      # Light Green
    'ERA5*': '#008000',     # Dark Green
    'Forecast': '#2525ff'   # Blue for future use
}

# File paths
# UPDATE_WITH_YOUR_MERGED_DATA_FILE_PATH
MERGED_DATA_FILE = r"UPDATE_WITH_YOUR_MERGED_DATA_FILE_PATH"

# Station information (from the table provided)
STATION_INFO = {
    'Castel San Giovanni': {'ID': 1, 'Latitude': 45.06, 'Longitude': 9.43, 'ERA5_Land_grid': (45.1, 9.4), 'Seasonal_forecast_grid': (45, 9)},
    'Caorso': {'ID': 2, 'Latitude': 45.05, 'Longitude': 9.87, 'ERA5_Land_grid': (45.1, 9.9), 'Seasonal_forecast_grid': (45, 10)},
    'Roncopascolo': {'ID': 3, 'Latitude': 44.83, 'Longitude': 10.27, 'ERA5_Land_grid': (44.8, 10.3), 'Seasonal_forecast_grid': (44, 10)},
    'Luzzara': {'ID': 4, 'Latitude': 44.95, 'Longitude': 10.68, 'ERA5_Land_grid': (45.0, 10.7), 'Seasonal_forecast_grid': (45, 11)},
    'Mirandola': {'ID': 5, 'Latitude': 44.88, 'Longitude': 11.07, 'ERA5_Land_grid': (44.9, 11.1), 'Seasonal_forecast_grid': (45, 11)},
    'Castel Maggiore': {'ID': 6, 'Latitude': 44.58, 'Longitude': 11.36, 'ERA5_Land_grid': (44.6, 11.4), 'Seasonal_forecast_grid': (45, 11)},
    'Guarda Ferrarese': {'ID': 7, 'Latitude': 44.93, 'Longitude': 11.75, 'ERA5_Land_grid': (44.9, 11.8), 'Seasonal_forecast_grid': (45, 12)},
    'Ostellato': {'ID': 8, 'Latitude': 44.75, 'Longitude': 11.94, 'ERA5_Land_grid': (44.8, 11.9), 'Seasonal_forecast_grid': (45, 12)},
    'Medicina': {'ID': 9, 'Latitude': 44.47, 'Longitude': 11.63, 'ERA5_Land_grid': (44.5, 11.6), 'Seasonal_forecast_grid': (45, 12)},
    'Forli': {'ID': 10, 'Latitude': 44.22, 'Longitude': 12.04, 'ERA5_Land_grid': (44.2, 12.0), 'Seasonal_forecast_grid': (44, 12)},
}


# --- New Data Loading Function for Merged File ---
def load_and_prepare_merged_data(file_path):
    """
    Loads the single, merged CSV file containing Station, ERA5-Land, and ERA5-Land* data
    for multiple locations, and prepares it for analysis.
    
    Returns a dictionary with location names as keys and pivoted dataframes as values.
    """
    print("--- Loading and preparing MERGED data for all locations ---")
    try:
        df = pd.read_csv(file_path)
        print(f"  Initial shape: {df.shape}")
        print(f"  Initial columns: {df.columns.tolist()}")

        # 1. Standardize Date Column
        date_col = None
        for col in df.columns:
            if 'Date' in col and 'dd/mm/yyyy' in col:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("Date column not found")
        
        df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['Date'])

        # 2. Rename columns for simplicity
        rename_map = {}
        for col in df.columns:
            if 'T (°C)' in col or (col == 'T' and 'T' not in rename_map.values()):
                rename_map[col] = 'T'
            elif 'RH (%)' in col or (col == 'RH' and 'RH' not in rename_map.values()):
                rename_map[col] = 'RH'
            elif 'Rain (mm)' in col or 'Precip' in col or (col == 'P' and 'P' not in rename_map.values()):
                rename_map[col] = 'P'
        
        df.rename(columns=rename_map, inplace=True)

        # 3. Normalize DATA column values (handle "ERA5-Land" and "ERA5-Land*")
        if 'DATA' in df.columns:
            df['DATA'] = df['DATA'].astype(str).str.strip()
            # Normalize: "ERA5-Land" -> "ERA5", "ERA5-Land*" -> "ERA5*"
            df['DATA'] = df['DATA'].replace({
                'ERA5-Land': 'ERA5',
                'ERA5-Land*': 'ERA5*',
                'Station': 'Station'
            })

        # 4. Check if Location column exists
        if 'Location' not in df.columns:
            print("  WARNING:  Warning: 'Location' column not found. Assuming single location.")
            locations = ['Unknown']
            df['Location'] = 'Unknown'
        else:
            locations = sorted(df['Location'].unique())
            print(f"  Found {len(locations)} locations: {locations}")

        # 5. Process each location separately
        location_data = {}
        
        for location in locations:
            loc_df = df[df['Location'] == location].copy()
            
            if loc_df.empty:
                continue
            
            # Pivot the data for this location
            pivot_df = loc_df.pivot_table(
                index='Date',
                columns='DATA',
                values=['T', 'RH', 'P'],
                aggfunc='first'
            )
            
            # Flatten the multi-level column index
            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = [f'{val}_{source}' for val, source in pivot_df.columns]
            else:
                # Already flattened or single level
                pass
            
            # Normalize column names (handle variations)
            rename_cols = {}
            for col in pivot_df.columns:
                # Handle ERA5-Land -> ERA5
                if '_ERA5-Land' in col and '*' not in col:
                    rename_cols[col] = col.replace('_ERA5-Land', '_ERA5')
                # Handle ERA5-Land* -> ERA5*
                elif '_ERA5-Land*' in col:
                    rename_cols[col] = col.replace('_ERA5-Land*', '_ERA5*')
                # Keep Station and ERA5* as is (already normalized)
                # No need to rename if already correct
            
            if rename_cols:
                pivot_df.rename(columns=rename_cols, inplace=True)
            
            # Filter for the required date range (2008-2025)
            start_date = '2008-01-01'
            end_date = '2025-12-31'
            pivot_df = pivot_df.loc[start_date:end_date]
            
            # Debug for Castel San Giovanni
            if location == 'Castel San Giovanni':
                print(f"  DEBUG: Columns after pivot and normalization for {location}:")
                print(f"    {pivot_df.columns.tolist()}")
                print(f"    Shape: {pivot_df.shape}")
                if 'T_Station' in pivot_df.columns:
                    print(f"    T_Station non-null: {pivot_df['T_Station'].notna().sum()}")
                if 'T_ERA5' in pivot_df.columns:
                    print(f"    T_ERA5 non-null: {pivot_df['T_ERA5'].notna().sum()}")
            
            location_data[location] = pivot_df
        
        print(f"  Processed {len(location_data)} locations")
        print(f"  Date range: {start_date} to {end_date}")
        print("OK Merged data ready for all locations.\n")
        
        return location_data

    except Exception as e:
        print(f"ERROR: Error processing merged data from {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_grid_localization_table():
    """Create a table showing station locations and their corresponding grid cells."""
    print("\n" + "="*80)
    print("CREATING GRID LOCALIZATION TABLE")
    print("="*80)
    
    table_data = []
    for station_name, info in STATION_INFO.items():
        era5_lat, era5_lon = info['ERA5_Land_grid']
        sf_lat, sf_lon = info['Seasonal_forecast_grid']
        
        table_data.append({
            'Station name': station_name,
            'ID': info['ID'],
            'Latitude': info['Latitude'],
            'Longitude': info['Longitude'],
            'ERA5-Land grid cell': f"({era5_lat}, {era5_lon})",
            'Seasonal forecast grid cell': f"({sf_lat}, {sf_lon})"
        })
    
    table_df = pd.DataFrame(table_data)
    output_path = os.path.join(OUTPUT_DIR, 'Grid_Localization_Table.csv')
    table_df.to_csv(output_path, index=False)
    
    print(table_df.to_string(index=False))
    print(f"\nOK Grid localization table saved to {output_path}")
    return table_df

def calculate_metrics(aligned_data):
    """Calculate metrics on an already aligned dataframe."""
    metrics = {}
    
    # Check for the presence of ERA5* columns
    has_era5_star = any('ERA5*' in col for col in aligned_data.columns)
    
    for var in ['T', 'RH', 'P']:
        station_col = f'{var}_Station' # Adjusted to match the pivot output
        era5_col = f'{var}_ERA5'
        
        # Check if all required columns for the variable exist
        if not all(col in aligned_data.columns for col in [station_col, era5_col]):
            print(f"WARNING: Missing columns for variable '{var}'. Skipping metrics calculation.")
            continue
        
        station_vals = aligned_data[station_col]
        era5_vals = aligned_data[era5_col]
        
        # Calculate metrics for ERA5
        bias_era5 = np.mean(era5_vals - station_vals)
        rmse_era5 = np.sqrt(np.mean((era5_vals - station_vals)**2))
        corr_era5, _ = stats.pearsonr(station_vals, era5_vals)
        
        # Initialize metrics for ERA5* as None
        bias_era5_pp, rmse_era5_pp, corr_era5_pp = None, None, None
        
        # Calculate metrics for ERA5* if present
        if has_era5_star:
            era5_pp_col = f'{var}_ERA5*'
            if era5_pp_col in aligned_data.columns:
                era5_pp_vals = aligned_data[era5_pp_col]
                bias_era5_pp = np.mean(era5_pp_vals - station_vals)
                rmse_era5_pp = np.sqrt(np.mean((era5_pp_vals - station_vals)**2))
                corr_era5_pp, _ = stats.pearsonr(station_vals, era5_pp_vals)
            else:
                bias_era5_pp, rmse_era5_pp, corr_era5_pp = None, None, None
        else:
            bias_era5_pp, rmse_era5_pp, corr_era5_pp = None, None, None
        
        # Store metrics
        metrics[var] = {
            'ERA5': {'bias': bias_era5, 'rmse': rmse_era5, 'corr': corr_era5},
            'ERA5*': {'bias': bias_era5_pp, 'rmse': rmse_era5_pp, 'corr': corr_era5_pp},
            'Station': {}
        }
        
        # Special frequency metrics
        if var == 'T':
            metrics[var]['ERA5']['hot_days'] = np.sum(era5_vals > HOT_DAY_THRESHOLD)
            metrics[var]['Station']['hot_days'] = np.sum(station_vals > HOT_DAY_THRESHOLD)
            if has_era5_star and era5_pp_col in aligned_data.columns:
                metrics[var]['ERA5*']['hot_days'] = np.sum(era5_pp_vals > HOT_DAY_THRESHOLD)
            
        elif var == 'RH':
            metrics[var]['ERA5']['high_rh_days'] = np.sum(era5_vals > HIGH_HUMIDITY_THRESHOLD)
            metrics[var]['Station']['high_rh_days'] = np.sum(station_vals > HIGH_HUMIDITY_THRESHOLD)
            if has_era5_star and era5_pp_col in aligned_data.columns:
                 metrics[var]['ERA5*']['high_rh_days'] = np.sum(era5_pp_vals > HIGH_HUMIDITY_THRESHOLD)
            
        elif var == 'P':
            metrics[var]['ERA5']['rainy_days'] = np.sum(era5_vals > 0)
            metrics[var]['Station']['rainy_days'] = np.sum(station_vals > 0)
            if has_era5_star and era5_pp_col in aligned_data.columns:
                metrics[var]['ERA5*']['rainy_days'] = np.sum(era5_pp_vals > 0)
    
    return metrics

def create_supplementary_table(location_data):
    """
    Creates Supplementary Table matching the screenshot format:
    Statistical comparison of ERA5-Land meteorological variables (T, RH, P) 
    against station observations (2008-2025) across all sites.
    
    Format: r, RMSE, MAE, MB (Mean Bias) for each variable and location.
    Includes rainy day delta after applying 2mm threshold to ERA5-Land.
    """
    print("\n" + "="*80)
    print("CREATING SUPPLEMENTARY TABLE (Statistical Comparison - Full Season)")
    print("="*80)
    print("Format: Pearson correlation (r), RMSE, MAE, MB for T, RH, P")
    print("Period: 2008-2025 (all months, all years)")
    print("Includes: Rainy day delta (Station vs ERA5-Land with 2mm threshold)")
    print("="*80)
    
    # Debug: Print available locations
    print(f"\n  Locations found in data: {sorted(location_data.keys())}")
    print(f"  Locations in STATION_INFO: {sorted(STATION_INFO.keys())}")
    
    # Create a mapping function to match location names (case-insensitive, handle spaces and abbreviations)
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
        # Normalize common abbreviations
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
    
    table_data = []
    
    for location in sorted(location_data.keys()):
        aligned_data = location_data[location]
        
        # Filter to 2008-2025 (whole period, all months)
        aligned_data = aligned_data[(aligned_data.index.year >= 2008) & (aligned_data.index.year <= 2025)]
        
        if aligned_data.empty:
            continue
        
        # Get station info - try to find matching location name
        matched_location = find_matching_location(location, STATION_INFO.keys())
        if matched_location:
            station_id = STATION_INFO[matched_location]['ID']
            lat = STATION_INFO[matched_location]['Latitude']
            lon = STATION_INFO[matched_location]['Longitude']
            print(f"    Matched '{location}' to '{matched_location}' in STATION_INFO")
        else:
            station_id = 0
            lat = 0.0
            lon = 0.0
            print(f"    WARNING:  Warning: '{location}' not found in STATION_INFO")
        
        # Debug: Check data availability for Castel San Giovanni
        if location == 'Castel San Giovanni':
            print(f"    DEBUG for Castel San Giovanni:")
            print(f"      Data shape after filtering: {aligned_data.shape}")
            print(f"      Available columns: {aligned_data.columns.tolist()}")
            print(f"      Date range: {aligned_data.index.min()} to {aligned_data.index.max()}")
            if 'T_Station' in aligned_data.columns:
                print(f"      T_Station non-null: {aligned_data['T_Station'].notna().sum()}")
            if 'T_ERA5' in aligned_data.columns:
                print(f"      T_ERA5 non-null: {aligned_data['T_ERA5'].notna().sum()}")
        
        # Calculate metrics for each variable
        for var in ['T', 'RH', 'P']:
            station_col = f'{var}_Station'
            era5_col = f'{var}_ERA5'
            
            if station_col not in aligned_data.columns or era5_col not in aligned_data.columns:
                if location == 'Castel San Giovanni':
                    print(f"      WARNING:  Missing columns for {var}: station_col={station_col in aligned_data.columns}, era5_col={era5_col in aligned_data.columns}")
                continue
            
            # Remove NaN values for comparison
            valid_mask = aligned_data[station_col].notna() & aligned_data[era5_col].notna()
            station_vals = aligned_data.loc[valid_mask, station_col]
            era5_vals = aligned_data.loc[valid_mask, era5_col]
            
            if len(station_vals) == 0:
                if location == 'Castel San Giovanni':
                    print(f"      WARNING:  No valid data pairs for {var}: station non-null={aligned_data[station_col].notna().sum()}, era5 non-null={aligned_data[era5_col].notna().sum()}")
                continue
            
            # Calculate metrics
            try:
                r, _ = stats.pearsonr(station_vals, era5_vals)
                rmse = np.sqrt(np.mean((era5_vals - station_vals)**2))
                mae = np.mean(np.abs(era5_vals - station_vals))
                mb = np.mean(era5_vals - station_vals)  # Mean Bias
            except Exception as e:
                if location == 'Castel San Giovanni':
                    print(f"      WARNING:  Error calculating metrics for {var}: {e}")
                continue
            
            table_data.append({
                'Station name': location,
                'ID': station_id,
                'Latitude': lat,
                'Longitude': lon,
                'Variable': var,
                'r': round(r, 2),
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'MB': round(mb, 2)
            })
    
    if not table_data:
        print("WARNING: No data available for supplementary table")
        return None
    
    table_df = pd.DataFrame(table_data)
    
    # Reshape to match screenshot format: one row per station with columns for each variable
    # Include ALL locations from STATION_INFO, even if not in location_data
    reshaped_data = []
    
    # First, process locations that are in both STATION_INFO and location_data
    for location in sorted(location_data.keys()):
        # Try to find matching location in STATION_INFO
        matched_location = find_matching_location(location, STATION_INFO.keys())
        if not matched_location:
            print(f"    WARNING:  Skipping '{location}': not in STATION_INFO")
            continue
        
        station_id = STATION_INFO[matched_location]['ID']
        lat = STATION_INFO[matched_location]['Latitude']
        lon = STATION_INFO[matched_location]['Longitude']
        
        row = {
            'Station name': matched_location,  # Use the matched name from STATION_INFO
            'ID': station_id,
            'Latitude': lat,
            'Longitude': lon
        }
        
        for var in ['T', 'RH', 'P']:
            # Try to find data using the original location name from location_data
            var_data = table_df[(table_df['Station name'] == location) & (table_df['Variable'] == var)]
            if not var_data.empty:
                row[f'{var}_r'] = var_data.iloc[0]['r']
                row[f'{var}_RMSE'] = var_data.iloc[0]['RMSE']
                row[f'{var}_MAE'] = var_data.iloc[0]['MAE']
                row[f'{var}_MBE'] = var_data.iloc[0]['MB']
            else:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
        
        # Calculate rainy day delta for precipitation (full season)
        aligned_data_full = location_data[location]
        aligned_data_full = aligned_data_full[(aligned_data_full.index.year >= 2008) & (aligned_data_full.index.year <= 2025)]
        
        if 'P_Station' in aligned_data_full.columns and 'P_ERA5' in aligned_data_full.columns:
            # Get aligned precipitation data
            valid_mask_p = aligned_data_full['P_Station'].notna() & aligned_data_full['P_ERA5'].notna()
            station_p = aligned_data_full.loc[valid_mask_p, 'P_Station']
            era5_p = aligned_data_full.loc[valid_mask_p, 'P_ERA5']
            
            if len(station_p) > 0:
                # Station rainy days (threshold = 0mm)
                station_rainy = (station_p > 0).sum()
                
                # ERA5-Land rainy days with 2mm threshold
                era5_p_filtered = era5_p.copy()
                era5_p_filtered[era5_p_filtered < PRECIP_THRESHOLD_MM] = 0
                era5_rainy = (era5_p_filtered > 0).sum()
                
                # Delta (difference)
                delta_rainy_days = era5_rainy - station_rainy
                
                row['P_RainyDays_Station'] = station_rainy
                row['P_RainyDays_ERA5_2mm'] = era5_rainy
                row['P_Delta_RainyDays'] = delta_rainy_days
            else:
                row['P_RainyDays_Station'] = np.nan
                row['P_RainyDays_ERA5_2mm'] = np.nan
                row['P_Delta_RainyDays'] = np.nan
        else:
            row['P_RainyDays_Station'] = np.nan
            row['P_RainyDays_ERA5_2mm'] = np.nan
            row['P_Delta_RainyDays'] = np.nan
        
        reshaped_data.append(row)
    
    # Second, add any locations from STATION_INFO that weren't in location_data
    processed_locations = {find_matching_location(loc, STATION_INFO.keys()) 
                          for loc in location_data.keys() 
                          if find_matching_location(loc, STATION_INFO.keys())}
    
    for location in sorted(STATION_INFO.keys()):
        if location not in processed_locations:
            print(f"    WARNING:  Adding '{location}' from STATION_INFO (no data found in location_data)")
            station_id = STATION_INFO[location]['ID']
            lat = STATION_INFO[location]['Latitude']
            lon = STATION_INFO[location]['Longitude']
            
            row = {
                'Station name': location,
                'ID': station_id,
                'Latitude': lat,
                'Longitude': lon
            }
            
            for var in ['T', 'RH', 'P']:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
            
            # Add empty rainy day columns
            row['P_RainyDays_Station'] = np.nan
            row['P_RainyDays_ERA5_2mm'] = np.nan
            row['P_Delta_RainyDays'] = np.nan
            
            reshaped_data.append(row)
    
    final_table = pd.DataFrame(reshaped_data)
    
    # Reorder columns
    col_order = ['Station name', 'ID', 'Latitude', 'Longitude']
    for var in ['T', 'RH', 'P']:
        col_order.extend([f'{var}_r', f'{var}_RMSE', f'{var}_MAE', f'{var}_MBE'])
    # Add precipitation rainy day columns after P metrics
    col_order.extend(['P_RainyDays_Station', 'P_RainyDays_ERA5_2mm', 'P_Delta_RainyDays'])
    
    final_table = final_table.reindex(columns=col_order)
    
    output_path = os.path.join(OUTPUT_DIR, 'Supplementary_Table_Statistical_Comparison.csv')
    final_table.to_csv(output_path, index=False)
    
    print("\n" + final_table.to_string(index=False))
    print(f"\nOK Supplementary table saved to {output_path}")
    
    return final_table

def create_supplementary_table_growing_season_only(location_data):
    """
    Creates Supplementary Table with ALL metrics calculated using ONLY growing season months.
    Statistical comparison of ERA5-Land meteorological variables (T, RH, P) 
    against station observations (2008-2025) across all sites.
    
    Format: r, RMSE, MAE, MB (Mean Bias) for each variable and location.
    Period: Growing season months (April-September) across whole period 2008-2025
    """
    print("\n" + "="*80)
    print("CREATING SUPPLEMENTARY TABLE (Statistical Comparison - Growing Season Only)")
    print("="*80)
    print("Format: Pearson correlation (r), RMSE, MAE, MB for T, RH, P")
    print("Period: Growing season months (April-September) across whole period 2008-2025")
    print("Note: ALL metrics calculated using ONLY growing season months")
    print("Includes: Rainy day delta (Station vs ERA5-Land with 2mm threshold)")
    print("="*80)
    
    # Debug: Print available locations
    print(f"\n  Locations found in data: {sorted(location_data.keys())}")
    print(f"  Locations in STATION_INFO: {sorted(STATION_INFO.keys())}")
    
    # Create a mapping function to match location names (case-insensitive, handle spaces and abbreviations)
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
        # Normalize common abbreviations
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
    
    table_data = []
    
    for location in sorted(location_data.keys()):
        aligned_data = location_data[location]
        
        # Filter to growing season months ONLY (April-September)
        aligned_data = aligned_data[aligned_data.index.month.isin(GROWING_SEASON_MONTHS)]
        
        # Filter to 2008-2025 (whole period, but only growing season months)
        aligned_data = aligned_data[(aligned_data.index.year >= 2008) & (aligned_data.index.year <= 2025)]
        
        if aligned_data.empty:
            continue
        
        # Get station info - try to find matching location name
        matched_location = find_matching_location(location, STATION_INFO.keys())
        if matched_location:
            station_id = STATION_INFO[matched_location]['ID']
            lat = STATION_INFO[matched_location]['Latitude']
            lon = STATION_INFO[matched_location]['Longitude']
            print(f"    Matched '{location}' to '{matched_location}' in STATION_INFO")
        else:
            station_id = 0
            lat = 0.0
            lon = 0.0
            print(f"    WARNING:  Warning: '{location}' not found in STATION_INFO")
        
        # Calculate metrics for each variable
        for var in ['T', 'RH', 'P']:
            station_col = f'{var}_Station'
            era5_col = f'{var}_ERA5'
            
            if station_col not in aligned_data.columns or era5_col not in aligned_data.columns:
                continue
            
            # Remove NaN values for comparison
            valid_mask = aligned_data[station_col].notna() & aligned_data[era5_col].notna()
            station_vals = aligned_data.loc[valid_mask, station_col]
            era5_vals = aligned_data.loc[valid_mask, era5_col]
            
            if len(station_vals) == 0:
                continue
            
            # Calculate metrics
            try:
                r, _ = stats.pearsonr(station_vals, era5_vals)
                rmse = np.sqrt(np.mean((era5_vals - station_vals)**2))
                mae = np.mean(np.abs(era5_vals - station_vals))
                mb = np.mean(era5_vals - station_vals)  # Mean Bias
            except:
                continue
            
            table_data.append({
                'Station name': location,
                'ID': station_id,
                'Latitude': lat,
                'Longitude': lon,
                'Variable': var,
                'r': round(r, 2),
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'MB': round(mb, 2)
            })
    
    if not table_data:
        print("WARNING: No data available for supplementary table (growing season only)")
        return None
    
    table_df = pd.DataFrame(table_data)
    
    # Reshape to match screenshot format: one row per station with columns for each variable
    # Include ALL locations from STATION_INFO, even if not in location_data
    reshaped_data = []
    
    # First, process locations that are in both STATION_INFO and location_data
    for location in sorted(location_data.keys()):
        # Try to find matching location in STATION_INFO
        matched_location = find_matching_location(location, STATION_INFO.keys())
        if not matched_location:
            print(f"    WARNING:  Skipping '{location}': not in STATION_INFO")
            continue
        
        station_id = STATION_INFO[matched_location]['ID']
        lat = STATION_INFO[matched_location]['Latitude']
        lon = STATION_INFO[matched_location]['Longitude']
        
        row = {
            'Station name': matched_location,  # Use the matched name from STATION_INFO
            'ID': station_id,
            'Latitude': lat,
            'Longitude': lon
        }
        
        for var in ['T', 'RH', 'P']:
            # Try to find data using the original location name from location_data
            var_data = table_df[(table_df['Station name'] == location) & (table_df['Variable'] == var)]
            if not var_data.empty:
                row[f'{var}_r'] = var_data.iloc[0]['r']
                row[f'{var}_RMSE'] = var_data.iloc[0]['RMSE']
                row[f'{var}_MAE'] = var_data.iloc[0]['MAE']
                row[f'{var}_MBE'] = var_data.iloc[0]['MB']
            else:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
        
        # Calculate rainy day delta for precipitation (growing season only)
        aligned_data_gs = location_data[location]
        # Filter to growing season months ONLY (April-September)
        aligned_data_gs = aligned_data_gs[aligned_data_gs.index.month.isin(GROWING_SEASON_MONTHS)]
        # Filter to 2008-2025 (whole period, but only growing season months)
        aligned_data_gs = aligned_data_gs[(aligned_data_gs.index.year >= 2008) & (aligned_data_gs.index.year <= 2025)]
        
        if 'P_Station' in aligned_data_gs.columns and 'P_ERA5' in aligned_data_gs.columns:
            # Get aligned precipitation data
            valid_mask_p = aligned_data_gs['P_Station'].notna() & aligned_data_gs['P_ERA5'].notna()
            station_p = aligned_data_gs.loc[valid_mask_p, 'P_Station']
            era5_p = aligned_data_gs.loc[valid_mask_p, 'P_ERA5']
            
            if len(station_p) > 0:
                # Station rainy days (threshold = 0mm)
                station_rainy = (station_p > 0).sum()
                
                # ERA5-Land rainy days with 2mm threshold
                era5_p_filtered = era5_p.copy()
                era5_p_filtered[era5_p_filtered < PRECIP_THRESHOLD_MM] = 0
                era5_rainy = (era5_p_filtered > 0).sum()
                
                # Delta (difference)
                delta_rainy_days = era5_rainy - station_rainy
                
                row['P_RainyDays_Station'] = station_rainy
                row['P_RainyDays_ERA5_2mm'] = era5_rainy
                row['P_Delta_RainyDays'] = delta_rainy_days
            else:
                row['P_RainyDays_Station'] = np.nan
                row['P_RainyDays_ERA5_2mm'] = np.nan
                row['P_Delta_RainyDays'] = np.nan
        else:
            row['P_RainyDays_Station'] = np.nan
            row['P_RainyDays_ERA5_2mm'] = np.nan
            row['P_Delta_RainyDays'] = np.nan
        
        reshaped_data.append(row)
    
    # Second, add any locations from STATION_INFO that weren't in location_data
    processed_locations = {find_matching_location(loc, STATION_INFO.keys()) 
                          for loc in location_data.keys() 
                          if find_matching_location(loc, STATION_INFO.keys())}
    
    for location in sorted(STATION_INFO.keys()):
        if location not in processed_locations:
            print(f"    WARNING:  Adding '{location}' from STATION_INFO (no data found in location_data)")
            station_id = STATION_INFO[location]['ID']
            lat = STATION_INFO[location]['Latitude']
            lon = STATION_INFO[location]['Longitude']
            
            row = {
                'Station name': location,
                'ID': station_id,
                'Latitude': lat,
                'Longitude': lon
            }
            
            for var in ['T', 'RH', 'P']:
                row[f'{var}_r'] = np.nan
                row[f'{var}_RMSE'] = np.nan
                row[f'{var}_MAE'] = np.nan
                row[f'{var}_MBE'] = np.nan
            
            # Add empty rainy day columns
            row['P_RainyDays_Station'] = np.nan
            row['P_RainyDays_ERA5_2mm'] = np.nan
            row['P_Delta_RainyDays'] = np.nan
            
            reshaped_data.append(row)
    
    final_table = pd.DataFrame(reshaped_data)
    
    # Reorder columns
    col_order = ['Station name', 'ID', 'Latitude', 'Longitude']
    for var in ['T', 'RH', 'P']:
        col_order.extend([f'{var}_r', f'{var}_RMSE', f'{var}_MAE', f'{var}_MBE'])
    # Add precipitation rainy day columns after P metrics
    col_order.extend(['P_RainyDays_Station', 'P_RainyDays_ERA5_2mm', 'P_Delta_RainyDays'])
    
    final_table = final_table.reindex(columns=col_order)
    
    output_path = os.path.join(OUTPUT_DIR, 'Supplementary_Table_Statistical_Comparison_GrowingSeason_Only.csv')
    final_table.to_csv(output_path, index=False)
    
    print("\n" + final_table.to_string(index=False))
    print(f"\nOK Supplementary table (growing season only) saved to {output_path}")
    
    return final_table

def create_full_timeseries_all_locations(location_data):
    """
    Create full time series plots (2008-2025) for all 10 locations.
    One figure per location with 3 subplots (T, RH, P).
    """
    print("\n" + "="*80)
    print("CREATING FULL TIME SERIES PLOTS FOR ALL LOCATIONS (2008-2025)")
    print("="*80)
    
    variables = [('T', 'Temperature (°C)'), ('RH', 'Relative Humidity (%)'), ('P', 'Precipitation (mm)')]
    
    for location in sorted(location_data.keys()):
        aligned_data = location_data[location]
        
        if aligned_data.empty:
            print(f"  WARNING:  Skipping {location}: No data available")
            continue
        
        print(f"  Creating time series for {location}...")
        
        fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
        fig.suptitle(f'Full Time Series 2008-2025: {location}', fontsize=16, fontweight='bold')
        
        for i, (var_code, var_name) in enumerate(variables):
            station_col = f'{var_code}_Station'
            era5_col = f'{var_code}_ERA5'
            era5_pp_col = f'{var_code}_ERA5*'
            
            if station_col in aligned_data.columns:
                axes[i].plot(aligned_data.index, aligned_data[station_col], 
                           color=COLORS['Station'], label='Station', lw=0.7, alpha=0.8)
            
            if era5_col in aligned_data.columns:
                axes[i].plot(aligned_data.index, aligned_data[era5_col], 
                           color=COLORS['ERA5'], label='ERA5-Land', alpha=0.7, lw=0.7)
            
            if era5_pp_col in aligned_data.columns:
                axes[i].plot(aligned_data.index, aligned_data[era5_pp_col], 
                           color=COLORS['ERA5*'], label='ERA5-Land*', alpha=0.7, lw=0.7)
            
            axes[i].set_ylabel(var_name, fontsize=12)
            axes[i].legend(loc='upper right', fontsize=10)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Clean location name for filename
        location_clean = location.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(OUTPUT_DIR, f'FullTimeSeries_2008-2025_{location_clean}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {output_path}")
    
    print("\nOK Full time series plots created for all locations")

def create_final_figure1(aligned_data, year, location=None):
    """
    Creates the final 3x3 Figure 1, combining climatology and correlation,
    and focusing on the Station vs ERA5 comparison.
    
    Args:
        aligned_data: DataFrame with aligned Station and ERA5 data
        year: Year to plot
        location: Location name (optional, for title and filename)
    """
    if location is None:
        location = "Location"  # Default if not provided
    
    print(f"      Creating Final Figure 1 for {location} - {year}...")
    plt.close('all')

    data_year = aligned_data[aligned_data.index.year == year]
    if data_year.empty:
        print(f"      ERROR: Missing data for {year}. Skipping.")
        return

    variables = [('T', 'Temperature (°C)'), ('RH', 'Relative Humidity (%)'), ('P', 'Precipitation (mm)')]
    fig, axes = plt.subplots(3, 3, figsize=(22, 15))
    fig.suptitle(f'{location} {year}', fontsize=22, fontweight='bold')

    subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)']
    label_idx = 0

    for col in range(3):
        for row in range(3):
            ax = axes[row, col]
            ax.text(-0.1, 1.05, subplot_labels[label_idx], transform=ax.transAxes, 
                    fontsize=16, fontweight='bold', va='top', ha='right')
            label_idx += 1

    for row, (var_code, var_name) in enumerate(variables):
        station_col = f'{var_code}_Station'
        era5_col = f'{var_code}_ERA5'
        era5_pp_col = f'{var_code}_ERA5*' # ERA5* for post-processed
        
        station_vals = data_year[station_col]
        era5_vals = data_year[era5_col]
        
        # Filter out NaN values for correlation calculation
        valid_mask = station_vals.notna() & era5_vals.notna()
        station_vals_clean = station_vals[valid_mask]
        era5_vals_clean = era5_vals[valid_mask]
        
        # --- Column 1: Monthly Climatology with Inset Correlation Scatterplot ---
        ax1 = axes[row, 0]
        months = np.arange(1, 13)

        if var_code == 'P':
            # Bar chart for Precipitation Climatology
            monthly_station = data_year.resample('M').sum()[station_col]
            monthly_era5 = data_year.resample('M').sum()[era5_col]
            bar_width = 0.35
            r1 = np.arange(12)
            r2 = [x + bar_width for x in r1]
            ax1.bar(r1, monthly_station, width=bar_width, color=COLORS['Station'], edgecolor='grey')
            ax1.bar(r2, monthly_era5, width=bar_width, color=COLORS['ERA5'], edgecolor='grey')
            ax1.set_xticks([r + bar_width/2 for r in r1])
            ax1.set_ylabel('Total Precipitation (mm)', fontsize=12)
        else:
            # Line chart for T and RH Climatology
            monthly_station = data_year.resample('M').mean()[station_col]
            monthly_era5 = data_year.resample('M').mean()[era5_col]
            ax1.plot(months, monthly_station, 'o-', color=COLORS['Station'])
            ax1.plot(months, monthly_era5, 's-', color=COLORS['ERA5'])
            ax1.set_ylabel(var_name, fontsize=12)
            ax1.set_xticks(months)
        
        if row < 2:
            ax1.set_xticklabels([])
        else:
            ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45, fontsize=12)
        
        ax1.tick_params(axis='y', labelsize=12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Inset Scatterplot for Correlation (slightly larger, grey color)
        inset_ax = ax1.inset_axes([0.05, 0.7, 0.35, 0.35]) # Slightly larger: 0.35x0.35 instead of 0.3x0.3
        inset_ax.set_facecolor('none') # Transparent background patch
        
        # Only plot and calculate correlation if we have valid data
        if len(station_vals_clean) > 0 and len(era5_vals_clean) > 0:
            inset_ax.scatter(station_vals_clean, era5_vals_clean, alpha=0.5, color='#666666', s=18)  # Grey color to distinguish from ERA5 green
            
            # Add 1:1 line
            if len(station_vals_clean) > 0:
                min_val = min(station_vals_clean.min(), era5_vals_clean.min())
                max_val = max(station_vals_clean.max(), era5_vals_clean.max())
                inset_ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
            
            # Calculate correlation only if we have enough valid pairs and variance
            if len(station_vals_clean) >= 2:
                try:
                    corr, _ = stats.pearsonr(station_vals_clean, era5_vals_clean)
                    if np.isnan(corr):
                        # Check if values are constant (no variance)
                        if station_vals_clean.std() == 0 or era5_vals_clean.std() == 0:
                            corr = 1.0 if np.allclose(station_vals_clean, era5_vals_clean) else np.nan
                        else:
                            corr = np.nan
                except:
                    corr = np.nan
            else:
                corr = np.nan
        else:
            corr = np.nan
        # Display correlation (handle NaN case)
        if not np.isnan(corr):
            inset_ax.text(0.05, 0.95, f'R = {corr:.3f}', transform=inset_ax.transAxes, fontsize=12, verticalalignment='top', fontweight='bold')
        else:
            inset_ax.text(0.05, 0.95, 'R = N/A', transform=inset_ax.transAxes, fontsize=12, verticalalignment='top', fontweight='bold', color='red')
        
        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['right'].set_visible(False)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

        # --- Column 2: Daily Bias Distribution ---
        ax2 = axes[row, 1]
        # Use cleaned values for bias calculation (only where both are valid)
        if len(station_vals_clean) > 0 and len(era5_vals_clean) > 0:
            bias = era5_vals_clean - station_vals_clean
        else:
            bias = pd.Series(dtype=float)
        
        # For precipitation, focus on -10 to 10 range for better legibility
        if var_code == 'P':
            sns.histplot(bias, bins=50, alpha=0.7, color=COLORS['ERA5'], ax=ax2, stat='density', kde=False)
            sns.kdeplot(bias, color='grey', lw=2, ax=ax2)
            ax2.set_xlim(-10, 10)  # Focus on -10 to 10 range
        else:
            sns.histplot(bias, bins=30, alpha=0.7, color=COLORS['ERA5'], ax=ax2, stat='density', kde=False)
            sns.kdeplot(bias, color='grey', lw=2, ax=ax2)
        
        ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel(f'{var_code} Bias (ERA5 - Station)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # --- Column 3: Frequency Analysis ---
        ax3 = axes[row, 2]
        bar_width = 0.25
        full_month_index = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='M')
        
        if var_code == 'T':
            freq_st = data_year[lambda x: x[station_col] > HOT_DAY_THRESHOLD][station_col].resample('M').count().reindex(full_month_index, fill_value=0)
            freq_e5 = data_year[lambda x: x[era5_col] > HOT_DAY_THRESHOLD][era5_col].resample('M').count().reindex(full_month_index, fill_value=0)
            df = pd.DataFrame({'Station': freq_st, 'ERA5': freq_e5})
            ax3.set_ylabel(f'Number of hot days (>{HOT_DAY_THRESHOLD}°C)', fontsize=12)
            ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        elif var_code == 'RH':
            freq_st = data_year[lambda x: x[station_col] > HIGH_HUMIDITY_THRESHOLD][station_col].resample('M').count().reindex(full_month_index, fill_value=0)
            freq_e5 = data_year[lambda x: x[era5_col] > HIGH_HUMIDITY_THRESHOLD][era5_col].resample('M').count().reindex(full_month_index, fill_value=0)
            df = pd.DataFrame({'Station': freq_st, 'ERA5': freq_e5})
            ax3.set_ylabel(f'Number of high RH days (>{HIGH_HUMIDITY_THRESHOLD}%)', fontsize=12)
            ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
        elif var_code == 'P':
            freq_st = data_year[lambda x: x[station_col] > 0][station_col].resample('M').count().reindex(full_month_index, fill_value=0)
            freq_e5 = data_year[lambda x: x[era5_col] > 0][era5_col].resample('M').count().reindex(full_month_index, fill_value=0)
            
            df_data = {'Station': freq_st, 'ERA5': freq_e5}
            # Check if the ERA5* column exists before using it
            if era5_pp_col in data_year.columns:
                freq_e5p = data_year[lambda x: x[era5_pp_col] > 0][era5_pp_col].resample('M').count().reindex(full_month_index, fill_value=0)
                df_data['ERA5*'] = freq_e5p
            
            df = pd.DataFrame(df_data)
            ax3.set_ylabel('Number of rainy days (>0mm)', fontsize=12)

        r1 = np.arange(12)
        r2 = [x + bar_width for x in r1]
        ax3.bar(r1, df['Station'], color=COLORS['Station'], width=bar_width, edgecolor='grey', label='Station')
        ax3.bar(r2, df['ERA5'], color=COLORS['ERA5'], width=bar_width, edgecolor='grey', label='ERA5')
        
        if var_code == 'P' and 'ERA5*' in df.columns:
            r3 = [x + bar_width for x in r2]
            ax3.bar(r3, df['ERA5*'], color=COLORS['ERA5*'], width=bar_width, edgecolor='grey', label='ERA5*')
            ax3.set_xticks([r + bar_width for r in range(12)])
        else:
            ax3.set_xticks([r + bar_width/2 for r in range(12)])
        
        if row < 2:
            ax3.set_xticklabels([])
        else:
            ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45, fontsize=12)
        
        ax3.tick_params(axis='y', labelsize=12)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

    # Create a single, master legend for the entire figure
    legend_elements = [
        Patch(facecolor=COLORS['Station'], edgecolor='grey', label='Station'),
        Patch(facecolor=COLORS['ERA5'], edgecolor='grey', label='ERA5'),
        Patch(facecolor=COLORS['ERA5*'], edgecolor='grey', label='ERA5*')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.93), fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Clean location name for filename
    location_clean = location.replace(' ', '_').replace('/', '_') if location else "Location"
    output_path = os.path.join(OUTPUT_DIR, f'Figure1_Final_Analysis_{location_clean}_{year}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show() to avoid blocking
    print(f"      OK Saved: {output_path}")

def main():
    """Main function to run the complete analysis for all 10 sites."""
    print("="*80)
    print("FIGURE 1 ANALYSIS - 10 SITES (2008-2025)")
    print("="*80)
    
    # Set output directory to merged data file directory
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.dirname(MERGED_DATA_FILE) if os.path.dirname(MERGED_DATA_FILE) else os.getcwd()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare the merged dataset for all locations
    location_data = load_and_prepare_merged_data(MERGED_DATA_FILE)
    
    if location_data is None or len(location_data) == 0:
        print("ERROR: Merged data file could not be processed. Halting analysis.")
        return
    
    # --- RUN THE ANALYSIS ---
    
    # 1. Create grid localization table
    create_grid_localization_table()
    
    # 2. Create supplementary table (statistical comparison)
    create_supplementary_table(location_data)
    
    # 2b. Create supplementary table with growing season only (all metrics using only growing season months)
    create_supplementary_table_growing_season_only(location_data)
    
    # 3. Create full time series plots for all locations
    create_full_timeseries_all_locations(location_data)
    
    # 4. Create Figure 1 final analysis for each location and available years
    print("\n" + "="*80)
    print("CREATING FIGURE 1 FINAL ANALYSIS FOR ALL LOCATIONS")
    print("="*80)
    
    # Define years to create figures for (or use all available years)
    years_to_plot = [2010, 2015, 2018, 2019]  # You can modify this list or use all available years
    
    for location in sorted(location_data.keys()):
        if location not in location_data:
            continue
        
        aligned_data = location_data[location]
        if aligned_data.empty:
            print(f"  WARNING:  Skipping {location}: No data available")
            continue
        
        # Get available years for this location
        available_years = sorted(aligned_data.index.year.unique())
        print(f"\n  Processing {location}...")
        print(f"    Available years: {available_years}")
        
        # Use intersection of requested years and available years
        years_for_location = [y for y in years_to_plot if y in available_years]
        
        if not years_for_location:
            print(f"    WARNING:  No matching years found for {location}")
            continue
        
        for year in years_for_location:
            print(f"    Creating Figure 1 for {year}...")
            create_final_figure1(aligned_data, year, location)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"All files saved in: {OUTPUT_DIR}")
    print("="*80)
    print("\nGenerated outputs:")
    print("  1. Grid_Localization_Table.csv - Station locations and grid cells")
    print("  2. Supplementary_Table_Statistical_Comparison.csv - Statistical metrics (Full Season)")
    print("      - Metrics: r, RMSE, MAE, MB for T, RH, P")
    print("      - Uses ALL months for all metrics (T, RH, P)")
    print("      - Period: 2008-2025 (all months, all years)")
    print("      - Includes: Rainy day delta (P_RainyDays_Station, P_RainyDays_ERA5_2mm, P_Delta_RainyDays)")
    print("  3. Supplementary_Table_Statistical_Comparison_GrowingSeason_Only.csv - Statistical metrics (Growing Season)")
    print("      - Metrics: r, RMSE, MAE, MB for T, RH, P")
    print("      - Uses ONLY growing season months (April-September) for ALL metrics (T, RH, P)")
    print("      - Period: 2008-2025 (whole period, but only growing season months)")
    print("      - Includes: Rainy day delta (P_RainyDays_Station, P_RainyDays_ERA5_2mm, P_Delta_RainyDays)")
    print("  4. FullTimeSeries_2008-2025_[Location].png - Time series plots for each location")
    print("  5. Figure1_Final_Analysis_[Location]_[Year].png - Final Figure 1 for each location and year")
    print("="*80)

if __name__ == "__main__":
    main()

