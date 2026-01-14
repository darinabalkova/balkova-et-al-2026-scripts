#!/usr/bin/env python3
"""
1. ERA5-Land Daily Extraction

This script extracts daily meteorological data from ERA5-Land NetCDF files for specific grid cells.
It automatically detects all available years in the input directory and processes all of them.

Inputs:
- ERA5-Land NetCDF files (from Copernicus CDS)
- Grid cell coordinates (latitude, longitude) for stations

Outputs:
- CSV file with daily data (T, RH, P) for all available years and locations

File Structure Requirements:
- Separate NetCDF files per variable per year
- Expected file naming pattern: ERA5-Land_[variable]_[region]_[year].nc
  where [variable] is one of:
    - pr (precipitation)
    - relative_humidity (or r2m)
    - tas (temperature)
- Files should contain CF-compliant units in attributes

Expected Units (ERA5-Land standard):
- Temperature: Kelvin (K) → converted to Celsius (°C)
- Relative Humidity: Fraction (0-1) → converted to Percentage (%)
- Precipitation: Meters (m) → converted to Millimeters (mm)
  OR kg m-2 s-1 (rate) → converted to mm total

The script automatically detects units from NetCDF attributes and applies appropriate conversions.
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import re
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory for ERA5-Land NetCDF files
# UPDATE: Replace with your directory path containing ERA5-Land NetCDF files
era5_base_dir = r"UPDATE_WITH_YOUR_ERA5_LAND_NETCDF_DIRECTORY"

# Output directory (same as input file directory)
output_dir = era5_base_dir

# Output CSV file
output_file = os.path.join(output_dir, "ERA5_Land_Daily_Extracted_All_Years.csv")

# ============================================================================
# STATION CONFIGURATION
# ============================================================================
# Define exact ERA5-Land grid cells to extract
# Format: (Grid_Latitude, Grid_Longitude) - ERA5-Land grid cell coordinates
era5_grid_cells = [
    (45.1, 9.4),   # Castel San Giovanni
    (45.1, 9.9),   # Caorso
    (44.8, 10.3),  # Roncopascolo
    (45.0, 10.7),  # Luzzara 
    (44.9, 11.1),  # Mirandola
    (44.6, 11.4),  # Castel Maggiore
    (44.9, 11.8),  # Guarda Ferrarese
    (44.8, 11.9),  # Ostellato
    (44.5, 11.6),  # Medicina
    (44.2, 12.0),  # Forli
]

# Station names for reference
station_names = [
    "Castel San Giovanni",
    "Caorso",
    "Roncopascolo",
    "Luzzara",
    "Mirandola",
    "Castel Maggiore",
    "Guarda Ferrarese",
    "Ostellato",
    "Medicina",
    "Forli",
]

# Station coordinates (actual station locations, not grid cells)
# Format: (Station_Latitude, Station_Longitude) - Actual station coordinates
station_coords = [
    (45.06, 9.43),   # Castel San Giovanni
    (45.05, 9.87),   # Caorso
    (44.83, 10.27),  # Roncopascolo
    (44.95, 10.68),  # Luzzara
    (44.88, 11.07),  # Mirandola
    (44.58, 11.36),  # Castel Maggiore
    (44.93, 11.75),  # Guarda Ferrarese
    (44.75, 11.94),  # Ostellato
    (44.47, 11.63),  # Medicina
    (44.22, 12.04),  # Forli
]

# Verify that grid_cells, station_names, and station_coords have the same length
if not (len(era5_grid_cells) == len(station_names) == len(station_coords)):
    raise ValueError("ERROR: era5_grid_cells, station_names, and station_coords must have the same length")

# ============================================================================
# AUTO-DETECT AVAILABLE YEARS FROM FILES
# ============================================================================
print("=" * 80)
print("STEP 1: Detecting available years from NetCDF files...")
print("=" * 80)

# Check if directory exists
if not os.path.exists(era5_base_dir):
    print(f"ERROR: Directory not found: {era5_base_dir}")
    print("Please update 'era5_base_dir' with your actual directory path.")
    exit(1)

# Scan directory for NetCDF files and extract years
# Expected file naming pattern: ERA5-Land_[variable]_[region]_[year].nc
# We'll look for files with 'pr', 'relative_humidity', or 'tas' in the name
all_years = set()

# Look for precipitation files (pr)
pr_files = glob.glob(os.path.join(era5_base_dir, "*pr*.nc"))
for file in pr_files:
    # Extract year from filename (look for 4-digit year)
    year_match = re.search(r'(\d{4})', os.path.basename(file))
    if year_match:
        all_years.add(int(year_match.group(1)))

# Look for relative humidity files
rh_files = glob.glob(os.path.join(era5_base_dir, "*relative_humidity*.nc"))
for file in rh_files:
    year_match = re.search(r'(\d{4})', os.path.basename(file))
    if year_match:
        all_years.add(int(year_match.group(1)))

# Look for temperature files (tas)
tas_files = glob.glob(os.path.join(era5_base_dir, "*tas*.nc"))
for file in tas_files:
    year_match = re.search(r'(\d{4})', os.path.basename(file))
    if year_match:
        all_years.add(int(year_match.group(1)))

if not all_years:
    print(f"ERROR: No NetCDF files found in directory: {era5_base_dir}")
    print("Expected file naming pattern: ERA5-Land_[variable]_[region]_[year].nc")
    print("Where [variable] is one of: pr, relative_humidity, tas")
    exit(1)

# Sort years
years = sorted(list(all_years))
print(f"\nDetected {len(years)} years: {years[0]} to {years[-1]}")
print(f"Years to process: {years}")

# ============================================================================
# CHECK NETCDF FILE STRUCTURE (using first year as reference)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Checking NetCDF file structure...")
print("=" * 80)

# Find files for the first year to check structure
# Try to find files matching the pattern
first_year = years[0]
pr_file = None
rh_file = None
tas_file = None

# Find precipitation file for first year
for file in pr_files:
    if str(first_year) in os.path.basename(file):
        pr_file = file
        break

# Find relative humidity file for first year
for file in rh_files:
    if str(first_year) in os.path.basename(file):
        rh_file = file
        break

# Find temperature file for first year
for file in tas_files:
    if str(first_year) in os.path.basename(file):
        tas_file = file
        break

# If pattern matching failed, try standard pattern
if not all([pr_file, rh_file, tas_file]):
    # Try standard naming pattern
    pr_file = os.path.join(era5_base_dir, f"ERA5-Land_pr_Northern_Italy_{first_year}.nc")
    rh_file = os.path.join(era5_base_dir, f"ERA5-Land_relative_humidity_Northern_Italy_{first_year}.nc")
    tas_file = os.path.join(era5_base_dir, f"ERA5-Land_tas_Northern_Italy_{first_year}.nc")

if not all(os.path.exists(f) for f in [pr_file, rh_file, tas_file]):
    print(f"ERROR: One or more files not found for year {first_year}")
    print(f"  Looking for: {pr_file}")
    print(f"  Looking for: {rh_file}")
    print(f"  Looking for: {tas_file}")
    exit(1)

# Open first year's files to check structure
ds_temp = xr.open_dataset(tas_file, engine="h5netcdf", decode_timedelta=False)
ds_rh = xr.open_dataset(rh_file, engine="h5netcdf", decode_timedelta=False)
ds_pr = xr.open_dataset(pr_file, engine="h5netcdf", decode_timedelta=False)

print(f"\nChecking files for year {first_year}...")
print(f"\nAvailable variables in temperature file: {list(ds_temp.variables.keys())}")
print(f"Available variables in RH file: {list(ds_rh.variables.keys())}")
print(f"Available variables in precipitation file: {list(ds_pr.variables.keys())}")
print(f"\nDimensions: {dict(ds_temp.dims)}")
print(f"\nCoordinates:")
for coord in ds_temp.coords:
    print(f"  {coord}: {ds_temp.coords[coord].values[:5]}... (shape: {ds_temp.coords[coord].shape})")

# Check time dimension
if 'time' in ds_temp.dims:
    time_values = ds_temp.time.values
    print(f"\nTime dimension:")
    print(f"  First time: {time_values[0]}")
    print(f"  Last time: {time_values[-1]}")
    print(f"  Total timesteps: {len(time_values)}")
    
    # Check if hourly or daily
    if len(time_values) > 1:
        time_diff = pd.to_datetime(time_values[1]) - pd.to_datetime(time_values[0])
        if time_diff.total_seconds() == 3600:  # 1 hour
            is_hourly = True
            print(f"  WARNING: Data is HOURLY (time difference: {time_diff})")
        elif time_diff.total_seconds() == 86400:  # 1 day
            is_hourly = False
            print(f"  Data is DAILY (time difference: {time_diff})")
        else:
            is_hourly = None
            print(f"  WARNING: Time difference: {time_diff} (check manually)")
    else:
        is_hourly = None
        print(f"  WARNING: Only one timestep found")
else:
    is_hourly = None
    print(f"\nWARNING: No 'time' dimension found")

# Check available variables (files are separated by variable)
# Temperature file should have 'tas'
# Precipitation file should have 'pr'
# Relative humidity file should have relative humidity variable
temp_var = None
rh_var = None
precip_var = None

# Initialize unit detection variables
temp_units_kelvin = None
rh_units_fraction = None
precip_units_meters = None
precip_units_rate = None  # For kg m-2 s-1 (rate-based precipitation)

print(f"\n{'='*80}")
print("Checking variable names and units...")
print(f"{'='*80}")

# Check temperature file
for var_name in ['tas', 't2m', '2m_temperature', 'temperature', 'T2M']:
    if var_name in ds_temp.variables:
        temp_var = var_name
        print(f"\n{'='*60}")
        print(f"TEMPERATURE VARIABLE:")
        print(f"{'='*60}")
        print(f"  Variable name: {var_name}")
        
        # Check units and attributes
        units_attr = ds_temp[var_name].attrs.get('units', 'NOT FOUND')
        long_name = ds_temp[var_name].attrs.get('long_name', 'NOT FOUND')
        
        print(f"  Units attribute: {units_attr}")
        print(f"  Description: {long_name}")
        
        # Check sample value to verify units
        sample_val = float(ds_temp[var_name].isel(time=0, latitude=0, longitude=0).values)
        print(f"  Sample value: {sample_val:.2f}")
        
        # Determine unit conversion needed (ERA5-Land standard: Kelvin)
        units_lower = str(units_attr).lower()
        if 'kelvin' in units_lower or units_lower == 'k':
            print(f"  Units: KELVIN → Will convert to Celsius (-273.15)")
            temp_units_kelvin = True
        elif 'celsius' in units_lower or 'degc' in units_lower or 'c' in units_lower:
            print(f"  Units: CELSIUS → No conversion needed")
            temp_units_kelvin = False
        else:
            # Fallback: check sample value (ERA5-Land uses Kelvin, values typically >200)
            print(f"  WARNING: Units attribute unclear. Checking sample value...")
            if sample_val > 200:
                print(f"     → Sample value {sample_val:.2f} > 200, assuming KELVIN")
                temp_units_kelvin = True
            else:
                print(f"     → Sample value {sample_val:.2f} ≤ 200, assuming CELSIUS")
                temp_units_kelvin = False
        break

# Check relative humidity file (already opened)
for var_name in ['r2m', 'relative_humidity', 'rh', 'RH', '__xarray_dataarray_variable__']:
    if var_name in ds_rh.variables:
        rh_var = var_name
        print(f"\n{'='*60}")
        print(f"RELATIVE HUMIDITY VARIABLE:")
        print(f"{'='*60}")
        print(f"  Variable name: {var_name}")
        
        # Check units and attributes
        units_attr = ds_rh[var_name].attrs.get('units', 'NOT FOUND')
        long_name = ds_rh[var_name].attrs.get('long_name', 'NOT FOUND')
        
        print(f"  Units attribute: {units_attr}")
        print(f"  Description: {long_name}")
        
        # Check sample value to verify units
        sample_val = float(ds_rh[var_name].isel(time=0, latitude=0, longitude=0).values)
        print(f"  Sample value: {sample_val:.4f}")
        
        # Determine unit conversion needed (ERA5-Land standard: fraction 0-1)
        units_lower = str(units_attr).lower()
        if '%' in units_lower or 'percent' in units_lower:
            print(f"  Units: PERCENTAGE → No conversion needed")
            rh_units_fraction = False
        elif '1' in units_lower or 'fraction' in units_lower or units_lower == '1':
            print(f"  Units: FRACTION (0-1) → Will convert to percentage (×100)")
            rh_units_fraction = True
        else:
            # Fallback: check sample value (ERA5-Land uses fraction, values typically 0-1)
            print(f"  WARNING: Units attribute unclear. Checking sample value...")
            if 0 <= sample_val <= 1:
                print(f"     → Sample value {sample_val:.4f} in range 0-1, assuming FRACTION")
                rh_units_fraction = True
            else:
                print(f"     → Sample value {sample_val:.4f} > 1, assuming PERCENTAGE")
                rh_units_fraction = False
        break

# Check precipitation file (already opened)
for var_name in ['pr', 'tp', 'total_precipitation', 'precipitation', 'TP']:
    if var_name in ds_pr.variables:
        precip_var = var_name
        print(f"\n{'='*60}")
        print(f"PRECIPITATION VARIABLE:")
        print(f"{'='*60}")
        print(f"  Variable name: {var_name}")
        
        # Check units and attributes
        units_attr = ds_pr[var_name].attrs.get('units', 'NOT FOUND')
        long_name = ds_pr[var_name].attrs.get('long_name', 'NOT FOUND')
        
        print(f"  Units attribute: {units_attr}")
        print(f"  Description: {long_name}")
        
        # Check sample value to verify units
        sample_val = float(ds_pr[var_name].isel(time=0, latitude=0, longitude=0).values)
        print(f"  Sample value: {sample_val:.6f}")
        
        # Determine unit conversion needed (ERA5-Land standard: meters)
        units_lower = str(units_attr).lower()
        
        # Check for rate-based precipitation (kg m-2 s-1)
        if 'kg m-2 s-1' in units_lower or 'kg/m2/s' in units_lower or 'kg m^-2 s^-1' in units_lower:
            print(f"  Units: KG M-2 S-1 (rate) → Will convert to mm total")
            if is_hourly:
                print(f"     → Hourly data: multiply by 3600 s/h, then sum for daily")
            else:
                print(f"     → Daily data: multiply by 86400 s/d")
            precip_units_rate = True
            precip_units_meters = False
        # Check for millimeters or kg m-2 (equivalent to mm)
        elif 'mm' in units_lower or 'millimeter' in units_lower or 'millimetre' in units_lower:
            print(f"  Units: MILLIMETERS → No conversion needed")
            precip_units_meters = False
            precip_units_rate = False
        elif 'kg m-2' in units_lower or 'kg/m2' in units_lower or 'kg m^-2' in units_lower:
            print(f"  Units: KG M-2 (equivalent to mm) → No conversion needed")
            precip_units_meters = False
            precip_units_rate = False
        # Check for meters (ERA5-Land standard)
        elif 'm' in units_lower and 'mm' not in units_lower and 'kg' not in units_lower:
            print(f"  Units: METERS → Will convert to mm (×1000)")
            precip_units_meters = True
            precip_units_rate = False
        else:
            # Fallback: check sample value (ERA5-Land uses meters, values typically <0.01)
            print(f"  WARNING: Units attribute unclear. Checking sample value...")
            if sample_val < 0.01:
                print(f"     → Sample value {sample_val:.6f} < 0.01, assuming METERS")
                precip_units_meters = True
                precip_units_rate = False
            else:
                print(f"     → Sample value {sample_val:.6f} ≥ 0.01, assuming MILLIMETERS")
                precip_units_meters = False
                precip_units_rate = False
        break

if not temp_var:
    print(f"\nERROR: Temperature variable not found!")
if not rh_var:
    print(f"\nERROR: Relative humidity variable not found!")
if not precip_var:
    print(f"\nERROR: Precipitation variable not found!")

if not all([temp_var, rh_var, precip_var]):
    print("\nERROR: Missing required variables. Please check the NetCDF files.")
    ds_temp.close()
    ds_rh.close()
    ds_pr.close()
    exit(1)

# Unit information is now determined in the check above
# Variables: temp_units_kelvin, rh_units_fraction, precip_units_meters
# (These are set during the variable checking loops above)

# Summary of what will be done
print(f"\n{'='*80}")
print("EXTRACTION SUMMARY:")
print(f"{'='*80}")
print(f"\nVariables to extract:")
print(f"  1. Temperature:   {temp_var}")
print(f"  2. Relative Humidity: {rh_var}")
print(f"  3. Precipitation: {precip_var}")

print(f"\nTime resolution: {'HOURLY (will aggregate to daily)' if is_hourly else 'DAILY (already daily)'}")

print(f"\nUnit conversions:")
if temp_units_kelvin:
    print(f"  • Temperature: KELVIN → CELSIUS (subtract 273.15)")
else:
    print(f"  • Temperature: CELSIUS (no conversion)")
    
if rh_units_fraction:
    print(f"  • Relative Humidity: FRACTION (0-1) → PERCENTAGE (multiply by 100)")
else:
    print(f"  • Relative Humidity: PERCENTAGE (no conversion)")
    
if precip_units_rate:
    if is_hourly:
        print(f"  • Precipitation: KG M-2 S-1 (rate) → MM (multiply by 3600 s/h, then sum for daily)")
    else:
        print(f"  • Precipitation: KG M-2 S-1 (rate) → MM (multiply by 86400 s/d for daily total)")
elif precip_units_meters:
    print(f"  • Precipitation: METERS → MILLIMETERS (multiply by 1000)")
else:
    print(f"  • Precipitation: MILLIMETERS (no conversion)")

print(f"\nExtraction parameters:")
print(f"  • Grid cells to extract: {len(era5_grid_cells)}")
print(f"  • Years to process: {years[0]} to {years[-1]} ({len(years)} years)")
print(f"  • Total files to process: {len(years)} years × 3 variables = {len(years) * 3} files")

print(f"\n{'='*80}")
print("WARNING: IMPORTANT: Please verify the variable names and unit conversions above!")
print("WARNING: Make sure the conversions are correct before proceeding!")
print(f"{'='*80}")

# Close reference files after check
ds_temp.close()
ds_rh.close()
ds_pr.close()

# Verify unit variables were set
if temp_units_kelvin is None or rh_units_fraction is None or precip_units_meters is None or precip_units_rate is None:
    print("\nERROR: Unit detection failed. Please check the script.")
    print(f"  temp_units_kelvin: {temp_units_kelvin}")
    print(f"  rh_units_fraction: {rh_units_fraction}")
    print(f"  precip_units_meters: {precip_units_meters}")
    print(f"  precip_units_rate: {precip_units_rate}")
    exit(1)

# ============================================================================
# VERIFY GRID CELLS ARE AVAILABLE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Verifying ERA5-Land grid cells are available...")
print("=" * 80)

# Get grid info from first year's file
ds_ref = xr.open_dataset(tas_file, engine="h5netcdf", decode_timedelta=False)
latitudes = ds_ref.latitude.values
longitudes = ds_ref.longitude.values
ds_ref.close()

print(f"\nAvailable ERA5-Land grid:")
print(f"  Latitude range: {latitudes.min():.2f} to {latitudes.max():.2f}")
print(f"  Longitude range: {longitudes.min():.2f} to {longitudes.max():.2f}")
print(f"  Latitude resolution: {np.diff(latitudes)[0]:.2f}°")
print(f"  Longitude resolution: {np.diff(longitudes)[0]:.2f}°")

print(f"\nExtracting data for {len(era5_grid_cells)} exact grid cells:")
for i, ((lat, lon), name) in enumerate(zip(era5_grid_cells, station_names), 1):
    # Check if grid cell exists (within tolerance)
    lat_exists = np.any(np.abs(latitudes - lat) < 0.05)  # 0.05° tolerance
    lon_exists = np.any(np.abs(longitudes - lon) < 0.05)
    status = "OK" if (lat_exists and lon_exists) else "WARNING"
    print(f"  {i:2d}. {status} {name:20s} - Grid cell: ({lat:.1f}°N, {lon:.1f}°E)")

# ============================================================================
# HELPER FUNCTION FOR FILE FINDING
# ============================================================================

def find_file_for_year(year, file_list, pattern_keywords):
    """
    Find a file for a specific year that contains the pattern keywords.
    
    Args:
        year: Year to search for
        file_list: List of file paths to search
        pattern_keywords: List of keywords that should be in filename
    
    Returns:
        File path if found, None otherwise
    """
    year_str = str(year)
    for file in file_list:
        filename = os.path.basename(file).lower()
        if year_str in filename:
            # Check if all keywords are in filename
            if all(keyword.lower() in filename for keyword in pattern_keywords):
                return file
    return None

# ============================================================================
# EXTRACT AND PROCESS DATA FOR ALL YEARS
# ============================================================================
print("\n" + "=" * 80)
print(f"STEP 4: Extracting and processing data for years {years[0]}-{years[-1]}...")
print("=" * 80)

all_grid_data = []

# Process each year
for year in years:
    print(f"\n{'='*80}")
    print(f"Processing year {year}...")
    print(f"{'='*80}")
    
    # Find files for this year
    pr_file = find_file_for_year(year, pr_files, ['pr'])
    rh_file = find_file_for_year(year, rh_files, ['relative', 'humidity'])
    tas_file = find_file_for_year(year, tas_files, ['tas'])
    
    # If not found with pattern matching, try standard naming
    if not pr_file:
        pr_file = os.path.join(era5_base_dir, f"ERA5-Land_pr_Northern_Italy_{year}.nc")
    if not rh_file:
        rh_file = os.path.join(era5_base_dir, f"ERA5-Land_relative_humidity_Northern_Italy_{year}.nc")
    if not tas_file:
        tas_file = os.path.join(era5_base_dir, f"ERA5-Land_tas_Northern_Italy_{year}.nc")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [pr_file, rh_file, tas_file]):
        missing = [f for f in [pr_file, rh_file, tas_file] if not os.path.exists(f)]
        print(f"WARNING: Skipping year {year}: One or more files not found")
        for f in missing:
            print(f"     Missing: {os.path.basename(f)}")
        continue
    
    # Open datasets for this year
    ds_temp = xr.open_dataset(tas_file, engine="h5netcdf", decode_timedelta=False)
    ds_rh = xr.open_dataset(rh_file, engine="h5netcdf", decode_timedelta=False)
    ds_pr = xr.open_dataset(pr_file, engine="h5netcdf", decode_timedelta=False)
    
    # Fix unnamed variable in humidity dataset if needed
    if "__xarray_dataarray_variable__" in ds_rh.variables:
        ds_rh = ds_rh.rename({"__xarray_dataarray_variable__": rh_var})
    
    # Process each grid cell
    for (grid_lat, grid_lon), station_name, (station_lat, station_lon) in zip(era5_grid_cells, station_names, station_coords):
        print(f"\n  Processing {station_name}: Grid cell ({grid_lat:.1f}°N, {grid_lon:.1f}°E)")
        
        # Extract data for this grid point
        temp_data = ds_temp[temp_var].sel(latitude=grid_lat, longitude=grid_lon, method="nearest")
        rh_data = ds_rh[rh_var].sel(latitude=grid_lat, longitude=grid_lon, method="nearest")
        precip_data = ds_pr[precip_var].sel(latitude=grid_lat, longitude=grid_lon, method="nearest")
    
        # Convert to DataFrame
        temp_df = temp_data.to_dataframe().reset_index()
        rh_df = rh_data.to_dataframe().reset_index()
        precip_df = precip_data.to_dataframe().reset_index()
        
        # Merge dataframes
        merged_df = temp_df.merge(rh_df, on=['time', 'latitude', 'longitude'], suffixes=('_temp', '_rh'))
        merged_df = merged_df.merge(precip_df, on=['time', 'latitude', 'longitude'])
        
        # Rename columns - find the actual column names
        temp_col = [col for col in merged_df.columns if temp_var in col.lower()][0]
        rh_col = [col for col in merged_df.columns if rh_var in col.lower() or 'relative' in col.lower()][0]
        precip_col = [col for col in merged_df.columns if precip_var in col.lower()][0]
        
        merged_df = merged_df.rename(columns={
            temp_col: 'temperature_K',
            rh_col: 'relative_humidity',
            precip_col: 'precipitation_m'
        })
        
        # Convert time to datetime
        merged_df['time'] = pd.to_datetime(merged_df['time'])
        
        # Convert units based on detected units from check phase
        # Temperature: Kelvin to Celsius (if needed)
        if temp_units_kelvin:
            merged_df['temperature_C'] = merged_df['temperature_K'] - 273.15
        else:
            merged_df['temperature_C'] = merged_df['temperature_K']  # Already in Celsius
        
        # Relative humidity: convert fraction to percentage (if needed)
        if rh_units_fraction:
            merged_df['relative_humidity_pct'] = merged_df['relative_humidity'] * 100
        else:
            merged_df['relative_humidity_pct'] = merged_df['relative_humidity']  # Already in percentage
        
        # Precipitation: convert based on detected units
        if precip_units_rate:
            # Rate-based (kg m-2 s-1): convert to total
            if is_hourly:
                # Hourly data: multiply by 3600 s/h to get hourly total in mm
                merged_df['precipitation_mm'] = merged_df['precipitation_m'] * 3600
            else:
                # Daily data: multiply by 86400 s/d to get daily total in mm
                merged_df['precipitation_mm'] = merged_df['precipitation_m'] * 86400
        elif precip_units_meters:
            # Meters: convert to mm
            merged_df['precipitation_mm'] = merged_df['precipitation_m'] * 1000
        else:
            # Already in mm (or kg m-2 which equals mm)
            merged_df['precipitation_mm'] = merged_df['precipitation_m']  # Already in mm
    
        # Aggregate to daily if hourly
        if is_hourly:
            merged_df['date'] = merged_df['time'].dt.date
            
            daily_df = merged_df.groupby('date').agg({
                'temperature_C': 'mean',
                'relative_humidity_pct': 'mean',
                'precipitation_mm': 'sum'  # Sum for total daily precipitation
            }).reset_index()
            
            daily_df['date'] = pd.to_datetime(daily_df['date'])
        else:
            # Already daily, just select and rename columns
            daily_df = merged_df[['time', 'temperature_C', 'relative_humidity_pct', 'precipitation_mm']].copy()
            daily_df = daily_df.rename(columns={'time': 'date'})
        
        # Add metadata columns to match standard output format
        daily_df['number'] = 0  # ERA5-Land is not an ensemble, so number = 0
        daily_df['latitude'] = grid_lat  # ERA5-Land grid cell latitude
        daily_df['longitude'] = grid_lon  # ERA5-Land grid cell longitude
        daily_df['station'] = station_name  # Station name
        daily_df['lat'] = station_lat  # Actual station latitude
        daily_df['lon'] = station_lon  # Actual station longitude
        
        # Rename columns to match standard output format
        daily_df = daily_df.rename(columns={
            'temperature_C': 'Tmean_C',
            'relative_humidity_pct': 'RH_%',
            'precipitation_mm': 'Precip_mm',
            'date': 'date'
        })
        
        # Format date as DD/MM/YYYY
        daily_df['date'] = daily_df['date'].dt.strftime('%d/%m/%Y')
        
        # Reorder columns to match standard output format: date, number, latitude, longitude, Tmean_C, RH_%, Precip_mm, station, lat, lon
        daily_df = daily_df[['date', 'number', 'latitude', 'longitude', 
                            'Tmean_C', 'RH_%', 'Precip_mm', 
                            'station', 'lat', 'lon']]
        
        all_grid_data.append(daily_df)
    
    # Close datasets for this year
    ds_temp.close()
    ds_rh.close()
    ds_pr.close()
    
    print(f"\nCompleted year {year}")

# ============================================================================
# COMBINE AND SAVE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Combining and saving data...")
print("=" * 80)

# Combine all grid points
final_df = pd.concat(all_grid_data, ignore_index=True)

# Sort by date and station
final_df = final_df.sort_values(['date', 'station']).reset_index(drop=True)

# Save to CSV
final_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"\nData extraction complete!")
print(f"   Total records: {len(final_df)}")
print(f"   Date range: {final_df['date'].min()} to {final_df['date'].max()}")
print(f"   Grid cells extracted: {len(era5_grid_cells)}")
print(f"   Unique dates: {final_df['date'].nunique()}")
print(f"   Stations: {final_df['station'].nunique()}")
print(f"\nSaved to: {output_file}")

# No need to close - datasets are closed after each year

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)

