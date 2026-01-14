#!/usr/bin/env python3
"""
2. Preprocess Seasonal Forecasts

Extracts data from seasonal forecast NetCDF files, calculates variables,
applies altitude correction, and saves CSV files in the required format.

Input: Seasonal forecast NetCDF files + orography files
Output: CSV files with columns: Date, Ensemble_Member, Location, T (°C), RH (%), Rain (mm), LW (h)
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory for seasonal forecast NetCDF files
# UPDATE: Replace with your directory path containing seasonal forecast NetCDF files
BASE_DIR = r"UPDATE_WITH_YOUR_SEASONAL_FORECAST_NETCDF_DIRECTORY"

# Output directory - same as input directory
OUTPUT_DIR = BASE_DIR

# Orography file paths
# UPDATE: Replace with your actual file paths
# These files are required for altitude correction
FORECAST_OROGRAPHY_FILE = r"UPDATE_WITH_YOUR_FORECAST_OROGRAPHY_NC_FILE"
ERA5_OROGRAPHY_FILE = r"UPDATE_WITH_YOUR_ERA5_LAND_GEOPOTENTIAL_NC_FILE"

# Target grid cells to extract (latitude, longitude)
# Format: (lat, lon) tuples - Seasonal forecast grid cell coordinates
# Multiple stations can share the same grid cell
TARGET_GRIDS = [
    (45, 9),   # Castel San Giovanni
    (45, 10),  # Caorso, Roncopascolo
    (45, 11),  # Luzzara, Mirandola, Castel Maggiore
    (45, 12),  # Guarda Ferrarese, Ostellato
    (44, 12),  # Forli, Medicina
]

# Lapse rate for temperature correction (°C/m)
LAPSE_RATE = 0.0065  # Standard atmospheric lapse rate

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def correct_temperature(T, alt_forecast, alt_era5):
    """
    Apply altitude correction to temperature using lapse rate.
    
    Args:
        T: Temperature in °C
        alt_forecast: Forecast grid altitude (m)
        alt_era5: ERA5 grid altitude (m)
    
    Returns:
        Corrected temperature in °C
    """
    if pd.isna(T) or pd.isna(alt_forecast) or pd.isna(alt_era5):
        return np.nan
    return T + (alt_forecast - alt_era5) * LAPSE_RATE

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main processing function."""
    if not os.path.exists(BASE_DIR):
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        return
    
    if not os.path.exists(FORECAST_OROGRAPHY_FILE):
        print(f"ERROR: Forecast orography file not found: {FORECAST_OROGRAPHY_FILE}")
        return
    
    if not os.path.exists(ERA5_OROGRAPHY_FILE):
        print(f"ERROR: ERA5 orography file not found: {ERA5_OROGRAPHY_FILE}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Auto-detect forecast NetCDF files
    all_nc_files = glob.glob(os.path.join(BASE_DIR, "*.nc"))
    forecast_files = [os.path.basename(f) for f in all_nc_files 
                     if 'orography' not in os.path.basename(f).lower() and 'orog' not in os.path.basename(f).lower()]
    
    if not forecast_files:
        print(f"ERROR: No forecast NetCDF files found in: {BASE_DIR}")
        return
    
    forecast_files = sorted(forecast_files)
    print(f"Processing {len(forecast_files)} forecast files...")
    
    # Load orography files
    ds_forecast_orog = xr.open_dataset(FORECAST_OROGRAPHY_FILE, engine="h5netcdf")
    alt_forecast = (ds_forecast_orog["z"] / 9.80665).mean(
        dim=["number", "forecast_reference_time", "forecast_period"]
    )
    
    ds_era5_orog = xr.open_dataset(ERA5_OROGRAPHY_FILE, engine="h5netcdf")
    alt_era5 = (ds_era5_orog["z"] / 9.80665).mean(dim="time")
    
    # Precompute mean ERA5 altitude per forecast grid cell
    era5_means = {}
    for lat in alt_forecast.latitude.values:
        for lon in alt_forecast.longitude.values:
            mask_lat = (alt_era5.latitude >= lat - 0.5) & (alt_era5.latitude < lat + 0.5)
            mask_lon = (alt_era5.longitude >= lon - 0.5) & (alt_era5.longitude < lon + 0.5)
            sub = alt_era5.sel(latitude=mask_lat, longitude=mask_lon)
            era5_means[(float(lat), float(lon))] = float(sub.mean().values)
    
    for fname in forecast_files:
        file_path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(file_path):
            continue
        
        try:
            ds = xr.open_dataset(file_path, engine="h5netcdf")
            
            # ========================================================================
            # STEP 1: EXTRACT and CONVERT to correct variables and units
            # ========================================================================
            # Convert temperatures from Kelvin to Celsius
            tmax = ds["mx2t24"] - 273.15
            tmin = ds["mn2t24"] - 273.15
            tmean = (tmax + tmin) / 2
            
            # Calculate RH from dewpoint and temperature (using Kelvin values)
            t = ds["t2m"]
            td = ds["d2m"]
            rh = 100 * (
                np.exp((17.625 * (td - 273.15)) / (243.04 + (td - 273.15))) /
                np.exp((17.625 * (t - 273.15)) / (243.04 + (t - 273.15)))
            )
            rh = rh.clip(0, 100)
            
            # Convert precipitation from meters to millimeters
            tp = ds["tp"] * 1000
            
            # Convert to DataFrame
            df = xr.Dataset({
                "Tmean_C": tmean,
                "RH_%": rh,
                "Precip_mm": tp,
            }).to_dataframe().reset_index()
            
            # Filter to target grid cells
            df = df[df[["latitude", "longitude"]].apply(tuple, axis=1).isin(TARGET_GRIDS)]
            
            if df.empty:
                continue
            
            # Calculate daily precipitation from cumulative
            df["Precip_mm_daily"] = df.groupby(
                ["number", "latitude", "longitude"]
            )["Precip_mm"].diff().clip(lower=0)
            
            # ========================================================================
            # STEP 2: APPLY altitude correction to mean temperature
            # ========================================================================
            # Add altitude information
            df["alt_forecast"] = df.apply(
                lambda r: float(alt_forecast.sel(
                    latitude=r["latitude"],
                    longitude=r["longitude"],
                    method="nearest"
                ).values),
                axis=1
            )
            
            df["alt_era5_mean"] = df.apply(
                lambda r: era5_means.get(
                    (float(r["latitude"]), float(r["longitude"])),
                    np.nan
                ),
                axis=1
            )
            
            # Apply altitude correction to mean temperature only (matching original workflow)
            df["Tmean_C"] = df.apply(
                lambda r: correct_temperature(
                    r["Tmean_C"],
                    r["alt_forecast"],
                    r["alt_era5_mean"]
                ),
                axis=1
            )
            
            # Drop intermediate columns
            df = df.drop(columns=["Precip_mm", "alt_forecast", "alt_era5_mean"])
            
            df["Location"] = df["latitude"].astype(int).astype(str) + "," + df["longitude"].astype(int).astype(str)
            
            df_output = df[[
                "time", "number", "Location", 
                "Tmean_C", "RH_%", "Precip_mm_daily"
            ]].copy()
            
            df_output = df_output.rename(columns={
                "time": "Date",
                "number": "Ensemble_Member",
                "Tmean_C": "T (°C)",
                "RH_%": "RH (%)",
                "Precip_mm_daily": "Rain (mm)"
            })
            
            df_output["LW (h)"] = np.nan
            
            df_output = df_output[[
                "Date", "Ensemble_Member", "Location",
                "T (°C)", "RH (%)", "Rain (mm)", "LW (h)"
            ]]
            
            if df_output["Date"].dtype == "object":
                df_output["Date"] = pd.to_datetime(df_output["Date"])
            df_output["Date"] = df_output["Date"].dt.strftime("%d/%m/%Y")
            
            output_path = os.path.join(OUTPUT_DIR, fname.replace(".nc", "_daily.csv"))
            df_output.to_csv(output_path, index=False)
            print(f"Saved: {os.path.basename(output_path)}")
            
            ds.close()
            
        except Exception as e:
            print(f"Error: {fname} - {e}")
            continue
    
    ds_forecast_orog.close()
    ds_era5_orog.close()
    print("Done.")

if __name__ == "__main__":
    main()

