#!/usr/bin/env python3
"""
6. Figure 5 Comprehensive Analysis - Model Evaluation

This script evaluates model performance using AFI outputs from the AFLA-maize model.
It calculates performance metrics (TP, TN, FP, FN) and creates comprehensive visualizations.

Input: CSV file with AFLA-maize model outputs (AFI summary)
Output: Performance evaluation figures and summary statistics

Note: This script requires outputs from the AFLA-maize model. The model code is available
upon request from the corresponding author.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import math
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# --- Step 1: Define Constants and Paths ---

# UPDATE_WITH_YOUR_AFI_SUMMARY_CSV_PATH
SUMMARY_CSV_PATH = r"UPDATE_WITH_YOUR_AFI_SUMMARY_CSV_PATH"
# Output directory - will be set to summary CSV file directory in main() function
OUTPUT_DIR = None

# Plot styling
sns.set_style("ticks")
plt.rcParams['figure.dpi'] = 150
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14

model_order = ['ERA5-Land*', 'Forecast April', 'Forecast June', 'Forecast July', 'Forecast August']
outcome_colors = {
    'background': '#ffffff',
    'False Negative': 'lightcoral',
    'False Positive': 'orange',
    'True Negative': 'lightgreen',
    'True Positive': 'darkgreen'
}

# --- Step 2: Data Loading and Analysis ---

def load_and_prepare_data():
    """
    Loads the summary CSV file and prepares it for analysis.
    
    Expected CSV columns:
    - location: Location name (e.g., "Castel San Giovanni")
    - year: Year (integer, e.g., 2008)
    - forecast_timing: Forecast timing ("April", "June", "July", "August")
    - run_type: Data source ("Station", "ERA5-Land*", "Forecast_Mean")
    - afi_max: AFI_max value (float)
    
    Returns dataframe with columns: location, year, run_type, forecast_month, afi_max
    """
    try:
        df = pd.read_csv(SUMMARY_CSV_PATH)
        print(f"Loaded CSV with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check required columns
        required_cols = ['location', 'year', 'forecast_timing', 'run_type', 'afi_max']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert year to integer if it's not already
        df['year'] = df['year'].astype(int)
        
        # Map run_type to standardized format
        run_type_mapping = {
            'Station': 'station',
            'ERA5-Land*': 'era5*',
            'Forecast_Mean': 'forecast_mean'
        }
        df['run_type'] = df['run_type'].map(run_type_mapping)
        
        # Map forecast_timing to forecast_month (standardize month names)
        timing_to_month = {
            'April': 'April',
            'June': 'June',
            'July': 'July',
            'August': 'August'
        }
        df['forecast_month'] = df['forecast_timing'].map(timing_to_month)
        
        # Remove rows with missing run_type
        df = df.dropna(subset=['run_type'])
        
        # For forecast_mean data, forecast_month is required
        # For Station and ERA5-Land*, forecast_month may be present but is not used in analysis
        
        # Keep only needed columns
        df = df[['location', 'year', 'run_type', 'forecast_month', 'afi_max']].copy()
        
        print(f"Prepared data with shape: {df.shape}")
        print(f"Unique locations: {sorted(df['location'].unique())}")
        print(f"Unique run_types: {sorted(df['run_type'].unique())}")
        print(f"Unique forecast_months: {sorted(df['forecast_month'].unique())}")
        print(f"Year range: {df['year'].min()} to {df['year'].max()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Summary file not found at '{SUMMARY_CSV_PATH}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_yearly_outcomes_all_locations(df):
    """
    Calculates the performance outcome (TP, TN, FP, FN) for each model for each year and location.
    Returns outcomes with location-year combinations as index (e.g., "Castel San Giovanni_2008").
    """
    AFI_THRESHOLD = 95.0
    
    df['exceeds'] = df['afi_max'] > AFI_THRESHOLD

    # 1. Filter to only the data we need for the performance matrix
    perf_df = df[df['run_type'].isin(['station', 'era5*', 'forecast_mean'])].copy()

    # 2. Create a consistent 'model' column name for pivoting
    perf_df['model'] = perf_df['run_type']
    fc_mask = perf_df['run_type'] == 'forecast_mean'
    perf_df.loc[fc_mask, 'model'] = 'Forecast ' + perf_df.loc[fc_mask, 'forecast_month'].astype(str)
    perf_df['model'] = perf_df['model'].replace({'station': 'Station', 'era5*': 'ERA5-Land*'})

    # 3. Handle duplicates: For Station and ERA5-Land*, there may be multiple rows per location-year
    # (one for each forecast_timing). We keep only one value per location-year-model.
    # For forecasts, each forecast_timing creates a different model, so no issue.
    perf_df = perf_df.drop_duplicates(subset=['location', 'year', 'model'], keep='first')

    # 4. Create location-year index for multi-index pivot
    perf_df['location_year'] = perf_df['location'] + '_' + perf_df['year'].astype(str)

    # 5. Pivot with location-year as index
    outcome_table = perf_df.pivot_table(index='location_year', columns='model', values='exceeds')

    # 6. Separate the ground truth (Station) from the model predictions
    if 'Station' not in outcome_table.columns:
        print("Error: 'Station' data not found in the summary file. Cannot calculate performance.")
        return pd.DataFrame(), {}
        
    station_exceeds = outcome_table['Station']
    predictions = outcome_table.drop(columns=['Station'])

    # 7. Calculate outcomes for each location-year combination
    yearly_outcomes = pd.DataFrame(index=station_exceeds.index)
    model_performance = {}

    for model_name in model_order:
        if model_name not in predictions.columns:
            print(f"Warning: Model '{model_name}' not found in prediction data. Skipping.")
            continue
            
        comparison_df = pd.concat([station_exceeds, predictions[model_name]], axis=1).dropna()
        obs = comparison_df['Station'].astype(bool)
        pred = comparison_df[model_name].astype(bool)

        TP = (pred & obs); TN = (~pred & ~obs); FP = (pred & ~obs); FN = (~pred & obs)
        
        # Calculate aggregate performance metrics across all location-years
        n_tp = TP.sum()
        n_tn = TN.sum()
        n_fp = FP.sum()
        n_fn = FN.sum()
        
        total_predictions = n_tp + n_tn + n_fp + n_fn
        correct_predictions = n_tp + n_tn
        
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else np.nan
        sensitivity = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else np.nan
        specificity = n_tn / (n_tn + n_fp) if (n_tn + n_fp) > 0 else np.nan
        
        # Calculate Precision and F1 Score
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else np.nan
        # F1 = 2 × (precision × recall) / (precision + recall)
        # where recall = sensitivity
        if not np.isnan(precision) and not np.isnan(sensitivity) and (precision + sensitivity) > 0:
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        else:
            f1_score = np.nan
        
        # Calculate False Positive Rate (FPR) and False Negative Rate (FNR)
        fpr = n_fp / (n_fp + n_tn) if (n_fp + n_tn) > 0 else np.nan  # FPR = 1 - Specificity
        fnr = n_fn / (n_fn + n_tp) if (n_fn + n_tp) > 0 else np.nan  # FNR = 1 - Sensitivity
        
        model_performance[model_name] = {
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1 Score': f1_score,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr,
            'TP': n_tp,
            'TN': n_tn,
            'FP': n_fp,
            'FN': n_fn
        }
        
        # Create outcome series for each location-year
        outcome_series = pd.Series(np.nan, index=station_exceeds.index, dtype='object')
        outcome_series.loc[TP.index[TP]] = 'True Positive'
        outcome_series.loc[TN.index[TN]] = 'True Negative'
        outcome_series.loc[FP.index[FP]] = 'False Positive'
        outcome_series.loc[FN.index[FN]] = 'False Negative'
        yearly_outcomes[model_name] = outcome_series
    
    # Add Station evaluation (comparing Station to itself - perfect scores)
    station_bool = station_exceeds.dropna().astype(bool)
    n_tp_station = station_bool.sum()  # All positives are TP when comparing Station to itself
    n_tn_station = (~station_bool).sum()  # All negatives are TN when comparing Station to itself
    n_fp_station = 0  # Station can't have false positives against itself
    n_fn_station = 0  # Station can't have false negatives against itself
    
    total_predictions_station = n_tp_station + n_tn_station + n_fp_station + n_fn_station
    correct_predictions_station = n_tp_station + n_tn_station
    
    accuracy_station = (correct_predictions_station / total_predictions_station) * 100 if total_predictions_station > 0 else np.nan
    sensitivity_station = 1.0 if n_tp_station > 0 else np.nan  # Perfect sensitivity (all positives detected)
    specificity_station = 1.0 if n_tn_station > 0 else np.nan  # Perfect specificity (all negatives correct)
    f1_score_station = 1.0 if n_tp_station > 0 else np.nan  # Perfect F1 (perfect precision and recall)
    fpr_station = 0.0  # No false positives
    fnr_station = 0.0  # No false negatives
    
    model_performance['Station'] = {
        'Accuracy': accuracy_station,
        'Sensitivity': sensitivity_station,
        'Specificity': specificity_station,
        'F1 Score': f1_score_station,
        'False Positive Rate': fpr_station,
        'False Negative Rate': fnr_station,
        'TP': n_tp_station,
        'TN': n_tn_station,
        'FP': n_fp_station,
        'FN': n_fn_station
    }
    
    # Create outcome series for Station (all are True Positive or True Negative)
    outcome_series_station = pd.Series(np.nan, index=station_exceeds.index, dtype='object')
    outcome_series_station.loc[station_bool.index[station_bool]] = 'True Positive'
    outcome_series_station.loc[station_bool.index[~station_bool]] = 'True Negative'
    yearly_outcomes['Station'] = outcome_series_station
        
    return yearly_outcomes, model_performance

def get_yearly_outcomes(df, location=None):
    """
    Calculates the performance outcome (TP, TN, FP, FN) for each model for each year based on the forecast mean AFI_max.
    If location is provided, filters data for that location only.
    (Kept for backward compatibility if needed)
    """
    if location:
        return get_yearly_outcomes_all_locations(df[df['location'] == location].copy())
    else:
        return get_yearly_outcomes_all_locations(df)

# --- Step 3: Figure Generation ---

def _build_multiindex_from_location_year(index):
    """Helper to split 'Location_Year' strings into a MultiIndex."""
    locations, years = [], []
    for label in index:
        label = str(label)
        if '_' in label:
            loc, year = label.rsplit('_', 1)
        else:
            loc, year = label, ''
        locations.append(loc)
        years.append(year)
    return pd.MultiIndex.from_arrays([locations, years], names=['Location', 'Year'])


def create_merged_heatmap_figure(yearly_outcomes):
    """Creates a single stacked heatmap grouped by location with sub-rows for each data source."""
    if yearly_outcomes.empty:
        print("No data available to build the merged heatmap.")
        return

    outcomes_multi = yearly_outcomes.copy()
    outcomes_multi.index = _build_multiindex_from_location_year(outcomes_multi.index)
    
    # Preserve user-defined location order
    location_order = [
        'Castel San Giovanni', 'Caorso', 'Roncopascolo', 'Luzzara', 'Mirandola',
        'Castel Maggiore', 'Guarda Ferrarese', 'Ostellato', 'Medicina', 'Forli'
    ]
    available_locations = list(outcomes_multi.index.get_level_values('Location').unique())
    locations = [loc for loc in location_order if loc in available_locations]
    remaining = [loc for loc in available_locations if loc not in locations]
    locations.extend(sorted(remaining))
    year_labels = outcomes_multi.index.get_level_values('Year').unique()
    years = sorted([int(y) for y in year_labels if str(y).isdigit()])
    years_str = [str(y) for y in years]

    available_models = [m for m in model_order if m in outcomes_multi.columns]
    if not available_models:
        print("No model columns found for the merged heatmap.")
        return

    outcome_map = {'False Negative': 1, 'False Positive': 2, 'True Negative': 3, 'True Positive': 4}
    cmap = ListedColormap([
        outcome_colors['background'],
        outcome_colors['False Negative'],
        outcome_colors['False Positive'],
        outcome_colors['True Negative'],
        outcome_colors['True Positive']
    ])

    frames = []
    for model in available_models:
        mat = outcomes_multi[model].unstack('Year')
        mat = mat.reindex(index=locations)
        mat.columns = [str(col) for col in mat.columns]
        mat = mat.reindex(columns=years_str)
        mat['Model'] = model
        mat = mat.set_index('Model', append=True)
        frames.append(mat)

    heatmap_df = pd.concat(frames)
    desired_index = pd.MultiIndex.from_product([locations, available_models], names=['Location', 'Model'])
    heatmap_df = heatmap_df.reindex(desired_index)
    heatmap_numeric = heatmap_df.map(lambda val: outcome_map.get(val, 0))

    n_rows = len(heatmap_numeric)
    fig_height = max(8, n_rows * 0.28)
    fig, ax = plt.subplots(figsize=(15, fig_height))

    base_linewidth = 1.0
    location_linewidth = 5.0
    sns.heatmap(
        heatmap_numeric.fillna(0),
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=4,
        linewidths=base_linewidth,
        linecolor='white',
        cbar=False,
        xticklabels=True,
        yticklabels=True
    )

    dataset_labels = []
    for _ in locations:
        for model in available_models:
            if 'Forecast' in model:
                month = model.split(' ')[1]
                dataset_labels.append(f"{month} Forecast")
            else:
                dataset_labels.append(model)
    ax.set_yticklabels(dataset_labels, rotation=0)
    ax.tick_params(axis='y', labelsize=LABEL_FONTSIZE-4, pad=10)

    ax.tick_params(axis='x', labelsize=LABEL_FONTSIZE-2)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    ax.set_xlabel('Year', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('')

    group_boundaries = np.arange(len(available_models), n_rows, len(available_models))
    for boundary in group_boundaries:
        ax.hlines(boundary, *ax.get_xlim(), colors='white', linewidth=location_linewidth)

    # Add vertical location labels on the left margin
    def _wrap_location(name):
        # Special case for "Castel San Giovanni"
        if name == "Castel San Giovanni":
            return "Castel San\nGiovanni"
        parts = name.split()
        if len(parts) <= 1:
            return name
        return "\n".join(parts)

    wrapped_locations = [_wrap_location(loc) for loc in locations]

    for idx, loc in enumerate(wrapped_locations):
        start = idx * len(available_models)
        end = start + len(available_models)
        center = (start + end) / 2
        ax.text(-3.0, center, loc, rotation=0, va='center', ha='center',
                fontsize=LABEL_FONTSIZE, fontweight='bold', transform=ax.transData)

    legend_patches = [
        Patch(facecolor=outcome_colors['True Positive'], edgecolor='black', label='True Positive'),
        Patch(facecolor=outcome_colors['True Negative'], edgecolor='black', label='True Negative'),
        Patch(facecolor=outcome_colors['False Positive'], edgecolor='black', label='False Positive'),
        Patch(facecolor=outcome_colors['False Negative'], edgecolor='black', label='False Negative')
    ]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=LABEL_FONTSIZE-1)
    fig.suptitle('Performance Outcomes by Location and Data Source', fontsize=TITLE_FONTSIZE+2, y=1.02)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save in multiple high-quality formats for publication
    base_filename = "Figure5_Performance_Heatmap_AllSites_Stacked"
    
    # 1. High-resolution PNG (300 DPI - standard for publications)
    png_path = os.path.join(OUTPUT_DIR, f"{base_filename}.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    print(f"Merged performance heatmap figure saved to: {png_path} (300 DPI)")
    
    # 2. PDF (vector format - no quality loss, best for manuscripts)
    pdf_path = os.path.join(OUTPUT_DIR, f"{base_filename}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Merged performance heatmap figure saved to: {pdf_path} (vector format)")
    
    plt.close()

def create_heatmap_figure(yearly_outcomes, location=None):
    """Creates the final, single-panel heatmap figure. (Kept for backward compatibility)"""
    create_merged_heatmap_figure(yearly_outcomes)

def calculate_and_print_performance_metrics(df, location=None):
    """Calculates detailed performance metrics and prints them to the terminal."""
    # Filter by location if specified
    if location:
        df = df[df['location'] == location].copy()
    
    # --- 1. Pivot the data to get a wide-format table ---
    pivot_df = df[df['run_type'].isin(['station', 'era5*', 'forecast_mean'])].copy()
    pivot_df['model'] = pivot_df['run_type']
    fc_mask = pivot_df['run_type'] == 'forecast_mean'
    # Use forecast_month instead of forecast_timing for consistency
    pivot_df.loc[fc_mask, 'model'] = 'Forecast ' + pivot_df.loc[fc_mask, 'forecast_month'].astype(str)
    pivot_df['model'] = pivot_df['model'].replace({'station': 'Station', 'era5*': 'ERA5-Land*'})
    
    # Aggregate across locations if location=None (take mean), otherwise just pivot
    if location is None:
        # Aggregate across all locations by taking mean for each year-model combination
        data_wide = pivot_df.pivot_table(index='year', columns='model', values='afi_max', aggfunc='mean').dropna()
    else:
        # For single location, drop duplicates and pivot
        pivot_df.drop_duplicates(subset=['year', 'model'], inplace=True)
        data_wide = pivot_df.pivot_table(index='year', columns='model', values='afi_max').dropna()

    if 'Station' not in data_wide.columns:
        print("Error: 'Station' data not found for performance metric calculation.")
        if location:
            print(f"  Location: {location}")
        return
        
    observed_data = data_wide['Station']
    
    # --- 2. Calculate metrics for each model ---
    results = []
    for model_name in model_order:
        if model_name in data_wide.columns:
            predicted_data = data_wide[model_name]
            
            # Pearson Correlation (with p-value)
            r, p_value = pearsonr(observed_data, predicted_data)
            # Root Mean Square Error (RMSE)
            rmse = np.sqrt(mean_squared_error(observed_data, predicted_data))
            # Mean Absolute Error (MAE)
            mae = mean_absolute_error(observed_data, predicted_data)
            # Mean Bias (MB)
            bias = np.mean(predicted_data - observed_data)
            
            metrics = {
                'r': r,
                'p-value': p_value,
                'RMSE': rmse,
                'MAE': mae,
                'MBE': bias
            }
            
            display_model_name = model_name
            if 'Forecast' in model_name:
                display_model_name += ' (mean)'

            metrics['Comparison'] = f"{display_model_name} vs. Station"
            results.append(metrics)
            
    if not results:
        print("No performance metric results were calculated.")
        return
        
    # --- 3. Format and print the results ---
    results_df = pd.DataFrame(results).set_index('Comparison')
    
    location_label = f" for {location}" if location else " (All Sites)"
    print(f"\n--- Statistical Performance Metrics{location_label} ---")
    print(results_df.to_string(float_format="%.2f"))
    print("---------------------------------------")
    
    # --- 4. Save the metrics as CSV ---
    if location:
        csv_path = os.path.join(OUTPUT_DIR, f"performance_metrics_{location.replace(' ', '_')}.csv")
    else:
        csv_path = os.path.join(OUTPUT_DIR, "performance_metrics_AllSites.csv")
    results_df.to_csv(csv_path)
    print(f"Performance metrics CSV saved to: {csv_path}")
    
    # --- 5. Save the metrics as a figure ---
    if location:
        figure_path = os.path.join(OUTPUT_DIR, f"performance_metrics_table_{location.replace(' ', '_')}.png")
    else:
        figure_path = os.path.join(OUTPUT_DIR, "performance_metrics_table_AllSites.png")
    save_metrics_as_figure(results_df, figure_path, location)

def save_metrics_as_figure(df, filepath, location=None):
    """Saves the metrics DataFrame as a clean, styled table image."""
    fig, ax = plt.subplots(figsize=(12, 4)) # Adjusted size for better fit
    ax.axis('tight')
    ax.axis('off')

    # Format the dataframe for display
    df_display = df.round(2)

    # Create the table
    table = ax.table(cellText=df_display.values,
                     colLabels=df_display.columns,
                     rowLabels=df_display.index,
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.set_fontsize(12)
    table.scale(1.1, 1.3)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(fontweight='bold')
    
    title = 'Statistical Performance Metrics'
    if location:
        title += f' - {location}'
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=20)
    plt.savefig(filepath, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Metrics table figure saved to: {filepath}")

def create_merged_summary_table_figure(df):
    """Creates a stacked table (no heatmap) with locations grouped like the heatmap layout."""
    plot_df = df[df['run_type'].isin(['station', 'era5*', 'forecast_mean'])].copy()
    plot_df['model'] = plot_df['run_type']
    forecast_mask = plot_df['run_type'] == 'forecast_mean'
    plot_df.loc[forecast_mask, 'model'] = plot_df.loc[forecast_mask, 'forecast_month'] + ' Forecast'
    plot_df['model'] = plot_df['model'].replace({'station': 'Station', 'era5*': 'ERA5-Land*'})

    location_order = [
        'Castel San Giovanni', 'Caorso', 'Roncopascolo', 'Luzzara', 'Mirandola',
        'Castel Maggiore', 'Guarda Ferrarese', 'Ostellato', 'Medicina', 'Forli'
    ]
    available_locations = list(plot_df['location'].unique())
    locations = [loc for loc in location_order if loc in available_locations]
    remaining = [loc for loc in available_locations if loc not in locations]
    locations.extend(sorted(remaining))

    years = sorted(plot_df['year'].unique())
    available_models = [m for m in [
        'Station', 'ERA5-Land*', 'April Forecast', 'June Forecast',
        'July Forecast', 'August Forecast'
    ] if m in plot_df['model'].unique()]

    pivot_table = plot_df.pivot_table(index=['location', 'model'], columns='year', values='afi_max')
    desired_index = pd.MultiIndex.from_product([locations, available_models], names=['location', 'model'])
    pivot_table = pivot_table.reindex(desired_index)
    pivot_table = pivot_table.reindex(columns=years)

    table_data = np.round(pivot_table.values, 2)
    col_labels = [str(y) for y in pivot_table.columns]
    
    # Show data source names in row labels
    row_labels = []
    location_col = []  # Location column for CSV
    for loc in locations:
        for model in available_models:
            row_labels.append(model)
            location_col.append(loc)  # Repeat location 5 times per location

    # Wrap location names (same logic as heatmap)
    def _wrap_location(name):
        # Special case for "Castel San Giovanni"
        if name == "Castel San Giovanni":
            return "Castel San\nGiovanni"
        parts = name.split()
        if len(parts) <= 1:
            return name
        return "\n".join(parts)

    n_rows = len(row_labels)
    fig_height = max(10, n_rows * 0.4)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # Create table with data source names in row labels
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     rowLabels=row_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.15, 0, 0.85, 1])  # Leave space on left for location labels
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.15)

    # Style the table - clean summary table without location separators
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Column headers
            cell.set_text_props(weight='bold')
        if col == -1:  # Row labels
            cell.set_text_props(weight='bold')

    # Location names are now included in the CSV export instead of on the figure

    ax.set_title('Yearly Maximum AFI Summary - All 10 Test Sites', fontsize=TITLE_FONTSIZE+2, pad=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    output_path = os.path.join(OUTPUT_DIR, "Figure6_AFI_Max_Summary_Table_AllSites_Stacked.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Merged summary table saved to: {output_path}")
    
    # Save as CSV with Location and Data Source columns
    csv_df = pd.DataFrame({
        'Location': location_col,
        'Data Source': row_labels
    })
    # Add year columns
    for year_idx, year in enumerate(years):
        csv_df[str(year)] = table_data[:, year_idx]
    
    csv_output_path = os.path.join(OUTPUT_DIR, "AFI_Max_Summary_Table_AllSites.csv")
    csv_df.to_csv(csv_output_path, index=False)
    print(f"Summary table CSV saved to: {csv_output_path}")

def create_summary_table_figure(df, location=None):
    """Creates a figure with a summary table of max AFI values. (Kept for backward compatibility)"""
    if location:
        # For individual location, use original logic
        plot_df = df[df['location'] == location].copy()
        plot_df = plot_df[plot_df['run_type'].isin(['station', 'era5*', 'forecast_mean'])].copy()
        plot_df['model'] = plot_df['run_type']
        forecast_mask = plot_df['run_type'] == 'forecast_mean'
        plot_df.loc[forecast_mask, 'model'] = 'Forecast ' + plot_df.loc[forecast_mask, 'forecast_month'] + ' (mean)'
        plot_df['model'] = plot_df['model'].replace({'station': 'Station', 'era5*': 'ERA5-Land*'})
        summary_table = plot_df.pivot_table(index='year', columns='model', values='afi_max')
        column_order = [m for m in ['Station', 'ERA5-Land*', 'Forecast April (mean)', 'Forecast June (mean)', 'Forecast July (mean)', 'Forecast August (mean)'] if m in summary_table.columns]
        summary_table = summary_table[column_order]
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        table_data = np.round(summary_table.values, 2)
        table = ax.table(cellText=table_data, colLabels=summary_table.columns, rowLabels=summary_table.index, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 1.5)
        for (row, col), cell in table.get_celld().items():
            if (row == 0):
                cell.set_text_props(weight='bold')
        ax.set_title(f'Yearly Maximum AFI Summary - {location}', fontsize=TITLE_FONTSIZE, pad=20)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"Figure6_AFI_Max_Summary_Table_{location.replace(' ', '_')}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Summary table figure saved to: {output_path}")
    else:
        create_merged_summary_table_figure(df)

# --- Step 4: Main Execution ---

def main():
    # Set output directory to summary CSV file directory
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.dirname(SUMMARY_CSV_PATH) if os.path.dirname(SUMMARY_CSV_PATH) else os.getcwd()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    """Main function to run the analysis and generate the final figure."""
    print("Starting Comprehensive AFLA-maize Performance Analysis for 10 Test Sites...")
    df = load_and_prepare_data()
    
    if df.empty:
        print("Could not load or process data. Aborting.")
        return
    
    # Get unique locations
    locations = df['location'].unique() if 'location' in df.columns else []
    print(f"\nFound {len(locations)} locations in the data: {', '.join(locations)}")
    
    # Create merged analysis for all locations
    print(f"\n{'='*60}")
    print("Creating merged analysis for all 10 test sites...")
    print(f"{'='*60}")
    
    # Get outcomes for all locations combined
    yearly_outcomes_all, model_performance_all = get_yearly_outcomes_all_locations(df)
    
    if yearly_outcomes_all.empty:
        print("  Error: No valid data found for merged analysis.")
        return
    
    # Print the calculated performance metrics to the terminal
    print("\n--- Model Performance (Contingency Matrix Metrics) - All Sites Combined ---")
    perf_df = pd.DataFrame.from_dict(model_performance_all, orient='index')
    
    # Reorder to show Station first, then other models
    if 'Station' in perf_df.index:
        station_row = perf_df.loc[['Station']]
        other_rows = perf_df.drop('Station')
        perf_df = pd.concat([station_row, other_rows])
    
    # Store FPR and FNR for separate printing
    fpr_fnr_data = {}
    if 'False Positive Rate' in perf_df.columns:
        fpr_fnr_data['FPR'] = perf_df['False Positive Rate']
    if 'False Negative Rate' in perf_df.columns:
        fpr_fnr_data['FNR'] = perf_df['False Negative Rate']
    
    # Remove FPR and FNR from main table (they're redundant with Specificity and Sensitivity)
    perf_df_display = perf_df.drop(columns=['False Positive Rate', 'False Negative Rate'], errors='ignore')
    
    perf_df_display['Accuracy'] = perf_df_display['Accuracy'].map('{:.2f}%'.format)
    perf_df_display['Sensitivity'] = perf_df_display['Sensitivity'].map('{:.2f}'.format)
    perf_df_display['Specificity'] = perf_df_display['Specificity'].map('{:.2f}'.format)
    if 'F1 Score' in perf_df_display.columns:
        perf_df_display['F1 Score'] = perf_df_display['F1 Score'].map('{:.2f}'.format)
    print(perf_df_display)
    
    # Print FPR and FNR separately for reference
    if fpr_fnr_data:
        print("\nAdditional Metrics (for reference):")
        fpr_fnr_df = pd.DataFrame(fpr_fnr_data)
        fpr_fnr_df['FPR'] = fpr_fnr_df['FPR'].map('{:.2f}'.format)
        fpr_fnr_df['FNR'] = fpr_fnr_df['FNR'].map('{:.2f}'.format)
        print(fpr_fnr_df)
        print("(Note: FPR = 1 - Specificity, FNR = 1 - Sensitivity)")
    
    # Print contingency matrix counts (TP, TN, FP, FN)
    print("\nContingency Matrix Counts (TP, TN, FP, FN):")
    counts_data = {}
    if 'TP' in perf_df.columns:
        counts_data['TP'] = perf_df['TP'].astype(int)
    if 'TN' in perf_df.columns:
        counts_data['TN'] = perf_df['TN'].astype(int)
    if 'FP' in perf_df.columns:
        counts_data['FP'] = perf_df['FP'].astype(int)
    if 'FN' in perf_df.columns:
        counts_data['FN'] = perf_df['FN'].astype(int)
    
    if counts_data:
        counts_df = pd.DataFrame(counts_data)
        print(counts_df)
        print("(TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative)")
    
    print("------------------------------------------------------\n")
    
    # Save contingency metrics table
    create_contingency_metrics_table(model_performance_all, None, is_aggregate=True)
    
    # Create merged heatmap
    create_merged_heatmap_figure(yearly_outcomes_all)
    
    # Create merged summary table
    create_merged_summary_table_figure(df)
    
    # Calculate and print performance metrics for all sites
    calculate_and_print_performance_metrics(df, location=None)
    
    print("\nAnalysis complete.")

def create_contingency_metrics_table(performance_data, location=None, is_aggregate=False):
    """Saves the contingency matrix performance metrics as a styled table image."""
    # Convert the dictionary to a DataFrame
    if isinstance(performance_data, dict):
        perf_df = pd.DataFrame.from_dict(performance_data, orient='index')
    else:
        perf_df = performance_data
    
    # Remove FPR and FNR from table/CSV (they're redundant with Specificity and Sensitivity)
    perf_df_display = perf_df.drop(columns=['False Positive Rate', 'False Negative Rate'], errors='ignore')
    
    # Store numeric version for CSV export (rounded to two decimals) - without FPR/FNR
    perf_df_csv = perf_df_display.copy().applymap(lambda x: round(x, 2) if isinstance(x, (int, float, np.floating)) else x)
    
    # Format the values for display
    if 'Accuracy' in perf_df_display.columns:
        perf_df_display['Accuracy'] = perf_df_display['Accuracy'].map('{:.2f}%'.format)
    if 'Sensitivity' in perf_df_display.columns:
        perf_df_display['Sensitivity'] = perf_df_display['Sensitivity'].map('{:.2f}'.format)
    if 'Specificity' in perf_df_display.columns:
        perf_df_display['Specificity'] = perf_df_display['Specificity'].map('{:.2f}'.format)
    if 'F1 Score' in perf_df_display.columns:
        perf_df_display['F1 Score'] = perf_df_display['F1 Score'].map('{:.2f}'.format)
    
    fig, ax = plt.subplots(figsize=(12, 4)) # Width adjusted for main metrics columns
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=perf_df_display.values,
                     colLabels=perf_df_display.columns,
                     rowLabels=perf_df_display.index,
                     cellLoc='center',
                     loc='center')
                     
    # Style the table
    table.set_fontsize(14)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(fontweight='bold')
            
    title = 'Model Performance Metrics'
    if is_aggregate:
        title += ' (Aggregate - All Sites)'
    elif location:
        title += f' - {location}'
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=20)
    
    plt.tight_layout()
    
    if is_aggregate:
        output_path = os.path.join(OUTPUT_DIR, "contingency_metrics_table_AllSites.png")
        csv_output_path = os.path.join(OUTPUT_DIR, "contingency_metrics_AllSites.csv")
    elif location:
        output_path = os.path.join(OUTPUT_DIR, f"contingency_metrics_table_{location.replace(' ', '_')}.png")
        csv_output_path = os.path.join(OUTPUT_DIR, f"contingency_metrics_{location.replace(' ', '_')}.csv")
    else:
        output_path = os.path.join(OUTPUT_DIR, "contingency_metrics_table.png")
        csv_output_path = os.path.join(OUTPUT_DIR, "contingency_metrics.csv")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Contingency metrics table saved to: {output_path}")
    
    perf_df_csv.to_csv(csv_output_path)
    print(f"Contingency metrics CSV saved to: {csv_output_path}")

if __name__ == "__main__":
    main()

