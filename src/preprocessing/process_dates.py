import os
import pandas as pd
import numpy as np

# 1. Path configurations
INPUT_FILE = "data/raw/Base_IMA_veineux_pseudonymise.xlsx"
OUTPUT_CSV = "data/processed/Matrice_Complete_IMA_Clean.csv"
OUTPUT_EXCEL = "data/processed/Matrice_Complete_IMA_Clean.xlsx"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Source file '{INPUT_FILE}' not found in data/raw/.")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Load full raw dataset
df_raw = pd.read_excel(INPUT_FILE)

# 2. Dynamic date column detection
def is_date_column(series):
    non_nulls = series.dropna()
    if non_nulls.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample_str = non_nulls.astype(str).iloc[0].strip()
    if '/' in sample_str or '-' in sample_str:
        parsed = pd.to_datetime(non_nulls, format='mixed', dayfirst=False, errors='coerce')
        return parsed.notna().sum() > 0
    return False

date_columns = [col for col in df_raw.columns if is_date_column(df_raw[col])]

# 3. Parse date columns in memory for calculation
parsed_dates_df = pd.DataFrame()
for col in date_columns:
    parsed_dates_df[col] = pd.to_datetime(df_raw[col], format='mixed', dayfirst=False, errors='coerce')

# 4. Calculate clinical time deltas relative to initial functional symptoms
sf_col = [c for c in date_columns if 'prem_SF' in c]

computed_deltas = pd.DataFrame(index=df_raw.index)

if sf_col:
    base_sf_col = sf_col[0]
    base_sf_dates = parsed_dates_df[base_sf_col]
    
    target_events = {
        'diagnostic': 'Delta_SF_to_Diagnostic_days',
        'PEC': 'Delta_SF_to_PEC_days',
        'CT1': 'Delta_SF_to_CT1_days',
        'bio': 'Delta_SF_to_Bio_days',
        'BJN': 'Delta_SF_to_BJN_days'
    }
    
    for key, delta_name in target_events.items():
        matching_cols = [c for c in date_columns if key.lower() in c.lower() and c != base_sf_col]
        if matching_cols:
            event_col = matching_cols[0]
            event_dates = parsed_dates_df[event_col]
            
            days_series = (event_dates - base_sf_dates).dt.days
            
            # Clean invalid negative values
            negative_mask = days_series < 0
            if negative_mask.sum() > 0:
                affected = df_raw.loc[negative_mask, 'Patient_ID'].tolist() if 'Patient_ID' in df_raw.columns else []
                print(f"[WARNING] Negative values found in {delta_name} for patients: {affected}. Set to NaN.")
                days_series.loc[negative_mask] = pd.NA
                
            computed_deltas[delta_name] = days_series

# 5. Concatenate all raw columns with computed deltas
existing_delta_cols = [c for c in df_raw.columns if c.startswith('Delta_')]
df_cleaned = df_raw.drop(columns=existing_delta_cols, errors='ignore')

df_final = pd.concat([df_cleaned, computed_deltas], axis=1)

# Export consolidated master dataset
df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
df_final.to_excel(OUTPUT_EXCEL, index=False)

print("\n--- MASTER DATASET PROCESSING COMPLETED ---")
print(f"Total rows (patients): {len(df_final)}")
print(f"Total columns preserved: {len(df_final.columns)}")
print(f"Exported to: {OUTPUT_EXCEL}")