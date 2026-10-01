import os
import pandas as pd
import numpy as np

INPUT_CSV = "data/processed/Matrice_Complete_IMA_Clean.csv"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"File '{INPUT_CSV}' not found. Please run process_dates.py first.")

df = pd.read_csv(INPUT_CSV)

print("==================================================")
print(" CLINICAL VARIABLE AUDIT & QUALITY CONTROL ")
print(f" Total Patients: {len(df)} | Total Columns: {len(df.columns)}")
print("==================================================\n")

# Exclusion candidates list
exclude_identifiers = ['Patient_ID', 'Necrose', 'Necrose_peritonitis', 'Necrose_purs']

# 1. Text / Object columns (non-numeric)
non_numeric_cols = []
for col in df.columns:
    if col not in exclude_identifiers and not col.startswith('Date_') and not col.startswith('date_'):
        # Check if non-numeric
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.isna().sum() == len(df):
            non_numeric_cols.append((col, df[col].dropna().unique()[:3]))

# 2. High missing value columns (> 40% missing)
high_missing_cols = []
for col in df.columns:
    if col not in exclude_identifiers:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 40:
            high_missing_cols.append((col, round(missing_pct, 1)))

# 3. Low variance / Constant columns
constant_cols = []
for col in df.columns:
    if col not in exclude_identifiers:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            constant_cols.append((col, n_unique))

# Display Results
print("--- 1. TEXT / CATEGORICAL / NON-NUMERIC COLUMNS ---")
if non_numeric_cols:
    for col, samples in non_numeric_cols:
        print(f"  • {col}: Sample values -> {list(samples)}")
else:
    print("  None found.")

print("\n--- 2. HIGH MISSING VALUES COLUMNS (> 40% NaN) ---")
if high_missing_cols:
    for col, pct in sorted(high_missing_cols, key=lambda x: x[1], reverse=True):
        print(f"  • {col}: {pct}% missing")
else:
    print("  None found.")

print("\n--- 3. CONSTANT / ZERO VARIANCE COLUMNS ---")
if constant_cols:
    for col, _ in constant_cols:
        print(f"  • {col}")
else:
    print("  None found.")