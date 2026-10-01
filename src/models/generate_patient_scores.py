import os
import argparse
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

## How to launch: python src/models/generate_patient_scores.py -n_var 3 or 5 or as many variables you want to be considered in the score

# Command line argument parser for number of variables (-n_var)
parser = argparse.ArgumentParser(description="Generate patient risk scores based on top Lasso L1 features.")
parser.add_argument("-n_var", type=int, default=3, help="Number of top features to select from Lasso L1 (default: 3)")
args = parser.parse_args()

INPUT_CSV = "data/processed/Matrice_Complete_IMA_Clean.csv"
OUTPUT_SCORES_CSV = f"reports/Patient_Level_Scores_n{args.n_var}.csv"
OUTPUT_SCORES_EXCEL = f"reports/Patient_Level_Scores_n{args.n_var}.xlsx"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"File '{INPUT_CSV}' not found. Please run data preparation first.")

df = pd.read_csv(INPUT_CSV)

if 'Patient_ID' not in df.columns:
    df['Patient_ID'] = [f"PAT_{i+1:03d}" for i in range(len(df))]

# Clean target
target_col = 'Necrose'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)

# Exclude non-predictive or leakage columns (same pipeline rules)
post_treatment_leakage = [
    'Resection', 'N_resection', 'J_resection', 'Date_resection', 'dates_resection',
    'Realim_J5', 'DouleurJ5_persist_ou_aggrav', 'DouleurJ5_aggrav', 'sorti_d\'hospit', 'DCD'
]
explicit_ignore = ['Patient_ID', 'Unnamed: 165', target_col, 'Necrose_peritonitis', 'Necrose_purs']
raw_date_cols = [c for c in df.columns if ('date' in c.lower() or 'date_' in c.lower()) and not c.startswith('Delta_')]
comment_cols = [c for c in df.columns if 'comment' in c.lower() or 'info' in c.lower() or c in ['HOS_SERVICE_L1', 'SGC_def']]

all_ignore_cols = list(set(explicit_ignore + post_treatment_leakage + raw_date_cols + comment_cols))
features = [c for c in df.columns if c not in all_ignore_cols]

# Numeric coercion & missing/constant filtering
for c in features:
    df[c] = pd.to_numeric(df[c], errors='coerce')

missing_pct = df[features].isna().mean()
features = [c for c in features if missing_pct[c] <= 0.90]
features = [c for c in features if df[c].isna().sum() < len(df)]
variances = df[features].var(skipna=True)
features = [c for c in features if variances[c] > 0]

X = df[features].values
y = df[target_col].values

# --- 1. FIT LASSO L1 TO EXTRACT RANKED COEFFICIENTS ---
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

model_full = LogisticRegressionCV(
    solver='liblinear', 
    cv=5, 
    l1_ratios=[1.0], 
    scoring='neg_log_loss',
    random_state=42, 
    max_iter=2000
)
model_full.fit(X_scaled, y)

coefs = model_full.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefs,
    'Abs_Coefficient': np.abs(coefs)
}).sort_values(by='Abs_Coefficient', ascending=False)

# Filter out zero coefficients and select top N variables
non_zero_importance = feature_importance[feature_importance['Coefficient'] != 0]
top_features = non_zero_importance.head(args.n_var)['Feature'].tolist()

print("==================================================")
print(f" TOP {args.n_var} FEATURES SELECTED DYNAMICALLY BY LASSO L1")
print("==================================================")
print(non_zero_importance.head(args.n_var)[['Feature', 'Coefficient']].to_string(index=False))
print("==================================================\n")

# --- 2. COMPUTE PATIENT CLINICAL SCORE ---
# Assign 1 point per positive criterion for the chosen top N features
df['Total_Score'] = 0
for feat in top_features:
    df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
    # Binary conversion if continuous (values > 0 get 1 point)
    binary_feat = (df[feat] > 0).astype(int)
    df['Total_Score'] += binary_feat

# Cutoff rule: Score >= 2 is predicted as High Risk / Necrosis (adjustable) 
# This cutoff can be modified
cutoff = 2 if args.n_var >= 3 else 1
df['Predicted_Necrosis'] = (df['Total_Score'] >= cutoff).astype(int)
df['Actual_Necrosis'] = df[target_col]

df['Outcome_Classification'] = np.where(
    df['Predicted_Necrosis'] == df['Actual_Necrosis'],
    'CORRECT',
    np.where((df['Predicted_Necrosis'] == 1) & (df['Actual_Necrosis'] == 0), 'FALSE POSITIVE', 'FALSE NEGATIVE')
)

# --- 3. EXPORT RESULTS ---
output_cols = ['Patient_ID'] + top_features + ['Total_Score', 'Predicted_Necrosis', 'Actual_Necrosis', 'Outcome_Classification']
results_df = df[output_cols]

os.makedirs(os.path.dirname(OUTPUT_SCORES_CSV), exist_ok=True)
results_df.to_csv(OUTPUT_SCORES_CSV, index=False, encoding='utf-8-sig')
results_df.to_excel(OUTPUT_SCORES_EXCEL, index=False)

print("==================================================")
print(f" EVALUATION SUMMARY (Top {args.n_var} Features, Cutoff >= {cutoff})")
print("==================================================")
print(f"Total Patients Evaluated: {len(results_df)}")
print(f"Correct Classifications: {(results_df['Outcome_Classification'] == 'CORRECT').sum()} / {len(results_df)} "
      f"({(results_df['Outcome_Classification'] == 'CORRECT').mean()*100:.1f}%)")
print(f"False Positives: {(results_df['Outcome_Classification'] == 'FALSE POSITIVE').sum()}")
print(f"False Negatives: {(results_df['Outcome_Classification'] == 'FALSE NEGATIVE').sum()}")
print("==================================================\n")

print(f"[SUCCESS] Exported reports to:\n  - {OUTPUT_SCORES_CSV}\n  - {OUTPUT_SCORES_EXCEL}")