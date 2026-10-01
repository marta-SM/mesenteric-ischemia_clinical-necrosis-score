import os
import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

INPUT_CSV = "data/processed/Matrice_Complete_IMA_Clean.csv"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"File '{INPUT_CSV}' not found. Please run process_dates.py first.")

df = pd.read_csv(INPUT_CSV)

# Target configuration
target_col = 'Necrose'
targets = ['Necrose', 'Necrose_peritonitis', 'Necrose_purs']

# Clean target and drop the patient with missing outcome (do not fillna(0) here)
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df[df[target_col].notna()].copy()
df[target_col] = df[target_col].astype(int)

# --- Step 1: Strict Pre-operative Filtering ---

# Post-outcome variables causing DATA LEAKAGE + non-predictive identifiers
post_treatment_leakage = [
    'Resection', 'N_resection', 'J_resection', 'Date_resection', 'dates_resection',
    'Realim_J5', 'DouleurJ5_persist_ou_aggrav', 'DouleurJ5_aggrav', 'sorti_d\'hospit', 'DCD',
    'durée d\'hospit J',
]

explicit_ignore = ['Patient_ID', 'Unnamed: 165'] + targets + post_treatment_leakage

# Pattern-based exclusion for raw dates and comments
raw_date_cols = [c for c in df.columns if ('date' in c.lower() or 'date_' in c.lower()) and not c.startswith('Delta_')]
comment_cols = [c for c in df.columns if 'comment' in c.lower() or 'info' in c.lower() or c in ['HOS_SERVICE_L1', 'SGC_def']]

all_ignore_cols = list(set(explicit_ignore + raw_date_cols + comment_cols))

# Retain predictive pre-operative features
features = [c for c in df.columns if c not in all_ignore_cols]

print(f"Loaded dataset: {df.shape[0]} patients, {df.shape[1]} total columns")
print(f"Excluded {len(post_treatment_leakage)} post-operative leakage variables.")

# --- Step 2: Quality Filtering & Data Type Enforcement ---

for c in features:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Filter A: Drop features with > 90% missing values
missing_pct = df[features].isna().mean()
high_missing = missing_pct[missing_pct > 0.90].index.tolist()
features = [c for c in features if c not in high_missing]

# Filter B: Drop features that are entirely NaN
all_nan = [c for c in features if df[c].isna().sum() == len(df)]
features = [c for c in features if c not in all_nan]

# Filter C: Drop zero-variance / constant features
variances = df[features].var(skipna=True)
constant_features = variances[variances == 0].index.tolist()
features = [c for c in features if c not in constant_features]

print(f"Final pre-operative predictors evaluated by Lasso L1: {len(features)}")

X = df[features].values
y = df[target_col].values

# --- Step 3: LOOCV Training Pipeline ---
loo = LeaveOneOut()
y_true = []
y_pred_prob = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    clf = LogisticRegressionCV(
        solver='liblinear', 
        cv=5, 
        l1_ratios=[1.0], 
        scoring='neg_log_loss',
        random_state=42, 
        max_iter=2000
    )
    clf.fit(X_train_scaled, y_train)
    
    prob = clf.predict_proba(X_test_scaled)[0, 1]
    y_pred_prob.append(prob)
    y_true.append(y_test[0])

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred_bin = (y_pred_prob >= 0.5).astype(int)

# Metrics
auc = roc_auc_score(y_true, y_pred_prob)
acc = accuracy_score(y_true, y_pred_bin)
prec = precision_score(y_true, y_pred_bin, zero_division=0)
rec = recall_score(y_true, y_pred_bin, zero_division=0)
f1 = f1_score(y_true, y_pred_bin, zero_division=0)

print("\n==================================================")
print(" LOOCV METRICS (Strict Pre-operative Model)")
print("==================================================")
print(f" ROC-AUC  : {auc:.3f}")
print(f" Accuracy : {acc:.3f}")
print(f" Precision: {prec:.3f}")
print(f" Recall    : {rec:.3f}")
print(f" F1-Score : {f1:.3f}")
print("==================================================\n")

# --- Step 4: Final Feature Importance ---
imputer_full = SimpleImputer(strategy='median')
X_full_imp = imputer_full.fit_transform(X)

scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full_imp)

model_full = LogisticRegressionCV(
    solver='liblinear', 
    cv=5, 
    l1_ratios=[1.0], 
    scoring='neg_log_loss',
    random_state=42, 
    max_iter=2000
)
model_full.fit(X_full_scaled, y)

coefs = model_full.coef_[0]
selected_indices = np.where(coefs != 0)[0]

selected_df = pd.DataFrame({
    'Feature': [features[i] for i in selected_indices],
    'Coefficient': coefs[selected_indices],
    'Abs_Coefficient': np.abs(coefs[selected_indices])
}).sort_values(by='Abs_Coefficient', ascending=False)

print("--- PRE-OPERATIVE VARIABLES SELECTED BY LASSO L1 ---")
if not selected_df.empty:
    print(selected_df[['Feature', 'Coefficient']].to_string(index=False))
else:
    print("No features selected.")