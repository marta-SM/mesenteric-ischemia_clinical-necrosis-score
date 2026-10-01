import os
import pandas as pd
import numpy as np
from scipy import stats

INPUT_CSV = "data/processed/Matrice_Complete_IMA_Clean.csv"
OUTPUT_CSV = "reports/Univariate_Analysis_Summary.csv"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"File '{INPUT_CSV}' not found. Please run process_dates.py first.")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

df = pd.read_csv(INPUT_CSV)
target_col = 'Necrose'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# --- Exclude identifiers, targets, and post-event leakage variables ---
# (same exclusion logic as train_lasso.py, for consistency across the pipeline)
other_targets = ['Necrose_peritonitis', 'Necrose_purs']
post_treatment_leakage = [
    'Resection', 'N_resection', 'J_resection', 'Date_resection', 'dates_resection',
    'Realim_J5', 'DouleurJ5_persist_ou_aggrav', 'DouleurJ5_aggrav', "sorti_d'hospit", 'DCD'
]
explicit_ignore = ['Patient_ID', 'Unnamed: 165', target_col] + other_targets + post_treatment_leakage
raw_date_cols = [c for c in df.columns if ('date' in c.lower()) and not c.startswith('Delta_')]
comment_cols = [c for c in df.columns if 'comment' in c.lower() or 'info' in c.lower()
                or c in ['HOS_SERVICE_L1', 'SGC_def']]

all_ignore_cols = set(explicit_ignore + raw_date_cols + comment_cols)
candidate_cols = [c for c in df.columns if c not in all_ignore_cols]

# Drop the one patient with missing outcome
df_valid = df[df[target_col].notna()].copy()
df_valid[target_col] = df_valid[target_col].astype(int)

results = []

for col in candidate_cols:
    series = pd.to_numeric(df_valid[col], errors='coerce')
    sub = pd.DataFrame({'y': df_valid[target_col], 'x': series}).dropna()

    if len(sub) < 5:
        continue

    group_0 = sub.loc[sub['y'] == 0, 'x']
    group_1 = sub.loc[sub['y'] == 1, 'x']

    if len(group_0) == 0 or len(group_1) == 0:
        continue

    n_unique = sub['x'].nunique()

    if n_unique <= 2:
        # Binary variable -> Fisher's exact test
        a = (group_1 == 1).sum()
        b = (group_1 == 0).sum()
        c = (group_0 == 1).sum()
        d = (group_0 == 0).sum()
        table = [[a, b], [c, d]]
        _, p_value = stats.fisher_exact(table)
        test_used = 'Fisher'
        summary_0 = f"{int(c)}/{len(group_0)} ({c / len(group_0) * 100:.0f}%)"
        summary_1 = f"{int(a)}/{len(group_1)} ({a / len(group_1) * 100:.0f}%)"
    else:
        # Continuous variable -> Mann-Whitney U
        _, p_value = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
        test_used = 'Mann-Whitney'
        summary_0 = f"{group_0.median():.1f} [{group_0.quantile(.25):.1f}-{group_0.quantile(.75):.1f}]"
        summary_1 = f"{group_1.median():.1f} [{group_1.quantile(.25):.1f}-{group_1.quantile(.75):.1f}]"

    # Variables kept in the univariate comparison (to match Paul's report) but
    # flagged as consequence-of-outcome rather than predictors, and excluded from Lasso
    leakage_risk_cols = ["durée d'hospit J"]  # ajustar al nombre exacto de columna

    results.append({
        'Variable': col,
        'Test': test_used,
        'N_valid': len(sub),
        'Pct_missing': round((1 - len(sub) / len(df_valid)) * 100, 1),
        'Group_Necrose_0': summary_0,
        'Group_Necrose_1': summary_1,
        'p_value': round(p_value, 4)
    })

results_df = pd.DataFrame(results).sort_values('p_value')
results_df.to_csv(OUTPUT_CSV, index=False)

if any(results_df['Variable'].isin(leakage_risk_cols)):
    print("\n[NOTE] 'durée d'hospit J' is included for comparison with Paul's univariate "
          "table, but is excluded from the Lasso model: hospital stay length is more "
          "likely a consequence of necrosis than a predictor of it.")

print("==================================================")
print(f" UNIVARIATE ANALYSIS — {len(results_df)} variables tested")
print("==================================================\n")
print(results_df.head(20).to_string(index=False))
print(f"\n[SUCCESS] Full results saved to '{OUTPUT_CSV}'")
print("\n[NOTE] With ~150 variables tested, expect ~7-8 significant results by chance alone (alpha=0.05).")