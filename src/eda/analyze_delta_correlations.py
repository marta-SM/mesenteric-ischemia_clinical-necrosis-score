import os
import pandas as pd
import numpy as np
from scipy import stats

# Rutas adaptadas a la nueva estructura del proyecto
INPUT_CSV = "data/processed/Matrice_Delais_IMA_Clean.csv"
OUTPUT_MATRIX = "reports/Delta_Inter_Correlation_Matrix.csv"
OUTPUT_PAIRWISE = "reports/Delta_Pairwise_Correlations.csv"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"File '{INPUT_CSV}' not found. Please run process_dates.py first.")

# Asegurar que la carpeta de destino para los reportes existe
os.makedirs(os.path.dirname(OUTPUT_MATRIX), exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# Select only delta variables
delta_cols = [c for c in df.columns if c.startswith('Delta_')]

print("==================================================")
print(" INTER-DELTA CORRELATION MATRIX (SPEARMAN RHO) ")
print("==================================================\n")

# Compute Spearman correlation matrix (pairwise complete observations)
spearman_matrix = df[delta_cols].corr(method='spearman')

# Format and display correlation matrix
print(spearman_matrix.round(3).to_string())

# Save correlation matrix to CSV
spearman_matrix.round(3).to_csv(OUTPUT_MATRIX)
print(f"\n[SUCCESS] Correlation matrix saved to '{OUTPUT_MATRIX}'")

# 2. Detailed pairwise correlation with p-values
pairwise_results = []

for i in range(len(delta_cols)):
    for j in range(i + 1, len(delta_cols)):
        col1 = delta_cols[i]
        col2 = delta_cols[j]
        
        # Pairwise complete cases
        sub_df = df[[col1, col2]].dropna()
        
        if len(sub_df) > 0:
            rho, p_val = stats.spearmanr(sub_df[col1], sub_df[col2])
            pairwise_results.append({
                'Variable_1': col1,
                'Variable_2': col2,
                'N_Obs': len(sub_df),
                'Spearman_rho': round(rho, 3),
                'p_value': round(p_val, 4)
            })

pairwise_df = pd.DataFrame(pairwise_results)
pairwise_df.to_csv(OUTPUT_PAIRWISE, index=False)

print("\n--- Top Pairwise Collinearities ---")
print(pairwise_df.sort_values(by='Spearman_rho', ascending=False).to_string(index=False))
print(f"\n[SUCCESS] Pairwise analysis saved to '{OUTPUT_PAIRWISE}'")