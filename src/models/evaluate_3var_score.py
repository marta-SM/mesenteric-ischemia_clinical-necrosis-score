import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("data/processed/Matrice_Complete_IMA_Clean.csv")

# Clean inputs
df['Morphine_survibase'] = pd.to_numeric(df['Morphine_survibase'], errors='coerce').fillna(0)
df['CTmax_intestinal_perforation'] = pd.to_numeric(df['CTmax_intestinal_perforation'], errors='coerce').fillna(0)
df['Lactate_H48_15'] = pd.to_numeric(df['Lactate_H48_15'], errors='coerce').fillna(0)

# Compute 3-Point Clinical Score
df['Clinical_Score'] = (
    (df['Morphine_survibase'] * 2) +
    (df['CTmax_intestinal_perforation'] * 2) +
    (df['Lactate_H48_15'] * 1)
)

auc = roc_auc_score(df['Necrose'], df['Clinical_Score'])

print(f"=== SCORE CLÍNICO DE 3 VARIABLES (10 EPV) ===")
print(f"ROC-AUC del Score Simplificado: {auc:.3f}")
print("\nDistribución de necrosis por Puntuación:")
print(pd.crosstab(df['Clinical_Score'], df['Necrose'], normalize='index') * 100)