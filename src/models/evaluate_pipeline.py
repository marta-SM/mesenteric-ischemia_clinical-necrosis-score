import os
import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

INPUT_CSV = "data/processed/Matrice_Complete_IMA_Clean.csv"
OUTPUT_CSV = "reports/Pipeline_Evaluation_Summary.csv"
N_SPLITS = 5
N_REPEATS = 20
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

df = pd.read_csv(INPUT_CSV)
target_col = 'Necrose'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df[df[target_col].notna()].copy()
df[target_col] = df[target_col].astype(int)

# --- Same exclusion logic as train_lasso.py (keep in sync) ---
other_targets = ['Necrose_peritonitis', 'Necrose_purs']
post_treatment_leakage = [
    'Resection', 'N_resection', 'J_resection', 'Date_resection', 'dates_resection',
    'Realim_J5', 'DouleurJ5_persist_ou_aggrav', 'DouleurJ5_aggrav', "sorti_d'hospit", 'DCD',
    "durée d'hospit J",  # consequence of outcome, not a predictor
]
explicit_ignore = ['Patient_ID', 'Unnamed: 165', target_col] + other_targets + post_treatment_leakage
raw_date_cols = [c for c in df.columns if ('date' in c.lower()) and not c.startswith('Delta_')]
comment_cols = [c for c in df.columns if 'comment' in c.lower() or 'info' in c.lower()
                or c in ['HOS_SERVICE_L1', 'SGC_def']]
all_ignore_cols = set(explicit_ignore + raw_date_cols + comment_cols)
features = [c for c in df.columns if c not in all_ignore_cols]

for c in features:
    df[c] = pd.to_numeric(df[c], errors='coerce')

X_full = df[features].values
y_full = df[target_col].values


def binary_points_score(X_fold, selected_idx, X_eval):
    """Current scheme: 1 point per selected feature with value > 0."""
    return (X_eval[:, selected_idx] > 0).sum(axis=1)


def select_and_score(X_train, y_train, X_test, scoring='binary_points'):
    # Fit preprocessing on TRAIN only
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Drop zero-variance columns within this fold's training data
    variances = X_train_imp.var(axis=0)
    keep_mask = variances > 0
    X_train_imp = X_train_imp[:, keep_mask]
    X_test_imp = X_test_imp[:, keep_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    model = LogisticRegressionCV(
        solver='liblinear', cv=5, l1_ratios=[1.0], scoring='neg_log_loss',
        class_weight='balanced', random_state=RANDOM_STATE, max_iter=2000
    )
    model.fit(X_train_scaled, y_train)
    coefs = model.coef_[0]
    selected_idx = np.where(coefs != 0)[0]

    if len(selected_idx) == 0:
        # Fallback: no variable survived shrinkage in this fold
        return np.zeros(X_test.shape[0]), 0

    if scoring == 'binary_points':
        test_score = binary_points_score(X_train_imp, selected_idx, X_test_imp)
    elif scoring == 'logistic_probability':
        test_score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        raise ValueError(f"Unknown scoring scheme: {scoring}")

    return test_score, len(selected_idx)


def run_evaluation(scoring='binary_points'):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    oof_scores = np.full((N_REPEATS, len(y_full)), np.nan)
    n_selected_per_fold = []

    for i, (train_idx, test_idx) in enumerate(rskf.split(X_full, y_full)):
        repeat_idx = i // N_SPLITS
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train = y_full[train_idx]

        test_score, n_sel = select_and_score(X_train, y_train, X_test, scoring=scoring)
        oof_scores[repeat_idx, test_idx] = test_score
        n_selected_per_fold.append(n_sel)

    # AUC per repeat (each repeat gives one full out-of-fold prediction set)
    auc_per_repeat = []
    for r in range(N_REPEATS):
        preds = oof_scores[r]
        auc_per_repeat.append(roc_auc_score(y_full, preds))

    auc_mean = np.mean(auc_per_repeat)

    # Bootstrap CI: resample patients (using the repeat-averaged score per patient)
    avg_scores = np.nanmean(oof_scores, axis=0)
    rng = np.random.default_rng(RANDOM_STATE)
    boot_aucs = []
    n = len(y_full)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_full[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_full[idx], avg_scores[idx]))
    ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])

    return {
        'scoring_scheme': scoring,
        'auc_mean_across_repeats': round(auc_mean, 3),
        'auc_ci95_low': round(ci_low, 3),
        'auc_ci95_high': round(ci_high, 3),
        'median_n_vars_selected': int(np.median(n_selected_per_fold)),
        'min_n_vars_selected': int(np.min(n_selected_per_fold)),
        'max_n_vars_selected': int(np.max(n_selected_per_fold)),
    }


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results = []
    for scheme in ['binary_points', 'logistic_probability']:
        print(f"Running evaluation for scoring scheme: {scheme}...")
        results.append(run_evaluation(scheme))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print("\n==================================================")
    print(" HONEST PIPELINE EVALUATION (Repeated Stratified K-Fold)")
    print(f" {N_SPLITS}-fold, {N_REPEATS} repeats, {N_BOOTSTRAP} bootstrap resamples")
    print("==================================================\n")
    print(results_df.to_string(index=False))
    print(f"\n[SUCCESS] Results saved to '{OUTPUT_CSV}'")
    print("\n[NOTE] 'median_n_vars_selected' shows how stable Lasso's variable count is "
          "across folds — wide min/max range signals instability given the small N.")