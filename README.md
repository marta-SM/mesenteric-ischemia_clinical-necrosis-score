# mesenteric-ischemia_clinical-necrosis-score

Clinical scoring pipeline (Python) to build a necrosis score specific to venous acute mesenteric ischemia (AMI), based on clinical and radiological (CT scan) data. Part of FHU TSUNAMI (Inserm U1148 LVTS).

Python reproduction and extension of the preliminary methodology developed by Paul Primard (Gastroenterology/IBD, Hôpital Beaujon APHP), originally implemented in R (RMarkdown + preliminary report).

## Status

<b>Stabilized pipeline & honest evaluation completed</b> — preprocessing, univariate analysis, variable selection (Lasso L1 with strict liblinear solver), and cross-validated evaluation pipeline are fully operational with strict top-N restriction.

**Modeling Rationale:** We intentionally selected regularized logistic regression over black-box machine learning algorithms (such as random forests or gradient boosting). In a high-stakes clinical setting like acute mesenteric ischemia, any marginal performance gain from complex models does not compensate for the complete loss of interpretability and transparency required at bedside decision-making.

## Setup

```bash
conda env create -f environment.yml
conda activate ima-clinical-score
```

## Pipeline — run order, inputs, and why each method was chosen

### 1. Preprocessing — `src/preprocessing/process_dates.py`
Converts raw dates into deltas relative to symptom onset (e.g. days to CT1, days to diagnosis). Raw dates are dropped afterwards; only deltas are kept as candidate predictors, since raw calendar dates themselves carry no clinical meaning and would just act as patient identifiers.

```bash
python src/preprocessing/process_dates.py
```
- <b>Input:</b> `data/raw/Base_IMA_veineux_pseudonymise.xlsx`
- <b>Output:</b> `data/processed/Matrice_Complete_IMA_Clean.csv` (+ `.xlsx`), `data/processed/Matrice_Delais_IMA_Clean.csv`
- Must be run first; every other script depends on its output.

### 2. Data quality audit — `src/eda/inspect_variables.py`
Flags non-numeric columns, columns with >40% missing values, and constant (zero-variance) columns. Pure diagnostic step, run once after any change to the source data, to catch data entry issues before they reach the models.

```bash
python src/eda/inspect_variables.py
```
- <b>Input:</b> `data/processed/Matrice_Complete_IMA_Clean.csv`
- <b>Output:</b> console only

### 3. Delta-delta correlations — `src/eda/analyze_delta_correlations.py`
Spearman correlation between all time-delta variables. Used to check which time variables are redundant with each other (e.g. date of diagnosis vs. date of care — PEC — are 98% correlated), which helps interpret why Lasso later keeps one and drops the other.

```bash
python src/eda/analyze_delta_correlations.py
```
- <b>Input:</b> `data/processed/Matrice_Delais_IMA_Clean.csv`
- <b>Output:</b> `reports/Delta_Inter_Correlation_Matrix.csv`, `reports/Delta_Pairwise_Correlations.csv`

### 4. Univariate analysis — `src/eda/univariate_analysis.py`
Every candidate variable tested individually against necrosis: Fisher's exact test for binary variables, Mann-Whitney U for continuous ones (same approach as Paul's own report, to allow direct comparison). This step exists independently of Lasso because it answers a different question — "does this variable alone distinguish the groups" — rather than "does this variable survive a joint, multivariable model". It is also the step clinicians can sanity-check without needing to understand regularized regression.

```bash
python src/eda/univariate_analysis.py
```
- <b>Input:</b> `data/processed/Matrice_Complete_IMA_Clean.csv`
- <b>Output:</b> `reports/Univariate_Analysis_Summary.csv`
- Excludes post-treatment/leakage variables and the one patient with missing `Necrose`. Includes `durée d'hospit J` for comparison with Paul's table only — flagged, not used downstream (see Limitations).
- Note: with ~150 variables tested, ~7-8 are expected to reach p<0.05 by chance alone.

### 5. Variable selection — `src/models/train_lasso.py`
L1-penalized (Lasso) logistic regression, chosen over L2/Ridge because L1 shrinks irrelevant or redundant coefficients to exactly zero, producing an actual subset of variables — required for a short, bedside-usable score, rather than a model that keeps a small weight on all ~150 candidates. LOOCV gives an initial, indicative performance estimate; a final fit on all data gives the selected variables and their coefficients.

```bash
python src/models/train_lasso.py
```
- <b>Input:</b> `data/processed/Matrice_Complete_IMA_Clean.csv`
- <b>Output:</b> console only (LOOCV metrics + selected variables with coefficients)
- ⚠️ Variable selection here happens once on the full dataset before LOOCV evaluates it — this makes the printed LOOCV metrics optimistic. Use step 7 for a trustworthy performance estimate.

### 6. Patient-level score — `src/models/generate_patient_scores.py`
Fits Lasso once, takes the top-N ranked variables, and computes a simple score (1 point per positive criterion, binary) per patient, with a configurable cutoff. Useful for quickly inspecting individual patients and comparing `n_var` values by hand.

```bash
python src/models/generate_patient_scores.py -n_var 4
```
- `-n_var`: number of top Lasso-ranked variables to include (default: 3)
- <b>Input:</b> `data/processed/Matrice_Complete_IMA_Clean.csv`
- <b>Output:</b> `reports/Patient_Level_Scores_n<N>.csv` (+ `.xlsx`)
- ⚠️ Same leakage caveat as step 5 — not a reliable performance estimate, for inspection only.

### 7. Honest pipeline evaluation — `src/models/evaluate_pipeline.py`
The methodologically correct evaluation. Variable selection (Lasso) is repeated <i>inside</i> each cross-validation fold rather than once on the full dataset, so no information from the test patients ever influences which variables get selected. Uses Repeated Stratified K-Fold (5 folds × 20 repeats) with class-balanced logistic regression (`class_weight='balanced'`) and bootstrap 95% CI on AUC, restricted to the top-5 variables per fold.

<b>Evaluation Results Summary:</b>
- <b>Binary points scheme:</b> AUROC = 0.830 (95% CI: 0.813–0.847), median selected variables = 5 (range: 5–5)
- <b>Logistic probability scheme:</b> AUROC = 0.864 (95% CI: 0.848–0.877), median selected variables = 5 (range: 5–5)

```bash
python src/models/evaluate_pipeline.py
```
- <b>Input:</b> `data/processed/Matrice_Complete_IMA_Clean.csv`
- <b>Output:</b> `reports/Pipeline_Evaluation_Summary.csv`
- `median/min/max_n_vars_selected` shows how stable Lasso's full (unrestricted) variable count is across folds — useful diagnostic, not the deployed score's size.
- **Once fixed, this is the script to cite for actual model performance** — not steps 5 or 6.

## Known limitations / open questions

- Scoring currently uses fixed 1-point-per-criterion; integer-weighted scoring (à la CHA₂DS₂-VASc, i.e. coefficients from an unpenalized logistic regression on the Lasso-selected variables, rounded to small integers) is designed but not yet applied, pending Paul's input
- 5 patients have a CT1 date preceding symptom onset in the source data (confirmed not a calculation error) — currently treated as a missing value for that variable
- Two overlapping morphine variables (`morphine`, `Morphine_survibase`) — relationship not yet clarified
- `durée d'hospit J` is included in the univariate comparison (to match Paul's table) but excluded from Lasso/scoring as a likely consequence of necrosis rather than a predictor

## Data

Real patient data is never committed to this repository. Development uses synthetic/placeholder data on Codespaces; real data processing happens locally.

## Related repositories

- [mesenteric-ischemia-multiomics](../mesenteric-ischemia-multiomics) — omics-based biomarker discovery axis (diagnostic/prognostic/physiopathology), same FHU TSUNAMI project