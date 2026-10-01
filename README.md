# mesenteric-ischemia_clinical-necrosis-score

Clinical scoring pipeline (Python) to build a necrosis predictive score specific to venous acute mesenteric ischemia (AMI), based on clinical, biological, and radiological (CT scan) time deltas. Part of FHU TSUNAMI (Inserm U1148 LVTS).

Python reproduction and extension of the preliminary methodology developed by Paul Primard (Gastroenterology/IBD, Hôpital Beaujon APHP), originally implemented in R (RMarkdown + preliminary report).

---

## 🛠 Project Status & Progress

🔄 **Pre-processing & EDA Phase Completed** — The pipeline now handles mixed-format clinical dates, converts time differentials to whole-day deltas, filters out clinically impossible negative values, and performs automated non-parametric correlation analyses.

---

## 📁 Repository Structure

```text
mesenteric-ischemia_clinical-necrosis-score/
│
├── data/
│   ├── raw/                  # Source clinical Excel datasets (uncommitted/gitignored)
│   └── processed/            # Cleaned matrices with standardized ISO dates & deltas
│
├── reports/                  # Generated statistical outputs (.csv)
│
├── src/                      # Source code
│   ├── preprocessing/        # Date parsing, delta calculations & quality control
│   ├── eda/                  # Correlation analyses (Spearman, Mann-Whitney U, inter-delta)
│   └── models/               # Downstream predictive modeling (Lasso L1, cross-validation)
│
└── README.md