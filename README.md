# mesenteric-ischemia_clinical-necrosis-score

Clinical scoring pipeline (Python) to build a necrosis score specific to venous acute mesenteric ischemia (AMI), based on clinical and radiological (CT scan) data. Part of FHU TSUNAMI (Inserm U1148 LVTS).

Python reproduction and extension of the preliminary methodology developed by Paul Primard (Gastroenterology/IBD, Hôpital Beaujon APHP), originally implemented in R (RMarkdown + preliminary report).

## Status

🚧 **Setup phase** — repository structure and environment being prepared. Input data (source Excel from Alexandre's cohort) still needs to be adapted before pipeline development begins. No modeling code yet.

## Planned approach

1. Univariate analysis
2. Grouping of variables by clinical significance
3. Multivariate models (3-variable models)
4. Model comparison (ANOVA / AIC / AUC)

## Data

Real patient data is never committed to this repository. Development uses synthetic/placeholder data on Codespaces; real data processing happens locally.

## Related repositories

- [mesenteric-ischemia-multiomics](../mesenteric-ischemia-multiomics) — omics-based biomarker discovery axis (diagnostic/prognostic/physiopathology), same FHU TSUNAMI project.