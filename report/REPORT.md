# Skill Radar — Project Report

> Decision Analytics project, mapped to the Business Analytics syllabus
> (Modules 1-7). All artefacts in this repository are reproducible from
> `data/results.csv` via `run_all.py`.

## 1. Summary

Enterprises spend billions on engineering L&D every year, yet course selection
is largely driven by instinct and trend pieces. **Skill Radar** is a
data-driven recommender that, given a developer's role, experience, current
stack and country, returns the *next* skills to learn, each backed by a
regression coefficient, a p-value and a satisfaction association drawn
from the 2025 Stack Overflow Developer Survey (~30k working developers
after cleaning).

## 2. Problem identification

L&D teams face a coordination problem: hundreds of possible trainings, dozens
of personas, and a noisy, fast-moving skill market. We need a method that
(a) ranks candidate skills on objective outcomes (salary, satisfaction) and
(b) tailors the ranking to the individual's existing profile.

## 3. Objectives

- **Primary.** Recommend, with statistical evidence, the top 5 next skills
  for a working developer.
- **Secondary.** Identify a small number of distinct developer personas in
  the population to support portfolio-level L&D allocation.

## 4. Data visualisation & exploration

The processed dataset has `data/processed.parquet` with:

- one row per professional developer
- numeric columns: `WorkExp`, `YearsCode`, `CompTotal`, `log_comp`, `JobSat_num`, `JobSat_high`, `ai_user`
- binary skill columns: top-N from `Language`, `Database`, `Platform`,
  `Webframe`, `DevEnvs`, `AIModels`
- 10 top-country dummies

Figures in `assets/`:

| File | Syllabus chart |
|---|---|
| `01_comp_hist.png` | histogram |
| `01_jobsat_bar.png` | bar chart |
| `01_scatter_exp_comp.png` | scatter |
| `01_ai_pie.png` | pie |
| `01_box_exp_comp.png` | box plot |
| `01_corr_heatmap.png` | tree/heatmap |
| `03_skill_roi.png` | regression coefficients |
| `04_roc_cm.png` | ROC + confusion matrix |
| `05_pca_scatter.png` | PCA scatter |
| `dendrogram.png` | hierarchical clustering |

## 5. Statistical analysis

### 5.1 Module 2 — Hypothesis testing

Welch's two-sample t-tests (unequal variance) and chi-square tests of
independence with Cramér's V. See `data/stats_tests.json` for the full
result table. Significance level α = 0.05 throughout.

- `log_comp`: AI users vs non-users
- `JobSat_num`: AI users vs non-users
- `ai_user` vs `RemoteWork`
- `DevType` (primary role) vs the most common language

### 5.2 Module 3 — Regression

- **SLR.** `log_comp ~ WorkExp`. R², F-statistic, residual diagnostics.
- **Full MLR.** `log_comp ~ WorkExp + country + skills`. Reports R²,
  adj-R², AIC/BIC, and a sample of variance-inflation factors.
- **Per-skill MLR.** For each skill s, `log_comp ~ WorkExp + country + s`.
  The standardised coefficient of `s` is that skill's **salary ROI** in
  log points, converted to percent via `(exp(β) − 1)·100`. This is
  robust to multicollinearity and yields a single interpretable number
  per skill, used directly by the recommender. See `data/skill_roi.csv`.

### 5.3 Module 4 — Classification

Binary logistic regression of `JobSat_high` (satisfied vs not) on
`WorkExp + ai_user + top-30 most common skills`. Trained with stratified
75/25 split, balanced class weights, and standardised features.

- ROC + AUC
- Optimal threshold via Youden's J statistic
- Confusion matrix at the optimal threshold
- Per-skill satisfaction coefficients saved to `data/satisfaction_coefs.csv`

### 5.4 Module 5 — Clustering & dimensionality reduction

- **PCA** on the standardised binary skill matrix, retaining enough
  components to explain ≥ 80% of variance.
- **K-means** on the PCA components, k chosen by silhouette score over
  3 ≤ k ≤ 8.
- **Hierarchical clustering** (Ward linkage) on a 1.5 k-row subsample
  for the dendrogram.
- **Personas.** For each cluster, the top 8 skills by *prevalence lift*
  (in-cluster prevalence / overall prevalence) are saved as the persona
  profile in `data/cluster_meta.json`.

### 5.5 Module 4 + 5 — Recommender

`src/recommend.py` instantiates a `SkillRadar` object that loads the
regression, classification and clustering artefacts. For a user with
current skills `S`:

1. Build the user's binary skill vector.
2. Assign the user to the nearest cluster in PCA space → **persona**.
3. For every skill the user does not have:
   - look up its salary ROI coefficient and p-value
   - look up its satisfaction coefficient
   - compute its lift within the user's persona
4. Score = `β_salary + 0.5 · β_satisfaction + 0.3 · log(1 + max(lift − 1, 0))`
   + a 0.1 bonus if p < 0.05.
5. Return the top 5 with the evidence strings used in the Streamlit UI.

## 6. Results

- **SLR R²** and **full MLR adj-R²**: see `data/regression_summary.json`.
- **Top 10 positive and bottom 10 negative skill ROI**: see
  `data/skill_roi.csv`.
- **Logistic AUC, optimal threshold, confusion matrix**: see
  `data/satisfaction_metrics.json`.
- **Persona profiles** with cluster size, share and top skills: see
  `data/cluster_meta.json` → `personas`.

## 7. Recommendations

- **For individual developers.** Use the Streamlit app to enter your
  profile, see your persona, and read the evidence behind each
  recommended next skill. The 95% confidence interval and p-value
  indicate how certain the signal is.
- **For L&D managers.** Treat each persona cluster as an L&D *track*.
  Build a curated curriculum per persona (the top 8–10 distinguishing
  skills in `data/cluster_meta.json`), and use the per-skill ROI table
  to prioritise budget where the salary / satisfaction uplift is
  largest and most statistically robust.
- **For HR analytics teams.** The coefficient tables are the missing
  bridge between engagement surveys and compensation benchmarks. Re-run
  the pipeline yearly to detect drift in skill ROI.

## 8. Limitations & future work

See the README for the full list. In short: sample bias (SO respondents),
self-reported compensation, cross-sectional associations ≠ causal effects,
currency heterogeneity, and single-year snapshot. Future work: explicit
FX conversion, panel methods across multiple survey years, causal
inference with a work-history subsample, and a privacy-preserving
on-prem deployment for enterprise L&D.
