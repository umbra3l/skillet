# Skill Radar — Project Report

> Decision Analytics project, mapped to the Business Analytics syllabus
> (Modules 1–7). All artefacts in this repository are reproducible from
> `data/results.csv` via `run_all.py`.

## 1. Summary

Enterprises spend billions on engineering L&D every year, yet course selection
is largely driven by instinct and trend pieces. **Skill Radar** is a
data-driven recommender that, given a developer's role, experience, current
stack and country, returns the *next* skills to learn, each backed by a
regression coefficient, a p-value and a satisfaction association drawn
from the 2025 Stack Overflow Developer Survey.

After cleaning, the working dataset contains **19,210 full-time professional
developers** across ~190 countries, with a binary skill matrix over
**122 distinct skills** (top languages, databases, cloud platforms, web
frameworks, dev environments and AI/LLM models).

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

- one row per professional developer;
- numeric columns: `WorkExp`, `YearsCode`, `CompTotal`, `log_comp`, `JobSat_num` (0–10), `JobSat_high` (≥ 8), `ai_user`;
- binary skill columns: top-N from `Language`, `Database`, `Platform`,
  `Webframe`, `DevEnvs`, `AIModels`;
- 10 top-country dummies.

Figures in `assets/`:

| File | Chart type (M2) |
|---|---|
| `01_comp_hist.png` | histogram |
| `01_jobsat_bar.png` | bar chart |
| `01_scatter_exp_comp.png` | scatter plot |
| `01_ai_pie.png` | pie chart |
| `01_box_exp_comp.png` | box plot |
| `01_corr_heatmap.png` | heatmap (tree/heatmap) |
| `03_skill_roi.png` | regression coefficient bar charts |
| `04_roc_cm.png` | ROC curve + confusion matrix |
| `05_pca_scatter.png` | PCA scatter (personas) |
| `dendrogram.png` | hierarchical clustering dendrogram |

## 5. Statistical analysis

### 5.1 Module 2 — Hypothesis testing

Significance level α = 0.05 throughout.

| Test | Statistic | p-value | Decision |
|---|---|---|---|
| AI users' log-comp vs non-users (Welch's t) | t = 7.67 | 1.9×10⁻¹⁴ | Reject H₀ — AI users earn significantly more |
| AI users' job satisfaction vs non-users (Welch's t) | t = 3.36 | 7.8×10⁻⁴ | Reject H₀ — AI users report higher satisfaction |
| AI usage × remote work (χ², 5 d.f.) | χ² = 10.59, V = 0.023 | 0.060 | Fail to reject H₀ (marginal) |
| Primary role × most common language (χ²) | χ² = 2416, V = 0.355 | < 1×10⁻³⁰ | Reject H₀ — strong association |

### 5.2 Module 3 — Regression

- **SLR** `log_comp ~ WorkExp`: R² = 0.002, slope = 0.0063 log-points/year
  (p = 3.3×10⁻⁹). The very low R² is expected: country fixed effects and
  skill mix dominate compensation, so work experience alone is a weak
  univariate predictor.
- **Full MLR** `log_comp ~ WorkExp + country + 122 skills`:
  R² = 0.238, adj-R² = 0.233, F p ≈ 0, with 134 features.
- **Per-skill MLR** (the recommender's salary prior): for each skill s,
  fit `log_comp ~ WorkExp + country + s` and record β_s. The
  standardised coefficient is that skill's **salary ROI** in log points,
  converted to percent via (e^β − 1)·100. 121 skills had enough users
  for a stable estimate.

**Top 5 positive ROI:**

| Skill | % salary impact | p-value | n |
|---|---|---|---|
| Scala | +29.8% | 4.1×10⁻⁶ | 471 |
| Terraform | +27.0% | 1.3×10⁻²² | 2,945 |
| Kubernetes | +26.5% | 9.5×10⁻³⁰ | 4,534 |
| Go | +26.0% | 9.3×10⁻²² | 3,027 |
| ClickHouse | +24.5% | 5.9×10⁻⁴ | 370 |

**Bottom 5 ROI** (included for honesty): Microsoft Access −32.9%, VB.NET −23.1%, WordPress −21.6%, PHP −21.0%, Ada −20.9% (all p < 0.05).

### 5.3 Module 4 — Classification

Binary logistic regression of `JobSat_high` (1 if JobSat ≥ 8 on the 0–10
scale) on `WorkExp + ai_user + RemoteWork dummies + OrgSize dummies +
top-30 most common skills`. Trained with stratified 75/25 split, balanced
class weights, and standardised features.

- AUC = **0.562**, optimal threshold (Youden's J) = 0.472,
  F1 = 0.602, accuracy = 0.549.
- The modest AUC is the truth: observable skill/experience features
  explain a small portion of satisfaction variance, consistent with the
  well-known finding that intrinsic factors (autonomy, team, manager)
  dominate developer satisfaction. This is reported as a *finding*, not a
  failure.
- Per-skill satisfaction coefficients are saved to
  `data/satisfaction_coefs.csv` and feed the recommender as a small
  bonus term.

### 5.4 Module 5 — Clustering & dimensionality reduction

- **PCA** on the 86-skill standardised matrix (skills known by ≥ 5% of the
  sample), retaining enough components to explain ≥ 80% of variance →
  **51 components**.
- **K-means** on the PCA components; k chosen by silhouette score over
  3 ≤ k ≤ 8 → **k = 3** (silhouette 0.085).
- **Hierarchical clustering** (Ward linkage) on a 1.5 k-row subsample
  for the dendrogram (`assets/dendrogram.png`).
- **Personas.** For each cluster, the top 8 skills by *prevalence lift*
  (in-cluster prevalence / overall prevalence):

| # | Persona | Share | Top 5 skills |
|---|---|---|---|
| 0 | **.NET / Microsoft stack** | 17.9% | ASP.NET, ASP.NET Core, NuGet, C# (lang_C_2), Visual Studio |
| 1 | **Systems / enterprise languages** | 46.2% | C++ (lang_C_3), C (lang_C), Swift, Java, Oracle |
| 2 | **Modern JS / web tooling** | 35.9% | pnpm, Express, Yarn, Webpack, Vite |

The low absolute silhouette is a property of sparse, high-dimensional
binary data; the three clusters are nonetheless sharply interpretable.

### 5.5 Module 4 + 5 — Recommender (KNN)

`src/recommend.py` instantiates a `SkillRadar` object that loads the
regression, classification and clustering artefacts. For a user with
current skills `S`:

1. Build the user's binary skill vector in the 86-skill cluster space.
2. Assign the user to a persona by overlap with each cluster's top skills
   (robust for sparse user inputs).
3. Find the k=300 nearest developers in the survey by **Jaccard distance**
   on the binary skill matrix (KNN — Module 4).
4. For every skill the user does NOT have, score:
   `score(s) = neighbour_prevalence(s) + 0.5 · β_s + 0.2 · peer_salary_premium`,
   plus a 0.05 bonus if the ROI is significant at p < 0.05.
5. Drop skills with β_s ≤ 0 (negative ROI) — they appear in the
   coefficient chart but are not "upskilling recommendations".
6. Return the top 5 with the evidence string used in the Streamlit UI.

A **cold-start** path handles empty input by recommending the globally
high-ROI, high-prevalence skills (Docker, Kubernetes, Terraform,
Homebrew) with the same evidence format.

## 6. Results

The headline numbers (all reproducible):

- 19,210 working developers analysed.
- 3 interpretable personas (silhouette 0.085).
- Full MLR R² = 0.238 (134 features, country + skills + work exp).
- Top 5 skill ROI: Scala +29.8%, Terraform +27.0%, Kubernetes +26.5%, Go +26.0%, ClickHouse +24.5% (all p < 0.001).
- Logistic satisfaction model AUC = 0.562.
- AI users earn more (t = 7.67, p ≈ 2×10⁻¹⁴) and are more satisfied (t = 3.36, p ≈ 8×10⁻⁴).

## 7. Recommendations

- **For individual developers.** Use the Streamlit app to enter your
  profile, see your persona, and read the evidence behind each
  recommended next skill. The 95% confidence interval and p-value
  indicate how certain the salary signal is.
- **For L&D managers.** Treat each persona cluster as an L&D *track*.
  Build a curated curriculum per persona (the top 8–10 distinguishing
  skills in `data/cluster_meta.json`), and use the per-skill ROI table
  to prioritise budget where the salary uplift is largest and most
  statistically robust.
- **For HR analytics teams.** The coefficient tables bridge engagement
  surveys and compensation benchmarks. Re-run the pipeline yearly to
  detect drift in skill ROI.

## 8. Limitations & future work

See the README for the full list. In short: sample bias (SO respondents),
self-reported compensation, cross-sectional associations ≠ causal effects,
currency heterogeneity, and single-year snapshot. Future work: explicit
FX conversion, panel methods across multiple survey years, causal
inference with a work-history subsample, and a privacy-preserving
on-prem deployment for enterprise L&D.
