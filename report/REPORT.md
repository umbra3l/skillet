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
   `score(s) = neighbour_prevalence(s) + 0.5 · β_s + 0.2 · peer_salary_premium
              + 0.3 · peer_want_fraction(s)`,
   plus a 0.05 bonus if the ROI is significant at p < 0.05. The
   `peer_want_fraction` is loaded once from the 2025
   `*WantToWorkWith` columns (`src/want.py`) and gives a real
   *demand* signal from the user's peers.
5. Drop skills with β_s ≤ 0 (negative ROI) — they appear in the
   coefficient chart but are not "upskilling recommendations".
6. Return the top 5 with the evidence string used in the Streamlit UI
   (now reports both "X% of peers know it" and "Y% of peers want to
   learn it").

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

## 9. Additional analyses (v2)

The pipeline was extended with four new analyses that deepen Modules 4
and 5 and turn the recommender into a real product.

### 9.1 Model comparison on the satisfaction target (Module 4)

A decision tree (`max_depth=6`, balanced class weights) and a Gaussian
Naive Bayes were fit on the *same* design matrix and 75/25 stratified
split as the logistic regression:

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic regression | 0.562 | 0.602 | 0.549 |
| **Naive Bayes** | **0.566** | 0.579 | 0.550 |
| Decision tree (max_depth=6) | 0.552 | 0.569 | 0.539 |

All three models agree on a single message: `JobSat_high` is only weakly
predictable from observable skills and work arrangement (AUC barely
above random), confirming the well-known dominance of intrinsic factors.
Top features by tree Gini importance: `WorkExp` (0.44), `rw_Flexible`
(0.05), `lang_JavaScript` (0.04). See `assets/04_roc_compare.png` and
`assets/04_tree_importance.png`.

### 9.2 Factor Analysis (Module 5)

18 factors chosen by Kaiser criterion (eigenvalue > 1) on the 86-skill
correlation matrix, varimax-rotated. Cumulative variance explained:
50.1%. The factors map cleanly onto technology ecosystems:

| Factor | Theme | Top skills |
|---|---|---|
| F1 | JS / web front-end | React, Node.js, Next.js |
| F2 | .NET / Microsoft | ASP.NET Core, C#, NuGet |
| F3 | Java / enterprise | Java, Spring Boot, Maven |
| F4 | OpenAI tooling | GPT chatbot, GPT reasoning, image models |
| F5 | Linux CLI / DevOps | APT, Docker, Pip |
| F7 | Native / systems | C++, C, Make |
| F8 | Mobile | Android Studio, Dart, Swift |

This is the *latent ecosystem* structure underneath the 51 PCA
components and the 3 K-means personas. Scree plot:
`assets/factor_scree.png`. Loadings: `data/factor_loadings.csv`.

### 9.3 One-way ANOVA across personas

| Statistic | Value |
|---|---|
| F (df1=2, df2=19207) | 16.90 |
| p | 4.66×10⁻⁸ |
| η² | 0.00176 (small effect) |
| Mean log-comp: .NET | 11.839 (n=3,439) |
| Mean log-comp: Systems/enterprise | 11.748 (n=8,868) |
| Mean log-comp: JS/web | 11.951 (n=6,903) |

Pairwise (Bonferroni-corrected Welch t): .NET vs Systems p_adj=0.073,
.NET vs JS/web p_adj=0.021, **Systems vs JS/web p_adj=3.57×10⁻⁸** —
JS/web developers earn meaningfully more than Systems/enterprise devs
(~20% higher compensation after exp). Box plot:
`assets/anova_personas.png`.

### 9.4 Skill co-occurrence network (Module 5)

86 skills as nodes; edges where Jaccard similarity > 0.30 → **182 edges,
38 connected components** (many rare skills are isolated at the
threshold). The "connective tissue" hubs (top-5 by degree centrality):

| Skill | Degree |
|---|---|
| `plat_npm` | 0.235 |
| `plat_Docker` | 0.224 |
| `lang_SQL` | 0.212 |
| `db_PostgreSQL` | 0.200 |
| `aimodel_openAI_GPT_chatbot_models` | 0.200 |

Modularity against the 3-persona partition is 0.21 (weak separation).
Greedy modularity maximisation finds a different partition at
modularity 0.38, dominated by a 20-node community — the persona
labels don't capture the true community structure; architectural /
ecosystem groupings do. Graph (Spring layout, persona-coloured):
`assets/skill_network.png`. Node-link JSON:
`data/skill_network.json`.

### 9.5 "What if I learn this?" salary calculator

A new widget in the Streamlit app (`src/salary_calc.py`) sums the
per-skill MLR coefficients of user-picked candidate skills, converts
the log-point sum to a percent via `exp()`, and shows the per-skill
breakdown with each skill's 95% CI. An honest caveat is displayed:
the sum assumes independent effects, so the true joint lift is
smaller than the naive sum (skills cluster).

### 9.6 WantToWorkWith demand signal

`src/want.py` loads the 2025 `*WantToWorkWith` columns for the
~19k respondents in the processed dataset (cached after the first
call). The recommender adds `0.3 · peer_want_fraction(s)` to each
candidate's score, where `peer_want_fraction(s)` is the share of the
user's k=300 nearest peers who said they *want* to learn skill `s`.
The evidence string now reports both "X% of peers know it" and "Y% of
peers want to learn it", giving the recommendation a real demand
signal on top of the salary ROI prior.

## 10. Multi-year skill ROI trend (2022 → 2025)

The per-skill MLR was re-fit on each of the last four Stack Overflow
Developer Surveys (2022, 2023, 2024, 2025) for the **32 skills present
in every year**, with USD-converted compensation (`ConvertedCompYearly`)
throughout and country fixed effects absorbed. Year-by-year working
developer counts after cleaning: 25,244 / 30,728 / 14,217 / 20,083.

| Per-year top skill (by ROI) | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| #1 | Rust (+4.6%) | Django (+1.5%) | Flask (+4.1%) | **Jupyter (+5.7%)** |
| #2 | Ruby (+4.1%) | C# (+1.1%) | Jupyter (+2.4%) | Flask (+4.1%) |
| #3 | Redis (+2.3%) | SQL (+1.0%) | Go (+2.2%) | VS Code (+3.6%) |
| #4 | Docker (+2.1%) | VS Code (+0.8%) | Docker (+2.2%) | Azure (+3.1%) |
| #5 | C# (+2.0%) | Go (+0.5%) | Django (+2.0%) | JavaScript (+3.0%) |

### 10.1 Biggest movers

| Direction | Skill | 2022 | 2025 | Δ |
|---|---|---|---|---|
| ⬆ | **Jupyter** | -3.3% | **+5.7%** | **+8.9 pts** |
| ⬆ | **Flask** | -4.8% | +4.1% | +8.8 pts |
| ⬆ | VS Code | -0.8% | +3.6% | +4.4 pts |
| ⬆ | JavaScript | -1.3% | +3.0% | +4.3 pts |
| ⬆ | HTML/CSS | -1.3% | +2.3% | +3.6 pts |
| ⬇ | C# | +2.0% | +0.3% | -1.7 pts |
| ⬇ | Docker | +2.1% | +0.0% | -2.1 pts |
| ⬇ | Ruby | +4.1% | +1.3% | -2.7 pts |
| ⬇ | Rust | +4.6% | +0.8% | -3.7 pts |
| ⬇ | **Redis** | +2.3% | -1.6% | -3.8 pts |

**Story:** the Python data-stack (Jupyter, Flask, VS Code) is the clear
winner of the AI / data-science wave. Redis, Rust, Ruby, Docker and C#
are the most notable *fallers* — the famous premium attached to those
skills in 2022 has largely been competed away as they've become
mainstream (and in Rust's case, the salary premium of "exotic" status
has shrunk as the language has matured).

### 10.2 Caveats

- 2022 / 2023 use `ConvertedCompYearly` (USD-converted by Stack
  Overflow); 2024 / 2025 *also* expose a `ConvertedCompYearly` column
  (used here), so the comparison is currency-clean.
- Survey year-to-year the same skill lives in **different columns**:
  e.g. "Docker" was in `ToolsTechHaveWorkedWith` in 2022-24 and
  `PlatformHaveWorkedWith` in 2025; "Jupyter" was `IPython/Jupyter`
  in 2022 and `Jupyter Notebook/JupyterLab` in 2023-25; "React" was
  `React.js` in 2022. The full per-year harmonisation map is in
  `data/trend_summary.json` under `skill_token_map` and `matrix_columns`.
- Each year's MLR is run on a different sample (different respondents
  in each year), so the coefficients are *within-year* estimates.
  Between-year drift could reflect both a genuine change in the
  skill's market premium AND a change in the population of respondents
  (not causally identified from a single year).
