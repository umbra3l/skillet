# 🍳 skillet — Project Report

> Decision Analytics project, mapped to the Business Analytics syllabus
> (Modules 1–7). All artefacts in this repository are reproducible from
> `data/results.csv` via `run_all.py`.

## 1. Summary

Enterprises spend billions on engineering L&D every year, yet course selection
is largely driven by instinct and trend pieces. **skillet** is a
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
- numeric columns: `WorkExp`, `YearsCode`, `CompTotal` (local currency, reference only), `CompUSD` (`ConvertedCompYearly`), `log_comp` = log(1 + `CompUSD`), `JobSat_num` (0–10), `JobSat_high` (≥ 8), `ai_user`;
- binary skill columns: top-N from `Language`, `Database`, `Platform`,
  `Webframe`, `DevEnvs`, `AIModels`;
- `Country`, `RemoteWork` and `OrgSize` as raw answers. No country dummies are stored: every model builds a fixed effect for each of the survey's countries, and the app offers all of them.

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
| AI users' log-comp vs non-users (Welch's t) | t = −6.15 | 8.3×10⁻¹⁰ | Reject H₀ — raw gap favours non-users (see note) |
| AI users' log-comp, holding exp + country fixed (OLS) | t = 7.53, +9.9% | 5.2×10⁻¹⁴ | Reject H₀ — AI users earn more at the same experience and country |
| AI users' job satisfaction vs non-users (Welch's t) | t = 3.36 | 7.8×10⁻⁴ | Reject H₀ — AI users report higher satisfaction |
| AI usage × remote work (χ², 5 d.f.) | χ² = 10.59, V = 0.023 | 0.060 | Fail to reject H₀ (marginal) |
| Primary role × most common language (χ²) | χ² = 2416, V = 0.355 | < 1×10⁻³⁰ | Reject H₀ — strong association |

### 5.2 Module 3 — Regression

- **SLR** `log_comp ~ WorkExp`: R² = 0.128, slope = 0.036 log-points/year
  (≈ +3.7% pay per year of experience, p ≈ 0).
- **Full MLR** `log_comp ~ WorkExp + country + 122 skills`:
  R² = 0.559, adj-R² = 0.553, F p ≈ 0, with 274 features (a fixed effect
  for each of the 151 countries in the model sample, US as reference).
- **Per-skill MLR** (the recommender's salary prior): for each skill s,
  fit `log_comp ~ WorkExp + country + s` and record β_s. The
  standardised coefficient is that skill's **salary ROI** in log points,
  converted to percent via (e^β − 1)·100. 121 skills had enough users
  for a stable estimate.

*Note on the compensation variable.* An earlier version of this pipeline
modelled `CompTotal`, which the survey records in each respondent's local
currency (median 1,300,000 for India vs 150,000 for the US). Only the ten
largest countries had dummies, so the other ~40% of respondents shared one
baseline with mixed currencies. Switching to USD (`ConvertedCompYearly`) with
a fixed effect for every country raised the SLR R² from 0.002 to 0.128, flipped
the sign of 12 of the 121 skill coefficients (all of them small or
non-significant before), and raised the number of significant skills from 75
to 98. The AI-user t-test is a Simpson's-paradox case: the raw gap favours
non-users (t = −6.15) because AI use is more common in lower-paying
countries, but holding work experience and country fixed AI users earn
**+9.9%** (95% CI +7.3% to +12.7%, p ≈ 5×10⁻¹⁴; `log_comp_ai_users_vs_non_controlled` in `data/stats_tests.json`).

**Top 5 positive ROI:**

| Skill | % salary impact | p-value | n |
|---|---|---|---|
| Scala | +35.2% | 2.5×10⁻²¹ | 471 |
| Snowflake | +33.0% | 4.1×10⁻²⁶ | 665 |
| Terraform | +31.0% | 8.9×10⁻⁸⁷ | 2,958 |
| ClickHouse | +30.7% | 1.2×10⁻¹³ | 366 |
| Databricks SQL | +30.1% | 2.8×10⁻¹⁷ | 493 |

**Bottom 5 ROI** (included for honesty): Microsoft Access −27.0%, Delphi −24.4%, VB.NET −21.5%, PHP −19.8%, Laravel −19.1% (all p < 10⁻¹³).

### 5.3 Module 4 — Classification

Binary logistic regression of `JobSat_high` (1 if JobSat ≥ 8 on the 0–10
scale) on `WorkExp + ai_user + RemoteWork dummies + OrgSize dummies +
top-30 most common skills`. Trained with stratified 75/25 split, balanced
class weights, and standardised features.

- AUC = **0.565**, optimal threshold (Youden's J) = 0.465,
  F1 = 0.614, accuracy = 0.548.
- Remote-work and organisation-size dummies map each 2025 answer exactly
  (an earlier version matched 2023-era answer formats, so all four
  organisation-size dummies were zero and two remote-work dummies were
  mislabelled).
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
| 1 | **Lean generalist stacks** | 46.2% | Smaller stacks (≈10 common skills vs 21–27); C++, C, Swift, Java, Oracle lift only ≈1.1× |
| 2 | **Modern JS / web tooling** | 35.9% | pnpm, Express, Yarn, Webpack, Vite |

The low absolute silhouette is a property of sparse, high-dimensional
binary data; the three clusters are nonetheless sharply interpretable.

### 5.5 Module 4 + 5 — Recommender (KNN)

`src/recommend.py` instantiates a `SkillRadar` object that loads the
regression, classification and clustering artefacts. For a user with
current skills `S`:

1. Build the user's binary skill vector over all 122 skill columns.
2. Assign the user to a persona with the saved scaler → PCA → K-means
   model (on the 86 cluster-space skills), so the app's persona is the
   same cluster the analysis in §5.4 would give that stack.
3. Keep the developers who share the user's role, country, experience
   (±10 years) and AI-tool use. If fewer than 50 match, relax one filter
   at a time — AI-tool use, then experience, then country, then role —
   and report which filters were used.
4. Among them, find the k=300 nearest by **Jaccard distance** on the
   binary skill matrix (KNN — Module 4), each weighted by its similarity
   (1 − distance). Weighting matters: most filtered pools have fewer than
   300 members, and with equal weights every pool member would count the
   same regardless of the user's stack.
5. For every skill the user does NOT have, score:
   `score(s) = peer_prevalence(s) + 0.5 · β_s + 0.3 · peer_want_fraction(s)`,
   plus a 0.05 bonus if the ROI is significant at p < 0.05 (and 0.02 for
   AI-model skills when the user uses AI tools). Prevalence and want
   fraction are similarity-weighted shares of the peers. The
   `peer_want_fraction` is loaded once from the 2025
   `*WantToWorkWith` columns (`src/want.py`) and gives a real
   *demand* signal from the user's peers.
   An earlier version also added `0.2 · peer_salary_premium`; that term is
   the same for every candidate skill, so it could never change the
   ranking, and it was removed. The **peer premium** is still reported: the
   peers' mean residual from `log_comp ~ WorkExp + country`, i.e. how much
   more they earn than developers with the same experience in the same
   country.
6. Drop skills with β_s ≤ 0 (negative ROI) — they appear in the
   coefficient chart but are not "upskilling recommendations".
7. Return the top 5 with the evidence string used in the Streamlit UI
   (now reports both "X% of peers know it" and "Y% of peers want to
   learn it").

A **cold-start** path handles empty input by recommending the
high-ROI, high-prevalence skills within the same comparison group
(for the whole survey: Docker, Kubernetes, Terraform, AWS, Snowflake)
with the same evidence format.

## 6. Results

The headline numbers (all reproducible):

- 19,210 working developers analysed.
- 3 interpretable personas (silhouette 0.085).
- Full MLR R² = 0.559 (274 features: work exp + 122 skills + a fixed effect per country; USD pay).
- Top 5 skill ROI: Scala +35.2%, Snowflake +33.0%, Terraform +31.0%, ClickHouse +30.7%, Databricks SQL +30.1% (all p < 10⁻¹²).
- Logistic satisfaction model AUC = 0.565.
- AI users earn +9.9% more holding experience and country fixed (p ≈ 5×10⁻¹⁴; the raw gap reverses, t = −6.15) and are more satisfied (t = 3.36, p ≈ 8×10⁻⁴).

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
USD pay at survey exchange rates (not purchasing-power adjusted), and a
single-year snapshot. Future work: PPP adjustment, panel methods across multiple survey years, causal
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
| Logistic regression | 0.565 | 0.614 | 0.548 |
| **Naive Bayes** | **0.569** | 0.580 | 0.551 |
| Decision tree (max_depth=6) | 0.560 | 0.461 | 0.553 |

All three models agree on a single message: `JobSat_high` is only weakly
predictable from observable skills and work arrangement (AUC barely
above random), confirming the well-known dominance of intrinsic factors.
Top features by tree Gini importance: `WorkExp` (0.44), `rw_Flexible`
(0.08), `org_Small` (0.04), `rw_Remote` (0.04). See `assets/04_roc_compare.png` and
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
| F (df1=2, df2=18954) | 16.88 |
| p | 4.73×10⁻⁸ |
| η² | 0.00178 (small effect) |
| Mean log USD comp: .NET | 11.096 (n=3,411) |
| Mean log USD comp: Lean generalist | 10.935 (n=8,721) |
| Mean log USD comp: JS/web | 10.999 (n=6,825) |

Pairwise (Bonferroni-corrected Welch t): **.NET vs lean generalist p_adj=3.96×10⁻¹⁰**,
.NET vs JS/web p_adj=5.2×10⁻⁴, lean generalist vs JS/web p_adj=0.018 — .NET
developers earn the most (~17% more than the lean generalist cluster). This is a raw
comparison: it does not control for country or experience. Box plot:
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

A widget in the Streamlit app (`src/salary_calc.py`), also used for
the headline number, fits **one** regression with all the chosen skills
together: `log_comp ~ WorkExp + country FE + s_1 + … + s_k`. The total
lift is the sum of the joint coefficients, tested as a single contrast,
so its 95% interval is (e^(Σβ ± t·SE) − 1)·100 with SE² = 1′Σ1 from the
coefficient covariance. Each skill's joint coefficient and CI is shown,
with its stand-alone ROI for reference.

An earlier version summed the separate per-skill coefficients and
"summed" their percent intervals. Both were wrong: skills that co-occur
each absorbed the same premium (Docker + Kubernetes + AWS + Python + Terraform read +152%;
jointly it is +57%, 95% CI +51% to +62%), and a sum of interval bounds
is not an interval (the old display put the +152% point estimate above
its own "+90% → +117%" interval).

### 9.6 WantToWorkWith demand signal

`src/want.py` loads the 2025 `*WantToWorkWith` columns for the
~19k respondents in the processed dataset (cached after the first
call). The recommender adds `0.3 · peer_want_fraction(s)` to each
candidate's score, where `peer_want_fraction(s)` is the share of the
user's k=300 nearest peers who said they *want* to learn skill `s`.
The evidence string now reports both "X% of peers know it" and "Y% of
peers want to learn it", giving the recommendation a real demand
signal on top of the salary ROI prior.

### 9.7 Recommender evaluation (held-out respondents)

`src/evaluate.py` tests the recommender on 1,000 real respondents (500
validation, 500 test). Each is removed from their own peer pool and
recommends from their own role, country, experience and AI-tool use.

- **Task A — hidden-skill recovery.** Hide two of the respondent's
  skills (among skills the recommender can suggest) and recommend from the
  rest. hit@5 = at least one hidden skill in the top 5.
- **Task B — wanted skills.** From the full stack, precision@5 = share of
  the top 5 the respondent says they *want* to work with.
- **Value.** Mean stand-alone salary effect of the recommended skills.

Test-set results (95% bootstrap CI):

| Method | hit@5 | want precision@5 | mean salary effect |
|---|---|---|---|
| Popularity (most common skills) | 0.718 [0.678, 0.756] | 0.126 [0.111, 0.141] | +9.5% |
| Salary only (highest ROI) | 0.054 [0.034, 0.074] | 0.052 [0.042, 0.062] | +31.6% |
| KNN, peer prevalence only | 0.786 [0.750, 0.822] | 0.137 [0.123, 0.151] | +10.2% |
| **KNN, current weights** (ROI 0.5, demand 0.3) | **0.780** [0.744, 0.818] | **0.158** [0.144, 0.174] | **+12.1%** |
| KNN, ROI 2.0, demand 2.0 | 0.746 [0.708, 0.782] | 0.188 [0.172, 0.206] | +14.4% |

The KNN beats both baselines on both tasks. Adding the salary and demand
terms to plain peer prevalence raises want-precision (0.137 → 0.158) and
the salary value of the plan (+10.2% → +12.1%) at no meaningful cost in
hit rate. A grid search over the two weights, selected on the validation
half, favours heavier weights; on the test half the heaviest point is a
genuine trade-off rather than an improvement — paired against the current
weights it gains +3.0 pts of want-precision [+2.2, +3.9] and +2.4 pts of
salary value [+2.1, +2.6] but loses 3.4 pts of hit rate [−5.6, −1.2] — and
the validation optimum keeps moving to the edge of the grid because Task B
rewards the demand term directly. The current weights are therefore kept:
they sit on the accuracy/value frontier, and heavier weights are a product
choice (more aspirational recommendations) rather than a correction.
Full grid: `data/recommender_eval.json`.

## 10. Multi-year skill ROI trend (2022 → 2025)

The per-skill MLR was re-fit on each of the last four Stack Overflow
Developer Surveys (2022, 2023, 2024, 2025) for the **32 skills present
in every year**, with USD-converted compensation (`ConvertedCompYearly`)
throughout, a fixed effect for every country, and the same
professional-developer filter as `src/preprocess.py`. Year-by-year working
developer counts after cleaning: 25,249 / 30,739 / 15,163 / 20,472.
The 2025 estimates agree with the main per-skill table in §5.2 to within
about a point (Docker +16.8% vs +16.4%, Terraform +31.5% vs +31.0%); the
remaining gap is the sample — the main pipeline also requires a
job-satisfaction answer, which the trend module does not.

| Per-year top skill (by ROI) | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| #1 | Terraform (+31.9%) | Terraform (+28.2%) | Terraform (+34.4%) | Terraform (+31.5%) |
| #2 | Kubernetes (+29.6%) | Kubernetes (+24.9%) | Kubernetes (+28.7%) | Kubernetes (+27.5%) |
| #3 | Docker (+27.0%) | Go (+21.8%) | AWS (+20.8%) | AWS (+21.9%) |
| #4 | Redis (+25.8%) | AWS (+21.8%) | Redis (+19.3%) | Go (+18.5%) |
| #5 | AWS (+25.6%) | Ruby (+17.9%) | Go (+17.8%) | Redis (+17.2%) |

### 10.1 Biggest movers

| Direction | Skill | 2022 | 2025 | Δ |
|---|---|---|---|---|
| ⬆ | **VS Code** | -2.4% | +4.3% | **+6.7 pts** |
| ⬆ | JavaScript | -8.1% | -5.6% | +2.6 pts |
| ⬆ | HTML/CSS | -12.7% | -10.4% | +2.3 pts |
| ⬇ | Django | +1.3% | -4.7% | -6.0 pts |
| ⬇ | Go | +25.2% | +18.5% | -6.6 pts |
| ⬇ | Flask | +9.6% | +2.6% | -7.0 pts |
| ⬇ | Redis | +25.8% | +17.2% | -8.6 pts |
| ⬇ | **Docker** | +27.0% | +16.8% | **-10.2 pts** |

**Story:** the top of the table is stable — Terraform and Kubernetes
command the largest premium in all four years and have not moved. The
clearest change is among once-specialist infrastructure skills: Docker,
Redis and Go lost a third or more of their premium between 2022 and 2023
and have held steady since, consistent with those skills becoming
mainstream. Few skills rose; VS Code (+6.7 pts) is the only gain above
three points. Python and Jupyter are flat.

*Correction.* An earlier version of this section reported Jupyter and
Flask as the big risers and Docker near zero. That came from a bug in
`src/trend.py`: after dropping salary outliers the code reset the row
index and then looked up skills by the new index, so each salary was
paired with a different respondent's skills (19,429 of 19,441 rows in
2025). Those estimates were noise; the tables above are the corrected fit.

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
