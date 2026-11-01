# 🍳 skillet

> **A data-driven recommender for tech upskilling.** Built on the 2025 Stack Overflow Developer Survey, skillet answers *"what should this specific engineer learn next?"* with statistical evidence from the Business Analytics syllabus.

[![2025 survey](https://img.shields.io/badge/data-2025%20Stack%20Overflow%20Survey-blue)](https://github.com/StackExchange/Survey/tree/main/packages/archive/2025)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-green)]()
[![Streamlit](https://img.shields.io/badge/demo-Streamlit-FF4B4B)]()

---

## 🏆 Headline results

- **19,210** full-time professional developers analysed.
- **Three interpretable personas** discovered by PCA + K-means on the skill space:
  - **.NET stack** (18%) — ASP.NET, NuGet, Visual Studio
  - **Lean generalist stacks** (46%) — smaller stacks (about 10 common skills vs 21–27 in the other two); C, C++, Java, Swift and Oracle set it apart only slightly
  - **Modern JS / web tooling** (36%) — pnpm, Express, Vite, Webpack
- **Top 5 skills by salary ROI** (per-skill MLR on USD pay, control = work exp + a fixed effect for every country):
  | Skill | Salary impact | p-value | n |
  |---|---|---|---|
  | Scala | **+35.2%** | 3×10⁻²¹ | 471 |
  | Snowflake | **+33.0%** | 4×10⁻²⁶ | 665 |
  | Terraform | **+31.0%** | 9×10⁻⁸⁷ | 2,958 |
  | ClickHouse | **+30.7%** | 1×10⁻¹³ | 366 |
  | Databricks SQL | **+30.1%** | 3×10⁻¹⁷ | 493 |
- **AI-tool users earn more once country is controlled** (+9.9%, 95% CI +7.3% to +12.7%, p ≈ 5×10⁻¹⁴, holding work exp + country fixed) and report higher job satisfaction (Welch's t = 3.36, p = 8×10⁻⁴). The raw comparison points the other way (Welch's t = −6.15 on log-comp) because AI use is more common in lower-paying countries — a Simpson's-paradox case.
- **Full MLR** of log USD compensation on work exp + country fixed effects + 122 skills → R² = 0.559, adj-R² = 0.553.
- **Logistic model** of high job satisfaction → AUC = 0.565 (honest, with discussion in the report).

![Personas in PCA skill-space](assets/05_pca_scatter.png)

---

## ✨ What it does

Given a developer's profile (role, experience, country, current stack, AI-tool usage), skillet:

1. assigns them to a **persona** with the fitted PCA + K-means model,
2. finds their **k nearest peers** in the survey by Jaccard distance on the binary matrix of all 122 skills (KNN — Module 4), keeping peers who share their role, country, experience (±10 years) and AI-tool use and weighting each peer by its similarity,
3. ranks every skill they don't have by
   *peer prevalence* + 0.5 × *salary ROI* + 0.3 × *peer demand* (+ small bonuses),
4. returns the **top 5 skills to learn next**, each with the regression coefficient, p-value, and the evidence sentence, plus their **joint** modelled salary lift with a 95% interval.

When fewer than 50 developers share the whole profile, the filters are relaxed one at a time (AI-tool use, then experience, then country, then role) and the app says which ones were used. A cold-start fallback recommends the most lucrative skills within that comparison group when no stack is entered.

**Validated offline** (`src/evaluate.py`, 500 held-out respondents): the recommender recovers a hidden skill in 78% of cases (popularity baseline 72%) and 15.8% of its picks are skills the respondent says they want to learn (baseline 12.6%), while recommending skills with a higher salary effect (+12.1% vs +9.5%). See `report/REPORT.md` §9.7.

---

## 🧪 Methodology (mapped to the BA syllabus)

| Syllabus module | Where it appears in this repo |
|---|---|
| **M1** Data types, scales, cleansing, supervised vs unsupervised | `src/preprocess.py` — filters to professional developers, explodes multi-select skills, IQR outlier flag, derives `log_comp` (log of USD-converted pay, `ConvertedCompYearly`) and `JobSat_high` (≥ 8 on 0–10 scale) |
| **M2** Descriptive analytics, all chart types, hypothesis testing, p-values, relationships | `notebooks/01_eda.py` (histogram, bar, scatter, pie, box, heatmap); `src/stats_tests.py` — Welch's t-tests (AI users vs not, on salary + satisfaction), chi-square tests of independence with Cramér's V |
| **M3** SLR, MLR, assumptions, goodness of fit, model comparison | `src/regression.py` — SLR `log_comp ~ WorkExp`; per-skill MLR `log_comp ~ WorkExp + country + skill_s` (121 skills); full MLR with VIF sample |
| **M4** Logistic regression, decision tree, KNN, Naive Bayes, LDA, ROC, confusion matrix | `src/classification.py` — logistic on `JobSat_high` with ROC, AUC, Youden-optimal threshold, confusion matrix; `src/recommend.py` — **similarity-weighted KNN core** (Jaccard, k=300) drives the personalised recommendations |
| **M5** K-means, hierarchical clustering, PCA, FA | `src/cluster.py` — PCA(80% var, 51 components) → K-means (silhouette-tuned, k=3) → hierarchical dendrogram |
| **M6** Domain application — HR analytics | The whole project: L&D / upskilling decision-making for engineering teams |
| **M7** Report writing (summary → problem → objectives → data viz → analysis → findings → recommendations) | `report/REPORT.md` |

---

## 📦 Repository layout

```
skillet/
├── README.md
├── Makefile
├── run_all.py                  # one-shot pipeline runner
├── requirements.txt
├── data/
│   ├── download.sh             # fetches results.csv (LFS) + schema.csv
│   ├── results.csv             # ~135MB, gitignored
│   ├── schema.csv              # column metadata
│   ├── processed.parquet       # 19,210 × 170 (committed; ≈ 0.8 MB)
│   ├── processed_meta.json
│   ├── skill_roi.csv           # per-skill salary impact
│   ├── satisfaction_coefs.csv
│   ├── cluster_model.joblib    # PCA + KMeans bundle
│   └── *.json                  # metric snapshots
├── src/
│   ├── preprocess.py
│   ├── stats_tests.py
│   ├── regression.py
│   ├── classification.py      # + run_tree_and_nb()
│   ├── cluster.py
│   ├── fa.py                  # Factor Analysis (M5)
│   ├── anova.py               # one-way ANOVA across personas
│   ├── network.py             # skill co-occurrence graph
│   ├── salary_calc.py         # "What if I learn this?" calculator
│   ├── want.py                # WantToWorkWith demand signal loader
│   ├── recommend.py           # SkillRadar engine (Jaccard KNN + persona + ROI + want)
│   └── trend.py               # multi-year skill ROI trend (2022→2025)
├── notebooks/                  # runnable Python analyses
│   ├── 01_eda.py
│   ├── 02_stats_tests.py
│   ├── 03_regression.py
│   ├── 04_classification.py   # logistic + decision tree + naive bayes
│   ├── 05_clustering.py
│   ├── 06_recommend.py
│   ├── 07_fa_anova.py
│   └── 08_network.py
├── app/streamlit_app.py       # recommender + salary calculator + methodology
├── report/REPORT.md
├── .streamlit/config.toml     # theme for Streamlit Cloud
├── runtime.txt                # python-3.12
└── assets/                    # generated figures (PNG)
```

---

## 🚀 Quickstart

```bash
# 1. set up
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. fetch the 2025 survey (~135 MB)
bash data/download.sh

# 3. run the full pipeline
.venv/bin/python run_all.py

# 4. generate the figures
.venv/bin/python notebooks/01_eda.py
.venv/bin/python notebooks/02_stats_tests.py
.venv/bin/python notebooks/03_regression.py
.venv/bin/python notebooks/04_classification.py
.venv/bin/python notebooks/05_clustering.py
.venv/bin/python notebooks/06_recommend.py

# 5. launch the interactive demo
.venv/bin/streamlit run app/streamlit_app.py
```

A short report lives at [`report/REPORT.md`](report/REPORT.md).

---

## 📊 More results

**Bottom of the salary table** (negative ROI, included for honesty):

| Skill | Salary impact | p-value |
|---|---|---|
| Microsoft Access | −27.0% | 5×10⁻²² |
| Delphi | −24.4% | 7×10⁻¹⁴ |
| Visual Basic .NET | −21.5% | 5×10⁻²⁰ |
| PHP | −19.8% | 6×10⁻⁶¹ |
| Laravel | −19.1% | 6×10⁻²⁵ |

**Hypothesis tests (Module 2):**

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| AI users' log-comp vs non-users (raw) | t = −6.15 | 8×10⁻¹⁰ | Raw gap favours non-users; reverses to +9.9% for AI users once country + exp are controlled |
| AI users' job satisfaction vs non-users | t = 3.36 | 8×10⁻⁴ | AI users are significantly more satisfied |
| AI usage × remote work (χ², 5 d.f.) | χ² = 10.59 | 0.060 | Marginal association (Cramér's V = 0.023) |
| Role × most common language (χ²) | χ² = 2416 | < 1×10⁻³⁰ | Strong association (Cramér's V = 0.355) |

**Clustering (Module 5):** PCA 51 components for 80% variance → K-means k=3 (silhouette 0.085). The low silhouette reflects the sparsity of the binary skill space; the three clusters are nonetheless sharply interpretable as .NET, enterprise-systems, and JS/web stacks.

**Logistic satisfaction model (Module 4):** AUC 0.565, F1 0.614, accuracy 0.548. The modest AUC is honest: observable skill/experience features explain a small portion of satisfaction variance, consistent with the well-known dominance of intrinsic factors (autonomy, team, manager) over skills in developer satisfaction.

**Model comparison on the same target (Module 4):**

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic regression | 0.565 | 0.614 | 0.548 |
| **Naive Bayes** | **0.569** | 0.580 | 0.551 |
| Decision tree (max_depth=6) | 0.560 | 0.461 | 0.553 |

All three sit barely above random — strong evidence that the survey's observable features (skills, work arrangement, experience) explain only a small slice of job-satisfaction variance. Top features by tree Gini importance: **WorkExp (0.44)**, `rw_Flexible` (0.08), `org_Small` (0.04), `rw_Remote` (0.04). In the logistic model, holding the rest fixed, people at small organisations (≤ 99 staff) are the most likely to report high satisfaction, and "your choice" working arrangements are the most satisfied of the remote-work answers.

**Factor Analysis (Module 5):** 18 factors chosen by Kaiser criterion (eigenvalue > 1) on the 86-skill matrix, varimax-rotated. They map cleanly onto technology ecosystems:

| Factor | Theme | Top skills |
|---|---|---|
| F1 | JS / web front-end | React, Node.js, Next.js |
| F2 | .NET / Microsoft | ASP.NET Core, C#, NuGet |
| F3 | Java / enterprise | Java, Spring Boot, Maven |
| F4 | OpenAI tooling | GPT chatbot, GPT reasoning, image models |
| F5 | Linux CLI / DevOps | APT, Docker, Pip |
| F7 | Native / systems | C++, C, Make |
| F8 | Mobile | Android Studio, Dart, Swift |

(18 factors explain 50.1% of the binary-skill variance — the "latent ecosystem" structure underneath the 51 PCA components.)

**ANOVA across personas:** F(2, 18954) = 16.88, p = 4.7×10⁻⁸, η² = 0.002 (small effect). Mean log USD comp: .NET 11.10, lean generalist 10.94, JS/web 11.00. The .NET vs lean-generalist gap is the driver (p_adj = 4×10⁻¹⁰, ~17% higher pay; raw, not controlled for country).

**Skill co-occurrence network (Module 5/2):** 86 nodes, 182 edges at Jaccard > 0.30. The "connective-tissue" hubs (highest degree centrality) are the cross-persona tools: `npm` 0.235, `Docker` 0.224, `SQL` 0.212, `PostgreSQL` 0.200, `OpenAI GPT chatbot` 0.200. Modularity against the 3-persona partition is 0.21 (weak) — connectivity is driven by architectural/ecosystem groupings rather than the persona labels.

**"What if I learn this?" calculator (app):** pick candidate skills and see the predicted salary lift from one MLR fitted with all of them together (work exp + country fixed effects + the chosen skills). The total and its 95% interval come from that joint model, so skills that usually come as a set (Docker, Kubernetes, Terraform…) do not each claim the same premium. Summing the separate per-skill coefficients instead overstates the lift — for the plan Docker + Kubernetes + AWS + Python + Terraform, +152% summed vs +57% jointly.

**WantToWorkWith demand signal (recommender):** the 2025 survey's `*WantToWorkWith` columns are loaded once and the share of a candidate skill's *wanting* among the user's k nearest peers is added (0.3 weight) to the recommendation score. The evidence string now reports both "X% of peers know it" and "Y% of peers want to learn it."

**Multi-year skill ROI trend (2022 → 2025):** the per-skill MLR was re-fit on each of the last four Stack Overflow surveys for the 32 skills present in every year, with USD-converted compensation, a fixed effect for every country, and the same sample filter as the main pipeline. The 2025 estimates agree with the main ROI table to within about a point (Docker +16.8% vs +16.4%; the main table also requires a job-satisfaction answer). Terraform and Kubernetes carry the largest premium in every year. Biggest moves:

| Skill | 2022 | 2025 | Δ |
|---|---|---|---|
| **VS Code** | -2.4% | +4.3% | **+6.7 pts** |
| JavaScript | -8.1% | -5.6% | +2.6 pts |
| HTML/CSS | -12.7% | -10.4% | +2.3 pts |
| Go | +25.2% | +18.5% | -6.6 pts |
| Flask | +9.6% | +2.6% | -7.0 pts |
| Redis | +25.8% | +17.2% | -8.6 pts |
| **Docker** | +27.0% | +16.8% | **-10.2 pts** |

The premiums for once-specialist infrastructure skills (Docker, Redis, Go) shrank by a third or more between 2022 and 2023 and have held steady since, consistent with those skills going mainstream; Terraform (+31%) and Kubernetes (+27%) have not moved. Chart: `assets/trend_roi.png`. Data: `data/trend_roi.csv`. Full per-year column-harmonisation map (column names change every year, e.g. "Docker" moved from `ToolsTechHaveWorkedWith` to `PlatformHaveWorkedWith` in 2025) is in `data/trend_summary.json`.

An earlier version of this analysis misaligned salaries and skills (an index reset after the outlier filter paired each salary with another respondent's skills), so its tables — e.g. "Jupyter and Flask are the big risers" — were noise. Fixed in `src/trend.py`.

---

## 🚀 Deploy

The repo is Streamlit-Cloud-ready. `runtime.txt` pins Python 3.12, `.streamlit/config.toml` sets the theme, and `requirements.txt` is the single source of truth.

1. Push to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) → **New app** → point at this repo, branch `main`, main file `app/streamlit_app.py` → **Deploy**.
3. ~3-5 min later you have a public URL like `https://<your-handle>-skillet.streamlit.app` — put it in the repo description and in your report.

> Note: the app runs entirely from committed files — `data/processed.parquet` (≈ 0.8 MB), `data/processed_meta.json`, `data/want_sets.parquet` (the WantToWorkWith answers extracted from the raw survey) and the model artefacts in `data/`. The raw `results.csv` (~135 MB) stays gitignored and is only needed to re-run the pipeline. The code is syntax-checked against Python 3.12 (the `runtime.txt` pin) and developed on 3.14.

## ⚠️ Limitations

- **Sample bias.** Stack Overflow respondents over-represent English-speaking,
  internet-active developers. Generalising to a non-tech or non-SO population
  is unwarranted.
- **Self-reported compensation** is plausibly missing-not-at-random; we keep
  USD-converted pay (`ConvertedCompYearly`) and flag IQR outliers but do not
  impute. The ~1% of respondents whose pay the survey could not convert are
  excluded from the salary models only.
- **Cross-sectional associations.** Coefficients are not causal effects — a
  skill's "ROI" reflects correlation with compensation among people who know
  it, not the effect of learning it. Reverse causality (high-paid engineers
  can afford to learn more) is plausible.
- **Currency and cost of living.** Pay is compared in USD at the survey's
  exchange rate, with a fixed effect for every country; skill coefficients are
  therefore within-country pay gaps, not purchasing-power adjusted.
- **Single year.** The 2025 cross-section is a snapshot; skill ROI drifts.
- **Personas use the 86 common skills only.** A stack made entirely of
  rare skills (< 5% prevalence) carries no persona signal and is shown
  the largest persona.
- **Small roles.** Nine roles (e.g. Product manager, Student, Retired) have
  fewer than 50 respondents in the whole survey, so their peer group is the
  whole survey and the role does not change their recommendations; the app
  says so.

---

## 📚 Data source

Stack Overflow 2025 Developer Survey — public GitHub archive
[`StackExchange/Survey`](https://github.com/StackExchange/Survey/tree/main/packages/archive/2025).
Pulled via the LFS media host
(`media.githubusercontent.com/.../packages/archive/2025/results.csv`).
