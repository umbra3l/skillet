# 🎯 Skill Radar

> **A data-driven recommender for tech upskilling.** Built on the 2025 Stack Overflow Developer Survey, Skill Radar answers *"what should this specific engineer learn next?"* with statistical evidence from the Business Analytics syllabus.

[![2025 survey](https://img.shields.io/badge/data-2025%20Stack%20Overflow%20Survey-blue)](https://github.com/StackExchange/Survey/tree/main/packages/archive/2025)
[![Python 3.14](https://img.shields.io/badge/python-3.14-green)]()
[![Streamlit](https://img.shields.io/badge/demo-Streamlit-FF4B4B)]()

---

## 🏆 Headline results

- **19,210** full-time professional developers analysed.
- **Three interpretable personas** discovered by PCA + K-means on the skill space:
  - **.NET stack** (18%) — ASP.NET, NuGet, Visual Studio
  - **Systems / enterprise languages** (46%) — C, C++, Java, Swift, Oracle
  - **Modern JS / web tooling** (36%) — pnpm, Express, Vite, Webpack
- **Top 5 skills by salary ROI** (per-skill MLR, control = work exp + country):
  | Skill | Salary impact | p-value | n |
  |---|---|---|---|
  | Scala | **+29.8%** | 4×10⁻⁶ | 471 |
  | Terraform | **+27.0%** | 1×10⁻²² | 2,945 |
  | Kubernetes | **+26.5%** | 9×10⁻³⁰ | 4,534 |
  | Go | **+26.0%** | 9×10⁻²² | 3,027 |
  | ClickHouse | **+24.5%** | 6×10⁻⁴ | 370 |
- **AI-tool users earn more** and report higher job satisfaction (Welch's t = 7.67, p ≈ 2×10⁻¹⁴ on log-comp; t = 3.36, p = 8×10⁻⁴ on satisfaction).
- **Full MLR** of log compensation on work exp + country + 122 skills → R² = 0.238, adj-R² = 0.233.
- **Logistic model** of high job satisfaction → AUC = 0.562 (honest, with discussion in the report).

![Personas in PCA skill-space](assets/05_pca_scatter.png)

---

## ✨ What it does

Given a developer's profile (role, experience, country, current stack, AI-tool usage), Skill Radar:

1. assigns them to a **persona** by overlap with each cluster's top skills,
2. finds their **k nearest peers** in the survey by Jaccard distance on the binary skill matrix (KNN — Module 4),
3. ranks every skill they don't have by
   *neighbour prevalence* + 0.5 × *salary ROI* + 0.2 × *peer salary premium*,
4. returns the **top 5 skills to learn next**, each with the regression coefficient, p-value, and the evidence sentence.

A cold-start fallback recommends the globally most lucrative skills when no stack is entered.

---

## 🧪 Methodology (mapped to the BA syllabus)

| Syllabus module | Where it appears in this repo |
|---|---|
| **M1** Data types, scales, cleansing, supervised vs unsupervised | `src/preprocess.py` — filters to professional developers, explodes multi-select skills, IQR outlier flag, derives `log_comp` and `JobSat_high` (≥ 8 on 0–10 scale) |
| **M2** Descriptive analytics, all chart types, hypothesis testing, p-values, relationships | `notebooks/01_eda.py` (histogram, bar, scatter, pie, box, heatmap); `src/stats_tests.py` — Welch's t-tests (AI users vs not, on salary + satisfaction), chi-square tests of independence with Cramér's V |
| **M3** SLR, MLR, assumptions, goodness of fit, model comparison | `src/regression.py` — SLR `log_comp ~ WorkExp`; per-skill MLR `log_comp ~ WorkExp + country + skill_s` (121 skills); full MLR with VIF sample |
| **M4** Logistic regression, decision tree, KNN, Naive Bayes, LDA, ROC, confusion matrix | `src/classification.py` — logistic on `JobSat_high` with ROC, AUC, Youden-optimal threshold, confusion matrix; `src/recommend.py` — **KNN core** (Jaccard, k=300) drives the personalised recommendations |
| **M5** K-means, hierarchical clustering, PCA, FA | `src/cluster.py` — PCA(80% var, 51 components) → K-means (silhouette-tuned, k=3) → hierarchical dendrogram |
| **M6** Domain application — HR analytics | The whole project: L&D / upskilling decision-making for engineering teams |
| **M7** Report writing (summary → problem → objectives → data viz → analysis → findings → recommendations) | `report/REPORT.md` |

---

## 📦 Repository layout

```
skill-radar/
├── README.md
├── Makefile
├── run_all.py                  # one-shot pipeline runner
├── requirements.txt
├── data/
│   ├── download.sh             # fetches results.csv (LFS) + schema.csv
│   ├── results.csv             # ~135MB, gitignored
│   ├── schema.csv              # column metadata
│   ├── processed.parquet       # 19,210 × 178
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
| Microsoft Access | −32.9% | 5×10⁻¹² |
| Visual Basic .NET | −23.1% | 2×10⁻⁸ |
| WordPress | −21.6% | 1×10⁻¹⁴ |
| PHP | −21.0% | 3×10⁻²³ |
| Ada | −20.9% | 0.045 |

**Hypothesis tests (Module 2):**

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| AI users' log-comp vs non-users | t = 7.67 | 2×10⁻¹⁴ | AI users earn significantly more |
| AI users' job satisfaction vs non-users | t = 3.36 | 8×10⁻⁴ | AI users are significantly more satisfied |
| AI usage × remote work (χ², 5 d.f.) | χ² = 10.59 | 0.060 | Marginal association (Cramér's V = 0.023) |
| Role × most common language (χ²) | χ² = 2416 | < 1×10⁻³⁰ | Strong association (Cramér's V = 0.355) |

**Clustering (Module 5):** PCA 51 components for 80% variance → K-means k=3 (silhouette 0.085). The low silhouette reflects the sparsity of the binary skill space; the three clusters are nonetheless sharply interpretable as .NET, enterprise-systems, and JS/web stacks.

**Logistic satisfaction model (Module 4):** AUC 0.562, F1 0.602, accuracy 0.549. The modest AUC is honest: observable skill/experience features explain a small portion of satisfaction variance, consistent with the well-known dominance of intrinsic factors (autonomy, team, manager) over skills in developer satisfaction.

**Model comparison on the same target (Module 4):**

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic regression | 0.562 | 0.602 | 0.549 |
| **Naive Bayes** | **0.566** | 0.579 | 0.550 |
| Decision tree (max_depth=6) | 0.552 | 0.569 | 0.539 |

All three sit barely above random — strong evidence that the survey's observable features (skills, work arrangement, experience) explain only a small slice of job-satisfaction variance. Top features by tree Gini importance: **WorkExp (0.44)**, `rw_Flexible` (0.05), `lang_JavaScript` (0.04).

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

**ANOVA across personas:** F(2, 19207) = 16.90, p = 4.7×10⁻⁸, η² = 0.002 (small effect). Mean log-comp: .NET 11.84, Systems 11.75, JS/web 11.95. The Systems vs JS/web gap is the driver (p_adj = 3.6×10⁻⁸, ~20% more compensation after exp).

**Skill co-occurrence network (Module 5/2):** 86 nodes, 182 edges at Jaccard > 0.30. The "connective-tissue" hubs (highest degree centrality) are the cross-persona tools: `npm` 0.235, `Docker` 0.224, `SQL` 0.212, `PostgreSQL` 0.200, `OpenAI GPT chatbot` 0.200. Modularity against the 3-persona partition is 0.21 (weak) — connectivity is driven by architectural/ecosystem groupings rather than the persona labels.

**"What if I learn this?" calculator (app):** pick 2-3 candidate skills and see the predicted salary lift, summed from the per-skill MLR coefficients. Each skill shows its own % and 95% CI; the total is flagged with a caveat that the sum assumes independent effects.

**WantToWorkWith demand signal (recommender):** the 2025 survey's `*WantToWorkWith` columns are loaded once and the share of a candidate skill's *wanting* among the user's k nearest peers is added (0.3 weight) to the recommendation score. The evidence string now reports both "X% of peers know it" and "Y% of peers want to learn it."

---

## 🚀 Deploy

The repo is Streamlit-Cloud-ready. `runtime.txt` pins Python 3.12, `.streamlit/config.toml` sets the theme, and `requirements.txt` is the single source of truth.

1. Push to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) → **New app** → point at this repo, branch `main`, main file `app/streamlit_app.py` → **Deploy**.
3. ~3-5 min later you have a public URL like `https://<your-handle>-skill-radar.streamlit.app` — put it in the repo description and in your report.

> Note: the first launch will re-run the pipeline (preprocess → recommend artefacts) on Streamlit's servers. The raw `results.csv` is gitignored and must be downloaded at build time. Either host it somewhere and download in the app, or bake the pipeline outputs into the repo (they're small — `processed.parquet` ≈ 4 MB).

## ⚠️ Limitations

- **Sample bias.** Stack Overflow respondents over-represent English-speaking,
  internet-active developers. Generalising to a non-tech or non-SO population
  is unwarranted.
- **Self-reported compensation** is plausibly missing-not-at-random; we keep
  the raw `CompTotal` and flag IQR outliers but do not impute.
- **Cross-sectional associations.** Coefficients are not causal effects — a
  skill's "ROI" reflects correlation with compensation among people who know
  it, not the effect of learning it. Reverse causality (high-paid engineers
  can afford to learn more) is plausible.
- **Currency heterogeneity.** `CompTotal` is in the respondent's day-to-day
  currency; we mitigate with country fixed effects rather than explicit FX
  conversion.
- **Single year.** The 2025 cross-section is a snapshot; skill ROI drifts.
- **Persona assignment for sparse users** uses skill-overlap with each
  cluster's top skills; a developer with a very unusual stack may be
  mis-classified.
- **`OrgSize` not in 2025 parquet** so the four `org_*` dummies in the
  logistic / tree / NB models are constant zero — they carry no
  information. All three models were trained on the same design matrix
  so the comparison is still fair, but the absolute AUCs could be a
  little higher with the right controls.

---

## 📚 Data source

Stack Overflow 2025 Developer Survey — public GitHub archive
[`StackExchange/Survey`](https://github.com/StackExchange/Survey/tree/main/packages/archive/2025).
Pulled via the LFS media host
(`media.githubusercontent.com/.../packages/archive/2025/results.csv`).
