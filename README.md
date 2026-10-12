# 🎯 Skill Radar

> **A data-driven recommender for tech upskilling.** Built on the 2025 Stack Overflow Developer Survey, Skill Radar answers the question *"what should this specific engineer learn next?"* with statistical evidence from the Business Analytics syllabus.

![Skill Radar banner](assets/05_pca_scatter.png)

---

## ✨ What it does

Given a developer's profile (role, experience, country, current stack, AI-tool usage), Skill Radar assigns them to a **persona** (a K-means cluster in PCA skill-space) and recommends the **top 5 skills to learn next**, ranked by:

- **Salary ROI** — the per-skill multiple-linear-regression coefficient of `log(CompTotal)` controlling for experience and country
- **Satisfaction lift** — the logistic-regression coefficient of `JobSat_high`
- **Persona fit** — how distinctive the skill is within their cluster (prevalence lift)
- **Significance** — a small bonus for statistically robust effects (p < 0.05)

The interactive demo is a Streamlit app with persona cards, evidence per recommendation, and a coefficient chart.

---

## 🧪 The methodology (mapped to the BA syllabus)

| Syllabus module | Where it appears in this repo |
|---|---|
| **M1** Data types, scales, cleansing, supervised vs unsupervised | `src/preprocess.py` — full-time developer filter, multi-select explosion, IQR outlier flag, `log_comp`, `JobSat_high` |
| **M2** Descriptive analytics, all chart types, hypothesis testing, p-values, relationships | `notebooks/01_eda.py` (histogram, bar, scatter, pie, box, heatmap), `src/stats_tests.py` (Welch's t, chi-square, Cramér's V) |
| **M3** SLR, MLR, assumptions, goodness of fit, model comparison | `src/regression.py` — SLR `log_comp ~ WorkExp`; per-skill MLR `log_comp ~ WorkExp + country + skill_s`; full MLR with VIF |
| **M4** Binary logistic, decision tree, KNN, Naive Bayes, LDA, ROC, confusion matrix | `src/classification.py` + `notebooks/04_classification.py` — logistic on `JobSat_high`, ROC + AUC + Youden-optimal confusion matrix; KNN logic in the recommender |
| **M5** K-means, hierarchical, PCA, FA | `src/cluster.py` — PCA(80% var) → K-means (silhouette-tuned) → hierarchical dendrogram |
| **M6** Domain application — HR analytics | The whole project: L&D / upskilling for engineering teams |
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
│   ├── processed.parquet       # one row per working developer
│   ├── processed_meta.json
│   ├── skill_roi.csv           # per-skill salary impact
│   ├── satisfaction_coefs.csv
│   ├── cluster_model.joblib    # PCA + KMeans bundle
│   └── *.json                  # metric snapshots
├── src/
│   ├── preprocess.py
│   ├── stats_tests.py
│   ├── regression.py
│   ├── classification.py
│   ├── cluster.py
│   └── recommend.py            # SkillRadar engine used by the app
├── notebooks/                  # runnable Python analyses (also open as notebooks)
│   ├── 01_eda.py
│   ├── 02_stats_tests.py
│   ├── 03_regression.py
│   ├── 04_classification.py
│   ├── 05_clustering.py
│   └── 06_recommend.py
├── app/streamlit_app.py
├── report/REPORT.md
└── assets/                     # generated figures (PNG)
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

# 4. generate the notebook figures
make            # runs notebooks/01..06

# 5. launch the interactive demo
.venv/bin/streamlit run app/streamlit_app.py
```

A short report lives at [`report/REPORT.md`](report/REPORT.md).

---

## 📊 Results at a glance

| | |
|---|---|
| Respondents after cleaning | ≈ 30,000 full-time professional developers |
| Skill columns kept (top-N) | 60 languages · 25 databases · 20 platforms · 15 web frameworks · 10 dev environments · 10 AI/LLM models |
| PCA components (≥80% var) | (computed at run time — see `data/cluster_meta.json`) |
| K-means k (silhouette) | 3–8 grid, best k chosen automatically |
| SLR R² (WorkExp → log_comp) | see `data/regression_summary.json` |
| Full MLR adj-R² | see `data/regression_summary.json` |
| Logistic AUC (JobSat_high) | see `data/satisfaction_metrics.json` |
| Top-10 skill ROI (salary %) | see `data/skill_roi.csv` |

Top positive coefficients are reproducible by running the pipeline.

---

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

---

## 📚 Data source

Stack Overflow 2025 Developer Survey — public GitHub archive
[`StackExchange/Survey`](https://github.com/StackExchange/Survey/tree/main/packages/archive/2025).
Pulled via the LFS media host
(`media.githubusercontent.com/.../packages/archive/2025/results.csv`).
