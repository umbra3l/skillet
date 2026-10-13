# Product — skillet

<!-- impeccable:product-schema 1 -->

## Platform

web (Streamlit)

## Users

Individual software engineers, engineering managers, and L&D / People
Analytics teams at technology organisations. They are deciding what to
learn next (or what to fund for their team) under time and budget
pressure, in an environment where skill demand shifts fast (especially
around AI tooling) and where instinct-driven training choices waste
money.

## Product Purpose

skillet turns the public Stack Overflow Developer Survey into a
per-developer, evidence-backed upskilling recommender. Given a
developer's role, experience, current stack and country, it returns
the **top 5 next skills to learn** with the salary-ROI coefficient,
p-value and confidence interval for each, and shows the **predicted
salary lift** of the recommended plan. Built for a Business Analytics
coursework deliverable and for HR-analytics practitioners.

## Positioning

The salary-ROI coefficients come from per-skill multiple-linear
regressions (`log_comp ~ work_exp + country + skill`) on the full
survey, so each recommendation is a *quantified marginal effect on
compensation*, not a popularity score. The clustering, factor analysis
and skill-network modules surface the latent structure of the
population; the multi-year trend module shows how skill ROI has
drifted 2022 → 2025. The KNN-with-Jaccard recommender personalises
this signal to the user's actual stack.

## Operating Context

Streamlit web app, runnable locally and deployable to Streamlit
Community Cloud. Inputs are a role select, an experience slider, a
country select, an AI-tool-usage checkbox, and a multi-select for the
current stack. Outputs are a hero predicted-lift number, a persona
card, five recommendation cards with evidence, a skill-ROI chart, and
a secondary "what-if" calculator. The full analytics pipeline is
executed by `run_all.py` from the raw survey CSV.

## Capabilities and Constraints

- Recommender: Jaccard-distance KNN over 19,210 working developers,
  combined with per-skill MLR coefficients, peer salary premium, and
  peer "want-to-learn" share from the 2025 `*WantToWorkWith` columns.
- Models: SLR + per-skill MLR, binary logistic (with Decision Tree and
  Gaussian Naive Bayes comparators), PCA + K-means clustering,
  Factor Analysis, one-way ANOVA across personas, skill co-occurrence
  network, multi-year (2022-2025) per-skill MLR trend.
- Data: Stack Overflow Developer Survey 2025 (~135 MB CSV); 2022-2024
  for the multi-year trend.
- Run once, cache aggressively (`@st.cache_data`, `@st.cache_resource`);
  the KNN index is built once per process.
- Open decisions: deployment target is the user's choice (Streamlit
  Cloud is the default; the app is local-runnable without it).

## Brand Commitments

- Name: **skillet** (lowercase). A wordplay on "skill" and the cast-iron
  skillet — both a kitchen tool and the thing that *cooks up* the next
  skill. Treated as a single token; never title-cased in display.
- Voice: confident, technical, honest about uncertainty. Reports
  coefficients with their p-values and 95% intervals; never rounds to
  zero without saying so.
- The cast-iron visual world is **pinned** by the user (see
  SURFACE_BRIEF.md and DESIGN.md). The app must inhabit that world;
  no later visual revision may drift to the AI-default look.

## Evidence on Hand

- 19,210 working developers in the processed dataset.
- 32 skills tracked across all four survey years (2022-2025).
- Coefficients and p-values in `data/skill_roi.csv`,
  `data/regression_summary.json`, `data/satisfaction_metrics.json`.
- The trend story in `data/trend_summary.json` and
  `assets/trend_roi.png`.
- Honest limitations: self-reported compensation; sample bias
  (English-speaking, internet-active developers); associations ≠
  causal effects; single-currency comparison (USD-converted columns
  used); `OrgSize` not present in the 2025 processed parquet (the
  four `org_*` dummies are constant zero across all classifiers — a
  data-pipeline gap, documented in the report and README).

## Product Principles

1. **Every number on the page traces to a named model.** Salary impact,
   p-value, CI, persona, peer-prevalence, peer-want — each is the
   output of a specific estimator in `src/`. The UI shows the evidence
   sentence, not just the number.
2. **Honesty over reassurance.** Weak predictions (logistic AUC ≈ 0.56)
   are reported as findings, not hidden. Negative ROI is filtered from
   *recommendations* but kept on the coefficient chart.
3. **Personalised, not generic.** The KNN-with-Jaccard recommender
   personalises to the user's actual stack; the cold-start path is a
   clearly labelled fallback, not a default.
4. **A persona is a vocabulary, not a verdict.** Three K-means
   personas are an interpretable compression of an 86-skill space;
   they are useful for the UI, not a claim about identity.

## Accessibility & Inclusion

Web app; standard keyboard navigation and focus rings through the
app. Animations respect `prefers-reduced-motion`. Colour contrast
meets WCAG AA on the cast-iron palette (body text on cream ≥ 4.5:1;
display on dark hero panel ≥ 7:1). No accessibility-only fallback for
the recommender's core value.
