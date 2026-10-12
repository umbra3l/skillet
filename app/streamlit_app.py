"""Skill Radar - interactive Streamlit demo.

Run with:  streamlit run app/streamlit_app.py
Requires that the pipeline has been run first (run_all.py) so that
data/processed.parquet and data/*.joblib exist.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

st.set_page_config(page_title="Skill Radar", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_artifacts():
    df = pd.read_parquet(DATA / "processed.parquet")
    roi = pd.read_csv(DATA / "skill_roi.csv")
    cluster = joblib.load(DATA / "cluster_model.joblib")
    meta = json.loads((DATA / "cluster_meta.json").read_text())
    return df, roi, cluster, meta


df, roi, cluster, meta = load_artifacts()

# --- friendly labels for skills (strip the prefix) -----------------------------
SKILL_GROUPS = {
    "Languages": [c for c in df.columns if c.startswith("lang_")],
    "Databases": [c for c in df.columns if c.startswith("db_")],
    "Cloud platforms": [c for c in df.columns if c.startswith("plat_")],
    "Web frameworks": [c for c in df.columns if c.startswith("web_")],
    "Dev environments": [c for c in df.columns if c.startswith("devenv_")],
    "AI / LLM models": [c for c in df.columns if c.startswith("aimodel_")],
}


def short(c: str) -> str:
    return c.split("_", 1)[1].replace("_", " ").title() if "_" in c else c


# --- sidebar inputs ------------------------------------------------------------
st.sidebar.title("🎯 Skill Radar")
st.sidebar.caption("Tell us about you, and we'll recommend what to learn next.")

devtype = st.sidebar.selectbox(
    "Your role",
    options=sorted([c for c in df.columns if c.startswith("role_")]),
    format_func=lambda c: short(c),
)
workexp = st.sidebar.slider("Years of professional experience", 0, 50, 5)
country = st.sidebar.selectbox("Country",
    options=sorted([c for c in df.columns if c.startswith("country_")]),
    format_func=lambda c: c.replace("country_", "").replace("_", " "),
    index=0)
ai_user = st.sidebar.checkbox("I currently use AI tools in my development workflow", value=True)

st.sidebar.markdown("**Your current stack**")
have: list[str] = []
for group, cols in SKILL_GROUPS.items():
    chosen = st.sidebar.multiselect(group, options=cols, format_func=short, max_selections=6)
    have.extend(chosen)

# --- persona & recommendations -------------------------------------------------
# Lazy import to keep the page snappy
import sys
sys.path.insert(0, str(ROOT))
from src.recommend import SkillRadar  # noqa: E402

sr = SkillRadar()
rec = sr.recommend(current_skills=have, top_n=5)

# --- main layout ---------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Your persona", f"#{rec['persona_id']}")
c2.metric("People like you in survey", f"{rec['persona_size']:,}")
c3.metric("Skills you listed", f"{len(have)}")

st.subheader("👥 Your developer persona")
if rec["persona_top_skills"]:
    st.write("**Most distinctive skills in your cluster:** "
             + ", ".join(short(s) for s in rec["persona_top_skills"]))

st.subheader("🚀 Top recommended next skills")
if not rec["recommendations"]:
    st.info("Add a few skills on the left so we can recommend complementary ones.")
else:
    for r in rec["recommendations"]:
        with st.container(border=True):
            st.markdown(f"### {short(r['skill'])}")
            cols = st.columns(4)
            cols[0].metric("Salary impact", f"{r['salary_impact_pct']:+.1f}%",
                           help="Multiple-linear-regression coefficient, holding experience and country constant.")
            cols[1].metric("p-value", f"{r['salary_p_value']:.3g}")
            cols[2].metric("Satisfaction effect", f"{r['satisfaction_coef']:+.2f}",
                           help="Standardized logistic-regression coefficient (JobSat model).")
            cols[3].metric("Survey users who know it", f"{r['n_known']:,}")
            st.caption(r["evidence"])
            ci = r.get("salary_ci_pct", [None, None])
            if ci[0] is not None and not np.isnan(ci[0]):
                st.caption(f"95% CI for salary impact: {ci[0]:+.1f}% to {ci[1]:+.1f}%")

# --- coefficient chart --------------------------------------------------------
st.subheader("📊 Where these recommendations come from")
top = roi.head(20).copy()
top["skill_label"] = top["skill"].map(short)
fig = px.bar(top, x="pct_salary_impact", y="skill_label", orientation="h",
             error_x=top["ci_hi"] - top["pct_salary_impact"],
             error_x_minus=top["pct_salary_impact"] - top["ci_lo"],
             color="p_value", color_continuous_scale="RdYlGn_r",
             title="Top 20 skills by salary impact (per-skill MLR, control = work exp + country)")
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=520)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Methodology & data"):
    st.markdown(f"""
**Skill Radar** maps the *2025 Stack Overflow Developer Survey*
({meta.get('n_samples', 0):,} professional developers in the processed dataset)
onto a skill-vs-outcome space, then recommends the next skill that maximises
salary uplift and job-satisfaction lift, conditioned on your role, experience
and country.

Pipeline:
1. **Preprocess** (`src/preprocess.py`) — explode multi-select skill columns
   (languages, databases, cloud platforms, frameworks, dev environments, AI/LLM
   models), build a binary skill matrix, IQR-flag salary outliers, derive
   `log_comp` and a binary `JobSat_high`.
2. **Hypothesis tests** (`src/stats_tests.py`) — Welch's t-tests (AI users vs
   not, on salary and satisfaction) and chi-square tests of independence
   (AI usage vs remote work, role vs top language), with Cramér's V.
3. **Regression** (`src/regression.py`) — SLR (work exp → log compensation) and
   per-skill MLR (`log_comp ~ work_exp + country + skill`) yielding a
   skill-by-skill **ROI** in log points, converted to percent salary impact.
4. **Classification** (`src/classification.py`) — logistic regression of
   `JobSat_high` on experience, AI usage, and the top-30 most common skills;
   ROC / AUC / Youden-optimal threshold and confusion matrix.
5. **Clustering** (`src/cluster.py`) — PCA on the standardized skill matrix →
   K-means personas chosen by silhouette score, hierarchical dendrogram for
   visualisation, persona profiles by prevalence lift.
6. **Recommender** (`src/recommend.py`) — assigns the user to a persona, ranks
   the candidate skills they don't have by `(salary ROI) + (satisfaction coef) +
   log(persona lift) + significance bonus`, returns the top 5 with the
   statistical evidence for each.

**Limitations.** Stack Overflow respondents over-represent English-speaking,
internet-active developers; self-reported compensation is plausibly
missing-not-at-random; coefficients are *associations*, not causal effects;
single-year cross-section.
""")

st.caption("Built with Python · pandas · statsmodels · scikit-learn · Streamlit · Plotly")
