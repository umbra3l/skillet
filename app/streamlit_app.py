"""Skill Radar — interactive Streamlit demo.

Run with:  streamlit run app/streamlit_app.py
Requires that the pipeline has been run first (run_all.py) so that
data/processed.parquet and data/*.joblib exist.

Layout:
  fixed header → hero (predicted salary lift) → persona + top-5 cards
  → methodology → skill ROI landscape → secondary what-if calculator
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Skill Radar — what to learn next", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

from src.recommend import SkillRadar          # noqa: E402
from src.salary_calc import expected_uplift   # noqa: E402

# --------------------------------------------------------------------------------------
# theme css
# --------------------------------------------------------------------------------------
_STYLES = ROOT / ".streamlit" / "style.css"
if _STYLES.exists():
    st.markdown("<style>" + _STYLES.read_text() + "</style>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# display labels
# --------------------------------------------------------------------------------------
PERSONA_NAMES = {
    0: ".NET stack",
    1: "Systems / enterprise",
    2: "Modern JS / web tooling",
}

ALIASES = {
    "lang_Bash_Shell_all_shells": "Bash / Shell",
    "lang_HTML_CSS": "HTML / CSS",
    "lang_C_2": "C#",
    "lang_C_3": "C++",
    "lang_JavaScript": "JavaScript",
    "lang_TypeScript": "TypeScript",
    "lang_VBA": "VBA",
    "lang_Visual_Basic_Net": "Visual Basic .NET",
    "lang_Jupyter": "Jupyter",
    "db_BigQuery": "BigQuery",
    "db_Clickhouse": "ClickHouse",
    "db_DuckDB": "DuckDB",
    "db_Dynamodb": "DynamoDB",
    "db_MariaDB": "MariaDB",
    "db_MongoDB": "MongoDB",
    "db_MySQL": "MySQL",
    "db_Neo4J": "Neo4j",
    "db_PostgreSQL": "PostgreSQL",
    "db_InfluxDB": "InfluxDB",
    "devenv_Visual_Studio_Code": "VS Code",
    "devenv_Visual_Studio": "Visual Studio",
    "devenv_Jupyter_Notebook_JupyterLab": "JupyterLab",
    "devenv_IntelliJ_IDEA": "IntelliJ IDEA",
    "devenv_Notepad": "Notepad++",
    "plat_Amazon_Web_Services_AWS": "AWS",
    "plat_Maven_build_tool": "Maven",
    "plat_Microsoft_Azure": "Azure",
    "plat_Google_Cloud": "Google Cloud",
    "plat_npm": "npm",
    "plat_pnpm": "pnpm",
    "web_ASP_NET": "ASP.NET",
    "web_ASP_NET_Core": "ASP.NET Core",
    "web_FastAPI": "FastAPI",
    "web_Next_js": "Next.js",
    "web_Node_js": "Node.js",
    "web_Vue_js": "Vue.js",
    "web_jQuery": "jQuery",
    "web_Spring_Boot": "Spring Boot",
    "lang_OCaml": "OCaml",
    "lang_MATLAB": "MATLAB",
    "aimodel_Anthropic": "Anthropic",
    "aimodel_DeepSeek_R_Reasoning_models": "DeepSeek-R (reasoning)",
    "aimodel_DeepSeek_V_General_purpose_models": "DeepSeek-V (general)",
    "aimodel_Gemini_Flash_general_purpose_models": "Gemini Flash",
    "aimodel_Gemini_Pro_Reasoning_models": "Gemini Pro (reasoning)",
    "aimodel_Meta_Llama_all_models": "Meta Llama",
    "aimodel_X_Grok_models": "xAI Grok",
    "aimodel_openAI_GPT_chatbot_models": "OpenAI GPT (chat)",
    "aimodel_openAI_Image_generating_models": "OpenAI image models",
    "aimodel_openAI_Reasoning_models": "OpenAI (reasoning)",
}

_KEEP = {"npm", "pnpm", "js", "css", "sql", "aws", "net", "io", "ml", "ai", "vue", "cli", "orm"}


def short(col: str) -> str:
    """Friendly label for a skill column name."""
    if col in ALIASES:
        return ALIASES[col]
    name = col.split("_", 1)[1] if "_" in col else col
    return " ".join(w if w.lower() in _KEEP else w.title() for w in name.replace("_", " ").split())


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def signed(x: float, dec: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{dec}f}"


def tone(x: float) -> str:
    return "pos" if (x is not None and not np.isnan(x) and x >= 0) else "neg"


# --------------------------------------------------------------------------------------
# cached loaders
# --------------------------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    df = pd.read_parquet(DATA / "processed.parquet")
    roi = pd.read_csv(DATA / "skill_roi.csv")
    meta = json.loads((DATA / "cluster_meta.json").read_text())
    return df, roi, meta


@st.cache_resource
def get_engine() -> SkillRadar:
    """One SkillRadar per process: the KNN index is built once, not per rerun."""
    return SkillRadar()


@st.cache_data
def lift_for(skills: tuple[str, ...]) -> dict:
    """Cached wrapper around src.salary_calc.expected_uplift."""
    return expected_uplift(list(skills))


df, roi, meta = load_artifacts()
N_SAMPLE = int(meta.get("n_samples", len(df)))
PERSONA_META = {int(p["cluster"]): p for p in meta["personas"]}
# The recommender rounds p-values for its payload; re-read the full precision
# ones from skill_roi.csv so the cards never print "0.0e+00".
P_MAP = dict(zip(roi["skill"], roi["p_value"]))

SKILL_GROUPS = {
    "Languages": [c for c in df.columns if c.startswith("lang_")],
    "Databases": [c for c in df.columns if c.startswith("db_")],
    "Cloud platforms": [c for c in df.columns if c.startswith("plat_")],
    "Web frameworks": [c for c in df.columns if c.startswith("web_")],
    "Dev environments": [c for c in df.columns if c.startswith("devenv_")],
    "AI / LLM models": [c for c in df.columns if c.startswith("aimodel_")],
}

# --------------------------------------------------------------------------------------
# sidebar — deliberately quiet
# --------------------------------------------------------------------------------------
st.sidebar.markdown("### Your profile")
st.sidebar.caption("Answer four questions, add the tools you already use, and the "
                   "evidence updates instantly.")

devtype = st.sidebar.selectbox(
    "Your role",
    options=sorted([c for c in df.columns if c.startswith("role_")]),
    format_func=lambda c: short(c),
)
workexp = st.sidebar.slider("Years of professional experience", 0, 50, 5)
country = st.sidebar.selectbox(
    "Country",
    options=sorted([c for c in df.columns if c.startswith("country_")]),
    format_func=lambda c: c.replace("country_", "").replace("_", " "),
    index=0,
)
ai_user = st.sidebar.checkbox("I use AI tools in my workflow", value=True)

st.sidebar.markdown('<div class="sr-side-rule"></div>', unsafe_allow_html=True)
st.sidebar.markdown("**YOUR CURRENT STACK**")
have: list[str] = []
for group, cols in SKILL_GROUPS.items():
    if not cols:          # e.g. no aimodel_* columns in this year's schema
        continue
    chosen = st.sidebar.multiselect(group, options=cols, format_func=short, max_selections=6)
    have.extend(chosen)

st.sidebar.markdown(
    '<div class="sr-side-note">Every number below is read from '
    f'{N_SAMPLE:,} survey responses. Leave the stack empty to see the '
    "cold-start plan (the globally highest-ROI skills).</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# recommendations + hero lift
# --------------------------------------------------------------------------------------
engine = get_engine()
rec = engine.recommend(current_skills=list(have), top_n=5)
recs = rec["recommendations"]
rec_labels = [short(r["skill"]) for r in recs]
hero = lift_for(tuple(r["skill"] for r in recs))
cold = not have

# --------------------------------------------------------------------------------------
# fixed header
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="sr-header">
      <div class="sr-brand"><span class="sr-logo">◎</span>Skill&nbsp;Radar
        <span class="sr-badge">demo</span></div>
      <div class="sr-header-sep"></div>
      <div class="sr-tagline">What should this engineer learn next? — a recommender
        built on the 2025 Stack Overflow Developer Survey.</div>
      <div class="sr-header-right">
        <span class="sr-chip"><b>{N_SAMPLE:,}</b> developers</span>
        <span class="sr-chip"><b>{len(roi)}</b> skills scored</span>
        <span class="sr-chip">p-values attached</span>
      </div>
    </div>
    <div class="sr-header-spacer"></div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# HERO — predicted salary lift (the one number the page is about)
# --------------------------------------------------------------------------------------
total = float(hero["total_pct"])
n_sig = int(hero["n_significant"])
n_eval = int(hero["n_skills"])
ci_lo = float(np.nansum([b["ci_lo"] for b in hero["breakdown"]]))
ci_hi = float(np.nansum([b["ci_hi"] for b in hero["breakdown"]]))
premium_pct = (np.exp(rec["salary_premium_log"]) - 1.0) * 100.0
bar_w = float(min(max(total, 1.5), 100.0))

eyebrow = "Predicted salary lift" + (" · cold start" if cold else "")
if cold:
    sub = ("You haven't entered a stack yet — this is the lift attached to the "
           "<b>5 highest-ROI skills in the whole survey</b>. Pick your stack on the "
           "left to personalise it.")
else:
    sub = (f"Modelled change in compensation if you learn "
           f"<b>these {n_eval} skills</b>, holding experience and country constant.")

st.markdown(
    f"""
    <section class="sr-hero">
      <div>
        <div class="sr-eyebrow"><span class="sr-pulse"></span>{eyebrow}</div>
        <div class="sr-hero-number"><span class="sr-plus">+</span>{total:.0f}<span class="sr-unit">%</span></div>
        <div class="sr-hero-sub">{sub}</div>
        <div class="sr-hero-skills">↳ {", ".join(rec_labels) if rec_labels else "—"}</div>
        <div class="sr-hero-bar"><i style="width:{bar_w:.1f}%"></i></div>
        <div class="sr-hero-scale"><span>0%</span><span>scale: +100%</span></div>
      </div>
      <div class="sr-hero-stats">
        <div class="sr-hstat good">
          <span class="k">Skills contributing</span>
          <span class="v">{n_eval} / {n_eval}</span>
          <span class="s">in your recommended plan</span>
        </div>
        <div class="sr-hstat good">
          <span class="k">Statistically significant</span>
          <span class="v">{n_sig} / {n_eval}</span>
          <span class="s">p &lt; 0.05 in the salary model</span>
        </div>
        <div class="sr-hstat">
          <span class="k">Approx. 95% interval</span>
          <span class="v mono">{signed(ci_lo, 0)}% → {signed(ci_hi, 0)}%</span>
          <span class="s">per-skill CIs summed (independence assumed)</span>
        </div>
        <div class="sr-hstat">
          <span class="k">Your peers' salary premium</span>
          <span class="v mono">{"—" if rec["n_neighbours"] == 0 else signed(premium_pct) + "%"}</span>
          <span class="s">{"cold start — no peer group yet" if rec["n_neighbours"] == 0
                            else f"{rec['n_neighbours']:,} nearest peers vs survey average"}</span>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if cold:
    st.markdown(
        '<div class="sr-prompt"><span class="arrow">→</span><span><b>Pick your stack on the '
        "left</b> to see your persona, your personalised top-5 skills to learn, and the "
        "predicted salary lift for <i>you</i>.</span></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# 01 persona  |  02 recommendations
# --------------------------------------------------------------------------------------
pid = int(rec["persona_id"])
p_meta = PERSONA_META.get(pid, {})
share = float(p_meta.get("share", rec["persona_size"] / max(N_SAMPLE, 1)))
distinguishing = " ".join(
    f'<span class="sr-chip-skill">{short(s)}</span>' for s in rec["persona_top_skills"]
) or "<span class='sr-chip-skill'>—</span>"

left, right = st.columns([1, 1.45], gap="large")

with left:
    st.markdown('<div class="sr-col-label">01 · Your persona</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sr-sticky">
        <div class="sr-card sr-card--persona">
          <span class="sr-rank">CLUSTER {pid}</span>
          <div class="sr-persona-name">{PERSONA_NAMES.get(pid, f'Cluster {pid}')}</div>
          <div class="sr-persona-meta">{rec['persona_size']:,} developers ·
            {share * 100:.0f}% of the {N_SAMPLE:,} surveyed</div>
          <div class="sr-share"><i style="width:{min(share * 100, 100):.1f}%"></i></div>
          <div class="sr-share-legend"><span>share of sample</span><span>{share * 100:.1f}%</span></div>
          <div class="sr-card-sub">Skills that define this persona</div>
          <div class="sr-chips">{distinguishing}</div>
          <div class="sr-card-sub">Your nearest peers</div>
          <div class="sr-mini-stats">
            <span class="sr-mini"><b>{rec['n_neighbours']:,}</b> matched</span>
            <span class="sr-mini">premium <b>{"—" if rec["n_neighbours"] == 0 else signed(premium_pct) + "%"}</b></span>
            <span class="sr-mini">skills listed <b>{len(have)}</b></span>
          </div>
        </div>
        <div class="sr-card sr-card--note">
          <div class="sr-card-sub" style="margin-top:0">How to read a card</div>
          <p class="sr-evidence"><b>Salary impact</b> is the regression coefficient for that
          skill expressed in percent, with experience and country held constant.
          <b>p-value</b> tests whether the association is distinguishable from zero
          (below 0.05 is the usual bar). The <b>95% interval</b> is the range the data
          supports for that coefficient.</p>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div class="sr-col-label">02 · Top skills to learn next</div>', unsafe_allow_html=True
    )
    if not recs:
        st.info("No candidate skills left — you've listed almost everything we score.")
    else:
        cards = []
        for i, r in enumerate(recs, start=1):
            label = short(r["skill"])
            imp = float(r["salary_impact_pct"])
            p_exact = float(P_MAP.get(r["skill"], r["salary_p_value"]))
            ci = r.get("salary_ci_pct", [None, None])
            has_ci = ci[0] is not None and not (isinstance(ci[0], float) and np.isnan(ci[0]))
            ci_text = (f"95% CI {signed(ci[0])}% → {signed(ci[1])}%" if has_ci
                       else "95% CI unavailable")
            if have:
                peer_text = (f"{r['nbr_prevalence_pct']:.0f}% of your peers know it · "
                             f"{r['nbr_want_pct']:.0f}% want to learn it")
            else:
                peer_text = f"{r['nbr_prevalence_pct']:.0f}% of surveyed developers know it"
            evidence = r["evidence"].replace(r["skill"], label)
            if not have:
                evidence = evidence.replace("Cold-start recommendation:", "Cold start —")
            cards.append(f"""
            <div class="sr-card {'neg' if imp < 0 else 'pos'}">
              <div class="sr-card-head">
                <span class="sr-rank">{i:02d}</span>
                <span class="sr-skill">{label}</span>
                <span class="sr-head-right">
                  <span class="sr-score" title="Internal recommender score: peer prevalence + 0.5× salary ROI + 0.3× peer demand + 0.2× peer salary premium (see Methodology)">score {r['score']:.3f}</span>
                  <span class="sr-pill {'pos' if imp >= 0 else 'neg'}">{signed(imp)}%</span>
                </span>
              </div>
              <div class="sr-metrics">
                <div><span class="k">Salary impact</span><span class="v {tone(imp)}">{signed(imp)}%</span></div>
                <div><span class="k">p-value</span><span class="v">{fmt_p(p_exact)}</span></div>
                <div><span class="k">Satisfaction coef.</span><span class="v {tone(r['satisfaction_coef'])}">{signed(r['satisfaction_coef'], 3)}</span></div>
                <div><span class="k">Users who know it</span><span class="v">{r['n_known']:,}</span></div>
              </div>
              <div class="sr-ci">{ci_text} · {peer_text}</div>
              <p class="sr-evidence">{evidence}</p>
            </div>
            """)
        st.markdown("".join(cards), unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# 03 methodology
# --------------------------------------------------------------------------------------
st.markdown('<div class="sr-section"><div class="sr-kicker">03 · Methodology</div></div>',
            unsafe_allow_html=True)

with st.expander("How these recommendations are scored", expanded=False):
    st.markdown(
        f"""
For every skill you don't have, the recommender computes

```
score(s)  =  peer_prevalence(s)                       # how common it is in your 300 nearest peers
           + 0.5 × salary_ROI(s)                      # per-skill MLR coefficient (log points)
           + 0.2 × peer_salary_premium                # how much your peer group earns overall
           + 0.3 × peer_demand(s)                     # share of peers who "want" to learn it
           + 0.05 if p(s) < 0.05                      # significance bonus
```

and returns the top 5. **Skill Radar** maps the *2025 Stack Overflow Developer Survey*
({N_SAMPLE:,} professional developers in the processed dataset) onto a
skill-vs-outcome space, then recommends the next skill that maximises salary uplift
and job-satisfaction lift, conditioned on your role, experience and country.

**Pipeline**

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
6. **Recommender** (`src/recommend.py`) — assigns the user to a persona, ranks the
   candidate skills they don't have by the score above, and returns the top 5 with
   the statistical evidence for each.

**Limitations.** Stack Overflow respondents over-represent English-speaking,
internet-active developers; self-reported compensation is plausibly
missing-not-at-random; coefficients are *associations*, not causal effects;
single-year cross-section.
"""
    )

# --------------------------------------------------------------------------------------
# 04 skill ROI landscape
# --------------------------------------------------------------------------------------
st.markdown(
    '<div class="sr-section"><div class="sr-kicker">04 · Skill ROI landscape</div>'
    '<div class="sr-section-row"><h2>Where these recommendations come from</h2>'
    '<span class="sr-section-note">per-skill MLR · controls = work experience + country</span>'
    "</div>"
    '<p class="sr-lead">Every recommendation traces back to a row in this chart: the top 20 '
    "skills by modelled salary impact, with 95% confidence intervals as whiskers. Bars are "
    "coloured by significance — deep green is far below p = 0.05, rose is not significant — "
    "and the skills in your recommended plan are outlined in indigo.</p></div>",
    unsafe_allow_html=True,
)

top = roi.head(20).copy()
top["skill_label"] = top["skill"].map(short)
top["neglogp"] = -np.log10(top["p_value"].clip(lower=1e-300))
# Highlight the skills that are in the user's current plan.
plan = {r["skill"] for r in recs}
# Scale is −log10(p) over 0..30, with the p = 0.05 threshold (~1.3) sitting at
# 4% of the ramp: anything statistically significant turns green immediately.
fig = px.bar(
    top,
    x="pct_salary_impact",
    y="skill_label",
    orientation="h",
    color="neglogp",
    color_continuous_scale=[
        (0.000, "#e11d48"), (0.020, "#f97316"), (0.043, "#facc15"),
        (0.140, "#6ee7b7"), (0.400, "#10b981"), (1.000, "#047857"),
    ],
    range_color=[0, 30],
    labels={"pct_salary_impact": "Modelled salary impact (%)", "skill_label": "",
            "neglogp": "−log₁₀(p)"},
    hover_data={"p_value": ":.2e", "users": ":,", "neglogp": False},
    error_x=top["ci_hi"] - top["pct_salary_impact"],
    error_x_minus=top["pct_salary_impact"] - top["ci_lo"],
)
fig.update_traces(
    error_x_color="#94a3b8",
    error_x_thickness=1.2,
    error_x_width=0,
    marker_line_width=[4 if s in plan else 0 for s in top["skill"]],
    marker_line_color="#312e81",
)
fig.update_layout(
    template="plotly_white",
    height=580,
    margin=dict(l=0, r=16, t=8, b=8),
    font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color="#475569"),
    yaxis={"categoryorder": "total ascending", "title": "",
           "showgrid": False, "linecolor": "#e2e8f0", "automargin": True},
    xaxis={"title": "", "gridcolor": "#eef2f7", "zeroline": True,
           "zerolinecolor": "#94a3b8", "zerolinewidth": 1},
    coloraxis_colorbar=dict(
        title="significance", thickness=10, len=0.45, outlinewidth=0,
        tickvals=[1.3, 5, 10, 20, 30],
        ticktext=["p=0.05", "1e−5", "1e−10", "1e−20", "1e−30"],
        ticks="outside", tickcolor="#e2e8f0", tickfont=dict(size=11),
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="#0f172a", font_color="#fff", font_family="JetBrains Mono, monospace"),
)
st.plotly_chart(fig, width="stretch", theme=None,
                config={"displayModeBar": False, "displaylogo": False})

# --------------------------------------------------------------------------------------
# 05 secondary what-if calculator
# --------------------------------------------------------------------------------------
st.markdown(
    '<div class="sr-section"><div class="sr-kicker">05 · What-if calculator</div>'
    '<div class="sr-section-row"><h2>What if I also learned these?</h2>'
    '<span class="sr-section-note">secondary tool — same model, your own shortlist</span></div>'
    '<p class="sr-lead">Add any other skills you are weighing up and the same per-skill '
    "coefficients are summed for you.</p></div>",
    unsafe_allow_html=True,
)

known = set(have) | {r["skill"] for r in recs}
extras = st.multiselect(
    "Pick additional skills you're considering",
    options=sorted([c for c in roi["skill"] if c not in known]),
    format_func=short, max_selections=8, key="salary_calc_extras",
    placeholder="Choose up to 8 skills…",
)

if extras:
    calc = lift_for(tuple(extras))
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted salary lift", f"{calc['total_pct']:+.1f}%",
                  help="Sum of per-skill MLR coefficients, converted to percent via exp(). "
                       "Assumes independent effects (see caveat below).")
        m2.metric("Skills evaluated", f"{calc['n_skills']}")
        m3.metric("Significant (p<0.05)", f"{calc['n_significant']}/{calc['n_skills']}")
        m4.metric("Not yet recommended", f"{len(extras)}")
    if calc["breakdown"]:
        bd = pd.DataFrame(calc["breakdown"])
        bd["skill_label"] = bd["skill"].map(short)
        fig2 = px.bar(
            bd, x="pct_impact", y="skill_label", orientation="h",
            color="p_value", color_continuous_scale=[
                (0.0, "#047857"), (0.5, "#facc15"), (1.0, "#f43f5e")],
            range_color=[0, 0.1],
            title="Per-skill contribution to the predicted lift",
            error_x=bd["ci_hi"] - bd["pct_impact"],
            error_x_minus=bd["pct_impact"] - bd["ci_lo"],
            labels={"pct_impact": "salary impact (%)", "skill_label": "",
                    "p_value": "p-value"},
        )
        fig2.update_layout(
            yaxis={"categoryorder": "total ascending", "title": "", "automargin": True},
            xaxis={"title": "", "gridcolor": "#eef2f7", "automargin": True},
            height=240 if len(bd) < 3 else 360,
            margin=dict(l=0, r=16, t=44, b=8),
            font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=13, color="#475569"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(title="p-value", thickness=10, len=0.6, outlinewidth=0,
                                    tickformat=".0e"),
        )
        fig2.update_traces(error_x_color="#64748b", error_x_thickness=1.3, error_x_width=0)
        st.plotly_chart(fig2, width="stretch", theme=None,
                        config={"displayModeBar": False, "displaylogo": False})
        st.caption(
            "Caveat: this sums per-skill MLR coefficients under an independence assumption. "
            "In reality, skills cluster (a Kubernetes user usually also knows Docker), so the "
            "true joint lift is smaller than the sum. We expose the 95% CI per skill but not a "
            "combined CI for that reason."
        )
else:
    st.caption("Nothing selected yet — pick one or more skills above to see their combined lift.")

# --------------------------------------------------------------------------------------
# footer
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="sr-footer">
      <span>Skill Radar · upskilling recommender · demo build</span>
      <span>Python · pandas · statsmodels · scikit-learn · Streamlit · Plotly</span>
      <span>Data: 2025 Stack Overflow Developer Survey ({N_SAMPLE:,} rows)</span>
    </div>
    """,
    unsafe_allow_html=True,
)
