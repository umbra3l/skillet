"""skillet — interactive Streamlit app.

Run with:  streamlit run app/streamlit_app.py
Requires that the pipeline has been run first (run_all.py) so that
data/processed.parquet and data/*.joblib exist.

Editorial Magazine visual world: a quarterly's cover story on tech
hiring. The first session load shows a quiet cover fold while the KNN
index and salary calculator are first-built; on every subsequent rerun
the app is instant thanks to @st.cache_resource and @st.cache_data.
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

st.set_page_config(
    page_title="skillet — what to learn next",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# theme css
_CSS = ROOT / ".streamlit" / "style.css"
if _CSS.exists():
    st.markdown("<style>" + _CSS.read_text() + "</style>",
                unsafe_allow_html=True)

from src.preprocess import slugify            # noqa: E402
from src.recommend import MIN_PEERS, SkillRadar  # noqa: E402
from src.salary_calc import expected_uplift   # noqa: E402

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
    if col in ALIASES:
        return ALIASES[col]
    name = col.split("_", 1)[1] if "_" in col else col
    return " ".join(w if w.lower() in _KEEP else w.title()
                   for w in name.replace("_", " ").split())


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_p_pub(p: float) -> str:
    """p-values set like a publication: 4×10⁻⁶, not 4.0e-06."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 1e-3:
        mantissa, exp = f"{p:.1e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}×10{str(int(exp)).translate(_SUP)}"
    return f"{p:.3f}"


def signed(x, dec: int = 1) -> str:
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
    """One SkillRadar per process: the KNN index is built once."""
    return SkillRadar()


@st.cache_data
def lift_for(skills: tuple[str, ...]) -> dict:
    return expected_uplift(list(skills))


# --------------------------------------------------------------------------------------
# cover fold: the quiet first-render while the cache warms.
# A top hairline, the wordmark arriving once, a rust rule infilling.
# Once caches are warm, the cover unfolds and this never shows again.
# --------------------------------------------------------------------------------------
COVERFOLD = """
<div class="sk-coverfold">
  <div class="sk-fold-rule"></div>
  <div class="sk-fold-word">skillet</div>
  <div class="sk-fold-bar"><i></i></div>
</div>
"""


def render_coverfold_and_warm() -> None:
    """First-load cover fold. Shows the wordmark, then triggers the heavy
    work (data + KNN build), then reruns so the cover body is instant."""
    st.html(COVERFOLD)
    # warm caches (this is the work the fold embodies)
    _ = load_artifacts()
    _ = get_engine()
    st.session_state["skillet_ready"] = True
    st.rerun()


if not st.session_state.get("skillet_ready", False):
    render_coverfold_and_warm()
    st.stop()

# --------------------------------------------------------------------------------------
# main app (reached on every rerun after the first session load)
# --------------------------------------------------------------------------------------
df, roi, meta = load_artifacts()
N_SAMPLE = int(meta.get("n_samples", len(df)))
PERSONA_META = {int(p["cluster"]): p for p in meta["personas"]}
# re-read full-precision p-values from the source CSV (the recommender
# payload rounds to 4 dp; the cards want the full value)
P_MAP = dict(zip(roi["skill"], roi["p_value"]))

SKILL_GROUPS = {
    "Languages":       [c for c in df.columns if c.startswith("lang_")],
    "Databases":       [c for c in df.columns if c.startswith("db_")],
    "Cloud platforms": [c for c in df.columns if c.startswith("plat_")],
    "Web frameworks":  [c for c in df.columns if c.startswith("web_")],
    "Dev environments":[c for c in df.columns if c.startswith("devenv_")],
    "AI / LLM models": [c for c in df.columns if c.startswith("aimodel_")],
}

# --- filter section: in-page (no sidebar) ---------------------------------------------
st.html('<div class="sk-filter-shell"><div class="sk-filter-rule"></div></div>')

st.html('<div class="sk-filter-sec"><h3 class="sk-filter-title">Your profile</h3></div>')

devtype = st.selectbox(
    "Role",
    options=sorted([c for c in df.columns if c.startswith("role_")]),
    format_func=short,
    key="sk_devtype",
)
workexp = st.slider("Years of professional experience", 0, 50, 5, key="sk_workexp")
country = st.selectbox(
    "Country",
    options=sorted([c for c in df.columns if c.startswith("country_")]),
    format_func=lambda c: c.replace("country_", "").replace("_", " "),
    index=0,
    key="sk_country",
)
ai_user = st.checkbox("I use AI tools in my workflow", value=True, key="sk_ai")

st.html('<div class="sk-filter-sec"><h3 class="sk-filter-title">Your current stack</h3></div>')
have: list[str] = []
for group, cols in SKILL_GROUPS.items():
    if not cols:
        continue
    chosen = st.multiselect(group, options=cols, format_func=short, max_selections=6,
                            key=f"sk_stack_{group}")
    have.extend(chosen)

st.html(
    f'<div class="sk-filter-dateline">{N_SAMPLE:,} developers · {len(roi)} skills scored</div>'
)
engine = get_engine()
# `country` selectbox returns a country_* column; the recommender wants the
# raw Country value ("United States of America") to match the peer pool.
_COUNTRY_RAW = {f"country_{slugify(c)}": c for c in df["Country"].dropna().unique()}
rec = engine.recommend(
    current_skills=list(have),
    top_n=5,
    country=_COUNTRY_RAW.get(country),
    workexp=int(workexp),
    role=devtype,
    ai_user=bool(ai_user),
)
recs = rec["recommendations"]
rec_labels = [short(r["skill"]) for r in recs]
hero = lift_for(tuple(r["skill"] for r in recs))
cold = not have

# --------------------------------------------------------------------------------------
# cover illustration: a drawn-in-brand publication diagram.
# A coefficient skyline — thin bars for the scored skills, a rust
# trend line across them, ember markers on the recommended plan —
# set over a hairline baseline, captioned like a cover figure.
# --------------------------------------------------------------------------------------
def _cover_illus_html(n_skills: int) -> str:
    bars = []
    heights = [34, 52, 44, 70, 58, 88, 76, 104, 92, 120, 108, 134, 122, 148,
               138, 162, 150, 172, 160, 182, 172, 196, 186, 210]
    for j, h in enumerate(heights):
        x = 60 + j * 25
        bars.append(
            f'<line x1="{x}" y1="{280 - h}" x2="{x}" y2="280" '
            f'stroke="#3d4046" stroke-width="3" opacity="0.55"/>'
        )
    trend = "M60,246 L110,228 L160,232 L210,204 L260,210 L310,182 L360,188 L410,158 L460,162 L510,132 L560,136 L610,104 L660,70"
    markers = [(210, 204), (310, 182), (410, 158), (510, 132), (610, 104)]
    dots = "\n".join(
        f'<circle cx="{x}" cy="{y}" r="5" fill="#f3eee5" stroke="#c44a1a" stroke-width="2"/>'
        for x, y in markers
    )
    return f"""
    <div class="sk-illus">
      <svg viewBox="0 0 720 300" role="img" aria-label="Stylised diagram of per-skill salary coefficients rising across the scored skills">
        <line x1="40" y1="280" x2="680" y2="280" stroke="#0e0d0a" stroke-width="1.5"/>
        <line x1="40" y1="274" x2="40" y2="286" stroke="#0e0d0a" stroke-width="1.5"/>
        <line x1="680" y1="274" x2="680" y2="286" stroke="#0e0d0a" stroke-width="1.5"/>
        <line x1="40" y1="200" x2="680" y2="200" stroke="#d8d0bf" stroke-width="1"/>
        <line x1="40" y1="120" x2="680" y2="120" stroke="#d8d0bf" stroke-width="1"/>
        {"".join(bars)}
        <path d="{trend}" fill="none" stroke="#a23a22" stroke-width="2"/>
        {dots}
        <text x="648" y="262" font-family="JetBrains Mono, monospace" font-size="11" fill="#6b6e74">+31%</text>
        <text x="52" y="184" font-family="JetBrains Mono, monospace" font-size="11" fill="#6b6e74">p &lt; .05</text>
      </svg>
      <div class="sk-illus-cap">Cover · per-skill salary coefficients across {n_skills} skills, 2025</div>
    </div>
    """


# --------------------------------------------------------------------------------------
# shared data assembly (the single reading spine uses this throughout)
# --------------------------------------------------------------------------------------
total = float(hero["total_pct"])
n_sig = int(hero["n_significant"])
n_eval = int(hero["n_skills"])
ci_lo = float(np.nansum([b["ci_lo"] for b in hero["breakdown"]]))
ci_hi = float(np.nansum([b["ci_hi"] for b in hero["breakdown"]]))
premium_pct = (np.exp(rec["salary_premium_log"]) - 1.0) * 100.0
pid = int(rec["persona_id"])
p_meta = PERSONA_META.get(pid, {})
share = float(p_meta.get("share", rec["persona_size"] / max(N_SAMPLE, 1)))
if cold:
    stand_sentence = ("This is the starting set — the five highest-ROI skills among developers "
                      "who share your profile. Add your stack in the Index to personalise the cover.")
else:
    stand_sentence = ("Modelled change in compensation if you learn these five skills, "
                      "holding experience and country constant.")
skills_line = " · ".join(rec_labels) if rec_labels else "—"
premium_text = "—" if rec["n_neighbours"] == 0 else f"{signed(premium_pct)}%"

# Which profile filters the peer group actually used. The recommender
# drops experience, then country, then role while too few developers match.
_FILTER_WORDS = {"role": "role", "country": "country", "experience": "experience (±10 years)"}


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


_pf = rec["peer_filter"]
if _pf["dropped"]:
    _kept = [_FILTER_WORDS[f] for f in _pf["applied"]]
    _basis = (f"your {_and(_kept)}" if _kept
              else "your stack alone, across the whole survey")
    peer_note = (
        f'<div class="sk-note">Fewer than {MIN_PEERS} developers in the survey share your '
        f'{_and([_FILTER_WORDS[f] for f in _pf["requested"]])}, so this comparison group is '
        f'matched on <b>{_basis}</b> instead ({_pf["n_pool"]:,} developers) — '
        f'{_and([_FILTER_WORDS[f] for f in _pf["dropped"]])} '
        f'{"is" if len(_pf["dropped"]) == 1 else "are"} not shaping these results.</div>'
    )
else:
    peer_note = ""


def _masthead_html() -> str:
    """Imprint, cover title, deck line, attributed byline."""
    return f"""
    <div class="sk-imprint">THE STACK OVERFLOW ANNUAL · 2025 · ISSUE 01</div>
    <h1 class="sk-cover-title">skillet<span class="sk-title-mark">.</span></h1>
    <p class="sk-deck-line">The dataset that decides what you should learn next.</p>
    <div class="sk-byline">By the <b>skillet editorial team</b> · Built on the 2025 Stack Overflow Developer Survey · <b>{N_SAMPLE:,}</b> working developers · <b>{len(roi)}</b> skills scored</div>
    """


def _standfirst_html() -> str:
    """The lead paragraph, in the publication's voice."""
    return """
    <p class="sk-standfirst">Every year, more than a hundred thousand professional developers tell
    Stack Overflow what they know, what they earn, and what they are reaching for next.
    skillet turns that record into an editorial decision: given your role, your stack,
    and your country, what should this engineer learn next? Following is the cover story —
    a per-skill salary coefficient, a peer-group computation, and five concrete
    recommendations, ranked.</p>
    """


def _covernote_html(first: bool = False) -> str:
    """The cover number: rust rules, the modelled lift, the skills byline,
    one stand sentence, and the stats row. `first` plays the one-shot rise."""
    cls = "sk-covernote sk-first" if first else "sk-covernote"
    return f"""
    <section class="{cls}">
      <div class="sk-covernum">
        <span class="sk-plus">+</span>{total:.0f}<span class="sk-unit">%</span>
      </div>
      <div class="sk-cover-skills"><b>↳</b> {skills_line}</div>
      <p class="sk-cover-stand">{stand_sentence}</p>
      <div class="sk-stats">
        <span>Skills in plan · <b>{n_eval}</b></span>
        <span>Significant · <b>{n_sig} of {n_eval}</b></span>
        <span>95% interval · <b>{signed(ci_lo, 0)}% → {signed(ci_hi, 0)}%</b></span>
        <span>Peer premium · <b>{premium_text}</b></span>
      </div>
      {peer_note}
    </section>
    """


def _persona_section_html() -> str:
    """The persona assignment as a full-width section: name, description
    and dateline on the left; defining-skill tags on the right."""
    tags = " ".join(
        f'<span class="sk-tag">{short(s)}</span>' for s in rec["persona_top_skills"]
    ) or '<span class="sk-tag">—</span>'
    top = ", ".join(short(s) for s in rec["persona_top_skills"][:3]) or "—"
    name = PERSONA_NAMES.get(pid, f"Cluster {pid}")
    return f"""
    <section class="sk-persona-sec">
      <h2>Who you are in the data.</h2>
      <div class="sk-persona-grid">
        <div>
          <div class="sk-persona-name">{name}</div>
          <p class="sk-persona-desc">Developers in this cluster reach for <b>{top}</b>;
          they number <b>{rec["persona_size"]:,}</b> in the processed sample. Your stack,
          role, experience and country place you with them — read the five entries above
          as this cluster's ranking, not a verdict on you.</p>
          <div class="sk-dateline">cluster {pid} · share of sample {share * 100:.1f}%</div>
        </div>
        <div class="sk-tags">{tags}</div>
      </div>
    </section>
    """


def _story_grid_html(enter: bool = False) -> str:
    """The cover story: five typeset page items in a two-up grid.

    Folio rank in italic display, skill name, evidence dek, a four-cell
    stat strip, and the 95% interval beneath. `enter=True` plays the
    one-shot staggered reveal (first render of the session only).
    """
    if not recs:
        return '<div class="sk-note">No candidate skills left — you have listed almost everything we score.</div>'
    parts = []
    for i, r in enumerate(recs, start=1):
        label = short(r["skill"])
        imp = float(r["salary_impact_pct"])
        p_exact = float(P_MAP.get(r["skill"], r["salary_p_value"]))
        ci = r.get("salary_ci_pct", [None, None])
        has_ci = ci[0] is not None and not (isinstance(ci[0], float) and np.isnan(ci[0]))
        ci_line = (f"95% interval {signed(ci[0])}% → {signed(ci[1])}%"
                   if has_ci else "95% interval unavailable")
        if have:
            want = f"{r['nbr_want_pct']:.0f}% want it"
        else:
            want = f"{r['nbr_prevalence_pct']:.0f}% know it"
        evidence = r["evidence"].replace(r["skill"], label)
        if not have:
            evidence = evidence.replace("Cold-start recommendation:", "Cold start —")
        parts.append(f"""
        <article class="sk-item">
          <div class="sk-folio">{i:02d}</div>
          <h3 class="sk-item-name">{label}</h3>
          <p class="sk-dek">{evidence}</p>
          <div class="sk-strip">
            <div><span class="k">Salary effect</span><span class="v {tone(imp)}">{signed(imp)}%</span></div>
            <div><span class="k">p-value</span><span class="v sk-small">p = {fmt_p_pub(p_exact)}</span></div>
            <div><span class="k">Survey users</span><span class="v sk-small">n = {r['n_known']:,}</span></div>
            <div><span class="k">Peer demand</span><span class="v sk-small">{want}</span></div>
          </div>
          <div class="sk-ciline">{ci_line}</div>
        </article>
        """)
    grid_cls = "sk-story-grid sk-enter" if enter else "sk-story-grid"
    head = '<div class="sk-story-head"><h2>The cover story.</h2></div>'
    return head + f'<div class="{grid_cls}">' + "".join(parts) + "</div>"


def _notes_html() -> str:
    """Methodology as Notes: a colophon page with the scoring formula,
    the numbered pipeline, the data source, and the limitations."""
    return f"""
    <section class="sk-notes-sec">
    <details class="sk-notes">
      <summary>How these recommendations are scored</summary>
      <div class="sk-notes-body">
        <p class="sk-notes-dek">For every skill you do not yet hold, the recommender
        computes a score from your peer group and the salary model, and returns the
        top five with the statistical evidence for each.</p>
        <pre>score(s)  =  peer_prevalence(s)                       # how common it is in your 300 nearest peers
           + 0.5 × salary_ROI(s)                      # per-skill MLR coefficient (log points)
           + 0.2 × peer_salary_premium                # how much your peer group earns overall
           + 0.3 × peer_demand(s)                     # share of peers who "want" to learn it
           + 0.05 if p(s) &lt; 0.05                      # significance bonus</pre>
        <p>Your peers must share your role, country and experience (±10 years).
        When fewer than {MIN_PEERS} developers match all three, experience is
        dropped first, then country, then role, and the cover says which
        filters were used.</p>
        <p>The pipeline behind the score, in order:</p>
        <ol>
          <li><b>Preprocess</b> (<code>src/preprocess.py</code>) — explode multi-select
          skill columns (languages, databases, cloud platforms, frameworks, dev
          environments, AI/LLM models), build a binary skill matrix, IQR-flag salary
          outliers, derive <code>log_comp</code> and a binary <code>JobSat_high</code>.</li>
          <li><b>Hypothesis tests</b> (<code>src/stats_tests.py</code>) — Welch's t-tests
          (AI users vs not, on salary and satisfaction) and chi-square tests of
          independence (AI usage vs remote work, role vs top language), with Cramér's V.</li>
          <li><b>Regression</b> (<code>src/regression.py</code>) — SLR (work exp →
          log compensation) and per-skill MLR
          (<code>log_comp ~ work_exp + country + skill</code>) yielding a skill-by-skill
          <b>ROI</b> in log points, converted to percent salary impact.</li>
          <li><b>Classification</b> (<code>src/classification.py</code>) — logistic
          regression of <code>JobSat_high</code> on experience, AI usage, and the top-30
          most common skills; ROC / AUC / Youden-optimal threshold and confusion matrix.</li>
          <li><b>Clustering</b> (<code>src/cluster.py</code>) — PCA on the standardized
          skill matrix → K-means personas chosen by silhouette score, hierarchical
          dendrogram for visualisation, persona profiles by prevalence lift.</li>
          <li><b>Recommender</b> (<code>src/recommend.py</code>) — assigns the reader to
          a persona, ranks the candidate skills they do not have by the score above,
          and returns the top five with the statistical evidence for each.</li>
        </ol>
        <p><b>Data source.</b> <b>skillet</b> maps the <b>2025 Stack Overflow Developer
        Survey</b> ({N_SAMPLE:,} professional developers in the processed dataset) onto
        a skill-vs-outcome space, then recommends the next skill that maximises salary
        uplift and job-satisfaction lift, conditioned on role, experience and country.</p>
        <p><b>Limitations.</b> Stack Overflow respondents over-represent English-speaking,
        internet-active developers; self-reported compensation is plausibly
        missing-not-at-random; coefficients are <b>associations</b>, not causal effects;
        single-year cross-section.</p>
      </div>
    </details>
    </section>
    """


# --- the reading spine ---------------------------------------------------------------
# One excellent view, well-paced. First-render entrances play once per
# session; later reruns emit the same markup without the enter classes.
_first = not st.session_state.get("skillet_entered", False)

st.html(_masthead_html())
st.html(_cover_illus_html(len(roi)))
st.html(_standfirst_html())
st.html(_covernote_html(first=_first))
st.html(_story_grid_html(enter=_first))
st.html(_persona_section_html())

st.session_state["skillet_entered"] = True

# --- notes ---------------------------------------------------------------------------
st.html(_notes_html())

# --- figure: skill ROI landscape -------------------------------------------------------
st.html('<div class="sk-figure-sec"><div class="sk-figlabel">FIGURE 1</div><h2>Skill ROI landscape.</h2></div>')
st.markdown(
    "<p class=\"sk-fig-lead\">Where the cover story comes from: the twenty skills "
    "with the largest modelled salary effects, each with its 95% interval. "
    "The ember-ringed bars are the five in your plan.</p>",
    unsafe_allow_html=True,
)

top = roi.head(20).copy()
top["skill_label"] = top["skill"].map(short)
top["neglogp"] = -np.log10(top["p_value"].clip(lower=1e-300))
plan = {r["skill"] for r in recs}

fig = px.bar(
    top,
    x="pct_salary_impact",
    y="skill_label",
    orientation="h",
    color="neglogp",
    color_continuous_scale=[
        (0.000, "#6e2d22"), (0.020, "#c44a1a"), (0.043, "#d8d0bf"),
        (0.140, "#7fa08a"), (0.400, "#2f5a3a"), (1.000, "#1e4028"),
    ],
    range_color=[0, 30],
    labels={"pct_salary_impact": "Modelled salary impact (%)", "skill_label": "",
            "neglogp": "−log₁₀(p)"},
    hover_data={"p_value": ":.2e", "users": ":,", "neglogp": False},
    error_x=top["ci_hi"] - top["pct_salary_impact"],
    error_x_minus=top["pct_salary_impact"] - top["ci_lo"],
)
fig.update_traces(
    error_x_color="#6b6e74",
    error_x_thickness=1.2,
    error_x_width=0,
    marker_line_width=[2 if s in plan else 0 for s in top["skill"]],
    marker_line_color="#c44a1a",
)
fig.update_layout(
    template="plotly_white",
    height=560,
    margin=dict(l=0, r=16, t=8, b=8),
    font=dict(family="Manrope, system-ui, sans-serif", size=13, color="#3d4046"),
    yaxis={"categoryorder": "total ascending", "title": "",
           "showgrid": False, "linecolor": "#d8d0bf", "automargin": True,
           "tickfont": dict(family="Newsreader, Georgia, serif", size=13, color="#0e0d0a")},
    xaxis={"title": dict(text="Modelled salary impact in percent — one MLR coefficient per skill.",
                         font=dict(family="JetBrains Mono, monospace", size=11, color="#3d4046")),
           "gridcolor": "#e8e1d2", "zeroline": True,
           "zerolinecolor": "#3d4046", "zerolinewidth": 1,
           "tickfont": dict(family="JetBrains Mono, monospace", size=11, color="#3d4046")},
    coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="#0e0d0a", font_color="#f3eee5",
                    font_family="JetBrains Mono, monospace"),
)
st.plotly_chart(fig, use_container_width=True, theme=None,
                config={"displayModeBar": False, "displaylogo": False})
st.html('<div class="sk-ramp"><span class="sk-ramp-lab">significance</span>'
        '<div class="sk-ramp-track"><i></i>'
        '<div class="sk-ramp-ticks"><span>p &gt; 0.05</span><span>1e−5</span>'
        '<span>1e−10</span><span>1e−20</span><span>1e−30</span></div>'
        '</div></div>')
st.markdown(
    "<p class=\"sk-fig-prose\">Read left to right, the landscape is the argument: "
    "a handful of infrastructure and data skills carry large, tightly-estimated "
    "premiums, while the long middle is a crowd of small effects the data cannot "
    "tell apart from noise. Your five sit where the bars are tall and the whiskers "
    "are short — that is the whole of the cover story, in one figure.</p>",
    unsafe_allow_html=True,
)

# --- calc: what-if as a secondary instrument -------------------------------------------
st.html('<div class="sk-calc-sec"><div class="sk-calc-head"><h2>What if I also learned these?</h2><span class="sk-marlabel">SECONDARY CALC</span></div></div>')

known = set(have) | {r["skill"] for r in recs}
extras = st.multiselect(
    "Pick additional skills you are considering",
    options=sorted([c for c in roi["skill"] if c not in known]),
    format_func=short, max_selections=8, key="skillet_extras",
    placeholder="Choose up to 8 skills…",
)

if extras:
    calc = lift_for(tuple(extras))
    names = ", ".join(short(s) for s in extras)
    st.html(
        f'<div class="sk-calc-read">'
        f'<div class="sk-calc-lift"><span class="sk-plus">+</span>{calc["total_pct"]:.1f}%</div>'
        f'<p class="sk-calc-interp">Adding <b>{names}</b> models at '
        f"<b>{calc['total_pct']:+.1f}%</b>, with "
        f"<b>{calc['n_significant']} of {calc['n_skills']}</b> coefficients clearing "
        f"p &lt; 0.05. Read it as an upper bound: the sum assumes each skill pays "
        f"independently, and real stacks overlap.</p>"
        f"</div>"
    )
    if calc["breakdown"]:
        bd = pd.DataFrame(calc["breakdown"])
        bd["skill_label"] = bd["skill"].map(short)
        fig2 = px.bar(
            bd, x="pct_impact", y="skill_label", orientation="h",
            color="p_value", color_continuous_scale=[
                (0.0, "#2f5a3a"), (0.5, "#d8d0bf"), (1.0, "#6e2d22")],
            range_color=[0, 0.1],
            title="Per-skill contribution.",
            error_x=bd["ci_hi"] - bd["pct_impact"],
            error_x_minus=bd["pct_impact"] - bd["ci_lo"],
            labels={"pct_impact": "salary impact (%)", "skill_label": "",
                    "p_value": "p-value"},
        )
        fig2.update_layout(
            yaxis={"categoryorder": "total ascending", "title": "", "automargin": True,
                   "tickfont": dict(family="Newsreader, Georgia, serif", size=12,
                                    color="#0e0d0a")},
            xaxis={"title": "", "gridcolor": "#e8e1d2", "automargin": True,
                   "zeroline": True, "zerolinecolor": "#3d4046", "zerolinewidth": 1,
                   "tickfont": dict(family="JetBrains Mono, monospace", size=11,
                                    color="#3d4046")},
            height=200 if len(bd) < 3 else 300,
            margin=dict(l=0, r=16, t=44, b=8),
            font=dict(family="Manrope, system-ui, sans-serif",
                      size=13, color="#3d4046"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            title=dict(x=0, xanchor="left", font=dict(
                family="Newsreader, Georgia, serif", size=15, color="#0e0d0a")),
            coloraxis_colorbar=dict(title="p-value", thickness=10, len=0.6,
                                    outlinewidth=0, tickformat=".0e",
                                    tickfont=dict(family="JetBrains Mono, monospace",
                                                  size=11, color="#3d4046")),
        )
        fig2.update_traces(error_x_color="#6b6e74", error_x_thickness=1.3, error_x_width=0)
        st.plotly_chart(fig2, use_container_width=True, theme=None,
                        config={"displayModeBar": False, "displaylogo": False})
        st.html('<div class="sk-cap">Per-skill contribution to the modelled lift, with 95% intervals.</div>')
else:
    st.html('<div class="sk-calc-quiet">Pick one or more skills above to see their combined predicted lift.</div>')

# --- footer: colophon line -------------------------------------------------------------
st.html(
    f"""
    <div class="sk-footer">
      <span>skillet · An annual on developer upskilling</span>
      <span>Newsreader · Manrope · JetBrains Mono</span>
      <span>2025 Stack Overflow Developer Survey · {N_SAMPLE:,} developers</span>
    </div>
    """)
