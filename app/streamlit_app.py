"""skillet — interactive Streamlit app.

Run with:  streamlit run app/streamlit_app.py
Requires that the pipeline has been run first (run_all.py) so that
data/processed.parquet and data/*.joblib exist.

Star Atlas visual world: skill space drawn as a night sky. The visitor's
stack is a constellation among all the skills, placed by how often they are
used together (data/skill_map.json, src/skymap.py); the five recommendations
are the bright stars nearest it, ringed in amber. The first session load
shows a short fold while the KNN index and salary model warm; every later
rerun is instant thanks to @st.cache_resource and @st.cache_data.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="skillet — what to learn next",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",  # no Streamlit sidebar: the rail is in the page
)

_CSS = ROOT / ".streamlit" / "style.css"
if _CSS.exists():
    st.markdown("<style>" + _CSS.read_text() + "</style>", unsafe_allow_html=True)

from src.recommend import MIN_PEERS, SkillRadar  # noqa: E402
from src.salary_calc import expected_uplift  # noqa: E402

# --------------------------------------------------------------------------------------
# palette (mirrors the tokens in .streamlit/style.css; Plotly needs literal colours)
# --------------------------------------------------------------------------------------
NAVY, PANEL, RULE, GRAT = "#0b1220", "#111a2d", "#25324d", "#18233a"
INK, INK_2, INK_3, DIM = "#e9edf5", "#b7c1d6", "#8593ad", "#4f5d7a"
AMBER, SKY, SAGE, ROSE = "#f0a65a", "#7fb3d5", "#9bd3a4", "#d9776b"
SERIF = "Spectral, Georgia, serif"
SANS = "'Hanken Grotesk', system-ui, sans-serif"

# --------------------------------------------------------------------------------------
# display labels
# --------------------------------------------------------------------------------------
PERSONA_NAMES = {
    0: ".NET stack",
    1: "Lean generalist stacks",
    2: "Modern JS / web tooling",
}

ALIASES = {
    "lang_Bash_Shell_all_shells": "Bash / Shell",
    "lang_HTML_CSS": "HTML / CSS",
    "lang_C_2": "C#",
    "lang_C_3": "C++",
    "lang_F": "F#",
    "lang_PHP": "PHP",
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


def fmt_p_html(p: float) -> str:
    """The same p-value for HTML, with a real superscript: the UI face lacks
    some Unicode superscript digits, which then mix fonts."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 1e-3:
        mantissa, exp = f"{p:.1e}".split("e")
        return f'{mantissa.rstrip("0").rstrip(".")}×10<sup>−{abs(int(exp))}</sup>'
    return f"{p:.3f}"


def signed(x, dec: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{dec}f}".replace("-", "−")


def tone(x: float) -> str:
    return "pos" if (x is not None and not np.isnan(x) and x >= 0) else "neg"


# st.html sanitises away inline SVG, so markers are CSS shapes on spans
RING = '<span class="sk-ring" aria-hidden="true"></span>'

# label slots tried around a star, in plotly textposition terms, with the
# direction (in map units) each one pushes the label
_SLOTS = [("middle right", (0.16, 0)), ("middle left", (-0.16, 0)), ("top center", (0, 0.07)),
          ("bottom center", (0, -0.07)), ("top right", (0.12, 0.06)), ("bottom right", (0.12, -0.06)),
          ("top left", (-0.12, 0.06)), ("bottom left", (-0.12, -0.06))]


def place_labels(pts: np.ndarray, avoid: np.ndarray, view=None) -> list[str]:
    """Greedy label placement: each label takes the slot farthest from the
    stars in `avoid` and from the labels already placed, and never a slot
    that would run off the plate (`view` = (cx, cy, half_width))."""
    taken: list[np.ndarray] = []
    out = []
    for p in pts:
        best, best_d = _SLOTS[0][0], -1.0
        for pos, (dx, dy) in _SLOTS:
            anchor = p + np.array([dx, dy])
            others = [a for a in avoid if not np.allclose(a, p)] + taken
            d = min((np.hypot(*(anchor - o)) for o in others), default=9.0)
            if view is not None:
                cx, cy, hw = view
                reach = p[0] + 2.2 * dx  # a label runs about twice its slot offset
                if abs(reach - cx) > 0.92 * hw or abs(anchor[1] - cy) > 0.9 * hw / 1.25:
                    d -= 10.0
            if d > best_d:
                best, best_d = pos, d
        out.append(best)
        taken.append(p + np.array(dict(_SLOTS)[best]))
    return out

# --------------------------------------------------------------------------------------
# cached loaders
# --------------------------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    df = pd.read_parquet(DATA / "processed.parquet")
    roi = pd.read_csv(DATA / "skill_roi.csv")
    meta = json.loads((DATA / "cluster_meta.json").read_text())
    sky = json.loads((DATA / "skill_map.json").read_text())["skills"]
    return df, roi, meta, sky


@st.cache_data
def cluster_breadth() -> dict[int, float]:
    """Average number of (cluster-space) skills per persona: cluster 1 is set
    apart mainly by smaller stacks, which its name has to say."""
    eng = get_engine()
    X = eng._data()[eng.cluster_skills].to_numpy()
    labels = np.asarray(eng._cluster["labels"])
    return {int(k): float(X[labels == k].sum(axis=1).mean()) for k in np.unique(labels)}


@st.cache_resource
def get_engine() -> SkillRadar:
    """One SkillRadar per process: the KNN index is built once."""
    return SkillRadar()


@st.cache_data
def lift_for(skills: tuple[str, ...]) -> dict:
    return expected_uplift(list(skills))


# --------------------------------------------------------------------------------------
# first-load fold: the wordmark and an amber hairline while the caches warm
# --------------------------------------------------------------------------------------
FOLD = """
<div class="sk-fold" role="status" aria-label="Loading skillet">
  <div class="sk-fold-word">skillet<span>.</span></div>
  <div class="sk-fold-line"><i></i></div>
  <div class="sk-fold-cap">Charting the survey</div>
</div>
"""

if not st.session_state.get("skillet_ready", False):
    st.html(FOLD)
    load_artifacts()
    get_engine()
    st.session_state["skillet_ready"] = True
    st.rerun()

# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
df, roi, meta, SKY_MAP = load_artifacts()
N_SAMPLE = int(meta.get("n_samples", len(df)))
PERSONA_META = {int(p["cluster"]): p for p in meta["personas"]}
P_MAP = dict(zip(roi["skill"], roi["p_value"]))
ROI_IDX = roi.set_index("skill")

SKILL_GROUPS = {
    "Languages":        [c for c in df.columns if c.startswith("lang_")],
    "Databases":        [c for c in df.columns if c.startswith("db_")],
    "Cloud platforms":  [c for c in df.columns if c.startswith("plat_")],
    "Web frameworks":   [c for c in df.columns if c.startswith("web_")],
    "Dev environments": [c for c in df.columns if c.startswith("devenv_")],
    "AI / LLM models":  [c for c in df.columns if c.startswith("aimodel_")],
}

# --------------------------------------------------------------------------------------
# left rail: profile and stack
# --------------------------------------------------------------------------------------
# The rail is an ordinary first column (sticky via CSS), not Streamlit's
# collapsible sidebar, so it is always on the page.
with st.container(key="sk_shell"):
    c_rail, c_main = st.columns([1, 3.4], gap="large")
with c_rail:
    with st.container(key="sk_rail"):
        st.html('<div class="sk-brand"><div class="sk-word">skillet<span>.</span></div>'
                f'<div class="sk-brand-sub">2025 Stack Overflow survey · {N_SAMPLE:,} developers</div></div>')
        st.html('<h2 class="sk-rail-h">Your profile</h2>')
        # Defaults are the most common answers, so a first visit lands on a
        # profile with a large peer group rather than the alphabetically first one.
        ROLE_OPTIONS = sorted([c for c in df.columns if c.startswith("role_")])
        devtype = st.selectbox(
            "Role", options=ROLE_OPTIONS, format_func=short,
            index=ROLE_OPTIONS.index(df[ROLE_OPTIONS].sum().idxmax()), key="sk_devtype",
        )
        workexp = st.slider("Years of professional experience", 0, 50, 5, key="sk_workexp")
        # Every country in the survey: the salary model has a fixed effect for
        # each, and the recommender relaxes the country filter when few match.
        COUNTRY_OPTIONS = sorted(df["Country"].dropna().unique())
        country = st.selectbox(
            "Country", options=COUNTRY_OPTIONS,
            index=COUNTRY_OPTIONS.index(df["Country"].value_counts().idxmax()), key="sk_country",
        )
        ai_user = st.checkbox("I use AI tools in my workflow", value=True, key="sk_ai")

        st.html('<h2 class="sk-rail-h">Your current stack</h2>')
        have: list[str] = []
        for group, cols in SKILL_GROUPS.items():
            if cols:
                have.extend(st.multiselect(group, options=cols, format_func=short,
                                           max_selections=6, key=f"sk_stack_{group}",
                                           placeholder="Choose skills…"))
        st.html(f'<div class="sk-rail-foot">{len(roi)} skills scored · salaries in USD</div>')


# --------------------------------------------------------------------------------------
# the recommendation
# --------------------------------------------------------------------------------------
engine = get_engine()
rec = engine.recommend(current_skills=list(have), top_n=5, country=country,
                       workexp=int(workexp), role=devtype, ai_user=bool(ai_user))
recs = rec["recommendations"]
plan = [r["skill"] for r in recs]
hero = lift_for(tuple(plan))
cold = not have

total = float(hero["total_pct"])
n_sig, n_eval = int(hero["n_significant"]), int(hero["n_skills"])
# 95% interval of the joint total (src/salary_calc.py), not a sum of intervals
ci_lo, ci_hi = float(hero["ci_lo_pct"]), float(hero["ci_hi_pct"])
premium_pct = (np.exp(rec["salary_premium_log"]) - 1.0) * 100.0
premium_text = f"{signed(premium_pct)}%"
_med = rec.get("group_median_usd", float("nan"))
median_text = "—" if np.isnan(_med) else f"${_med:,.0f}"

# Rank movement since the last rerun, so a profile change is visible even when
# the same five skills come back in a different order.
_prev = st.session_state.get("sk_prev_plan")
MOVES: dict[str, str] = {}
if _prev is not None and _prev != plan:
    for i, s_ in enumerate(plan):
        if s_ not in _prev:
            MOVES[s_] = "new"
        elif _prev.index(s_) != i:
            d = _prev.index(s_) - i
            MOVES[s_] = f"up {d}" if d > 0 else f"down {-d}"
st.session_state["sk_prev_plan"] = plan
pid = int(rec["persona_id"])
p_meta = PERSONA_META.get(pid, {})
share = float(p_meta.get("share", rec["persona_size"] / max(N_SAMPLE, 1)))

_FILTER_WORDS = {"role": "role", "country": "country",
                 "experience": "experience (±10 years)", "ai": "AI-tool use"}


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


_pf = rec["peer_filter"]
_lo, _hi = max(0, int(workexp) - 10), int(workexp) + 10
_parts = {"role": short(devtype).lower(), "country": country,
          "experience": f"{_lo}–{_hi} years", "ai": "AI users" if ai_user else "not using AI tools"}
_group = " · ".join(_parts[f] for f in ("role", "country", "experience", "ai") if f in _pf["applied"])
_grp = (f'<b>{_pf["n_pool"]:,} developers</b> who share your profile'
        + (f' ({html.escape(_group)})' if _group else ' (the whole survey)'))
if cold:
    lead = (f"Comparing you with {_grp}. Add your current stack to draw your constellation; "
            "until then these are the highest-ROI skills in that group.")
else:
    lead = (f"Your {rec['n_neighbours']} nearest peers in skill space, drawn from {_grp}.")
peer_note = ""
if _pf["dropped"]:
    _kept = [_FILTER_WORDS[f] for f in _pf["applied"]]
    _basis = f"your {_and(_kept)}" if _kept else "your stack alone, across the whole survey"
    peer_note = (
        f'<p class="sk-relax">Fewer than {MIN_PEERS} developers share your '
        f'{_and([_FILTER_WORDS[f] for f in _pf["requested"]])}, so peers are matched on '
        f'<b>{_basis}</b> ({_pf["n_pool"]:,} developers); '
        f'{_and([_FILTER_WORDS[f] for f in _pf["dropped"]])} '
        f'{"is" if len(_pf["dropped"]) == 1 else "are"} not shaping these results.</p>'
    )

# --------------------------------------------------------------------------------------
# the sky chart (signature): every skill a star, your stack a constellation
# --------------------------------------------------------------------------------------
def sky_chart() -> go.Figure:
    names = [s for s in SKY_MAP]
    xy = np.array([SKY_MAP[s] for s in names])
    eff = np.array([ROI_IDX["pct_salary_impact"].get(s, np.nan) for s in names])
    pvals = np.array([ROI_IDX["p_value"].get(s, np.nan) for s in names])
    users = np.array([ROI_IDX["users"].get(s, 0) for s in names])
    # magnitude ramp: dot diameter carries the salary effect
    size = 3.0 + 13.0 * np.clip(np.nan_to_num(eff, nan=0.0), 0, 30) / 30.0
    hover = np.array([[short(s), f"{signed(e)}%" if not np.isnan(e) else "not estimated",
                       fmt_p_pub(p), f"{int(u):,}"] for s, e, p, u in zip(names, eff, pvals, users)])
    idx = {s: i for i, s in enumerate(names)}
    mine = [s for s in have if s in idx]
    picks = [s for s in plan if s in idx]
    rest = [i for i, s in enumerate(names) if s not in set(mine) | set(picks)]

    # frame the plate around your constellation and the five picks, like an
    # atlas plate cropped around its brightest objects
    focus = xy[[idx[s] for s in (mine + picks)]] if (mine or picks) else xy
    lo, hi = focus.min(axis=0), focus.max(axis=0)
    cx, cy = (lo + hi) / 2
    half = max(float((hi - lo).max()) * 0.75, 0.32)
    fig = go.Figure()
    # graticule: concentric rings and radial meridians around the plate centre
    for k in (1 / 3, 2 / 3, 1.0, 4 / 3):
        t = np.linspace(0, 2 * np.pi, 181)
        fig.add_trace(go.Scatter(x=cx + k * half * np.cos(t), y=cy + k * half * np.sin(t), mode="lines",
                                 line=dict(color=GRAT, width=1), hoverinfo="skip", showlegend=False))
    for a in np.arange(0, np.pi, np.pi / 6):
        r = 1.5 * half
        fig.add_trace(go.Scatter(x=[cx - r * np.cos(a), cx + r * np.cos(a)],
                                 y=[cy - r * np.sin(a), cy + r * np.sin(a)], mode="lines",
                                 line=dict(color=GRAT, width=1), hoverinfo="skip", showlegend=False))
    tmpl = ("<b>%{customdata[0]}</b><br>salary effect %{customdata[1]}"
            "<br>p = %{customdata[2]} · n = %{customdata[3]}<extra></extra>")
    # every other skill: dim when its salary effect is negative
    fig.add_trace(go.Scatter(
        x=xy[rest, 0], y=xy[rest, 1], mode="markers",
        marker=dict(size=size[rest], color=[INK_3 if (e > 0) else DIM for e in np.nan_to_num(eff[rest])],
                    line=dict(width=0)),
        customdata=hover[rest], hovertemplate=tmpl, showlegend=False))
    # your constellation: minimum spanning tree over your skills
    if len(mine) >= 2:
        pts = xy[[idx[s] for s in mine]]
        mst = minimum_spanning_tree(cdist(pts, pts)).tocoo()
        lx, ly = [], []
        for i, j in zip(mst.row, mst.col):
            lx += [pts[i, 0], pts[j, 0], None]
            ly += [pts[i, 1], pts[j, 1], None]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=INK_2, width=1.2),
                                 opacity=0.75, hoverinfo="skip", showlegend=False))
    anchors = xy[[idx[s] for s in (picks + mine)]] if (mine or picks) else np.empty((0, 2))
    # name the brightest other stars on the plate, faintly, for orientation
    in_view = [i for i in rest if abs(xy[i, 0] - cx) < half and abs(xy[i, 1] - cy) < half
               and not np.isnan(eff[i]) and eff[i] > 0]
    # keep a name only if it clears every other label and pick by a margin
    # sized for a phone-width plate (about 340 px across 2.5 half-widths)
    unit = 2.5 * half / 340.0
    def _clear(i, kept):
        w = (len(short(names[i])) * 6.5 + 12) * unit
        for j in kept + [idx[s] for s in picks + mine]:
            if j == i:
                continue
            wj = (len(short(names[j])) * 6.5 + 12) * unit
            if abs(xy[i, 0] - xy[j, 0]) < (w + wj) / 2 and abs(xy[i, 1] - xy[j, 1]) < 34 * unit:
                return False
        return True
    named: list[int] = []
    for i in sorted(in_view, key=lambda i: -eff[i]):
        if len(named) == 5:
            break
        if _clear(i, named):
            named.append(i)
    if named:
        fig.add_trace(go.Scatter(
            x=xy[named, 0], y=xy[named, 1], mode="markers+text", text=[short(names[i]) for i in named],
            marker=dict(size=size[named] + 4, color="rgba(0,0,0,0)"),
            textposition=place_labels(xy[named], np.vstack([anchors, xy[named]]), (cx, cy, 1.25 * half)),
            textfont=dict(family=SERIF, size=12, color=INK_3), hoverinfo="skip", showlegend=False))
    if mine:
        mi = [idx[s] for s in mine]
        fig.add_trace(go.Scatter(
            x=xy[mi, 0], y=xy[mi, 1], mode="markers+text", text=[short(s) for s in mine],
            textposition=place_labels(xy[mi], anchors, (cx, cy, 1.25 * half)), textfont=dict(family=SANS, size=11, color=INK_2),
            marker=dict(size=9, color=INK, line=dict(width=0)),
            customdata=hover[mi], hovertemplate=tmpl, showlegend=False))
    if picks:
        pi = [idx[s] for s in picks]
        fig.add_trace(go.Scatter(x=xy[pi, 0], y=xy[pi, 1], mode="markers", hoverinfo="skip",
                                 marker=dict(size=30, color="rgba(0,0,0,0)",
                                             line=dict(width=1, color=AMBER)),
                                 opacity=0.55, showlegend=False))
        fig.add_trace(go.Scatter(
            x=xy[pi, 0], y=xy[pi, 1], mode="markers+text", text=[short(s) for s in picks],
            textposition=place_labels(xy[pi], anchors, (cx, cy, 1.25 * half)),
            textfont=dict(family=SERIF, size=16, color=INK, style="italic"),
            marker=dict(size=14, color=INK, line=dict(width=2.5, color=AMBER)),
            customdata=hover[pi], hovertemplate=tmpl, showlegend=False))
    ticks = []
    for k in range(41):
        t = k / 40
        L = 0.022 if k % 5 == 0 else 0.011
        ticks += [dict(type="line", xref="paper", yref="paper", x0=t, x1=t, y0=0, y1=L),
                  dict(type="line", xref="paper", yref="paper", x0=t, x1=t, y0=1 - L, y1=1),
                  dict(type="line", xref="paper", yref="paper", x0=0, x1=L, y0=t, y1=t),
                  dict(type="line", xref="paper", yref="paper", x0=1 - L, x1=1, y0=t, y1=t)]
    for d in ticks:
        d["line"] = dict(color=RULE, width=1)
    ticks.append(dict(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=0, y1=1,
                      line=dict(color=RULE, width=1)))
    fig.update_layout(
        shapes=ticks,
        height=640, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[cx - 1.25 * half, cx + 1.25 * half], fixedrange=True),
        yaxis=dict(visible=False, range=[cy - half, cy + half], fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        font=dict(family=SANS, color=INK_2), dragmode=False,
        hoverlabel=dict(bgcolor=PANEL, bordercolor=RULE, font=dict(family=SANS, size=13, color=INK)),
    )
    return fig


def plan_html(first: bool) -> str:
    """The plan panel: the joint lift, its interval, the five ranked stars."""
    sign = "+" if total >= 0 else "−"
    rows = []
    for i, r in enumerate(recs, start=1):
        share_word = "of developers who share your profile know it" if cold else "of your peers know it"
        move = f'<span class="sk-move">{MOVES[r["skill"]]}</span>' if r["skill"] in MOVES else ""
        rows.append(
            f'<li class="sk-plan-row">{RING}<div class="sk-plan-name">{short(r["skill"])}{move}'
            f'<small>{r["nbr_prevalence_pct"]:.0f}% {share_word} · p = '
            f'{fmt_p_html(P_MAP.get(r["skill"], r["salary_p_value"]))}</small></div>'
            f'<span class="sk-eff {tone(r["salary_impact_pct"])}">{signed(r["salary_impact_pct"])}%</span></li>')
    if not recs:
        rows.append('<li class="sk-plan-empty">No candidate skills left: your stack already '
                    'covers nearly everything skillet scores.</li>')
    cls = "sk-plan sk-first" if first else "sk-plan"
    return f"""
    <section class="{cls}" aria-labelledby="sk-plan-h">
      <h2 id="sk-plan-h">All five, estimated together</h2>
      <div class="sk-total"><span class="sk-sign">{sign}</span>{abs(total):.0f}<span class="sk-unit">%</span></div>
      <p class="sk-ci">95% interval {signed(ci_lo, 0)}% → {signed(ci_hi, 0)}% · {n_sig} of {n_eval} significant</p>
      <ol class="sk-plan-list">{''.join(rows)}</ol>
      <dl class="sk-facts">
        <div><dt>Median pay, your group</dt><dd>{median_text}</dd></div>
        <div><dt>Premium vs same country and experience</dt><dd>{premium_text}</dd></div>
        <div><dt>Persona (from your stack)</dt><dd>{"add your stack" if cold else PERSONA_NAMES.get(pid, f"Cluster {pid}")}</dd></div>
      </dl>
    </section>"""


def index_html() -> str:
    """Index of the five: the full evidence for each recommendation."""
    if not recs:
        return ""
    body = []
    for i, r in enumerate(recs, start=1):
        ci = r.get("salary_ci_pct", [np.nan, np.nan])
        if cold:
            evidence = f'{r["nbr_prevalence_pct"]:.0f}% of developers who share your profile know it.'
        else:
            evidence = (f'Weighted by similarity, {r["nbr_prevalence_pct"]:.0f}% of your '
                        f'{rec["n_neighbours"]} nearest peers know it and {r["nbr_want_pct"]:.0f}% '
                        f'want to learn it.')
        body.append(f"""
        <tr>
          <td class="sk-ix-rank">{RING}<span>{i}</span></td>
          <th scope="row"><span class="sk-ix-name">{short(r["skill"])}</span><span class="sk-ix-ev">{html.escape(evidence)}</span></th>
          <td class="sk-eff {tone(r["salary_impact_pct"])}">{signed(r["salary_impact_pct"])}%</td>
          <td class="sk-nowrap">{signed(ci[0])}% → {signed(ci[1])}%</td>
          <td class="sk-nowrap">{fmt_p_html(P_MAP.get(r["skill"], r["salary_p_value"]))}</td>
          <td>{r["n_known"]:,}</td>
          <td>{r["nbr_prevalence_pct"]:.0f}%</td>
          <td>{"—" if cold else f'{r["nbr_want_pct"]:.0f}%'}</td>
        </tr>""")
    return f"""
    <section class="sk-index" aria-labelledby="sk-index-h">
      <h2 id="sk-index-h">The five, in detail</h2>
      <p class="sk-lead">Each salary effect here is that skill on its own (one regression per skill,
      holding experience and country fixed). The plan total estimates the five together, so it is
      usually smaller than their sum: skills that come as a set share one premium.</p>
      <div class="sk-table-wrap"><table>
        <thead><tr><th scope="col">#</th><th scope="col">Skill and evidence</th><th scope="col">Salary effect</th>
        <th scope="col">95% interval</th><th scope="col">p-value</th><th scope="col">Survey users</th>
        <th scope="col">{"Profile matches know" if cold else "Peers know"}</th><th scope="col">Peers want</th></tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table></div>
    </section>"""


def persona_html() -> str:
    if cold:
        return ("""
    <section class="sk-persona" aria-labelledby="sk-persona-h">
      <div>
        <h2 id="sk-persona-h">Who you are in the data</h2>
        <p class="sk-lead">Personas are K-means clusters of <b>skills</b>, so they come from your stack
        alone: role, experience, country and AI use do not move them. Add your current stack to see which
        of the three clusters it falls in.</p>
      </div>
      <ul class="sk-tags" aria-label="The three personas"><li>.NET stack</li><li>Lean generalist stacks</li>
      <li>Modern JS / web tooling</li></ul>
    </section>""")
    tags = "".join(f'<li>{short(s)}</li>' for s in rec["persona_top_skills"]) or "<li>—</li>"
    top = ", ".join(short(s) for s in rec["persona_top_skills"][:3]) or "—"
    return f"""
    <section class="sk-persona" aria-labelledby="sk-persona-h">
      <div>
        <h2 id="sk-persona-h">Who you are in the data</h2>
        <p class="sk-persona-name">{PERSONA_NAMES.get(pid, f"Cluster {pid}")}</p>
        <p class="sk-lead">The K-means model places your stack in this cluster of
        <b>{rec["persona_size"]:,}</b> developers ({share * 100:.1f}% of the sample). Their stacks list
        <b>{cluster_breadth().get(pid, float("nan")):.0f} common skills</b> on average, and the skills that
        set the cluster apart most are {top}. Personas come from your stack alone, so experience, country and
        AI use do not change them. A persona is a vocabulary for the data, not a verdict on you.</p>
      </div>
      <ul class="sk-tags" aria-label="Defining skills">{tags}</ul>
    </section>"""


# --------------------------------------------------------------------------------------
# first viewport
# --------------------------------------------------------------------------------------
with c_main:
    _first = not st.session_state.get("skillet_entered", False)
    st.html(f'<header class="sk-top"><h1>Your stack, <em>and the skills nearest it.</em></h1>'
            f'<p class="sk-lead">{lead}</p>{peer_note}</header>')
    with st.container(key="sk_sky_row"):
        c_sky, c_plan = st.columns([1.7, 1], gap="large")
        with c_sky, st.container(key="sk_plate"):
            st.html(f'<div class="sk-plate-cap"><span><em>Plate I.</em> Skill space, 2025</span>'
                    f'<span>{len(SKY_MAP)} skills, placed by how often developers use them together</span></div>')
            st.plotly_chart(sky_chart(), width="stretch", theme=None,
                            config={"displayModeBar": False, "displaylogo": False, "scrollZoom": False})
            st.html('<div class="sk-legend">'
                    '<span><i class="sk-dot sk-dot-mine" aria-hidden="true"></i>your skill</span>'
                    '<span><i class="sk-dot sk-dot-pick" aria-hidden="true"></i>recommended</span>'
                    '<span class="sk-mag">salary effect <i style="width:5px;height:5px"></i>+5% '
                    '<i style="width:10px;height:10px"></i>+15% <i style="width:16px;height:16px"></i>+30%</span></div>')
        with c_plan:
            st.html(plan_html(_first))
    st.session_state["skillet_entered"] = True

    # --------------------------------------------------------------------------------------
    # below the fold
    # --------------------------------------------------------------------------------------
    if recs:
        st.html(index_html())
    st.html(persona_html())

    # --- figure: skill ROI landscape -------------------------------------------------------
    top20 = roi.head(20).copy()
    top20["label"] = top20["skill"].map(short)
    in_plan = [s in set(plan) for s in top20["skill"]]
    st.html('<section class="sk-fig"><h2>Skill ROI landscape</h2><p class="sk-lead">The twenty skills with '
            'the largest modelled salary effects. Each star is the estimate and its line the 95% interval; '
            'brighter stars are more significant. '
            f'{"Amber-ringed stars are in your plan." if any(in_plan) else "None of your five ranks in this top twenty; their effects are in the table above."}'
            '</p></section>')
    neglogp = -np.log10(top20["p_value"].clip(lower=1e-300))
    fig = go.Figure()
    # whiskers: the 95% interval, drawn as one hairline per skill
    wx, wy = [], []
    for lo_, hi_, lab in zip(top20["ci_lo"], top20["ci_hi"], top20["label"]):
        wx += [lo_, hi_, None]
        wy += [lab, lab, None]
    fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines", line=dict(color=INK_3, width=1.2),
                             hoverinfo="skip", showlegend=False))
    # the estimate as a star: brighter the more significant; amber ring = in your plan
    fig.add_trace(go.Scatter(
        x=top20["pct_salary_impact"], y=top20["label"], mode="markers",
        marker=dict(size=11, color=neglogp, cmin=0, cmax=100,
                    colorscale=[(0, "#41678f"), (0.5, "#7fa7cb"), (1, INK)],
                    line=dict(width=[2.5 if p else 0 for p in in_plan], color=AMBER)),
        customdata=np.stack([top20["p_value"].map(fmt_p_pub), top20["users"].map("{:,}".format),
                             top20["ci_lo"].map("{:+.1f}".format), top20["ci_hi"].map("{:+.1f}".format)], axis=1),
        hovertemplate=("<b>%{y}</b><br>%{x:+.1f}% (95% interval %{customdata[2]}% to %{customdata[3]}%)"
                       "<br>p = %{customdata[0]} · n = %{customdata[1]}<extra></extra>"),
        showlegend=False))
    fig.update_layout(
        height=640, margin=dict(l=16, r=12, t=4, b=52),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, size=13, color=INK_2),
        yaxis=dict(categoryorder="array", categoryarray=list(top20["label"][::-1]), showgrid=True,
                   gridcolor=GRAT, automargin=True, fixedrange=True,
                   tickfont=dict(family=SERIF, size=15, color=INK)),
        xaxis=dict(title=dict(text="Salary effect, % (one regression per skill), with 95% interval",
                              font=dict(family=SERIF, size=13, color=INK_3)),
                   gridcolor=GRAT, zeroline=True, zerolinecolor=RULE, ticksuffix="%", fixedrange=True,
                   automargin=True, tickfont=dict(size=12, color=INK_3)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=RULE, font=dict(family=SANS, size=13, color=INK)),
    )
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False, "displaylogo": False})
    st.html('<div class="sk-ramp" aria-hidden="true"><span>significance</span><div class="sk-ramp-track">'
            '<i></i><div class="sk-ramp-ticks"><span>p ≈ 1</span><span>10⁻²⁵</span><span>10⁻⁵⁰</span>'
            '<span>10⁻⁷⁵</span><span>≤ 10⁻¹⁰⁰</span></div></div></div>')

    # --- calc: what if I also learned these? ----------------------------------------------
    st.html('<section class="sk-calc"><h2>What if I also learned these?</h2><p class="sk-lead">Pick up to '
            'eight more skills to see their combined modelled effect, estimated together.</p></section>')
    known = set(have) | set(plan)
    extras = st.multiselect(
        "Skills you are considering", options=sorted([c for c in roi["skill"] if c not in known]),
        format_func=short, max_selections=8, key="skillet_extras", placeholder="Choose up to 8 skills…",
    )
    if extras:
        calc = lift_for(tuple(extras))
        names = ", ".join(short(s) for s in extras)
        st.html(
            f'<div class="sk-calc-read"><div class="sk-calc-lift">'
            f'<span class="sk-sign">{"+" if calc["total_pct"] >= 0 else "−"}</span>{abs(calc["total_pct"]):.1f}%</div>'
            f'<p>Adding <b>{names}</b> models at <b>{signed(calc["total_pct"])}%</b> '
            f'(95% interval {signed(calc["ci_lo_pct"])}% → {signed(calc["ci_hi_pct"])}%), with '
            f'<b>{calc["n_significant"]} of {calc["n_skills"]}</b> coefficients clearing p &lt; 0.05. '
            f'Overlap between the skills is not double-counted; it is still an association, not a promise.</p></div>')
        if calc["breakdown"]:
            bd = pd.DataFrame(calc["breakdown"])
            bd["label"] = bd["skill"].map(short)
            fig2 = go.Figure()
            wx, wy = [], []
            for lo_, hi_, lab in zip(bd["ci_lo"], bd["ci_hi"], bd["label"]):
                wx += [lo_, hi_, None]
                wy += [lab, lab, None]
            fig2.add_trace(go.Scatter(x=wx, y=wy, mode="lines", line=dict(color=INK_3, width=1.2),
                                      hoverinfo="skip", showlegend=False))
            fig2.add_trace(go.Scatter(
                x=bd["pct_impact"], y=bd["label"], mode="markers",
                marker=dict(size=11, color=[SAGE if v >= 0 else ROSE for v in bd["pct_impact"]]),
                hovertemplate="<b>%{y}</b><br>%{x:+.1f}% jointly<extra></extra>", showlegend=False))
            fig2.update_layout(
                height=max(180, 56 * len(bd) + 70), margin=dict(l=16, r=12, t=4, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=SANS, size=13, color=INK_2),
                yaxis=dict(categoryorder="array", categoryarray=list(bd["label"][::-1]), automargin=True,
                           fixedrange=True, showgrid=True, gridcolor=GRAT,
                           tickfont=dict(family=SERIF, size=15, color=INK)),
                xaxis=dict(gridcolor=GRAT, zeroline=True, zerolinecolor=RULE, ticksuffix="%",
                           fixedrange=True, automargin=True, tickfont=dict(size=12, color=INK_3)),
                hoverlabel=dict(bgcolor=PANEL, bordercolor=RULE, font=dict(family=SANS, size=13, color=INK)),
            )
            st.plotly_chart(fig2, width="stretch", theme=None, config={"displayModeBar": False, "displaylogo": False})
            st.html('<p class="sk-cap">Each skill\'s contribution when the chosen skills are estimated together, with 95% intervals.</p>')
    else:
        st.html('<p class="sk-quiet">Nothing chosen yet. The combined effect appears here.</p>')

    # --- method notes ---------------------------------------------------------------------
    st.html(f"""
    <section class="sk-notes-sec">
    <details class="sk-notes">
      <summary>How these recommendations are scored</summary>
      <div class="sk-notes-body">
        <p>For every skill you do not yet hold, the recommender computes a score from your peer group
        and the salary model, and returns the top five with the statistical evidence for each.</p>
        <pre>score(s) =  peer_prevalence(s)        similarity-weighted share of your 300 nearest peers who know s
              + 0.5 × salary_ROI(s)       per-skill regression coefficient, log points
              + 0.3 × peer_demand(s)      similarity-weighted share of peers who want to learn s
              + 0.02 if you use AI tools and s is an AI model
              + 0.05 if p(s) &lt; 0.05</pre>
        <p>Your peers share your role, country, experience (±10 years) and AI-tool use. When fewer than
        {MIN_PEERS} developers match, AI-tool use is dropped first, then experience, then country, then role,
        and the page says which filters were used.</p>
        <p>The headline number fits one regression with all five skills together, plus experience and a
        fixed effect for every country, so skills that usually come as a set do not each claim the same
        premium; its 95% interval comes from that model. The peer premium is how much more your peers earn
        than developers with the same experience in the same country; it is shown for context and is not
        part of the score.</p>
        <p>The sky chart places every skill by how often developers use it together with every other
        skill (t-SNE on 1 − Jaccard similarity); star size is the salary effect.</p>
        <ol>
          <li><b>Preprocess</b> (<code>src/preprocess.py</code>): binary skill matrix, USD pay, IQR outlier flag, <code>JobSat_high</code>.</li>
          <li><b>Hypothesis tests</b> (<code>src/stats_tests.py</code>): Welch's t-tests, the AI pay gap holding experience and country fixed, chi-square tests with Cramér's V.</li>
          <li><b>Regression</b> (<code>src/regression.py</code>): <code>log_comp ~ work_exp + country + skill</code> for each skill, giving its salary ROI.</li>
          <li><b>Classification</b> (<code>src/classification.py</code>): logistic, decision tree and Naive Bayes models of high job satisfaction.</li>
          <li><b>Clustering</b> (<code>src/cluster.py</code>): PCA and K-means personas.</li>
          <li><b>Recommender</b> (<code>src/recommend.py</code>), validated on held-out respondents in <code>src/evaluate.py</code>.</li>
        </ol>
        <p><b>Data.</b> The 2025 Stack Overflow Developer Survey, {N_SAMPLE:,} working professional developers.
        <b>Limitations.</b> Respondents over-represent English-speaking, internet-active developers;
        pay is self-reported; coefficients are associations, not causal effects; one survey year.</p>
      </div>
    </details>
    </section>
    <footer class="sk-footer"><span>skillet<b>.</b></span><span>2025 Stack Overflow Developer Survey · {N_SAMPLE:,} developers · {len(roi)} skills</span></footer>
    """)
