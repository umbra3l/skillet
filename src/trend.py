"""Module 6: Multi-year StackOverflow skill salary-ROI trend analyser.

For each StackOverflow Developer Survey year (2022, 2023, 2024, 2025) this
module:

  1. downloads ``results.csv`` + ``schema.csv`` into ``data/archive_{year}/``
     (gitignored) from the StackExchange LFS media host;
  2. harmonises the year-specific column names (compensation, work
     experience, country, and the ``*HaveWorkedWith`` matrix questions);
  3. filters to professional developers currently in work using the same
     ``MainBranch`` + ``Employment`` heuristic as ``src/preprocess.py``;
  4. fits a per-skill multiple linear regression, exactly like
     ``src/regression.py::per_skill_roi``:

         log1p(comp) ~ WorkExp + top-10 country dummies + skill_binary

     and records the skill coefficient, p-value, user count and the
     approximate percentage salary impact ``(exp(coef) - 1) * 100``;
  5. writes a long-format ``data/trend_roi.csv`` and a summary JSON
     ``data/trend_summary.json``;
  6. renders a line chart of salary impact vs. year for the widely tracked
     skills to ``assets/trend_roi.png``.

Running standalone::

    .venv/bin/python -m src.trend
"""
from __future__ import annotations

import json
import re
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSETS = ROOT / "assets"
REPORT_DIR = ROOT / "report"

OUT_CSV = DATA_DIR / "trend_roi.csv"
OUT_JSON = DATA_DIR / "trend_summary.json"
OUT_PNG = ASSETS / "trend_roi.png"
OUT_README = REPORT_DIR / "TREND_README.md"

YEARS = (2022, 2023, 2024, 2025)
LFS_BASE = ("https://media.githubusercontent.com/media/StackExchange/Survey/"
            "main/packages/archive/{year}/{name}")

SLUG = re.compile(r"[^A-Za-z0-9]+")


def slugify(t: str) -> str:
    """Same slug rule as ``src/preprocess.py`` (used for reporting only)."""
    return SLUG.sub("_", t).strip("_")


# ---------------------------------------------------------------------------
# Column-name discovery (names drifted between survey years)
# ---------------------------------------------------------------------------
# First present candidate wins.  All matrix questions use the
# "HaveWorkedWith"-style column only (WantToWorkWith / Admired / HaveEntry are
# deliberately ignored).
COMP_CANDIDATES = ("ConvertedCompYearly", "CompTotal")
WORKEXP_CANDIDATES = ("WorkExp",)
COUNTRY_CANDIDATES = ("Country",)

MATRIX_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "lang": ("LanguageHaveWorkedWith", "LanguageWorkedWith"),
    "db": ("DatabaseHaveWorkedWith", "DatabaseWorkedWith"),
    "plat": ("PlatformHaveWorkedWith", "PlatformWorkedWith"),
    "web": ("WebframeHaveWorkedWith", "WebframeWorkedWith"),
    "devenv": ("DevEnvsHaveWorkedWith", "NEWCollabToolsHaveWorkedWith",
               "NEWCollabToolsWorkedWith"),
    "misc": ("MiscTechHaveWorkedWith", "MiscTechWorkedWith"),
    "tools": ("ToolsTechHaveWorkedWith", "ToolsTechWorkedWith"),
    "os": ("OpSysProfessional use", "OpSysProfessional",
           "OpSysPersonal use", "OpSysPersonal"),
    "vcs": ("VersionControlSystem", "VersionControl"),
    "ai": ("AIModelsHaveWorkedWith", "AISearchDevHaveWorkedWith",
           "AISearchHaveWorkedWith", "AIDevHaveWorkedWith"),
}

# Core skill set: display name -> ordered raw-token aliases (the token names
# also drifted slightly between years, e.g. IntelliJ -> IntelliJ IDEA).
CORE_SKILLS: dict[str, list[str]] = {
    "Python": ["Python"],
    "JavaScript": ["JavaScript"],
    "TypeScript": ["TypeScript"],
    "Java": ["Java"],
    "C++": ["C++"],
    "C#": ["C#"],
    "Go": ["Go"],
    "Rust": ["Rust"],
    "Ruby": ["Ruby"],
    "PHP": ["PHP"],
    "SQL": ["SQL"],
    "HTML/CSS": ["HTML/CSS"],
    "React": ["React", "React.js"],
    "Angular": ["Angular", "Angular.js"],
    "Vue.js": ["Vue.js", "Vue"],
    "Node.js": ["Node.js"],
    "Django": ["Django"],
    "Flask": ["Flask"],
    "Spring": ["Spring Boot", "Spring"],
    "Docker": ["Docker"],
    "Kubernetes": ["Kubernetes"],
    "AWS": ["Amazon Web Services (AWS)", "AWS"],
    "Azure": ["Microsoft Azure", "Azure"],
    "GCP": ["Google Cloud", "Google Cloud Platform"],
    "PostgreSQL": ["PostgreSQL"],
    "MySQL": ["MySQL"],
    "MongoDB": ["MongoDB"],
    "Redis": ["Redis"],
    "Terraform": ["Terraform"],
    "Git": ["Git"],
    "Linux": ["Linux", "Linux-based"],
    "VS Code": ["Visual Studio Code", "Visual Studio Code (VS Code)", "VS Code"],
    "IntelliJ": ["IntelliJ IDEA", "IntelliJ"],
    "Jupyter": ["Jupyter Notebook/JupyterLab", "Jupyter Notebook",
                "JupyterLab", "Jupyter", "IPython/Jupyter"],
}

MIN_SKILL_USERS = 100  # skip a skill-year with fewer users than this


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with coarse progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "skill-radar/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        last_pct = -10
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                pct = int(got * 100 / total)
                if pct >= last_pct + 10:
                    last_pct = pct
                    print(f"    {dest.name}: {pct:3d}%  "
                          f"({got // (1 << 20)}/{total // (1 << 20)} MB)")
    tmp.replace(dest)


def download_years(years: tuple[int, ...] = YEARS, min_bytes: int = 1_000_000) -> dict[int, Path]:
    """Ensure ``data/archive_{year}/results.csv`` exists for every year."""
    paths: dict[int, Path] = {}
    for year in years:
        out_dir = DATA_DIR / f"archive_{year}"
        results = out_dir / "results.csv"
        schema = out_dir / "schema.csv"
        paths[year] = results
        if results.exists() and results.stat().st_size > min_bytes:
            print(f"  {year}: results.csv already present "
                  f"({results.stat().st_size // (1 << 20)} MB)")
        else:
            print(f"  {year}: downloading results.csv ...")
            _download(LFS_BASE.format(year=year, name="results.csv"), results)
        if not schema.exists():
            try:
                _download(LFS_BASE.format(year=year, name="schema.csv"), schema)
            except Exception as exc:  # pragma: no cover - schema is optional
                print(f"    {year}: schema.csv download skipped ({exc})")
    return paths


# ---------------------------------------------------------------------------
# Harmonisation helpers
# ---------------------------------------------------------------------------
def _pick(header: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in header:
            return c
    return None


def _discover_columns(header: list[str]) -> dict[str, str]:
    """Map our logical groups to the concrete columns present in *header*."""
    found: dict[str, str] = {}
    for group, cands in MATRIX_COLUMN_CANDIDATES.items():
        col = _pick(header, cands)
        if col:
            found[group] = col
    return found


def _delim_match(series: pd.Series, token: str) -> pd.Series:
    """Exact (delimiter-bounded) membership test on a ';'-separated cell."""
    packed = ("|" + series.fillna("").astype(str)
              .str.replace(r"\s*;\s*", "|", regex=True) + "|")
    return packed.str.contains("|" + token + "|", regex=False)


def _detect_skill(df: pd.DataFrame, matrix_cols: list[str],
                  aliases: list[str]) -> tuple[str, str, int] | None:
    """Return (alias, column, n_matches) for the first alias that appears."""
    for alias in aliases:
        best_col, best_n = None, 0
        for col in matrix_cols:
            n = int(_delim_match(df[col], alias).sum())
            if n > best_n:
                best_n, best_col = n, col
        if best_col is not None:
            return alias, best_col, best_n
    return None


def _years_to_int(s):
    """Same tolerant conversion as ``src/preprocess.py``."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s == "Less than 1 year":
        return 0
    if s == "More than 50 years":
        return 51
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return np.nan


def load_year(year: int, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read one survey year, harmonise it and return (base, skills, info)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()

    comp_col = _pick(header, COMP_CANDIDATES)
    workexp_col = _pick(header, WORKEXP_CANDIDATES)
    country_col = _pick(header, COUNTRY_CANDIDATES)
    if comp_col is None or workexp_col is None or country_col is None:
        raise RuntimeError(f"{year}: missing essential columns "
                           f"(comp={comp_col}, workexp={workexp_col}, country={country_col})")

    matrix = _discover_columns(header)
    usecols = list(dict.fromkeys(
        [comp_col, workexp_col, country_col, "MainBranch", "Employment"]
        + list(matrix.values())))

    print(f"  {year}: comp='{comp_col}', matrix={matrix}")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    print(f"  {year}: raw shape {df.shape}")

    # --- filter: professional developers currently in work ---------------
    branch = df["MainBranch"].fillna("")
    is_dev = branch.str.contains("developer by profession", case=False)
    emp = df["Employment"].fillna("")
    working = emp.str.contains(r"Employed|Independent|freelancer|self-employed",
                               case=False, regex=True)
    working &= ~emp.str.contains(r"Student|Not employed|Retired", case=False, regex=True)
    df = df[is_dev & working].copy()
    print(f"  {year}: working developers {df.shape}")

    # --- compensation / experience ---------------------------------------
    comp = pd.to_numeric(df[comp_col], errors="coerce")
    comp = comp.where(comp > 0)
    df["log_comp"] = np.log1p(comp)
    df["WorkExp"] = df[workexp_col].apply(_years_to_int)
    df = df.dropna(subset=["log_comp", "WorkExp"]).reset_index(drop=True)

    # IQR inlier flag (mirrors preprocess.comp_inlier; don't drop here)
    q1, q3 = df["log_comp"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["comp_inlier"] = ((df["log_comp"] >= q1 - 3 * iqr)
                         & (df["log_comp"] <= q3 + 3 * iqr)).astype(int)

    # top-10 country dummies (re-derived per year)
    top_countries = df[country_col].value_counts().head(10).index.tolist()
    for c in top_countries:
        df[f"country_{slugify(c)}"] = (df[country_col] == c).astype(int)

    # --- skill binaries --------------------------------------------------
    matrix_cols = sorted(set(matrix.values()))
    skills = pd.DataFrame(index=df.index)
    found: dict[str, str] = {}
    for name, aliases in CORE_SKILLS.items():
        hit = _detect_skill(df, matrix_cols, aliases)
        if hit is None:
            continue
        alias, col, n = hit
        skills[name] = _delim_match(df[col], alias).astype(int)
        found[name] = f"{alias} @ {col}"

    base = df[["log_comp", "WorkExp", "comp_inlier"] + [c for c in df.columns
                                                        if c.startswith("country_")]].copy()
    info = {
        "comp_column": comp_col,
        "matrix_columns": matrix,
        "top_countries": top_countries,
        "n_working_devs": int(df.shape[0]),
        "n_model_sample": int((df["comp_inlier"] == 1).sum()),
        "skills_found": found,
    }
    return base, skills, info


# ---------------------------------------------------------------------------
# Per-skill MLR (replicates src/regression.py::per_skill_roi)
# ---------------------------------------------------------------------------
def per_skill_roi_year(base: pd.DataFrame, skills: pd.DataFrame,
                       year: int) -> list[dict]:
    country_cols = [c for c in base.columns if c.startswith("country_")]
    sub = base[(base["comp_inlier"] == 1) & base["log_comp"].notna()] \
        .dropna(subset=["WorkExp"] + country_cols).reset_index(drop=True)
    sub_skills = skills.loc[sub.index]
    rows: list[dict] = []
    for name in skills.columns:
        d = sub_skills[name]
        users = int(d.sum())
        if users < MIN_SKILL_USERS:
            continue
        X = sm.add_constant(pd.concat([sub[["WorkExp"] + country_cols],
                                       d.rename(name)], axis=1))
        try:
            m = sm.OLS(sub["log_comp"], X).fit()
        except Exception:
            continue
        coef = float(m.params[name])
        p = float(m.pvalues[name])
        if np.isnan(coef) or np.isnan(p):
            continue
        rows.append({
            "year": year,
            "skill": name,
            "coef_log_comp": coef,
            "pct_salary_impact": float((np.exp(coef) - 1) * 100),
            "p_value": p,
            "n_users": users,
        })
    return rows


# ---------------------------------------------------------------------------
# Summary + plot
# ---------------------------------------------------------------------------
def build_summary(df: pd.DataFrame, infos: dict, per_year_n: dict) -> dict:
    """Per-year N/headline plus the biggest 2022->2025 salary-impact drifts."""
    pivot = df.pivot_table(index="skill", columns="year",
                           values="pct_salary_impact")
    years = sorted(df["year"].unique())
    top_by_year: dict[str, list[dict]] = {}
    for y in years:
        sub = df[df["year"] == y].sort_values("pct_salary_impact", ascending=False)
        top_by_year[str(y)] = sub.head(5)[
            ["skill", "pct_salary_impact", "p_value", "n_users"]].round(3).to_dict("records")

    first, last = years[0], years[-1]
    drifts: list[dict] = []
    if first in pivot.columns and last in pivot.columns:
        both = pivot[[first, last]].dropna()
        for skill, row in both.iterrows():
            drifts.append({
                "skill": skill,
                "pct_first": float(row[first]),
                "pct_last": float(row[last]),
                "delta": float(row[last] - row[first]),
            })
    drifts.sort(key=lambda r: r["delta"], reverse=True)

    if drifts:
        riser, faller = drifts[0], drifts[-1]
        headline = (f"{riser['skill']} salary impact rose from "
                    f"{riser['pct_first']:.1f}% in {first} to "
                    f"{riser['pct_last']:.1f}% in {last} "
                    f"({riser['delta']:+.1f} pts), while {faller['skill']} fell "
                    f"from {faller['pct_first']:.1f}% to {faller['pct_last']:.1f}% "
                    f"({faller['delta']:+.1f} pts).")
    else:
        headline = "Insufficient overlapping skills to compute 2022->2025 drift."

    tracked = sorted(pivot.dropna(how="all").index.tolist())
    return {
        "years": [int(y) for y in years],
        "per_year_n_working_devs": {str(y): infos[y]["n_working_devs"] for y in infos},
        "per_year_n_model_sample": {str(y): infos[y]["n_model_sample"] for y in infos},
        "per_year_top5_skills": top_by_year,
        "skills_tracked": tracked,
        "n_skills_tracked": len(tracked),
        "headline": headline,
        "biggest_risers": drifts[:5],
        "biggest_fallers": drifts[-5:][::-1],
        "comp_columns": {str(y): infos[y]["comp_column"] for y in infos},
        "matrix_columns": {str(y): infos[y]["matrix_columns"] for y in infos},
        "skill_token_map": {str(y): infos[y]["skills_found"] for y in infos},
        "min_skill_users": MIN_SKILL_USERS,
    }


def plot_trend(df: pd.DataFrame, path: Path) -> None:
    """Line chart: pct salary impact vs. year, one line per tracked skill."""
    pivot = df.pivot_table(index="year", columns="skill",
                           values="pct_salary_impact")
    # Keep every skill that has any data (gaps are fine).
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    palette: list = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        make = plt.get_cmap(cmap_name)
        palette += [make(i / 20) for i in range(20)]

    fig, ax = plt.subplots(figsize=(14, 9))
    for i, skill in enumerate(pivot.columns):
        s = pivot[skill].dropna()
        if s.empty:
            continue
        color = palette[i % len(palette)]
        ax.plot(s.index, s.values, marker="o", linewidth=1.8, markersize=5,
                color=color, label=skill)
        # label the line at its last point
        ax.annotate(skill, (s.index[-1], s.values[-1]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=6.5, color=color)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Survey year", fontsize=11)
    ax.set_ylabel("% salary impact (exp(coef) - 1) x 100", fontsize=11)
    ax.set_title("Stack Overflow skill salary ROI trend, 2022-2025\n"
                 "(per-skill MLR: log comp ~ WorkExp + top-10 country + skill)",
                 fontsize=13)
    ax.set_xticks(sorted(pivot.index))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              ncol=1, frameon=False)
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved chart -> {path}")


def write_readme(df: pd.DataFrame, summary: dict) -> None:
    """Human-readable notes: method, gaps, renamed columns, surprises."""
    years = summary["years"]
    per_year = df.groupby("year")["skill"].nunique().to_dict()
    lines = [
        "# Multi-year skill salary-ROI trend (2022-2025)",
        "",
        "Generated by `src/trend.py` (`python -m src.trend`).",
        "",
        "## Method",
        "- For each survey year, professional developers currently in work are",
        "  selected with the `MainBranch` + `Employment` heuristic from",
        "  `src/preprocess.py`.",
        "- Per skill: `log1p(comp) ~ WorkExp + top-10 country dummies + skill`,",
        "  fitted separately per skill (avoids multicollinearity), exactly like",
        "  `src/regression.py::per_skill_roi`. Rows need >= "
        f"{MIN_SKILL_USERS} users and a non-NaN p-value.",
        "- `% salary impact = (exp(coef) - 1) * 100`.",
        "",
        "## Compensation / harmonisation notes",
        f"- Compensation columns used: {summary['comp_columns']}.",
        "  All four years ship a USD-converted `ConvertedCompYearly`, so it is",
        "  preferred over raw `CompTotal` everywhere.",
        f"- Matrix columns discovered per year: {summary['matrix_columns']}.",
        "- Only `*HaveWorkedWith`-style columns are used (WantToWorkWith /",
        "  Admired / HaveEntry are ignored).",
        "",
        "## Coverage",
        f"- Skills tracked in the chart: {summary['n_skills_tracked']}.",
        f"- Skills with data per year: {per_year}.",
        "- Gaps are expected: a skill-year is omitted when the skill is absent",
        "  that year or has fewer than the minimum number of users, so a line",
        "  may be broken between years.",
        "",
        "## Headline",
        summary["headline"],
        "",
        "## Biggest drifts 2022 -> 2025",
    ]
    for r in summary["biggest_risers"]:
        lines.append(f"- UP   {r['skill']}: {r['pct_first']:.1f}% -> "
                     f"{r['pct_last']:.1f}% ({r['delta']:+.1f} pts)")
    for r in summary["biggest_fallers"]:
        lines.append(f"- DOWN {r['skill']}: {r['pct_first']:.1f}% -> "
                     f"{r['pct_last']:.1f}% ({r['delta']:+.1f} pts)")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_README.write_text("\n".join(lines) + "\n")
    print(f"  saved notes -> {OUT_README}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(years: tuple[int, ...] = YEARS) -> dict:
    """Download, harmonise, regress, plot. Returns the summary dict."""
    print("=== Skill Radar multi-year ROI trend ===")
    print("[1/5] downloading surveys ...")
    paths = download_years(years)

    print("[2/5] loading & harmonising years ...")
    all_rows: list[dict] = []
    infos: dict[int, dict] = {}
    per_year_n: dict[int, int] = {}
    for year in years:
        base, skills, info = load_year(year, paths[year])
        infos[year] = info
        per_year_n[year] = info["n_working_devs"]
        rows = per_skill_roi_year(base, skills, year)
        print(f"  {year}: {len(skills.columns)} candidate skills -> "
              f"{len(rows)} skill-year rows")
        all_rows.extend(rows)
        del base, skills

    df = pd.DataFrame(all_rows).sort_values(["year", "skill"]).reset_index(drop=True)
    print(f"  assembled {df.shape[0]} rows, {df['skill'].nunique()} skills")

    # retain only the ~25 skills present (with data) in all years for the chart
    counts = df.groupby("skill")["year"].nunique()
    tracked = counts[counts == len(years)].index
    df = df[df["skill"].isin(tracked)].reset_index(drop=True)
    print(f"  tracked in all {len(years)} years: {df['skill'].nunique()} skills")

    print("[3/5] writing trend_roi.csv ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  saved {df.shape} -> {OUT_CSV}")

    print("[4/5] building summary ...")
    summary = build_summary(df, infos, per_year_n)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  saved -> {OUT_JSON}")
    print(f"  headline: {summary['headline']}")

    print("[5/5] rendering chart ...")
    plot_trend(df, OUT_PNG)
    write_readme(df, summary)
    return summary


if __name__ == "__main__":
    run()
