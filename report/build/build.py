"""Render a report Markdown file to print-ready HTML with KaTeX math.

    uv run --with markdown python report/build/build.py DA_REPORT.md

Writes report/build/out/<name>.html, which print.js turns into a PDF
(see report/build/README.md).
"""
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
REPO = REPORT.parent
SRC = REPORT / sys.argv[1]
OUT = HERE / "out" / (SRC.stem + ".html")

STUDENT = [("Name", "Vibhaansh Bhardwaj"), ("Registration No.", "24BCE1401"),
           ("Course", "Business Analytics"), ("Faculty", "Dr. Balaji")]

COVERS = {
    "PROJECT_REPORT": dict(
        kicker="Business Analytics · Course Project Report",
        subtitle="An evidence-based skill recommender for software developers",
        desc="Full report on the data, the mathematical modelling, the results and the interactive application.",
        meta=[("Data", "Stack Overflow Developer Survey 2025 (2022–2024 for the trend analysis)"),
              ("Sample", "19,210 working developers · 122 skills · 156 countries")],
        blanks=False),
    "PRESENTATION_GUIDE": dict(
        kicker="Business Analytics · Presentation Study Guide",
        subtitle="What every number on the app means, the maths in plain language, and how to present it",
        desc="A cheat sheet, a guide to the web app, the modelling explained, a timed presentation plan "
             "with a demo script, likely questions with answers, and test scenarios.",
        meta=[("Companion", "PROJECT_REPORT.pdf (full report with derivations)")],
        blanks=False),
    **{f"DA{n}_REPORT": dict(
        kicker=f"Business Analytics · DA {n}",
        subtitle=sub,
        desc=desc,
        meta=[("Project", "A recommendation model for what software developers should learn next"),
              ("Data", "Stack Overflow Developer Survey 2025")],
        blanks=STUDENT) for n, sub, desc in [
        (1, "Problem / Idea of the Model",
         "Title, background, problem statement, objectives, proposed dataset, proposed methodology "
         "and statistical techniques, and expected recommendations and outcomes."),
        (2, "Software / Model Implementation",
         "Introduction and objectives, dataset, preprocessing, model methodology and implementation, "
         "statistical techniques, screenshots and sample outputs, and initial results."),
        (3, "Final Report and Results",
         "Introduction, objectives, dataset and methodology, statistical analysis and interpretation, "
         "model outputs and recommendations, conclusion and future scope, and references."),
    ]},
    "DA_REPORT": dict(
        kicker="Business Analytics · Recommendation Model Project",
        subtitle="A recommendation model for what software developers should learn next",
        desc="DA 1: Problem and idea of the model · DA 2: Software and model implementation · "
             "DA 3: Final report and results",
        meta=[("Data", "Stack Overflow Developer Survey 2025"),
              ("Model", "Log-pay regression with country fixed effects · similarity-weighted KNN · "
                        "joint plan regression · Streamlit web application")],
        blanks=STUDENT),
}
cover_cfg = COVERS.get(SRC.stem, COVERS["PROJECT_REPORT"])

text = SRC.read_text()
# The cover page replaces the Markdown title block (everything before the first rule).
body_md = text.split("\n---\n", 1)[1]

store: list[tuple[str, bool]] = []


def keep(tex: str, display: bool) -> str:
    store.append((tex, display))
    return f"MATHPH{len(store) - 1}X"


codes: list[str] = []


def keep_code(m):
    codes.append(m.group(0))
    return f"CODEPH{len(codes) - 1}X"


body_md = re.sub(r"```.*?```", keep_code, body_md, flags=re.S)
# Python-Markdown only starts a list after a blank line; GitHub does not need one.
LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s")
lines = body_md.split("\n")
fixed: list[str] = []
for ln in lines:
    if LIST_ITEM.match(ln) and fixed and fixed[-1].strip() and not LIST_ITEM.match(fixed[-1]) \
            and not fixed[-1].startswith(("  ", "|")):
        fixed.append("")
    fixed.append(ln)
body_md = "\n".join(fixed)
body_md = body_md.replace(r"\$", "LITDOLLARX")
body_md = re.sub(r"\$\$(.+?)\$\$", lambda m: "\n\n" + keep(m.group(1).strip(), True) + "\n\n",
                 body_md, flags=re.S)
body_md = re.sub(r"\$([^\s$][^$\n]*?)\$", lambda m: keep(m.group(1), False), body_md)
for i, c in enumerate(codes):
    body_md = body_md.replace(f"CODEPH{i}X", c)

html = markdown.markdown(body_md, extensions=["tables", "toc", "fenced_code", "sane_lists"])

rendered = json.loads(subprocess.run(
    ["node", str(HERE / "katex_batch.js")], input=json.dumps(store),
    capture_output=True, text=True, check=True, cwd=HERE).stdout)
for i, h in enumerate(rendered):
    html = html.replace(f"<p>MATHPH{i}X</p>", f'<div class="eq">{h}</div>')
    html = html.replace(f"MATHPH{i}X", h)
html = html.replace("LITDOLLARX", "$")
html = html.replace('src="../assets/', f'src="file://{REPO}/assets/')
html = html.replace('src="screenshots/', f'src="file://{REPORT}/screenshots/')
# Images that sit alone in a paragraph become figures.
html = re.sub(r'<p>((?:<img [^>]+>\s*)+)</p>', r'<figure>\1</figure>', html)

meta = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in cover_cfg["meta"])
meta += '<dt>Live app</dt><dd><a href="https://skillet.streamlit.app/">https://skillet.streamlit.app/</a></dd>'
meta += f"<dt>Date</dt><dd>{date.today():%B %Y}</dd>"
blanks = ""
if cover_cfg["blanks"]:
    rows = "".join(f'<div class="blank"><span>{lab}</span><b>{val}</b></div>'
                   for lab, val in cover_cfg["blanks"])
    blanks = f'<div class="blanks">{rows}</div>'
cover = f"""
<section class="cover">
  <p class="kicker">{cover_cfg["kicker"]}</p>
  <h1>skillet</h1>
  <p class="subtitle">{cover_cfg["subtitle"]}</p>
  <p class="desc">{cover_cfg["desc"]}</p>
  {blanks}
  <dl class="meta">{meta}</dl>
</section>
"""

title = cover_cfg["kicker"].split("·")[-1].strip()
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>skillet — {title}</title>
<link rel="stylesheet" href="file://{HERE}/node_modules/katex/dist/katex.min.css">
<link rel="stylesheet" href="file://{HERE}/report.css">
</head><body>{cover}<main>{html}</main></body></html>"""
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(page)
print(f"wrote {OUT} ({len(store)} formulas)")
