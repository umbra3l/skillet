# Surface brief — skillet Streamlit app

## Scope and visitor mode

Operate. A single-page Streamlit application at `app/streamlit_app.py`,
backed by a Python analytics pipeline under `src/`. The visitor is a
reader: they open the cover, read the standfirst, take the number,
read five ranked entries, check who they are in the data, study the
figure, optionally run the secondary calc, optionally read the notes.
Code-led (no comp round — the ambition lives here, in the FIRST
VIEWPORT and SIGNATURE INTERACTION blocks).

## Audience, job, action, proof, constraints

- **Audience:** software engineers and engineering managers (see
  PRODUCT.md). Office, daytime; a cover story read on a laptop.
- **Job:** decide whether to act on the recommended top-5 skills, and
  which to fund (manager) or learn (individual).
- **Action:** read the cover number, read the five entries, open the
  notes if skeptical, optionally try the calc.
- **Proof:** the 19,210-row survey, the MLR coefficients with
  p-values, the 95% intervals, the persona assignment, the top-20
  coefficient figure. Every number on the page is backed by a file in
  `data/`.
- **Constraints:** no new pip deps; Streamlit + Plotly only; runs in
  ~1.5 GB RAM on a laptop; deployable to Streamlit Cloud (Python 3.12
  per `runtime.txt`); self-hostable fonts in the future (currently
  Google Fonts via the CSS with system fallbacks — acceptable for the
  demo, noted as future work).

## Chosen direction and memorable moment

**Editorial Magazine** (pinned by the user; replaces the retired
cast-iron world entirely). The page is a serious quarterly's cover
story on tech hiring: bone paper (`#f3eee5`), true publication ink
(`#0e0d0a`), warm rust accent (`#a23a22`), sharp corners (0–2 px),
Newsreader (display, optical sizing on) for the cover title,
numerals, folios and heads, Manrope for body, JetBrains Mono for
datelines and captions. The **memorable moment** is a quiet
first-render cover fold — a top hairline, the wordmark arriving
once, a rust rule infilling — that plays while the KNN index and
salary calculator warm, then hands off to the instant cover.

## Direction contract (the six blocks)

**THESIS** — A data-driven upskilling recommender, presented as an
editorial product that happens to be backed by data. The hero is a
cover number in its own ruled block ("if you learn these 5 skills,
your compensation is modelled at +X%"), not a dashboard metric. The
number stands; the rest of the issue earns it.

**OWN-WORLD** — Editorial Magazine. Palette: bone paper
(`#f3eee5`–`#e8e1d2`), true ink (`#0e0d0a`), mid slate secondaries
(`#3d4046`, `#6b6e74`), rust primary (`#a23a22`), ember hot state
(`#c44a1a`), cool forest positive (`#2f5a3a`), cooler brick negative
(`#6e2d22`). Typography: Newsreader (display, opsz 6–72), Manrope
(body), JetBrains Mono (numerals, datelines). Radii: 0 (sharp) with
2 px on form controls. No glass, no gradient text, no emoji icons,
no colour side borders above 1 px on items, no rounded SaaS cards,
no shadows-as-depth (rules, never offset blur), no section numbers.
The browser-default surfaces (selection, caret, focus rings,
scrollbars, tabular numerals) are themed from the palette.

**STORY** — The reader sets a profile in the Index. The issue
recomputes: a masthead with imprint and byline, a drawn cover
diagram, a standfirst, a full-width cover number with a stats row,
five ranked page items in a two-up grid, a persona section, a
framed Figure 1 with ramp legend and prose, a Notes colophon, a
secondary Calc, a three-cell colophon footer.

**FIRST VIEWPORT** — Imprint (`THE STACK OVERFLOW ANNUAL · 2025 ·
ISSUE 01`) on a 2 px ink rule. Cover title `skillet.` in Newsreader
with a rust full stop. Deck line ("The dataset that decides what
you should learn next."). Attributed mono byline (editorial team,
survey, 19,210 developers, skills scored). Then the drawn cover
diagram (coefficient skyline, rust trend, ember markers, mono
caption) and the 19 px standfirst. Then the cover number between
rust rules: `+NN%` in Newsreader at clamp(96–168 px), the `↳ five
skills` byline, one stand sentence, and the stats row on a single
ruled line (skills in plan · significant · 95% interval · peer
premium, all mono tabular).

**FORM** — Editorial Magazine, code-led. The form is the inherited
Operate vocabulary rebuilt as print furniture: no sticky dashboard
header (a static masthead), a quiet paper Index sidebar (Profile /
Your stack, Newsreader heads on cream hairlines, sentence-case
Manrope labels, mono dateline), content on a single measured
reading spine (1020 px), items as paper-edged page furniture with
double rules instead of shadows. The signature interaction is the
**cover fold** on first load, plus the session-gated staggered
reveal of the five story items. Respect `prefers-reduced-motion`:
composed stills, no transform.

**FINISH** — "unreviewed and undocumented is unfinished; this build
ends with the finish review, the verdict, DESIGN.md, and every
shipping raster carrying its provenance"
