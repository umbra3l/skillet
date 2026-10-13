# Surface brief — skillet Streamlit app

## Scope and visitor mode

Operate. A single-page Streamlit application at `app/streamlit_app.py`,
backed by a Python analytics pipeline under `src/`. The visitor is in
a task: enter a profile, read a hero number, read five recommendation
cards, optionally run a what-if, optionally read the methodology.
Code-led (no comp round — Operate is code-led by contract; the
ambition lives here, in the FIRST VIEWPORT and SIGNATURE INTERACTION
blocks).

## Audience, job, action, proof, constraints

- **Audience:** software engineers and engineering managers (see
  PRODUCT.md). Office, daytime; data dashboards read on a laptop.
- **Job:** decide whether to act on the recommended top-5 skills, and
  which to fund (manager) or learn (individual).
- **Action:** read the hero number, skim the cards, open the
  methodology if skeptical, optionally try the what-if.
- **Proof:** the 19,210-row survey, the MLR coefficients with
  p-values, the 95% intervals, the persona assignment, the 4-year
  trend chart. Every number on the page is backed by a file in
  `data/`.
- **Constraints:** no new pip deps; Streamlit + Plotly only; runs in
  ~1.5 GB RAM on a laptop; deployable to Streamlit Cloud (Python 3.12
  per `runtime.txt`); self-hostable fonts in the future (currently
  Google Fonts via the CSS with system fallbacks — acceptable for the
  demo, noted as future work).

## Chosen direction and memorable moment

**Cast-iron heritage** (pinned by the user). The page is a kitchen
workbench: a warm parchment / vellum surface, a deep cast-iron
charcoal hero panel that *is* the predicted-lift number, copper-amber
accents, sharp corners (radius 0–2 px), a confident editorial serif
(Fraunces, upright) for the display number and section headings, a
workhorse sans (IBM Plex Sans) for body, JetBrains Mono for numbers.
The **memorable moment** is a smooth first-load cast-iron pan
animation — a skillet heating up, a heat shimmer, an ember progress
line — that plays once per session and hands off to the app. After
that, the app is instant.

## Direction contract (the six blocks)

**THESIS** — A data-driven upskilling recommender, presented as a
serious artisan tool rather than a SaaS dashboard. The hero is a
single quantified number ("if you learn these 5 skills, your
compensation is modelled at +X%"), not the category's "big number +
4 glass stat cells + accent gradient" template. The number stands;
the rest of the page earns it.

**OWN-WORLD** — Cast-iron heritage. Palette: warm parchment paper
(`#f7f0e1`–`#ede2c8` surface), deep cast-iron charcoal
(`#1c1815` ink, `#2a2520` panel), copper-amber accent
(`#b8743a` for primary actions, focus, the **+** in the hero), one
ember hot state (`#c44a1a`). Muted patinaed-copper positive
(`#3f5a3a`) and brick negative (`#7a3528`) for ROI signs.
Typography: Fraunces (display, upright, optical sizing), IBM Plex
Sans (body, UI), JetBrains Mono (numerals, p-values, code).
Radii: 0 (sharp) with 2 px on form controls. No gradient text, no
glass-as-decoration, no coloured side borders above 1 px, no
pulsing dot, no italic display serif, no AI-beige ground, no
"01 / 02 / 03" section numbers, no kicker labels above headings.
The browser-default surfaces (text selection colour, caret, focus
rings, scrollbars, tabular numerals) are themed from the palette.

**STORY** — The visitor enters a profile on the left. The page
recomputes: a single bold predicted-lift number appears in a dark
cast-iron panel, a persona card on the left names the cluster the
user belongs to, five recommendation cards on the right show the next
skills with their salary-ROI coefficient, p-value, CI and a
one-sentence evidence line. A short methodology expander shows the
scoring formula. A coefficient chart shows where the recommendations
come from in the wider skill-ROI landscape. A secondary what-if lets
the visitor add their own shortlist and see the summed lift.

**FIRST VIEWPORT** — Header (skillet brand, demo badge, three
data chips: 19,210 developers · 122 skills scored · p-values
attached). Then the dark cast-iron hero panel: the **+NN%** number
in Fraunces at clamp(64–84 px), a one-sentence caption naming the
five contributing skills, four compact typographic info pairs
(skills contributing · significant · 95% interval · peer premium),
no glass cards, no gradient, no kicker. Below: a 1:1.45 two-column
section — left, a persona card with the cluster name, share, share
bar, distinguishing-skill chips, peer count, peer premium; right,
five recommendation cards stacked, each a sharp-cornered panel with
the rank, the skill name in Fraunces, a 4-cell mono metric grid
(salary impact · p-value · satisfaction coef · n users), a CI
caption, and the evidence sentence. A short, quiet methodology
expander sits below.

**FORM** — Cast-iron heritage, code-led. The form is the inherited
Operate vocabulary (top bar, side inputs, content columns) rebuilt
in cast-iron's own language: a flat top bar (not a sticky blurred
floating bar), a quiet sidebar (muted, not glassed), content on a
warm parchment surface, the hero as the one dark cast-iron plate
in the page. The signature interaction is the **pan loading
animation** on first load: an SVG cast-iron skillet (dark ellipse +
handle), a gentle heat shimmer rising above it, a thin ember
progress line below, the wordmark "skillet" in Fraunces and the
caption "Cooking up your recommendations." The animation runs while
the KNN index and salary calculator are first-built; on subsequent
runs (cache hit) the splash is suppressed and the app is instant.
Respect `prefers-reduced-motion`: replace the shimmer with a static
ember bar, no transform.

**FINISH** — "unreviewed and undocumented is unfinished; this build
ends with the finish review, the verdict, DESIGN.md, and every
shipping raster carrying its provenance"
