# Design — skillet

The visual world is **cast-iron heritage**, pinned by the user
(see `SURFACE_BRIEF.md`). This file records what was built so a later
agent (or a later edit) can extend it without drifting to the
category default.

## The world

A cast-iron skillet on a parchment workbench. The page is the
workbench: warm vellum surface, deep cast-iron charcoal hero
panel, copper-amber accents, sharp corners, a slab-serif display
(Bitter) and a workhorse sans (Sora). No gradient text, no glass,
no coloured side borders above 1 px, no pulsing dot, no italic
display serif, no AI-beige ground, no `01 / 02 / 03` section
numbers, no kicker labels above headings.

The signature moment is a smooth first-load cast-iron pan
animation: an SVG skillet heats up while the KNN index and salary
calculator are first-built; on every subsequent rerun (cache hit)
the splash is suppressed and the app is instant. `prefers-reduced-
motion` replaces the shimmer with a static ember bar.

## Tokens (CSS custom properties on `:root`)

```
/* ink (cast-iron charcoal) */
--ink:    #1c1815;   /* primary text, the hero panel, the pan */
--ink-2:  #2a2520;   /* pan interior, secondary dark */
--ink-3:  #3a342c;   /* rim highlight */

/* paper (warm vellum) */
--paper:   #f7f0e1;  /* app surface */
--paper-2: #ede2c8;  /* cards, panels */
--paper-3: #e3d5b6;  /* lines, deeper tints */

/* mid neutrals */
--steel:   #6a6258;  /* secondary text on paper */
--steel-2: #8a8275;  /* tertiary / muted, hero-on-dark caption */

/* accent (copper) */
--copper:   #b8743a; /* primary action, focus, the "+" in the hero */
--copper-2: #a86830;
--ember:    #c44a1a; /* hot state — the hero number, the heat */

/* semantic (muted, world-aligned) */
--patina:   #3f5a3a; /* positive ROI (muted forest) */
--brick:    #7a3528; /* negative ROI (muted brick) */

/* type (off the AI overused list) */
--font-display: 'Bitter', Georgia, 'Times New Roman', serif;
--font-body:    'Sora', system-ui, -apple-system, 'Segoe UI', sans-serif;
--font-mono:    'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;

/* rhythm */
--r-1..--r-9: 4, 8, 12, 16, 24, 32, 48, 64, 96 px;

/* line widths (sharp) */
--line: 1px;  --line-2: 2px;
```

All radii are `0` (sharp) with `2px` on form controls. The chatty
"modern SaaS rounded card" is banned.

## Type

Display: **Bitter** — a slab serif. Sturdy, editorial, slightly
industrial; the flat slab terminals echo the cast-iron pan's flat
rim. Controlled by `font-weight` (300–700); no optical-sizing
axis.

Body: **Sora** — a clean geometric workhorse. Used for UI labels,
sidebar, captions, the cold-start prompt, the methodology.

Mono: **JetBrains Mono** — numerals (the hero number, p-values,
sample sizes), CI captions, the scoring formula in the methodology
expander. `font-variant-numeric: tabular-nums lining-nums` on any
number that sits in a column.

Hero number (`+49%`-style): Bitter at `font-weight: 500`,
`clamp(64–112 px)`, `letter-spacing: -0.045em`, `line-height:
0.9`, color `--ember`, mono numerals via `font-variant-numeric`.

## Layout

- **Fixed top header.** Three regions: brand (`skillet` in Bitter),
  tagline (one sentence about the product, in `--steel` with a
  thin left rule), data chips (sample size, skills scored,
  p-values attached — mono on `--paper-2`). Sticky, no blur, no
  glass.
- **Hero panel.** One `--ink` rectangle per page. The number, a
  caption sentence (no kicker above the number), a `↳ skills`
  line, and a two-column info grid separated from the number by a
  2 px `--copper` left rule. Restrained palette commits in the
  dark hero; the rest of the page is warm paper.
- **Persona + recommendations.** `1fr : 1.45fr` two-column
  section (collapses to single column under 1100 px). Left: a
  single persona card on `--paper-2` with the cluster name as
  h2, a thin ember share bar, defining-skill chips, three mono
  mini-stats (peers · premium · skills listed). Right: five
  recommendation cards stacked, each a sharp `--paper-2` panel
  with a 1 px `--paper-3` border, a 1 px `--paper-3` bottom
  rule under the head, a 4-cell mono metric grid (salary impact ·
  p-value · satisfaction · n users), a CI caption, and the
  evidence sentence.
- **Methodology.** A `st.expander` rendered as a `--paper-2`
  panel with a 1 px border. The scoring formula sits in a
  `<pre>` with a 2 px `--copper` left border (the cast-iron
  cooking line).
- **ROI landscape.** A `plotly_white` chart on a transparent
  paper background, bars coloured on a 6-stop significance ramp
  from `--brick` (insignificant) to deep forest (p < 1e-30), CI
  whiskers in `--steel-2`, the recommended plan outlined in
  `--ember` (3 px) to tie the chart to the hero.
- **What-if.** A bordered section with a 1 px `--paper-3` top
  rule. A `st.multiselect` + 4 `st.metric` tiles + a contribution
  bar chart. The "secondary tool" copy is replaced with a
  multiselect placeholder ("Choose up to 8 skills…").
- **Footer.** Three mono cells in `--steel`, separated by `--paper-3`
  middle dots, on a 1 px top rule.

## Components

- **`.sk-card`** — sharp, 1 px border, no coloured side border.
  The previous "left border green/red by sign" is removed
  (banned by craft-floor: coloured `border-left` > 1 px on
  cards). Sign is carried by `.sk-pill` colour (`--patina` /
  `--brick`) and by the `Salary impact` cell's colour.
- **`.sk-pill`** — a 1 px bordered pill, mono, sharp.
- **`.sk-chip`** / **`.sk-chip-skill`** — small mono or body
  tags on `--paper-2` / `--paper`.
- **`.sk-info .v` / `.sk-mono`** — the hero info grid uses
  Bitter for the value (size 28) and mono for the CI/premium
  cell, with `--steel-2` captions underneath.
- **Browser surfaces** are themed: `::selection` is copper on
  cream, the caret is `--ember`, scrollbar is `--steel` on
  `--paper-2` track, focus rings are 2 px `--copper` with 2 px
  offset (no glow, no shadow). Tabular numerals wherever a
  number sits next to a label.

## The pan loading animation (splash)

`session_state.skillet_ready` is `False` on the first script run.
The app renders the splash HTML (inline SVG skillet + wordmark +
caption + ember bar), then warms the caches (`load_artifacts`,
`get_engine`), sets the flag, and `st.rerun()`. The second run
skips the splash and renders the full app.

The splash HTML:
- Cast-iron ellipse body (`fill: #1c1815`) with a long handle
  extending to the right and a small hanging hole.
- Three `.sk-pan-heat` thin wavy paths above the pan, each a
  warm copper/ember stroke, animated with a 3.2 s ease-out
  infinite rise (`translateY` + opacity), staggered by ~1 s.
- A `.sk-pan-glow` stroke around the pan rim that breathes
  (2.6 s ease-in-out infinite, `opacity: 0 → 0.35 → 0`).
- A `.sk-ember-bar` at the bottom: 220 × 2 px, with an inner
  `<i>` that breathes `scaleX(0.45 → 0.95 → 0.45)` over 1.8 s
  with a 12 px `--ember` box-shadow.
- The wordmark "skillet" in Bitter at 56 px, `letter-spacing:
  -0.04em`.
- The caption "Cooking up your recommendations." in Sora at
  14 px, `--steel`.

`prefers-reduced-motion: reduce` overrides all of the above
animations to `none`, the heat paths drop to `opacity: 0.3`,
and the ember bar holds at `scaleX(0.8)`.

## What is refused (and why)

- **No `Fraunces` / `IBM Plex` / `Inter` / `Space Grotesk` /
  `Plus Jakarta` / `DM Sans` etc.** — these are on the
  AI-convergence overused list. Bitter (slab) and Sora are not.
  If a later edit swaps to one of the overused faces, the
  impeccable detector will flag it (`overused-font`,
  `slop/category`).
- **No kickers / eyebrows above headings.** The hero number
  stands alone; the h2s stand alone. The "01 · Methodology"
  pattern is gone.
- **No section numbers** (`01 / 02 / 03`) unless the sequence
  itself carries information the reader needs. The
  recommendation cards keep their rank numbers (1–5) because
  the ordering *is* the information.
- **No coloured `border-left` / `border-right` above 1 px** on
  cards. The previous green/red sign-border is removed; the
  sign is carried by colour of the pill and the metric value.
- **No glass-as-decoration**, no gradient text, no pulsing
  dot, no italic display serif, no AI-beige ground, no
  emoji-as-icon system, no custom scrollbar / form-control
  reinvention. The pan animation is the one authored moment;
  everything else is quiet.
- **No hero-metric template** in its lazy form (big number +
  four rounded glass stat cells + accent gradient). The
  user's brief earns the metric, but the form is a single
  `--ink` panel with a copper left rule separating the number
  from a typographic info grid (Bitter values, `--steel-2`
  captions). Earned familiarity, not a costume.

## Self-check (impeccable mechanical detector)

`/home/umbra3l/.config/opencode/skills/impeccable/scripts/impeccable
detect --json .streamlit/style.css app/streamlit_app.py`

The current run returns `[]` (zero findings). Re-run on every
material edit; if it returns a non-empty list, fix the
`severity: error` findings before shipping. The
`overused-font` finding on Fraunces is what drove the swap to
Bitter + Sora; keep that constraint.

## Where to extend

- **Add a skill.** The recommendation cards, the chart, the
  hero, and the splash all derive from the same token set
  (`--paper-2` panel, `--ink` ink, `--ember` accent, `0`/sharp
  radius, Bitter for the name, mono for the metrics, evidence
  sentence with a CI caption). Add it the way the existing
  five were added — nothing new in the design system.
- **Add a new section.** A new `h2` with no kicker, a
  short `--steel` lead, content in `--paper-2` panels, mono
  for numbers, Bitter for headings, accent in `--copper` /
  `--ember` / `--patina` / `--brick` only.
- **Change the world.** Don't. Cast-iron heritage is pinned
  (PRODUCT.md + SURFACE_BRIEF.md). A visual revision that
  drifts to another world is a rebrand, not an edit.
