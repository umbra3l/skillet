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
  **One hover micro-interaction:** `.sk-card:hover` sets
  `border-left-color: var(--ember)` over 160 ms ease-out. The
  border stays exactly 1 px and it appears only on a state
  change, never as static decoration. The card does not lift,
  scale, or bloom a shadow. The `.sk-pill` deepens its
  background to `--paper-3` on the same hover.
- **`.sk-metrics`** — a 4-cell mono grid. Values are
  `tabular-nums lining-nums`; the p-value cell is smaller
  (`.sk-pval`, 14 px) with a faint `p =` prefix (`.pre`,
  `--steel-2`, 10 px) so the eye reads `p = 1.2e-15` cleanly.
- **`.sk-persona .share`** — a 4 px bar with a copper→ember
  gradient fill and a two-layer `inset` box-shadow so it reads
  as cast into the card, not drawn on top of it.
- **`.sk-prompt`** — the cold-start call to action: `--paper-2`
  ground, 2 px `--copper` left border, ink text, ember arrow.
  The arrow nudges 4 px right on hover (160 ms ease-out) and
  returns on leave. (This 2 px side border is the one static
  exception the brief authorises — it is a call-to-action band,
  not a card or list item.)
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
  number sits next to a label. `.streamlit/config.toml` carries
  the same tokens (`primaryColor` copper, `baseRadius = "0px"`),
  so Streamlit's own widgets inherit the world.

## Structural rules (Streamlit-specific)

These are not design choices; they are how Streamlit renders, and
getting them wrong silently breaks the layout:

- **One `st.html` call per visual block.** Each `st.html` call
  renders as its own isolated element — an opening
  `st.html('<div class="sk-cols">')` and a closing
  `st.html('</div>')` produce two *empty* divs and nothing is
  ever wrapped. The persona + recommendation columns are built
  as strings and emitted in a single call so the 1fr : 1.45fr
  grid actually contains them.
- **Streamlit widgets cannot nest inside `st.html` divs.** The
  expander, the multiselect and the metric tiles are widgets, so
  `.sk-method`, `.sk-whatif` etc. wrap only their heading; the
  widget styles are scoped to their `data-testid` directly
  (there is exactly one of each on the page).
- **Streamlit 1.65 renders select/multiselect with react-aria,
  not baseweb.** The old `[data-baseweb="select"]` /
  `[data-baseweb="tag"]` selectors are dead. The real hooks are
  `.react-aria-ComboBox > [role="group"]` for the box (with
  `[data-focused]` for the copper focus ring) and
  `span[data-tag]` for selected-skill tags.

## The pan loading animation (splash)

`session_state.skillet_ready` is `False` on the first script run.
The app renders the splash HTML (inline SVG skillet + wordmark +
caption + ember bar), then warms the caches (`load_artifacts`,
`get_engine`), sets the flag, and `st.rerun()`. The second run
skips the splash and renders the full app.

The splash is one rehearsed sequence — the pan is *cooking*. Every
motion is CSS-driven (no SMIL, so `prefers-reduced-motion` can kill
it cleanly) and uses `--ease-settle` (`cubic-bezier(0.16, 1, 0.3, 1)`,
confident deceleration) for entrances or `--ease-breath`
(`cubic-bezier(0.45, 0, 0.55, 1)`, long ease-in-out) for continuous
breathing. No linear, no bounce, no elastic.

The splash HTML:
- Cast-iron ellipse body (`fill: #1c1815`) with a long handle and a
  small **hanging hole** (`#f7f0e1` ellipse in the handle) — this
  replaces the old rectangular end-cap, which read as a step.
- `.sk-pan-burner` — a blurred (`blur(6px)`) ember radial-gradient
  ellipse under the pan: the burner is on. Breathe
  `opacity 0.045 → 0.09 → 0.045` over 4.4 s.
- `.sk-pan-haze` — a warm copper/ember radial-gradient band above
  the pan, displaced by an SVG `feTurbulence` + `feDisplacementMap`
  filter (`#sk-shimmer`) and Gaussian-blurred. Breathe
  `opacity 0.22 → 0.5`, `translateY 4px → -3px`, `scaleX 0.97 → 1.02`
  over 5.2 s. This is the continuous warm motion the eye feels but
  the camera doesn't catch.
- `.sk-pan-heat` — two thin wavy paths (down from three) inside the
  same shimmer filter, staggered by half a cycle (`-2.4s`), rising
  `translateY 6px → -18px` with a fade over 4.8 s.
- `.sk-pan-glow` — a stroke around the pan rim that breathes
  `opacity 0 → 0.3 → 0` over 3.4 s.
- `.sk-wordmark` — "skillet" in Bitter at 56 px, `letter-spacing:
  -0.04em`. **One-shot** `sk-arrive` entrance: `opacity 0 → 1`,
  `translateY 12px → 0` over 700 ms `--ease-settle`, then it sits
  still. The heat is the motion; the type is the anchor.
- `.sk-splash-caption` — "Cooking up your recommendations." in Sora
  at 14 px, `--steel`, same `sk-arrive` entrance delayed 120 ms.
- `.sk-ember-bar` — 220 × 2 px with an inner `<i>`. **One-shot**
  `sk-ember-fill`: `scaleX 0 → 1` from the left over 2.6 s
  `--ease-settle`, with the ember box-shadow blooming then settling
  to nothing. The pan is heating up; it fills, then holds.

`prefers-reduced-motion: reduce` overrides all of the above to
`animation: none`, with a composed still as the fallback: heat paths
at `opacity 0.3`, haze at `0.35`, glow at `0.25`, burner at `0.07`,
the wordmark/caption/hero-number at `opacity 1` with no transform,
and the ember bar at `scaleX(1)`. No half-states, no invisible
elements.

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
