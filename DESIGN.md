# Design — skillet

The visual world is **Editorial Magazine**, pinned by the user
(see `SURFACE_BRIEF.md`). A serious quarterly's cover story on tech
hiring — imprint, cover title, hero illustration, standfirst, cover
number, ranked entries, persona section, figure, notes, calc,
colophon. This file records what was built so a later agent (or a
later edit) can extend it without drifting to the dashboard default
or back to the retired cast-iron world.

## The world

A printed annual on bone paper in true publication ink. One reading
spine, top to bottom: masthead → illustration → standfirst → cover
number → cover story → persona → figure → notes → calc → footer.
Rules do the structuring (1–2 px ink, rust, and cream lines); there
are no shadows, no glass, no coloured side borders above 1 px, no
pulsing dots, no emoji icons, no gradient text, no rounded SaaS
cards, no section numbers. The single deliberate exception is the
significance ramp legend, a thin data gradient that replaces the
chart colour bar.

The signature moment is a quiet first-render cover fold: a top
hairline, the wordmark arriving once in Newsreader, a rust rule
infilling left to right while the KNN index and salary calculator
warm. On every subsequent rerun the cover is instant.
`prefers-reduced-motion` replaces all of it with a composed still.

## Tokens (CSS custom properties on `:root`)

```
/* bone paper */
--paper:    #f3eee5;   /* page ground */
--paper-2:  #e8e1d2;   /* panels, page items */
--paper-3:  #d8d0bf;   /* tints, dividers */

/* publication ink */
--ink:      #0e0d0a;   /* true ink, not warm */
--ink-2:    #1f1d18;
--ink-3:    #3a3833;

/* mid slate */
--steel:    #3d4046;   /* secondary text */
--steel-2:  #6b6e74;   /* captions, tertiary */

/* accents */
--rust:     #a23a22;   /* warm rust primary accent */
--rust-2:   #862d18;
--ember:    #c44a1a;   /* hot state (plan outlines, markers) */

/* semantic */
--patina:   #2f5a3a;   /* positive — cool forest */
--brick:    #6e2d22;   /* negative — cooler brick */

/* type */
--font-display: 'Newsreader', Georgia, 'Times New Roman', serif;
--font-body:    'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
--font-mono:    'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;

/* rhythm */
--r-1..--r-9: 4, 8, 12, 16, 24, 32, 48, 64, 96 px;

/* line widths (sharp) */
--line: 1px;  --line-2: 2px;
```

All radii are `0` (sharp) with `2px` on form controls.

## Type

Display: **Newsreader** — variable with an `opsz` 6–72 axis, served
via Google Fonts. Used like a real publication face: large, tight,
`font-optical-sizing: auto` on headings, the cover title, the folio
numbers, and the cover numeral. The story item names pin
`"opsz" 36`.

Body: **Manrope** — standfirst (19 px), deks, descriptions, figure
prose, widget labels in sentence case.

Mono: **JetBrains Mono** — the imprint, bylines, datelines, the
skills line, the stats row, strip values, CI lines, figure labels,
the ramp legend, the notes formula, the calc caption.
`font-variant-numeric: tabular-nums lining-nums` on every numeral
that sits beside a label.

Cover number (`+49%`-style): Newsreader 500 at
`clamp(96–168 px)`, `letter-spacing: -0.04em`, `line-height: 0.9`,
ink with a rust `+`, steel `%`.

## Layout (the single spine)

- **Masthead.** Imprint mono line on a 2 px ink top rule
  (`THE STACK OVERFLOW ANNUAL · 2025 · ISSUE 01`), the cover title
  `skillet.` in Newsreader up to 128 px with a rust full stop, a
  one-sentence deck line, and an attributed mono byline on a cream
  hairline.
- **Hero illustration.** A drawn-in-brand SVG publication diagram
  (`_cover_illus_html`): a coefficient skyline — thin slate bars for
  the scored skills, a rust trend line, ember-ringed markers on the
  recommended plan, hairline baseline with end ticks, two small mono
  annotations. Centred, max 360 px tall, mono caption beneath
  (`Cover · per-skill salary coefficients across N skills, 2025`).
- **Standfirst.** One Manrope 19 px paragraph, 68 ch, centred
  measure, directly below the illustration.
- **Cover number.** Full-width block with 2 px rust rules above and
  below: the numeral, the `↳ skills` mono byline, one stand
  sentence, and the four info pairs on a single ruled line
  (`Skills in plan · 5 │ Significant · 5 of 5 │ 95% interval ·
  +14% → +82% │ Peer premium · +8.4%`).
- **Cover story.** `The cover story.` head, then a two-up grid
  (single column under 880 px) of five page items. Each item:
  folio rank in Newsreader italic 44 px rust, skill name in
  Newsreader 500 opsz 36, one-sentence Manrope dek, a four-cell
  stat strip with sharp `--paper-3` dividers
  (`effect │ p = 4×10⁻⁶ │ n = 471 │ 22% want it`), the 95% interval
  in small mono beneath. Sign is carried by the value colour
  (`--patina` / `--brick`) only. Paper edge: `--paper-2` ground, 1
  px `--paper-3` outer rule, second 1 px rule 4 px inside (no
  `box-shadow` anywhere).
- **Persona.** `Who you are in the data.` between ink hairlines:
  two columns (one under 880 px) — left, persona name in
  Newsreader, Manrope description, mono dateline
  (`cluster N · share of sample N%`); right, defining-skill tags in
  small ruled mono.
- **Figure.** `FIGURE 1` mono label (the figure label the chart
  needs), `Skill ROI landscape.` head, short lead, the Plotly chart
  in a 1 px `--paper-3` frame on `--paper` (Newsreader skill
  names, `--steel-2` whiskers, no colour bar), the six-stop
  significance ramp as a thin gradient rule with mono ticks
  (`p > 0.05 · 1e−5 · 1e−10 · 1e−20 · 1e−30`), then real editorial
  prose under the figure.
- **Notes.** Methodology as a native `<details class="sk-notes">`
  colophon block: a real-sentence summary, dek paragraph, the
  scoring formula in a `<pre>` with the one allowed 2 px rust
  rule, the numbered pipeline with ember mono markers, a data
  source paragraph, a limitations paragraph.
- **Calc.** `What if I also learned these?` with a
  `SECONDARY CALC` marginal label at the heading's right margin
  (not above it): a compact typed read-out (lift in Newsreader
  display, one-sentence interpretation with the independence
  caveat), then the shrunk contribution chart in the standard
  thin frame with a real caption.
- **Footer.** Three mono cells on an ink top hairline, middle-dot
  separated: the annual line, the type credits, the survey line.
- **Sidebar — Editorial Index.** Paper ground, no shadow. Two
  sections (`Profile`, `Your stack`), each a Newsreader 18 px
  heading with a `--paper-3` hairline beneath. Widget labels in
  Manrope sentence case, rust focus rings. A mono dateline at the
  bottom (`19,210 developers · 122 skills`).

## Components

- **`.sk-item`** — paper edge from two rules, never shadow. One
  hover move: both rules deepen to `--steel-2` over 140 ms.
- **`.sk-strip`** — four cells on a 1 px `--paper-3` grid (the 1 px
  gap shows the ground through as dividers), cells on `--paper`,
  mono values with tabular numerals, sign-coloured effect.
- **`.sk-tag`** — small ruled mono tag on `--paper`; border
  deepens to `--rust` on hover over 140 ms.
- **`.sk-notes`** — native details; `+` / `–` mono markers in rust
  before the summary; body on `--paper` over a cream top rule.
- **`.sk-ramp`** — the colour bar's replacement: label, 6 px
  gradient rule across the six significance stops, mono ticks
  spread beneath.
- **Browser surfaces** are themed: rust selection, ember caret,
  slate scrollbar on bone, 2 px rust focus rings. Streamlit's
  theme tokens in `.streamlit/config.toml` carry the same palette
  (`primaryColor` rust, `baseRadius = "0px"`).

## Structural rules (Streamlit-specific)

These are not design choices; they are how Streamlit renders, and
getting them wrong silently breaks the layout:

- **One `st.html` call per visual block.** Each `st.html` call
  renders as its own isolated element — wrapper divs must open
  and close in the *same* call. The story grid (head + all five
  items) is built as one string and emitted once.
- **Streamlit widgets cannot nest inside `st.html` divs.** The
  multiselects and both Plotly charts are widgets; heading blocks
  wrap only their own markup and widget styles are scoped to
  their `data-testid` directly.
- **Streamlit renders select/multiselect with react-aria.** The
  hooks are `.react-aria-ComboBox > [role="group"]` (with
  `[data-focused]` for the rust focus ring) and `span[data-tag]`
  for selected-skill tags.
- **There is exactly one reading spine.** The old four-variant
  dispatch (`?variant=A|B|C|D`, `body:has([data-sk-variant])`
  scoping, the variant bar) is deleted. Do not reintroduce per-
  variant CSS.

## The cover fold (first render)

`session_state.skillet_ready` is `False` on the first script run.
The app renders the fold (top hairline + `skillet` in Newsreader
via one-shot `sk-arrive` + rust rule infilling `scaleX 0 → 1`),
warms the caches, sets the flag, and `st.rerun()`. The second run
renders the cover instantly. `prefers-reduced-motion` shows the
wordmark set and the rule full, with no animation.

## Motion

First render of the session only, gated through
`session_state.skillet_entered`: the wordmark arrives once, the
cover number rises 8 px with opacity over 600 ms
(`--ease-settle`), the five story items fade up staggered 60 ms
apart. All hovers are 140 ms ease-out, no bounce. Reduced motion
resets everything to a composed still — no half-states.

## What is refused (and why)

- **No `Fraunces` / `IBM Plex` / `Inter` / `Space Grotesk` /
  `Plus Jakarta` / `DM Sans` / `Geist` etc.** Newsreader (with
  opsz), Manrope, and JetBrains Mono are the publication's faces.
- **No dashboard chrome**: no sticky header rows of chips, no
  bordered metric tiles, no glass, no gradient text, no pulsing
  dots, no emoji icons, no `box-shadow` depth, no rounded SaaS
  cards, no coloured side borders above 1 px on items.
- **No section numbers** (`01 / 02 / 03`). Folio ranks (`01`–`05`)
  stay because the ranking *is* the information; `FIGURE 1` stays
  because the chart needs its label.
- **No cast-iron residue**: no pan SVG, no copper tokens, no
  Bitter/Sora faces, no variant scopes. The rebrand is total.

## Self-check (impeccable mechanical detector)

`/home/umbra3l/.config/opencode/skills/impeccable/scripts/impeccable
detect --json .streamlit/style.css app/streamlit_app.py`

Must return `[]`. Re-run on every material edit; fix
`severity: error` findings before shipping.

## Where to extend

- **Add a skill.** It flows through the story grid, the figure,
  and the calc automatically — same item anatomy, same strip,
  same evidence dek. Nothing new in the design system.
- **Add a new section.** An `h2` in Newsreader, short steel lead,
  content on `--paper-2` items or plain measure, mono for
  numbers, rust reserved for accents and folio-scale moments.
- **Change the world.** Don't. Editorial Magazine is pinned
  (PRODUCT.md + SURFACE_BRIEF.md). A revision that drifts to
  another world is a rebrand, not an edit.
