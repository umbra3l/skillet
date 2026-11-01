---
name: skillet
description: What to learn next, charted as a night sky of skills.
colors:
  navy: "#0b1220"
  panel: "#111a2d"
  panel-2: "#17223a"
  rule: "#25324d"
  graticule: "#18233a"
  ink: "#e9edf5"
  ink-2: "#b7c1d6"
  ink-3: "#8593ad"
  dim: "#4f5d7a"
  amber: "#f0a65a"
  sky: "#7fb3d5"
  sage: "#9bd3a4"
  rose: "#d9776b"
  ramp-low: "#41678f"
  ramp-mid: "#7fa7cb"
typography:
  display:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "clamp(80px, 7vw, 116px)"
    fontWeight: 300
    lineHeight: 0.9
    letterSpacing: "-0.045em"
    fontFeature: "lnum, tnum"
  readout:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "72px"
    fontWeight: 300
    lineHeight: 0.95
    letterSpacing: "-0.04em"
    fontFeature: "lnum, tnum"
  headline:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "clamp(34px, 3.4vw, 52px)"
    fontWeight: 300
    lineHeight: 1.04
    letterSpacing: "-0.02em"
  wordmark:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "44px"
    fontWeight: 300
    lineHeight: 0.95
    letterSpacing: "-0.03em"
  persona-name:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "clamp(30px, 2.6vw, 40px)"
    fontWeight: 300
    lineHeight: 1.1
  section-head:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "clamp(28px, 2.4vw, 36px)"
    fontWeight: 300
    letterSpacing: "-0.015em"
  plate-head:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "22px"
    fontWeight: 400
    lineHeight: 1.2
  object-name:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "21px"
    fontWeight: 400
    lineHeight: 1.2
  rail-head:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.2
  list:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  note:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  figure:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "-0.01em"
    fontFeature: "tnum"
  body:
    fontFamily: "Hanken Grotesk, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.65
  caption:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
  tick:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1
    fontFeature: "tnum"
rounded:
  square: "0px"
  track: "2px"
  tag: "3px"
  base: "6px"
  marker: "50%"
spacing:
  row: "14px"
  cell: "18px"
  head-gap: "18px"
  section-top: "22px"
  rail-inset: "26px"
  plan-inset: "30px"
  column-gap: "48px"
  footer: "56px"
  section: "64px"
components:
  left-rail:
    textColor: "{colors.ink}"
    padding: "2px 26px 18px 2px"
    width: "clamp(250px, 21%, 320px)"
  plan-column:
    textColor: "{colors.ink}"
    padding: "0 0 4px 30px"
  catalogue-section:
    textColor: "{colors.ink-2}"
    padding: "22px 0 0"
  index-table-head:
    textColor: "{colors.ink-3}"
    typography: "{typography.note}"
    padding: "10px 14px 10px 0"
  index-table-cell:
    textColor: "{colors.ink-2}"
    padding: "18px 14px 18px 0"
  ruled-field:
    textColor: "{colors.ink}"
    rounded: "{rounded.square}"
  multiselect-tag:
    textColor: "{colors.ink}"
    rounded: "{rounded.tag}"
  persona-list:
    textColor: "{colors.ink-2}"
    typography: "{typography.list}"
  calc-readout:
    textColor: "{colors.ink}"
    typography: "{typography.readout}"
    padding: "18px 0"
  significance-track:
    rounded: "{rounded.track}"
    height: "3px"
  sky-plate:
    backgroundColor: "{colors.navy}"
    rounded: "{rounded.square}"
    height: "640px"
  hover-label:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    typography: "{typography.caption}"
---

# Design System: skillet

## Overview

**Creative North Star: "The Star Atlas"**

Skill space is drawn as a night sky. Every skill in the survey is a star on a deep navy chart ground, placed by how often developers use skills together and sized by its salary effect; the visitor's own stack is a constellation, and the five recommendations are the bright objects nearest it, ringed in amber. The rest of the page reads like the catalogue pages of an atlas: an index of the five objects with their full evidence, a persona entry, two star-and-whisker figures, a calculator and method notes. The user chose this world in October 2026 (seed a54a0412). It replaces the retired Editorial Magazine world entirely. Mode is Operate: this is a working instrument, not a poster.

Since the October 2026 refinement the page is set as open atlas plates rather than boxed cards. Nothing below the chart sits on a panel ground, a border box or a rounded corner; sections, rows and the rail are divided by 1px hairlines on the navy ground, and only the sky chart keeps a frame. Light Spectral, mostly italic, names objects and sets every heading, label and caption; Hanken Grotesk carries controls, lead text and every tabular figure. There are no shadows, no glow and no decorative gradients; the one gradient is the significance ramp, a data legend.

Rejected outright: a neon or terminal look, SaaS KPI tiles over a table, glow, decorative gradients, eyebrow labels over headings, tracked uppercase micro-labels, and Unicode-glyph icons.

**Key Characteristics:**
- Deep navy ground, chalk ink, a fine graticule; open plates divided by hairlines, not cards.
- Amber marks the five recommended skills, the focus ring, the wordmark stop and the joint total's sign, nothing else.
- The sky chart is the only framed object: a square plate with a double neatline, graduated ticks and a plate caption.
- Italic Spectral does every label and caption; Hanken Grotesk does the counting.
- Every estimate is drawn as a star on a whisker: a marker for the value, a hairline for its 95% interval.

## Colors

A cold night palette of navy and chalk with one warm star colour and two quiet effect colours.

### Primary
- **Recommendation Amber** (`amber`): rings the five recommended skills wherever they appear: the outer ring and stroke on the sky chart picks, the ring marker in each plan row and index row, and the 2.5px ring on ROI-figure stars that are in the plan. It is also the sign of the joint total, the `:focus-visible` outline (2px, offset 2px), the stop on the "skillet." wordmark and footer, and the first-load hairline.

### Secondary
- **Control Sky** (`sky`): every Streamlit control (theme `primaryColor`), links, the focused bottom rule of a field, the relaxed-filter note's left hairline, the notes summary on hover, the movement tag ("up 1", "new") in plan rows, the sign of the what-if lift, text selection (`rgba(127,179,213,0.28)`), the caret, and the 3.5% row wash on the index table (`rgba(127,179,213,0.035)`).

### Tertiary
- **Positive Sage** (`sage`): salary effects at or above zero, in plan rows, index rows and calculator markers.
- **Negative Rose** (`rose`): salary effects below zero, in the same places.

### Neutral
- **Chart Navy** (`navy`): the page, the sky plate, the scrollbar track. Every plate below the chart sits directly on it.
- **Raised Panel** (`panel`): chart hover labels only in the page; also Streamlit's code and (hidden) sidebar background in the theme.
- **Inset Panel** (`panel-2`): Streamlit's secondary background (dropdown menus, slider track) via the theme. No page block uses it.
- **Hairline Rule** (`rule`): section rules, the rail and plan hairlines, field bottom rules, multiselect tag outlines, the plate border, neatline ticks, chart zero lines.
- **Graticule** (`graticule`): the sky chart's rings and meridians, the plate's outer outline, row hairlines in the plan and index, the calculator readout rules, figure gridlines.
- **Chalk** (`ink`): headings, object names, your own stars, the picks' fill, key numbers, the top stop of the significance ramp.
- **Secondary Chalk** (`ink-2`): lead text, the plan heading, persona tags, constellation lines, your skills' labels, table cells.
- **Faint Chalk** (`ink-3`): captions, table heads, widget labels, legend, whisker lines, field hover rule, the heavier rule over a list or table head, other stars with a positive effect.
- **Dim Star** (`dim`): chart-only; other stars whose salary effect is zero or negative.
- **Significance Ramp** (`ramp-low` to `ramp-mid` to `ink`): steel blue to chalk, used only as the ROI star colourscale by -log10 p (0 to 100) and its matching legend track.

### Named Rules
**The One Star Colour Rule.** Amber marks the five recommended skills, the focus ring, the wordmark stop and the joint total's sign. Controls, links, selection, caret and the calculator use sky. If amber appears anywhere else, it is a bug.

**The Effect Sign Rule.** Sage means the effect is positive and rose means it is negative. They never decorate anything else.

## Typography

**Display Font:** Spectral (with Georgia, serif), weights 300 to 500, roman and italic
**Body Font:** Hanken Grotesk (with system-ui, -apple-system, Segoe UI, sans-serif), weights 400 to 700
**Label/Mono Font:** ui-monospace (SF Mono, Menlo) only for the scoring formula in the method notes

Both come from Google Fonts in a single import at the top of the stylesheet.

**Character:** A light, literary serif names the objects and writes the margins the way an atlas does, in italic. A clear grotesque handles the instrument work: controls, lead text and tabular numbers.

### Hierarchy
- **Display** (300, clamp(80px, 7vw, 116px), 0.9, -0.045em, lining tabular figures): the joint total in the plan only; 76px under 900px. Its unit is raised and smaller (0.42em, `ink-3`, lifted 0.9em); its sign is amber.
- **Readout** (300, 72px, 0.95, -0.04em): the what-if lift, with a sky sign. The first-load wordmark uses the same size at 1 / -0.035em.
- **Headline** (300, clamp(34px, 3.4vw, 52px), 1.04): the page heading; its second clause is italic in `ink-2`.
- **Wordmark** (300, 44px, 0.95): "skillet." at the top of the rail. The footer repeats it at 26px.
- **Persona name** (italic 300, clamp(30px, 2.6vw, 40px), 1.1).
- **Section head** (300, clamp(28px, 2.4vw, 36px), -0.015em): the four catalogue sections below the fold.
- **Plate head** (italic 400, 22px/1.2): the plan heading in `ink-2` and the method-notes summary in `ink` (22px/1.3).
- **Object name** (400, 21px/1.2): skill names in plan rows and the index table.
- **Rail head** (italic 400, 18px/1.2, `ink`): "Your profile" and "Your current stack", each under a hairline.
- **List** (italic 400, 16px/1.5): persona tags; the index rank numeral uses italic 16px/1.
- **Body** (400, 15px/1.65, max 68ch): lead paragraphs, the calculator explanation. Notes body is 14.5px/1.7 at 76ch.
- **Figure** (500, 17px, -0.01em, tabular): salary effects in sage or rose, and the facts values in `ink`.
- **Note** (italic 400, 14px/1.5, `ink-3`): figure captions, empty states, the footer line, the brand line under the wordmark (14px/1.45). Its smaller steps: table heads and facts labels at 13.5px; the plate caption, rail footnote, movement tag and ramp label at 13px. The relaxed-filter note and the interval line are italic 15px in `ink-2`.
- **Caption** (400, 13px/1.5, `ink-3`): evidence lines in the index and the chart legend; plan-row evidence and Streamlit widget labels at 12.5px (500, 0.01em, sentence case).
- **Tick** (400, 12px, tabular): ramp ticks and figure axis ticks. Figure category names are Spectral 15px `ink`; axis titles Spectral 13px `ink-3`.

### Named Rules
**The Italic Margin Rule.** Labels, table heads, captions, notes and empty states are italic Spectral in sentence case. There are no tracked uppercase micro-labels anywhere.

**The Names Are Serif Rule.** Anything that names a skill or a persona is Spectral; the five picks on the chart are Spectral italic 16px. Your own skills' labels on the chart are Hanken Grotesk 11px, so the picks read first.

**The Tabular Rule.** Every number that can sit beside another number uses tabular figures.

## Layout

Streamlit wide layout, content capped at 1640px with 2.4rem 2.4rem 3rem padding and 0.9rem block gaps. The left rail is an in-page first column (1 : 3.4 against the main column, flexed to clamp(250px, 21%, 320px)), not Streamlit's collapsible sidebar, which is hidden. It has no ground of its own: a right hairline in `rule` separates it, and it is `position: sticky` (top 1.4rem, its own scroll when taller than the window). It holds the wordmark and dataset line, "Your profile" (role, years slider, country, AI checkbox), "Your current stack" (six multiselects) and a footnote, each group under a hairline.

Main column, top to bottom: header (headline, lead, optional relaxed-filter note); the sky row at 1.7 : 1 with a large gap (the captioned sky plate and its legend, then the plan column, which carries a left hairline and 30px inset); then the index, persona, ROI figure with its significance ramp, the what-if calculator, the method notes and the footer.

Below the fold every section uses one grammar: a 64px gap, a 1px `rule` above, 22px in, then a two-column grid (0.9fr : 1.5fr, 48px column gap, 18px row gap, baseline-aligned) with the heading left and the lead right. Tables and the persona tags span both columns; the persona name sits under its heading and the lead spans both rows on the right. Method notes sit between two `rule` hairlines; the footer sits 56px down under one.

Under 1200px the sky row stacks with the plan first (2rem gap) and the plan drops its left hairline and inset. Under 900px the container padding becomes 1.4rem 1rem 2rem; the rail stacks above the content (1.6rem gap), stops being sticky and swaps its right hairline for a bottom hairline; sections collapse to one column with 48px gaps; the calculator readout stacks; the display total drops to 76px.

**The One Block Rule.** Each visual block is a single `st.html` call. Widgets cannot be nested inside `st.html`, so controls sit between blocks, never inside them.

**The Section Grammar Rule.** A section below the fold is a hairline, then heading left and lead right. A new section uses the same grid; it does not get a box.

## Elevation & Depth

The system is flat and, below the chart, single-toned: everything sits on the navy ground. Depth and grouping come from hairlines in three weights: `ink-3` over a list or table head, `rule` between sections and around the plate, `graticule` between rows. The only tone steps left are Streamlit's own menus (`panel-2`) and chart hover labels (`panel`). There are no box shadows, no glows and no blur, except the 800ms settle-in blur on the joint total during first-render motion. The only gradient is the significance ramp legend.

**The Hairline, Not Box Rule.** A section, row or column is separated by a 1px hairline on the navy ground, never by a panel background, a border box or a shadow.

**The Data-Only Gradient Rule.** A gradient is allowed only when it encodes a value, and the significance ramp is the only one.

## Shapes

Square by default. The sky plate is the single framed object: square corners (0px), a 1px `rule` border and a 1px `graticule` outline 4px out, read as a double neatline, with graduated ticks drawn inside by Plotly. Form fields are bare ruled lines with no corners at all. The only radii are small: 3px on multiselect tags, 2px on the 3px significance track, 6px on the scrollbar thumb and on remaining Streamlit widgets through the theme's `baseRadius`. Markers are circles everywhere (50%): filled dots for stars, open 2px amber rings for picks. The disclosure chevron is two CSS borders rotated, not a glyph.

**The CSS Marker Rule.** `st.html` strips inline SVG, so every page marker (rings, legend dots, magnitude key, chevron) is a CSS shape on a span or pseudo-element. Never use a Unicode glyph as an icon; the middot between persona tags is a typographic separator, not an icon.

## Components

### Sky Chart (signature)
The atlas plate. A Plotly scatter, 640px tall, with transparent paper on the navy plate, no axes, no zoom or drag, and no mode bar.
- **Plate caption:** above the plate, Spectral 13px `ink-3` split across the width: an italic `ink` 15px plate number ("Plate I.") with its title on the left, the placement note on the right. It is the figure's caption, not an eyebrow over a heading.
- **Placement:** every skill is positioned by t-SNE on 1 - Jaccard similarity of the developers who know each pair (`src/skymap.py` writes `data/skill_map.json`).
- **Magnitude:** dot diameter is the salary effect, 3px plus up to 13px at +30% and above; the legend's magnitude key shows +5% / +15% / +30%. Other stars are `ink-3` when the effect is positive and `dim` otherwise.
- **Constellation:** your skills are 9px `ink` dots joined by a minimum spanning tree in `ink-2`, 1.2px at 0.75 opacity, labelled in Hanken Grotesk 11px.
- **Picks:** 14px `ink` dots with a 2.5px amber stroke, an outer 30px amber ring at 1px and 0.55 opacity, and Spectral italic 16px labels.
- **Orientation:** up to five of the highest-effect other stars in view get faint Spectral 12px `ink-3` names, kept only if they clear every other label.
- **Framing:** cropped around your constellation and the picks (0.75 of their extent, minimum half-width 0.32) at a 1.25 : 1 aspect.
- **Graticule:** four concentric rings and six meridians in `graticule`. A neatline carries 41 graduated ticks per side in `rule`, every fifth tick long.
- **Hover:** a `panel` label with a `rule` border in Hanken Grotesk 13px: name, salary effect, p and n.

### Plan Column
An open column beside the plate: an italic 22px `ink-2` heading, the joint total at display size (amber sign, raised `ink-3` unit), an italic 15px interval line, then the five rows under an `ink-3` rule. Each row is a 22px / 1fr / auto grid of amber ring, name (21px serif, with an optional italic sky movement tag and a 12.5px evidence line) and a 17px effect figure, 14px above and 13px below, divided by `graticule` hairlines. A two-column facts list (italic 13.5px labels, 17px values) closes it.

### Index Table
"The five, in detail": an open table that scrolls horizontally. Heads are italic 13.5px `ink-3` over an `ink-3` rule; cells have 18px vertical padding, 14px right, `graticule` hairlines and 14px `ink-2` text. Each row has an italic rank with an amber ring, the 21px object name with its evidence line, the effect figure, the interval, p and counts. Rows take the sky wash on hover over 160ms.

### Persona Entry
The section grammar with the italic persona name under the heading and the lead on the right. Defining skills are an italic 16px `ink-2` serif list separated by `ink-3` middots, spanning both columns. No pills, no borders.

### ROI Figure and Significance Ramp
A star-and-whisker plot of the twenty largest effects, 640px tall: each skill a 1.2px `ink-3` interval line and an 11px star coloured by -log10 p along `ramp-low` to `ramp-mid` to `ink`. Stars in the plan carry a 2.5px amber ring. Names are Spectral 15px `ink`; gridlines `graticule`, zero line `rule`. Below sits the ramp legend: an italic "significance" label, a 3px track at 2px radius in the same three stops, and 12px tabular ticks (p ≈ 1 to ≤ 10⁻¹⁰⁰).

### What-if Calculator
A Streamlit multiselect, then an open readout between two `graticule` rules: the lift at 72px with a sky sign beside a 15px explanation (auto / 1fr, 36px gap). A star-and-whisker breakdown follows with the same whisker and 11px markers, sage or rose by sign, then an italic note caption. The empty state is an italic note line.

### Inputs (Streamlit widgets)
Fields are ruled lines. Select and multiselect groups are transparent with no corners and only a 1px bottom border: `rule` at rest, `ink-3` on hover, sky on focus-within, over 160ms. Input text is 15px `ink`. Multiselect tags are transparent with a 1px `rule` outline at 3px radius and `ink` text. Widget labels are 12.5px 500 `ink-3`; the checkbox label 14px `ink-2`. The theme supplies the sky primary, `panel-2` menus and 6px radius for the rest. Focus shows the global amber outline.

### Method Notes
A `<details>` between two `rule` hairlines. The summary is an italic 22px serif line with a rotating CSS chevron (200ms) and turns sky on hover. The body is 14.5px `ink-2`, inset 23px to align with the summary text; the formula sits in a `pre` with no ground, only a left `rule` hairline.

### Motion
Motion plays only on the first render of a session, gated by a session flag. The joint total settles in over 800ms (12px rise, 6px blur) and the plan rows rise 6px over 560ms, staggered from 120ms in 60ms steps, on `cubic-bezier(0.16, 1, 0.3, 1)`. Reruns from input changes do not animate. Everything sits behind `prefers-reduced-motion: no-preference`. The first-load fold shows the 72px wordmark, an amber hairline filling over 1.6s and an italic caption; the fill stops under reduced motion.

## Do's and Don'ts

### Do:
- **Do** reserve amber for the five recommended skills, the focus ring, the wordmark stop and the total's sign, and use sky for every control, link, selection and caret.
- **Do** use sage only for positive effects and rose only for negative effects.
- **Do** divide sections, rows and columns with 1px hairlines on the navy ground (`ink-3` over heads, `rule` between sections, `graticule` between rows).
- **Do** set new sections in the section grammar: hairline above, heading left, lead right, one column under 900px.
- **Do** set labels, table heads, captions and empty states in italic Spectral, sentence case.
- **Do** draw estimates as a star on a whisker: a 1.2px `ink-3` interval line and an 11px marker.
- **Do** name skills and personas in Spectral, and set every number in Hanken Grotesk with tabular figures.
- **Do** build markers as CSS shapes on spans, because `st.html` strips inline SVG.
- **Do** emit one `st.html` per visual block and keep widgets outside it.
- **Do** gate entrance motion to the first render of a session and respect `prefers-reduced-motion`.
- **Do** keep the sky chart square-cornered with its double neatline, graduated ticks, graticule and plate caption.

### Don't:
- **Don't** use a neon or terminal look.
- **Don't** use SaaS KPI tiles.
- **Don't** box content in cards: no panel backgrounds, border boxes or 10px radii below the chart.
- **Don't** use glow, shadows or gradients, except the significance ramp data legend.
- **Don't** put eyebrow labels over headings, or tracked uppercase micro-labels anywhere.
- **Don't** use pills for tags or movement markers; tags are an italic serif list, movement is italic sky text.
- **Don't** use Unicode-glyph icons.
- **Don't** put amber on controls, links, selection or the calculator. Those are sky.
- **Don't** animate reruns. The chart redraws on every input change and must stay still.
