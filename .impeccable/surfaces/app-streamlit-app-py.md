---
version: 1
slug: "app-streamlit-app-py"
primary_target: "app/streamlit_app.py"
related_targets: [".streamlit/style.css"]
---

# Surface brief — skillet app (app/streamlit_app.py)

Scope: the whole Streamlit app. Mode: Operate (the visitor completes a task: describe a profile and stack, read the recommended next skills and their evidence).

Audience and job: Business Analytics graders on laptops, reading up close at their own pace; developers and L&D leads deciding what to learn or fund. They must see the five recommended skills, the joint salary lift with its 95% interval, and the evidence behind each, and be able to change any input and watch the answer move. Must avoid: neon or hacker-terminal looks, generic SaaS dashboard tiles, gloomy low-contrast greys. Layout may change freely; every number, input and evidence sentence stays.

## Direction contract

THESIS: Skill space as a night sky. The visitor's stack is drawn as a constellation among all 122 skills (placed by how often they are used together, sized by salary effect); the five recommendations are the bright stars nearest it, ringed in amber. Refuses the dashboard arrangement of KPI tiles over a table, and the incumbent printed-magazine spine.

OWN-WORLD: deep navy chart ground (#0b1220) with two raised navy panels, chalk-white stars and ink, a fine coordinate graticule, blue-grey secondary text, sage for positive effects, rose for negative. Amber (#f0a65a) is reserved for one job: marking the five recommended skills wherever they appear (sky chart rings, index, ROI figure, calculator) and the focus ring. Light serif with italics (Spectral) names objects and headings; a grotesque (Hanken Grotesk) carries UI and tabular figures. Thin 1px ruled panels, no shadows, no gradients, no glow.

STORY: The visitor sets role, experience, country, AI use and stack in the left rail; sees their constellation and the five ringed stars; reads the joint lift (+37%, 95% CI +33% to +42%) and the ranked list beside the chart; drops into the Index of objects for full evidence; checks persona, the ROI landscape and the what-if calculator; reads the method notes.

FIRST VIEWPORT: Left rail (an in-page sticky first column, not Streamlit's sidebar): wordmark "skillet." with amber stop, dataset line, Your profile (role, years slider, country, AI checkbox), Your current stack (six multiselects). Main, two columns: left ~62% the sky chart (interactive Plotly scatter on the graticule: all skills as points, the user's skills white and joined by constellation lines, the five picks with amber rings and italic labels; hover shows name, salary effect, p); right ~38% a ruled panel: "All five, estimated together", the joint lift at display size with its interval and "4 of 5 significant", then the five ranked rows (amber ring, name, peer share, p, effect). A quiet line above the columns states the peer group and any relaxed filter.

FORM: Celestial star atlas (catalog challenger notation-diagram-systems-star-atlas, also my own top-ranked grounded candidate), chosen by the user as the IMPECCABLE'S PICK card; seed key a54a0412. Signature interaction: the live sky chart — your stack as a constellation, recommendations ringed amber, hover for evidence; it redraws whenever an input changes.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Unresolved

- None beyond the build.
