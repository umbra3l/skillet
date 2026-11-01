# Report PDF build

Renders the report Markdown files in `report/` to PDF: KaTeX typesets the
math, and Electron's Chromium prints the page.

```bash
cd report/build
npm install                                   # once: KaTeX
uv run --with markdown python build.py DA_REPORT.md
FOOTER="skillet · Business Analytics DA Report" electron43 print.js out/DA_REPORT.html ../DA_REPORT.pdf
```

The same two steps work for `PROJECT_REPORT.md` and `PRESENTATION_GUIDE.md`.
Literal dollar signs in the Markdown must be written `\$` so they are not read as math.

`shoot.js` recaptures the app screenshots in `report/screenshots/` from a
running app (`streamlit run app/streamlit_app.py --server.port 8051`):

```bash
electron43 shoot.js ../screenshots
```
