// Drive the running skillet app and capture screenshots for the report.
// electron shoot.js <outdir>
const { app, BrowserWindow } = require("electron");
const fs = require("fs");
const path = require("path");

const OUT = process.argv[process.argv.length - 1];
const URL = "http://localhost:8051";
// 1x capture (offscreen rendering)


const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let win;
const js = (code) => win.webContents.executeJavaScript(code);

async function idle(extra = 600) {
  // wait until Streamlit has finished rerunning
  await sleep(500);
  for (let i = 0; i < 120; i++) {
    const running = await js(`!!document.querySelector('[data-testid="stStatusWidget"] [data-testid="stStatusWidgetRunningIcon"], [data-testid="stStatusWidgetRunningManIcon"]') || document.body.innerText.includes('Running...')`);
    if (!running) break;
    await sleep(250);
  }
  await sleep(extra);
}

async function click(x, y) {
  for (const type of ["mouseDown", "mouseUp"]) {
    win.webContents.sendInputEvent({ type, x: Math.round(x), y: Math.round(y), button: "left", clickCount: 1 });
    await sleep(40);
  }
}

async function key(k, n = 1) {
  for (let i = 0; i < n; i++) {
    win.webContents.sendInputEvent({ type: "keyDown", keyCode: k });
    win.webContents.sendInputEvent({ type: "keyUp", keyCode: k });
    await sleep(60);
  }
}

async function pick(label, text, exact = text) {
  await js(`(() => { const i = document.querySelector('input[aria-label="${label}"]');
    i.scrollIntoView({block: "center"}); i.focus(); return true; })()`);
  await sleep(250);
  await win.webContents.insertText(text);
  await sleep(700);
  const r = await js(`(() => { const o = [...document.querySelectorAll('[role="option"]')]
      .find(e => e.innerText.trim() === ${JSON.stringify(exact)});
    if (!o) return null; o.scrollIntoView({block: "nearest"}); const b = o.getBoundingClientRect();
    return [b.x + b.width / 2, b.y + b.height / 2]; })()`);
  if (!r) throw new Error(`option not found: ${label} / ${exact}`);
  await click(r[0], r[1]);
  await key("Escape");
  await idle();
}

async function scrollTo(selector, offset = 24) {
  await js(`(() => { const e = document.querySelector(${JSON.stringify(selector)});
    e.scrollIntoView({block: "start"});
    const sc = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
    sc.scrollBy(0, -${offset}); return true; })()`);
  await sleep(700);
}

async function shot(name) {
  const img = await win.webContents.capturePage();
  fs.writeFileSync(path.join(OUT, name), img.toPNG());
  console.log("saved", name, img.getSize());
}

app.whenReady().then(async () => {
  win = new BrowserWindow({ width: 1440, height: 900, show: false, webPreferences: { offscreen: true } });
  await win.loadURL(URL);
  console.log("loaded");
  for (let i = 0; i < 120; i++) {
    if (await js(`document.body.innerText.includes("All five, estimated together")`)) break;
    await sleep(500);
  }
  await idle(1500);

  // 1. cold start, defaults
  await scrollTo("header.sk-top", 40);
  await shot("01_cold_start.png");

  // 2. the B1 stack
  await pick("Languages", "Python");
  await pick("Languages", "JavaScript");
  await pick("Languages", "SQL");
  await pick("Databases", "PostgreSQL");
  await pick("Cloud platforms", "Docker");
  await key("Escape");
  await idle(1500);
  await scrollTo("header.sk-top", 40);
  await shot("02_b1_main.png");

  // 3. evidence table
  await scrollTo("section.sk-index");
  await shot("03_evidence.png");

  // 4. persona + start of ROI landscape
  await scrollTo("section.sk-persona");
  await shot("04_persona.png");

  // 5. ROI landscape
  await scrollTo("section.sk-fig");
  await shot("05_roi.png");

  // 6. calculator
  await pick("Skills you are considering", "Kubernetes");
  await pick("Skills you are considering", "Terraform");
  await pick("Skills you are considering", "Snowflake");
  await idle(1500);
  await scrollTo("section.sk-calc");
  await shot("06_calculator.png");

  // 7. method notes, expanded
  await js(`(() => { const d = document.querySelector("details.sk-notes"); d.open = true; return true; })()`);
  await sleep(500);
  await scrollTo("section.sk-notes-sec");
  await shot("07_notes.png");

  // 8. Nepal: relaxed filters
  await pick("Country", "Nepal");
  await idle(1500);
  await scrollTo("header.sk-top", 40);
  await shot("08_nepal_relaxed.png");

  app.quit();
}).catch((e) => { console.error(e); app.exit(1); });
