// Print an HTML file to PDF with Electron's Chromium:
//   FOOTER="skillet · DA Report" electron print.js in.html out.pdf
const { app, BrowserWindow } = require("electron");
const fs = require("fs");
const path = require("path");

const [input, output] = process.argv.slice(-2);
const footer = process.env.FOOTER || "skillet";
app.disableHardwareAcceleration();
app.whenReady().then(async () => {
  const win = new BrowserWindow({ show: false, width: 1000, height: 1400 });
  await win.loadFile(path.resolve(input));
  await win.webContents.executeJavaScript(`
    document.fonts.ready.then(() => Promise.all([...document.images].map(i =>
      i.complete ? 1 : new Promise(r => { i.onload = i.onerror = r; }))))`);
  const broken = await win.webContents.executeJavaScript(
    "[...document.images].filter(i => !i.naturalWidth).map(i => i.src)");
  if (broken.length) console.error("broken images:", broken);
  const pdf = await win.webContents.printToPDF({
    pageSize: "A4",
    printBackground: true,
    margins: { top: 0.75, bottom: 0.75, left: 0.8, right: 0.8 },
    displayHeaderFooter: true,
    headerTemplate: "<div></div>",
    footerTemplate: `<div style="font-family: Georgia, serif; font-size: 8.5px; color: #6b7280;
      width: 100%; padding: 0 0.8in; display: flex; justify-content: space-between;">
      <span>${footer}</span><span class="pageNumber"></span></div>`,
  });
  fs.writeFileSync(output, pdf);
  console.log(`wrote ${output} (${(pdf.length / 1024).toFixed(0)} KB)`);
  app.quit();
});
