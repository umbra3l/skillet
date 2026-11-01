// Read [[tex, display], ...] on stdin, print the rendered HTML strings as JSON.
const katex = require("katex");
let buf = "";
process.stdin.on("data", (d) => (buf += d));
process.stdin.on("end", () => {
  const out = JSON.parse(buf).map(([tex, display]) => {
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: "ignore" });
    } catch (e) {
      process.stderr.write(`KaTeX error in: ${tex}\n  ${e.message}\n`);
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" });
    }
  });
  process.stdout.write(JSON.stringify(out));
});
