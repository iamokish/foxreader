// Helper for test_symbol_fallbacks.py: run the actual frontend with API fonts.
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolve } from "node:path";

const root = fileURLToPath(new URL("../../", import.meta.url));
const require = createRequire(resolve(root, "frontend/package.json"));
const { JSDOM } = require("jsdom");
const { rolldown } = await import(pathToFileURL(require.resolve("rolldown")).href);
const bundle = await rolldown({
    input: "symbol-contract",
    plugins: [{
        name: "symbol-contract",
        resolveId(id) { if (id === "symbol-contract") return "\0symbol-contract"; },
        load(id) {
            if (id !== "\0symbol-contract") return;
            const path = name => JSON.stringify(resolve(root, `frontend/src/${name}.ts`).replaceAll("\\", "/"));
            return `export { state } from ${path("state")}; export * from ${path("entries/textFit")};`;
        },
    }],
});
const { output } = await bundle.generate({ format: "esm" });
await bundle.close();
const dom = new JSDOM("<!doctype html><body></body>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const renderer = await import(`data:text/javascript;base64,${Buffer.from(output[0].code).toString("base64")}`);
let input = "";
for await (const chunk of process.stdin) input += chunk;
const data = JSON.parse(input);
renderer.state.currentFontsData = data.fonts;
renderer.state.currentSymbolFonts = data.symbols;
renderer.state.currentSymbolFont = data.symbols.at(-1);
// jsdom supplies the DOM, and the real unhinted font advances supply its SVG
// measurement primitive. The fit, run selection and placement are production TS.
dom.window.SVGElement.prototype.getComputedTextLength = function () {
    const widths = data.advances[this.getAttribute("font-family")];
    assert.ok(widths, "unknown measured family");
    return [...this.textContent].reduce((sum, character) => sum + widths[character], 0);
};
const ns = "http://www.w3.org/2000/svg";
const svg = document.createElementNS(ns, "svg");
document.body.appendChild(svg);
const text = document.createElementNS(ns, "text");
svg.appendChild(text);
renderer.resetTextMetrics();
for (const item of data.cases) {
    try {
        const font = data.fonts.find(font => font.font_filename === item.filename);
        const family = font.font_name.replaceAll(/\s+/g, "_");
        const geom = { wordSpacing: item.wordSpacing, lineSpacing: item.lineSpacing, scaleX: 1, scaleY: 1 };
        const layout = renderer.fitText(svg, item.text, family, item.align, item.box, item.explicitSize,
            geom, renderer.fontMetricsFromFontData(font));
        assert.equal(layout.size, item.size);
        assert.deepEqual(layout.lines.map(line => line.tokens.join(" ")), item.lines);
        const outline = renderer.drawFitLayout(text, layout, item.align, item.box,
            { family, fill: "black", stroke: "white", strokeWidth: item.strokeWidth }, "", svg);
        if (outline) svg.insertBefore(outline, text);
        const calls = [];
        const layers = [...svg.children].filter(node => node === outline || node === text);
        for (const layer of layers) {
            for (const span of layer.querySelectorAll("tspan")) {
                const spanFamily = span.getAttribute("font-family") ?? family;
                const size = Number(span.getAttribute("font-size") ?? layout.size);
                const widths = data.advances[spanFamily];
                const advance = character => widths[character] * size / 100;
                const width = [...span.textContent].reduce((sum, character) => sum + advance(character), 0);
                const anchor = span.getAttribute("text-anchor") ?? layer.getAttribute("text-anchor");
                let x = Number(span.getAttribute("x")) - (anchor === "middle" ? width / 2 : anchor === "end" ? width : 0);
                for (const character of span.textContent) {
                    if (character !== " ") calls.push({ text: character, x, y: Number(span.getAttribute("y")),
                        family: spanFamily, size, fill: layer.getAttribute("fill"),
                        strokeWidth: Number(layer.getAttribute("stroke-width")) / renderer.STROKE_ATTR_COEFF });
                    x += advance(character);
                }
            }
        }
        assert.equal(calls.length, item.calls.length);
        calls.forEach((call, index) => {
            const expected = item.calls[index];
            assert.equal(call.text, expected.text);
            assert.equal(call.family, expected.family);
            assert.equal(call.size, expected.size);
            assert.equal(call.fill, expected.fill);
            assert.equal(call.strokeWidth, expected.strokeWidth);
            assert.ok(Math.abs(call.x - expected.x) < 1e-8, `x ${call.x} != ${expected.x}`);
            assert.ok(Math.abs(call.y - expected.y) < 1e-8, `y ${call.y} != ${expected.y}`);
        });
        outline?.remove();
    } catch (error) {
        throw new Error(`${JSON.stringify({ ...item, calls: undefined })}: ${error.message}`, { cause: error });
    }
}
console.log(`${data.cases.length} frontend/backend symbol layouts matched`);
dom.window.close();
