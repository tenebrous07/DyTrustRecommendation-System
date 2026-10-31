// Assembles docs/DyTrustRec_Research_Paper.docx. Run: node make_paper.js (from docs/build).
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, AlignmentType, SectionType, Footer, PageNumber } = require("docx");
const L = require("./lib");
const S = require("./paper_static");
const B = require("./build_paper");
const S2 = require("./second_dataset");
const { F } = B;

const DOCS = path.resolve(__dirname, "..");
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 900, bottom: 1080, left: 900 } };
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
  new TextRun({ children: [PageNumber.CURRENT], size: 16, font: L.FONT })] })] });

// each block is {cols, children}; consecutive blocks with the same column count merge
const blocks = [];
const push = (cols, child, newPage = false) => {
  const last = blocks[blocks.length - 1];
  if (!newPage && last && last.cols === cols) last.children.push(child);
  else blocks.push({ cols, newPage, children: [child] });
};
// items wrapped as {wide: element} span both columns; everything else flows in two
const add = (cols, children) => children.forEach((c) => (c && c.wide ? push(1, c.wide, !!c.newPage) : push(cols, c)));

const fig1 = path.join(DOCS, "diagrams", "DyTrustRec_Architecture.png");
const method = S.method(F);

add(1, B.title);
add(2, B.abstract());
add(2, B.introduction());
add(2, [B.H1("II. Related Work"), ...S.relatedWork(F)]);
add(2, [B.H1("III. Dataset and Problem Formulation"), ...S.dataSection(F)]);
// Fig. 1 opens a fresh page so it cannot strand a half-empty column; Section IV text follows it.
add(1, [{ wide: L.imageParagraph(fig1, 7.1, 7.1 * 1580 / 1900, { title: "architecture" }), newPage: true },
  { wide: B.cap("Fig. 1. DyTrustRec end-to-end architecture, from raw review ingestion through the trust-calibrated, explainable top-K output.") }]);
add(2, [B.H1("IV. Proposed Methodology"), ...method]);
add(2, [B.H1("V. Experimental Setup"), ...S.setup(F)]);

const results = B.resultsSection();
add(2, results);
add(2, B.comparisonSection());
add(2, B.ablationSection());
add(2, B.horizonSection());
add(2, B.trustSection());
add(2, B.confidenceSection());
add(2, B.confidenceCalibrationParagraph());
add(2, B.coldStartSection());
add(2, B.adaptiveSection());
add(2, B.fidelitySection());
add(2, S2.section());
if (B.limitations) add(2, B.limitations());
if (B.conclusion) add(2, B.conclusion());
add(2, B.references());

const sections = blocks.map((b, i) => ({
  properties: {
    type: (i === 0 || b.newPage) ? SectionType.NEXT_PAGE : SectionType.CONTINUOUS,
    page: PAGE,
    column: b.cols === 2 ? { count: 2, space: 360, equalWidth: true } : { count: 1 },
  },
  footers: { default: footer },
  children: b.children,
}));

const doc = new Document({
  creator: "DyTrustRec", title: "DyTrustRec",
  styles: { default: { document: { run: { font: L.FONT, size: 20 } } } },
  numbering: L.numbering,
  sections,
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(DOCS, "DyTrustRec_Research_Paper.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
