// Shared docx-js helpers for the paper and the architecture explainer.
// Text is written as plain strings with light inline markup: **bold**, *italic*, `code`, ^sup^, ~sub~.

const fs = require("fs");
const path = require("path");
const {
  Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  AlignmentType, ImageRun, LevelFormat, VerticalAlign, HeadingLevel,
} = require("docx");

const FONT = "Times New Roman";
const MONO = "Courier New";

function runs(text, base = {}) {
  // tokenise inline markup
  const out = [];
  const re = /(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const tok = m[0];
    if (tok.startsWith("***")) out.push(new TextRun({ text: tok.slice(3, -3), bold: true, italics: true, font: FONT, ...base }));
    else if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (tok.startsWith("*")) out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: MONO, ...base, size: Math.max(14, (base.size || 20) - 2) }));
    else if (tok.startsWith("^")) out.push(new TextRun({ text: tok.slice(1, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), subScript: true, font: FONT, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

function para(text, o = {}) {
  return new Paragraph({
    alignment: o.align ?? AlignmentType.JUSTIFIED,
    spacing: { before: o.before ?? 0, after: o.after ?? 80, line: o.line ?? 240 },
    indent: o.indent,
    keepNext: o.keepNext,
    keepLines: o.keepLines,
    children: runs(text, { size: o.size ?? 20, ...(o.run || {}) }),
  });
}

function bullets(items, o = {}) {
  return items.map((t) => new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 40, line: 240 },
    children: runs(t, { size: o.size ?? 20 }),
  }));
}

function imageParagraph(file, widthIn, heightIn, o = {}) {
  const data = fs.readFileSync(file);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: o.before ?? 60, after: o.after ?? 40 },
    keepNext: true,
    children: [new ImageRun({
      type: path.extname(file).slice(1).replace("jpg", "jpg"),
      data, transformation: { width: Math.round(widthIn * 96), height: Math.round(heightIn * 96) },
      altText: { title: o.title ?? "figure", description: o.description ?? "figure", name: o.title ?? "figure" },
    })],
  });
}

const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

// Booktabs-like table: rules above/below the header and at the foot, no vertical lines.
function table(columns, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const size = o.size ?? 16;
  const cell = (text, i, kind, rowIndex) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 25, bottom: 25, left: 55, right: 55 },
    borders: {
      top: kind === "head" && rowIndex === 0 ? thin : none,
      bottom: kind === "head" ? thin : (kind === "last" ? thin : none),
      left: none, right: none,
    },
    shading: o.stripe && kind === "body" && rowIndex % 2 === 1
      ? { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" } : undefined,
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : (o.align?.[i] ?? AlignmentType.RIGHT),
      spacing: { after: 0, line: 220 },
      children: runs(String(text), { size, bold: kind === "head" }),
    })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: columns.map((c, i) => cell(c, i, "head", 0)) }),
      ...rows.map((r, ri) => new TableRow({
        cantSplit: true,
        children: r.map((c, i) => cell(c, i, ri === rows.length - 1 ? "last" : "body", ri)),
      })),
    ],
  });
}

function caption(text, o = {}) {
  return new Paragraph({
    alignment: o.align ?? AlignmentType.CENTER,
    spacing: { before: o.before ?? 40, after: o.after ?? 100, line: 220 },
    keepLines: true,
    children: runs(text, { size: o.size ?? 16 }),
  });
}

const numbering = {
  config: [{
    reference: "bullets",
    levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 300, hanging: 200 } } } }],
  }],
};

function fmt(x, d = 4) { return Number(x).toFixed(d); }
function pm(mean, sd, d = 4) { return `${fmt(mean, d)} ± ${fmt(sd, d)}`; }
function pct(x, d = 1) { return `${(100 * x).toFixed(d)}%`; }
function signedPct(x, d = 1) { return `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(d)}%`; }

module.exports = { FONT, MONO, runs, para, bullets, imageParagraph, table, caption, numbering,
  fmt, pm, pct, signedPct, AlignmentType, HeadingLevel };
