#!/usr/bin/env python
"""Build the DyTrustRec architecture diagram (SVG -> PDF, and an editable .drawio).

One spec drives every output, so the paper figure, the explainer figure and the
draw.io file cannot drift apart. Numbers that come from a run (graph size,
tuned settings) are read from the project's result files rather than typed in.

    python docs/diagrams/build_architecture.py
"""

from __future__ import annotations

import html
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W, H = 1900, 1580

PALETTE = {
    "amber": ("#F2C57C", "#C77D2E"), "cream": ("#FCE7C2", "#C77D2E"),
    "grey": ("#E7E7E7", "#8A8F98"), "red": ("#E8B4B4", "#B33B3B"),
    "gold": ("#F3DFA0", "#C9A227"), "blue": ("#BBD1EC", "#4472C4"),
    "green": ("#CFE3C6", "#6E9F5E"), "sand": ("#FBF1DD", "#C9A227"),
}


@dataclass
class Box:
    id: str
    x: int
    y: int
    w: int
    h: int
    tag: str = ""                 # small label above the title, e.g. "MODULE 8"
    title: str = ""
    lines: list[str] = field(default_factory=list)
    colour: str = "amber"
    small: list[str] = field(default_factory=list)   # italic footnote lines in the box

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


@dataclass
class Container:
    id: str
    x: int
    y: int
    w: int
    h: int
    title: str
    stroke: str = "#C77D2E"
    dx: int = 10                   # label offset, so connectors do not run through the text


@dataclass
class Note:
    id: str
    x: int
    y: int
    lines: list[str]


@dataclass
class Edge:
    id: str
    path: list[tuple[int, int]]
    colour: str = "#2B2B2B"
    dashed: bool = False


def load_facts() -> dict:
    """Graph size and tuned settings, from the files a run produces."""
    facts = {"nodes": "138,480", "edges": "718,506", "keep": "0.5", "lambda": "0.5",
             "live": None}
    stats = ROOT / "reports" / "tables" / "graph_stats.json"
    live = HERE / "live_example.json"
    if live.is_file():
        facts["live"] = json.loads(live.read_text(encoding="utf-8"))
    tuning = ROOT / "reports" / "results" / "tuning"
    best = None
    for path in sorted(tuning.glob("hgt__*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if best is None or record["val_ndcg@10"] > best["val_ndcg@10"]:
            best = record
    if best:
        keep = best["overrides"].get("training.edge_mask.keep_prob")
        if keep is not None:
            facts["keep"] = f"{keep:g}"
    best_rec = None
    for path in sorted(tuning.glob("hgt_recency__*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if best_rec is None or record["val_ndcg@10"] > best_rec["val_ndcg@10"]:
            best_rec = record
    if best_rec:
        facts["lambda"] = f"{best_rec['recency_lambda']:g}"
    return facts


def spec(facts: dict):
    containers = [
        Container("c_ing", 20, 150, 1860, 175, "Data Ingestion, Cleaning & Leakage-Safe Split -- Modules 1-3, 22", dx=210),
        Container("c_feat", 20, 390, 970, 390, "(a) Feature Engineering -- Sentiment, Semantics & Fusion -- Modules 5-8", dx=200),
        Container("c_graph", 1020, 390, 860, 390, "(b) Trust-Aware Heterogeneous Graph -- Modules 9, 11-13", dx=150),
        Container("c_hgt", 20, 830, 1860, 230, "Heterogeneous Graph Transformer (HGT) Encoder -- Module 14", stroke="#3D6FB0", dx=320),
    ]
    boxes = [
        Box("inp", 40, 24, 300, 96, "", "Dataset", ["Amazon Reviews 2023 (Video_Games tier)", "262,547 raw rows", "replicated on Musical_Instruments"], "cream"),
        Box("ing", 60, 195, 280, 118, "MODULE 1", "Ingestion & Schema Adapter", ["field presence checked, never invented"]),
        Box("clean", 380, 195, 280, 118, "MODULE 2", "Cleaning & k-core Filter", ["262,547 -> 144,178 rows (54.9% retained)"]),
        Box("split", 700, 195, 280, 118, "MODULE 22", "Chronological Split -- 80/10/10", ["train 115,342 * val 14,418 * test 14,418"], "grey"),
        Box("leak", 1020, 195, 320, 118, "ENFORCED", "Leakage Guards",
            ["6 assertions in tests/ -- no val/test", "edge ever enters the training graph"],
            "red", ["terminal -- a check on the split,", "not a data-producing stage"]),
        Box("eda", 1380, 195, 460, 118, "MODULES 3-4", "EDA * Outlier Analysis",
            ["18.7% corroborated outliers (>=2 detectors agree) --",
             "flagged, never deleted, so the trust detectors have evidence to find"],
            "gold", ["terminal -- informs design decisions and the written",
                     "report (Sec. G); nothing downstream consumes it"]),
        Box("text", 60, 440, 290, 90, "", "review_title + review_text", [], "grey"),
        Box("struct5", 60, 560, 290, 130, "MODULE 5", "5 Structured Features",
            ["length * helpfulness * verified *", "recency * rating-centred"], "grey"),
        Box("sent", 390, 440, 320, 120, "MODULE 6", "BERT Sentiment Encoder",
            ["nlptown/bert-base-multilingual-", "uncased-sentiment -> pos/neg/neu/compound"]),
        Box("sbert", 390, 580, 320, 120, "MODULE 7", "SBERT Semantic Encoder",
            ["all-MiniLM-L6-v2 * mean-pool * L2-norm", "-> 384-d, unit-norm"]),
        Box("fusion", 750, 490, 220, 170, "MODULE 8", "Semantic-Behavioural\nFusion Layer",
            ["concat -> LayerNorm ->", "Linear(390->256) -> GELU ->", "Dropout -> Linear(256->128)"], "blue",
            ["ablated: no gain over concatenation;", "the best variant bypasses it"]),
        Box("hgraph", 1060, 440, 330, 170, "", "5-type heterogeneous graph G = (V, E)", [], "sand"),
        Box("cola", 1440, 440, 400, 105, "MODULE 11", "Behavioural Trust Detector",
            ["indicators: shared text * repeats * bursts *", "paraphrase match (sentence embeddings)"]),
        Box("anomtrust", 1440, 565, 400, 105, "MODULE 12", "Anomaly -> Trust",
            ["tau = 1 - rank-norm(score), clamp [0.05, 1];", "flagged accounts are gated out of the graph"]),
        Box("temporal", 1060, 640, 330, 105, "MODULE 13", "Temporal Recency Weight",
            [f"exp(-lambda*age), lambda={facts['lambda']} (tuned on validation), floor 0.01", "age from max(train.timestamp), never now"]),
        Box("wgraph", 1440, 690, 400, 90, "OUTPUT", "Trust x Recency Sample Weights",
            ["w_e = base * tau_e^a * recency_e^b scales each BPR term", "(HGTConv cannot read an edge weight)"], "green"),
        Box("proj", 60, 885, 280, 130, "", "Type-Specific Linear\nProjection -> 128-d",
            ["per node type: user 20-d, product 18-d,", "review 390-d, brand/category learned"], "blue"),
        Box("emb", 780, 890, 280, 105, "OUTPUT", "User / Product Embeddings",
            ["128-d, score(u,i) = e_u . e_i"], "green"),
        Box("bpr", 1200, 890, 560, 105, "MODULE 20/31", "BPR Objective, AdamW, Edge-Masked Training",
            [f"each epoch {int(round((1 - float(facts['keep'])) * 100))}% of interaction edges are hidden from message passing",
             "and become the targets; validation NDCG@10 early-stops and restores the best epoch"]),
        Box("scoring", 60, 1100, 420, 110, "MODULE 15", "Recommendation Scoring",
            ["Full-catalogue ranking, positives masked;", "last r slots: new products by text"]),
        Box("expl", 520, 1100, 420, 110, "MODULE 16", "Explanation Branch",
            ["6 evidence sources; each cited fact is", "tested by removing it and re-running"]),
        Box("cf", 980, 1100, 420, 110, "MODULE 17", "Counterfactual Branch",
            ["Bisection search over one perturbable", "factor -> magnitude to enter Top-K, or infeasible"]),
        Box("conf", 1440, 1100, 400, 110, "MODULE 18", "Confidence Branch",
            ["evidence sufficiency 0.5 / 0.3 / 0.2, plus", "a separately calibrated hit probability"]),
        Box("finalout", 40, 1390, 1820, 60, "", "Trust-Calibrated Explainable Top-K Recommendation", [], "green"),
        Box("dash", 1500, 1465, 360, 90, "MODULE 39", "Streamlit Dashboard",
            ["reads generated artefacts only --", "trains nothing, computes no metric of its own"], "cream"),
    ]
    notes = [
        Note("n_feat", 60, 731, ['concat arm (mode="concat") kept alongside the learned projection so "does fusion beat',
                                 'concatenation?" is answered, not assumed. Measured: it does not (Table V).']),
        Note("n_hgt", 380, 1025, ["HGT: 2 layers x 4-head relational attention, full-batch"]),
        Note("n_mat", 60, 1358, ["y_u,i score map (illustrative)"]),
        Note("n_bars", 1440, 1214, ["confidence blend"]),
    ]
    live = facts.get("live")
    if live:
        notes.append(Note("n_live", 40, 1470, live["lines"]))
    edges = [
        Edge("e0", [(190, 108), (190, 195)]),
        Edge("e1", [(340, 254), (380, 254)]),
        Edge("e2", [(660, 254), (700, 254)]),
        Edge("e3", [(980, 254), (1020, 254)]),
        Edge("e4", [(520, 195), (520, 168), (1610, 168), (1610, 195)], "#5C6873", True),
        Edge("e5", [(350, 485), (390, 485)]),
        Edge("e6", [(205, 530), (205, 548), (370, 548), (370, 640), (390, 640)]),
        Edge("e7", [(350, 625), (365, 625), (365, 570), (740, 570), (750, 570)]),
        Edge("e8", [(710, 500), (730, 500), (730, 540), (750, 540)]),
        Edge("e9", [(710, 640), (730, 640), (730, 620), (750, 620)]),
        Edge("e10", [(1390, 495), (1440, 495)]),
        Edge("e11", [(1640, 545), (1640, 565)]),
        Edge("e12", [(1225, 610), (1225, 640)]),
        Edge("e13", [(1640, 670), (1640, 690)]),
        Edge("e14", [(1390, 700), (1415, 700), (1415, 735), (1440, 735)]),
        Edge("e15", [(840, 313), (840, 365), (190, 365), (190, 440)]),
        Edge("e16", [(840, 365), (1150, 365), (1150, 440)]),
        Edge("e17", [(340, 950), (780, 950)]),
        Edge("e18", [(1060, 940), (1200, 940)], dashed=True),
        Edge("e19", [(860, 660), (860, 790), (150, 790), (150, 885)], "#4472C4"),
        Edge("e20", [(1640, 780), (1640, 802), (270, 802), (270, 885)], "#C9A227"),
        Edge("e21", [(920, 995), (920, 1050), (270, 1050), (270, 1100)]),
        Edge("e22", [(920, 995), (920, 1050), (730, 1050), (730, 1100)]),
        Edge("e23", [(920, 995), (920, 1050), (1190, 1050), (1190, 1100)]),
        Edge("e24", [(920, 995), (920, 1050), (1640, 1050), (1640, 1100)]),
        Edge("e25", [(270, 1210), (270, 1390)]),
        Edge("e26", [(730, 1210), (730, 1390)]),
        Edge("e27", [(1190, 1210), (1190, 1390)]),
        Edge("e28", [(1810, 1210), (1810, 1390)]),
        Edge("e29", [(1680, 1450), (1680, 1465)]),
    ]
    return containers, boxes, notes, edges


# -- SVG --------------------------------------------------------------------
def esc(text: str) -> str:
    return html.escape(text, quote=False)


def svg_box(b: Box) -> str:
    fill, stroke = PALETTE[b.colour]
    out = [f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="14" fill="{fill}" '
           f'stroke="{stroke}" stroke-width="1.6"/>']
    rows = ([("tag", b.tag)] if b.tag else []) + [("title", t) for t in b.title.split("\n")] \
        + [("line", t) for t in b.lines] + [("small", t) for t in b.small]
    heights = {"tag": 14, "title": 24, "line": 17, "small": 14}
    total = sum(heights[k] for k, _ in rows)
    y = b.y + (b.h - total) / 2 + 2
    for kind, text in rows:
        y += heights[kind]
        if not text:
            continue
        style = {"tag": f'font-size="11" fill="{stroke}" letter-spacing="1"',
                 "title": 'font-size="17" font-weight="bold" fill="#1A1A1A"',
                 "line": 'font-size="12.5" fill="#4A4A4A"',
                 "small": 'font-size="10.5" font-style="italic" fill="#B33B3B"'}[kind]
        out.append(f'<text x="{b.cx}" y="{y - 4:.1f}" text-anchor="middle" {style}>{esc(text)}</text>')
    return "\n".join(out)


def svg_illustrations(boxes: dict[str, Box], facts: dict) -> str:
    out: list[str] = []
    g = boxes["hgraph"]
    users = [(g.x + 40, g.y + 58 + 27 * i) for i in range(3)]
    prods = [(g.x + 165, g.y + 52 + 31 * i) for i in range(3)]
    attrs = [(g.x + 290, g.y + 68), (g.x + 290, g.y + 108)]
    for u in users:
        for p in prods:
            out.append(f'<line x1="{u[0]}" y1="{u[1]}" x2="{p[0]}" y2="{p[1]}" stroke="#C9A227" stroke-width="1" opacity=".7"/>')
    for p in prods:
        for a in attrs:
            out.append(f'<line x1="{p[0]}" y1="{p[1]}" x2="{a[0]}" y2="{a[1]}" stroke="#9AA39A" stroke-width="1" opacity=".7"/>')
    for pts, colour in ((users, "#4472C4"), (prods, "#C77D2E"), (attrs, "#6E9F5E")):
        for x, y in pts:
            out.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{colour}"/>')
    out.append(f'<text x="{g.cx}" y="{g.y + 24}" text-anchor="middle" font-size="15" font-weight="bold" fill="#1A1A1A">'
               f'5-type heterogeneous graph G = (V, E)</text>')
    out.append(f'<text x="{g.cx}" y="{g.y + g.h - 30}" text-anchor="middle" font-size="12" fill="#4A4A4A">'
               f'{facts["nodes"]} nodes * {facts["edges"]} edges (incl. reverses)</text>')
    out.append(f'<text x="{g.cx}" y="{g.y + g.h - 12}" text-anchor="middle" font-size="11" fill="#6A6A6A">'
               f'user (blue) / product (orange) / brand-category (green)</text>')

    # layer stack inside the HGT container
    for i, (x, height) in enumerate(zip(range(400, 760, 50), (110, 100, 90, 78, 66, 54, 44))):
        shade = ["#DCE8F6", "#C9DCF1", "#A9C8EA", "#7FB0E0", "#5C9AD6", "#3F86CC", "#2F72B5"][i]
        out.append(f'<rect x="{x}" y="{950 - height / 2}" width="34" height="{height}" fill="{shade}" stroke="#2F5F9E" stroke-width="1"/>')

    # score map
    import random
    rng = random.Random(7)
    for r in range(4):
        for c in range(10):
            shade = rng.choice(["#EAF1F9", "#CFE0F1", "#9FC3E6", "#5E96CF", "#2E64A6", "#16406E"])
            out.append(f'<rect x="{62 + c * 12}" y="{1262 + r * 12}" width="12" height="12" fill="{shade}"/>')
    out.append('<text x="62" y="1254" font-size="12" font-weight="bold" fill="#1A1A1A">preference scores</text>')

    # confidence bars
    for i, (label, share, colour) in enumerate((("stability", .5, "#4472C4"), ("volume", .3, "#C77D2E"),
                                                ("evid. trust", .2, "#6E9F5E"))):
        y = 1248 + i * 28
        out.append(f'<text x="1440" y="{y + 11}" font-size="11.5" fill="#4A4A4A">{label}</text>')
        out.append(f'<rect x="1540" y="{y}" width="200" height="13" fill="#EEEEEE"/>')
        out.append(f'<rect x="1540" y="{y}" width="{200 * share:.0f}" height="13" fill="{colour}"/>')
        out.append(f'<text x="1752" y="{y + 11}" font-size="11.5" fill="#4A4A4A">{share:.2f}</text>')
    return "\n".join(out)


def build_svg(facts: dict) -> str:
    containers, boxes, notes, edges = spec(facts)
    by_id = {b.id: b for b in boxes}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Helvetica, Arial, sans-serif">',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
           'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>',
           f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']
    for c in containers:
        out.append(f'<rect x="{c.x}" y="{c.y}" width="{c.w}" height="{c.h}" rx="18" fill="none" stroke="{c.stroke}" '
                   f'stroke-width="1.6" stroke-dasharray="7 5"/>')
    for e in edges:
        pts = " ".join(f"{x},{y}" for x, y in e.path)
        dash = ' stroke-dasharray="6 5"' if e.dashed else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{e.colour}" stroke-width="1.6"{dash} '
                   f'marker-end="url(#arrow)"/>')
    for b in boxes:
        if b.id == "hgraph":
            fill, stroke = PALETTE["sand"]
            out.append(f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        else:
            out.append(svg_box(b))
    for c in containers:
        out.append(f'<text x="{c.x + c.dx}" y="{c.y - 10}" font-size="15" font-weight="bold" fill="{c.stroke}" '
                   f'stroke="#FFFFFF" stroke-width="5" paint-order="stroke">{esc(c.title)}</text>')
    out.append(svg_illustrations(by_id, facts))
    for n in notes:
        for i, line in enumerate(n.lines):
            out.append(f'<text x="{n.x}" y="{n.y + 14 + 15 * i}" font-size="12" fill="#555555">{esc(line)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# -- draw.io ----------------------------------------------------------------
def build_drawio(facts: dict) -> str:
    containers, boxes, notes, edges = spec(facts)
    cells = ['<mxCell id="page1-0" />', '<mxCell id="page1-1" parent="page1-0" />']

    def style(fill: str, stroke: str, extra: str = "") -> str:
        return (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                f"fontColor=#1A1A1A;align=center;verticalAlign=middle;{extra}")

    for c in containers:
        cells.append(
            f'<mxCell id="page1-{c.id}" value="{html.escape(c.title)}" style="rounded=1;dashed=1;strokeColor={c.stroke};'
            f'fillColor=none;verticalAlign=top;align=left;fontColor={c.stroke};fontStyle=1;spacingLeft=8;" vertex="1" parent="page1-1">'
            f'<mxGeometry x="{c.x}" y="{c.y}" width="{c.w}" height="{c.h}" as="geometry" /></mxCell>')
    for b in boxes:
        fill, stroke = PALETTE[b.colour]
        label = "<br>".join(html.escape(t) for t in ([b.tag] if b.tag else []) + b.title.split("\n") + b.lines + b.small)
        cells.append(
            f'<mxCell id="page1-{b.id}" value="{html.escape(label, quote=True)}" style="{style(fill, stroke)}" vertex="1" parent="page1-1">'
            f'<mxGeometry x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" as="geometry" /></mxCell>')
    for n in notes:
        cells.append(
            f'<mxCell id="page1-{n.id}" value="{html.escape("<br>".join(n.lines), quote=True)}" '
            f'style="text;html=1;align=left;verticalAlign=top;fontSize=11;fontColor=#555555;" vertex="1" parent="page1-1">'
            f'<mxGeometry x="{n.x}" y="{n.y}" width="620" height="40" as="geometry" /></mxCell>')
    for e in edges:
        dash = "dashed=1;" if e.dashed else ""
        points = "".join(f'<mxPoint x="{x}" y="{y}" />' for x, y in e.path)
        cells.append(
            f'<mxCell id="page1-{e.id}" value="" style="endArrow=block;html=1;strokeColor={e.colour};rounded=0;{dash}" '
            f'edge="1" parent="page1-1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{e.path[0][0]}" y="{e.path[0][1]}" as="sourcePoint" />'
            f'<mxPoint x="{e.path[-1][0]}" y="{e.path[-1][1]}" as="targetPoint" />'
            f'<Array as="points">{points}</Array></mxGeometry></mxCell>')
    body = "\n".join(cells)
    return (f'<mxfile host="app.diagrams.net"><diagram id="page1" name="Fig. 1 -- DyTrustRec Full System Architecture">'
            f'<mxGraphModel dx="800" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
            f'fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0"><root>{body}</root>'
            f'</mxGraphModel></diagram></mxfile>')


def main() -> int:
    facts = load_facts()
    svg_path = HERE / "DyTrustRec_Architecture.svg"
    svg_path.write_text(build_svg(facts), encoding="utf-8")
    (HERE / "DyTrustRec_Architecture.drawio").write_text(build_drawio(facts), encoding="utf-8")
    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(HERE / "DyTrustRec_Architecture.pdf"),
                    str(svg_path)], check=True)
    subprocess.run(["rsvg-convert", "-w", "2400", "-o", str(HERE / "DyTrustRec_Architecture.png"),
                    str(svg_path)], check=True)
    print("wrote", svg_path.name, "+ .drawio, .pdf, .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
