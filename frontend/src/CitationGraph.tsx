import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphLinkOut, PaperOut } from "./api";
import { useDarkMode } from "./useDarkMode";

// Same categorical palette and surfaces as GraphExplorer, so both graphs read as one system.
const CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"] as const;
const SERIES = {
  light: { "cs.AI": "#2a78d6", "cs.LG": "#008300", "cs.CL": "#d55181", other: "#5b6572" },
  dark: { "cs.AI": "#3987e5", "cs.LG": "#008300", "cs.CL": "#d55181", other: "#9aa3b2" },
};
const SURFACE = { light: "#f6f7f9", dark: "#16181d" };
const LINK = { light: "#c6ccd6", dark: "#3a4150" };
const ACCENT = { light: "#2563eb", dark: "#6ea8fe" };

// force-graph renders nodeLabel as HTML, and titles come from the corpus; escape them.
function esc(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

interface CiteNode {
  id: string;
  title: string;
  category: string;
  degree: number;
  x?: number;
  y?: number;
}

// A relationship map of just the papers cited in the answer, linked by semantic similarity
// (the same stored edges the landscape map uses). Nothing outside the citation set appears.
// Mirrors the research map: clicking a node selects it and fills the detail strip below rather
// than navigating away.
export default function CitationGraph({
  papers,
  links,
}: {
  papers: PaperOut[];
  links: GraphLinkOut[];
}) {
  const dark = useDarkMode();
  const mode = dark ? "dark" : "light";
  const colors = SERIES[mode];
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const data = useMemo(() => {
    const byId = new Map<string, CiteNode>();
    const nodes: CiteNode[] = papers.map((p) => {
      const node: CiteNode = {
        id: p.arxiv_id,
        title: p.title,
        category: p.primary_category,
        degree: 0,
      };
      byId.set(p.arxiv_id, node);
      return node;
    });
    const kept = links.filter((l) => byId.has(l.source) && byId.has(l.target));
    for (const l of kept) {
      byId.get(l.source)!.degree += 1;
      byId.get(l.target)!.degree += 1;
    }
    // force-graph mutates link endpoints into node references; give it copies.
    return { nodes, links: kept.map((l) => ({ ...l })) };
  }, [papers, links]);

  const colorFor = (category: string): string =>
    colors[category as keyof typeof colors] ?? colors.other;

  const present = useMemo(() => {
    const cats = new Set(data.nodes.map((n) => n.category));
    const known = CATEGORIES.filter((c) => cats.has(c));
    const hasOther = data.nodes.some((n) => !(CATEGORIES as readonly string[]).includes(n.category));
    return hasOther ? [...known, "other"] : known;
  }, [data.nodes]);

  const chosen = selected ? data.nodes.find((n) => n.id === selected) : undefined;

  return (
    <section className="card map-card">
      <h2>How the cited papers connect</h2>
      <p className="hint">
        Only the papers cited above, linked by semantic similarity (not citations). Click a paper
        for its details.
      </p>
      <div className="legend">
        {present.map((c) => (
          <span key={c} className="legend-item">
            <span className="swatch" style={{ background: colorFor(c) }} />
            {c}
          </span>
        ))}
      </div>
      <div className="graph-well" ref={wrapRef}>
        {width > 0 && (
          <ForceGraph2D
            width={width}
            height={360}
            graphData={data}
            backgroundColor={SURFACE[mode]}
            nodeLabel={(node) => {
              const n = node as CiteNode;
              return `<div class="graph-tip">${esc(n.title)}<br/><small>${esc(n.id)} &middot; ${esc(n.category)}</small></div>`;
            }}
            linkColor={() => LINK[mode]}
            linkWidth={(link) => 1 + ((link as GraphLinkOut).weight - 0.6) * 5}
            nodeCanvasObject={(node, ctx) => {
              const n = node as Required<CiteNode>;
              const r = 4 + Math.min(n.degree, 8) * 0.5;
              if (n.id === selected) {
                ctx.beginPath();
                ctx.arc(n.x, n.y, r + 3, 0, 2 * Math.PI);
                ctx.strokeStyle = ACCENT[mode];
                ctx.lineWidth = 2;
                ctx.stroke();
              }
              ctx.beginPath();
              ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
              ctx.fillStyle = colorFor(n.category);
              ctx.fill();
              // surface ring so overlapping marks stay separable
              ctx.lineWidth = 2;
              ctx.strokeStyle = SURFACE[mode];
              ctx.stroke();
            }}
            nodePointerAreaPaint={(node, color, ctx) => {
              const n = node as Required<CiteNode>;
              ctx.beginPath();
              ctx.arc(n.x, n.y, 12, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
            onNodeClick={(node) => {
              const id = (node as CiteNode).id;
              setSelected(id === selected ? null : id);
            }}
            onBackgroundClick={() => setSelected(null)}
            cooldownTicks={90}
          />
        )}
      </div>
      {chosen ? (
        <div className="map-detail">
          <span className="swatch" style={{ background: colorFor(chosen.category) }} />
          <div className="map-detail-main">
            <a href={`https://arxiv.org/abs/${chosen.id}`} target="_blank" rel="noreferrer">
              {chosen.title}
            </a>
            <span className="map-detail-meta">
              {chosen.category} &middot; {chosen.degree} connection
              {chosen.degree === 1 ? "" : "s"} among the cited papers
            </span>
          </div>
        </div>
      ) : (
        <p className="hint">
          {data.nodes.length} papers &middot; {data.links.length} similarity link
          {data.links.length === 1 ? "" : "s"}
          {data.links.length === 0 && " (these citations share no strong similarity edge)"}
        </p>
      )}
    </section>
  );
}
