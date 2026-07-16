import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import type { Direction, GraphLinkOut, TimelinePoint } from "./api";
import { SERIES } from "./palette";
import { useDarkMode } from "./useDarkMode";

const SURFACE = { light: "#f6f7f9", dark: "#16181d" };
const LINK = { light: "#c6ccd6", dark: "#3a4150" };
const ACCENT = { light: "#2563eb", dark: "#6ea8fe" };

interface MapNode {
  id: string;
  title: string;
  di: number; // direction index, the color slot
  degree: number;
  month: string;
  x?: number;
  y?: number;
}

// The whole-topic map: every landscape paper, colored by direction, sized by how connected
// it is within the topic set (stored similarity edges, never citations). Clicking selects;
// the detail strip below keeps the selection readable without leaving the map.
export default function ResearchMap({
  directions,
  links,
  selected,
  onSelect,
}: {
  directions: Direction[];
  links: GraphLinkOut[];
  selected: string | null;
  onSelect: (arxivId: string | null) => void;
}) {
  const dark = useDarkMode();
  const mode = dark ? "dark" : "light";
  const colors = SERIES[mode];
  const fgRef = useRef<ForceGraphMethods | undefined>(undefined);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const data = useMemo(() => {
    const nodes: MapNode[] = [];
    const byId = new Map<string, MapNode>();
    directions.forEach((d, di) =>
      d.papers.forEach((p) => {
        const node: MapNode = {
          id: p.arxiv_id,
          title: p.title,
          di,
          degree: 0,
          month: p.published_month,
        };
        nodes.push(node);
        byId.set(p.arxiv_id, node);
      }),
    );
    const kept = links.filter((l) => byId.has(l.source) && byId.has(l.target));
    for (const l of kept) {
      byId.get(l.source)!.degree += 1;
      byId.get(l.target)!.degree += 1;
    }
    // force-graph mutates link endpoints into node references; give it copies
    return { nodes, links: kept.map((l) => ({ ...l })) };
  }, [directions, links]);

  // Center on the selection, whether it came from a node click or a paper-list click.
  useEffect(() => {
    if (!selected) return;
    const node = data.nodes.find((n) => n.id === selected);
    if (node && node.x != null && node.y != null) {
      fgRef.current?.centerAt(node.x, node.y, 500);
    }
  }, [selected, data]);

  const chosen = selected ? data.nodes.find((n) => n.id === selected) : undefined;

  return (
    <section className="card map-card">
      <h2>Research map</h2>
      <p className="hint">
        Every paper in the landscape, colored by direction and sized by how connected it is
        within the topic. Links are semantic similarity, not citations. Click a paper.
      </p>
      <div className="legend">
        {directions.map((d, di) => (
          <span key={d.name} className="legend-item">
            <span className="swatch" style={{ background: colors[di % colors.length] }} />
            {d.name}
          </span>
        ))}
      </div>
      <div className="graph-well" ref={wrapRef}>
        {width > 0 && (
          <ForceGraph2D
            ref={fgRef}
            width={width}
            height={430}
            graphData={data}
            backgroundColor={SURFACE[mode]}
            nodeLabel={(node) => {
              const n = node as MapNode;
              return `<div class="graph-tip">${esc(n.title)}<br/><small>${esc(
                directions[n.di]?.name ?? "",
              )} &middot; ${esc(n.month)}</small></div>`;
            }}
            linkColor={() => LINK[mode]}
            linkWidth={(link) => 0.5 + ((link as GraphLinkOut).weight - 0.6) * 4}
            nodeCanvasObject={(node, ctx) => {
              const n = node as Required<MapNode>;
              const r = 3.5 + Math.min(n.degree, 10) * 0.35;
              if (n.id === selected) {
                ctx.beginPath();
                ctx.arc(n.x, n.y, r + 3, 0, 2 * Math.PI);
                ctx.strokeStyle = ACCENT[mode];
                ctx.lineWidth = 2;
                ctx.stroke();
              }
              ctx.beginPath();
              ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
              ctx.fillStyle = colors[n.di % colors.length];
              ctx.fill();
              // surface ring so overlapping marks stay separable
              ctx.lineWidth = 1.5;
              ctx.strokeStyle = SURFACE[mode];
              ctx.stroke();
            }}
            nodePointerAreaPaint={(node, color, ctx) => {
              const n = node as Required<MapNode>;
              ctx.beginPath();
              ctx.arc(n.x, n.y, 11, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
            onNodeClick={(node) => {
              const id = (node as MapNode).id;
              onSelect(id === selected ? null : id);
            }}
            onBackgroundClick={() => onSelect(null)}
            cooldownTicks={90}
          />
        )}
      </div>
      {chosen ? (
        <div className="map-detail">
          <span className="swatch" style={{ background: colors[chosen.di % colors.length] }} />
          <div className="map-detail-main">
            <a href={`https://arxiv.org/abs/${chosen.id}`} target="_blank" rel="noreferrer">
              {chosen.title}
            </a>
            <span className="map-detail-meta">
              {directions[chosen.di]?.name} &middot; {chosen.month} &middot; {chosen.degree}{" "}
              connection{chosen.degree === 1 ? "" : "s"} in this landscape
            </span>
          </div>
        </div>
      ) : (
        <p className="hint">
          {data.nodes.length} papers &middot; {data.links.length} similarity links
        </p>
      )}
    </section>
  );
}

// force-graph renders nodeLabel as HTML, and titles come from the corpus; escape them.
function esc(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Stacked bars, papers per month per direction: answers "where is the activity flowing".
// Plain SVG in a fixed viewBox; segment identity is doubled by the shared map legend and
// each segment's native tooltip, so color is never the only carrier.
const CHART_W = 640;
const CHART_H = 150;
const LABEL_H = 18;
const GAP = 2;

export function Timeline({
  timeline,
  directions,
}: {
  timeline: TimelinePoint[];
  directions: Direction[];
}) {
  const dark = useDarkMode();
  const colors = SERIES[dark ? "dark" : "light"];
  const months = [...new Set(timeline.map((t) => t.month))].sort();
  const names = directions.map((d) => d.name);
  const byKey = new Map(timeline.map((t) => [`${t.month}|${t.direction}`, t.count]));
  const totals = months.map((m) =>
    names.reduce((sum, n) => sum + (byKey.get(`${m}|${n}`) ?? 0), 0),
  );
  const maxTotal = Math.max(...totals, 1);
  const plotH = CHART_H - LABEL_H;
  const slot = CHART_W / months.length;
  const barW = Math.min(40, slot * 0.6);
  // With many months, label every other tick so labels never collide.
  const labelEvery = months.length > 8 ? 2 : 1;

  return (
    <section className="card">
      <h2>Activity by month</h2>
      <figure className="timeline">
        <svg
          viewBox={`0 0 ${CHART_W} ${CHART_H}`}
          role="img"
          aria-label="Papers per month, stacked by research direction"
        >
          {months.map((month, mi) => {
            const x = mi * slot + (slot - barW) / 2;
            let y = plotH;
            return (
              <g key={month}>
                {names.map((name, ni) => {
                  const count = byKey.get(`${month}|${name}`) ?? 0;
                  if (count === 0) return null;
                  const h = Math.max((count / maxTotal) * (plotH - 8) - GAP, 1.5);
                  y -= h + GAP;
                  return (
                    <rect
                      key={name}
                      x={x}
                      y={y}
                      width={barW}
                      height={h}
                      rx={2}
                      fill={colors[ni % colors.length]}
                    >
                      <title>{`${name}: ${count} paper${count === 1 ? "" : "s"} in ${month}`}</title>
                    </rect>
                  );
                })}
                {mi % labelEvery === 0 && (
                  <text
                    className="tick"
                    x={mi * slot + slot / 2}
                    y={CHART_H - 4}
                    textAnchor="middle"
                  >
                    {month}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
        <figcaption className="hint">
          Colors match the research map; hover a bar segment for its direction and count.
        </figcaption>
      </figure>
    </section>
  );
}
