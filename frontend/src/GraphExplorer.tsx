import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { getGraph } from "./api";
import { merge, NODE_CAP, type GLink, type GNode, type GraphData } from "./graphMerge";

// Categorical palette, slots assigned in fixed order to the corpus categories; anything else
// folds into muted ink rather than minting a fourth hue. Both modes validated against the
// app's --bg surfaces with the dataviz six-checks script (CVD, lightness, chroma, contrast).
const CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"] as const;
const SERIES = {
  light: { "cs.AI": "#2a78d6", "cs.LG": "#008300", "cs.CL": "#d55181", other: "#5b6572" },
  dark: { "cs.AI": "#3987e5", "cs.LG": "#008300", "cs.CL": "#d55181", other: "#9aa3b2" },
};
const SURFACE = { light: "#f6f7f9", dark: "#16181d" };
const LINK = { light: "#c6ccd6", dark: "#3a4150" };

// force-graph renders nodeLabel as HTML, and titles come from the corpus; escape them.
function esc(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function useDarkMode(): boolean {
  const [dark, setDark] = useState(
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return dark;
}

export default function GraphExplorer({ rootId }: { rootId: string }) {
  const dark = useDarkMode();
  const mode = dark ? "dark" : "light";
  const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
  const [capped, setCapped] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const expanded = useRef(new Set<string>());
  // Bumped when the root changes so a fetch still in flight for the old root is discarded
  // instead of merging into the new graph.
  const generation = useRef(0);
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

  const expand = useCallback(async (id: string, root = false) => {
    if (expanded.current.has(id)) return;
    expanded.current.add(id);
    const gen = generation.current;
    try {
      const resp = await getGraph(id);
      if (gen !== generation.current) return;
      setData((current) => {
        const seeded: GraphData = root ? { nodes: [], links: [] } : current;
        const { data: next, capped: hitCap } = merge(seeded, resp);
        if (root) {
          const rootNode = next.nodes.find((n) => n.id === id);
          if (rootNode) rootNode.root = true;
        }
        if (hitCap) setCapped(true);
        return next;
      });
    } catch (err) {
      if (gen !== generation.current) return;
      expanded.current.delete(id); // allow a retry click
      setError(err instanceof Error ? err.message : "could not load the neighborhood");
    }
  }, []);

  useEffect(() => {
    generation.current += 1;
    expanded.current = new Set();
    setCapped(false);
    setError(null);
    void expand(rootId, true);
  }, [rootId, expand]);

  const colors = SERIES[mode];
  const present = useMemo(() => {
    const cats = new Set(data.nodes.map((n) => n.category));
    const known = CATEGORIES.filter((c) => cats.has(c));
    const hasOther = data.nodes.some(
      (n) => !(CATEGORIES as readonly string[]).includes(n.category),
    );
    return hasOther ? [...known, "other"] : known;
  }, [data.nodes]);

  const colorFor = (category: string): string =>
    colors[category as keyof typeof colors] ?? colors.other;

  return (
    <section className="card graph">
      <h2>Neighborhood of {rootId}</h2>
      <p className="hint">
        Click a node to pull in its semantic neighbors. Edge length is layout only; edge width
        tracks similarity.
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
            height={440}
            graphData={data}
            backgroundColor={SURFACE[mode]}
            nodeLabel={(node) => {
              const n = node as GNode;
              return `<div class="graph-tip">${esc(n.title)}<br/><small>${esc(n.id)} &middot; ${esc(n.category)}</small></div>`;
            }}
            linkColor={() => LINK[mode]}
            linkWidth={(link) => 1 + ((link as GLink).weight - 0.6) * 5}
            nodeCanvasObject={(node, ctx) => {
              const n = node as Required<GNode>;
              const r = n.root ? 8 : 5.5;
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
              const n = node as Required<GNode>;
              // hit target larger than the mark
              ctx.beginPath();
              ctx.arc(n.x, n.y, 12, 0, 2 * Math.PI);
              ctx.fillStyle = color;
              ctx.fill();
            }}
            onNodeClick={(node) => void expand((node as GNode).id)}
            cooldownTicks={90}
          />
        )}
      </div>
      <p className="hint">
        {data.nodes.length} papers shown
        {capped && <> &middot; node cap of {NODE_CAP} reached; expansion is paused</>}
      </p>
      {error && <p className="graph-error">{error}</p>}
    </section>
  );
}
