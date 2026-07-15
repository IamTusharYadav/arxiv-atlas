import type { GraphResponse } from "./api";

// The plan's ceiling: expansion stops here so the canvas stays legible and the layout stays
// responsive; the note below the graph says so when it bites.
export const NODE_CAP = 75;

export interface GNode {
  id: string;
  title: string;
  category: string;
  root?: boolean;
  x?: number;
  y?: number;
}

export interface GLink {
  source: string | GNode;
  target: string | GNode;
  weight: number;
}

export interface GraphData {
  nodes: GNode[];
  links: GLink[];
}

// force-graph rewrites link endpoints from ids to node objects once the layout runs.
export function idOf(end: string | GNode): string {
  return typeof end === "string" ? end : end.id;
}

function linkKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

// Merge a neighborhood into the existing graph, reusing node objects so the running layout
// keeps its positions. Links to nodes dropped by the cap are dropped with them; edges are
// undirected, so a link seen from either end is the same link.
export function merge(
  current: GraphData,
  resp: GraphResponse,
  cap: number = NODE_CAP,
): { data: GraphData; capped: boolean } {
  const byId = new Map(current.nodes.map((n) => [n.id, n]));
  let capped = false;
  for (const n of resp.nodes) {
    if (byId.has(n.arxiv_id)) continue;
    if (byId.size >= cap) {
      capped = true;
      break;
    }
    byId.set(n.arxiv_id, { id: n.arxiv_id, title: n.title, category: n.primary_category });
  }
  const links = new Map(current.links.map((l) => [linkKey(idOf(l.source), idOf(l.target)), l]));
  for (const l of resp.links) {
    if (!byId.has(l.source) || !byId.has(l.target)) continue;
    const key = linkKey(l.source, l.target);
    if (!links.has(key)) links.set(key, { source: l.source, target: l.target, weight: l.weight });
  }
  return { data: { nodes: [...byId.values()], links: [...links.values()] }, capped };
}
