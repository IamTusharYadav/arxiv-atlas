import { describe, expect, it } from "vitest";
import type { GraphResponse } from "./api";
import { merge, type GraphData } from "./graphMerge";

function resp(center: string, neighbors: string[], weight = 0.8): GraphResponse {
  return {
    center,
    nodes: [center, ...neighbors].map((id) => ({
      arxiv_id: id,
      title: `title ${id}`,
      primary_category: "cs.AI",
    })),
    links: neighbors.map((n) => ({ source: center, target: n, weight })),
  };
}

const empty: GraphData = { nodes: [], links: [] };

describe("merge", () => {
  it("dedupes nodes and undirected links across expansions", () => {
    const first = merge(empty, resp("a", ["b", "c"])).data;
    // b's neighborhood repeats the a-b edge from the other direction
    const second = merge(first, {
      center: "b",
      nodes: [
        { arxiv_id: "b", title: "t", primary_category: "cs.AI" },
        { arxiv_id: "a", title: "t", primary_category: "cs.AI" },
        { arxiv_id: "d", title: "t", primary_category: "cs.LG" },
      ],
      links: [
        { source: "b", target: "a", weight: 0.8 },
        { source: "b", target: "d", weight: 0.7 },
      ],
    });
    expect(second.data.nodes.map((n) => n.id).sort()).toEqual(["a", "b", "c", "d"]);
    expect(second.data.links).toHaveLength(3); // a-b, a-c, b-d: the reversed a-b did not duplicate
    expect(second.capped).toBe(false);
  });

  it("keeps node object identity across merges so the layout is preserved", () => {
    const first = merge(empty, resp("a", ["b"])).data;
    const nodeB = first.nodes.find((n) => n.id === "b");
    const second = merge(first, resp("b", ["c"])).data;
    expect(second.nodes.find((n) => n.id === "b")).toBe(nodeB);
  });

  it("caps nodes and drops links that point past the cap", () => {
    const first = merge(empty, resp("a", ["b", "c"])).data; // 3 nodes
    const { data, capped } = merge(first, resp("c", ["d", "e", "f"]), 4);
    expect(capped).toBe(true);
    expect(data.nodes).toHaveLength(4); // a, b, c, d; e and f dropped
    // links only between surviving nodes: a-b, a-c, c-d
    expect(data.links).toHaveLength(3);
    for (const l of data.links) {
      const ids = data.nodes.map((n) => n.id);
      expect(ids).toContain(typeof l.source === "string" ? l.source : l.source.id);
      expect(ids).toContain(typeof l.target === "string" ? l.target : l.target.id);
    }
  });
});
