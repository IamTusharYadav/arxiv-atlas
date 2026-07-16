import { describe, expect, it } from "vitest";
import { closestDemo, DEMO_RUNS } from "./demo";

describe("demo runs", () => {
  it("ships real captured answers, each cited and traced", () => {
    expect(DEMO_RUNS.length).toBeGreaterThan(0);
    for (const run of DEMO_RUNS) {
      expect(run.result.papers.length).toBeGreaterThan(0);
      expect(run.result.trace.length).toBeGreaterThan(0);
      expect(run.result.cost_usd).toBeGreaterThan(0);
      // Never label a saved run as a live cache hit, and never as a capped partial.
      expect(run.result.cached).toBe(false);
      expect(run.result.partial).toBe(false);
      // The brief cites at least one of the papers it came back with.
      const cited = run.result.papers.some((p) => run.result.brief.includes(p.arxiv_id));
      expect(cited).toBe(true);
    }
  });

  it("picks the saved run that shares the most words with the question", () => {
    const target = DEMO_RUNS[DEMO_RUNS.length - 1];
    expect(closestDemo(target.question)).toBe(target);
  });

  it("falls back to a run rather than nothing when nothing matches", () => {
    expect(closestDemo("zzzz qqqq")).toBeDefined();
  });
});
