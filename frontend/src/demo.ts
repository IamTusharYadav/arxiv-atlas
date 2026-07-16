import type { QueryResult } from "./api";
import runs from "./demoRuns.json";

// Real answers captured from the live API against the 98k corpus, not written by hand. They are
// only ever shown behind an explicit "saved example" banner, so a demo answer is never mistaken
// for a fresh one. Regenerate by capturing GET /api/query/{id} results (see docs).
export interface DemoRun {
  question: string;
  result: QueryResult;
}

export const DEMO_RUNS: DemoRun[] = runs as DemoRun[];

// Word overlap, not embeddings: the fallback only needs to beat "show the first one" when the
// visitor's question happens to resemble a saved one, and the client has no embedder.
function overlap(a: string, b: string): number {
  const words = (s: string) => new Set(s.toLowerCase().match(/[a-z]{4,}/g) ?? []);
  const left = words(a);
  const right = words(b);
  let shared = 0;
  for (const w of left) if (right.has(w)) shared += 1;
  return shared / Math.max(left.size, 1);
}

export function closestDemo(question: string): DemoRun {
  let best = DEMO_RUNS[0];
  let bestScore = -1;
  for (const run of DEMO_RUNS) {
    const score = overlap(question, run.question);
    if (score > bestScore) {
      best = run;
      bestScore = score;
    }
  }
  return best;
}
