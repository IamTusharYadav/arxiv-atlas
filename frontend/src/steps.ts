// The agent's internal step names, translated for people. Unknown names pass through so a
// new backend step degrades to its raw name instead of vanishing.
const LABELS: Record<string, string> = {
  planner: "Plan",
  retriever: "Search",
  reranker: "Rank",
  extractor: "Read",
  check: "Verify",
  synthesizer: "Write",
  cluster: "Group",
  direction: "Name",
  landscape: "Map",
};

export function stepLabel(step: string): string {
  return LABELS[step] ?? step;
}
