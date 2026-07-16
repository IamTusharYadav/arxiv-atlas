import type { TraceStep } from "./api";
import { stepLabel } from "./steps";

// Reader-facing names for each pipeline stage. The raw step keys (planner, retriever, ...) are
// internal, so they never surface here; an unknown step falls back to the terse shared label.
const STAGE_NAMES: Record<string, string> = {
  planner: "Understand the request",
  retriever: "Search for papers",
  reranker: "Rank by relevance",
  extractor: "Extract key findings",
  check: "Assess coverage",
  synthesizer: "Write the answer",
  cluster: "Group related papers",
  direction: "Identify a research direction",
  landscape: "Compose the landscape",
};

const stageName = (step: string): string => STAGE_NAMES[step] ?? stepLabel(step);

// A plain-language walk-through of how the answer was produced, in a collapsed panel for the
// curious. Token counts and cost stay out of it: those are diagnostics, not part of the answer.
export default function TracePanel({ trace }: { trace: TraceStep[] }) {
  if (trace.length === 0) return null;
  return (
    <details className="card trace">
      <summary>
        How this answer was made
        <span className="trace-meta">{trace.length} steps</span>
      </summary>
      <ol className="trace-steps">
        {trace.map((step, i) => (
          <li key={i}>
            <span className="trace-num" aria-hidden="true">
              {i + 1}
            </span>
            <div className="trace-body">
              <span className="trace-name">{stageName(step.step)}</span>
              <span className="trace-what">{step.summary}</span>
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}
