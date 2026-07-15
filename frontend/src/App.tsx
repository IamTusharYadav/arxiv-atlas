import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  ApiError,
  getJob,
  getStatus,
  postQuery,
  type ProgressStep,
  type QueryResult,
} from "./api";
import GraphExplorer from "./GraphExplorer";
import TracePanel from "./TracePanel";

const POLL_MS = 2500;
// A few network blips are survivable; a run of them means the API is gone, so stop burning
// the poll loop. The server itself reports a dead worker after 900s of silence.
const MAX_POLL_FAILURES = 5;

type Phase =
  | { kind: "idle" }
  | { kind: "working"; progress: ProgressStep[] }
  | { kind: "done"; result: QueryResult }
  | { kind: "failed"; message: string; progress: ProgressStep[] };

export default function App() {
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [corpusSize, setCorpusSize] = useState<number | null>(null);
  const [exploreId, setExploreId] = useState<string | null>(null);
  // Bumped on every submit so a stale poll loop from an abandoned run stops writing state.
  const runId = useRef(0);

  useEffect(() => {
    getStatus()
      .then((s) => setCorpusSize(s.corpus_size))
      .catch(() => setCorpusSize(null)); // header stat only; the page works without it
  }, []);

  async function submit() {
    const q = question.trim();
    if (!q || phase.kind === "working") return;
    const run = ++runId.current;
    setExploreId(null); // a new question invalidates the old neighborhood
    setPhase({ kind: "working", progress: [] });
    try {
      const outcome = await postQuery(q);
      if (run !== runId.current) return;
      if (outcome.kind === "done") {
        setPhase({ kind: "done", result: outcome.result });
        return;
      }
      await poll(outcome.jobId, run);
    } catch (err) {
      if (run !== runId.current) return;
      setPhase({ kind: "failed", message: friendly(err), progress: [] });
    }
  }

  async function poll(jobId: string, run: number) {
    let progress: ProgressStep[] = [];
    let failures = 0;
    for (;;) {
      await sleep(POLL_MS);
      if (run !== runId.current) return;
      let job;
      try {
        job = await getJob(jobId);
        failures = 0;
      } catch (err) {
        if (++failures >= MAX_POLL_FAILURES) {
          setPhase({ kind: "failed", message: friendly(err), progress });
          return;
        }
        continue;
      }
      progress = job.progress;
      if (job.status === "done" && job.result) {
        setPhase({ kind: "done", result: job.result });
        return;
      }
      if (job.status === "error") {
        setPhase({ kind: "failed", message: job.error ?? "the query failed", progress });
        return;
      }
      setPhase({ kind: "working", progress });
    }
  }

  return (
    <main>
      <header>
        <h1>ArXiv Atlas</h1>
        <p className="tagline">
          A semantic research graph over cs.AI / cs.LG / cs.CL abstracts
          {corpusSize !== null && <> &middot; {corpusSize.toLocaleString()} papers indexed</>}
        </p>
      </header>

      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about AI/ML research, e.g. how do KV-cache methods trade memory for accuracy?"
          maxLength={500}
          disabled={phase.kind === "working"}
        />
        <button type="submit" disabled={phase.kind === "working" || !question.trim()}>
          {phase.kind === "working" ? "Working" : "Ask"}
        </button>
      </form>

      {phase.kind === "working" && <Progress steps={phase.progress} />}

      {phase.kind === "failed" && (
        <section className="card error">
          <p>{phase.message}</p>
          {phase.progress.length > 0 && <Progress steps={phase.progress} stalled />}
        </section>
      )}

      {phase.kind === "done" && <Result result={phase.result} onExplore={setExploreId} />}

      {phase.kind === "done" && exploreId && <GraphExplorer rootId={exploreId} />}
    </main>
  );
}

function Progress({ steps, stalled }: { steps: ProgressStep[]; stalled?: boolean }) {
  return (
    <section className="card progress">
      <ol>
        {steps.map((s, i) => (
          <li key={i}>
            <span className="step-name">{s.step}</span>
            <span className="step-summary">{s.summary}</span>
          </li>
        ))}
        {!stalled && <li className="pending">thinking&hellip;</li>}
      </ol>
    </section>
  );
}

function Result({
  result,
  onExplore,
}: {
  result: QueryResult;
  onExplore: (arxivId: string) => void;
}) {
  return (
    <>
      <section className="card brief">
        <div className="meta">
          {result.cached ? <span className="badge">cached</span> : null}
          {result.partial ? <span className="badge warn">stopped early</span> : null}
          <span className="cost">${result.cost_usd.toFixed(4)}</span>
        </div>
        <ReactMarkdown>{result.brief}</ReactMarkdown>
      </section>
      <section className="card papers">
        <h2>Papers</h2>
        <ul>
          {result.papers.map((p) => (
            <li key={p.arxiv_id}>
              <a href={`https://arxiv.org/abs/${p.arxiv_id}`} target="_blank" rel="noreferrer">
                {p.arxiv_id}
              </a>
              <span className="title">{p.title}</span>
              <span className="badge">{p.primary_category}</span>
              <button className="explore" onClick={() => onExplore(p.arxiv_id)}>
                graph
              </button>
            </li>
          ))}
        </ul>
      </section>
      <TracePanel trace={result.trace} total={result.cost_usd} />
    </>
  );
}

function friendly(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 429) {
      const wait = err.retryAfterS ? ` Try again in ${err.retryAfterS}s.` : "";
      return `Rate limit hit.${wait}`;
    }
    return err.message;
  }
  return "Could not reach the API.";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
