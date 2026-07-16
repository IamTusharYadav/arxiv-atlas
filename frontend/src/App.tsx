import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  ApiError,
  getJob,
  getLandscapeJob,
  getStatus,
  postLandscape,
  postQuery,
  type JobStatus,
  type LandscapeResult,
  type PostOutcome,
  type ProgressStep,
  type QueryResult,
} from "./api";
import GraphExplorer from "./GraphExplorer";
import LandscapeView from "./LandscapeView";
import StatusPage from "./StatusPage";
import TracePanel from "./TracePanel";
import { closestDemo, DEMO_RUNS } from "./demo";
import { stepLabel } from "./steps";

const POLL_MS = 2500;
// A few network blips are survivable; a run of them means the API is gone, so stop burning
// the poll loop. The server itself reports a dead worker after 900s of silence.
const MAX_POLL_FAILURES = 5;

// Topics for the map mode's examples; phrased the way the planner likes topics, and broad
// enough that retrieval fills every direction.
const MAP_EXAMPLES = [
  "efficient inference for large language models",
  "prompt injection and jailbreak defenses",
  "parameter-efficient fine-tuning",
];

type Mode = "map" | "ask";

type Phase =
  | { kind: "idle" }
  | { kind: "working"; progress: ProgressStep[] }
  | { kind: "done"; result: QueryResult }
  | { kind: "mapped"; result: LandscapeResult }
  // A saved run, shown when the live agent cannot answer. `asked` is what the visitor typed,
  // which is not what the saved run answered; the banner says so.
  | { kind: "demo"; result: QueryResult; answered: string; asked: string }
  | { kind: "failed"; message: string; progress: ProgressStep[] };

// The URL hash is the whole router: one extra view, deep-linkable, browser back works, and no
// router dependency for it.
function useHashView(): string {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const onHash = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return hash;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("map");
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [corpusSize, setCorpusSize] = useState<number | null>(null);
  const [exploreId, setExploreId] = useState<string | null>(null);
  const view = useHashView();
  // Bumped on every submit so a stale poll loop from an abandoned run stops writing state.
  const runId = useRef(0);
  const startedAt = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getStatus()
      .then((s) => setCorpusSize(s.corpus_size))
      .catch(() => setCorpusSize(null)); // header stat only; the page works without it
  }, []);

  // "/" focuses the search box from anywhere except another text field.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      e.preventDefault();
      inputRef.current?.focus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function submit(text?: string, as?: Mode) {
    const m = as ?? mode;
    const q = (text ?? question).trim();
    if (!q || phase.kind === "working") return;
    if (text !== undefined) setQuestion(text);
    if (as !== undefined) setMode(as);
    const run = ++runId.current;
    startedAt.current = Date.now();
    setExploreId(null); // a new run invalidates the old neighborhood
    setPhase({ kind: "working", progress: [] });
    try {
      if (m === "map") {
        await drive(run, await postLandscape(q), getLandscapeJob, (result) =>
          setPhase({ kind: "mapped", result }),
        );
      } else {
        await drive(run, await postQuery(q), getJob, (result) =>
          setPhase({ kind: "done", result }),
        );
      }
    } catch (err) {
      if (run !== runId.current) return;
      // 503 is the agent declining: the daily cap (the drain backstop), an exhausted loop, or
      // a worker that could not start. The ask mode has saved runs to fall back on; the map
      // mode reports honestly.
      if (err instanceof ApiError && err.status === 503 && m === "ask") {
        const demo = closestDemo(q);
        setPhase({ kind: "demo", result: demo.result, answered: demo.question, asked: q });
        return;
      }
      setPhase({ kind: "failed", message: friendly(err), progress: [] });
    }
  }

  async function drive<T>(
    run: number,
    outcome: PostOutcome<T>,
    get: (jobId: string) => Promise<JobStatus<T>>,
    finish: (result: T) => void,
  ) {
    if (run !== runId.current) return;
    if (outcome.kind === "done") {
      finish(outcome.result);
      return;
    }
    let progress: ProgressStep[] = [];
    let failures = 0;
    for (;;) {
      await sleep(POLL_MS);
      if (run !== runId.current) return;
      let job;
      try {
        job = await get(outcome.jobId);
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
        finish(job.result);
        return;
      }
      if (job.status === "error") {
        setPhase({ kind: "failed", message: job.error ?? "the run failed", progress });
        return;
      }
      setPhase({ kind: "working", progress });
    }
  }

  const topbar = (
    <header className="topbar">
      <h1>
        <a href="#">ArXiv Atlas</a>
      </h1>
      <nav>
        {view === "#status" ? (
          <a href="#">Search</a>
        ) : (
          <>
            {corpusSize !== null && (
              <span className="corpus">{corpusSize.toLocaleString()} papers</span>
            )}
            <a href="#status">Status</a>
          </>
        )}
      </nav>
    </header>
  );

  if (view === "#status") {
    return (
      <main>
        {topbar}
        <StatusPage />
      </main>
    );
  }

  const queryResult =
    phase.kind === "done" || phase.kind === "demo" ? phase.result : null;
  const landscape = phase.kind === "mapped" ? phase.result : null;
  const trace = queryResult ?? landscape;
  const exploreTitle =
    exploreId === null
      ? undefined
      : (queryResult?.papers.find((p) => p.arxiv_id === exploreId)?.title ??
        landscape?.directions
          .flatMap((d) => d.papers)
          .find((p) => p.arxiv_id === exploreId)?.title);

  return (
    <main>
      {topbar}

      {phase.kind === "idle" && (
        <section className="hero">
          <h2>Understand a research area in minutes.</h2>
          <p>
            Name a topic and Atlas maps the current landscape from{" "}
            {corpusSize !== null ? corpusSize.toLocaleString() : "almost 100,000"} recent arXiv
            abstracts: the major directions, where activity is flowing, which papers to read
            first, and what remains open. Or ask a pointed question and get a brief in which
            every claim cites its paper.
          </p>
        </section>
      )}

      <div className="mode" role="tablist" aria-label="Mode">
        <button
          role="tab"
          aria-selected={mode === "map"}
          className={mode === "map" ? "active" : ""}
          onClick={() => setMode("map")}
        >
          Map a topic
        </button>
        <button
          role="tab"
          aria-selected={mode === "ask"}
          className={mode === "ask" ? "active" : ""}
          onClick={() => setMode("ask")}
        >
          Ask a question
        </button>
      </div>

      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <input
          ref={inputRef}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={
            mode === "map"
              ? "e.g. efficient inference for large language models"
              : "e.g. How do KV-cache methods trade memory for accuracy?"
          }
          aria-label={mode === "map" ? "Research topic" : "Research question"}
          maxLength={mode === "map" ? 200 : 500}
          disabled={phase.kind === "working"}
        />
        <button type="submit" disabled={phase.kind === "working" || !question.trim()}>
          {phase.kind === "working" ? "Working" : mode === "map" ? "Map" : "Ask"}
        </button>
      </form>

      {phase.kind === "idle" && (
        <>
          <section className="examples" aria-label="Examples">
            <p className="section-label">
              Or start from an example
              <span className="kbd-hint">
                <kbd>/</kbd> focuses the search box
              </span>
            </p>
            <ul>
              {mode === "map"
                ? MAP_EXAMPLES.map((topic) => (
                    <li key={topic}>
                      <button type="button" onClick={() => void submit(topic, "map")}>
                        {topic}
                      </button>
                    </li>
                  ))
                : DEMO_RUNS.map((r) => (
                    <li key={r.question}>
                      <button type="button" onClick={() => void submit(r.question, "ask")}>
                        {r.question}
                      </button>
                    </li>
                  ))}
            </ul>
          </section>

          <section className="how" aria-label="How it works">
            <div>
              <h3>Search</h3>
              <p>Your topic is split into focused subqueries and matched against the corpus.</p>
            </div>
            <div>
              <h3>Group and read</h3>
              <p>
                Matching papers are clustered into research directions, and an agent reads and
                names each one.
              </p>
            </div>
            <div>
              <h3>Map</h3>
              <p>
                You get the landscape: key ideas, activity over time, a reading order, and the
                open problems, all cited.
              </p>
            </div>
          </section>

          <footer>
            <span>
              A semantic research graph over arXiv cs.AI, cs.LG and cs.CL abstracts, refreshed
              nightly.
            </span>
          </footer>
        </>
      )}

      {phase.kind === "working" && (
        <Progress steps={phase.progress} startedAt={startedAt.current} />
      )}

      {phase.kind === "failed" && (
        <section className="card error" role="alert">
          <p className="error-title">That did not work</p>
          <p>{phase.message}</p>
          <button className="retry" onClick={() => void submit()}>
            Try again
          </button>
          {phase.progress.length > 0 && (
            <Progress steps={phase.progress} startedAt={startedAt.current} stalled />
          )}
        </section>
      )}

      {phase.kind === "demo" && (
        <section className="card demo-banner">
          <p>
            <strong>Live answers are paused right now, so this is a saved example run.</strong>{" "}
            It answers a different question than the one you asked, and it is a real recorded
            answer over the same corpus, not a live one.
          </p>
          <p className="asked">
            You asked: <em>{phase.asked}</em>
            <br />
            Showing: <em>{phase.answered}</em>
          </p>
        </section>
      )}

      {queryResult && (
        <Result result={queryResult} exploreId={exploreId} onExplore={setExploreId} />
      )}

      {landscape && (
        <LandscapeView result={landscape} exploreId={exploreId} onExplore={setExploreId} />
      )}

      {trace && exploreId && <GraphExplorer rootId={exploreId} rootTitle={exploreTitle} />}

      {trace && <TracePanel trace={trace.trace} total={trace.cost_usd} />}
    </main>
  );
}

function Progress({
  steps,
  startedAt,
  stalled,
}: {
  steps: ProgressStep[];
  startedAt: number;
  stalled?: boolean;
}) {
  return (
    <section className="card progress">
      {!stalled && (
        <div className="progress-head">
          <span className="spinner" aria-hidden="true" />
          <span className="progress-title">Researching</span>
          <Elapsed since={startedAt} />
        </div>
      )}
      {/* the ticking clock stays outside this region so screen readers hear steps, not seconds */}
      <div aria-live="polite">
        {steps.length > 0 && (
          <ol>
            {steps.map((s, i) => (
              <li key={i}>
                <span className="step-check" aria-hidden="true">
                  &#10003;
                </span>
                <span className="step-name">{stepLabel(s.step)}</span>
                <span className="step-summary">{s.summary}</span>
              </li>
            ))}
          </ol>
        )}
        {!stalled && steps.length === 0 && (
          <p className="hint">
            Answers usually take 30 to 90 seconds; multi-part questions can take longer. Each
            step shows up here as it completes.
          </p>
        )}
      </div>
    </section>
  );
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  return <span className="elapsed">{Math.max(0, Math.round((now - since) / 1000))}s</span>;
}

function Result({
  result,
  exploreId,
  onExplore,
}: {
  result: QueryResult;
  exploreId: string | null;
  onExplore: (arxivId: string | null) => void;
}) {
  return (
    <>
      <section className="card brief">
        {(result.cached || result.partial) && (
          <div className="meta">
            {result.cached && (
              <span className="badge" title="A very similar question was answered recently">
                cached
              </span>
            )}
            {result.partial && <span className="badge warn">stopped early</span>}
          </div>
        )}
        {result.partial && (
          <p className="partial-note">
            The run stopped before a full brief could be written; below is the evidence it had
            gathered, grouped by paper.
          </p>
        )}
        <ReactMarkdown>{result.brief}</ReactMarkdown>
      </section>
      <section className="card papers">
        <h2>
          Cited papers <span className="count">{result.papers.length}</span>
        </h2>
        <ul>
          {result.papers.map((p) => (
            <li key={p.arxiv_id}>
              <div className="paper-main">
                <a
                  className="paper-title"
                  href={`https://arxiv.org/abs/${p.arxiv_id}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {p.title}
                </a>
                <span className="paper-meta">
                  <span className="cat" data-cat={p.primary_category}>
                    {p.primary_category}
                  </span>
                  <span className="paper-id">{p.arxiv_id}</span>
                </span>
              </div>
              <button
                className={exploreId === p.arxiv_id ? "explore active" : "explore"}
                onClick={() => onExplore(exploreId === p.arxiv_id ? null : p.arxiv_id)}
              >
                {exploreId === p.arxiv_id ? "hide graph" : "graph"}
              </button>
            </li>
          ))}
        </ul>
      </section>
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
