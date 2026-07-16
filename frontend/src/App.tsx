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
import ResearchMap, { Timeline } from "./ResearchMap";
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
  | { kind: "working"; progress: ProgressStep[]; mode: Mode }
  | { kind: "done"; result: QueryResult }
  | { kind: "mapped"; result: LandscapeResult }
  // A saved run, shown when the live agent cannot answer. `asked` is what the visitor typed,
  // which is not what the saved run answered; the banner says so.
  | { kind: "demo"; result: QueryResult; answered: string; asked: string }
  | { kind: "failed"; message: string; progress: ProgressStep[] };

// Everything a tab owns. The two tabs run fully independently: switching between them never
// shows the other's results, and a run in one never touches the other's state.
type TabState = {
  question: string;
  phase: Phase;
  exploreId: string | null; // selected paper / open direction
  activeQuery: string | null;
};

const EMPTY_TAB: TabState = {
  question: "",
  phase: { kind: "idle" },
  exploreId: null,
  activeQuery: null,
};

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
  const [corpusSize, setCorpusSize] = useState<number | null>(null);
  const [tabs, setTabs] = useState<Record<Mode, TabState>>({
    map: { ...EMPTY_TAB },
    ask: { ...EMPTY_TAB },
  });
  const view = useHashView();
  // Bumped on every submit so a stale poll loop from an abandoned run stops writing state; kept
  // per tab so submitting in one tab never cancels the other's in-flight poll loop.
  const runId = useRef<Record<Mode, number>>({ map: 0, ask: 0 });
  const startedAt = useRef<Record<Mode, number>>({ map: 0, ask: 0 });
  const inputRef = useRef<HTMLInputElement>(null);

  const { question, phase, exploreId, activeQuery } = tabs[mode];
  const patchTab = (m: Mode, patch: Partial<TabState>) =>
    setTabs((prev) => ({ ...prev, [m]: { ...prev[m], ...patch } }));

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
    const q = (text ?? tabs[m].question).trim();
    if (!q || tabs[m].phase.kind === "working") return;
    if (as !== undefined) setMode(as);
    const run = ++runId.current[m];
    startedAt.current[m] = Date.now();
    // A new query resets this tab; the other tab is untouched.
    patchTab(m, {
      question: text ?? tabs[m].question,
      activeQuery: q,
      exploreId: null,
      phase: { kind: "working", progress: [], mode: m },
    });
    try {
      if (m === "map") {
        await drive(run, m, await postLandscape(q), getLandscapeJob, (result) =>
          patchTab(m, { phase: { kind: "mapped", result } }),
        );
      } else {
        await drive(run, m, await postQuery(q), getJob, (result) =>
          patchTab(m, { phase: { kind: "done", result } }),
        );
      }
    } catch (err) {
      if (run !== runId.current[m]) return;
      // 503 is the agent declining: the daily cap (the drain backstop), an exhausted loop, or
      // a worker that could not start. The ask mode has saved runs to fall back on; the map
      // mode reports honestly.
      if (err instanceof ApiError && err.status === 503 && m === "ask") {
        const demo = closestDemo(q);
        patchTab(m, { phase: { kind: "demo", result: demo.result, answered: demo.question, asked: q } });
        return;
      }
      patchTab(m, { phase: { kind: "failed", message: friendly(err), progress: [] } });
    }
  }

  async function drive<T>(
    run: number,
    m: Mode,
    outcome: PostOutcome<T>,
    get: (jobId: string) => Promise<JobStatus<T>>,
    finish: (result: T) => void,
  ) {
    if (run !== runId.current[m]) return;
    if (outcome.kind === "done") {
      finish(outcome.result);
      return;
    }
    let progress: ProgressStep[] = [];
    let failures = 0;
    for (;;) {
      await sleep(POLL_MS);
      if (run !== runId.current[m]) return;
      let job;
      try {
        job = await get(outcome.jobId);
        failures = 0;
      } catch (err) {
        if (++failures >= MAX_POLL_FAILURES) {
          patchTab(m, { phase: { kind: "failed", message: friendly(err), progress } });
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
        patchTab(m, { phase: { kind: "failed", message: job.error ?? "the run failed", progress } });
        return;
      }
      patchTab(m, { phase: { kind: "working", progress, mode: m } });
    }
  }

  const showContext = phase.kind !== "idle" && activeQuery !== null;
  const topbar = (
    <header className="topbar">
      <h1>
        <a href="#">ArXiv Atlas</a>
      </h1>
      {showContext && <span className="topbar-context">{activeQuery}</span>}
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

  return (
    <main>
      {topbar}

      {phase.kind === "idle" && (
        <section className="hero narrow">
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

      <div className="searchbar">
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
            onChange={(e) => patchTab(mode, { question: e.target.value })}
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
      </div>

      {phase.kind === "idle" && (
        <div className="narrow">
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
                You get the landscape: a research map, key ideas, activity over time, a reading
                order, and the open problems, all cited.
              </p>
            </div>
          </section>

          <footer>
            <span>
              A semantic research graph over arXiv cs.AI, cs.LG and cs.CL abstracts, refreshed
              nightly.
            </span>
          </footer>
        </div>
      )}

      {phase.kind === "working" && (
        <div className="narrow-result">
          <Progress steps={phase.progress} startedAt={startedAt.current[mode]} mode={phase.mode} />
        </div>
      )}

      {phase.kind === "failed" && (
        <div className="narrow-result">
          <section className="card error" role="alert">
            <p className="error-title">That did not work</p>
            <p>{phase.message}</p>
            <button className="retry" onClick={() => void submit()}>
              Try again
            </button>
            {phase.progress.length > 0 && (
              <Progress steps={phase.progress} startedAt={startedAt.current[mode]} stalled />
            )}
          </section>
        </div>
      )}

      {/* Both tabs' results stay mounted so switching away and back keeps a tab exactly as it
          was (open direction, force layout, loaded graph) with no refetch; only the active one
          is shown. */}
      <Workspace
        phase={tabs.map.phase}
        exploreId={tabs.map.exploreId}
        onSelect={(id) => patchTab("map", { exploreId: id })}
        hidden={mode !== "map"}
      />
      <Workspace
        phase={tabs.ask.phase}
        exploreId={tabs.ask.exploreId}
        onSelect={(id) => patchTab("ask", { exploreId: id })}
        hidden={mode !== "ask"}
      />
    </main>
  );
}

function Workspace({
  phase,
  exploreId,
  onSelect,
  hidden,
}: {
  phase: Phase;
  exploreId: string | null;
  onSelect: (arxivId: string | null) => void;
  hidden: boolean;
}) {
  const queryResult = phase.kind === "done" || phase.kind === "demo" ? phase.result : null;
  const landscape = phase.kind === "mapped" ? phase.result : null;
  if (!queryResult && !landscape) return null;
  const split =
    (landscape !== null && !landscape.declined) || (queryResult !== null && exploreId !== null);
  const exploreTitle =
    exploreId === null
      ? undefined
      : queryResult?.papers.find((p) => p.arxiv_id === exploreId)?.title;

  return (
    // Inline display beats the .workspace grid rule; the subtree stays mounted while hidden.
    <div className={split ? "workspace split" : "workspace"} style={hidden ? { display: "none" } : undefined}>
      <div className="ws-main">
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
          <Result result={queryResult} exploreId={exploreId} onExplore={onSelect} />
        )}

        {landscape && (
          <LandscapeView result={landscape} selectedId={exploreId} onSelect={onSelect} />
        )}

        {(queryResult ?? landscape) && (
          <TracePanel
            trace={(queryResult ?? landscape)!.trace}
            total={(queryResult ?? landscape)!.cost_usd}
          />
        )}
      </div>

      {split && (
        <aside className="ws-side">
          {landscape && !landscape.declined && (
            <>
              <ResearchMap
                directions={landscape.directions}
                links={landscape.links}
                selected={exploreId}
                onSelect={onSelect}
              />
              {landscape.timeline.length > 0 && (
                <Timeline timeline={landscape.timeline} directions={landscape.directions} />
              )}
            </>
          )}
          {queryResult && exploreId && (
            <GraphExplorer rootId={exploreId} rootTitle={exploreTitle} />
          )}
        </aside>
      )}
    </div>
  );
}

// The landscape pipeline is a fixed sequence, so its progress renders as named stages that
// fill in, with the direction names appearing live as the agent produces them. The ask
// pipeline loops (evidence rounds vary), so it keeps the append-style checklist.
const MAP_STAGES = [
  { key: "planner", label: "Understanding the topic" },
  { key: "retriever", label: "Finding relevant papers" },
  { key: "cluster", label: "Identifying research directions" },
  { key: "direction", label: "Reading each direction" },
  { key: "landscape", label: "Writing the landscape" },
];

function MapStages({ steps }: { steps: ProgressStep[] }) {
  const stageOf = (step: string) => MAP_STAGES.findIndex((s) => s.key === step);
  const arrived = steps.map((s) => stageOf(s.step)).filter((i) => i >= 0);
  const reached = arrived.length ? Math.max(...arrived) : -1;
  // A step arrives when its stage completes, so stages up to `reached` are done, except
  // "direction", which emits one step per cluster and stays live until synthesis starts.
  const isDone = (i: number) => (i === 3 ? reached >= 4 : reached >= i);
  const current = MAP_STAGES.findIndex((_, i) => !isDone(i));
  const directionNames = steps.filter((s) => s.step === "direction").map((s) => s.summary);
  const detailFor = (i: number) => {
    if (i === 3) return undefined;
    const matching = steps.filter((s) => stageOf(s.step) === i);
    return matching[matching.length - 1]?.summary;
  };

  return (
    <ol className="stages">
      {MAP_STAGES.map((stage, i) => {
        const state = isDone(i) ? "done" : i === current ? "current" : "pending";
        return (
          <li key={stage.key} className={state}>
            <span className="stage-icon" aria-hidden="true">
              {state === "done" ? (
                <span className="step-check">&#10003;</span>
              ) : state === "current" ? (
                <span className="spinner" />
              ) : (
                <span className="stage-dot" />
              )}
            </span>
            <div className="stage-body">
              <span className="stage-label">{stage.label}</span>
              {detailFor(i) && <span className="stage-detail">{detailFor(i)}</span>}
              {i === 3 && directionNames.length > 0 && (
                <span className="dir-chips">
                  {directionNames.map((name) => (
                    <span key={name} className="chip">
                      {name}
                    </span>
                  ))}
                </span>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function Progress({
  steps,
  startedAt,
  stalled,
  mode,
}: {
  steps: ProgressStep[];
  startedAt: number;
  stalled?: boolean;
  mode?: Mode;
}) {
  const staged = mode === "map" && !stalled;
  return (
    <section className="card progress">
      {!stalled && (
        <div className="progress-head">
          <span className="spinner" aria-hidden="true" />
          <span className="progress-title">
            {mode === "map" ? "Mapping the landscape" : "Researching"}
          </span>
          <Elapsed since={startedAt} />
        </div>
      )}
      {/* the ticking clock stays outside this region so screen readers hear steps, not seconds */}
      <div aria-live="polite">
        {staged ? (
          <MapStages steps={steps} />
        ) : (
          steps.length > 0 && (
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
          )
        )}
        {!staged && !stalled && steps.length === 0 && (
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
