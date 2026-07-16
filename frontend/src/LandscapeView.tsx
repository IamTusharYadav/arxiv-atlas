import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Direction, LandscapeResult } from "./api";
import { SERIES } from "./palette";
import { useDarkMode } from "./useDarkMode";

// The content pane of the landscape workspace. The visualizations (research map, activity
// timeline) live in the sticky side pane; App composes the two.
export default function LandscapeView({
  result,
  selectedId,
  onSelect,
}: {
  result: LandscapeResult;
  selectedId: string | null;
  onSelect: (arxivId: string | null) => void;
}) {
  if (result.declined) {
    return (
      <section className="card">
        <p>{result.overview}</p>
      </section>
    );
  }

  return (
    <>
      <section className="card brief">
        {result.cached && (
          <div className="meta">
            <span className="badge" title="A very similar topic was mapped recently">
              cached
            </span>
          </div>
        )}
        <h2>The landscape</h2>
        <ReactMarkdown>{result.overview}</ReactMarkdown>
        {result.key_ideas.length > 0 && (
          <>
            <h3 className="landscape-sub">Key ideas</h3>
            <ul className="key-ideas">
              {result.key_ideas.map((idea) => (
                <li key={idea}>{idea}</li>
              ))}
            </ul>
          </>
        )}
        <p className="hint">
          Mapped from the last ~12 months of arXiv abstracts; groupings are semantic, not
          citation-based.
        </p>
      </section>

      <DirectionsPanel
        directions={result.directions}
        selectedId={selectedId}
        onSelect={onSelect}
      />

      <div className="two-col">
        {result.reading_order.length > 0 && (
          <section className="card">
            <h2>Where to start</h2>
            <ol className="reading-order">
              {result.reading_order.map((r) => (
                <li key={r.arxiv_id}>
                  <a
                    href={`https://arxiv.org/abs/${r.arxiv_id}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {r.title || r.arxiv_id}
                  </a>
                  <span className="reading-reason">{r.reason}</span>
                </li>
              ))}
            </ol>
          </section>
        )}
        {result.open_problems.length > 0 && (
          <section className="card">
            <h2>Open problems</h2>
            <ul className="open-problems">
              {result.open_problems.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </>
  );
}

function DirectionsPanel({
  directions,
  selectedId,
  onSelect,
}: {
  directions: Direction[];
  selectedId: string | null;
  onSelect: (arxivId: string | null) => void;
}) {
  const dark = useDarkMode();
  const colors = SERIES[dark ? "dark" : "light"];
  const [active, setActive] = useState(0);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const direction = directions[Math.min(active, directions.length - 1)];

  // Roving tabs: arrow keys move focus and selection, per the WAI-ARIA tabs pattern.
  function onKey(e: React.KeyboardEvent) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const delta = e.key === "ArrowRight" ? 1 : -1;
    const next = (active + delta + directions.length) % directions.length;
    setActive(next);
    tabRefs.current[next]?.focus();
  }

  return (
    <section className="card">
      <h2>Research directions</h2>
      <div className="tabs" role="tablist" aria-label="Research directions" onKeyDown={onKey}>
        {directions.map((d, i) => (
          <button
            key={d.name}
            ref={(el) => {
              tabRefs.current[i] = el;
            }}
            role="tab"
            id={`dir-tab-${i}`}
            aria-controls="dir-panel"
            aria-selected={i === active}
            tabIndex={i === active ? 0 : -1}
            className={i === active ? "tab active" : "tab"}
            onClick={() => setActive(i)}
          >
            <span className="swatch" style={{ background: colors[i % colors.length] }} />
            {d.name}
            <span className="count">{d.papers.length}</span>
          </button>
        ))}
      </div>
      {direction && (
        <div
          id="dir-panel"
          role="tabpanel"
          aria-labelledby={`dir-tab-${Math.min(active, directions.length - 1)}`}
        >
          <p className="direction-problem">{direction.problem}</p>
          <ul className="direction-papers">
            {direction.papers.map((p) => (
              <li key={p.arxiv_id}>
                <a
                  href={`https://arxiv.org/abs/${p.arxiv_id}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {p.title}
                </a>
                <span className="paper-month">{p.published_month}</span>
                <button
                  className={selectedId === p.arxiv_id ? "explore active" : "explore"}
                  onClick={() => onSelect(selectedId === p.arxiv_id ? null : p.arxiv_id)}
                >
                  {selectedId === p.arxiv_id ? "on map" : "show on map"}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
