import { type Ref, useEffect, useRef, useState } from "react";
import type { Direction, LandscapeResult } from "./api";
import { arxivUrl, CitedMarkdown, flashPaper, linkifyCitations } from "./citations";
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
  // A cited paper lives inside its direction's (collapsed, lazy) list, so revealing it means
  // expanding that direction and then, once it has rendered, scrolling to the entry. `pending`
  // carries the id across that render into the effect below.
  const [open, setOpen] = useState<ReadonlySet<number>>(new Set());
  const [pending, setPending] = useState<string | null>(null);
  const panelRef = useRef<HTMLElement>(null);

  const toggleDir = (i: number) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  const reveal = (id: string) => {
    const idx = result.directions.findIndex((d) => d.papers.some((p) => p.arxiv_id === id));
    if (idx < 0) {
      window.open(arxivUrl(id), "_blank", "noopener");
      return;
    }
    setOpen((prev) => (prev.has(idx) ? prev : new Set(prev).add(idx)));
    setPending(id);
  };

  useEffect(() => {
    if (!pending) return;
    flashPaper(panelRef.current?.querySelector(`[data-paper="${pending}"]`));
    setPending(null);
  }, [pending]);

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
        <CitedMarkdown onCite={reveal}>{result.overview}</CitedMarkdown>
        {result.key_ideas.length > 0 && (
          <>
            <h3 className="landscape-sub">Key ideas</h3>
            <ul className="key-ideas">
              {result.key_ideas.map((idea) => (
                <li key={idea}>{linkifyCitations(idea, reveal)}</li>
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
        open={open}
        onToggle={toggleDir}
        panelRef={panelRef}
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
                  <span className="reading-reason">{linkifyCitations(r.reason, reveal)}</span>
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
                <li key={p}>{linkifyCitations(p, reveal)}</li>
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
  open,
  onToggle,
  panelRef,
}: {
  directions: Direction[];
  selectedId: string | null;
  onSelect: (arxivId: string | null) => void;
  // Expansion is owned by the parent so a citation click can open the direction holding the
  // cited paper before scrolling to it.
  open: ReadonlySet<number>;
  onToggle: (i: number) => void;
  panelRef: Ref<HTMLElement>;
}) {
  const dark = useDarkMode();
  const colors = SERIES[dark ? "dark" : "light"];

  return (
    <section className="card" ref={panelRef}>
      <h2>Research directions</h2>
      <ul className="directions">
        {directions.map((d, i) => {
          const expanded = open.has(i);
          return (
            <li key={d.name} className="direction">
              <div className="direction-head">
                <span className="swatch" style={{ background: colors[i % colors.length] }} />
                <h3 className="direction-name">{d.name}</h3>
                <span className="count">{d.papers.length}</span>
              </div>
              <p className="direction-problem">{linkifyCitations(d.problem)}</p>
              <button
                className="view-papers"
                aria-expanded={expanded}
                aria-controls={`dir-papers-${i}`}
                onClick={() => onToggle(i)}
              >
                {expanded ? "Hide papers" : `View papers (${d.papers.length})`}
              </button>
              {expanded && (
                <ul id={`dir-papers-${i}`} className="direction-papers">
                  {d.papers.map((p) => (
                    <li key={p.arxiv_id} data-paper={p.arxiv_id}>
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
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
