import ReactMarkdown from "react-markdown";
import type { Direction, LandscapeResult, TimelinePoint } from "./api";
import { useDarkMode } from "./useDarkMode";

// Categorical palette for directions, fixed slot order, never cycled (k is capped at 6
// server-side). Slots 1-3 are the app's category colors; all six steps validated per mode
// against the app surfaces with the dataviz six-checks script (worst adjacent CVD dE 9.1
// light / 8.4 dark). Light steps 4-6 sit under 3:1 contrast on the light surface, which is
// the documented relief case: the legend is always shown, every segment carries a tooltip,
// and the direction lists repeat the same counts as text.
const SERIES = {
  light: ["#2a78d6", "#008300", "#d55181", "#eda100", "#1baf7a", "#eb6834"],
  dark: ["#3987e5", "#008300", "#d55181", "#c98500", "#199e70", "#d95926"],
};

export default function LandscapeView({
  result,
  exploreId,
  onExplore,
}: {
  result: LandscapeResult;
  exploreId: string | null;
  onExplore: (arxivId: string | null) => void;
}) {
  const dark = useDarkMode();
  const colors = SERIES[dark ? "dark" : "light"];

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

      <section className="card">
        <h2>Research directions</h2>
        {result.timeline.length > 0 && (
          <Timeline
            timeline={result.timeline}
            directions={result.directions}
            colors={colors}
          />
        )}
        <div className="directions">
          {result.directions.map((d, i) => (
            <DirectionBlock
              key={d.name}
              direction={d}
              color={colors[i % colors.length]}
              exploreId={exploreId}
              onExplore={onExplore}
            />
          ))}
        </div>
      </section>

      {result.reading_order.length > 0 && (
        <section className="card">
          <h2>Where to start</h2>
          <ol className="reading-order">
            {result.reading_order.map((r) => (
              <li key={r.arxiv_id}>
                <a href={`https://arxiv.org/abs/${r.arxiv_id}`} target="_blank" rel="noreferrer">
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
    </>
  );
}

const PREVIEW_PAPERS = 3;

function DirectionBlock({
  direction,
  color,
  exploreId,
  onExplore,
}: {
  direction: Direction;
  color: string;
  exploreId: string | null;
  onExplore: (arxivId: string | null) => void;
}) {
  const preview = direction.papers.slice(0, PREVIEW_PAPERS);
  const rest = direction.papers.slice(PREVIEW_PAPERS);
  return (
    <div className="direction">
      <h3>
        <span className="swatch" style={{ background: color }} aria-hidden="true" />
        {direction.name}
        <span className="count">{direction.papers.length}</span>
      </h3>
      <p className="direction-problem">{direction.problem}</p>
      <ul className="direction-papers">
        {preview.map((p) => (
          <PaperRow key={p.arxiv_id} paper={p} exploreId={exploreId} onExplore={onExplore} />
        ))}
      </ul>
      {rest.length > 0 && (
        <details>
          <summary className="hint">
            {rest.length} more paper{rest.length === 1 ? "" : "s"}
          </summary>
          <ul className="direction-papers">
            {rest.map((p) => (
              <PaperRow key={p.arxiv_id} paper={p} exploreId={exploreId} onExplore={onExplore} />
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function PaperRow({
  paper,
  exploreId,
  onExplore,
}: {
  paper: { arxiv_id: string; title: string; published_month: string };
  exploreId: string | null;
  onExplore: (arxivId: string | null) => void;
}) {
  const active = exploreId === paper.arxiv_id;
  return (
    <li>
      <a href={`https://arxiv.org/abs/${paper.arxiv_id}`} target="_blank" rel="noreferrer">
        {paper.title}
      </a>
      <span className="paper-month">{paper.published_month}</span>
      <button
        className={active ? "explore active" : "explore"}
        onClick={() => onExplore(active ? null : paper.arxiv_id)}
      >
        {active ? "hide graph" : "graph"}
      </button>
    </li>
  );
}

// Stacked bars, papers per month per direction: answers "where is the activity flowing".
// Plain SVG in a fixed viewBox; segment identity is doubled by the legend and each
// segment's native tooltip, so color is never the only carrier.
const CHART_W = 640;
const CHART_H = 150;
const LABEL_H = 18;
const GAP = 2;

function Timeline({
  timeline,
  directions,
  colors,
}: {
  timeline: TimelinePoint[];
  directions: Direction[];
  colors: string[];
}) {
  const months = [...new Set(timeline.map((t) => t.month))].sort();
  const names = directions.map((d) => d.name);
  const byKey = new Map(timeline.map((t) => [`${t.month}|${t.direction}`, t.count]));
  const totals = months.map((m) =>
    names.reduce((sum, n) => sum + (byKey.get(`${m}|${n}`) ?? 0), 0),
  );
  const maxTotal = Math.max(...totals, 1);
  const plotH = CHART_H - LABEL_H;
  const slot = CHART_W / months.length;
  const barW = Math.min(40, slot * 0.6);
  // With many months, label every other tick so labels never collide.
  const labelEvery = months.length > 8 ? 2 : 1;

  return (
    <figure className="timeline">
      <svg
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        role="img"
        aria-label="Papers per month, stacked by research direction"
      >
        {months.map((month, mi) => {
          const x = mi * slot + (slot - barW) / 2;
          let y = plotH;
          return (
            <g key={month}>
              {names.map((name, ni) => {
                const count = byKey.get(`${month}|${name}`) ?? 0;
                if (count === 0) return null;
                const h = Math.max((count / maxTotal) * (plotH - 8) - GAP, 1.5);
                y -= h + GAP;
                return (
                  <rect
                    key={name}
                    x={x}
                    y={y}
                    width={barW}
                    height={h}
                    rx={2}
                    fill={colors[ni % colors.length]}
                  >
                    <title>{`${name}: ${count} paper${count === 1 ? "" : "s"} in ${month}`}</title>
                  </rect>
                );
              })}
              {mi % labelEvery === 0 && (
                <text className="tick" x={mi * slot + slot / 2} y={CHART_H - 4} textAnchor="middle">
                  {month}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption className="legend">
        {names.map((name, ni) => (
          <span key={name} className="legend-item">
            <span className="swatch" style={{ background: colors[ni % colors.length] }} />
            {name}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
