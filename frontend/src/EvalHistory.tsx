import history from "../../evals/history.json";
import { SERIES } from "./palette";
import { useDarkMode } from "./useDarkMode";

type Run = {
  timestamp: string;
  relevance: number;
  faithfulness: number;
  citation_correctness: number;
  n: number;
};

const DIMENSIONS = [
  { key: "relevance", label: "Relevance" },
  { key: "faithfulness", label: "Faithfulness" },
  { key: "citation_correctness", label: "Citation correctness" },
] as const;

const W = 460;
const H = 170;
const PAD = { top: 12, right: 14, bottom: 24, left: 30 };

const runs = history as Run[];

// Scores are 1-5, but runs that stay in the top band would draw as flat lines pinned to the
// ceiling on a full axis. Take the tightest floor the data actually clears, and print it on
// the axis so a zoomed scale is never mistaken for the full one.
export function axisFloor(values: number[]): number {
  const min = Math.min(...values);
  return [4.5, 4, 3].find((floor) => values.length > 0 && min >= floor) ?? 1;
}

const day = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });

export default function EvalHistory() {
  const dark = useDarkMode();
  const colors = SERIES[dark ? "dark" : "light"];
  const latest = runs[runs.length - 1];
  if (!latest) return null;

  const lo = axisFloor(runs.flatMap((r) => DIMENSIONS.map((d) => r[d.key])));
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (runs.length === 1 ? plotW / 2 : (i * plotW) / (runs.length - 1));
  const y = (v: number) => PAD.top + plotH - ((v - lo) / (5 - lo)) * plotH;
  const ticks = [lo, (lo + 5) / 2, 5];

  return (
    <figure className="eval-history">
      <figcaption>
        Golden-set scores, one point per full scoring run. Judge ratings are 1 to 5.
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="eval-chart"
        role="img"
        aria-label={`Eval history over ${runs.length} runs. Latest: relevance ${latest.relevance}, faithfulness ${latest.faithfulness}, citation correctness ${latest.citation_correctness}, over ${latest.n} queries.`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="eval-grid" />
            <text x={PAD.left - 6} y={y(t) + 4} className="eval-tick" textAnchor="end">
              {t}
            </text>
          </g>
        ))}
        <text x={PAD.left} y={H - 6} className="eval-tick">
          {day(runs[0].timestamp)}
        </text>
        {runs.length > 1 && (
          <text x={W - PAD.right} y={H - 6} className="eval-tick" textAnchor="end">
            {day(latest.timestamp)}
          </text>
        )}

        {DIMENSIONS.map((d, di) => (
          <g key={d.key}>
            {runs.length > 1 && (
              <polyline
                className="eval-line"
                stroke={colors[di]}
                points={runs.map((r, i) => `${x(i)},${y(r[d.key])}`).join(" ")}
              />
            )}
            {runs.map((r, i) => (
              <g key={r.timestamp}>
                <circle cx={x(i)} cy={y(r[d.key])} r={4} fill={colors[di]} className="eval-dot" />
                {/* Native SVG tooltip: a hover layer without a line of tooltip state. The
                    invisible disc is the hit target, since a 4px dot is a hard thing to hit. */}
                <circle cx={x(i)} cy={y(r[d.key])} r={11} fill="transparent">
                  <title>{`${d.label} ${r[d.key]} on ${day(r.timestamp)} (${r.n} queries)`}</title>
                </circle>
              </g>
            ))}
          </g>
        ))}
      </svg>

      <ul className="eval-legend">
        {DIMENSIONS.map((d, di) => (
          <li key={d.key}>
            <span className="swatch" style={{ background: colors[di] }} />
            {d.label} <strong>{latest[d.key]}</strong>
          </li>
        ))}
      </ul>

      <p className="hint">
        {lo > 1 && `Axis covers ${lo} to 5, the band every run has stayed in. `}
        Latest run scored {latest.n} queries. A change is blocked when relevance or faithfulness
        falls more than 0.3 below the recorded baseline.
      </p>

      <details className="eval-table">
        <summary>Show the numbers</summary>
        <table>
          <thead>
            <tr>
              <th scope="col">Run</th>
              {DIMENSIONS.map((d) => (
                <th scope="col" key={d.key}>
                  {d.label}
                </th>
              ))}
              <th scope="col">Queries</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.timestamp}>
                <th scope="row">{day(r.timestamp)}</th>
                {DIMENSIONS.map((d) => (
                  <td key={d.key}>{r[d.key]}</td>
                ))}
                <td>{r.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
