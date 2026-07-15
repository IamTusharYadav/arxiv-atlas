import type { TraceStep } from "./api";

// Steps with no model call (retrieval) carry no tokens and cost nothing; showing "0" for them
// reads as a measurement rather than an absence, so they get a dash.
function tokens(step: TraceStep): string {
  if (!step.input_tokens && !step.output_tokens) return "-";
  return `${step.input_tokens.toLocaleString()} / ${step.output_tokens.toLocaleString()}`;
}

function cost(step: TraceStep): string {
  return step.cost_usd ? `$${step.cost_usd.toFixed(4)}` : "-";
}

export default function TracePanel({ trace, total }: { trace: TraceStep[]; total: number }) {
  if (trace.length === 0) return null;
  return (
    <details className="card trace">
      <summary>
        Agent trace
        <span className="trace-meta">
          {trace.length} steps &middot; ${total.toFixed(4)}
        </span>
      </summary>
      <div className="trace-scroll">
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>What it did</th>
              <th className="num">Tokens in / out</th>
              <th className="num">Cost</th>
            </tr>
          </thead>
          <tbody>
            {trace.map((step, i) => (
              <tr key={i}>
                <td className="step-cell">{step.step}</td>
                <td>{step.summary}</td>
                <td className="num">{tokens(step)}</td>
                <td className="num">{cost(step)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr>
              <td colSpan={3}>Total</td>
              <td className="num">${total.toFixed(4)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </details>
  );
}
