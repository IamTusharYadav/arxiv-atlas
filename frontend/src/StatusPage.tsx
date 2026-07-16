import { useEffect, useState } from "react";
import { getStatus } from "./api";

type Health =
  | { kind: "checking" }
  | { kind: "up"; corpusSize: number }
  | { kind: "down"; message: string };

export default function StatusPage() {
  const [health, setHealth] = useState<Health>({ kind: "checking" });

  useEffect(() => {
    let live = true;
    getStatus()
      .then((s) => live && setHealth({ kind: "up", corpusSize: s.corpus_size }))
      .catch(
        (err: unknown) =>
          live &&
          setHealth({
            kind: "down",
            message: err instanceof Error ? err.message : "unreachable",
          }),
      );
    return () => {
      live = false;
    };
  }, []);

  return (
    <section className="card status-page">
      <h2>Status</h2>
      <dl>
        <dt>Query API</dt>
        <dd>
          {health.kind === "checking" && <span className="muted">checking&hellip;</span>}
          {health.kind === "up" && (
            <>
              <span className="dot up" aria-hidden="true" />
              operational
            </>
          )}
          {health.kind === "down" && (
            <>
              <span className="dot down" aria-hidden="true" />
              unreachable <span className="muted">({health.message})</span>
            </>
          )}
        </dd>

        <dt>Corpus</dt>
        <dd>
          {health.kind === "up" ? (
            <>{health.corpusSize.toLocaleString()} papers indexed</>
          ) : (
            <span className="muted">unknown</span>
          )}
        </dd>

        <dt>Scope</dt>
        <dd>
          arXiv abstracts in cs.AI, cs.LG and cs.CL, refreshed nightly
        </dd>

        <dt>Spend controls</dt>
        <dd>
          A per-query cap and a fail-closed daily cap bound cost. When the daily cap is reached,
          live queries pause and saved example runs are served until the next UTC day.
        </dd>
      </dl>
      <p className="hint">
        Eval history lands here in the next phase, once nightly scoring is publishing results.
      </p>
    </section>
  );
}
