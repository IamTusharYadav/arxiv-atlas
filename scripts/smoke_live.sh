#!/usr/bin/env bash
# Post-deploy smoke against the live endpoint (commit 42). Two tiers, on purpose:
#
#   Gate (exits non-zero on failure, so deploy.yml repoints the alias to the previous version):
#     GET /api/status returns corpus_size, and GET /api/graph/<id> returns a center. Both are
#     deterministic and free (no Bedrock), so a rollback never thrashes on a model flake or spends
#     per deploy. The graph check exercises the real read path (store.get + edge resolution) that
#     status, which only counts, cannot.
#
#   Alert-only (never exits non-zero): one canned async query end to end, so a broken agent loop is
#     visible in the deploy log without gating the rollback on Bedrock. Costs ~$0.05 per deploy.
#
# Usage: scripts/smoke_live.sh <base_url> [paper_id]
# paper_id is a stable in-corpus arXiv id; unset skips the graph gate (still passes).
set -euo pipefail

BASE="${1:?usage: smoke_live.sh <base_url> [paper_id]}"
PAPER_ID="${2:-}"

echo "smoke: GET $BASE/api/status"
status=$(curl -fsS --max-time 30 --retry 5 --retry-connrefused --retry-delay 10 "$BASE/api/status")
echo "$status"
echo "$status" | grep -q '"corpus_size"' || { echo "::error::status has no corpus_size"; exit 1; }

if [ -n "$PAPER_ID" ]; then
  echo "smoke: GET $BASE/api/graph/$PAPER_ID"
  graph=$(curl -fsS --max-time 30 --retry 3 --retry-delay 5 "$BASE/api/graph/$PAPER_ID")
  echo "$graph" | grep -q '"center"' || { echo "::error::graph has no center for $PAPER_ID"; exit 1; }
else
  echo "::notice::SMOKE_PAPER_ID unset; skipping the /api/graph gate (set a stable in-corpus id to enable)"
fi

echo "gate passed."

# --- alert-only: one live query end to end. Warns, never fails. ---
alert_query() {
  local q resp job_id poll st
  q='{"question":"What is the current state of KV-cache compression?"}'
  echo "alert-only: POST $BASE/api/query"
  resp=$(curl -fsS --max-time 30 -H 'content-type: application/json' -d "$q" "$BASE/api/query") \
    || { echo "::warning::canned query POST failed"; return 0; }
  # A cache hit answers inline (has "brief"); a miss returns a job id to poll.
  if echo "$resp" | grep -q '"brief"'; then echo "alert-only: inline cache hit, ok"; return 0; fi
  job_id=$(echo "$resp" | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
  [ -n "$job_id" ] || { echo "::warning::no job_id in POST response: $resp"; return 0; }
  echo "alert-only: polling job $job_id"
  for _ in $(seq 1 40); do  # ~200s: covers a cold worker plus a multi-round loop
    sleep 5
    poll=$(curl -fsS --max-time 30 "$BASE/api/query/$job_id") || continue
    st=$(echo "$poll" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
    case "$st" in
      done)  echo "alert-only: query done"; return 0 ;;
      error) echo "::warning::canned query errored: $poll"; return 0 ;;
    esac
  done
  echo "::warning::canned query did not finish in ~200s"
}
alert_query || true
echo "smoke complete."
