# ADR 0004: a versioned read-only API surface for an MCP server

Date: 2026-07-23
Status: accepted

## Context

The next increment (v2.0.0) publishes an MCP server, `arxiv-atlas-mcp`, so an assistant such
as Claude can search Atlas, pull a paper, walk the similarity graph, cluster a topic, and
find papers bridging two areas. The value is that Atlas's corpus and graph become tools an
agent reaches for.

Two facts about the existing API force decisions before any MCP code:

- The web app calls `POST /api/query`, `POST /api/landscape`, and reads `GET /api/graph` and
  `GET /api/status`. Those paths are compiled into the deployed frontend bundle. There is no
  general search endpoint, and `GET /api/graph` returns only id/title/category per node, no
  abstract.
- Once someone installs an MCP package pinned to a path, changing that path breaks *their*
  client silently, from a deploy they never see. That is the one kind of technical debt that
  is trivial now and expensive later.

The corpus constraints from ADR 0001 and ADR 0003 still bind and shape what the tools can
honestly claim:

- Abstracts only, three categories, a rolling ~12-month window, semantic-similarity edges
  and no citation data.
- Edges are stored one-directionally on the newer paper, so graph traversal only reaches
  *older* similar work. Measured over 20k papers: the first backfill month (2507) has ~51%
  of papers with zero edges because it was ingested before a corpus existed to link into;
  everything from 2509 on is fully connected. So "what built on this paper" is unanswerable
  today, and a fresh paper can legitimately return few neighbours.

## Decision

Add a **versioned, read-only surface under `/api/v1/`** that the MCP package pins against,
served alongside the unversioned `/api/*` the frontend uses. Four routes, each a thin wrapper
over an existing `VectorStore` method, none calling Bedrock:

- `GET /api/v1/search?q&k`: embed the query, one vector search, truncated results.
- `GET /api/v1/paper/{id}`: full detail for one paper; accepts every common id form.
- `GET /api/v1/clusters?q&k`: retrieve wide, k-means, return **unnamed** groups.
- `GET /api/v1/graph/{id}`: the graph neighbourhood with an abstract lead per node.

The load-bearing choices:

- **No Bedrock on any v1 route.** The web app's budget guard gates only `/api/query` and
  `/api/landscape`. Keeping v1 model-free means an MCP client keeps working when the day's
  LLM budget is exhausted, which is the surface's central promise. `get_topic_clusters` is
  the one that made this non-trivial: the only clustering that *names* directions is the
  landscape pipeline, which runs Haiku and Sonnet through the budget guard. So v1 clusters
  return membership only (k-means over stored vectors, pure math) and the calling model names
  them from the titles and leads it gets back. Cheaper, budget-free, and it leans on the fact
  that a capable model is already on the other end.
- **Truncated abstracts in list responses (~400 chars), full text only via `paper`.** Ten
  full abstracts is several thousand tokens of a client's context per call and multi-hop
  traversal multiplies it. The lead is enough to judge relevance; the detail call is there
  when it is not.
- **Every response carries `{categories, size, last_ingested_at}`.** A client can caveat
  coverage honestly, and a stalled nightly ingest becomes visible instead of silently serving
  stale results. `last_ingested_at` is TTL-cached (5 min) because the underlying scan is
  O(corpus); a five-minute-stale freshness stamp on a nightly corpus is acceptable.
- **Empty results are an empty list plus a note, never an error.** Models handle "no matches;
  the corpus covers cs.AI/LG/CL" far better than an exception.
- **Abstracts are untrusted third-party text.** They flow into a client model's context, so
  they are a prompt-injection vector; the fields are documented as data, never instructions,
  and never concatenated into instruction-shaped prose.
- **Graph traversal is framed as related earlier work, never lineage.** The reframe is the
  honest response to the one-directional edge fact above, not a limitation to paper over. A
  paper with no strong edges returns a note saying so rather than an empty mystery. No reverse
  edge index is built; that would be real ingestion work, deferred until measured need.

## Consequences

The MCP surface is honest and cheap by construction: every tool sits on infrastructure that
already exists, and none of it can drain or be blocked by the LLM budget. The cost is a second
API contract to keep stable, which is the point of versioning it: `/api/v1/*` shapes are frozen
once published, and a breaking change means `/api/v2/*` beside it, never a silent edit. The
frontend's unversioned paths are unaffected and stay served.

What this surface deliberately cannot do, from the same data fence: forward-in-time or
citation traversal (needs a reverse index or citation ingestion, out of scope per ADR 0001),
named clusters server-side (would reintroduce model spend and break the budget-free promise),
and full-text anything (abstracts only). The 2507 orphan cohort is left as-is; the graph route
notes low connectivity rather than a re-link running. If the MCP `find_bridge_papers` tool does
not convince on hand-validation, it ships as four tools and the ADR-style honesty is that the
scoring was not good enough, not that the surface failed.
