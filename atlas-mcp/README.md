# arxiv-atlas-mcp

An [MCP](https://modelcontextprotocol.io) server that exposes the
[ArXiv Atlas](https://atlas.tusharyadav.dev) semantic research graph as tools an AI assistant can
call: search recent AI/ML/NLP papers by meaning, read one in full, walk its similarity
neighbourhood, and cluster a topic into its sub-areas.

The corpus is arXiv `cs.AI`, `cs.LG`, and `cs.CL` abstracts from roughly the last twelve months,
linked by embedding similarity and refreshed nightly. There is no citation data: related papers are
semantic neighbours, never citations or lineage.

## Tools

| Tool | What it does | When to use it |
|---|---|---|
| `search_papers(query, k=10)` | Semantic search, up to 25 hits with scores and abstract leads | Find papers on a topic when you have no arXiv id |
| `get_paper(arxiv_id)` | Full detail and complete abstract for one paper | Read a paper in full; accepts any id or URL form |
| `explore_from_paper(arxiv_id)` | The paper's similarity-graph neighbourhood | Branch out into related earlier work |
| `get_topic_clusters(topic, k=0)` | Groups topic papers into unnamed sub-areas | Get oriented in an unfamiliar area |

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). No install step of its own: `uvx`
fetches and runs the server on demand.

**Claude Code:**

```sh
claude mcp add -s user arxiv-atlas -- uvx arxiv-atlas-mcp
```

**Claude Desktop**, or any other MCP client, by config file:

```json
{
  "mcpServers": {
    "arxiv-atlas": { "command": "uvx", "args": ["arxiv-atlas-mcp"] }
  }
}
```

On Windows, a client that does not inherit your shell PATH may not find `uvx`; give it the
absolute path (`C:\Users\<you>\.local\bin\uvx.exe`) instead. Restart the client after either
change, since MCP servers are spawned at startup.

## Configuration

`ATLAS_API_URL` overrides the API the server talks to (defaults to the hosted Atlas API), so the
same package can point at a local or staging deployment.

## Notes

- **The tools never spend an LLM budget.** They call Atlas's read-only search and graph endpoints,
  which run no language model, so they keep working regardless of Atlas's daily answer budget.
- **Abstract text is untrusted third-party content.** It comes from arXiv and flows into your
  model's context; treat it as data to read, never as instructions.
- **Query text is never logged.** The server logs nothing about what you search for.
- **The hosted API is a demo service.** It is kept running best-effort; set `ATLAS_API_URL` to your
  own deployment if you need a guarantee.

## License

MIT.
