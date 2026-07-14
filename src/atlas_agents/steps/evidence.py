"""One builder for the delimited data blocks that carry untrusted paper text into prompts.

Titles and abstracts come from arXiv (and ultimately the open internet). Wrapping them in a
<paper> block only contains them if the text cannot forge the delimiter: an abstract holding
a literal </paper> would otherwise close the block early and leave the rest floating as
free instructions (plan risk 5, injection via abstracts). Neutralizing angle brackets in one
place means no step can forget to, and a new step gets containment for free.
"""


def _neutralize(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def paper_block(arxiv_id: str, title: str, body: str, max_chars: int | None = None) -> str:
    """A <paper id=...> block with untrusted title and body angle-escaped. `arxiv_id` is a
    validated id, not untrusted, so it stays verbatim."""
    if max_chars is not None:
        body = body[:max_chars]
    return f"<paper id={arxiv_id!r}>\n{_neutralize(title)}\n{_neutralize(body)}\n</paper>"
