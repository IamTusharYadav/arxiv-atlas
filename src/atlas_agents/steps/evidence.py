def neutralize(text: str) -> str:
    # Escape angle brackets so a crafted abstract can't forge a </paper> tag, break out of its
    # data block, and get its trailing text read as instructions. Also applied to extractor
    # output: claims are derived from untrusted abstracts, so injection can survive laundering.
    return text.replace("<", "&lt;").replace(">", "&gt;")


def paper_block(arxiv_id: str, title: str, body: str, max_chars: int | None = None) -> str:
    # arxiv_id is validated upstream so it goes in as-is; title and body are untrusted.
    if max_chars is not None:
        body = body[:max_chars]
    return f"<paper id={arxiv_id!r}>\n{neutralize(title)}\n{neutralize(body)}\n</paper>"
