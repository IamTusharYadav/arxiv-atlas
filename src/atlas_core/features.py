import re
import unicodedata

_MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)
# Commands whose argument is a reference key or URL, not prose.
_DROP_ARG = re.compile(r"\\(?:cite[tp]?|ref|eqref|label|url|href)\*?\s*\{[^{}]*\}")
_KEEP_ARG = re.compile(r"\\[a-zA-Z]+\*?\s*\{([^{}]*)\}")
_ESCAPED = re.compile(r"\\([%&#_])")
_BARE_COMMAND = re.compile(r"\\[a-zA-Z]+\*?")
_WS = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize_for_embedding(text: str) -> str:
    # Does not handle escaped \$ inside math; rare in abstracts and the cost is a few
    # lost characters, not a crash.
    text = unicodedata.normalize("NFC", text)
    text = _MATH.sub(" ", text)
    text = _DROP_ARG.sub(" ", text)
    for _ in range(3):  # unwrap nesting like \textbf{\emph{x}}
        unwrapped = _KEEP_ARG.sub(r"\1", text)
        if unwrapped == text:
            break
        text = unwrapped
    text = _ESCAPED.sub(r"\1", text)
    text = _BARE_COMMAND.sub(" ", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    return collapse_whitespace(text)
