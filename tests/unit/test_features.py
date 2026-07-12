import unicodedata

from atlas_core.features import collapse_whitespace, normalize_for_embedding


def test_strips_inline_and_display_math() -> None:
    text = "We prove $O(n \\log n)$ bounds and $$\\sum_i x_i$$ convergence."
    assert normalize_for_embedding(text) == "We prove bounds and convergence."


def test_unwraps_formatting_commands_including_nested() -> None:
    assert normalize_for_embedding("\\textbf{\\emph{Transformers}} are \\emph{great}") == (
        "Transformers are great"
    )


def test_drops_citation_and_url_arguments() -> None:
    text = "As \\cite{smith2024} shows, see \\url{https://example.com} for code."
    assert normalize_for_embedding(text) == "As shows, see for code."


def test_unescapes_special_characters() -> None:
    assert normalize_for_embedding("50\\% accuracy at 3\\&4 bits") == "50% accuracy at 3&4 bits"


def test_removes_bare_commands_and_braces() -> None:
    assert normalize_for_embedding("results \\newline shown {here}") == "results shown here"


def test_normalizes_unicode_to_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Café")
    assert decomposed != "Café"
    assert normalize_for_embedding(decomposed) == "Café"


def test_collapse_whitespace() -> None:
    assert collapse_whitespace("  a\n\t b\r\n c  ") == "a b c"
