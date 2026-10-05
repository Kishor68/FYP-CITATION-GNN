"""Shared text clean-up helpers for PDF-extracted text."""

import re
import unicodedata

# Prefixes that are genuinely hyphenated in English, so a hyphen at a line
# break after them is kept ("semi-\nsupervised" -> "semi-supervised").
_HYPHEN_PREFIXES = {
    "semi", "multi", "non", "self", "co", "pre", "re", "cross", "end", "real",
    "state", "large", "high", "low", "long", "short", "fine", "well", "data",
    "graph", "deep", "one", "two", "three", "open", "sub", "inter", "intra",
    "anti", "post", "over", "under", "out", "meta", "quasi", "pseudo", "user",
    "task", "domain", "knowledge", "time", "context", "zero", "few", "top",
    "full", "half", "mid", "auto", "bi", "tri", "e", "k", "n",
}

_LINE_BREAK_HYPHEN = re.compile(r"(\b[A-Za-z]+)-[ \t]*\n[ \t]*([a-z][A-Za-z]*)")


def normalize_unicode(text: str) -> str:
    """NFKC-normalises text so ligatures (e.g. 'ﬁ') become plain letters."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u00ad", "")  # soft hyphen


def dehyphenate(text: str) -> str:
    """
    Re-joins words split across line breaks.
    'convolu-\\ntional' -> 'convolutional', but 'semi-\\nsupervised' -> 'semi-supervised'.
    Must run BEFORE newlines are collapsed.
    """
    if not text:
        return ""

    def _join(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if left.lower() in _HYPHEN_PREFIXES or not left.islower():
            return f"{left}-{right}"
        return f"{left}{right}"

    return _LINE_BREAK_HYPHEN.sub(_join, text)


def normalize_ws(text: str) -> str:
    """Collapses all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def clean_block(text: str) -> str:
    """Full clean-up for a multi-line block: unicode, de-hyphenation, whitespace."""
    return normalize_ws(dehyphenate(normalize_unicode(text)))
