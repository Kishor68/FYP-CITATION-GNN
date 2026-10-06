"""Shared text clean-up helpers for PDF-extracted text and reference parsing."""

import re
import unicodedata

# Prefixes that are genuinely hyphenated in English compounds, so a hyphen at a line
# break or space after them is kept ("semi-supervised" -> "semi-supervised").
_HYPHEN_PREFIXES = {
    "semi", "multi", "non", "self", "co", "pre", "re", "cross", "end", "real",
    "state", "large", "high", "low", "long", "short", "fine", "well", "data",
    "graph", "deep", "one", "two", "three", "open", "sub", "inter", "intra",
    "anti", "post", "over", "under", "out", "meta", "quasi", "pseudo", "user",
    "task", "domain", "knowledge", "time", "context", "zero", "few", "top",
    "full", "half", "mid", "auto", "bi", "tri", "micro", "macro", "super",
    "node", "hyper", "spatio", "spatial", "temporal", "channel",
}

# Common English suffixes and word endings that indicate broken split words
_BROKEN_WORD_SUFFIXES = {
    "ing", "ings", "tion", "tions", "sion", "sions", "ed", "er", "ers",
    "ment", "ments", "al", "ally", "able", "ible", "ity", "ities", "ive",
    "ness", "ic", "ical", "ics", "less", "ful", "est", "ance", "ence",
    "ant", "ent", "ism", "ist", "ists", "ize", "ized", "izing", "ous",
    "ory", "ary", "ks", "d", "es", "t", "s", "one", "ods"
}


def normalize_unicode(text: str) -> str:
    """NFKC-normalises text so ligatures (e.g. 'ﬁ') become plain letters."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u00ad", "")  # soft hyphen


def dehyphenate(text: str) -> str:
    """
    Re-joins words split across line breaks or with spurious spaces after hyphens.
    Examples:
    'learn-\\ning' -> 'learning'
    'learn- ing' -> 'learning'
    'Understand-\\ning' -> 'Understanding'
    'Understand- ing' -> 'Understanding'
    'con-\\ntrast' -> 'contrast'
    'semi-\\nsupervised' -> 'semi-supervised'
    """
    if not text:
        return ""

    def _join(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        left_lower = left.lower()
        right_lower = right.lower()

        # If left is a standard compound prefix, preserve hyphen
        if left_lower in _HYPHEN_PREFIXES:
            return f"{left}-{right}"
            
        # If right is a common suffix or left is incomplete, join without hyphen
        if right_lower in _BROKEN_WORD_SUFFIXES or len(right) <= 3 or not left_lower.endswith(('e', 'a', 'i', 'o', 'u', 'y', 'l', 'r', 's', 't', 'n', 'c', 'k', 'g', 'p', 'm')):
            return f"{left}{right}"
            
        # Default for line-break split words: join without hyphen
        return f"{left}{right}"

    # First pass: hyphens across newlines
    text = re.sub(r"(\b[A-Za-z]+)-[ \t]*\n[ \t]*([a-zA-Z]+)", _join, text)
    # Second pass: hyphens followed by whitespace (after collapsing newlines or from PDF streams)
    text = re.sub(r"(\b[A-Za-z]+)-[ \t]+([a-zA-Z]+)", _join, text)
    # Third pass: space before hyphen in broken word (e.g. "learn -ing" or "learn - ing")
    text = re.sub(r"(\b[A-Za-z]+)[ \t]+-[ \t]*([a-zA-Z]+)", _join, text)

    return text


def normalize_ws(text: str) -> str:
    """Collapses all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def clean_reference_text(text: str) -> str:
    """
    Full clean-up for reference entries and raw_text from PDF extraction:
    - Normalizes unicode & ligatures
    - Repairs broken hyphenated words ('learn- ing' -> 'learning', 'Understand- ing' -> 'Understanding')
    - Cleans spurious whitespace around punctuation & quotes
    - Collapses extra whitespace
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = normalize_unicode(text)
    
    # Clean PDF diacritic / ligature artifacts (e.g. 'Veli ˇckovi´c' -> 'Velickovic')
    text = re.sub(r'[ˇ´`^~¨]', '', text)

    # 2. Repair broken hyphenated words
    text = dehyphenate(text)

    # 3. Fix spurious whitespace around punctuation
    # Remove space before comma, period, colon, semicolon, question mark
    text = re.sub(r'\s+([,\.:;\?!])', r'\1', text)
    # Fix spaces around opening/closing quotes
    text = re.sub(r'([“"\'\(])\s+', r'\1', text)
    text = re.sub(r'\s+([”"\'\)])', r'\1', text)
    # Fix author initials space like "Y . " -> "Y. " or "C. - I." -> "C.-I."
    text = re.sub(r'\b([A-Z])\s+\.', r'\1.', text)
    text = re.sub(r'\b([A-Z]\.)\s*-\s*([A-Z]\.)', r'\1-\2', text)

    # 4. Collapse whitespace
    text = normalize_ws(text)
    return text


def clean_block(text: str) -> str:
    """Full clean-up for a multi-line block: unicode, de-hyphenation, whitespace."""
    return clean_reference_text(text)

