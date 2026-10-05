import re
from typing import Dict, Any, Optional, List
from src.utils.text_utils import clean_block, normalize_ws

def normalize_title(title: str) -> str:
    """Normalizes title string for exact comparison (lowercase, alphanumeric only)."""
    if not title:
        return ""
    normalized = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower())
    return re.sub(r'\s+', ' ', normalized).strip()

def _split_into_sentences(text: str) -> List[str]:
    """
    Splits reference text by periods, ignoring periods in author initials (e.g. 'T. N.'),
    abbreviations ('et al.', 'ed.', 'pp.', 'Vol.', 'No.'), and numbers ('v1.0').
    """
    # Temporarily mask initials and common abbreviations
    masked = text
    # Mask single capital letter initials like " T. " or " T. N. "
    masked = re.sub(r'\b([A-Z])\.', r'\1___DOT___', masked)
    # Mask common reference abbreviations
    abbrevs = ["et al", "ed", "eds", "vol", "no", "pp", "p", "fig", "dr", "prof", "vs", "ie", "eg", "dept"]
    for abb in abbrevs:
        masked = re.sub(rf'\b({abb})\.', r'\1___DOT___', masked, flags=re.IGNORECASE)
        
    # Split by period followed by whitespace or end of string
    raw_parts = re.split(r'\.\s+|\.$', masked)
    
    parts = []
    for p in raw_parts:
        # Restore dots
        unmasked = p.replace('___DOT___', '.').strip()
        if unmasked:
            parts.append(unmasked)
            
    return parts

def parse_reference_fields(raw_text: str) -> Dict[str, Any]:
    """
    Parses a raw reference string into candidate fields:
    - title
    - lead_author
    - all_authors
    - year
    - venue
    """
    cleaned = clean_block(raw_text)
    
    # Strip leading label like [1], (1), 1., etc.
    cleaned_no_label = re.sub(r'^\s*(\[\d+\]|\(\d+\)|\d+\.)\s*', '', cleaned)
    
    # Extract year (4 digit number 1900-2029)
    year_match = re.search(r'\b(19\d\d|20[0-2]\d)\b', cleaned_no_label)
    year = int(year_match.group(1)) if year_match else None

    # Title extraction strategies:
    title = None
    
    # Strategy 1: Title inside quotes "..." or “...”
    quote_match = re.search(r'["“](.*?)["”]', cleaned_no_label)
    if quote_match and len(quote_match.group(1).strip()) > 5:
        title = quote_match.group(1).strip()

    # Strategy 2: Split by sentence/period while respecting initials
    parts = _split_into_sentences(cleaned_no_label)
    
    if not title and parts:
        # Check parts for title candidate
        # If year is in parentheses like (2017), parts often look like:
        # Part 0: "Kipf, T. N., & Welling, M. (2017)"
        # Part 1: "Semi-supervised classification with graph convolutional networks"
        # Part 2: "ICLR"
        candidate_idx = 1 if len(parts) > 1 else 0
        
        # If part 0 doesn't contain year or authors, or if part 0 is author list
        if len(parts) > 1:
            # Check if part 0 is predominantly author names / year
            part0 = parts[0]
            # Strip year from part0 if present
            part0_clean = re.sub(r'\(?\b(19\d\d|20[0-2]\d)\b\)?', '', part0).strip()
            
            # If part0 looks like authors (contains commas, &, et al., or initials)
            if re.search(r'(,|&|\bet al\b|[A-Z]\.)', part0_clean) or len(part0_clean) < 40:
                title = parts[1]
            else:
                title = parts[0]
        else:
            title = parts[0]

    # Clean extracted title
    if title:
        # Remove trailing/leading punctuation and year patterns
        title = re.sub(r'^\s*[\"\“\']|[\"\”\']\s*$', '', title)
        title = re.sub(r'\s*\(?\b(19\d\d|20[0-2]\d)\b\)?\s*$', '', title).strip()

    # Lead author extraction
    author_match = re.search(r'^([A-Z][a-zA-Z\-]+)', cleaned_no_label)
    lead_author = author_match.group(1) if author_match else None
    
    # Venue candidate (last part if available)
    venue = parts[-1] if len(parts) > 2 else None
    
    return {
        "raw_text": raw_text,
        "clean_text": cleaned_no_label,
        "title": title or cleaned_no_label,
        "lead_author": lead_author,
        "year": year,
        "venue": venue,
    }
