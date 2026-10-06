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
    # Mask journal category tags like [J]., [C]., [M]., [D]., [R]., [P].
    masked = re.sub(r'\[([A-Z])\]\.', r'[\1]___DOT___', masked)
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
    
    # Strip leading label like [1], (1), 1., C001, etc.
    cleaned_no_label = re.sub(r'^\s*(\[\d+\]|\(\d+\)|\d+\.|C\d+)\s*', '', cleaned)
    
    # Extract year (4 digit number 1900-2029)
    year_match = re.search(r'\b(19\d\d|20[0-2]\d)\b', cleaned_no_label)
    year = int(year_match.group(1)) if year_match else None

    # Title extraction strategies:
    title = None
    
    # Strategy 1: Title inside quotes "..." or “...”
    quote_match = re.search(r'["“](.*?)["”]', cleaned_no_label)
    if quote_match and len(quote_match.group(1).strip()) > 5:
        title = quote_match.group(1).strip()

    # Strategy 2: Check for journal tag [J]., [C]., [M]. or et al. boundary
    if not title:
        # Check for [J]., [C]., [M]., [D]. tag which separates Title from Venue
        tag_match = re.search(r'(.*?)\s*\[[A-Z](/OL)?\]\.?\s*(.*)', cleaned_no_label)
        if tag_match:
            before_tag = tag_match.group(1).strip()
            # If before_tag contains et al. or author list
            et_match = re.search(r'\bet\s+al\b\.?\s*,?\s*(.*)', before_tag, re.IGNORECASE)
            if et_match:
                title = et_match.group(1).strip()
            else:
                # Split before_tag by period to separate author from title
                b_parts = _split_into_sentences(before_tag)
                title = b_parts[-1] if b_parts else before_tag
        else:
            # Check for et al. boundary
            et_match = re.search(r'\bet\s+al\b\.?\s*,?\s*(.*)', cleaned_no_label, re.IGNORECASE)
            if et_match:
                after_et = et_match.group(1).strip()
                a_parts = _split_into_sentences(after_et)
                title = a_parts[0] if a_parts else after_et
            else:
                parts = _split_into_sentences(cleaned_no_label)
                valid_parts = [p.strip() for p in parts if not re.match(r'^\(?\s*(19\d\d|20[0-2]\d)\s*\)?$', p.strip())]
                if len(valid_parts) >= 2:
                    p0 = valid_parts[0]
                    p0_clean = re.sub(r'\(?\b(19\d\d|20[0-2]\d)\b\)?', '', p0).strip()
                    is_author = bool(re.search(r'(,|&|\bet al\b|\b[A-Z]\.|\band\b)', p0_clean)) or len(p0_clean) < 45
                    title = valid_parts[1] if is_author else valid_parts[0]
                elif valid_parts:
                    title = valid_parts[0]

    # Clean extracted title
    if title:
        # Repair broken PDF line-break hyphens like "Convolu- tional" -> "Convolutional"
        title = re.sub(r'(\b[a-zA-Z]+)-\s+([a-zA-Z]+\b)', r'\1\2', title)
        # Strip journal category tags like [J], [C], [M]
        title = re.sub(r'\[[A-Z](/OL)?\]', '', title)
        # Strip trailing/leading quotes, commas, colons, and punctuation
        title = re.sub(r'^[,\.:\s"“\']+|[,\.:\s"”\']+$', '', title)
        title = re.sub(r'\s*\(?\b(19\d\d|20[0-2]\d)\b\)?\s*$', '', title)
        # Strip venue prefix/suffix if present
        title = re.sub(r'\s+In\s+[A-Z].*$', '', title)
        title = re.sub(r'^[,\.:\s"“\']+|[,\.:\s"”\']+$', '', title)
        title = title.strip()

    # Fallback to cleaned text if title was not isolated cleanly
    if not title or len(title) < 5:
        title = cleaned_no_label

    # Lead author extraction
    author_match = re.search(r'^([A-Z][a-zA-Z\-]+)', cleaned_no_label)
    lead_author = author_match.group(1) if author_match else None
    
    parts_all = _split_into_sentences(cleaned_no_label)
    venue = parts_all[-1] if len(parts_all) > 2 else None
    
    return {
        "raw_text": raw_text,
        "clean_text": cleaned_no_label,
        "title": title,
        "lead_author": lead_author,
        "year": year,
        "venue": venue,
    }
