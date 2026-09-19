import re
from typing import Dict, Any, Optional

def parse_reference_fields(raw_text: str) -> Dict[str, Any]:
    """
    Parses a raw reference text into structured candidate fields:
    - title
    - author (lead author surname/name)
    - year
    """
    # Remove leading numbers/brackets like [1] or 1.
    clean_text = re.sub(r'^\s*(\[\d+\]|\d+\.)\s*', '', raw_text)
    
    # Extract year (4 digit number between 1900 and 2029)
    year_match = re.search(r'\b(19\d\d|20[0-2]\d)\b', clean_text)
    year = int(year_match.group(1)) if year_match else None
    
    # Extract candidate title
    # Strategy 1: Title in quotes "..."
    quote_match = re.search(r'["“](.*?)["”]', clean_text)
    title = None
    if quote_match:
        title = quote_match.group(1).strip()
        
    # Strategy 2: Title between author/year and source
    if not title:
        # Split by periods
        parts = [p.strip() for p in clean_text.split('.') if len(p.strip()) > 5]
        if len(parts) >= 2:
            # Usually part 0 is author, part 1 is title
            title = parts[1]
        elif len(parts) == 1:
            title = parts[0]
        else:
            title = clean_text
            
    # Normalize title
    if title:
        title = re.sub(r'^\d+\s*', '', title).strip()
        
    # Extract lead author surname
    author_match = re.search(r'^([A-Z][a-zA-Z\-]+)', clean_text)
    lead_author = author_match.group(1) if author_match else None
    
    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "title": title or clean_text,
        "lead_author": lead_author,
        "year": year,
    }

def normalize_title(title: str) -> str:
    """Normalizes title string for exact comparison (lowercase, alphanumeric only)."""
    if not title:
        return ""
    normalized = re.sub(r'[^a-zA-Z0-9\s]', '', title.lower())
    return re.sub(r'\s+', ' ', normalized).strip()
