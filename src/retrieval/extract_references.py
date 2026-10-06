import re
import io
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pypdf

from src.retrieval.paper_metadata import extract_full_paper_profile

def extract_text_from_pdf(pdf_source: Any) -> Tuple[str, List[str]]:
    """
    Extracts text from PDF file path or file-like stream.
    Returns (full_text, pages_text).
    """
    if isinstance(pdf_source, (str, Path)):
        reader = pypdf.PdfReader(str(pdf_source))
    else:
        reader = pypdf.PdfReader(pdf_source)
        
    pages_text = []
    full_text = ""
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages_text.append(txt)
        full_text += txt + "\n"
        
    return full_text, pages_text

def isolate_bibliography(full_text: str) -> str:
    """Finds and extracts the bibliography / references section text."""
    headings = [
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*References\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*Literature Cited\s*\n',
    ]
    
    split_pos = -1
    for h in headings:
        match = re.search(h, full_text)
        if match:
            split_pos = match.start()
            break
            
    if split_pos != -1:
        return full_text[split_pos:]
    
    # Fallback search if heading isn't surrounded by empty lines
    match = re.search(r'(?i)\n\s*(references|bibliography)\s*\n', full_text)
    if match:
        return full_text[match.start():]
        
    return ""

def split_references(bib_text: str) -> List[str]:
    """
    Splits raw bibliography text into individual reference entries.
    Supports bracketed [1], numbered 1., and unnumbered Author-Year formats.
    """
    if not bib_text:
        return []

    # 1. Square bracketed pattern like [1] ... [2] ...
    bracket_pattern = r'(\[\d+\])'
    parts = re.split(bracket_pattern, bib_text)
    if len(parts) > 2:
        entries = []
        for i in range(1, len(parts), 2):
            label = parts[i]
            content = parts[i+1] if i+1 < len(parts) else ""
            clean_entry = f"{label} {content.strip()}"
            clean_entry = re.sub(r'\s+', ' ', clean_entry)
            if len(clean_entry) > 10:
                entries.append(clean_entry)
        if len(entries) >= 2:
            return entries

    # 1b. Parenthesized pattern like (1) ... (2) ... if square brackets not present
    paren_pattern = r'(\(\d+\))'
    parts = re.split(paren_pattern, bib_text)
    if len(parts) > 2:
        entries = []
        for i in range(1, len(parts), 2):
            label = parts[i]
            content = parts[i+1] if i+1 < len(parts) else ""
            clean_entry = f"{label} {content.strip()}"
            clean_entry = re.sub(r'\s+', ' ', clean_entry)
            if len(clean_entry) > 10:
                entries.append(clean_entry)
        if len(entries) >= 2:
            return entries

    # 2. Numbered pattern like 1. ... 2. ... or 1) ... 2) ...
    num_pattern = r'(\n\s*\d+[\.\)]\s+)'
    parts = re.split(num_pattern, bib_text)
    if len(parts) > 2:
        entries = []
        for i in range(1, len(parts), 2):
            label = parts[i].strip()
            content = parts[i+1] if i+1 < len(parts) else ""
            clean_entry = f"{label} {content.strip()}"
            clean_entry = re.sub(r'\s+', ' ', clean_entry)
            if len(clean_entry) > 10:
                entries.append(clean_entry)
        if len(entries) >= 2:
            return entries

    # 3. Unnumbered format (Author-Year or plain lines)
    lines = [l.strip() for l in bib_text.splitlines() if l.strip()]
    entries = []
    curr = ""

    def is_author_start(line: str) -> bool:
        if re.match(r'^[A-Z][a-zA-Z\-\']+(?:,\s*[A-Z]\.|\s+[A-Z][a-zA-Z\-\']+)', line):
            return True
        return False

    for l in lines:
        if re.match(r'(?i)^(references|bibliography|literature cited)$', l):
            continue
        if not curr:
            curr = l
        else:
            has_year = bool(re.search(r'\b(19\d\d|20[0-2]\d)\b', curr))
            if has_year and is_author_start(l):
                entries.append(curr)
                curr = l
            else:
                curr += " " + l

    if curr and not re.match(r'(?i)^(references|bibliography)$', curr.strip()):
        entries.append(curr)

    cleaned = [re.sub(r'\s+', ' ', e).strip() for e in entries if len(e.strip()) > 15]
    return cleaned if cleaned else [re.sub(r'\s+', ' ', bib_text).strip()]

def find_citation_context(full_text: str, citation_num: int, raw_ref: str) -> str:
    """Finds the in-text citation context for a given reference (searched in body text)."""
    # Isolate body text before bibliography to avoid searching within bibliography
    bib_pos = full_text.find("REFERENCES")
    if bib_pos == -1:
        bib_pos = full_text.find("References")
    body_text = full_text[:bib_pos] if bib_pos != -1 else full_text

    # Try searching for [N]
    pattern = rf'([^.\n]*?\[{citation_num}\][^.\n]*?\.)'
    match = re.search(pattern, body_text)
    if match:
        return match.group(1).strip()
        
    # Try searching by author surname from raw reference
    author_match = re.search(r'([A-Z][a-z]+)', raw_ref)
    if author_match:
        surname = author_match.group(1)
        if len(surname) > 3:
            surname_pattern = rf'([^.\n]*?\b{surname}\b[^.\n]*?\.)'
            s_match = re.search(surname_pattern, body_text)
            if s_match:
                return s_match.group(1).strip()
                
    return "In-text citation context extracted from body."

def process_pdf_document(pdf_source: Any, run_id: str = "RUN_001") -> Dict[str, Any]:
    """
    Main function to ingest PDF, extract references, assign citation IDs,
    and return structured paper profile & citation representation.
    """
    full_text, pages_text = extract_text_from_pdf(pdf_source)
    paper_profile = extract_full_paper_profile(pdf_source, full_text, pages_text)
    
    bib_text = isolate_bibliography(full_text)
    raw_references = split_references(bib_text)
    
    extracted_citations = []
    for idx, raw_ref in enumerate(raw_references, start=1):
        citation_id = f"C{idx:03d}"
        context = find_citation_context(full_text, idx, raw_ref)
        extracted_citations.append({
            "citation_id": citation_id,
            "raw_text": raw_ref,
            "citation_context": context,
            "index": idx,
        })
        
    return {
        "run_id": run_id,
        "paper_title": paper_profile["title"],
        "abstract": paper_profile["abstract"],
        "paper_profile": paper_profile,
        "total_pages": len(pages_text),
        "total_references_found": len(extracted_citations),
        "citations": extracted_citations,
    }
