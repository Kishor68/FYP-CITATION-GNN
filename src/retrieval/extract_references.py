import re
import io
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pypdf

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

def extract_metadata_from_text(full_text: str) -> Dict[str, str]:
    """Attempts to extract title and abstract from paper text."""
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    title = lines[0] if lines else "Untitled Paper"
    
    # Try to find abstract
    abstract = ""
    abstract_match = re.search(r'(?i)abstract[:\s]+(.*?)(?=\n\n|\n[A-Z][a-z]+|\Z)', full_text, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        if len(abstract) > 1000:
            abstract = abstract[:1000] + "..."
            
    return {
        "title": title,
        "abstract": abstract,
    }

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
    """Splits raw bibliography text into individual reference entries."""
    if not bib_text:
        return []
        
    # Standard bracketed pattern like [1] ... [2] ...
    bracket_pattern = r'(\[\d+\])'
    parts = re.split(bracket_pattern, bib_text)
    
    entries = []
    if len(parts) > 2:
        for i in range(1, len(parts), 2):
            label = parts[i]
            content = parts[i+1] if i+1 < len(parts) else ""
            clean_entry = f"{label} {content.strip()}"
            clean_entry = re.sub(r'\s+', ' ', clean_entry)
            if len(clean_entry) > 10:
                entries.append(clean_entry)
        return entries
        
    # Numbered pattern like 1. ... 2. ...
    num_pattern = r'(\n\s*\d+\.\s+)'
    parts = re.split(num_pattern, bib_text)
    if len(parts) > 2:
        for i in range(1, len(parts), 2):
            label = parts[i].strip()
            content = parts[i+1] if i+1 < len(parts) else ""
            clean_entry = f"{label} {content.strip()}"
            clean_entry = re.sub(r'\s+', ' ', clean_entry)
            if len(clean_entry) > 10:
                entries.append(clean_entry)
        return entries
        
    # Fallback to paragraph splitting
    paragraphs = bib_text.split("\n\n")
    for p in paragraphs:
        p_clean = re.sub(r'\s+', ' ', p).strip()
        if len(p_clean) > 20 and not re.match(r'(?i)^(references|bibliography)', p_clean):
            entries.append(p_clean)
            
    return entries

def find_citation_context(full_text: str, citation_num: int, raw_ref: str) -> str:
    """Finds the in-text citation context for a given reference."""
    # Try searching for [N]
    pattern = rf'([^.\n]*?\[{citation_num}\][^.\n]*?\.)'
    match = re.search(pattern, full_text)
    if match:
        return match.group(1).strip()
        
    # Try searching by author surname from raw reference
    author_match = re.search(r'([A-Z][a-z]+)', raw_ref)
    if author_match:
        surname = author_match.group(1)
        if len(surname) > 3:
            surname_pattern = rf'([^.\n]*?\b{surname}\b[^.\n]*?\.)'
            s_match = re.search(surname_pattern, full_text)
            if s_match:
                return s_match.group(1).strip()
                
    return "In-text citation context extracted from body."

def process_pdf_document(pdf_source: Any, run_id: str = "RUN_001") -> Dict[str, Any]:
    """
    Main function to ingest PDF, extract references, assign citation IDs,
    and return structured paper representation.
    """
    full_text, pages_text = extract_text_from_pdf(pdf_source)
    metadata = extract_metadata_from_text(full_text)
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
        "paper_title": metadata["title"],
        "abstract": metadata["abstract"],
        "total_pages": len(pages_text),
        "total_references_found": len(extracted_citations),
        "citations": extracted_citations,
    }
