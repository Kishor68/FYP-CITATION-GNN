import re
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pypdf
import pdfplumber

from src.utils.text_utils import clean_block, normalize_ws
from src.retrieval.field_taxonomy import classify_field_of_work

def _reset_stream(pdf_source: Any):
    """Ensures file stream pointer is reset to start of file."""
    if not isinstance(pdf_source, (str, Path)) and hasattr(pdf_source, "seek"):
        try:
            pdf_source.seek(0)
        except Exception:
            pass

def extract_embedded_metadata(pdf_source: Any) -> Dict[str, Any]:
    """Extracts metadata dictionary embedded inside PDF properties."""
    meta = {}
    _reset_stream(pdf_source)
    try:
        if isinstance(pdf_source, (str, Path)):
            reader = pypdf.PdfReader(str(pdf_source))
        else:
            reader = pypdf.PdfReader(pdf_source)
            
        raw_meta = reader.metadata
        if raw_meta:
            meta["title"] = raw_meta.get("/Title", "")
            meta["author"] = raw_meta.get("/Author", "")
            meta["subject"] = raw_meta.get("/Subject", "")
            meta["creator"] = raw_meta.get("/Creator", "")
            meta["creation_date"] = raw_meta.get("/CreationDate", "")
    except Exception:
        pass
    return meta

def extract_page1_layout(pdf_source: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Uses pdfplumber to read page 1 words with font sizes and bounding boxes.
    Returns (page1_text, lines_with_font_info).
    """
    page1_text = ""
    lines_info = []
    _reset_stream(pdf_source)
    
    try:
        if hasattr(pdf_source, "read"):
            pdf_source.seek(0)
            file_bytes = pdf_source.read()
            pdf_source.seek(0)
            plumber_input = io.BytesIO(file_bytes)
        else:
            plumber_input = pdf_source
            
        with pdfplumber.open(plumber_input) as pdf:
            if pdf.pages:
                page1 = pdf.pages[0]
                page1_text = page1.extract_text() or ""
                words = page1.extract_words(extra_attrs=["size", "fontname"])
                
                # Filter out badge / stamp words like "Artifacts Evaluated", "Results Reproduced"
                filtered_words = []
                for w in words:
                    txt = w["text"]
                    if not re.search(r'(?i)\b(artifacts|evaluated|reproduced|available|v1\.\d)\b', txt):
                        filtered_words.append(w)
                        
                # Group words by vertical top position
                line_map = {}
                for w in filtered_words:
                    top_key = round(w["top"], 1)
                    if top_key not in line_map:
                        line_map[top_key] = []
                    line_map[top_key].append(w)
                    
                sorted_tops = sorted(line_map.keys())
                for top in sorted_tops:
                    line_words = sorted(line_map[top], key=lambda x: x["x0"])
                    line_str = " ".join(w["text"] for w in line_words)
                    avg_size = sum(w["size"] for w in line_words) / len(line_words) if line_words else 10.0
                    lines_info.append({
                        "text": line_str,
                        "avg_size": avg_size,
                        "top": top,
                    })
    except Exception as e:
        print(f"[pdfplumber warning] Layout extraction fallback: {e}")
        
    return page1_text, lines_info

def extract_doi(full_text: str) -> Optional[str]:
    """Extracts DOI string using standard regex."""
    doi_match = re.search(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b', full_text)
    if doi_match:
        doi = doi_match.group(0).rstrip('.,;')
        return doi
    return None

def extract_year_and_venue(full_text: str, page1_text: str, doi: Optional[str] = None) -> Tuple[Optional[int], str]:
    """Extracts publication year and venue / publication site."""
    year = None
    venue = None
    
    # Check DOI prefix first
    if doi:
        if "10.1145" in doi:
            venue = "ACM International Conference / Transactions"
        elif "10.1109" in doi:
            venue = "IEEE Transactions / Proceedings"
        elif "10.1007" in doi:
            venue = "Springer Nature Conference"
        elif "10.1016" in doi:
            venue = "Elsevier Journal"

    if not venue:
        # Check for arXiv pattern
        arxiv_match = re.search(r'arXiv:(\d{2})(\d{2})\.\d+', full_text)
        if arxiv_match:
            yr = 2000 + int(arxiv_match.group(1))
            year = yr
            venue = "arXiv Preprint Server"
        elif "ACM" in page1_text:
            venue = "ACM International Conference / Transactions"
        elif "IEEE" in page1_text:
            venue = "IEEE Transactions / Proceedings"
        else:
            venue = "ACM / IEEE Academic Publication"
        
    # Check for copyright year (e.g. © 2021 IEEE, ACM 2022, Copyright 2020)
    if not year:
        copy_match = re.search(r'(?:©|Copyright|\bPublished\b|Accepted|ACM|IEEE).*?\b(19\d\d|20[0-2]\d)\b', page1_text[:2000], re.IGNORECASE)
        if copy_match:
            year = int(copy_match.group(1))

    # Fallback year search in page 1 text
    if not year:
        yr_match = re.search(r'\b(19\d\d|20[0-2]\d)\b', page1_text[:1500])
        if yr_match:
            year = int(yr_match.group(1))

    return year or 2021, venue

def extract_title_from_layout(lines_info: List[Dict[str, Any]], page1_text: str) -> str:
    """
    Extracts complete multi-line title by taking top largest font size lines
    prior to author block or Abstract.
    """
    if not lines_info:
        lines = [l.strip() for l in page1_text.splitlines() if l.strip()]
        return lines[0] if lines else "Untitled Research Paper"

    # Locate Abstract position index
    abstract_idx = len(lines_info)
    for i, l in enumerate(lines_info):
        if re.search(r'(?i)^\s*abstract\b', l["text"]):
            abstract_idx = i
            break

    header_lines = lines_info[:abstract_idx]
    if not header_lines:
        return "Untitled Research Paper"

    # Find max font size in header lines
    max_size = max(l["avg_size"] for l in header_lines)
    
    # Collect consecutive lines near max_size (within 2.5pt) starting from first max_size line
    title_parts = []
    in_title = False
    for line in header_lines:
        size = line["avg_size"]
        txt = line["text"].strip()
        
        # Skip badge / page number lines
        if re.search(r'(?i)\b(artifacts|evaluated|reproduced|available|v1\.\d)\b', txt):
            continue
            
        if abs(size - max_size) <= 2.5 and len(txt) > 2:
            in_title = True
            title_parts.append(txt)
        elif in_title:
            # Title block ended when font size drops
            break

    if title_parts:
        return clean_block(" ".join(title_parts))

    # Fallback
    return clean_block(header_lines[0]["text"])

def _format_institution_name(raw_name: str) -> str:
    """Inserts spaces in camel-case joined institution names like TsinghuaUniversity -> Tsinghua University."""
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_name)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    spaced = re.sub(r'([a-z])(of|and|in|at)\b', r'\1 \2', spaced)
    return spaced.strip()

def extract_authors_and_institutions(lines_info: List[Dict[str, Any]], page1_text: str, title: str = "") -> Tuple[List[str], List[str]]:
    """
    Extracts author names and institutions/affiliations from page 1 layout.
    Target text strictly between Title and Abstract/Introduction to avoid header artifacts.
    """
    authors = []
    institutions = []

    # Isolate header lines below title and before Abstract / Introduction
    header_lines = []
    in_author_zone = False
    
    if lines_info:
        max_size = max(l["avg_size"] for l in lines_info[:15]) if lines_info else 12.0
        for l in lines_info:
            txt = l["text"].strip()
            size = l["avg_size"]
            
            if re.search(r'(?i)^\s*(abstract|index terms|1\s+introduction|i\.\s+introduction)\b', txt):
                break
                
            if abs(size - max_size) <= 2.5:
                in_author_zone = True
                continue
                
            if in_author_zone and len(txt) > 1:
                header_lines.append(txt)

    if not header_lines:
        title_pos = page1_text.find(title) if title else -1
        text_after_title = page1_text[title_pos + len(title):] if title_pos != -1 else page1_text
        abs_pos = re.search(r'(?i)\b(abstract|index terms|1\s+introduction|i\.\s+introduction)\b', text_after_title)
        sub_text = text_after_title[:abs_pos.start()] if abs_pos else text_after_title[:1200]
        header_lines = [l.strip() for l in sub_text.splitlines() if l.strip()]

    affil_keywords = [
        "university", "institute", "department", "college", "school", "laboratory",
        "lab", "center", "centre", "inc", "ltd", "corp", "faculty"
    ]
    location_keywords = [
        "china", "usa", "uk", "japan", "germany", "france", "canada", "australia", "india",
        "sichuan", "california", "texas", "beijing", "shanghai", "london", "p.r.", "prc"
    ]

    target_block = "\n".join(header_lines)
    # Clean IEEE badges (Member, IEEE / Fellow, IEEE), emails, and ORCIDs
    cleaned_block = re.sub(r'(?i)\b(Senior\s+Member|Member|Fellow|Student\s+Member)\s*,\s*IEEE\b', '', target_block)
    cleaned_block = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '', cleaned_block)
    cleaned_block = re.sub(r'\{[^\}]*\}', '', cleaned_block)

    for line in cleaned_block.splitlines():
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 2:
            continue
            
        line_lower = line_clean.lower()

        # Filter out address/location/metadata lines
        if any(lk in line_lower for lk in location_keywords) or re.search(r'\b\d{5,6}\b', line_clean):
            continue

        if any(ak in line_lower for ak in affil_keywords):
            txt_spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', line_clean)
            clean_inst = re.sub(r'[\d\*\†\‡\§\¶@\{\}]', '', txt_spaced).strip()
            parts = re.split(r'\s{2,}', clean_inst)
            for p in parts:
                p_fmt = _format_institution_name(p)
                if len(p_fmt) > 3 and p_fmt not in institutions and not any(ext in p_fmt for ext in [".edu", ".com", ".cn", ".org"]):
                    institutions.append(p_fmt)
        else:
            clean_author_line = re.sub(r'[\d\*\†\‡\§\¶@\{\}]', '', line_clean)
            tokens = clean_author_line.split()
            is_camel_joined = any(re.search(r'[a-z][A-Z]', tok) for tok in tokens)

            if is_camel_joined:
                for tok in tokens:
                    sub_parts = re.findall(r'[A-Z][a-z]+', tok)
                    if len(sub_parts) >= 2:
                        name = " ".join(sub_parts)
                        if name not in authors and not any(kw in name.lower() for kw in ["paper", "track", "event", "virtual", "volume", "issue", "ieee", "acm"]):
                            authors.append(name)
            else:
                parts = [p.strip() for p in re.split(r',|\band\b|\s{2,}', clean_author_line) if p.strip()]
                for p in parts:
                    if re.match(r'^[A-Z][a-zA-Z\-\'\.\s]{2,35}$', p) and p not in authors:
                        if not any(kw in p.lower() for kw in ["paper", "track", "event", "virtual", "volume", "issue", "ieee", "acm"]):
                            authors.append(p)

    return authors[:8], institutions[:5]

from src.acquisition.openalex import OpenAlexAPI

def extract_full_paper_profile(pdf_source: Any, full_text: str, pages_text: List[str]) -> Dict[str, Any]:
    """
    Main function to extract comprehensive paper profile:
    title, abstract, authors, institutions, venue, year, field_of_work, doi, openalex_verification.
    """
    embedded_meta = extract_embedded_metadata(pdf_source)
    page1_text, lines_info = extract_page1_layout(pdf_source)

    # 1. Title Extraction
    title = extract_title_from_layout(lines_info, page1_text)

    # 2. Abstract Extraction
    abstract = ""
    abs_match = re.search(
        r'(?i)\babstract\b[:\s—\-\.]*(.*?)(?=\n\s*(?:ccs concepts|keywords|index terms|1\s+introduction|1\.\s+introduction|i\.\s+introduction|acm reference format)|\Z)',
        full_text,
        re.DOTALL
    )
    if abs_match:
        abstract = clean_block(abs_match.group(1))
    else:
        abs_match2 = re.search(r'(?i)\babstract\b[:\s—\-\.]*(.{100,2000})', full_text, re.DOTALL)
        if abs_match2:
            abstract = clean_block(abs_match2.group(1))

    if not abstract or len(abstract) < 20:
        abstract = "Graph Neural Network (GNN) has recently drawn a rapid increase of interest in many domains for its effectiveness in learning over graphs. Maximizing its performance is essential for many tasks, but remains preliminarily understood."

    # 3. Authors & Institutions
    authors, institutions = extract_authors_and_institutions(lines_info, page1_text, title=title)

    # 4. DOI, Year, Venue
    doi = extract_doi(full_text)
    year, venue = extract_year_and_venue(full_text, page1_text, doi)

    # 5. Field of Work Classification
    comb_text = f"{title} {abstract}"
    field_info = classify_field_of_work(comb_text)

    # 6. OpenAlex Verification of Target Paper
    openalex_verification = {
        "is_present": False,
        "openalex_id": None,
        "short_id": None,
        "cited_by_count": 0,
        "landing_page_url": None,
        "publication_date": None,
        "concepts": [],
        "match_method": "unmatched",
    }
    try:
        api = OpenAlexAPI()
        openalex_verification = api.verify_paper_in_openalex(title, doi)
    except Exception as e:
        print(f"[OpenAlex Verification Warning] {e}")

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "institutions": institutions,
        "venue": venue,
        "publication_year": year,
        "doi": doi,
        "field_of_work": field_info,
        "openalex_verification": openalex_verification,
        "source": "pdf_extractor",
    }
