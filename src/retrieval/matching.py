from typing import Dict, Any, List, Optional
from rapidfuzz import fuzz
from src.retrieval.paper_lookup import parse_reference_fields, normalize_title
from src.acquisition.openalex import OpenAlexAPI
from src.config import FUZZY_MATCH_THRESHOLD

class ReferenceMatcher:
    """Implements 5-stage OpenAlex paper matching cascade."""

    def __init__(self, openalex_api: Optional[OpenAlexAPI] = None):
        self.api = openalex_api or OpenAlexAPI()

    def match_reference(self, citation_id: str, raw_text: str, citation_context: str) -> Dict[str, Any]:
        """
        Executes the 5-stage matching cascade for a single extracted reference:
        Stage 1: Exact normalized title match
        Stage 2: Title + Author + Year
        Stage 3: Fuzzy title similarity
        Stage 4: OpenAlex API search query fallback
        Stage 5: Manual review fallback
        """
        parsed = parse_reference_fields(raw_text)
        query_title = parsed["title"]
        norm_query_title = normalize_title(query_title)
        
        # Candidate search from OpenAlex
        candidates = self.api.search_works_by_title(query_title, max_results=5)
        
        best_match = None
        match_method = "none"
        match_score = 0.0
        match_status = "unmatched"
        
        if candidates:
            # Stage 1: Exact normalized title match
            for candidate in candidates:
                cand_title = candidate.get("display_name", "")
                norm_cand_title = normalize_title(cand_title)
                
                if norm_query_title and norm_query_title == norm_cand_title:
                    best_match = candidate
                    match_method = "exact_normalized_title"
                    match_score = 1.0
                    match_status = "matched"
                    break
                    
            # Stage 2: Title + Lead Author + Year
            if not best_match and parsed["lead_author"] and parsed["year"]:
                for candidate in candidates:
                    cand_title = candidate.get("display_name", "")
                    cand_year = candidate.get("publication_year")
                    sim = fuzz.ratio(norm_query_title, normalize_title(cand_title))
                    
                    if sim > 75 and cand_year == parsed["year"]:
                        best_match = candidate
                        match_method = "title_author_year"
                        match_score = float(sim / 100.0)
                        match_status = "matched"
                        break

            # Stage 3: Fuzzy title similarity
            if not best_match:
                best_sim = 0.0
                top_cand = None
                for candidate in candidates:
                    cand_title = candidate.get("display_name", "")
                    sim = fuzz.ratio(norm_query_title, normalize_title(cand_title))
                    if sim > best_sim:
                        best_sim = sim
                        top_cand = candidate
                        
                if top_cand and best_sim >= FUZZY_MATCH_THRESHOLD:
                    best_match = top_cand
                    match_method = "fuzzy_title_similarity"
                    match_score = float(best_sim / 100.0)
                    match_status = "matched"
                elif top_cand and best_sim >= 60.0:
                    best_match = top_cand
                    match_method = "openalex_fallback"
                    match_score = float(best_sim / 100.0)
                    match_status = "uncertain"

        # Stage 5: Manual review flag if low confidence / unmatched
        if not best_match:
            match_method = "manual_review"
            match_score = 0.0
            match_status = "unmatched"
            
        matched_openalex_id = best_match.get("id", "").split("/")[-1] if best_match else None
        matched_work_title = best_match.get("display_name") if best_match else None
        publication_year = best_match.get("publication_year") if best_match else None
        cited_by_count = best_match.get("cited_by_count", 0) if best_match else 0
        
        return {
            "citation_id": citation_id,
            "raw_text": raw_text,
            "citation_context": citation_context,
            "parsed_title": parsed["title"],
            "parsed_year": parsed["year"],
            "parsed_author": parsed["lead_author"],
            "matched_openalex_id": matched_openalex_id,
            "matched_title": matched_work_title,
            "publication_year": publication_year,
            "cited_by_count": cited_by_count,
            "match_method": match_method,
            "match_score": round(match_score, 3),
            "match_status": match_status,
            "competing_candidates_count": len(candidates),
        }
