from typing import Dict, Any, List
from src.config import DEFAULT_ALPHA, DEFAULT_BETA, get_classification_label

def calculate_fused_risk(
    graph_score: float,
    semantic_score: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA
) -> float:
    """
    Calculates transparent weighted risk score:
    final_risk = alpha * graph_suspicion + beta * semantic_suspicion
    """
    # Normalize weights if sum != 1
    total_w = alpha + beta
    if total_w <= 0:
        a_norm, b_norm = 0.5, 0.5
    else:
        a_norm, b_norm = alpha / total_w, beta / total_w
        
    fused = (a_norm * graph_score) + (b_norm * semantic_score)
    return round(fused, 3)

def fuse_citation_evidence(
    retrieval_record: Dict[str, Any],
    graph_record: Dict[str, Any],
    semantic_record: Dict[str, Any],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA
) -> Dict[str, Any]:
    """
    Joins retrieval, graph, and semantic records by citation_id and computes final fused record.
    """
    cid = retrieval_record["citation_id"]
    g_score = graph_record.get("graph_score", 0.0)
    s_score = semantic_record.get("semantic_score", 0.0)
    
    risk_score = calculate_fused_risk(g_score, s_score, alpha, beta)
    classification = get_classification_label(risk_score)
    
    # Combine evidence list
    g_ev = graph_record.get("graph_evidence", [])
    s_ev = semantic_record.get("semantic_evidence", [])
    combined_evidence = list(dict.fromkeys(g_ev + s_ev))  # unique preserving order
    
    explanation = semantic_record.get("reason", "Analysis completed.")
    
    return {
        "citation_id": cid,
        "raw_text": retrieval_record.get("raw_text", ""),
        "citation_context": retrieval_record.get("citation_context", ""),
        "matched_openalex_id": retrieval_record.get("matched_openalex_id"),
        "matched_title": retrieval_record.get("matched_title"),
        "publication_year": retrieval_record.get("publication_year"),
        "cited_by_count": retrieval_record.get("cited_by_count", 0),
        "match_method": retrieval_record.get("match_method", "unknown"),
        "match_score": retrieval_record.get("match_score", 0.0),
        "match_status": retrieval_record.get("match_status", "unmatched"),
        "graph_score": g_score,
        "semantic_score": s_score,
        "semantic_similarity": semantic_record.get("semantic_similarity", 0.0),
        "integrity_risk_score": risk_score,
        "classification": classification,
        "explanation": explanation,
        "evidence": combined_evidence,
    }
