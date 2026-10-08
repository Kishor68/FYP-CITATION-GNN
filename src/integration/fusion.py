from typing import Dict, Any, List
from src.config import DEFAULT_ALPHA, DEFAULT_BETA, get_classification_label

def calculate_fused_risk(
    graph_score: float,
    semantic_score: Any,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA
) -> float:
    """
    Calculates transparent weighted risk score:
    If semantic_score is None (unmatched), returns graph_score (0.0).
    If graph_score is 0.0 (unintegrated module), relies on active semantic_score.
    When both are present, computes weighted average: alpha * graph + beta * semantic.
    """
    if semantic_score is None:
        return round(float(graph_score), 3)
    s_val = float(semantic_score)
    if graph_score <= 0.0:
        return round(s_val, 3)
    if s_val <= 0.0:
        return round(graph_score, 3)
        
    total_w = alpha + beta
    if total_w <= 0:
        a_norm, b_norm = 0.5, 0.5
    else:
        a_norm, b_norm = alpha / total_w, beta / total_w
        
    fused = (a_norm * graph_score) + (b_norm * s_val)
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
    Unintegrated modules (e.g. Graph GNN) default to 0.0.
    """
    cid = retrieval_record["citation_id"]
    
    # Graph Sub-scores (None if unmatched)
    raw_ag = graph_record.get("author_group_score")
    author_group_score = float(raw_ag) if isinstance(raw_ag, (float, int)) else None

    raw_cc = graph_record.get("citation_circle_score")
    citation_circle_score = float(raw_cc) if isinstance(raw_cc, (float, int)) else None

    raw_amp = graph_record.get("amplification_score")
    amplification_score = float(raw_amp) if isinstance(raw_amp, (float, int)) else None

    raw_g = graph_record.get("graph_score")
    g_score = float(raw_g) if isinstance(raw_g, (float, int)) else 0.0

    # Semantic Sub-scores (Person 2 SPECTER2 active backend)
    s_score = semantic_record.get("semantic_score")  # None if unmatched
    weak_citation_score = semantic_record.get("weak_citation_score")  # None if unmatched
    semantic_alignment_score = float(semantic_record.get("semantic_alignment_score", 0.0))

    risk_score = calculate_fused_risk(g_score, s_score, alpha, beta)
    classification = get_classification_label(risk_score)
    
    # Combine evidence list
    raw_g_ev = graph_record.get("graph_evidence", [])
    if isinstance(raw_g_ev, dict):
        g_ev = []
        for v in raw_g_ev.values():
            if isinstance(v, list):
                g_ev.extend(v)
            elif isinstance(v, str):
                g_ev.append(v)
    elif isinstance(raw_g_ev, list):
        g_ev = raw_g_ev
    else:
        g_ev = []

    s_ev = semantic_record.get("semantic_evidence", [])
    if not isinstance(s_ev, list):
        s_ev = []

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
        "author_group_score": author_group_score,
        "citation_circle_score": citation_circle_score,
        "amplification_score": amplification_score,
        "semantic_score": s_score,
        "weak_citation_score": weak_citation_score,
        "semantic_alignment_score": semantic_alignment_score,
        "semantic_similarity": semantic_record.get("semantic_similarity", 0.0),
        "integrity_risk_score": risk_score,
        "classification": classification,
        "explanation": explanation,
        "evidence": combined_evidence,
    }
