"""
=============================================================================
MODULE: Semantic Alignment & LLM Reasoning Analysis
OWNER: Person 2 (Semantic Module Lead)
=============================================================================
Instructions for Person 2:
- Replace or extend the logic in `analyze_semantic_suspicion()` with your trained semantic similarity / LLM explanation model.
- Input contract: List of matched citation records and paper metadata.
- Output contract per citation:
    {
        "citation_id": "C001",
        "semantic_score": 0.77,             # Composite semantic suspicion (0.00 to 1.00)
        "weak_citation_score": 0.80,        # Weak relevance / padding citation suspicion
        "semantic_alignment_score": 0.74,   # Topic drift & contextual misalignment suspicion
        "semantic_similarity": 0.26,        # Cosine / Embedding similarity (1.0 - alignment_score)
        "reason": "Textual context discusses quantum computing but cited work is on agriculture.",
        "semantic_evidence": ["topic mismatch", "irrelevance_abstract"]
    }
"""

from typing import Dict, Any, List

def analyze_semantic_suspicion(paper_data: Dict[str, Any], citation_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes semantic suspicion score partitioned into:
    - weak_citation_score (superficial or padding citation suspicion)
    - semantic_alignment_score (topic drift / contextual misalignment suspicion)
    """
    results = []
    for idx, cite in enumerate(citation_records):
        citation_id = cite["citation_id"]
        
        # MOCK SCORE GENERATION FOR DEMO (Person 2 will replace this block)
        weak_citation_score = round(min(0.95, max(0.05, 0.12 + (idx * 0.27) % 0.80)), 2)
        semantic_alignment_score = round(min(0.95, max(0.05, 0.18 + (idx * 0.23) % 0.75)), 2)
        
        # Composite Semantic Score
        semantic_score = round((weak_citation_score + semantic_alignment_score) / 2.0, 2)
        semantic_similarity = round(max(0.0, 1.0 - semantic_alignment_score), 2)
        
        if semantic_score >= 0.65:
            reason = "Significant topic mismatch and weak citation relevance between in-text paragraph and target abstract."
            evidence = ["Topic drift", "Superficial claim citation", "Weak contextual relevance"]
        elif semantic_score >= 0.40:
            reason = "Moderate semantic distance; citation claim is only weakly supported by the cited paper."
            evidence = ["Weak contextual alignment"]
        else:
            reason = "Strong semantic alignment between in-text citation sentence and cited paper."
            evidence = ["High semantic relevance", "Strong claim grounding"]
            
        results.append({
            "citation_id": citation_id,
            "semantic_score": semantic_score,
            "weak_citation_score": weak_citation_score,
            "semantic_alignment_score": semantic_alignment_score,
            "semantic_similarity": semantic_similarity,
            "reason": reason,
            "semantic_evidence": evidence,
        })
        
    return results
