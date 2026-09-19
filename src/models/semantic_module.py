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
        "semantic_score": 0.77,  # 0.00 (valid/aligned) to 1.00 (high suspicion / misaligned)
        "semantic_similarity": 0.32,
        "reason": "Textual context discusses quantum computing but cited work is on agriculture.",
        "evidence": ["topic mismatch", "irrelea_abstract"]
    }
"""

from typing import Dict, Any, List

def analyze_semantic_suspicion(paper_data: Dict[str, Any], citation_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [PERSON 2 PLACEHOLDER]
    Computes semantic suspicion score and LLM explanation for each citation.
    Currently returns mock baseline scores until Person 2 implements the model.
    """
    results = []
    for idx, cite in enumerate(citation_records):
        citation_id = cite["citation_id"]
        
        # MOCK SCORE GENERATION FOR DEMO (Person 2 will replace this block)
        # Deterministic mock score based on citation index for testing UI
        mock_score = round(0.20 + (idx * 0.31) % 0.70, 2)
        mock_similarity = round(1.0 - mock_score, 2)
        
        if mock_score >= 0.65:
            reason = "Significant topic mismatch between in-text citation paragraph and target abstract."
            evidence = ["Topic drift", "Superficial claim citation"]
        elif mock_score >= 0.40:
            reason = "Moderate semantic distance; citation claim is only weakly supported."
            evidence = ["Weak contextual alignment"]
        else:
            reason = "Strong semantic alignment between citation sentence and cited paper."
            evidence = ["High semantic relevance"]
            
        results.append({
            "citation_id": citation_id,
            "semantic_score": mock_score,
            "semantic_similarity": mock_similarity,
            "reason": reason,
            "semantic_evidence": evidence,
            "status": "mock_placeholder_active",
        })
        
    return results
