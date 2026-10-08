"""
=============================================================================
MODULE: Semantic Alignment & SPECTER2 Inference Analysis
OWNER: Person 2 (Semantic Module Lead)
=============================================================================
"""

import numpy as np
from typing import Dict, Any, List

try:
    import torch
    from scipy.special import expit
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    HAS_TORCH_SPECTER = True
except ImportError:
    HAS_TORCH_SPECTER = False
    torch = None
    expit = None
    cosine_similarity = None
    AutoTokenizer = None
    AutoAdapterModel = None

# Global model cache to avoid reloading weights on every upload
_TOKENIZER = None
_MODEL = None
_DEVICE = None

# Baseline statistics calibrated from genuine citations
GENUINE_MEAN = 0.9274
GENUINE_STD = 0.0251

def get_specter2_model():
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is None:
        MODEL_NAME = "allenai/specter2_base"
        ADAPTER_NAME = "allenai/specter2"
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoAdapterModel.from_pretrained(MODEL_NAME)
        _MODEL.load_adapter(ADAPTER_NAME, source="hf", load_as="proximity")
        _MODEL.set_active_adapters("proximity")
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _TOKENIZER, _MODEL, _DEVICE

def compute_weak_citation_score(similarity: float) -> float:
    sig = max(GENUINE_STD, 1e-6)
    z = (GENUINE_MEAN - float(similarity)) / sig
    return float(expit(z))

def analyze_semantic_suspicion(paper_data: Dict[str, Any], citation_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Person 2's SPECTER2 Semantic Inference:
    Computes real semantic suspicion scores based on title + abstract similarity.
    """
    if not HAS_TORCH_SPECTER:
        # Fallback for lightweight environment without PyTorch
        results = []
        for cite in citation_records:
            citation_id = cite["citation_id"]
            raw_text = cite.get("raw_text", "").lower()
            weak_score = 0.15 if "neural" in raw_text or "graph" in raw_text else 0.45
            results.append({
                "citation_id": citation_id,
                "semantic_score": weak_score,
                "weak_citation_score": weak_score,
                "semantic_alignment_score": 0.0,
                "semantic_similarity": round(1.0 - weak_score, 4),
                "reason": "Lightweight semantic similarity estimate (SPECTER2 torch omitted).",
                "semantic_evidence": ["Fallback Similarity Estimate"],
            })
        return results

    citing_title = paper_data.get("paper_title", "")
    citing_abstract = paper_data.get("abstract", "")
    
    citing_text = (citing_title.strip() + " [SEP] " + citing_abstract.strip()).strip()
    
    tokenizer, model, device = get_specter2_model()
    
    results = []
    
    # Collect cited texts
    cited_texts = []
    for cite in citation_records:
        matched_title = cite.get("matched_title") or cite.get("raw_text") or ""
        matched_abstract = cite.get("abstract") or cite.get("citation_context") or ""
        text = (matched_title.strip() + " [SEP] " + matched_abstract.strip()).strip()
        cited_texts.append(text)
        
    if not cited_texts:
        return results
        
    # Generate embeddings
    all_texts = [citing_text] + cited_texts
    inputs = tokenizer(all_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    citing_emb = embeddings[0:1]
    cited_embs = embeddings[1:]
    
    sims = cosine_similarity(citing_emb, cited_embs)[0]
    
    for idx, cite in enumerate(citation_records):
        citation_id = cite["citation_id"]
        sim = float(sims[idx])
        
        wcs = compute_weak_citation_score(sim)
        weak_citation_score = round(max(0.0, min(1.0, wcs)), 4)
        semantic_score = weak_citation_score
        
        if sim < 0.80:
            reason = f"Low SPECTER2 similarity ({sim:.3f}) between citing paper and cited reference. Significant topic disparity detected."
            evidence = ["SPECTER2 Topic Disparity", "Weak Contextual Alignment", f"Similarity: {sim:.3f}"]
        elif weak_citation_score >= 0.50:
            reason = f"Moderate-High SPECTER2 similarity ({sim:.3f}), but slightly below the strict genuine citation baseline ({GENUINE_MEAN:.3f})."
            evidence = ["Moderate Semantic Alignment", f"Similarity: {sim:.3f}", f"Baseline Std Dev: {GENUINE_STD:.3f}"]
        else:
            reason = f"High SPECTER2 similarity ({sim:.3f}) confirming strong topic relevance."
            evidence = ["High Semantic Alignment", f"Similarity: {sim:.3f}"]
            
        results.append({
            "citation_id": citation_id,
            "semantic_score": semantic_score,
            "weak_citation_score": weak_citation_score,
            "semantic_alignment_score": 0.0,
            "semantic_similarity": round(sim, 4),
            "reason": reason,
            "semantic_evidence": evidence,
        })
        
    return results
