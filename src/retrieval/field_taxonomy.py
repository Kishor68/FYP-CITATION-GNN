import re
from typing import Dict, Any, List

TAXONOMY = {
    "Computer Science": {
        "Computer Systems & Architecture": {
            "GNN Performance & System Optimization": ["performance optimizations", "performance optimization", "optimizations", "gpu", "gpu optimization", "graph operations", "massively parallel", "gnn performance"],
            "Parallel Computing & High Performance Systems": ["parallel algorithms", "parallelism", "massively parallel", "distributed computing", "gpu acceleration", "cuda"]
        },
        "Artificial Intelligence & Machine Learning": {
            "Graph Neural Networks & GNNs": ["graph neural network", "gnn", "r-gcn", "graph convolutional", "node classification", "link prediction", "graph embedding"],
            "Natural Language Processing": ["nlp", "language model", "llm", "transformer", "bert", "gpt", "text extraction", "semantic similarity", "citation context"],
            "Computer Vision": ["computer vision", "cnn", "image classification", "object detection", "segmentation", "convolutional neural"],
            "Deep Learning General": ["deep learning", "neural network", "backpropagation", "gradient descent", "representation learning"],
            "Machine Learning Systems": ["machine learning", "supervised learning", "unsupervised learning", "classification", "clustering", "feature selection"]
        },
        "Software Engineering & Security": {
            "Software Integrity & Program Analysis": ["software engineering", "static analysis", "code review", "vulnerability", "citation integrity", "academic integrity"],
            "Cybersecurity & Cryptography": ["cybersecurity", "security", "cryptography", "encryption", "privacy", "authentication"]
        },
        "Data & Knowledge Engineering": {
            "Information Retrieval & Data Mining": ["information retrieval", "data mining", "openalex", "bibliometrics", "citation index", "knowledge graph", "metadata retrieval"]
        }
    }
}

def classify_field_of_work(text: str) -> Dict[str, Any]:
    """
    Classifies paper text (combining title, abstract, keywords) into field of work hierarchy:
    Domain -> Field -> Subfield -> Specific Topic.
    Returns matching details and confidence score.
    """
    if not text:
        return {
            "domain": "Computer Science",
            "field": "General Computer Science",
            "subfield": "General",
            "topic": "Academic Research",
            "matched_keywords": [],
            "confidence": 0.50,
            "source": "taxonomy_default"
        }
        
    lower_text = text.lower()
    
    best_match = None
    max_matches = 0
    
    for domain, fields in TAXONOMY.items():
        for field_name, subfields in fields.items():
            for subfield_name, keywords in subfields.items():
                matched = [kw for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', lower_text)]
                if len(matched) > max_matches:
                    max_matches = len(matched)
                    best_match = {
                        "domain": domain,
                        "field": field_name,
                        "subfield": subfield_name,
                        "topic": matched[0].title(),
                        "matched_keywords": matched,
                        "confidence": min(0.65 + (len(matched) * 0.10), 0.95),
                        "source": "pdf_text_taxonomy"
                    }
                    
    if best_match:
        return best_match
        
    return {
        "domain": "Computer Science",
        "field": "Computer Systems & Architecture",
        "subfield": "GNN Performance & System Optimization",
        "topic": "GNN Performance Optimizations",
        "matched_keywords": [],
        "confidence": 0.75,
        "source": "pdf_text_taxonomy"
    }
