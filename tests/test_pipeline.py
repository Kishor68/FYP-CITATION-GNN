import unittest
import json
from pathlib import Path
from src.retrieval.paper_lookup import normalize_title, parse_reference_fields
from src.retrieval.extract_references import split_references, isolate_bibliography
from src.integration.fusion import calculate_fused_risk, fuse_citation_evidence
from src.config import get_classification_label
from src.models.graph_module import analyze_graph_suspicion
from src.models.semantic_module import analyze_semantic_suspicion
from src.utils.json_helpers import validate_final_record

class TestPipelineComponents(unittest.TestCase):

    def test_normalize_title(self):
        title = "Graph Neural Networks: A Review (2020)!"
        norm = normalize_title(title)
        self.assertEqual(norm, "graph neural networks a review 2020")

    def test_parse_reference_fields(self):
        ref = "[1] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR."
        parsed = parse_reference_fields(ref)
        self.assertEqual(parsed["year"], 2017)
        self.assertEqual(parsed["lead_author"], "Kipf")

    def test_fusion_calculation(self):
        # alpha=0.5, beta=0.5
        score = calculate_fused_risk(0.80, 0.60, alpha=0.5, beta=0.5)
        self.assertEqual(score, 0.70)
        label = get_classification_label(score)
        self.assertEqual(label, "suspicious")

        # Valid score test
        v_score = calculate_fused_risk(0.20, 0.30, alpha=0.5, beta=0.5)
        self.assertEqual(v_score, 0.25)
        self.assertEqual(get_classification_label(v_score), "valid")

    def test_model_placeholders(self):
        paper_data = {"paper_title": "Test Paper"}
        matched = [{"citation_id": "C001", "raw_text": "Ref text"}]
        
        g_out = analyze_graph_suspicion(paper_data, matched)
        self.assertEqual(len(g_out), 1)
        self.assertIn("graph_score", g_out[0])
        
        s_out = analyze_semantic_suspicion(paper_data, matched)
        self.assertEqual(len(s_out), 1)
        self.assertIn("semantic_score", s_out[0])

    def test_fused_record_contract(self):
        retrieval_rec = {
            "citation_id": "C001",
            "raw_text": "Test Ref",
            "citation_context": "Context text",
            "matched_openalex_id": "W12345",
            "matched_title": "Matched Paper Title",
            "match_status": "matched",
            "match_method": "exact_normalized_title",
            "match_score": 1.0,
        }
        graph_rec = {"citation_id": "C001", "graph_score": 0.8, "graph_evidence": ["anomaly"]}
        semantic_rec = {"citation_id": "C001", "semantic_score": 0.7, "semantic_evidence": ["drift"], "reason": "Topic drift."}
        
        fused = fuse_citation_evidence(retrieval_rec, graph_rec, semantic_rec)
        self.assertTrue(validate_final_record(fused))
        self.assertEqual(fused["integrity_risk_score"], 0.75)
        self.assertEqual(fused["classification"], "suspicious")

if __name__ == "__main__":
    unittest.main()
