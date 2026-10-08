"""
=============================================================================
UNIT TESTS FOR HETEROGENEOUS R-GCN GRAPH ANOMALY ANALYSIS MODULE
=============================================================================
"""

import unittest
import torch
from src.models.graph_module import (
    analyze_graph_suspicion,
    GraphManager,
    CitationMLP,
    compute_amplification_score,
    compute_author_group_score,
    compute_citation_circle_score
)
from src.integration.fusion import fuse_citation_evidence


class TestGraphModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = GraphManager.get_instance()

    def test_01_module_import_and_checkpoint_load(self):
        """1 & 2. Verify module imports and trained checkpoint loads with correct architecture."""
        self.assertIsNotNone(self.mgr.citation_mlp)
        self.assertIsInstance(self.mgr.citation_mlp, CitationMLP)
        
        # Check network weights
        self.assertEqual(self.mgr.citation_mlp.network[0].in_features, 71)
        self.assertEqual(self.mgr.citation_mlp.network[0].out_features, 64)
        self.assertEqual(self.mgr.citation_mlp.network[3].out_features, 32)
        self.assertEqual(self.mgr.citation_mlp.network[5].out_features, 1)

    def test_02_record_count_and_id_preservation(self):
        """3 & 4. Verify output length, citation IDs, and input ordering preservation."""
        paper_data = {"paper_id": "W100", "paper_title": "Test Paper"}
        records = [
            {"citation_id": "CIT_001", "matched_openalex_id": "W1"},
            {"citation_id": "CIT_002", "matched_openalex_id": "W2"},
            {"citation_id": "CIT_003", "matched_openalex_id": "W3"},
        ]
        out = analyze_graph_suspicion(paper_data, records)
        self.assertEqual(len(out), 3)
        self.assertEqual([r["citation_id"] for r in out], ["CIT_001", "CIT_002", "CIT_003"])

    def test_03_score_bounds_and_structure(self):
        """5 & 6. Verify scores are bounded [0.0, 1.0] and calculated independently."""
        paper_data = {"paper_id": "W100", "paper_title": "Test Paper"}
        records = [{"citation_id": "C_TEST", "matched_openalex_id": "W200"}]
        out = analyze_graph_suspicion(paper_data, records)[0]

        for key in ["author_group_score", "citation_circle_score", "amplification_score"]:
            self.assertIn(key, out)
            val = out[key]
            self.assertIsInstance(val, float)
            self.assertTrue(0.0 <= val <= 1.0, f"Score {key}={val} out of bounds [0, 1]")

        self.assertIn("graph_evidence", out)
        self.assertIn("author_group", out["graph_evidence"])
        self.assertIn("citation_circle", out["graph_evidence"])
        self.assertIn("amplification", out["graph_evidence"])

    def test_04_author_overlap_calculation(self):
        """8. Test Module B (Author-Group Analysis) author overlap scoring."""
        # Simulated co-author overlap
        feat_overlap_0 = {"author_overlap": 0, "same_venue": 0, "citing_authors": ["a1"], "cited_authors": ["a2"]}
        score_0, ev_0 = compute_author_group_score(self.mgr, "P1", "P2", feat_overlap_0)
        self.assertEqual(score_0, 0.0)

        feat_overlap_2 = {"author_overlap": 2, "same_venue": 1, "citing_authors": ["a1", "a2"], "cited_authors": ["a1", "a2"]}
        score_2, ev_2 = compute_author_group_score(self.mgr, "P1", "P2", feat_overlap_2)
        self.assertTrue(score_2 >= 0.70)
        self.assertTrue(any("co-author" in e.lower() for e in ev_2))

    def test_05_citation_circle_reciprocal_detection(self):
        """7. Test Module C (Citation-Circle Analysis) reciprocal and cycle detection."""
        # Add synthetic reciprocal edge in graph manager
        self.mgr.citation_adj["PAPER_A"] = {"PAPER_B"}
        self.mgr.citation_adj["PAPER_B"] = {"PAPER_A"}

        score_recip, ev_recip = compute_citation_circle_score(self.mgr, "PAPER_A", "PAPER_B")
        self.assertTrue(score_recip >= 0.85)
        self.assertTrue(any("reciprocal" in e.lower() for e in ev_recip))

        # Test non-reciprocal pair
        score_none, _ = compute_citation_circle_score(self.mgr, "PAPER_X", "PAPER_Y")
        self.assertEqual(score_none, 0.0)

    def test_06_amplification_degree_responsiveness(self):
        """9. Test Module A (Citation Amplification Analysis) volume/degree responsiveness."""
        pair_tensor = torch.randn(1, 71)
        feat_high = {"cited_in_degree": 600, "citing_out_degree": 60}
        score_high, ev_high = compute_amplification_score(self.mgr, pair_tensor, feat_high)
        self.assertTrue(0.0 <= score_high <= 1.0)
        self.assertTrue(any("concentration" in e.lower() for e in ev_high))

    def test_07_safe_handling_missing_metadata(self):
        """10. Test safe handling of unmatched citations, missing metadata, and unknown IDs."""
        paper_data = {}  # Empty paper data
        records = [
            {"citation_id": "C_MISSING", "matched_openalex_id": None, "match_status": "unmatched"},
            {"citation_id": "C_UNKNOWN", "matched_openalex_id": "UNKNOWN_ID_999999", "match_status": "matched"}
        ]
        out = analyze_graph_suspicion(paper_data, records)
        self.assertEqual(len(out), 2)
        self.assertIsNone(out[0]["author_group_score"])
        self.assertIsNone(out[0]["citation_circle_score"])
        self.assertIsNone(out[0]["amplification_score"])
        self.assertTrue(0.0 <= out[1]["author_group_score"] <= 1.0)

    def test_08_repeated_inference_no_grad_leak(self):
        """11. Verify repeated inference runs smoothly without autograd memory leaks."""
        paper_data = {"paper_id": "W1"}
        records = [{"citation_id": f"CIT_{i}", "matched_openalex_id": f"W{i}"} for i in range(10)]
        
        for _ in range(5):
            out = analyze_graph_suspicion(paper_data, records)
            self.assertEqual(len(out), 10)

    def test_09_pipeline_fusion_compatibility(self):
        """12. Test compatibility with existing pipeline and fusion contracts."""
        retrieval_rec = {"citation_id": "C001", "raw_text": "Ref string", "match_status": "matched"}
        paper_data = {"paper_id": "W1"}
        matched = [{"citation_id": "C001", "matched_openalex_id": "W2"}]

        graph_rec = analyze_graph_suspicion(paper_data, matched)[0]
        semantic_rec = {"citation_id": "C001", "semantic_score": 0.60, "semantic_evidence": ["drift"]}

        fused = fuse_citation_evidence(retrieval_rec, graph_rec, semantic_rec)
        self.assertIn("author_group_score", fused)
        self.assertIn("citation_circle_score", fused)
        self.assertIn("amplification_score", fused)
        self.assertIn("evidence", fused)
        self.assertIsInstance(fused["evidence"], list)


if __name__ == "__main__":
    unittest.main()
