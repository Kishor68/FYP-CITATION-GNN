import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Ensure Windows terminal outputs do not crash on non-ASCII characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.retrieval.extract_references import process_pdf_document
from src.retrieval.matching import ReferenceMatcher
from src.models.graph_module import analyze_graph_suspicion
from src.models.semantic_module import analyze_semantic_suspicion
from src.integration.fusion import fuse_citation_evidence
from src.utils.json_helpers import save_json
from src.config import (
    PROCESSED_DIR,
    MATCHES_DIR,
    FINAL_OUTPUT_DIR,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
)

class AnalysisPipeline:
    """Coordinates end-to-end processing pipeline from PDF upload to dashboard results."""

    def __init__(self, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA):
        self.alpha = alpha
        self.beta = beta
        self.matcher = ReferenceMatcher()

    def run_pipeline(
        self,
        pdf_source: Any,
        run_id: str = "RUN_DEMO",
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes full analysis pipeline:
        Stage 1: Ingestion & Bibliography extraction
        Stage 2: OpenAlex matching cascade
        Stage 3: Graph model analysis (Person 1)
        Stage 4: Semantic model analysis (Person 2)
        Stage 5: Fusion calculation & final JSON persistence
        """
        def update_progress(msg: str, percent: int):
            if progress_callback:
                progress_callback(msg, percent)

        # STAGE 1: PDF Ingestion & Bibliography extraction
        update_progress("Extracting bibliography and text from PDF...", 10)
        paper_data = process_pdf_document(pdf_source, run_id=run_id)
        
        # Save processed paper representation
        save_json(paper_data, PROCESSED_DIR / f"{run_id}_paper.json")
        update_progress(f"Extracted {paper_data['total_references_found']} references.", 25)

        # STAGE 2: OpenAlex Reference Matching Cascade
        update_progress("Resolving references via OpenAlex cascade...", 40)
        matched_records = []
        for citation in paper_data["citations"]:
            cid = citation["citation_id"]
            raw_txt = citation["raw_text"]
            ctx = citation["citation_context"]
            match_res = self.matcher.match_reference(cid, raw_txt, ctx)
            matched_records.append(match_res)
            
        # Save matched records
        save_json(matched_records, MATCHES_DIR / f"{run_id}_matches.json")
        update_progress("OpenAlex reference resolution complete.", 60)

        # STAGE 3: Graph Model Analysis (Person 1)
        update_progress("Running Graph Neural Network structural analysis...", 75)
        graph_outputs = analyze_graph_suspicion(paper_data, matched_records)
        graph_map = {g["citation_id"]: g for g in graph_outputs}

        # STAGE 4: Semantic Model Analysis (Person 2)
        update_progress("Running Semantic alignment & LLM explanation analysis...", 85)
        semantic_outputs = analyze_semantic_suspicion(paper_data, matched_records)
        semantic_map = {s["citation_id"]: s for s in semantic_outputs}

        # STAGE 5: Fusion & Output Persistence
        update_progress("Fusing evidence & computing integrity risk scores...", 95)
        final_citations = []
        for m_rec in matched_records:
            cid = m_rec["citation_id"]
            g_rec = graph_map.get(cid, {})
            s_rec = semantic_map.get(cid, {})
            fused_rec = fuse_citation_evidence(m_rec, g_rec, s_rec, self.alpha, self.beta)
            final_citations.append(fused_rec)

        # Aggregate summary metrics
        total_count = len(final_citations)
        matched_count = sum(1 for c in final_citations if c["match_status"] == "matched")
        avg_risk = round(sum(c["integrity_risk_score"] for c in final_citations) / total_count, 3) if total_count > 0 else 0.0
        
        valid_semantics = [c["semantic_score"] for c in final_citations if isinstance(c.get("semantic_score"), (float, int))]
        avg_semantic = round(sum(valid_semantics) / len(valid_semantics), 3) if valid_semantics else 0.0

        # Print terminal logging for backend verification
        print("\n" + "=" * 80)
        print(" BACKEND PIPELINE INFERENCE OUTPUT & CITATION SCORES ")
        print("=" * 80)
        print(f"Paper Title : {paper_data['paper_title']}")
        print(f"Run ID      : {run_id}")
        print(f"Total Citations Processed: {total_count}")
        print("-" * 80)

        for c in final_citations:
            sem_str = f"{c['semantic_score']:.4f}" if isinstance(c.get("semantic_score"), (float, int)) else "N/A"
            sim_str = f"{c['semantic_similarity']:.4f}" if isinstance(c.get("semantic_similarity"), (float, int)) else "N/A"
            print(f"[+] Citation ID: {c['citation_id']}")
            print(f"   Matched Title         : {c['matched_title'] or 'Unmatched'}")
            print(f"   Match Method          : {c['match_method']} (Confidence: {c['match_score']:.2f})")
            print(f"   SPECTER2 Similarity   : {sim_str}")
            print(f"   Semantic Score        : {sem_str}  (Person 2 SPECTER2 Model)")
            print(f"   Author Group Score    : {c.get('author_group_score', 0.0):.4f}  (Person 1 GNN - Default 0.0)")
            print(f"   Citation Circle Score : {c.get('citation_circle_score', 0.0):.4f}  (Person 1 GNN - Default 0.0)")
            print(f"   Amplification Score   : {c.get('amplification_score', 0.0):.4f}  (Person 1 GNN - Default 0.0)")
            print(f"   Graph Score           : {c['graph_score']:.4f}  (Person 1 GNN - Default 0.0)")
            print(f"   -> Fused Risk Score   : {c['integrity_risk_score']:.4f}")
            print("-" * 80)

        print(f"SUMMARY METRICS:")
        print(f"  Total Extracted Citations : {total_count}")
        print(f"  OpenAlex Matched Count    : {matched_count}")
        print(f"  Average Semantic Score    : {avg_semantic:.4f}")
        print(f"  Average Fused Risk Score  : {avg_risk:.4f}")
        print("=" * 80 + "\n")

        pipeline_result = {
            "run_id": run_id,
            "paper_title": paper_data["paper_title"],
            "abstract": paper_data["abstract"],
            "paper_profile": paper_data.get("paper_profile", {}),
            "summary_metrics": {
                "total_citations": total_count,
                "matched_citations": matched_count,
                "avg_risk_score": avg_risk,
                "avg_semantic_score": avg_semantic,
            },
            "fusion_weights": {"alpha": self.alpha, "beta": self.beta},
            "citations": final_citations,
        }

        # Save final application result
        save_json(pipeline_result, FINAL_OUTPUT_DIR / f"{run_id}_final.json")
        update_progress("Analysis complete!", 100)
        
        return pipeline_result
