import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from src.integration.pipeline import AnalysisPipeline


def create_sample_pdf(pdf_path: Path):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Graph Convolutional Networks for Citation Analysis")
    c.drawString(100, 730, "Abstract: This paper evaluates citation integrity using GNNs.")
    c.drawString(100, 700, "1. Introduction")
    c.drawString(100, 680, "Graph convolutional networks were introduced by Kipf and Welling [1].")
    c.drawString(100, 660, "Attention mechanisms were introduced by Vaswani et al [2].")
    c.drawString(100, 620, "References")
    c.drawString(100, 600, "[1] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.")
    c.drawString(100, 580, "[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., & Polosukhin, I. (2017). Attention is all you need. NeurIPS.")
    c.save()
    print(f"Sample PDF created at: {pdf_path}")

if __name__ == "__main__":
    sample_pdf = Path("data/sample_paper.pdf")
    create_sample_pdf(sample_pdf)

    pipeline = AnalysisPipeline()
    res = pipeline.run_pipeline(sample_pdf, run_id="DEMO_RUN_01")

    print("\n=================== PIPELINE DEMO RESULTS ===================")
    print(f"Run ID: {res['run_id']}")
    print(f"Paper Title: {res['paper_title']}")
    print(f"Summary Metrics: {res['summary_metrics']}")
    print("\nCitation Records:")
    for cite in res["citations"]:
        print(f"  - [{cite['citation_id']}] Classification: {cite['classification']} | Risk: {cite['integrity_risk_score']}")
        print(f"    Raw Ref: {cite['raw_text']}")
        print(f"    Matched OpenAlex Work: {cite['matched_title']} (OpenAlex ID: {cite['matched_openalex_id']})")
        print(f"    Match Method: {cite['match_method']} | Confidence: {cite['match_score']}")
        print(f"    Graph Score: {cite['graph_score']} | Semantic Score: {cite['semantic_score']}")
        print(f"    Explanation: {cite['explanation']}")
        print(f"    Evidence: {cite['evidence']}\n")
