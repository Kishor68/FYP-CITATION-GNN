import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import sys

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.integration.pipeline import AnalysisPipeline
from src.config import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    THRESHOLD_VALID_MAX,
    THRESHOLD_MANUAL_MIN,
    THRESHOLD_MANUAL_MAX,
    THRESHOLD_SUSPICIOUS_MIN,
    OPENALEX_CONTACT_EMAIL,
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Citation Integrity Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject High-Contrast CSS Styling
st.markdown("""
<style>
    /* Dark Theme High-Contrast Base Colors */
    .stApp {
        background-color: #0B0F17;
        color: #F3F4F6;
    }
    
    /* Headers & Labels */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    label, .stMarkdown p, .stText {
        color: #E5E7EB !important;
        font-size: 1rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937 !important;
    }
    
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #F9FAFB !important;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #9CA3AF;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Badges */
    .badge-suspicious {
        background-color: #7F1D1D;
        color: #FCA5A5;
        border: 1px solid #EF4444;
        padding: 4px 10px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-manual {
        background-color: #78350F;
        color: #FDE68A;
        border: 1px solid #F59E0B;
        padding: 4px 10px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-valid {
        background-color: #064E3B;
        color: #A7F3D0;
        border: 1px solid #10B981;
        padding: 4px 10px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* File Uploader styling override for high contrast */
    div[data-testid="stFileUploader"] {
        background-color: #1F2937 !important;
        border: 2px dashed #4B5563 !important;
        border-radius: 8px !important;
        padding: 15px !important;
    }
    
    div[data-testid="stFileUploader"] section {
        background-color: #1F2937 !important;
    }

    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small {
        color: #F9FAFB !important;
    }
    
    /* Streamlit Buttons High-Contrast Overrides */
    button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        border: none !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
    }
    
    button:disabled, button[disabled] {
        background-color: #374151 !important;
        color: #9CA3AF !important;
        border: 1px solid #4B5563 !important;
        cursor: not-allowed !important;
        opacity: 0.8 !important;
    }
    
    /* Dataframes & Tables */
    .stDataFrame {
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "current_page" not in st.session_state:
    st.session_state.current_page = "Upload Paper"
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "selected_citation_id" not in st.session_state:
    st.session_state.selected_citation_id = None
if "alpha" not in st.session_state:
    st.session_state.alpha = DEFAULT_ALPHA
if "beta" not in st.session_state:
    st.session_state.beta = DEFAULT_BETA

# Sidebar Navigation
st.sidebar.title("Citation Integrity Analyzer")
st.sidebar.markdown("**Data Retrieval & Risk Analysis**")
st.sidebar.divider()

nav_options = [
    "Upload Paper",
    "Analysis Overview & Results",
    "Citation Evidence Detail",
    "OpenAlex Retrieval Logs",
    "Settings & Parameters",
]

page = st.sidebar.radio("Navigation", nav_options, index=nav_options.index(st.session_state.current_page))
st.session_state.current_page = page

st.sidebar.divider()
if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result
    st.sidebar.success(f"Active Paper: {res.get('paper_title', 'Loaded')[:30]}...")
    st.sidebar.info(f"Run ID: {res.get('run_id')}")
else:
    st.sidebar.warning("No active analysis loaded.")


# =============================================================================
# SCREEN 1: UPLOAD PAPER
# =============================================================================
if page == "Upload Paper":
    st.title("Paper Upload & Pipeline Execution")
    st.markdown("Upload a research paper in PDF format to initiate bibliography extraction, OpenAlex metadata resolution, GNN/semantic analysis, and risk score fusion.")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("1. Select PDF Document")
        uploaded_file = st.file_uploader("Choose a research paper PDF file", type=["pdf"], key="pdf_uploader")
        
        is_valid_pdf = False
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 25:
                st.error("File size exceeds maximum 25 MB limit.")
            else:
                is_valid_pdf = True
                st.success(f"Valid PDF Selected: {uploaded_file.name} ({file_size_mb:.2f} MB)")
                
        analyze_btn = st.button("Analyze Paper", disabled=not is_valid_pdf, type="primary")

    with col2:
        st.subheader("2. Workflow Stage Monitor")
        st.markdown("""
        **3-Stage Analysis Pipeline:**
        1. **Extraction & OpenAlex Retrieval** *(Person 3)*
        2. **Graph & Semantic Model Scoring** *(Person 1 & 2 Modules)*
        3. **Transparent Risk Score Fusion** *(Person 3)*
        """)

    if analyze_btn and uploaded_file is not None:
        st.divider()
        st.subheader("Processing Analysis Pipeline...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_ui_progress(msg: str, percent: int):
            status_text.markdown(f"**{msg}**")
            progress_bar.progress(percent)

        try:
            pipeline = AnalysisPipeline(alpha=st.session_state.alpha, beta=st.session_state.beta)
            timestamp = int(time.time())
            run_id = f"RUN_{timestamp}"
            
            result = pipeline.run_pipeline(
                pdf_source=uploaded_file,
                run_id=run_id,
                progress_callback=update_ui_progress
            )
            
            st.session_state.pipeline_result = result
            st.session_state.selected_citation_id = result["citations"][0]["citation_id"] if result["citations"] else None
            
            st.success("Analysis Completed Successfully! Click below to view results.")
            if st.button("View Analysis Results"):
                st.session_state.current_page = "Analysis Overview & Results"
                st.rerun()
                
        except Exception as e:
            st.error(f"Analysis Pipeline Failed: {str(e)}")
            st.info("Provide a valid research paper PDF with references to retry.")


# =============================================================================
# SCREEN 2: ANALYSIS RESULTS
# =============================================================================
elif page == "Analysis Overview & Results":
    st.title("Analysis Overview & Citation Results")
    
    if not st.session_state.pipeline_result:
        st.warning("No analysis results available. Please upload a paper first.")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = "Upload Paper"
            st.rerun()
    else:
        res = st.session_state.pipeline_result
        metrics = res["summary_metrics"]
        
        st.markdown(f"### **Paper**: *{res['paper_title']}*")
        st.caption(f"Run ID: {res['run_id']} | Fused Weights: alpha={res['fusion_weights']['alpha']}, beta={res['fusion_weights']['beta']}")
        
        # Metric Cards Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Extracted Citations", metrics["total_citations"])
        c2.metric("OpenAlex Matched", metrics["matched_citations"])
        c3.metric("Flagged Suspicious", metrics["suspicious_citations"])
        c4.metric("Manual Review", metrics["manual_review_citations"])
        
        st.divider()
        
        # Filters and Sorting
        st.subheader("Citation Evidence Table")
        
        f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
        with f_col1:
            filter_status = st.selectbox(
                "Filter by Classification",
                ["All Citations", "Suspicious Only", "Manual Review Only", "Valid Only", "Unmatched Only"]
            )
        with f_col2:
            sort_by = st.selectbox(
                "Sort Citations By",
                ["Fused Risk Score (High -> Low)", "Graph Score (High -> Low)", "Semantic Score (High -> Low)", "OpenAlex Match Confidence"]
            )
            
        citations = res["citations"]
        
        # Apply Filters
        filtered = citations
        if filter_status == "Suspicious Only":
            filtered = [c for c in filtered if c["classification"] == "suspicious"]
        elif filter_status == "Manual Review Only":
            filtered = [c for c in filtered if c["classification"] == "manual_review"]
        elif filter_status == "Valid Only":
            filtered = [c for c in filtered if c["classification"] == "valid"]
        elif filter_status == "Unmatched Only":
            filtered = [c for c in filtered if c["match_status"] == "unmatched"]
            
        # Apply Sorting
        if sort_by == "Fused Risk Score (High -> Low)":
            filtered = sorted(filtered, key=lambda x: x["integrity_risk_score"], reverse=True)
        elif sort_by == "Graph Score (High -> Low)":
            filtered = sorted(filtered, key=lambda x: x["graph_score"], reverse=True)
        elif sort_by == "Semantic Score (High -> Low)":
            filtered = sorted(filtered, key=lambda x: x["semantic_score"], reverse=True)
        elif sort_by == "OpenAlex Match Confidence":
            filtered = sorted(filtered, key=lambda x: x["match_score"], reverse=True)

        # Build Interactive Table Dataframe
        table_rows = []
        for c in filtered:
            status_badge = "Valid"
            if c["classification"] == "suspicious":
                status_badge = "Suspicious"
            elif c["classification"] == "manual_review":
                status_badge = "Manual Review"
                
            table_rows.append({
                "Citation ID": c["citation_id"],
                "Status": status_badge,
                "Fused Risk": c["integrity_risk_score"],
                "Graph Score (S_graph)": c["graph_score"],
                "Semantic Score (S_sem)": c["semantic_score"],
                "Match Method": c["match_method"],
                "Match Confidence": c["match_score"],
                "Matched OpenAlex Work": c["matched_title"] or "Unmatched",
            })
            
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Select row for detailed inspection
        st.divider()
        st.subheader("Select Citation for In-Depth Evidence View")
        selected_cid = st.selectbox(
            "Choose Citation ID to inspect:",
            [c["citation_id"] for c in filtered] if filtered else [c["citation_id"] for c in citations]
        )
        
        if st.button("Open Detailed Citation Evidence"):
            st.session_state.selected_citation_id = selected_cid
            st.session_state.current_page = "Citation Evidence Detail"
            st.rerun()


# =============================================================================
# SCREEN 3: CITATION DETAIL
# =============================================================================
elif page == "Citation Evidence Detail":
    st.title("Citation Evidence & Explanation View")
    
    if not st.session_state.pipeline_result:
        st.warning("No analysis results loaded.")
    else:
        res = st.session_state.pipeline_result
        citations = res["citations"]
        
        cids = [c["citation_id"] for c in citations]
        current_id = st.session_state.selected_citation_id or cids[0]
        
        col_select, col_back = st.columns([3, 1])
        with col_select:
            selected_cid = st.selectbox("Inspect Citation ID:", cids, index=cids.index(current_id) if current_id in cids else 0)
            st.session_state.selected_citation_id = selected_cid
        with col_back:
            if st.button("Back to Results"):
                st.session_state.current_page = "Analysis Overview & Results"
                st.rerun()
                
        # Find record
        record = next((c for c in citations if c["citation_id"] == selected_cid), None)
        if record:
            st.divider()
            
            # Top Banner & Risk Badge
            risk = record["integrity_risk_score"]
            label = record["classification"].upper()
            
            badge_color = "red" if label == "SUSPICIOUS" else ("orange" if label == "MANUAL_REVIEW" else "green")
            st.markdown(f"## Citation ID: **{record['citation_id']}**  | Classification: :{badge_color}[**{label}**]")
            
            # Score Overview Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fused Risk Score", f"{risk:.2f}")
            m2.metric("Graph Suspicion (S_graph)", f"{record['graph_score']:.2f}")
            m3.metric("Semantic Suspicion (S_semantic)", f"{record['semantic_score']:.2f}")
            m4.metric("OpenAlex Match Confidence", f"{record['match_score']:.2f}")
            
            st.divider()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("1. Citation In-Text Context & Source")
                st.markdown("**Raw Reference Entry:**")
                st.info(record["raw_text"])
                
                st.markdown("**In-Text Citation Context:**")
                st.warning(f"\"{record['citation_context']}\"")
                
                st.markdown("**Matched OpenAlex Work:**")
                if record["matched_openalex_id"]:
                    url = f"https://openalex.org/{record['matched_openalex_id']}"
                    st.markdown(f"- **Title**: [{record['matched_title']}]({url})")
                    st.markdown(f"- **OpenAlex ID**: `{record['matched_openalex_id']}`")
                    st.markdown(f"- **Publication Year**: {record['publication_year']}")
                    st.markdown(f"- **Citations Count**: {record['cited_by_count']}")
                else:
                    st.error("No OpenAlex work matched (Unmatched / Manual Review required).")

            with col_b:
                st.subheader("2. Model Evidence & Reasoning")
                st.markdown("**Explanation & Topic Analysis:**")
                st.write(record["explanation"])
                
                st.markdown("**Highlighted Evidence Items:**")
                for ev in record.get("evidence", []):
                    st.markdown(f"- Evidence Tag: `{ev}`")
                    
                st.divider()
                st.subheader("Reviewer Actions")
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    if st.button("Confirm Valid", use_container_width=True):
                        record["classification"] = "valid"
                        st.success(f"Citation {selected_cid} marked as Valid.")
                with r_col2:
                    if st.button("Flag Suspicious", use_container_width=True):
                        record["classification"] = "suspicious"
                        st.error(f"Citation {selected_cid} flagged as Suspicious.")


# =============================================================================
# SCREEN 4: RETRIEVAL LOGS
# =============================================================================
elif page == "OpenAlex Retrieval Logs":
    st.title("OpenAlex Retrieval Audit Logs")
    st.markdown("Detailed breakdown of paper matching stages, candidate works, match methods, and raw confidence scores.")
    
    if not st.session_state.pipeline_result:
        st.warning("No active analysis loaded.")
    else:
        res = st.session_state.pipeline_result
        citations = res["citations"]
        
        for c in citations:
            with st.expander(f"Citation {c['citation_id']} - Match Status: {c['match_status'].upper()} ({c['match_method']})"):
                st.json(c)


# =============================================================================
# SCREEN 5: SETTINGS
# =============================================================================
elif page == "Settings & Parameters":
    st.title("System Settings & Fusion Parameters")
    
    st.subheader("1. Fusion Score Weights")
    st.markdown("Adjust transparent fusion weights alpha (Graph) and beta (Semantic).")
    
    alpha = st.slider("Alpha (Graph Suspicion Weight)", 0.0, 1.0, float(st.session_state.alpha), 0.05)
    beta = st.slider("Beta (Semantic Suspicion Weight)", 0.0, 1.0, float(st.session_state.beta), 0.05)
    
    if alpha + beta != 1.0:
        st.caption(f"Note: Alpha + Beta = {alpha+beta:.2f}. Weights will be normalized automatically during fusion.")
        
    st.session_state.alpha = alpha
    st.session_state.beta = beta
    
    st.divider()
    st.subheader("2. OpenAlex API Contact Settings")
    email = st.text_input("Contact Email (for OpenAlex Polite Pool)", value=OPENALEX_CONTACT_EMAIL)
    
    st.success("Settings saved for session.")
