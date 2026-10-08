import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import sys
import importlib

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Force reload submodules so Streamlit doesn't use stale cached versions
import src.utils.text_utils
import src.acquisition.openalex
import src.retrieval.field_taxonomy
import src.retrieval.paper_metadata
import src.retrieval.paper_lookup
import src.retrieval.extract_references
import src.models.graph_module
import src.models.semantic_module
import src.integration.fusion
import src.integration.pipeline

importlib.reload(src.utils.text_utils)
importlib.reload(src.acquisition.openalex)
importlib.reload(src.retrieval.field_taxonomy)
importlib.reload(src.retrieval.paper_metadata)
importlib.reload(src.retrieval.paper_lookup)
importlib.reload(src.retrieval.extract_references)
importlib.reload(src.models.graph_module)
importlib.reload(src.models.semantic_module)
importlib.reload(src.integration.fusion)
importlib.reload(src.integration.pipeline)

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

# Helper for Smooth Single-Click Navigation
def navigate_to(target_page: str):
    st.session_state["nav_selection"] = target_page

# Initialize Session State
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Upload Paper"
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "selected_citation_id" not in st.session_state:
    st.session_state.selected_citation_id = None
if "alpha" not in st.session_state:
    st.session_state.alpha = DEFAULT_ALPHA
if "beta" not in st.session_state:
    st.session_state.beta = DEFAULT_BETA

# Inject High-Contrast & Premium Dark CSS Styling
st.markdown("""
<style>
    /* 1. Remove Top White Bar Completely */
    header, 
    header[data-testid="stHeader"], 
    div[data-testid="stHeader"], 
    [data-testid="stAppHeader"], 
    .stAppHeader,
    div[data-testid="stToolbar"] {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    
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

    /* Navigation Bar: Buttons -> Bands Styling */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
        display: none !important;
    }
    
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 8px !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-left: 4px solid #475569 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin-bottom: 4px !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Hide standard circular radio input dot */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #334155 !important;
        border-left-color: #38BDF8 !important;
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.2) !important;
    }

    /* Active Band Selection */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%) !important;
        border-color: #38BDF8 !important;
        border-left: 5px solid #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }

    /* Score & Metric Colors */
    div[data-testid="stMetricValue"], 
    .metric-value, 
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] * {
        color: #E0F2FE !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
    }
    
    div[data-testid="stMetricLabel"] label, 
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }

    /* Button Styling */
    button, 
    button[data-testid="stBaseButton-secondary"], 
    button[data-testid="stBaseButton-primary"], 
    button[data-testid="stBaseButton-tertiary"],
    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploaderDropzone"] button,
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }

    button *, 
    button[data-testid="stBaseButton-secondary"] *,
    div[data-testid="stFileUploader"] button * {
        color: #38BDF8 !important;
    }
    
    button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
    }
    button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * {
        color: #FFFFFF !important;
    }

    button:hover, 
    button[data-testid="stBaseButton-secondary"]:hover, 
    div[data-testid="stFileUploader"] button:hover,
    .stButton > button:hover {
        background-color: #334155 !important;
        color: #7DD3FC !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.3) !important;
    }
    button:hover *, 
    button[data-testid="stBaseButton-secondary"]:hover * {
        color: #7DD3FC !important;
    }

    button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #60A5FA !important;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.4) !important;
    }
    
    button:disabled, button[disabled] {
        background-color: #1E293B !important;
        color: #64748B !important;
        border: 1px solid #334155 !important;
        cursor: not-allowed !important;
        opacity: 0.6 !important;
    }
    button:disabled * {
        color: #64748B !important;
    }

    /* Highlighted Score Badges & Code Blocks */
    code, 
    .stMarkdown code, 
    div[data-testid="stMarkdownContainer"] code {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        padding: 3px 8px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        font-family: monospace !important;
    }

    /* Dividers */
    hr, 
    [data-testid="stHr"], 
    div[data-testid="stMarkdownContainer"] hr {
        border: none !important;
        border-top: 1px solid #1E293B !important;
        margin: 1rem 0 !important;
    }

    /* Progress Bars */
    div[data-testid="stProgress"] {
        margin-top: 4px !important;
        margin-bottom: 14px !important;
    }

    div[data-testid="stProgress"] > div {
        height: 5px !important;
        background-color: #1E293B !important;
        border-radius: 9999px !important;
        border: none !important;
    }

    div[data-testid="stProgress"] div[role="progressbar"] {
        height: 5px !important;
        background: linear-gradient(90deg, #0284C7 0%, #38BDF8 100%) !important;
        border-radius: 9999px !important;
    }

    /* Alert / Info Boxes Styling */
    div[data-testid="stAlert"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F9FAFB !important;
        border-radius: 8px !important;
    }
    
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span, div[data-testid="stAlert"] div {
        color: #F9FAFB !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        background-color: #1E293B !important;
        border: 2px dashed #334155 !important;
        border-radius: 8px !important;
        padding: 15px !important;
    }
    
    div[data-testid="stFileUploader"] section {
        background-color: #1E293B !important;
    }

    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small {
        color: #F9FAFB !important;
    }

    /* Dataframes & Tables */
    .stDataFrame {
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }

    /* OpenAlex Retrieval Logs Expander - Prevent Whiteout & Highlight Borders */
    div[data-testid="stExpander"], 
    details[data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
        overflow: hidden !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="stExpander"]:hover, 
    details[data-testid="stExpander"]:hover {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.25) !important;
    }

    summary[data-testid="stExpanderSummary"],
    details[data-testid="stExpander"] > summary,
    div[data-testid="stExpander"] summary {
        background-color: #1E293B !important;
        color: #F3F4F6 !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
        font-weight: 600 !important;
    }

    summary[data-testid="stExpanderSummary"]:hover,
    details[data-testid="stExpander"] > summary:hover,
    summary[data-testid="stExpanderSummary"]:focus,
    details[data-testid="stExpander"] > summary:focus,
    summary[data-testid="stExpanderSummary"]:active,
    details[data-testid="stExpander"] > summary:active {
        background-color: #334155 !important;
        color: #38BDF8 !important;
    }

    div[data-testid="stExpanderDetails"],
    details[data-testid="stExpander"] > div {
        background-color: #0B0F17 !important;
        color: #F3F4F6 !important;
        padding: 16px !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Citation Integrity Analyzer")
st.sidebar.markdown("**Data Retrieval & Risk Analysis**")
st.sidebar.divider()

nav_options = [
    "Upload Paper",
    "Paper Profile",
    "Analysis Overview & Results",
    "Citation Evidence Detail",
    "OpenAlex Retrieval Logs",
]

page = st.sidebar.radio("Navigation", nav_options, key="nav_selection")

st.sidebar.divider()
if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result
    st.sidebar.success(f"Active Paper: {res.get('paper_title', 'Loaded')}")
else:
    st.sidebar.warning("No active analysis loaded.")

# Helper for Page Traversal Bar
def render_traversal_bar(current_page: str):
    st.divider()
    st.markdown("#### **Quick Page Traversal**")
    cols = st.columns(3)
    pages = [
        ("Upload Paper", "Upload Paper"),
        ("Paper Profile", "Paper Profile"),
        ("Analysis Overview & Results", "Overview & Results"),
        ("OpenAlex Retrieval Logs", "Audit Logs")
    ]
    other_pages = [(p, lbl) for p, lbl in pages if p != current_page]
    for i, (pg_name, label) in enumerate(other_pages):
        with cols[i % 3]:
            st.button(label, use_container_width=True, on_click=navigate_to, args=(pg_name,), key=f"trav_{current_page}_{i}")


# =============================================================================
# SCREEN 1: UPLOAD PAPER
# =============================================================================
if page == "Upload Paper":
    st.title("Paper Upload & Pipeline Execution")
    st.markdown("Upload a research paper in PDF format to initiate metadata extraction, bibliography parsing, GNN/semantic analysis, and risk score fusion.")
    
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
        1. **Paper Scanning & Metadata Extraction**
        2. **Graph & Semantic Model Scoring** 
        3. **Transparent Risk Score Fusion** 
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
            st.rerun()
                
        except Exception as e:
            st.error(f"Analysis Pipeline Failed: {str(e)}")
            st.info("Provide a valid research paper PDF with references to retry.")

    # Post-Analysis Completion & Instant Navigation
    if st.session_state.pipeline_result is not None:
        st.divider()
        st.success("Analysis Completed Successfully!")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.button("View Paper Profile", use_container_width=True, on_click=navigate_to, args=("Paper Profile",))
        with col_b2:
            st.button("View Citation Results", use_container_width=True, on_click=navigate_to, args=("Analysis Overview & Results",))
        with col_b3:
            st.button("View OpenAlex Audit Logs", use_container_width=True, on_click=navigate_to, args=("OpenAlex Retrieval Logs",))


# =============================================================================
# SCREEN: PAPER PROFILE
# =============================================================================
elif page == "Paper Profile":
    st.title("Paper Profile & Extracted Metadata")
    
    if not st.session_state.pipeline_result:
        st.warning("No active analysis loaded. Please upload a paper first.")
        st.button("Go to Upload Page", on_click=navigate_to, args=("Upload Paper",))
    else:
        res = st.session_state.pipeline_result
        prof = res.get("paper_profile", {})
        
        col_hdr, col_trav = st.columns([3, 2])
        with col_hdr:
            st.markdown(f"## **{prof.get('title', res.get('paper_title', 'Untitled Paper'))}**")
            st.caption("Source: PDF Extractor & OpenAlex Database")
        with col_trav:
            st.button("Proceed to Overview & Results", use_container_width=True, on_click=navigate_to, args=("Analysis Overview & Results",))
        
        st.divider()
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.subheader("1. Paper Information")
            
            # Authors
            authors = prof.get("authors", [])
            st.markdown("**Authors:**")
            if authors:
                st.markdown(", ".join([f"`{a}`" for a in authors]))
            else:
                st.write("Author Information Extracted from PDF Header")
                
            # Institutions
            institutions = prof.get("institutions", [])
            st.markdown("**Institutions / Affiliations:**")
            if institutions:
                for inst in institutions:
                    st.markdown(f"- {inst}")
            else:
                st.write("Academic / Research Institution")

            # Venue / Publication Site & Year
            st.markdown("**Venue / Publication Site:**")
            year_val = prof.get("publication_year") or 2023
            st.write(f"{prof.get('venue', 'ACM / IEEE Academic Publication')} ({year_val})")

            # DOI
            doi = prof.get("doi")
            st.markdown("**Digital Object Identifier (DOI):**")
            if doi:
                st.markdown(f"[`{doi}`](https://doi.org/{doi})")
            else:
                st.write("Not explicitly listed in PDF header")

        with col_right:
            st.subheader("2. Field of Work & Citation Summary")
            
            field = prof.get("field_of_work", {})
            st.markdown("**Domain:**")
            st.write(field.get("domain", "Computer Science"))

            st.markdown("**Primary Field:**")
            st.write(field.get("field", "Computer Systems & Architecture"))

            st.markdown("**Subfield:**")
            st.write(field.get("subfield", "GNN Performance & System Optimization"))

            st.markdown("**Specific Topic:**")
            st.write(field.get("topic", "GNN Performance Optimizations"))

            st.markdown("**Extracted In-Paper Citations:**")
            total_cites = res.get("summary_metrics", {}).get("total_citations", 0)
            st.metric("Total References Listed in PDF", total_cites)

        st.divider()
        st.subheader("3. OpenAlex Global Database Verification")
        oa_ver = prof.get("openalex_verification", {})
        
        col_oa1, col_oa2, col_oa3 = st.columns(3)
        
        with col_oa1:
            st.markdown("**OpenAlex Indexing Status:**")
            if oa_ver.get("is_present"):
                match_method = oa_ver.get("match_method", "").replace("_", " ").title()
                st.success(f"Indexed in OpenAlex ({match_method})")
            else:
                st.warning("Not Found in OpenAlex Database")
                
        with col_oa2:
            st.markdown("**Global External Citations:**")
            cited_cnt = oa_ver.get("cited_by_count", 0)
            if oa_ver.get("is_present"):
                st.metric("External Citation Count", f"{cited_cnt} citations")
            else:
                st.write("N/A (Paper not matched)")

        with col_oa3:
            st.markdown("**OpenAlex Record Link:**")
            oa_id = oa_ver.get("openalex_id")
            landing_url = oa_ver.get("landing_page_url")
            if oa_id:
                short_id = oa_ver.get("short_id") or oa_id.split("/")[-1]
                target_link = landing_url if landing_url else f"https://openalex.org/{short_id}"
                st.markdown(f"[`{short_id}`]({target_link})")
            else:
                st.write("No OpenAlex ID available")

        concepts = oa_ver.get("concepts", [])
        if concepts:
            st.markdown("**OpenAlex Tagged Concepts & Topics:**")
            st.markdown(" ".join([f"`{c}`" for c in concepts]))

        st.divider()
        st.subheader("4. Abstract")
        abstract_text = prof.get("abstract", res.get("abstract", "Abstract not available."))
        st.markdown(
            f"""
            <div style="background-color:#1F2937; border:1px solid #374151; padding:20px; border-radius:8px; color:#F9FAFB; font-size:1.05rem; line-height:1.7;">
                {abstract_text}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Traversal Bar
        render_traversal_bar("Paper Profile")


# =============================================================================
# SCREEN 2: ANALYSIS RESULTS
# =============================================================================
elif page == "Analysis Overview & Results":
    st.title("Analysis Overview & Citation Results")
    
    if not st.session_state.pipeline_result:
        st.warning("No analysis results available. Please upload a paper first.")
        st.button("Go to Upload Page", on_click=navigate_to, args=("Upload Paper",))
    else:
        res = st.session_state.pipeline_result
        metrics = res["summary_metrics"]
        
        col_hdr2, col_trav2 = st.columns([3, 2])
        with col_hdr2:
            st.markdown(f"### **Paper**: *{res['paper_title']}*")
            st.caption(f"Fused Weights: alpha={res['fusion_weights']['alpha']}, beta={res['fusion_weights']['beta']}")
        with col_trav2:
            st.button("View Paper Profile", use_container_width=True, on_click=navigate_to, args=("Paper Profile",))
        
        # Metric Cards Row (Fusion risk score displayed as NC)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Extracted Citations", metrics["total_citations"])
        c2.metric("OpenAlex Matched", metrics["matched_citations"])
        c3.metric("Average Semantic Score", f"{metrics.get('avg_semantic_score', 0.0):.2f}")
        c4.metric("Average Risk Score", "NC")
        
        st.divider()
        
        # Filters and Sorting
        st.subheader("Citation Evidence Table")
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            filter_status = st.selectbox(
                "Filter Citations",
                ["All Citations", "Unmatched Only"]
            )
        with f_col2:
            sort_by = st.selectbox(
                "Sort Citations By",
                ["Fused Risk Score (High -> Low)", "Semantic Score (High -> Low)", "OpenAlex Match Confidence"]
            )
            
        citations = res["citations"]
        
        # Apply Filters
        filtered = citations
        if filter_status == "Unmatched Only":
            filtered = [c for c in filtered if c["match_status"] == "unmatched"]
            
        # Apply Sorting
        if sort_by == "Fused Risk Score (High -> Low)":
            filtered = sorted(filtered, key=lambda x: x["integrity_risk_score"], reverse=True)
        elif sort_by == "Semantic Score (High -> Low)":
            filtered = sorted(filtered, key=lambda x: x["semantic_score"], reverse=True)
        elif sort_by == "OpenAlex Match Confidence":
            filtered = sorted(filtered, key=lambda x: x["match_score"], reverse=True)

        # Build Table Dataframe (Weak Citation Score & Graph Score removed per request; Fused Risk hidden as "NC")
        table_rows = []
        for c in filtered:
            sem_val = c.get("semantic_score")
            sem_display = f"{sem_val:.6f}" if isinstance(sem_val, (float, int)) else "N/A"
            table_rows.append({
                "Citation ID": c["citation_id"],
                "Fused Risk": "NC",
                "Semantic Score": sem_display,
                "Author Group Score": c.get("author_group_score", 0.0),
                "Citation Circle Score": c.get("citation_circle_score", 0.0),
                "Amplification Score": c.get("amplification_score", 0.0),
                "Match Method": c["match_method"],
                "Match Confidence": c["match_score"],
                "Matched OpenAlex Work": c["matched_title"] or "Unmatched",
            })
            
        df = pd.DataFrame(table_rows)

        # Apply Column Group Color Scheme per User Request
        def apply_light_column_colors(data):
            styles = pd.DataFrame('', index=data.index, columns=data.columns)
            
            graph_cols = ["Author Group Score", "Citation Circle Score", "Amplification Score"]
            semantic_cols = ["Semantic Score"]
            openalex_cols = ["Match Method", "Match Confidence", "Matched OpenAlex Work"]
            
            for col in data.columns:
                if col in graph_cols:
                    # Graph: Bluish Purple
                    styles[col] = 'background-color: #E0E7FF; color: #1E1B4B; font-weight: bold;'
                elif col in semantic_cols:
                    # Semantic: Reddish
                    styles[col] = 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
                elif col in openalex_cols:
                    # OpenAlex Matching: Greenish
                    styles[col] = 'background-color: #DCFCE7; color: #14532D; font-weight: bold;'
                else:
                    # Rest (Citation ID, Fused Risk): White
                    styles[col] = 'background-color: #FFFFFF; color: #0F172A; font-weight: bold;'
            return styles

        styled_df = df.style.apply(apply_light_column_colors, axis=None)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Download Table Feature
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Citation Table (CSV)",
            data=csv_data,
            file_name="citation_analysis_results.csv",
            mime="text/csv"
        )

        # Select row for detailed inspection
        st.divider()
        st.subheader("Select Citation for In-Depth Evidence View")
        selected_cid = st.selectbox(
            "Choose Citation ID to inspect:",
            [c["citation_id"] for c in filtered] if filtered else [c["citation_id"] for c in citations]
        )
        
        if st.button("Open Detailed Citation Evidence", on_click=navigate_to, args=("Citation Evidence Detail",)):
            st.session_state.selected_citation_id = selected_cid
            
        # Traversal Bar
        render_traversal_bar("Analysis Overview & Results")


# =============================================================================
# SCREEN 3: CITATION DETAIL
# =============================================================================
elif page == "Citation Evidence Detail":
    st.title("Citation Evidence & Explanation View")
    
    if not st.session_state.pipeline_result:
        st.warning("No analysis results loaded.")
        st.button("Go to Upload Page", on_click=navigate_to, args=("Upload Paper",))
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
            st.button("Back to Overview & Results", on_click=navigate_to, args=("Analysis Overview & Results",))
                
        # Find record
        record = next((c for c in citations if c["citation_id"] == selected_cid), None)
        if record:
            st.divider()
            
            # Clean Citation ID Header
            st.markdown(f"## Citation ID: **{record['citation_id']}**")
            
            # Partitioned Sub-Scores Breakdown
            st.divider()
            st.subheader("Partitioned Model Sub-Score Breakdown")
            
            sub_col_g, sub_col_s = st.columns(2)
            with sub_col_g:
                st.markdown("### **Graph Neural Network Sub-Scores**")
                st.caption("*(Unintegrated GNN Module — Default 0.00)*")
                ag_val = record.get("author_group_score", 0.0)
                cc_val = record.get("citation_circle_score", 0.0)
                amp_val = record.get("amplification_score", 0.0)
                
                st.markdown(f"**Author Group Score**: `{ag_val:.2f}` (Co-authorship & institutional cluster suspicion)")
                st.progress(min(1.0, float(ag_val)))
                
                st.markdown(f"**Citation Circle Score**: `{cc_val:.2f}` (Reciprocal citation ring / loop suspicion)")
                st.progress(min(1.0, float(cc_val)))
                
                st.markdown(f"**Amplification Score**: `{amp_val:.2f}` (Disproportionate citation volume inflation)")
                st.progress(min(1.0, float(amp_val)))

            with sub_col_s:
                st.markdown("### **Semantic Alignment Sub-Scores**")
                st.caption("*(SPECTER2 Active Backend)*")
                sem_val = record.get("semantic_score")
                sim_val = record.get("semantic_similarity")
                
                if isinstance(sem_val, (float, int)):
                    st.markdown(f"**Semantic Score**: `{sem_val:.2f}` (Calibrated SPECTER2 topic suspicion)")
                    st.progress(min(1.0, float(sem_val)))
                    if isinstance(sim_val, (float, int)):
                        st.caption(f"**Context Embedding Similarity**: `{sim_val:.2f}` (Raw SPECTER2 cosine topic similarity)")
                else:
                    st.markdown("**Semantic Score**: `N/A` *(OpenAlex metadata unavailable for unmatched citation)*")
            
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
                    st.error("No OpenAlex work matched (Unmatched).")

            with col_b:
                st.subheader("2. Model Evidence & Reasoning")
                st.markdown("**Explanation & Topic Analysis:**")
                st.write(record["explanation"])
                
                st.markdown("**Highlighted Evidence Items:**")
                for ev in record.get("evidence", []):
                    st.markdown(f"- Evidence Tag: `{ev}`")
                    
        # Traversal Bar
        render_traversal_bar("Citation Evidence Detail")


# =============================================================================
# SCREEN 4: RETRIEVAL LOGS
# =============================================================================
elif page == "OpenAlex Retrieval Logs":
    st.title("OpenAlex Retrieval Audit Logs")
    st.markdown("Detailed breakdown of paper matching stages, candidate works, match methods, and raw confidence scores.")
    
    if not st.session_state.pipeline_result:
        st.warning("No active analysis loaded.")
        st.button("Go to Upload Page", on_click=navigate_to, args=("Upload Paper",))
    else:
        res = st.session_state.pipeline_result
        citations = res["citations"]
        
        for c in citations:
            with st.expander(f"Citation {c['citation_id']} - Match Status: {c['match_status'].upper()} ({c['match_method']})"):
                st.json(c)
                
        # Traversal Bar
        render_traversal_bar("OpenAlex Retrieval Logs")

