import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_OPENALEX_DIR = DATA_DIR / "raw" / "openalex"
PROCESSED_DIR = DATA_DIR / "processed"
MATCHES_DIR = DATA_DIR / "matches"

OUTPUTS_DIR = BASE_DIR / "outputs"
GRAPH_OUTPUT_DIR = OUTPUTS_DIR / "graph"
LLM_OUTPUT_DIR = OUTPUTS_DIR / "llm"
FINAL_OUTPUT_DIR = OUTPUTS_DIR / "final"

# Ensure all directories exist
for directory in [
    RAW_OPENALEX_DIR,
    PROCESSED_DIR,
    MATCHES_DIR,
    GRAPH_OUTPUT_DIR,
    LLM_OUTPUT_DIR,
    FINAL_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# OpenAlex API settings
OPENALEX_CONTACT_EMAIL = os.getenv("OPENALEX_CONTACT_EMAIL", "researcher@example.com")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
OPENALEX_BASE_URL = "https://api.openalex.org/works"

# Fusion Parameters
DEFAULT_ALPHA = 0.50  # Graph suspicion weight
DEFAULT_BETA = 0.50   # Semantic suspicion weight

# Classification Policy Thresholds
THRESHOLD_VALID_MAX = 0.39
THRESHOLD_MANUAL_MIN = 0.40
THRESHOLD_MANUAL_MAX = 0.64
THRESHOLD_SUSPICIOUS_MIN = 0.65

# Matching Cascade Thresholds
FUZZY_MATCH_THRESHOLD = 85.0  # RapidFuzz similarity score 0-100
EXACT_TITLE_SIMILARITY = 95.0

# Citation ID Format
CITATION_ID_PREFIX = "C"
CITATION_ID_DIGITS = 3

def get_classification_label(score: float) -> str:
    """Classifies a fused risk score into valid, manual review, or suspicious."""
    if score < THRESHOLD_MANUAL_MIN:
        return "valid"
    elif score <= THRESHOLD_MANUAL_MAX:
        return "manual_review"
    else:
        return "suspicious"
