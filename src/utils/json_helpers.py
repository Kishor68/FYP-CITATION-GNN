import json
from pathlib import Path
from typing import Dict, Any, List, Union

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads JSON file and returns data dictionary/list."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Saves data to JSON file with formatting."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def validate_retrieval_record(record: Dict[str, Any]) -> bool:
    """Validates that a retrieval record conforms to the team data contract."""
    required_keys = {"citation_id", "raw_text", "match_status"}
    return required_keys.issubset(record.keys())

def validate_final_record(record: Dict[str, Any]) -> bool:
    """Validates that a final record has all required fused output fields."""
    required_keys = {
        "citation_id",
        "graph_score",
        "semantic_score",
        "integrity_risk_score",
        "classification",
        "explanation",
        "evidence",
    }
    return required_keys.issubset(record.keys())
