import requests
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.config import (
    OPENALEX_BASE_URL,
    OPENALEX_CONTACT_EMAIL,
    OPENALEX_API_KEY,
    RAW_OPENALEX_DIR,
)

class OpenAlexAPI:
    """Handles paper search and work metadata retrieval from OpenAlex API."""

    def __init__(self, cache_dir: Path = RAW_OPENALEX_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "User-Agent": f"CitationIntegrityAnalyzer/1.0 (mailto:{OPENALEX_CONTACT_EMAIL})"
        }
        if OPENALEX_API_KEY:
            self.headers["api_key"] = OPENALEX_API_KEY

    def _get_cache_path(self, query_key: str) -> Path:
        safe_key = "".join(c if c.isalnum() else "_" for c in query_key)[:100]
        return self.cache_dir / f"{safe_key}.json"

    def search_works_by_title(self, title: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Searches OpenAlex works by title string with caching.
        """
        cache_key = f"title_search_{title}"
        cache_file = self._get_cache_path(cache_key)

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        params = {
            "search": title,
            "per_page": max_results,
        }

        try:
            response = requests.get(OPENALEX_BASE_URL, headers=self.headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Save raw response to cache
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
                return results
            else:
                return []
        except Exception as e:
            print(f"[OpenAlexAPI Warning] Failed to search title '{title}': {e}")
            return []

    def get_work_by_id(self, openalex_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific work metadata by OpenAlex ID (e.g. W12345678)."""
        clean_id = openalex_id.split("/")[-1]
        cache_file = self._get_cache_path(f"work_{clean_id}")

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"{OPENALEX_BASE_URL}/{clean_id}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return data
            return None
        except Exception:
            return None
