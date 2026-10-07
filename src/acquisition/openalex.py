import requests
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.config import (
    OPENALEX_BASE_URL,
    OPENALEX_CONTACT_EMAIL,
    OPENALEX_API_KEY,
    RAW_OPENALEX_DIR,
)
from src.utils.text_utils import clean_reference_text

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
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _get_cache_path(self, query_key: str) -> Path:
        safe_key = "".join(c if c.isalnum() else "_" for c in query_key)[:100]
        return self.cache_dir / f"{safe_key}.json"

    def search_works_by_title(self, title: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Searches OpenAlex works by title string with caching and retries.
        """
        if not title or len(title.strip()) < 4 or title.startswith("http"):
            return []

        clean_title = clean_reference_text(title)
        clean_title = re.sub(r'^[,\.:\s"“\']+|[,\.:\s"”\']+$', '', clean_title).strip()
        cache_key = f"title_search_{clean_title}"
        cache_file = self._get_cache_path(cache_key)

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    if isinstance(cached_data, list) and len(cached_data) > 0:
                        return cached_data
            except Exception:
                pass

        params = {
            "search": clean_title,
            "per_page": max_results,
            "mailto": OPENALEX_CONTACT_EMAIL,
        }

        time.sleep(0.15)
        for attempt in range(5):
            try:
                response = self.session.get(OPENALEX_BASE_URL, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        
                    return results
                elif response.status_code in (429, 503):
                    time.sleep(2.0 * (attempt + 1))
            except Exception as e:
                if attempt == 4:
                    print(f"[OpenAlexAPI Warning] Failed to search title '{clean_title}': {e}")
                time.sleep(1.0 * (attempt + 1))
                # Reset session on connection error
                try:
                    self.session = requests.Session()
                    self.session.headers.update(self.headers)
                except Exception:
                    pass
                
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
        for attempt in range(3):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    return data
                elif response.status_code == 429:
                    time.sleep(1.5)
            except Exception:
                time.sleep(1.0)
                
        return None

    def get_work_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific work metadata by DOI (e.g. 10.1145/3437801.3441585)."""
        if not doi:
            return None
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "").strip()
        if not clean_doi:
            return None

        cache_key = f"doi_work_{clean_doi}"
        cache_file = self._get_cache_path(cache_key)

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"{OPENALEX_BASE_URL}/https://doi.org/{clean_doi}"
        for attempt in range(3):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    return data
                elif response.status_code == 429:
                    time.sleep(1.5)
            except Exception as e:
                if attempt == 2:
                    print(f"[OpenAlexAPI Warning] Failed DOI lookup '{clean_doi}': {e}")
                time.sleep(1.0)
                
        return None

    def verify_paper_in_openalex(self, title: str, doi: Optional[str] = None) -> Dict[str, Any]:
        """
        Verifies if target paper exists in OpenAlex by title search (with DOI fallback if needed).
        Returns detailed OpenAlex metadata including citation count and topics.
        """
        work = None
        match_method = "unmatched"

        # 1. Primary: Search OpenAlex by Title
        if title:
            results = self.search_works_by_title(title, max_results=5)
            if results:
                clean_t1 = "".join(c for c in title.lower() if c.isalnum())
                for cand in results:
                    cand_title = cand.get("title", "")
                    clean_t2 = "".join(c for c in cand_title.lower() if c.isalnum())
                    if clean_t1 and clean_t2 and (clean_t1 in clean_t2 or clean_t2 in clean_t1):
                        work = cand
                        match_method = "title_search"
                        break

        # 2. Fallback: Try DOI direct lookup if title search did not match
        if not work and doi:
            work = self.get_work_by_doi(doi)
            if work:
                match_method = "doi_exact"

        if work:
            concepts = [c.get("display_name") for c in work.get("concepts", []) if c.get("display_name")]
            return {
                "is_present": True,
                "openalex_id": work.get("id"),
                "short_id": work.get("id", "").split("/")[-1] if work.get("id") else None,
                "cited_by_count": work.get("cited_by_count", 0),
                "landing_page_url": work.get("doi") or work.get("id"),
                "publication_date": work.get("publication_date"),
                "concepts": concepts[:5],
                "match_method": match_method,
            }

        return {
            "is_present": False,
            "openalex_id": None,
            "short_id": None,
            "cited_by_count": 0,
            "landing_page_url": None,
            "publication_date": None,
            "concepts": [],
            "match_method": "unmatched",
        }
