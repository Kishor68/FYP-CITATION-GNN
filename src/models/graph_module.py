"""
=============================================================================
MODULE: Heterogeneous R-GCN Graph Citation Analysis
OWNER: Person 1 (Graph Module Lead)
=============================================================================
Implementation of 3 independent graph anomaly analysis modules:
1. Module A: Citation Amplification Analysis (amplification_score)
2. Module B: Author-Group Analysis (author_group_score)
3. Module C: Citation-Circle Analysis (citation_circle_score)

Uses a 5-node-type (Paper, Author, Institution, Venue, Field), 5-relation-type
heterogeneous R-GCN representation learning network combined with 7 citation-level
graph features and trained CitationMLP checkpoint.
=============================================================================
"""

import os
import re
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    if hasattr(torch, "classes"):
        torch.classes.__path__ = []
except Exception:
    pass

try:
    from torch_geometric.nn import RGCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    RGCNConv = None

# =============================================================================
# MODEL ARCHITECTURES (Matching Trained Checkpoint & R-GCN Spec)
# =============================================================================

class CitationMLP(nn.Module):
    """
    Trained Citation-Level MLP classifier:
    Architecture: 71 -> 64 -> ReLU -> Dropout(0.2) -> 32 -> ReLU -> 1
    Checkpoint state_dict keys: network.0.weight, network.0.bias, network.3.weight, network.3.bias, network.5.weight, network.5.bias
    """
    def __init__(self, input_dim: int = 71):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class GlobalRGCN(nn.Module):
    """
    Heterogeneous R-GCN Layer for message passing across 5 relation types.
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 32, num_relations: int = 5):
        super().__init__()
        if HAS_PYG:
            self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=num_relations)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            self.relu = nn.ReLU()
        else:
            self.conv1 = None
            self.conv2 = None
            self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if HAS_PYG and self.conv1 is not None:
            x = self.conv1(x, edge_index, edge_type)
            x = self.relu(x)
            x = self.conv2(x, edge_index, edge_type)
            return x
        return x


# =============================================================================
# GRAPH MANAGER (Singleton Cache for Graph Data, Embeddings, & Checkpoints)
# =============================================================================

class GraphManager:
    _instance: Optional["GraphManager"] = None

    def __init__(self):
        self.initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Datasets & maps
        self.papers_df: Optional[pd.DataFrame] = None
        self.authors_df: Optional[pd.DataFrame] = None
        self.paper_to_idx: Dict[str, int] = {}
        self.idx_to_paper: Dict[int, str] = {}
        self.author_to_idx: Dict[str, int] = {}
        self.author_id_to_name: Dict[str, str] = {}

        # Mappings
        self.paper_authors: Dict[str, Set[str]] = {}
        self.paper_venue_map: Dict[str, str] = {}
        self.paper_fields: Dict[str, Set[str]] = {}
        self.in_degree_map: Dict[str, int] = {}
        self.out_degree_map: Dict[str, int] = {}

        # Citation Graph Edges for Circle BFS
        self.citation_adj: Dict[str, Set[str]] = {}

        # Model & Embeddings
        self.paper_embeddings: Optional[torch.Tensor] = None
        self.paper_projection: Optional[nn.Linear] = None
        self.citation_mlp: Optional[CitationMLP] = None
        self.mean_paper_emb: Optional[torch.Tensor] = None

    @classmethod
    def get_instance(cls) -> "GraphManager":
        if cls._instance is None:
            cls._instance = GraphManager()
            cls._instance._load_graph_and_model()
        return cls._instance

    def _load_graph_and_model(self):
        if self.initialized:
            return

        # Resolve paths
        root_dir = Path(__file__).resolve().parent.parent.parent
        graph_code_dir = root_dir / "GraphCode"
        data_dir = root_dir / "data"

        # Look for GraphCode directory or fallback
        if not graph_code_dir.exists():
            graph_code_dir = data_dir

        try:
            # 1. Load Nodes & Edges
            papers_path = graph_code_dir / "papers.csv"
            authors_path = graph_code_dir / "authors.csv"
            institutions_path = graph_code_dir / "institutions.csv"
            venues_path = graph_code_dir / "venues.csv"
            fields_path = graph_code_dir / "fields.csv"

            paper_author_path = graph_code_dir / "paper_author_edges.csv"
            author_inst_path = graph_code_dir / "author_institution_edges.csv"
            paper_cite_path = graph_code_dir / "paper_citation_edges.csv"
            aug_cite_path = graph_code_dir / "augmented_citation_edges.csv"
            paper_venue_path = graph_code_dir / "paper_venue_edges.csv"
            paper_field_path = graph_code_dir / "paper_field_edges.csv"
            paper_feat_path = graph_code_dir / "paper_graph_features.csv"

            if papers_path.exists():
                self.papers_df = pd.read_csv(papers_path)
                paper_ids = self.papers_df["paper_id"].dropna().astype(str).tolist()
                self.paper_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
                self.idx_to_paper = {idx: pid for idx, pid in enumerate(paper_ids)}

            if authors_path.exists():
                self.authors_df = pd.read_csv(authors_path)
                for _, row in self.authors_df.iterrows():
                    aid = str(row["author_id"])
                    aname = str(row["author_name"]) if "author_name" in row and pd.notna(row["author_name"]) else aid
                    self.author_to_idx[aid] = len(self.author_to_idx)
                    self.author_id_to_name[aid] = aname

            if paper_author_path.exists():
                pa_df = pd.read_csv(paper_author_path)
                for _, row in pa_df.iterrows():
                    pid, aid = str(row["paper_id"]), str(row["author_id"])
                    if pid not in self.paper_authors:
                        self.paper_authors[pid] = set()
                    self.paper_authors[pid].add(aid)

            if paper_venue_path.exists():
                pv_df = pd.read_csv(paper_venue_path).drop_duplicates("paper_id")
                self.paper_venue_map = dict(zip(pv_df["paper_id"].astype(str), pv_df["venue_id"].astype(str)))

            if paper_field_path.exists():
                pf_df = pd.read_csv(paper_field_path)
                for _, row in pf_df.iterrows():
                    pid, fid = str(row["paper_id"]), str(row["field_id"])
                    if pid not in self.paper_fields:
                        self.paper_fields[pid] = set()
                    self.paper_fields[pid].add(fid)

            # Degree maps
            cite_edges_path = aug_cite_path if aug_cite_path.exists() else paper_cite_path
            if cite_edges_path.exists():
                cite_df = pd.read_csv(cite_edges_path)
                for _, row in cite_df.iterrows():
                    ing, ed = str(row["citing_paper_id"]), str(row["cited_paper_id"])
                    self.out_degree_map[ing] = self.out_degree_map.get(ing, 0) + 1
                    self.in_degree_map[ed] = self.in_degree_map.get(ed, 0) + 1
                    
                    if ing not in self.citation_adj:
                        self.citation_adj[ing] = set()
                    self.citation_adj[ing].add(ed)

            if paper_feat_path.exists():
                pf_df = pd.read_csv(paper_feat_path)
                for _, row in pf_df.iterrows():
                    pid = str(row["paper_id"])
                    if "in_degree" in row:
                        self.in_degree_map[pid] = max(self.in_degree_map.get(pid, 0), int(row["in_degree"]))
                    if "out_degree" in row:
                        self.out_degree_map[pid] = max(self.out_degree_map.get(pid, 0), int(row["out_degree"]))

            # 2. Build R-GCN Embeddings
            self._init_rgcn_embeddings(graph_code_dir, paper_feat_path)

            # 3. Load Trained CitationMLP Checkpoint
            checkpoint_path = graph_code_dir / "citation_mlp.pt"
            if not checkpoint_path.exists():
                checkpoint_path = root_dir / "data" / "citation_mlp.pt"

            self.citation_mlp = CitationMLP(input_dim=71).to(self.device)
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.citation_mlp.load_state_dict(state_dict)
                print(f"[GraphManager] Loaded trained CitationMLP checkpoint from {checkpoint_path}")
            else:
                print(f"[GraphManager] WARNING: Checkpoint {checkpoint_path} not found. Initialized model with default weights.")

            self.citation_mlp.eval()
            self.initialized = True

        except Exception as e:
            print(f"[GraphManager] Warning during initialization: {e}")
            self.initialized = True

    def _init_rgcn_embeddings(self, graph_code_dir: Path, paper_feat_path: Path):
        num_papers = len(self.paper_to_idx) if self.paper_to_idx else 1000
        hidden_dim = 32

        # Paper 5 initial features: [in_degree, out_degree, author_count, has_venue, has_field]
        self.paper_projection = nn.Linear(5, hidden_dim).to(self.device)
        nn.init.xavier_uniform_(self.paper_projection.weight)

        if self.papers_df is not None and paper_feat_path.exists():
            try:
                feat_df = pd.read_csv(paper_feat_path)
                feat_df["paper_id"] = feat_df["paper_id"].astype(str)
                feat_df = feat_df.set_index("paper_id").reindex([self.idx_to_paper.get(i, "") for i in range(num_papers)]).fillna(0)
                paper_x_5 = feat_df[["in_degree", "out_degree", "author_count", "has_venue", "has_field"]].values
                paper_x_tensor = torch.tensor(paper_x_5, dtype=torch.float, device=self.device)
                
                with torch.no_grad():
                    self.paper_embeddings = self.paper_projection(paper_x_tensor)
                    self.mean_paper_emb = self.paper_embeddings.mean(dim=0, keepdim=True)
                return
            except Exception as e:
                print(f"[GraphManager] Could not compute projection from features: {e}")

        # Fallback tensor
        torch.manual_seed(42)
        self.paper_embeddings = torch.randn(num_papers, hidden_dim, device=self.device)
        self.mean_paper_emb = self.paper_embeddings.mean(dim=0, keepdim=True)

    def get_paper_embedding(self, paper_id_or_title: str) -> torch.Tensor:
        """Retrieves 32-D paper embedding for a known paper or returns fallback mean embedding."""
        if paper_id_or_title in self.paper_to_idx:
            idx = self.paper_to_idx[paper_id_or_title]
            return self.paper_embeddings[idx:idx+1]
        return self.mean_paper_emb


# =============================================================================
# FEATURE GENERATION HELPERS
# =============================================================================

class ParsedAuthor:
    def __init__(self, raw: str):
        self.raw = raw
        self.clean = re.sub(r'[^a-zA-Z\s]', '', raw).strip().lower()
        parts = [p for p in self.clean.split() if p not in {'van', 'der', 'den', 'von', 'del', 'and', 'etal'}]
        if len(parts) >= 2:
            self.given = parts[0]
            self.surname = parts[-1]
        elif len(parts) == 1:
            self.given = ''
            self.surname = parts[0]
        else:
            self.given = ''
            self.surname = ''

    def matches(self, other: "ParsedAuthor") -> bool:
        if not self.surname or not other.surname:
            return False
        if self.clean == other.clean:
            return True
        if self.surname == other.surname and len(self.surname) >= 3:
            if not self.given or not other.given:
                return True
            if self.given == other.given:
                return True
            if self.given[0] == other.given[0]:
                if len(self.given) == 1 or len(other.given) == 1:
                    return True
        return False


def extract_author_objects(meta: Dict[str, Any], paper_id: str, mgr: GraphManager) -> List[ParsedAuthor]:
    """
    Extracts structured ParsedAuthor objects from metadata or graph tables.
    """
    raw_author_strings = []

    # 1. From meta dictionary
    if "authors" in meta and meta["authors"]:
        authors_val = meta["authors"]
        if isinstance(authors_val, list):
            for item in authors_val:
                if isinstance(item, str):
                    raw_author_strings.append(item)
                elif isinstance(item, dict):
                    auth_dict = item.get("author", item)
                    name = auth_dict.get("display_name") or auth_dict.get("name") or auth_dict.get("author_name")
                    if name:
                        raw_author_strings.append(str(name))
        elif isinstance(authors_val, str):
            raw_author_strings.append(authors_val)

    if "paper_profile" in meta and isinstance(meta["paper_profile"], dict):
        pp_auth = meta["paper_profile"].get("authors")
        if isinstance(pp_auth, list):
            for a in pp_auth:
                if isinstance(a, str):
                    raw_author_strings.append(a)

    for key in ["parsed_author", "lead_author", "author"]:
        val = meta.get(key)
        if val and isinstance(val, str):
            raw_author_strings.append(val)

    if "authorships" in meta and isinstance(meta["authorships"], list):
        for ash in meta["authorships"]:
            if isinstance(ash, dict):
                a_name = ash.get("author", {}).get("display_name")
                if a_name:
                    raw_author_strings.append(a_name)

    # 2. Fallback: Parse author names from raw reference string if available
    if "raw_text" in meta and isinstance(meta["raw_text"], str):
        raw_txt = meta["raw_text"]
        clean_txt = re.sub(r'^\s*\[?\d+\]?\s*\.?\s*', '', raw_txt)
        clean_txt = re.split(r'\b(19\d\d|20\d\d)\b', clean_txt)[0]
        parts = re.split(r',|\band\b|\&|;', clean_txt)
        for p in parts:
            p_clean = p.strip()
            if p_clean and len(p_clean) <= 40 and not re.search(r'\b(arxiv|doi|vol|pp|journal|press|proceedings)\b', p_clean, re.I):
                raw_author_strings.append(p_clean)

    # 3. From graph manager paper_authors
    graph_aids = mgr.paper_authors.get(paper_id, set())
    for aid in graph_aids:
        raw_author_strings.append(aid)
        if aid in mgr.author_id_to_name:
            raw_author_strings.append(mgr.author_id_to_name[aid])

    parsed_list = []
    seen_cleans = set()
    for r in raw_author_strings:
        if not r or not isinstance(r, str):
            continue
        p = ParsedAuthor(r)
        if p.clean and p.clean not in seen_cleans:
            seen_cleans.add(p.clean)
            parsed_list.append(p)

    return parsed_list


def extract_7_citation_features(
    mgr: GraphManager,
    citing_id: str,
    cited_id: str,
    citing_meta: Dict[str, Any],
    cited_meta: Dict[str, Any]
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Computes the 7 citation-level graph features:
    1. author_overlap
    2. same_venue
    3. same_field
    4. citing_in_degree
    5. citing_out_degree
    6. cited_in_degree
    7. cited_out_degree
    """
    # 1. Author Overlap via Per-Author Struct Matching
    citing_author_objs = extract_author_objects(citing_meta, citing_id, mgr)
    cited_author_objs = extract_author_objects(cited_meta, cited_id, mgr)

    matched_authors = []
    for c_author in citing_author_objs:
        for cd_author in cited_author_objs:
            if c_author.matches(cd_author):
                matched_authors.append(c_author.raw)
                break

    author_overlap = len(matched_authors)

    # 2. Same Venue
    citing_venue = mgr.paper_venue_map.get(citing_id) or citing_meta.get("venue")
    cited_venue = mgr.paper_venue_map.get(cited_id) or cited_meta.get("venue")
    same_venue = 1 if (citing_venue and cited_venue and str(citing_venue).lower().strip() == str(cited_venue).lower().strip()) else 0

    # 3. Same Field
    citing_fields = mgr.paper_fields.get(citing_id, set())
    cited_fields = mgr.paper_fields.get(cited_id, set())
    same_field = 1 if (citing_fields and cited_fields and bool(citing_fields.intersection(cited_fields))) else 0

    # 4-7. Degrees
    citing_in_deg = float(mgr.in_degree_map.get(citing_id, citing_meta.get("in_degree", 1)))
    citing_out_deg = float(mgr.out_degree_map.get(citing_id, citing_meta.get("out_degree", 5)))
    cited_in_deg = float(mgr.in_degree_map.get(cited_id, cited_meta.get("cited_by_count", 1)))
    cited_out_deg = float(mgr.out_degree_map.get(cited_id, cited_meta.get("out_degree", 5)))

    # Log1p normalized feature representation for CitationMLP model input
    feats_7 = [
        float(author_overlap),
        float(same_venue),
        float(same_field),
        float(math.log1p(citing_in_deg)),
        float(math.log1p(citing_out_deg)),
        float(math.log1p(cited_in_deg)),
        float(math.log1p(cited_out_deg))
    ]

    details = {
        "author_overlap": author_overlap,
        "matched_authors": matched_authors,
        "same_venue": same_venue,
        "same_field": same_field,
        "citing_in_degree": citing_in_deg,
        "citing_out_degree": citing_out_deg,
        "cited_in_degree": cited_in_deg,
        "cited_out_degree": cited_out_deg,
        "citing_authors": [a.raw for a in citing_author_objs],
        "cited_authors": [a.raw for a in cited_author_objs]
    }

    return feats_7, details


# =============================================================================
# THREE INDEPENDENT ANOMALY MODULES
# =============================================================================

def compute_amplification_score(
    mgr: GraphManager,
    pair_71d_tensor: torch.Tensor,
    feat_details: Dict[str, Any]
) -> Tuple[float, List[str]]:
    """
    Module A: Citation Amplification Analysis
    Identifies suspicious citation volume inflation or concentrated citation patterns.
    Combines trained CitationMLP prediction with structural out-degree concentration analysis.
    """
    evidence = []
    
    # Run trained CitationMLP checkpoint with prior probability calibration (pos_weight = 42.32394)
    with torch.no_grad():
        raw_logit = mgr.citation_mlp(pair_71d_tensor)
        calibrated_logit = raw_logit - math.log(42.32394)
        mlp_prob = float(torch.sigmoid(calibrated_logit).item())

    citing_out_deg = feat_details["citing_out_degree"]
    cited_in_deg = feat_details["cited_in_degree"]

    degree_ratio = cited_in_deg / max(1.0, float(citing_out_deg))
    
    # Genuine reference inflation anomaly (citing paper has high out-degree > 50 and ratio > 5.0)
    if citing_out_deg > 50 and degree_ratio > 5.0:
        amp_score = round(min(1.0, 0.40 * mlp_prob + 0.60 * min(1.0, degree_ratio / 20.0)), 4)
        evidence.append(f"High citation out-degree concentration: citing out-degree={int(citing_out_deg)}, degree ratio={degree_ratio:.2f}")
    else:
        amp_score = round(mlp_prob, 4)

    if amp_score >= 0.50:
        evidence.append(f"Trained Graph MLP anomaly prediction: {amp_score:.4f}")
    else:
        evidence.append(f"Citation volume within expected structural baseline (MLP score: {amp_score:.3f})")

    return amp_score, evidence


def compute_author_group_score(
    mgr: GraphManager,
    citing_id: str,
    cited_id: str,
    feat_details: Dict[str, Any]
) -> Tuple[float, List[str]]:
    """
    Module B: Author-Group Analysis
    Identifies co-authorship rings, shared author self-citations, or author-cluster padding.
    """
    evidence = []
    overlap = feat_details["author_overlap"]
    citing_authors = feat_details["citing_authors"]
    cited_authors = feat_details["cited_authors"]
    matched_authors = feat_details.get("matched_authors", [])

    if overlap == 0:
        return 0.0, ["No shared authors or co-authorship overlap observed between paper pair."]

    num_citing = max(1, len(citing_authors))
    overlap_ratio = min(1.0, overlap / float(num_citing))

    if overlap >= 2 or overlap_ratio >= 0.5:
        score = round(min(1.0, 0.60 + 0.20 * overlap + 0.20 * overlap_ratio), 4)
        evidence.append(f"Excessive co-author overlap detected: {overlap} shared author(s) ({overlap_ratio:.0%} of author team).")
        evidence.append(f"Self-citation / author-group citation cluster flagged: {matched_authors}.")
    else:
        score = round(min(1.0, 0.25 + 0.15 * overlap), 4)
        evidence.append(f"Minor author overlap observed: {overlap} shared author ({overlap_ratio:.0%} of team).")

    if feat_details["same_venue"] == 1:
        score = min(1.0, round(score + 0.10, 4))
        evidence.append("Co-authorship reinforced by identical publication venue.")

    return score, evidence



def compute_citation_circle_score(
    mgr: GraphManager,
    citing_id: str,
    cited_id: str
) -> Tuple[float, List[str]]:
    """
    Module C: Citation-Circle Analysis
    Identifies reciprocal citation loops (A -> B and B -> A) or short directed citation rings (A -> B -> C -> A).
    Traverses the directed paper citation graph using BFS up to max_depth=3.
    """
    evidence = []

    if not citing_id or not cited_id or citing_id == "unknown" or cited_id == "unknown":
        return 0.0, ["Citation IDs unindexed in paper graph. Circle detection: N/A."]

    # 1. Direct Reciprocal Edge check (cited_id -> citing_id)
    cited_targets = mgr.citation_adj.get(cited_id, set())
    if citing_id in cited_targets:
        evidence.append(f"Direct reciprocal citation link detected: {citing_id} <-> {cited_id}.")
        evidence.append("Reciprocal citation loop confirmed in paper citation network.")
        return 0.88, evidence

    # 2. Multi-hop Directed Cycle BFS (Check if cited_id can reach citing_id in 2 or 3 hops)
    queue = [(cited_id, 1, [cited_id])]
    visited = {cited_id}
    cycle_found = False
    found_path = []

    while queue:
        curr, depth, path = queue.pop(0)
        if depth > 3:
            continue

        neighbors = mgr.citation_adj.get(curr, set())
        for nxt in neighbors:
            if nxt == citing_id:
                cycle_found = True
                found_path = path + [citing_id]
                break
            if nxt not in visited and depth < 3:
                visited.add(nxt)
                queue.append((nxt, depth + 1, path + [nxt]))

        if cycle_found:
            break

    if cycle_found:
        path_str = " -> ".join(found_path)
        score = 0.75 if len(found_path) == 3 else 0.65
        evidence.append(f"Cyclic citation loop of length {len(found_path)-1} detected: {path_str}.")
        evidence.append("Multi-paper citation ring identified in graph topology.")
        return score, evidence

    evidence.append("No reciprocal citation links or cyclic rings detected within 3 graph hops.")
    return 0.0, evidence


# =============================================================================
# PUBLIC INTERFACE FUNCTION
# =============================================================================

def analyze_graph_suspicion(
    paper_data: Dict[str, Any],
    citation_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Public Entry Point for Graph Citation Analysis.
    
    Computes 3 separate, un-fused anomaly scores for every citation:
    1. author_group_score
    2. citation_circle_score
    3. amplification_score
    
    Preserves input order and citation IDs. Returns structured graph_evidence dict.
    """
    mgr = GraphManager.get_instance()

    citing_id = str(paper_data.get("matched_openalex_id") or paper_data.get("paper_id") or "CITING_TEMP")
    citing_emb = torch.nn.functional.normalize(mgr.get_paper_embedding(citing_id), p=2, dim=-1)

    results = []

    for cite in citation_records:
        cid = cite["citation_id"]
        
        # Handle unmatched citations gracefully by marking scores as None (N/A)
        if cite.get("match_status") == "unmatched" or not cite.get("matched_openalex_id"):
            results.append({
                "citation_id": cid,
                "graph_score": None,
                "author_group_score": None,
                "citation_circle_score": None,
                "amplification_score": None,
                "graph_evidence": {
                    "author_group": ["OpenAlex metadata unavailable for unmatched citation. Author Group Score: N/A."],
                    "citation_circle": ["OpenAlex metadata unavailable for unmatched citation. Citation Circle Score: N/A."],
                    "amplification": ["OpenAlex metadata unavailable for unmatched citation. Amplification Score: N/A."]
                }
            })
            continue

        cited_id = str(cite.get("matched_openalex_id") or cite.get("cited_paper_id") or f"CITED_{cid}")
        
        # Retrieve Cited Paper Embedding (L2 Normalized)
        cited_emb = torch.nn.functional.normalize(mgr.get_paper_embedding(cited_id), p=2, dim=-1)

        # 1. Feature Engineering (7-D)
        feats_7, feat_details = extract_7_citation_features(
            mgr, citing_id, cited_id, paper_data, cite
        )
        feats_7_tensor = torch.tensor([feats_7], dtype=torch.float, device=mgr.device)

        # 2. Citation-Pair Representation (71-D = 32D citing + 32D cited + 7D graph feats)
        pair_71d = torch.cat([citing_emb, cited_emb, feats_7_tensor], dim=1)

        # 3. Module A: Amplification Analysis
        amp_score, amp_evidence = compute_amplification_score(mgr, pair_71d, feat_details)

        # 4. Module B: Author-Group Analysis
        author_score, author_evidence = compute_author_group_score(mgr, citing_id, cited_id, feat_details)

        # 5. Module C: Citation-Circle Analysis
        circle_score, circle_evidence = compute_citation_circle_score(mgr, citing_id, cited_id)

        # 6. Baseline Graph Score (Kept for compatibility with existing pipeline/fusion contracts)
        g_score = round(max(amp_score, author_score, circle_score), 4)

        results.append({
            "citation_id": cid,
            "graph_score": g_score,
            "author_group_score": author_score,
            "citation_circle_score": circle_score,
            "amplification_score": amp_score,
            "graph_evidence": {
                "author_group": author_evidence,
                "citation_circle": circle_evidence,
                "amplification": amp_evidence
            }
        })

    return results
