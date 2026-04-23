#!/usr/bin/env python3
"""
Evaluation pipeline using manual relevance judgments.
Runs 4 retrieval configurations and calculates MRR@5, Recall@5.
"""
import csv, hashlib, json, os, pickle, random, time, sys
from dotenv import load_dotenv
csv.field_size_limit(10 * 1024 * 1024)
from collections import deque

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import GraphDatabase


import sys, pathlib
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "retrieval"))

from query_aware_graph import MetaPathBestFirstGraph

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv(ROOT / ".env")

FAISS_VECS = ROOT / "data" / "vectors.npy"
FAISS_PIDS = ROOT / "data" / "vector_paperids.json"
NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

# ═══════════════════════════════════════════════════════════════════════════════
# DISK CACHE
# Persists vector search, graph BFS, and reranker results across runs.
# Keyed by a hash of the function name + inputs; stored as pickle files.
# Delete results/eval_cache/ to force a full re-run.
# ═══════════════════════════════════════════════════════════════════════════════
CACHE_DIR = ROOT / "results" / "eval_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key(*args) -> str:
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()

def _cache_get(fn: str, *args):
    p = CACHE_DIR / f"{fn}_{_cache_key(*args)}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return True, pickle.load(f)
    return False, None

def _cache_put(fn: str, value, *args) -> None:
    p = CACHE_DIR / f"{fn}_{_cache_key(*args)}.pkl"
    with open(p, "wb") as f:
        pickle.dump(value, f)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS + CONNECT DBs
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading models...")
embed_model = SentenceTransformer("jordyvl/scibert_scivocab_uncased_sentence_transformer")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def create_embedding(texts):
    return embed_model.encode(texts)

print("Loading FAISS index...")
if not FAISS_VECS.exists() or not FAISS_PIDS.exists():
    sys.exit(
        "FAISS data not found. Run the one-time extraction in WSL2 first:\n"
        "  python scripts/extract_vectors.py"
    )
_raw_vecs    = np.load(str(FAISS_VECS))             # (N, 768) float32
_vector_pids = json.load(open(FAISS_PIDS))           # list of N paper IDs
faiss.normalize_L2(_raw_vecs)                        # cosine sim via inner product
_faiss_index = faiss.IndexFlatIP(_raw_vecs.shape[1])
_faiss_index.add(_raw_vecs)
print(f"  FAISS: {_faiss_index.ntotal} vectors ({_raw_vecs.shape[1]}-dim)")

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ═══════════════════════════════════════════════════════════════════════════════
# PaperBFS
# ═══════════════════════════════════════════════════════════════════════════════
class PaperBFS:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    def _get_neighbors(self, session, label, node_id, rel_type):
        query = f"MATCH (n:{label} {{id: $id}})-[:{rel_type}]-(m) RETURN labels(m)[0] AS label, m.id AS id"
        return list(session.run(query, id=node_id))
    def bfs_nearest_papers(self, start_paper_id, k, max_hops=4):
        # Four independent BFS channels -- one per relationship dimension.
        # Each channel explores only its own edge types so that high-degree
        # relationship types (like CITES) cannot starve the others.
        channels = {
            "cites":    [("Paper", "CITES")],
            "coauthor": [("Paper", "WROTE"), ("Author", "WROTE")],
            "venue":    [("Paper", "PUBLISHED_IN"), ("Venue", "PUBLISHED_IN")],
            "field":    [("Paper", "HAS_FIELD"), ("FieldOfStudy", "HAS_FIELD")],
        }
        channel_results = {}
        with self.driver.session() as session:
            for ch_name, ch_plan in channels.items():
                queue = deque([("Paper", start_paper_id, 0)])
                visited = {("Paper", start_paper_id)}
                found = []
                while queue and len(found) < k:
                    label, node_id, depth = queue.popleft()
                    if depth >= max_hops:
                        continue
                    if label == "Paper" and node_id != start_paper_id:
                        found.append({"paper_id": node_id, "distance": depth})
                    for src_label, rel in ch_plan:
                        if label != src_label:
                            continue
                        records = self._get_neighbors(session, label, node_id, rel)
                        for r in records:
                            key = (r["label"], r["id"])
                            if key not in visited:
                                visited.add(key)
                                queue.append((r["label"], r["id"], depth + 1))
                channel_results[ch_name] = found
        # Round-robin interleave across channels for fair representation
        found_papers = []
        seen = set()
        max_len = max((len(v) for v in channel_results.values()), default=0)
        for i in range(max_len):
            for ch_name in channels:
                results = channel_results[ch_name]
                if i < len(results):
                    p = results[i]
                    if p["paper_id"] not in seen:
                        seen.add(p["paper_id"])
                        found_papers.append(p)
                        if len(found_papers) == k:
                            return found_papers
        return found_papers

bfs = PaperBFS(driver)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading papers and ground truth...")
all_papers = {}
with open(ROOT / "data" / "papers.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row["id"]
        if pid not in all_papers:
            all_papers[pid] = {
                "paperId": pid, "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
            }
print(f"  {len(all_papers)} papers loaded")

# Load pre-computed paper embeddings (run scripts/precompute_embeddings.py to build cache)
EMB_NPY  = ROOT / "data" / "paper_embeddings.npy"
IDS_JSON = ROOT / "data" / "paper_ids.json"

if EMB_NPY.exists() and IDS_JSON.exists():
    import json as _json
    print("Loading cached paper embeddings from disk...")
    _emb_matrix  = np.load(str(EMB_NPY))
    _cached_ids  = _json.load(open(IDS_JSON, encoding="utf-8"))
    paper_embeddings = {pid: _emb_matrix[i] for i, pid in enumerate(_cached_ids)}
    # Fill in any papers added after the cache was built
    _missing = [pid for pid in all_papers if pid not in paper_embeddings]
    if _missing:
        print(f"  Encoding {len(_missing)} papers not in cache...")
        _texts  = [(all_papers[p]["title"] + " " + all_papers[p]["abstract"]).strip() for p in _missing]
        _vecs   = embed_model.encode(_texts, batch_size=128, show_progress_bar=False)
        for pid, vec in zip(_missing, _vecs):
            paper_embeddings[pid] = vec
else:
    print("No embedding cache found — computing now (run scripts/precompute_embeddings.py to pre-build)...")
    _paper_id_list = list(all_papers.keys())
    _paper_texts = [
        (all_papers[pid]["title"] + " " + all_papers[pid]["abstract"]).strip()
        for pid in _paper_id_list
    ]
    _paper_emb_matrix = embed_model.encode(_paper_texts, batch_size=128, show_progress_bar=True)
    paper_embeddings = {pid: emb for pid, emb in zip(_paper_id_list, _paper_emb_matrix)}

metapath_graph = MetaPathBestFirstGraph(driver, embed_model, paper_embeddings)
print(f"  {len(paper_embeddings)} paper embeddings ready")

# Load manual relevance judgments
relevance_by_query = {}  # query_id -> set of relevant paperIds
ground_truth_pids = set()  # ALL 250 judged papers (relevant or not)
with open(ROOT / "data" / "ground_truth_relevance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qi = int(row["query_id"])
        if qi not in relevance_by_query:
            relevance_by_query[qi] = set()
        ground_truth_pids.add(row["paperId"])
        if int(row["relevant"]) == 1:
            relevance_by_query[qi].add(row["paperId"])

QUERIES = [
    "How do graph-based and multi-step retrieval methods enhance retrieval-augmented generation systems?",
    "How are knowledge graphs constructed and applied to organize scholarly and scientific information?",
    "What methods are used for automated information extraction from scientific and biomedical text?",
    "How can NLP tools support researchers in conducting systematic literature reviews and evidence synthesis?",
    "What bibliometric and citation analysis methods reveal research trends and collaboration patterns?",
    "What techniques reduce factual hallucination in large language model outputs?",
    "What deep learning and language model approaches are used for classifying, tagging, or organizing scientific text?",
    "How are pretrained language models and embedding techniques used for semantic understanding of scientific text?",
    "What benchmarks, evaluation methods, and datasets are used to assess RAG and information retrieval system performance?",
    "How can AI systems assist in peer review, quality assessment, and research impact evaluation of scholarly work?",
]

for qi in range(1, 11):
    print(f"  Q{qi}: {len(relevance_by_query.get(qi, set()))} relevant papers")

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def mrr_at_k(ranked_ids, relevant_ids, k=5):
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(ranked_ids, relevant_ids, k=5):
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)

# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def do_vector_search(query, n_papers=5):
    """Retrieve n_papers unique papers ranked by cosine similarity via FAISS."""
    hit, cached = _cache_get("vec", query, n_papers)
    if hit:
        print(f"    [vector] {len(cached)} papers (cached)", flush=True)
        return cached
    query_vec = create_embedding([query])[0].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)
    _, indices = _faiss_index.search(query_vec, _faiss_index.ntotal)
    result, seen = [], set()
    for idx in indices[0]:
        if idx < 0:
            continue
        pid = _vector_pids[idx]
        if pid not in seen:
            seen.add(pid)
            result.append(pid)
            if len(result) == n_papers:
                break
    _cache_put("vec", result, query, n_papers)
    print(f"    [vector] {len(result)} unique papers", flush=True)
    return result

def _bfs_single(sid, k, max_hops):
    """BFS from one seed paper — cached per (seed, k, max_hops)."""
    hit, cached = _cache_get("bfs", sid, k, max_hops)
    if hit:
        return cached
    try:
        neighbors = bfs.bfs_nearest_papers(sid, k=k, max_hops=max_hops)
        result = [n_["paper_id"] for n_ in neighbors]
    except Exception:
        result = []
    _cache_put("bfs", result, sid, k, max_hops)
    return result

def do_graph_search(seed_ids, k=50, max_hops=10):
    results = {}
    n = len(seed_ids)
    hits = 0
    for i, sid in enumerate(seed_ids):
        neighbors = _bfs_single(sid, k, max_hops)
        results[sid] = neighbors
        if (CACHE_DIR / f"bfs_{_cache_key(sid, k, max_hops)}.pkl").exists():
            hits += 1
        print(f"\r    [graph] seed {i+1}/{n}  (cache hits: {hits})", end="", flush=True)
    print()
    return results

def _metapath_single(query, sid, k, max_hops):
    """Meta-path BFS from one seed — cached per (query, seed, k, max_hops)."""
    hit, cached = _cache_get("meta", query, sid, k, max_hops)
    if hit:
        return cached
    try:
        neighbors = metapath_graph.retrieve(query, sid, k=k, max_hops=max_hops)
        result = [n_["paper_id"] for n_ in neighbors]
    except Exception:
        result = []
    _cache_put("meta", result, query, sid, k, max_hops)
    return result

def do_metapath_graph_search(seed_ids, query, k=15, max_hops=4):
    results = {}
    n = len(seed_ids)
    hits = 0
    for i, sid in enumerate(seed_ids):
        neighbors = _metapath_single(query, sid, k, max_hops)
        results[sid] = neighbors
        if (CACHE_DIR / f"meta_{_cache_key(query, sid, k, max_hops)}.pkl").exists():
            hits += 1
        print(f"\r    [metapath] seed {i+1}/{n}  (cache hits: {hits})", end="", flush=True)
    print()
    return results

def do_rerank(paper_ids, query, top_k=10):
    candidates = [pid for pid in paper_ids if pid in ground_truth_pids]
    if not candidates:
        return []
    hit, cached = _cache_get("rerank", query, tuple(sorted(candidates)), top_k)
    if hit:
        print(f"    [rerank] {len(cached)} papers (cached)", flush=True)
        return cached
    pairs, valid_ids = [], []
    for pid in candidates:
        p = all_papers.get(pid)
        if p:
            pairs.append((query, (p["title"] + " " + p["abstract"]).strip()))
            valid_ids.append(pid)
    if not pairs:
        return []
    print(f"    [rerank] scoring {len(pairs)} ground-truth papers...", flush=True)
    scores = reranker.predict(pairs)
    scored = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    result = [pid for pid, _ in scored[:top_k]]
    _cache_put("rerank", result, query, tuple(sorted(candidates)), top_k)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# 4 RETRIEVAL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def _unique_pool(vec_ids, graph_results):
    """Combine vector + graph IDs, deduped, preserving order."""
    all_ids = list(vec_ids)
    seen = set(vec_ids)
    for sid in vec_ids:
        for gid in graph_results.get(sid, []):
            if gid not in seen:
                seen.add(gid)
                all_ids.append(gid)
    return all_ids

HYBRID_VEC = 60   # vector seeds for hybrid (deduped pool ~300)
VECTOR_VEC = HYBRID_VEC  # vector pool size matches hybrid's vector component
GRAPH_K    = 15   # graph neighbors per seed paper

from collections import Counter

# ── shared hybrid helper: returns (vec_ids, graph_results, all_ids) ──────────
def _do_hybrid(query):
    vec_ids = do_vector_search(query, n_papers=HYBRID_VEC)
    graph_results = do_graph_search(vec_ids, k=GRAPH_K, max_hops=10)
    all_ids = _unique_pool(vec_ids, graph_results)
    return vec_ids, graph_results, all_ids

def retrieval_hybrid_reranker(query):
    vec_ids, graph_results, all_ids = _do_hybrid(query)
    return all_ids, do_rerank(all_ids, query, top_k=10)

def retrieval_hybrid_freq_no_reranker(query):
    vec_ids, graph_results, all_ids = _do_hybrid(query)
    # Rank by retrieval frequency: +1 for vector hit, +1 per graph seed that found it
    freq = Counter()
    for pid in vec_ids:
        freq[pid] += 1
    for sid, neighbors in graph_results.items():
        for pid in neighbors:
            freq[pid] += 1
    vec_rank = {pid: i for i, pid in enumerate(vec_ids)}
    gt_candidates = [pid for pid in freq if pid in ground_truth_pids]
    ranked = sorted(
        gt_candidates,
        key=lambda pid: (-freq[pid], vec_rank.get(pid, float('inf'))),
    )
    return all_ids, ranked[:10]

def retrieval_hybrid_interleave_no_reranker(query):
    vec_ids, graph_results, all_ids = _do_hybrid(query)
    # Interleave: all vector results first, then round-robin graph neighbors
    seen = set()
    ordered = []
    for vid in vec_ids:
        if vid not in seen:
            seen.add(vid)
            ordered.append(vid)
    max_graph_len = max((len(v) for v in graph_results.values()), default=0)
    for round_idx in range(max_graph_len):
        for vid in vec_ids:
            neighbors = graph_results.get(vid, [])
            if round_idx < len(neighbors):
                gid = neighbors[round_idx]
                if gid not in seen:
                    seen.add(gid)
                    ordered.append(gid)
    gt_ordered = [pid for pid in ordered if pid in ground_truth_pids]
    return all_ids, gt_ordered[:10]

def retrieval_vector_reranker(query):
    vec_ids = do_vector_search(query, n_papers=VECTOR_VEC)
    return vec_ids, do_rerank(vec_ids, query, top_k=10)

def retrieval_vector_only(query):
    vec_ids = do_vector_search(query, n_papers=VECTOR_VEC)
    gt_ranked = [pid for pid in vec_ids if pid in ground_truth_pids]
    return vec_ids, gt_ranked[:10]

def retrieval_vector_poolmatch_reranker(query):
    # Vector-only retrieval matching the hybrid pool size
    _, _, hybrid_pool = _do_hybrid(query)
    n = len(hybrid_pool)
    vec_ids = do_vector_search(query, n_papers=n)
    return vec_ids, do_rerank(vec_ids, query, top_k=10)

def retrieval_vector_poolmatch_only(query):
    # Vector-only retrieval matching the hybrid pool size
    _, _, hybrid_pool = _do_hybrid(query)
    n = len(hybrid_pool)
    vec_ids = do_vector_search(query, n_papers=n)
    gt_ranked = [pid for pid in vec_ids if pid in ground_truth_pids]
    return vec_ids, gt_ranked[:10]

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANT-BUDGET HYBRID (budget ~800 pre-dedup: VEC=60, GRAPH_K=12 -> 60*13=780)
# ═══════════════════════════════════════════════════════════════════════════════
BUDGET_VEC   = 80
BUDGET_GK    = 14   # 80 + 80*14 = 1200 pre-dedup retrievals

def _do_budget_hybrid(query):
    vec_ids = do_vector_search(query, n_papers=BUDGET_VEC)
    graph_results = do_graph_search(vec_ids, k=BUDGET_GK, max_hops=10)
    all_ids = _unique_pool(vec_ids, graph_results)
    return vec_ids, graph_results, all_ids

def retrieval_budget800_hybrid_reranker(query):
    _, _, all_ids = _do_budget_hybrid(query)
    return all_ids, do_rerank(all_ids, query, top_k=10)

def retrieval_budget800_hybrid_freq(query):
    vec_ids, graph_results, all_ids = _do_budget_hybrid(query)
    freq = Counter()
    for pid in vec_ids:
        freq[pid] += 1
    for sid, neighbors in graph_results.items():
        for pid in neighbors:
            freq[pid] += 1
    vec_rank = {pid: i for i, pid in enumerate(vec_ids)}
    gt_candidates = [pid for pid in freq if pid in ground_truth_pids]
    ranked = sorted(
        gt_candidates,
        key=lambda pid: (-freq[pid], vec_rank.get(pid, float('inf'))),
    )
    return all_ids, ranked[:10]

def retrieval_budget800_hybrid_interleave(query):
    vec_ids, graph_results, all_ids = _do_budget_hybrid(query)
    seen = set()
    ordered = []
    for vid in vec_ids:
        if vid not in seen:
            seen.add(vid)
            ordered.append(vid)
    max_graph_len = max((len(v) for v in graph_results.values()), default=0)
    for round_idx in range(max_graph_len):
        for vid in vec_ids:
            neighbors = graph_results.get(vid, [])
            if round_idx < len(neighbors):
                gid = neighbors[round_idx]
                if gid not in seen:
                    seen.add(gid)
                    ordered.append(gid)
    gt_ordered = [pid for pid in ordered if pid in ground_truth_pids]
    return all_ids, gt_ordered[:10]

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY-AWARE META-PATH HYBRID (VEC=60, GRAPH_K=15 to match regular hybrid)
# ═══════════════════════════════════════════════════════════════════════════════
METAPATH_VEC = 200
METAPATH_GK  = 20

def _do_metapath_hybrid(query):
    vec_ids = do_vector_search(query, n_papers=METAPATH_VEC)
    graph_results = do_metapath_graph_search(vec_ids, query, k=METAPATH_GK, max_hops=4)
    all_ids = _unique_pool(vec_ids, graph_results)
    return vec_ids, graph_results, all_ids

def retrieval_metapath_hybrid_reranker(query):
    _, _, all_ids = _do_metapath_hybrid(query)
    return all_ids, do_rerank(all_ids, query, top_k=10)

def retrieval_metapath_hybrid_freq(query):
    vec_ids, graph_results, all_ids = _do_metapath_hybrid(query)
    freq = Counter()
    for pid in vec_ids:
        freq[pid] += 1
    for sid, neighbors in graph_results.items():
        for pid in neighbors:
            freq[pid] += 1
    vec_rank = {pid: i for i, pid in enumerate(vec_ids)}
    gt_candidates = [pid for pid in freq if pid in ground_truth_pids]
    ranked = sorted(
        gt_candidates,
        key=lambda pid: (-freq[pid], vec_rank.get(pid, float('inf'))),
    )
    return all_ids, ranked[:10]

def retrieval_metapath_hybrid_interleave(query):
    vec_ids, graph_results, all_ids = _do_metapath_hybrid(query)
    seen = set()
    ordered = []
    for vid in vec_ids:
        if vid not in seen:
            seen.add(vid)
            ordered.append(vid)
    max_graph_len = max((len(v) for v in graph_results.values()), default=0)
    for round_idx in range(max_graph_len):
        for vid in vec_ids:
            neighbors = graph_results.get(vid, [])
            if round_idx < len(neighbors):
                gid = neighbors[round_idx]
                if gid not in seen:
                    seen.add(gid)
                    ordered.append(gid)
    gt_ordered = [pid for pid in ordered if pid in ground_truth_pids]
    return all_ids, gt_ordered[:10]

# ═══════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Running retrieval experiments...")
print("="*70)

configs = [
    # --- Hybrid: deduped pool ~300 (VEC=60, GRAPH_K=15) ---
    ("hybrid_reranker",              retrieval_hybrid_reranker,              ["MRR@5", "Recall@5"]),
    ("hybrid_freq_no_reranker",      retrieval_hybrid_freq_no_reranker,      ["MRR@5", "Recall@5"]),
    ("hybrid_interleave_no_reranker",retrieval_hybrid_interleave_no_reranker,["MRR@5", "Recall@5"]),
    # --- Hybrid: budget ~800 pre-dedup (VEC=60, GRAPH_K=12) ---
    ("b800_hybrid_reranker",         retrieval_budget800_hybrid_reranker,    ["MRR@5", "Recall@5"]),
    ("b800_hybrid_freq",             retrieval_budget800_hybrid_freq,        ["MRR@5", "Recall@5"]),
    ("b800_hybrid_interleave",       retrieval_budget800_hybrid_interleave,  ["MRR@5", "Recall@5"]),
    # --- Query-aware meta-path hybrid (VEC=60, GRAPH_K=15) ---
    ("metapath_hybrid_reranker",     retrieval_metapath_hybrid_reranker,     ["MRR@5", "Recall@5"]),
    ("metapath_hybrid_freq",         retrieval_metapath_hybrid_freq,         ["MRR@5", "Recall@5"]),
    ("metapath_hybrid_interleave",   retrieval_metapath_hybrid_interleave,   ["MRR@5", "Recall@5"]),
    # --- Vector baselines ---
    ("vector_reranker",              retrieval_vector_reranker,              ["MRR@5", "Recall@5"]),
    ("vector_only",                  retrieval_vector_only,                  ["MRR@5", "Recall@5"]),
    ("vector_poolmatch_reranker",    retrieval_vector_poolmatch_reranker,    ["MRR@5", "Recall@5"]),
    ("vector_poolmatch_only",        retrieval_vector_poolmatch_only,        ["MRR@5", "Recall@5"]),
]

all_results = {}
pool_sizes = {}  # config -> list of pool sizes per query
for config_name, config_fn, metrics in configs:
    print(f"\n  [{config_name}]")
    all_results[config_name] = {}
    pool_sizes[config_name] = []
    for qi, query in enumerate(QUERIES):
        relevant_ids = relevance_by_query.get(qi + 1, set())
        print(f"    Q{qi+1}/10: {query[:60]}...", flush=True)
        t0 = time.time()
        try:
            pool, top10 = config_fn(query)
        except Exception as e:
            print(f"    Q{qi+1} ERROR: {e}")
            pool, top10 = [], []
        elapsed = time.time() - t0
        pool_sizes[config_name].append(len(pool))
        mrr5       = mrr_at_k(top10, relevant_ids, k=5)
        rec5       = recall_at_k(top10, relevant_ids, k=5)
        rec10      = recall_at_k(top10, relevant_ids, k=10)
        n_rel_pool = len(set(pool) & relevant_ids)
        all_results[config_name][qi] = {
            "retrieved": top10, "pool_size": len(pool),
            "mrr5": mrr5, "recall5": rec5, "recall10": rec10,
            "n_relevant": len(relevant_ids), "n_relevant_in_pool": n_rel_pool,
            "elapsed": elapsed,
        }
        hits = [pid[:12] for pid in top10[:5] if pid in relevant_ids]
        print(f"    Q{qi+1}: MRR@5={mrr5:.3f}  Recall@5={rec5:.3f}  Recall@10={rec10:.3f}  "
              f"rel_in_pool={n_rel_pool}/{len(relevant_ids)}  pool={len(pool)}  ({elapsed:.1f}s) hits={hits}")
    avg_pool = np.mean(pool_sizes[config_name])
    print(f"  >> Average unique papers retrieved: {avg_pool:.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nSaving results...")

# Per-query detail
detail_rows = []
for config_name, _, metrics_list in configs:
    for qi, query in enumerate(QUERIES):
        r = all_results[config_name][qi]
        row = {
            "config": config_name, "query_id": qi + 1, "query": query,
            "n_relevant": r["n_relevant"],
            "pool_size": r["pool_size"],
            "retrieved_top10": "; ".join(r["retrieved"][:10]),
            "n_relevant_in_pool": r["n_relevant_in_pool"],
            "MRR@5": round(r["mrr5"], 4),
            "Recall@5": round(r["recall5"], 4),
            "Recall@10": round(r["recall10"], 4),
            "time_s": round(r["elapsed"], 2),
        }
        detail_rows.append(row)

with open(ROOT / "results" / "evaluation_results_detail.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
    writer.writeheader()
    writer.writerows(detail_rows)
print(f"  evaluation_results_detail.csv: {len(detail_rows)} rows")

# Summary averages
summary_rows = []
for config_name, _, metrics_list in configs:
    mrr_vals      = [all_results[config_name][qi]["mrr5"]              for qi in range(len(QUERIES))]
    rec5_vals     = [all_results[config_name][qi]["recall5"]           for qi in range(len(QUERIES))]
    rec10_vals    = [all_results[config_name][qi]["recall10"]          for qi in range(len(QUERIES))]
    rel_pool_vals = [all_results[config_name][qi]["n_relevant_in_pool"]for qi in range(len(QUERIES))]
    time_vals     = [all_results[config_name][qi]["elapsed"]           for qi in range(len(QUERIES))]
    avg_pool      = np.mean([all_results[config_name][qi]["pool_size"] for qi in range(len(QUERIES))])
    row = {
        "config": config_name,
        "avg_pool_size": round(avg_pool, 1),
        "avg_relevant_in_pool": round(np.mean(rel_pool_vals), 2),
        "avg_MRR@5": round(np.mean(mrr_vals), 4),
        "avg_Recall@5": round(np.mean(rec5_vals), 4),
        "avg_Recall@10": round(np.mean(rec10_vals), 4),
        "avg_time_s": round(np.mean(time_vals), 2),
    }
    summary_rows.append(row)

with open(ROOT / "results" / "evaluation_results_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"  evaluation_results_summary.csv: {len(summary_rows)} rows")

# Print final table
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"{'Config':<35} {'Avg Pool':>8} {'Rel in Pool':>12} {'Avg MRR@5':>12} {'Avg Recall@5':>14} {'Avg Recall@10':>15} {'Avg Time':>10}")
print("-"*106)
for row in summary_rows:
    print(f"{row['config']:<35} {row['avg_pool_size']:>8.0f} {row['avg_relevant_in_pool']:>12.1f} {row['avg_MRR@5']:>12.4f} {row['avg_Recall@5']:>14.4f} {row['avg_Recall@10']:>15.4f} {row['avg_time_s']:>9.2f}s")
print("="*94)

driver.close()
print("\nDone!")
