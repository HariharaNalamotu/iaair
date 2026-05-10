#!/usr/bin/env python3
"""
Final evaluation sweep — optimised for Ultra 9 275HX + RTX 5080 Mobile + 32 GB RAM.

Hardware utilisation:
  GPU  — FAISS index lives in VRAM; all queries batch-searched in one GPU pass
          at startup, eliminating per-call GPU latency.
          SciBERT and CrossEncoder run on CUDA; CrossEncoder batches all 10
          queries in a single inference call per (budget, VEC) point.
  CPU  — 64-thread pool for concurrent Neo4j BFS/metapath queries (I/O-bound,
          so thread count >> core count is correct). Neo4j connection pool = 100.
  RAM  — Full disk cache loaded into memory at startup so every cache read is a
          pure dict lookup; no disk I/O on subsequent runs.

Sweep:
  Budgets : 100, 125, 150, ..., 350  (step 25)
  VEC     : 1, 25, 50, ..., budget   (step 25, always include 1 and budget)
  GRAPH_K : auto-tuned per (budget, VEC, type) via binary search ±5 %
  Configs : bfs × {reranker, freq, interleave}
            metapath × {reranker, freq, interleave}
            vector   × {reranker, none}  (only at VEC = budget)

Output  : results/final_evaluation_sweep.csv  (incremental, resumable)

Run:
    conda activate torchtest
    python /mnt/c/Users/harih/hybrid-graphrag/evaluation/final_evaluation.py
"""

import csv, hashlib, json, os, pathlib, pickle, signal, sys, time, warnings
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import faiss
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import GraphDatabase

warnings.filterwarnings("ignore")
csv.field_size_limit(10 * 1024 * 1024)

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "retrieval"))
from query_aware_graph import MetaPathBestFirstGraph

load_dotenv(ROOT / ".env")

CACHE_DIR  = ROOT / "results" / "eval_cache"
SWEEP_CSV  = ROOT / "results" / "final_evaluation_sweep.csv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

# ── Hardware config ───────────────────────────────────────────────────────────
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS  = 64          # I/O-bound Neo4j threads; far exceeds core count
NEO4J_POOL = 100       # Neo4j bolt connection pool

# ── Sweep params ──────────────────────────────────────────────────────────────
BUDGETS     = list(range(100, 351, 25))
TOLERANCE   = 0.05
MAX_GK      = 200
BFS_HOPS    = 10
META_HOPS   = 4
TUNE_N      = 3        # queries sampled for GRAPH_K tuning

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
NQ = len(QUERIES)

# ── Graceful Ctrl-C ───────────────────────────────────────────────────────────
_shutdown = False
def _handle_sigint(sig, frame):
    global _shutdown
    print("\n[Interrupted] Finishing current point then saving…")
    _shutdown = True
signal.signal(signal.SIGINT, _handle_sigint)

# ═════════════════════════════════════════════════════════════════════════════
# STARTUP: load everything into memory / GPU
# ═════════════════════════════════════════════════════════════════════════════
t0 = time.time()

print(f"Device : {DEVICE}" +
      (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
print(f"Workers: {WORKERS}  Neo4j pool: {NEO4J_POOL}")

# ── Models ────────────────────────────────────────────────────────────────────
print("Loading SciBERT + CrossEncoder on GPU…")
embed_model    = SentenceTransformer(
    "jordyvl/scibert_scivocab_uncased_sentence_transformer", device=DEVICE
)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)

# ── FAISS — move to GPU VRAM ──────────────────────────────────────────────────
print("Loading FAISS index → GPU…")
_raw_vecs    = np.load(str(ROOT / "data" / "vectors.npy"))
_vector_pids = json.load(open(ROOT / "data" / "vector_paperids.json"))
faiss.normalize_L2(_raw_vecs)
_faiss_cpu = faiss.IndexFlatIP(_raw_vecs.shape[1])
_faiss_cpu.add(_raw_vecs)
if DEVICE == "cuda":
    _gpu_res  = faiss.StandardGpuResources()
    _faiss_idx = faiss.index_cpu_to_gpu(_gpu_res, 0, _faiss_cpu)
else:
    _faiss_idx = _faiss_cpu
print(f"  {_faiss_idx.ntotal} vectors in {'GPU VRAM' if DEVICE=='cuda' else 'RAM'}")

# ── Neo4j ─────────────────────────────────────────────────────────────────────
print("Connecting to Neo4j…")
driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS),
    max_connection_pool_size=NEO4J_POOL,
)

# ── Paper data ────────────────────────────────────────────────────────────────
print("Loading paper data…")
all_papers: dict[str, dict] = {}
with open(ROOT / "data" / "papers.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row["id"]
        if pid not in all_papers:
            all_papers[pid] = {
                "title":    row.get("title", ""),
                "abstract": row.get("abstract", ""),
            }

_emb_matrix = np.load(str(ROOT / "data" / "paper_embeddings.npy"))
_emb_ids    = json.load(open(ROOT / "data" / "paper_ids.json"))
paper_embeddings = {pid: _emb_matrix[i] for i, pid in enumerate(_emb_ids)}
metapath_graph   = MetaPathBestFirstGraph(driver, embed_model, paper_embeddings)

# ── Ground truth ──────────────────────────────────────────────────────────────
print("Loading ground truth…")
relevance_by_query: dict[int, set] = {}
ground_truth_pids:  set[str]       = set()
with open(ROOT / "data" / "ground_truth_relevance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qi = int(row["query_id"])
        ground_truth_pids.add(row["paperId"])
        relevance_by_query.setdefault(qi, set())
        if int(row["relevant"]) == 1:
            relevance_by_query[qi].add(row["paperId"])

# ─────────────────────────────────────────────────────────────────────────────
# In-memory cache (write-through; load entire disk cache at startup)
# ─────────────────────────────────────────────────────────────────────────────
_mem: dict[str, object] = {}

print("Loading disk cache into RAM…")
n_loaded = 0
for p in CACHE_DIR.glob("*.pkl"):
    try:
        _mem[p.stem] = pickle.load(open(p, "rb"))
        n_loaded += 1
    except Exception:
        pass
print(f"  {n_loaded} entries loaded ({n_loaded*1e-3:.1f}k)")

def _ckey(*args) -> str:
    return hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()

def _cget(ns: str, *args):
    k = f"{ns}_{_ckey(*args)}"
    v = _mem.get(k)
    return (True, v) if v is not None else (False, None)

def _cput(ns: str, val, *args):
    k = f"{ns}_{_ckey(*args)}"
    _mem[k] = val
    with open(CACHE_DIR / f"{k}.pkl", "wb") as f:
        pickle.dump(val, f)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute ALL vector searches in one GPU batch pass
# ─────────────────────────────────────────────────────────────────────────────
def vec_values(budget: int) -> list[int]:
    return sorted(set([1] + list(range(25, budget, 25)) + [budget]))

ALL_VEC_NS = sorted(set(vn for b in BUDGETS for vn in vec_values(b)))

print(f"Pre-computing {NQ} × {len(ALL_VEC_NS)} vector searches (single GPU pass)…")
_qvecs = embed_model.encode(QUERIES, batch_size=NQ, normalize_embeddings=False,
                             show_progress_bar=False).astype(np.float32)
faiss.normalize_L2(_qvecs)
_, _all_idx = _faiss_idx.search(_qvecs, _faiss_idx.ntotal)   # (NQ, ntotal)

# Build sorted PID list per query, then slice to each needed vec_n
_VEC_CACHE: dict[tuple, list[str]] = {}
for qi, query in enumerate(QUERIES):
    sorted_pids: list[str] = []
    seen: set[str] = set()
    for raw_i in _all_idx[qi]:
        if raw_i < 0:
            continue
        pid = _vector_pids[raw_i]
        if pid not in seen:
            seen.add(pid)
            sorted_pids.append(pid)
    for vn in ALL_VEC_NS:
        result = sorted_pids[:vn]
        _VEC_CACHE[(query, vn)] = result
        _cput("vs", result, query, vn)   # persist to disk cache too

print(f"  Done. Startup took {time.time()-t0:.1f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval functions
# ─────────────────────────────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=WORKERS)

def vec_search(query: str, n: int) -> list[str]:
    return _VEC_CACHE.get((query, n)) or _cget("vs", query, n)[1] or []

def _bfs_seed(sid: str, k: int, hops: int) -> list[str]:
    hit, v = _cget("bfs", sid, k, hops)
    if hit:
        return v
    channels = {
        "cites":    [("Paper", "CITES")],
        "coauthor": [("Paper", "WROTE"), ("Author", "WROTE")],
        "venue":    [("Paper", "PUBLISHED_IN"), ("Venue", "PUBLISHED_IN")],
        "field":    [("Paper", "HAS_FIELD"), ("FieldOfStudy", "HAS_FIELD")],
    }
    ch_res: dict[str, list] = {}
    with driver.session() as session:
        for ch, plan in channels.items():
            queue   = deque([("Paper", sid, 0)])
            visited = {("Paper", sid)}
            found: list[str] = []
            while queue and len(found) < k:
                label, nid, depth = queue.popleft()
                if depth >= hops:
                    continue
                if label == "Paper" and nid != sid:
                    found.append(nid)
                for src, rel in plan:
                    if label != src:
                        continue
                    q = (f"MATCH (n:{label} {{id:$id}})-[:{rel}]-(m) "
                         "RETURN labels(m)[0] AS l, m.id AS id")
                    for r in session.run(q, id=nid):
                        key = (r["l"], r["id"])
                        if key not in visited:
                            visited.add(key)
                            queue.append((r["l"], r["id"], depth + 1))
            ch_res[ch] = found
    result, seen = [], set()
    max_len = max((len(v) for v in ch_res.values()), default=0)
    for i in range(max_len):
        for ch in channels:
            lst = ch_res[ch]
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i])
                result.append(lst[i])
                if len(result) == k:
                    break
        if len(result) == k:
            break
    _cput("bfs", result, sid, k, hops)
    return result

def bfs_search(vec_ids: list[str], k: int) -> dict[str, list[str]]:
    if k == 0:
        return {sid: [] for sid in vec_ids}
    futs = {_executor.submit(_bfs_seed, sid, k, BFS_HOPS): sid for sid in vec_ids}
    return {futs[f]: f.result() for f in as_completed(futs)}

def _meta_seed(query: str, sid: str, k: int) -> list[str]:
    hit, v = _cget("meta", query, sid, k, META_HOPS)
    if hit:
        return v
    try:
        neighbors = metapath_graph.retrieve(query, sid, k=k, max_hops=META_HOPS)
        result = [n["paper_id"] for n in neighbors]
    except Exception:
        result = []
    _cput("meta", result, query, sid, k, META_HOPS)
    return result

def meta_search(vec_ids: list[str], query: str, k: int) -> dict[str, list[str]]:
    if k == 0:
        return {sid: [] for sid in vec_ids}
    futs = {_executor.submit(_meta_seed, query, sid, k): sid for sid in vec_ids}
    return {futs[f]: f.result() for f in as_completed(futs)}

def unique_pool(vec_ids: list[str], graph: dict[str, list[str]]) -> list[str]:
    pool, seen = list(vec_ids), set(vec_ids)
    for sid in vec_ids:
        for pid in graph.get(sid, []):
            if pid not in seen:
                seen.add(pid)
                pool.append(pid)
    return pool

# ─────────────────────────────────────────────────────────────────────────────
# Batch CrossEncoder — score all NQ queries in one GPU inference call
# ─────────────────────────────────────────────────────────────────────────────
def rerank_all(cands_by_qi: list[list[str]]) -> list[list[str]]:
    """
    cands_by_qi[qi] = ground-truth paper IDs in the pool for query qi.
    Returns top-10 ranked IDs for each query in a single GPU pass.
    """
    all_pairs: list[tuple] = []
    pair_meta: list[tuple] = []   # (qi, pid)
    for qi, (query, cands) in enumerate(zip(QUERIES, cands_by_qi)):
        for pid in cands:
            p = all_papers.get(pid)
            if p:
                all_pairs.append((query, (p["title"] + " " + p["abstract"]).strip()))
                pair_meta.append((qi, pid))

    if not all_pairs:
        return [[] for _ in QUERIES]

    scores = reranker_model.predict(all_pairs, batch_size=512, show_progress_bar=False)

    qi_scored: dict[int, list] = {i: [] for i in range(NQ)}
    for (qi, pid), sc in zip(pair_meta, scores):
        qi_scored[qi].append((float(sc), pid))

    return [[pid for _, pid in sorted(qi_scored[qi], reverse=True)[:10]]
            for qi in range(NQ)]

# ─────────────────────────────────────────────────────────────────────────────
# Rankers
# ─────────────────────────────────────────────────────────────────────────────
def rank_freq(pool, vec_ids, graph, query):
    freq = Counter(vec_ids)
    for nbrs in graph.values():
        for pid in nbrs:
            freq[pid] += 1
    vr = {p: i for i, p in enumerate(vec_ids)}
    gt = [p for p in freq if p in ground_truth_pids]
    return sorted(gt, key=lambda p: (-freq[p], vr.get(p, 9e9)))[:10]

def rank_interleave(pool, vec_ids, graph, query):
    seen, ordered = set(), []
    for v in vec_ids:
        if v not in seen:
            seen.add(v); ordered.append(v)
    mx = max((len(g) for g in graph.values()), default=0)
    for i in range(mx):
        for v in vec_ids:
            nbrs = graph.get(v, [])
            if i < len(nbrs) and nbrs[i] not in seen:
                seen.add(nbrs[i]); ordered.append(nbrs[i])
    return [p for p in ordered if p in ground_truth_pids][:10]

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def mrr5(ranked, relevant):
    for i, p in enumerate(ranked[:5]):
        if p in relevant:
            return 1.0 / (i + 1)
    return 0.0

def recall(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH_K auto-tuner
# ─────────────────────────────────────────────────────────────────────────────
def measure_pool(vec_n: int, gk: int, rtype: str) -> float:
    sizes = []
    for query in QUERIES[:TUNE_N]:
        vids  = vec_search(query, vec_n)
        graph = (bfs_search(vids, gk) if rtype == "bfs"
                 else meta_search(vids, query, gk))
        sizes.append(len(unique_pool(vids, graph)))
    return float(np.mean(sizes))

def autotune_gk(vec_n: int, budget: int, rtype: str) -> tuple[int, float]:
    lo, hi         = 0, MAX_GK
    history: list  = []
    last_above      = MAX_GK

    while lo <= hi:
        k = (lo + hi) // 2
        if k in history:
            actual = measure_pool(vec_n, last_above, rtype)
            return last_above, actual
        history.append(k)
        actual = measure_pool(vec_n, k, rtype)
        err    = (actual - budget) / budget
        if abs(err) <= TOLERANCE:
            return k, actual
        elif actual < budget:
            lo = k + 1
        else:
            last_above = min(last_above, k)
            hi = k - 1

    actual = measure_pool(vec_n, last_above, rtype)
    return last_above, actual

# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
FIELDS = [
    "budget", "vec_n", "graph_k", "retrieval_type", "ranker",
    "actual_avg_pool", "query_id",
    "mrr5", "recall5", "recall10", "n_relevant_in_pool", "pool_size",
]

def load_done() -> set:
    done: set = set()
    if not SWEEP_CSV.exists():
        return done
    with open(SWEEP_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((int(row["budget"]), int(row["vec_n"]),
                      row["retrieval_type"], row["ranker"]))
    return done

# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────
def make_rows(budget, vec_n, gk, rtype, avg_pool,
              all_vids, all_graphs, all_pools,
              reranked_by_qi, freq_by_qi, interleave_by_qi,
              done) -> list[dict]:
    rows = []
    for ranker_name, top10_by_qi in [
        ("reranker",   reranked_by_qi),
        ("freq",       freq_by_qi),
        ("interleave", interleave_by_qi),
    ]:
        if (budget, vec_n, rtype, ranker_name) in done:
            continue
        for qi in range(NQ):
            relevant = relevance_by_query.get(qi + 1, set())
            top10    = top10_by_qi[qi]
            pool     = all_pools[qi]
            rows.append({
                "budget":           budget,
                "vec_n":            vec_n,
                "graph_k":          gk,
                "retrieval_type":   rtype,
                "ranker":           ranker_name,
                "actual_avg_pool":  round(avg_pool, 1),
                "query_id":         qi + 1,
                "mrr5":             round(mrr5(top10, relevant), 4),
                "recall5":          round(recall(top10, relevant, 5), 4),
                "recall10":         round(recall(top10, relevant, 10), 4),
                "n_relevant_in_pool": len(set(pool) & relevant),
                "pool_size":        len(pool),
            })
    return rows

def main():
    done = load_done()
    exists = SWEEP_CSV.exists()
    csv_f  = open(SWEEP_CSV, "a", newline="", encoding="utf-8")
    csv_w  = csv.DictWriter(csv_f, fieldnames=FIELDS)
    if not exists:
        csv_w.writeheader()

    total_points = sum(len(vec_values(b)) for b in BUDGETS)
    done_points  = 0

    try:
        for budget in BUDGETS:
            if _shutdown: break
            print(f"\n{'='*64}\nBudget = {budget}\n{'='*64}")

            for vec_n in vec_values(budget):
                if _shutdown: break
                done_points += 1
                pct = 100 * done_points / total_points

                # ── Pure vector ───────────────────────────────────────────
                if vec_n == budget:
                    skip_r = (budget, vec_n, "vector", "reranker") in done
                    skip_n = (budget, vec_n, "vector", "none")     in done
                    if skip_r and skip_n:
                        print(f"  [{pct:.0f}%] SKIP vector vec={vec_n}")
                    else:
                        print(f"  [{pct:.0f}%] vector vec={vec_n}", flush=True)
                        all_vids = [vec_search(q, vec_n) for q in QUERIES]
                        # Batch rerank all queries
                        gt_cands = [[p for p in vids if p in ground_truth_pids]
                                    for vids in all_vids]
                        reranked = rerank_all(gt_cands)
                        rows = []
                        for qi in range(NQ):
                            pool     = all_vids[qi]
                            relevant = relevance_by_query.get(qi + 1, set())
                            for ranker_name, top10 in [
                                ("reranker", reranked[qi]),
                                ("none", [p for p in pool if p in ground_truth_pids][:10]),
                            ]:
                                if (budget, vec_n, "vector", ranker_name) in done:
                                    continue
                                rows.append({
                                    "budget": budget, "vec_n": vec_n, "graph_k": 0,
                                    "retrieval_type": "vector", "ranker": ranker_name,
                                    "actual_avg_pool": float(vec_n),
                                    "query_id": qi + 1,
                                    "mrr5":    round(mrr5(top10, relevant), 4),
                                    "recall5": round(recall(top10, relevant, 5), 4),
                                    "recall10":round(recall(top10, relevant, 10), 4),
                                    "n_relevant_in_pool": len(set(pool) & relevant),
                                    "pool_size": len(pool),
                                })
                        csv_w.writerows(rows); csv_f.flush()

                # ── BFS hybrid ────────────────────────────────────────────
                all_bfs_done = all(
                    (budget, vec_n, "bfs", r) in done
                    for r in ("reranker", "freq", "interleave")
                )
                if all_bfs_done:
                    print(f"  [{pct:.0f}%] SKIP bfs vec={vec_n}")
                else:
                    gk_bfs, avg_bfs = autotune_gk(vec_n, budget, "bfs")
                    print(f"  [{pct:.0f}%] bfs  vec={vec_n} gk={gk_bfs} "
                          f"pool≈{avg_bfs:.0f}", flush=True)

                    # Retrieve for all queries
                    all_vids   = [vec_search(q, vec_n) for q in QUERIES]
                    all_graphs = [bfs_search(vids, gk_bfs) for vids in all_vids]
                    all_pools  = [unique_pool(v, g) for v, g in zip(all_vids, all_graphs)]

                    # Batch rerank (single GPU call)
                    gt_cands    = [[p for p in pool if p in ground_truth_pids]
                                   for pool in all_pools]
                    reranked    = rerank_all(gt_cands)
                    freq_top10  = [rank_freq(all_pools[qi], all_vids[qi],
                                             all_graphs[qi], QUERIES[qi])
                                   for qi in range(NQ)]
                    inter_top10 = [rank_interleave(all_pools[qi], all_vids[qi],
                                                    all_graphs[qi], QUERIES[qi])
                                   for qi in range(NQ)]

                    rows = make_rows(budget, vec_n, gk_bfs, "bfs", avg_bfs,
                                     all_vids, all_graphs, all_pools,
                                     reranked, freq_top10, inter_top10, done)
                    csv_w.writerows(rows); csv_f.flush()

                # ── Metapath hybrid ───────────────────────────────────────
                all_meta_done = all(
                    (budget, vec_n, "metapath", r) in done
                    for r in ("reranker", "freq", "interleave")
                )
                if all_meta_done:
                    print(f"  [{pct:.0f}%] SKIP metapath vec={vec_n}")
                else:
                    gk_meta, avg_meta = autotune_gk(vec_n, budget, "metapath")
                    print(f"  [{pct:.0f}%] meta vec={vec_n} gk={gk_meta} "
                          f"pool≈{avg_meta:.0f}", flush=True)

                    all_vids   = [vec_search(q, vec_n) for q in QUERIES]
                    all_graphs = [meta_search(vids, QUERIES[qi], gk_meta)
                                  for qi, vids in enumerate(all_vids)]
                    all_pools  = [unique_pool(v, g) for v, g in zip(all_vids, all_graphs)]

                    gt_cands    = [[p for p in pool if p in ground_truth_pids]
                                   for pool in all_pools]
                    reranked    = rerank_all(gt_cands)
                    freq_top10  = [rank_freq(all_pools[qi], all_vids[qi],
                                             all_graphs[qi], QUERIES[qi])
                                   for qi in range(NQ)]
                    inter_top10 = [rank_interleave(all_pools[qi], all_vids[qi],
                                                    all_graphs[qi], QUERIES[qi])
                                   for qi in range(NQ)]

                    rows = make_rows(budget, vec_n, gk_meta, "metapath", avg_meta,
                                     all_vids, all_graphs, all_pools,
                                     reranked, freq_top10, inter_top10, done)
                    csv_w.writerows(rows); csv_f.flush()

    finally:
        csv_f.close()
        _executor.shutdown(wait=False)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"Results: {SWEEP_CSV}")

if __name__ == "__main__":
    main()
