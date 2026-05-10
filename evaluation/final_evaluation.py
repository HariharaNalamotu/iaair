#!/usr/bin/env python3
"""
Final evaluation sweep: all configs × 11 pool-size budgets × VEC sweep.

Budgets : 100, 125, 150, ..., 350  (step 25)
VEC     : 1, 25, 50, ..., budget   (step 25, always include 1 and budget)
GRAPH_K : auto-tuned per (budget, VEC, retrieval_type) via binary search
          so the average deduped pool lands within 5 % of the target budget.
          If binary search bounces (can't converge), GRAPH_K is set to the
          smallest value that puts the pool just above the 5 % threshold.

Configs at each (budget, VEC, GRAPH_K):
    bfs       × reranker / freq / interleave
    metapath  × reranker / freq / interleave
    vector    × reranker / none               (VEC = budget, GRAPH_K = 0)

Results saved incrementally to results/final_evaluation_sweep.csv.

Run:
    conda activate torchtest
    python /mnt/c/Users/harih/hybrid-graphrag/evaluation/final_evaluation.py
"""

import csv, hashlib, json, os, pathlib, pickle, sys, time, warnings
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

CACHE_DIR   = ROOT / "results" / "eval_cache"
SWEEP_CSV   = ROOT / "results" / "final_evaluation_sweep.csv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NEO4J_URI   = os.environ["NEO4J_URI"]
NEO4J_USER  = os.environ["NEO4J_USER"]
NEO4J_PASS  = os.environ["NEO4J_PASS"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS     = min(32, os.cpu_count() or 8)

# ── Sweep parameters ──────────────────────────────────────────────────────────
BUDGETS      = list(range(100, 351, 25))          # [100, 125, ..., 350]
TUNE_QUERIES = 3      # queries sampled for GRAPH_K tuning (first N)
TOLERANCE    = 0.05   # ± 5 % of target budget
MAX_GK       = 200    # hard cap on GRAPH_K search
BFS_HOPS     = 10
META_HOPS    = 4

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

# ── Startup ───────────────────────────────────────────────────────────────────
print("Loading models...")
embed_model = SentenceTransformer(
    "jordyvl/scibert_scivocab_uncased_sentence_transformer", device=DEVICE
)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)

print("Loading FAISS index...")
_raw_vecs    = np.load(str(ROOT / "data" / "vectors.npy"))
_vector_pids = json.load(open(ROOT / "data" / "vector_paperids.json"))
faiss.normalize_L2(_raw_vecs)
_faiss_index = faiss.IndexFlatIP(_raw_vecs.shape[1])
_faiss_index.add(_raw_vecs)
print(f"  {_faiss_index.ntotal} vectors")

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

print("Loading paper data...")
all_papers: dict[str, dict] = {}
with open(ROOT / "data" / "papers.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row["id"]
        if pid not in all_papers:
            all_papers[pid] = {
                "title":    row.get("title", ""),
                "abstract": row.get("abstract", ""),
            }

print("Loading paper embeddings...")
_emb_matrix = np.load(str(ROOT / "data" / "paper_embeddings.npy"))
_emb_ids    = json.load(open(ROOT / "data" / "paper_ids.json"))
paper_embeddings = {pid: _emb_matrix[i] for i, pid in enumerate(_emb_ids)}
metapath_graph   = MetaPathBestFirstGraph(driver, embed_model, paper_embeddings)

print("Loading ground truth...")
relevance_by_query: dict[int, set] = {}
ground_truth_pids:  set[str]       = set()
with open(ROOT / "data" / "ground_truth_relevance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qi = int(row["query_id"])
        ground_truth_pids.add(row["paperId"])
        relevance_by_query.setdefault(qi, set())
        if int(row["relevant"]) == 1:
            relevance_by_query[qi].add(row["paperId"])

print(f"Ready — {len(all_papers)} papers, {len(ground_truth_pids)} ground-truth papers\n")

# ── Cache helpers ─────────────────────────────────────────────────────────────
def _ckey(*args) -> str:
    return hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()

def _cget(ns, *args):
    p = CACHE_DIR / f"{ns}_{_ckey(*args)}.pkl"
    return (True, pickle.load(open(p, "rb"))) if p.exists() else (False, None)

def _cput(ns, val, *args):
    with open(CACHE_DIR / f"{ns}_{_ckey(*args)}.pkl", "wb") as f:
        pickle.dump(val, f)

# ── Core retrieval ────────────────────────────────────────────────────────────
def vec_search(query: str, n: int) -> list[str]:
    hit, v = _cget("vs", query, n)
    if hit:
        return v
    qv = embed_model.encode([query])[0].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(qv)
    _, idx = _faiss_index.search(qv, _faiss_index.ntotal)
    result, seen = [], set()
    for i in idx[0]:
        if i < 0: continue
        pid = _vector_pids[i]
        if pid not in seen:
            seen.add(pid); result.append(pid)
            if len(result) == n: break
    _cput("vs", result, query, n)
    return result

def _bfs_seed(sid: str, k: int, hops: int) -> list[str]:
    hit, v = _cget("bfs", sid, k, hops)
    if hit: return v
    channels = {
        "cites":    [("Paper", "CITES")],
        "coauthor": [("Paper", "WROTE"), ("Author", "WROTE")],
        "venue":    [("Paper", "PUBLISHED_IN"), ("Venue", "PUBLISHED_IN")],
        "field":    [("Paper", "HAS_FIELD"), ("FieldOfStudy", "HAS_FIELD")],
    }
    ch_res = {}
    with driver.session() as session:
        for ch, plan in channels.items():
            queue   = deque([("Paper", sid, 0)])
            visited = {("Paper", sid)}
            found   = []
            while queue and len(found) < k:
                label, nid, depth = queue.popleft()
                if depth >= hops: continue
                if label == "Paper" and nid != sid:
                    found.append(nid)
                for src, rel in plan:
                    if label != src: continue
                    q = f"MATCH (n:{label} {{id:$id}})-[:{rel}]-(m) RETURN labels(m)[0] AS l, m.id AS id"
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
                seen.add(lst[i]); result.append(lst[i])
                if len(result) == k: break
        if len(result) == k: break
    _cput("bfs", result, sid, k, hops)
    return result

def bfs_search(vec_ids: list[str], k: int, hops: int = BFS_HOPS) -> dict[str, list]:
    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_bfs_seed, sid, k, hops): sid for sid in vec_ids}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return results

def _meta_seed(query: str, sid: str, k: int, hops: int) -> list[str]:
    hit, v = _cget("meta", query, sid, k, hops)
    if hit: return v
    try:
        neighbors = metapath_graph.retrieve(query, sid, k=k, max_hops=hops)
        result = [n["paper_id"] for n in neighbors]
    except Exception:
        result = []
    _cput("meta", result, query, sid, k, hops)
    return result

def meta_search(vec_ids: list[str], query: str, k: int,
                hops: int = META_HOPS) -> dict[str, list]:
    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_meta_seed, query, sid, k, hops): sid for sid in vec_ids}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    return results

def unique_pool(vec_ids: list[str], graph: dict[str, list]) -> list[str]:
    pool, seen = list(vec_ids), set(vec_ids)
    for sid in vec_ids:
        for pid in graph.get(sid, []):
            if pid not in seen:
                seen.add(pid); pool.append(pid)
    return pool

def do_rerank(pool: list[str], query: str, top_k: int = 10) -> list[str]:
    cands = [p for p in pool if p in ground_truth_pids]
    if not cands: return []
    hit, v = _cget("rr", query, tuple(sorted(cands)), top_k)
    if hit: return v
    pairs = [(query, (all_papers[p]["title"] + " " + all_papers[p]["abstract"]).strip())
             for p in cands if p in all_papers]
    valid = [p for p in cands if p in all_papers]
    if not pairs: return []
    scores = reranker_model.predict(pairs)
    result = [valid[i] for i in np.argsort(scores)[::-1]][:top_k]
    _cput("rr", result, query, tuple(sorted(cands)), top_k)
    return result

# ── Rankers ───────────────────────────────────────────────────────────────────
def rank_reranker(pool, vec_ids, graph, query):
    return do_rerank(pool, query)

def rank_freq(pool, vec_ids, graph, query):
    freq = Counter(vec_ids)
    for sid, neighbors in graph.items():
        for pid in neighbors:
            freq[pid] += 1
    vr = {p: i for i, p in enumerate(vec_ids)}
    gt = [p for p in freq if p in ground_truth_pids]
    return sorted(gt, key=lambda p: (-freq[p], vr.get(p, 9e9)))[:10]

def rank_interleave(pool, vec_ids, graph, query):
    seen, ordered = set(), []
    for v in vec_ids:
        if v not in seen: seen.add(v); ordered.append(v)
    mx = max((len(g) for g in graph.values()), default=0)
    for i in range(mx):
        for v in vec_ids:
            nbrs = graph.get(v, [])
            if i < len(nbrs) and nbrs[i] not in seen:
                seen.add(nbrs[i]); ordered.append(nbrs[i])
    return [p for p in ordered if p in ground_truth_pids][:10]

RANKERS = {
    "reranker":   rank_reranker,
    "freq":       rank_freq,
    "interleave": rank_interleave,
}

# ── Metrics ───────────────────────────────────────────────────────────────────
def mrr5(ranked, relevant):
    for i, p in enumerate(ranked[:5]):
        if p in relevant: return 1.0 / (i + 1)
    return 0.0

def recall(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0

# ── Pool-size measurement ─────────────────────────────────────────────────────
def measure_pool(vec_n: int, gk: int, rtype: str) -> float:
    """Average deduped pool size over the first TUNE_QUERIES queries."""
    sizes = []
    for query in QUERIES[:TUNE_QUERIES]:
        vids = vec_search(query, vec_n)
        graph = (bfs_search(vids, gk) if rtype == "bfs"
                 else meta_search(vids, query, gk))
        sizes.append(len(unique_pool(vids, graph)))
    return float(np.mean(sizes))

# ── GRAPH_K auto-tuner ────────────────────────────────────────────────────────
def autotune_gk(vec_n: int, budget: int, rtype: str) -> tuple[int, float]:
    """
    Binary-search for GRAPH_K so measured pool ≈ budget ± TOLERANCE.
    If search bounces, use the smallest GK whose pool is ≥ budget.
    Returns (graph_k, actual_avg_pool).
    """
    lo, hi = 0, MAX_GK
    history: list[int] = []
    last_above = MAX_GK   # smallest GK seen that overshoots (fallback)

    while lo <= hi:
        k = (lo + hi) // 2
        if k in history:
            # Bouncing — set to last_above (pool slightly over budget)
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

    # Binary search exhausted without convergence
    actual = measure_pool(vec_n, last_above, rtype)
    return last_above, actual

# ── CSV writer ────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "budget", "vec_n", "graph_k", "retrieval_type", "ranker",
    "actual_avg_pool", "query_id",
    "mrr5", "recall5", "recall10", "n_relevant_in_pool", "pool_size",
]

def open_csv():
    exists = SWEEP_CSV.exists()
    f = open(SWEEP_CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not exists:
        w.writeheader()
    return f, w

# ── Already-done check ────────────────────────────────────────────────────────
def load_done() -> set[tuple]:
    """Return set of (budget, vec_n, retrieval_type, ranker) already in CSV."""
    done = set()
    if not SWEEP_CSV.exists():
        return done
    with open(SWEEP_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((int(row["budget"]), int(row["vec_n"]),
                      row["retrieval_type"], row["ranker"]))
    return done

# ── VEC sweep points ──────────────────────────────────────────────────────────
def vec_values(budget: int) -> list[int]:
    """[1, 25, 50, ..., budget]  — always include 1 and budget."""
    return sorted(set([1] + list(range(25, budget, 25)) + [budget]))

# ── Main sweep ────────────────────────────────────────────────────────────────
def run_config(vec_ids, graph, pool, query, rtype, ranker_name, ranker_fn,
               budget, vec_n, gk, avg_pool, qi):
    relevant = relevance_by_query.get(qi + 1, set())
    top10    = ranker_fn(pool, vec_ids, graph, query)
    return {
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
    }

def main():
    done = load_done()
    csv_f, csv_w = open_csv()

    try:
        for budget in BUDGETS:
            print(f"\n{'='*64}")
            print(f"Budget = {budget}")
            print(f"{'='*64}")

            for vec_n in vec_values(budget):

                # ── Pure vector (vec_n == budget, no graph) ───────────────
                if vec_n == budget:
                    for ranker_name in ("reranker", "none"):
                        if (budget, vec_n, "vector", ranker_name) in done:
                            print(f"  [SKIP] vector/{ranker_name} vec={vec_n}")
                            continue
                        print(f"  vector/{ranker_name}  vec={vec_n}", flush=True)
                        avg_pool = float(vec_n)
                        rows = []
                        for qi, query in enumerate(QUERIES):
                            vids  = vec_search(query, vec_n)
                            pool  = vids
                            graph = {}
                            if ranker_name == "reranker":
                                top10 = do_rerank(pool, query)
                            else:
                                top10 = [p for p in pool if p in ground_truth_pids][:10]
                            relevant = relevance_by_query.get(qi + 1, set())
                            rows.append({
                                "budget": budget, "vec_n": vec_n, "graph_k": 0,
                                "retrieval_type": "vector", "ranker": ranker_name,
                                "actual_avg_pool": avg_pool,
                                "query_id": qi + 1,
                                "mrr5":    round(mrr5(top10, relevant), 4),
                                "recall5": round(recall(top10, relevant, 5), 4),
                                "recall10":round(recall(top10, relevant, 10), 4),
                                "n_relevant_in_pool": len(set(pool) & relevant),
                                "pool_size": len(pool),
                            })
                        csv_w.writerows(rows)
                        csv_f.flush()

                # ── BFS hybrid ────────────────────────────────────────────
                if (budget, vec_n, "bfs", "reranker") not in done:
                    gk_bfs, avg_bfs = autotune_gk(vec_n, budget, "bfs")
                    print(f"  bfs  vec={vec_n}  gk={gk_bfs}  avg_pool={avg_bfs:.1f}",
                          flush=True)
                    for qi, query in enumerate(QUERIES):
                        vids  = vec_search(query, vec_n)
                        graph = bfs_search(vids, gk_bfs)
                        pool  = unique_pool(vids, graph)
                        for rname, rfn in RANKERS.items():
                            if (budget, vec_n, "bfs", rname) in done:
                                continue
                            csv_w.writerow(run_config(
                                vids, graph, pool, query,
                                "bfs", rname, rfn,
                                budget, vec_n, gk_bfs, avg_bfs, qi))
                    csv_f.flush()
                else:
                    print(f"  [SKIP] bfs vec={vec_n}")

                # ── Metapath hybrid ───────────────────────────────────────
                if (budget, vec_n, "metapath", "reranker") not in done:
                    gk_meta, avg_meta = autotune_gk(vec_n, budget, "metapath")
                    print(f"  meta vec={vec_n}  gk={gk_meta}  avg_pool={avg_meta:.1f}",
                          flush=True)
                    for qi, query in enumerate(QUERIES):
                        vids  = vec_search(query, vec_n)
                        graph = meta_search(vids, query, gk_meta)
                        pool  = unique_pool(vids, graph)
                        for rname, rfn in RANKERS.items():
                            if (budget, vec_n, "metapath", rname) in done:
                                continue
                            csv_w.writerow(run_config(
                                vids, graph, pool, query,
                                "metapath", rname, rfn,
                                budget, vec_n, gk_meta, avg_meta, qi))
                    csv_f.flush()
                else:
                    print(f"  [SKIP] metapath vec={vec_n}")

    finally:
        csv_f.close()

    print(f"\nDone. Results saved to:\n  {SWEEP_CSV}")

if __name__ == "__main__":
    main()
