#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep HYBRID_VEC from 5 to 250, dynamically calibrating GRAPH_K so avg pool ~ 500."""
import csv, json, os, time, sys, functools, pathlib
csv.field_size_limit(10 * 1024 * 1024)
from collections import deque
from functools import lru_cache
from dotenv import load_dotenv

import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import GraphDatabase

print = functools.partial(print, flush=True)

ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

MILVUS_DB  = str(ROOT / "RAG.db")
COLLECTION = "ingestion_v0"
NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

TARGET_POOL = 500
TOLERANCE   = 30   # acceptable deviation from target

print("=" * 80)
print("HYBRID RERANKER SWEEP - Dynamic GRAPH_K Calibration")
print("=" * 80)

print("\n[INIT] Loading SciBERT embedding model...")
embed_model = SentenceTransformer("jordyvl/scibert_scivocab_uncased_sentence_transformer")
print("[INIT] SciBERT loaded.")

print("[INIT] Loading CrossEncoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("[INIT] CrossEncoder loaded.")

def create_embedding(texts):
    return embed_model.encode(texts)

print("[INIT] Connecting to Milvus...")
client = MilvusClient(MILVUS_DB)
print("[INIT] Milvus connected.")

print("[INIT] Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
print("[INIT] Neo4j connected.")

class PaperBFS:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    def _get_neighbors(self, session, label, node_id, rel_type):
        query = f"MATCH (n:{label} {{id: $id}})-[:{rel_type}]-(m) RETURN labels(m)[0] AS label, m.id AS id"
        return list(session.run(query, id=node_id))
    @lru_cache(maxsize=10_000)
    def bfs_nearest_papers(self, start_paper_id, k, max_hops=4):
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

# Load data
print("\n[DATA] Loading papers.csv...")
all_papers = {}
with open("papers.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row["id"]
        if pid not in all_papers:
            all_papers[pid] = {"paperId": pid, "title": row.get("title",""), "abstract": row.get("abstract","")}
print(f"[DATA] Loaded {len(all_papers)} papers.")

print("[DATA] Loading ground_truth_relevance.csv...")
relevance_by_query = {}
ground_truth_pids = set()
with open("ground_truth_relevance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qi = int(row["query_id"])
        if qi not in relevance_by_query:
            relevance_by_query[qi] = set()
        ground_truth_pids.add(row["paperId"])
        if int(row["relevant"]) == 1:
            relevance_by_query[qi].add(row["paperId"])
print(f"[DATA] {len(ground_truth_pids)} ground truth papers across {len(relevance_by_query)} queries.")
for qi in sorted(relevance_by_query):
    print(f"  Q{qi}: {len(relevance_by_query[qi])} relevant")

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

def mrr_at_k(ranked_ids, relevant_ids, k=5):
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(ranked_ids, relevant_ids, k=5):
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)

def do_vector_search(query, n_papers=5):
    BATCH = 500
    query_vec = create_embedding([query])[0]
    seen, result, offset = set(), [], 0
    print(f"      [VECTOR] Searching for {n_papers} papers...")
    while len(result) < n_papers:
        limit = min(BATCH, 16000 - offset)
        if limit <= 0:
            break
        search_res = client.search(collection_name=COLLECTION, data=[query_vec],
                                   limit=limit, offset=offset, output_fields=["paperId"])
        hits = search_res[0]
        if not hits:
            break
        for hit in hits:
            pid = hit["entity"]["paperId"]
            if pid not in seen:
                seen.add(pid)
                result.append(pid)
                if len(result) == n_papers:
                    break
        offset += limit
    print(f"      [VECTOR] Found {len(result)} unique papers.")
    return result

def do_graph_search(seed_ids, k=50, max_hops=10):
    results = {}
    n = len(seed_ids)
    for i, sid in enumerate(seed_ids):
        print(f"      [GRAPH] BFS from seed {i+1}/{n} (k={k})...", end="")
        try:
            neighbors = bfs.bfs_nearest_papers(sid, k=k, max_hops=max_hops)
            results[sid] = [n_["paper_id"] for n_ in neighbors]
            print(f" found {len(results[sid])} neighbors")
        except Exception as e:
            results[sid] = []
            print(f" ERROR: {e}")
    return results

def do_rerank(paper_ids, query, top_k=5):
    candidates = [pid for pid in paper_ids if pid in ground_truth_pids]
    if not candidates:
        print(f"      [RERANK] No ground truth papers in pool, returning empty.")
        return []
    pairs, valid_ids = [], []
    for pid in candidates:
        p = all_papers.get(pid)
        if p:
            pairs.append((query, (p["title"] + " " + p["abstract"]).strip()))
            valid_ids.append(pid)
    if not pairs:
        return []
    print(f"      [RERANK] Scoring {len(pairs)} ground-truth candidates...")
    scores = reranker.predict(pairs)
    scored = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    for rank, (pid, score) in enumerate(top, 1):
        title = all_papers.get(pid, {}).get("title", "???")[:60]
        print(f"        #{rank} score={score:.4f} {title}...")
    return [pid for pid, _ in top]

def unique_pool(vec_ids, graph_results):
    all_ids = list(vec_ids)
    seen = set(vec_ids)
    for sid in vec_ids:
        for gid in graph_results.get(sid, []):
            if gid not in seen:
                seen.add(gid)
                all_ids.append(gid)
    return all_ids

# ===============================================================================
# CALIBRATION: binary search for GRAPH_K that yields avg pool ~ TARGET_POOL
# ===============================================================================

_vec_cache = {}

def get_vec_ids(query, vec_n):
    key = (query, vec_n)
    if key not in _vec_cache:
        _vec_cache[key] = do_vector_search(query, n_papers=vec_n)
    return _vec_cache[key]

CAL_QUERIES = [0, 4, 8]  # subset indices for fast calibration

def measure_avg_pool(vec_n, graph_k, query_indices=None):
    """Measure average deduped pool size across queries for given vec_n and graph_k."""
    if query_indices is None:
        query_indices = CAL_QUERIES
    bfs.bfs_nearest_papers.cache_clear()
    pools = []
    for qi in query_indices:
        vec_ids = get_vec_ids(QUERIES[qi], vec_n)
        graph_results = do_graph_search(vec_ids, k=graph_k, max_hops=10)
        all_ids = unique_pool(vec_ids, graph_results)
        pools.append(len(all_ids))
        print(f"      [CAL] Q{qi+1}: pool={len(all_ids)}")
    avg = np.mean(pools)
    print(f"      [CAL] Avg pool for GRAPH_K={graph_k}: {avg:.0f}")
    return avg

def calibrate_graph_k(vec_n):
    """Binary search for GRAPH_K so that avg deduped pool ~ TARGET_POOL."""
    lo, hi = 1, 200
    best_k, best_diff = lo, float('inf')
    best_pool = 0

    print(f"\n  [CALIBRATE] Starting binary search for VEC={vec_n}, target pool={TARGET_POOL}+/-{TOLERANCE}")

    # If vec_n alone already >= target, graph_k=1 (minimal graph expansion)
    bfs.bfs_nearest_papers.cache_clear()
    print(f"  [CALIBRATE] Testing lower bound GRAPH_K=1...")
    pool_at_1 = measure_avg_pool(vec_n, 1)
    if pool_at_1 >= TARGET_POOL:
        print(f"  [CALIBRATE] VEC={vec_n} already yields pool={pool_at_1:.0f} with GRAPH_K=1. Done.")
        return 1, pool_at_1

    # Check upper bound
    print(f"  [CALIBRATE] Testing upper bound GRAPH_K={hi}...")
    pool_at_hi = measure_avg_pool(vec_n, hi)
    if pool_at_hi < TARGET_POOL:
        print(f"  [CALIBRATE] VEC={vec_n}: even GRAPH_K={hi} only gives pool={pool_at_hi:.0f}. Using max.")
        return hi, pool_at_hi

    # Binary search
    iteration = 0
    while lo <= hi:
        iteration += 1
        mid = (lo + hi) // 2
        print(f"  [CALIBRATE] Iteration {iteration}: trying GRAPH_K={mid} (range [{lo}, {hi}])...")
        avg_pool = measure_avg_pool(vec_n, mid)
        diff = abs(avg_pool - TARGET_POOL)

        if diff < best_diff:
            best_diff = diff
            best_k = mid
            best_pool = avg_pool

        if diff <= TOLERANCE:
            print(f"  [CALIBRATE] SUCCESS: GRAPH_K={mid}, avg_pool={avg_pool:.0f} (within tolerance)")
            return mid, avg_pool

        if avg_pool < TARGET_POOL:
            print(f"  [CALIBRATE] Pool {avg_pool:.0f} < {TARGET_POOL}, increasing GRAPH_K...")
            lo = mid + 1
        else:
            print(f"  [CALIBRATE] Pool {avg_pool:.0f} > {TARGET_POOL}, decreasing GRAPH_K...")
            hi = mid - 1

    print(f"  [CALIBRATE] Converged: best GRAPH_K={best_k}, avg_pool={best_pool:.0f}")
    return best_k, best_pool


# ===============================================================================
# SWEEP
# ===============================================================================
VEC_VALUES = [5, 10, 15, 20, 30, 50, 75, 100, 125, 150, 175, 200, 250]

print(f"\n{'='*80}")
print(f"SWEEP CONFIG")
print(f"  VEC values: {VEC_VALUES}")
print(f"  Target avg pool: {TARGET_POOL} (tolerance: +/-{TOLERANCE})")
print(f"  Calibration queries: Q{CAL_QUERIES[0]+1}, Q{CAL_QUERIES[1]+1}, Q{CAL_QUERIES[2]+1}")
print(f"  Evaluation: all 10 queries with hybrid_reranker")
print(f"{'='*80}\n")

sweep_rows = []
t_sweep_start = time.time()

for step, vec_n in enumerate(VEC_VALUES):
    print(f"\n{'#'*80}")
    print(f"# STEP {step+1}/{len(VEC_VALUES)}: VEC={vec_n}")
    print(f"{'#'*80}")
    _vec_cache.clear()

    # Step 1: Calibrate GRAPH_K
    t_cal_start = time.time()
    graph_k, cal_pool = calibrate_graph_k(vec_n)
    t_cal = time.time() - t_cal_start
    print(f"\n  [RESULT] Calibrated GRAPH_K={graph_k} for VEC={vec_n} (took {t_cal:.1f}s)")

    # Step 2: Run full evaluation with calibrated params
    print(f"\n  [EVAL] Running full evaluation: VEC={vec_n}, GRAPH_K={graph_k}")
    bfs.bfs_nearest_papers.cache_clear()
    mrr_vals, rec_vals, pool_vals, time_vals = [], [], [], []

    for qi, query in enumerate(QUERIES):
        relevant_ids = relevance_by_query.get(qi + 1, set())
        print(f"\n    [EVAL Q{qi+1}/10] \"{query[:70]}...\"")
        print(f"    [EVAL Q{qi+1}/10] {len(relevant_ids)} relevant papers in ground truth")
        t0 = time.time()

        vec_ids = get_vec_ids(query, vec_n)
        graph_results = do_graph_search(vec_ids, k=graph_k, max_hops=10)
        all_ids = unique_pool(vec_ids, graph_results)
        print(f"    [EVAL Q{qi+1}/10] Deduped pool: {len(all_ids)} papers ({vec_n} vector + graph expansion)")

        top5 = do_rerank(all_ids, query, top_k=5)
        elapsed = time.time() - t0

        mrr5 = mrr_at_k(top5, relevant_ids, k=5)
        rec5 = recall_at_k(top5, relevant_ids, k=5)
        mrr_vals.append(mrr5)
        rec_vals.append(rec5)
        pool_vals.append(len(all_ids))
        time_vals.append(elapsed)

        hits = [pid[:12] for pid in top5 if pid in relevant_ids]
        print(f"    [EVAL Q{qi+1}/10] MRR@5={mrr5:.4f}  Recall@5={rec5:.4f}  "
              f"pool={len(all_ids)}  time={elapsed:.1f}s  hits={hits}")

    avg_pool = np.mean(pool_vals)
    avg_mrr = np.mean(mrr_vals)
    avg_rec = np.mean(rec_vals)
    avg_time = np.mean(time_vals)

    row = {
        "vec_n": vec_n, "graph_k": graph_k,
        "avg_pool": round(avg_pool, 1),
        "avg_MRR@5": round(avg_mrr, 4),
        "avg_Recall@5": round(avg_rec, 4),
        "avg_time_s": round(avg_time, 2),
    }
    sweep_rows.append(row)

    print(f"\n  {'='*60}")
    print(f"  STEP {step+1} SUMMARY: VEC={vec_n:>3}  GRAPH_K={graph_k:>3}  "
          f"pool={avg_pool:>6.0f}  MRR@5={avg_mrr:.4f}  Recall@5={avg_rec:.4f}  time={avg_time:.1f}s")
    print(f"  {'='*60}")

    # Save incrementally
    with open("hybrid_vec_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"  [SAVE] Incremental save to hybrid_vec_sweep.csv ({len(sweep_rows)} rows so far)")

    elapsed_total = time.time() - t_sweep_start
    print(f"  [TIME] Total elapsed: {elapsed_total:.0f}s")

print(f"\n\n{'='*80}")
print("FINAL SWEEP RESULTS")
print(f"{'='*80}")
print(f"{'VEC':>5} {'GRAPH_K':>8} {'Pool':>6} {'MRR@5':>8} {'Recall@5':>10} {'Time':>7}")
print("-" * 50)
for row in sweep_rows:
    print(f"{row['vec_n']:>5} {row['graph_k']:>8} {row['avg_pool']:>6.0f} "
          f"{row['avg_MRR@5']:>8.4f} {row['avg_Recall@5']:>10.4f} {row['avg_time_s']:>6.1f}s")

# Find best
best = max(sweep_rows, key=lambda r: (r["avg_MRR@5"], r["avg_Recall@5"]))
print(f"\nBEST CONFIG: VEC={best['vec_n']}, GRAPH_K={best['graph_k']}, "
      f"pool={best['avg_pool']}, MRR@5={best['avg_MRR@5']}, Recall@5={best['avg_Recall@5']}")

total_time = time.time() - t_sweep_start
print(f"\nTotal sweep time: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Saved to hybrid_vec_sweep.csv ({len(sweep_rows)} rows)")
driver.close()
print("\nDone!")
