#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare hybrid_reranker vs vector_reranker across pool sizes 100-500.
For hybrid: VEC fixed at 20% of target pool, GRAPH_K calibrated to hit target.
For vector: VECTOR_VEC = target pool size.
"""
import csv, os, time, sys, functools, pathlib
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

TARGET_POOLS = [100, 150, 200, 250, 300, 350, 400, 450, 500]
TOLERANCE    = 25
CAL_QUERIES  = [0, 4, 8]  # subset for fast calibration

print("[INIT] Loading models...")
embed_model = SentenceTransformer("jordyvl/scibert_scivocab_uncased_sentence_transformer")
reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def create_embedding(texts):
    return embed_model.encode(texts)

print("[INIT] Connecting to Milvus...")
client = MilvusClient(MILVUS_DB)
print("[INIT] Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
print("[INIT] Ready.")

class PaperBFS:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    def _get_neighbors(self, session, label, node_id, rel_type):
        q = f"MATCH (n:{label} {{id: $id}})-[:{rel_type}]-(m) RETURN labels(m)[0] AS label, m.id AS id"
        return list(session.run(q, id=node_id))
    @lru_cache(maxsize=10_000)
    def bfs_nearest_papers(self, start_paper_id, k, max_hops=10):
        queue = deque()
        visited = set()
        queue.append(("Paper", start_paper_id, 0))
        visited.add(("Paper", start_paper_id))
        found_papers = []
        traversal_plan = [
            ("Paper", "CITES"), ("Paper", "HAS_FIELD"), ("Paper", "WROTE"),
            ("Paper", "PUBLISHED_IN"), ("Author", "AFFILIATED_WITH"),
        ]
        with self.driver.session() as session:
            while queue and len(found_papers) < k:
                label, node_id, depth = queue.popleft()
                if depth >= max_hops:
                    continue
                if label == "Paper" and node_id != start_paper_id:
                    found_papers.append({"paper_id": node_id, "distance": depth})
                    if len(found_papers) == k:
                        break
                for src_label, rel in traversal_plan:
                    if label != src_label:
                        continue
                    for r in self._get_neighbors(session, label, node_id, rel):
                        key = (r["label"], r["id"])
                        if key not in visited:
                            visited.add(key)
                            queue.append((r["label"], r["id"], depth + 1))
        return found_papers

bfs = PaperBFS(driver)

# Load data
all_papers = {}
with open("papers.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row["id"]
        if pid not in all_papers:
            all_papers[pid] = {"paperId": pid, "title": row.get("title",""), "abstract": row.get("abstract","")}

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

def do_vector_search(query, n_papers):
    BATCH = 500
    query_vec = create_embedding([query])[0]
    seen, result, offset = set(), [], 0
    while len(result) < n_papers:
        limit = min(BATCH, 16000 - offset)
        if limit <= 0:
            break
        hits = client.search(collection_name=COLLECTION, data=[query_vec],
                             limit=limit, offset=offset, output_fields=["paperId"])[0]
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
    return result

def do_graph_search(seed_ids, k):
    results = {}
    for sid in seed_ids:
        try:
            neighbors = bfs.bfs_nearest_papers(sid, k=k, max_hops=10)
            results[sid] = [n["paper_id"] for n in neighbors]
        except Exception:
            results[sid] = []
    return results

def do_rerank(paper_ids, query, top_k=5):
    candidates = [pid for pid in paper_ids if pid in ground_truth_pids]
    if not candidates:
        return []
    pairs, valid_ids = [], []
    for pid in candidates:
        p = all_papers.get(pid)
        if p:
            pairs.append((query, (p["title"] + " " + p["abstract"]).strip()))
            valid_ids.append(pid)
    if not pairs:
        return []
    scores = reranker.predict(pairs)
    scored = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:top_k]]

def unique_pool(vec_ids, graph_results):
    all_ids = list(vec_ids)
    seen = set(vec_ids)
    for sid in vec_ids:
        for gid in graph_results.get(sid, []):
            if gid not in seen:
                seen.add(gid)
                all_ids.append(gid)
    return all_ids

# Vector cache
_vec_cache = {}
def get_vec_ids(query, n):
    if (query, n) not in _vec_cache:
        _vec_cache[(query, n)] = do_vector_search(query, n)
    return _vec_cache[(query, n)]

def measure_avg_pool(vec_n, graph_k):
    bfs.bfs_nearest_papers.cache_clear()
    pools = []
    for qi in CAL_QUERIES:
        vec_ids = get_vec_ids(QUERIES[qi], vec_n)
        graph_res = do_graph_search(vec_ids, k=graph_k)
        pools.append(len(unique_pool(vec_ids, graph_res)))
    return np.mean(pools)

def calibrate_graph_k(vec_n, target_pool):
    lo, hi = 1, 300
    best_k, best_pool = lo, 0

    pool_lo = measure_avg_pool(vec_n, lo)
    print(f"    CAL GRAPH_K=1 -> pool={pool_lo:.0f}")
    if pool_lo >= target_pool:
        return lo, pool_lo

    pool_hi = measure_avg_pool(vec_n, hi)
    print(f"    CAL GRAPH_K={hi} -> pool={pool_hi:.0f}")
    if pool_hi < target_pool:
        return hi, pool_hi

    best_diff = float('inf')
    for _ in range(10):
        mid = (lo + hi) // 2
        avg_pool = measure_avg_pool(vec_n, mid)
        diff = abs(avg_pool - target_pool)
        print(f"    CAL GRAPH_K={mid} -> pool={avg_pool:.0f} (diff={diff:.0f})")
        if diff < best_diff:
            best_diff = diff
            best_k = mid
            best_pool = avg_pool
        if diff <= TOLERANCE:
            return mid, avg_pool
        if avg_pool < target_pool:
            lo = mid + 1
        else:
            hi = mid - 1
    return best_k, best_pool

def run_config(queries, vec_fn, graph_fn):
    """Run evaluation for a config. vec_fn(q)->vec_ids, graph_fn(vec_ids)->graph_results or None."""
    mrr_vals, rec_vals, pool_vals, time_vals = [], [], [], []
    for qi, query in enumerate(queries):
        relevant_ids = relevance_by_query.get(qi + 1, set())
        t0 = time.time()
        vec_ids = vec_fn(query)
        if graph_fn:
            graph_res = graph_fn(vec_ids)
            pool = unique_pool(vec_ids, graph_res)
        else:
            pool = vec_ids
        top5 = do_rerank(pool, query, top_k=5)
        elapsed = time.time() - t0
        mrr_vals.append(mrr_at_k(top5, relevant_ids))
        rec_vals.append(recall_at_k(top5, relevant_ids))
        pool_vals.append(len(pool))
        time_vals.append(elapsed)
        print(f"      Q{qi+1}: pool={len(pool)} MRR@5={mrr_vals[-1]:.3f} Recall@5={rec_vals[-1]:.3f}")
    return {
        "avg_pool":    round(np.mean(pool_vals), 1),
        "avg_MRR@5":  round(np.mean(mrr_vals), 4),
        "avg_Recall@5": round(np.mean(rec_vals), 4),
        "avg_time_s": round(np.mean(time_vals), 2),
    }

# ===============================================================================
# MAIN SWEEP
# ===============================================================================
print(f"\n{'='*80}")
print(f"Pool Size Comparison Sweep: Hybrid Reranker vs Vector Reranker")
print(f"Target pools: {TARGET_POOLS}")
print(f"Hybrid VEC = 20% of target pool, GRAPH_K calibrated to hit target")
print(f"{'='*80}\n")

rows = []
t_start = time.time()

for target_pool in TARGET_POOLS:
    vec_n = max(5, round(target_pool * 0.20))  # 20% of target as vector seeds
    print(f"\n{'#'*70}")
    print(f"# TARGET POOL={target_pool}  (VEC={vec_n}, calibrating GRAPH_K...)")
    print(f"{'#'*70}")

    _vec_cache.clear()
    graph_k, cal_pool = calibrate_graph_k(vec_n, target_pool)
    print(f"  => Calibrated: VEC={vec_n}, GRAPH_K={graph_k}, cal_pool={cal_pool:.0f}")

    # Hybrid reranker
    bfs.bfs_nearest_papers.cache_clear()
    print(f"\n  [HYBRID RERANKER] VEC={vec_n}, GRAPH_K={graph_k}")
    hybrid_res = run_config(
        QUERIES,
        vec_fn=lambda q, v=vec_n: get_vec_ids(q, v),
        graph_fn=lambda vids, k=graph_k: do_graph_search(vids, k=k),
    )
    print(f"  => Hybrid: pool={hybrid_res['avg_pool']} MRR@5={hybrid_res['avg_MRR@5']} Recall@5={hybrid_res['avg_Recall@5']}")

    # Vector reranker (same target pool size)
    print(f"\n  [VECTOR RERANKER] VEC={target_pool}")
    vector_res = run_config(
        QUERIES,
        vec_fn=lambda q, n=target_pool: get_vec_ids(q, n),
        graph_fn=None,
    )
    print(f"  => Vector: pool={vector_res['avg_pool']} MRR@5={vector_res['avg_MRR@5']} Recall@5={vector_res['avg_Recall@5']}")

    rows.append({
        "target_pool":        target_pool,
        "hybrid_vec_n":       vec_n,
        "hybrid_graph_k":     graph_k,
        "hybrid_avg_pool":    hybrid_res["avg_pool"],
        "hybrid_MRR@5":       hybrid_res["avg_MRR@5"],
        "hybrid_Recall@5":    hybrid_res["avg_Recall@5"],
        "hybrid_time_s":      hybrid_res["avg_time_s"],
        "vector_avg_pool":    vector_res["avg_pool"],
        "vector_MRR@5":       vector_res["avg_MRR@5"],
        "vector_Recall@5":    vector_res["avg_Recall@5"],
        "vector_time_s":      vector_res["avg_time_s"],
    })

    # Incremental save
    with open("pool_comparison_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [SAVE] pool_comparison_sweep.csv updated ({len(rows)} rows)")
    print(f"  [TIME] Total elapsed: {time.time()-t_start:.0f}s")

# ===============================================================================
# PRINT TABLE
# ===============================================================================
print(f"\n\n{'='*90}")
print("FINAL RESULTS")
print(f"{'='*90}")
print(f"{'Target':>8} {'H-Pool':>8} {'H-MRR@5':>9} {'H-Rec@5':>9} | {'V-Pool':>8} {'V-MRR@5':>9} {'V-Rec@5':>9}")
print("-" * 90)
for r in rows:
    print(f"{r['target_pool']:>8} {r['hybrid_avg_pool']:>8.0f} {r['hybrid_MRR@5']:>9.4f} {r['hybrid_Recall@5']:>9.4f} | "
          f"{r['vector_avg_pool']:>8.0f} {r['vector_MRR@5']:>9.4f} {r['vector_Recall@5']:>9.4f}")

# ===============================================================================
# PLOT
# ===============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

targets   = [r["target_pool"]     for r in rows]
h_mrr     = [r["hybrid_MRR@5"]    for r in rows]
h_rec     = [r["hybrid_Recall@5"] for r in rows]
v_mrr     = [r["vector_MRR@5"]    for r in rows]
v_rec     = [r["vector_Recall@5"] for r in rows]
h_pool    = [r["hybrid_avg_pool"] for r in rows]
v_pool    = [r["vector_avg_pool"] for r in rows]
h_time    = [r["hybrid_time_s"]   for r in rows]
v_time    = [r["vector_time_s"]   for r in rows]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Hybrid Reranker vs Vector Reranker: Performance Across Pool Sizes", fontsize=13, fontweight="bold")

# Plot 1: MRR@5
ax = axes[0, 0]
ax.plot(targets, h_mrr, "o-", color="#2563eb", linewidth=2, markersize=8, label="Hybrid Reranker")
ax.plot(targets, v_mrr, "s--", color="#dc2626", linewidth=2, markersize=8, label="Vector Reranker")
ax.set_xlabel("Target Pool Size")
ax.set_ylabel("MRR@5")
ax.set_title("MRR@5 vs Pool Size")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(targets)

# Plot 2: Recall@5
ax = axes[0, 1]
ax.plot(targets, h_rec, "o-", color="#2563eb", linewidth=2, markersize=8, label="Hybrid Reranker")
ax.plot(targets, v_rec, "s--", color="#dc2626", linewidth=2, markersize=8, label="Vector Reranker")
ax.set_xlabel("Target Pool Size")
ax.set_ylabel("Recall@5")
ax.set_title("Recall@5 vs Pool Size")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(targets)

# Plot 3: Actual pool sizes achieved
ax = axes[1, 0]
ax.plot(targets, h_pool, "o-", color="#2563eb", linewidth=2, markersize=8, label="Hybrid (actual)")
ax.plot(targets, v_pool, "s--", color="#dc2626", linewidth=2, markersize=8, label="Vector (actual)")
ax.plot(targets, targets, "k:", linewidth=1, alpha=0.5, label="Target")
ax.set_xlabel("Target Pool Size")
ax.set_ylabel("Actual Avg Pool Size")
ax.set_title("Actual Pool Size vs Target")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(targets)

# Plot 4: Latency
ax = axes[1, 1]
x = np.arange(len(targets))
width = 0.35
ax.bar(x - width/2, h_time, width, color="#2563eb", alpha=0.8, label="Hybrid Reranker")
ax.bar(x + width/2, v_time, width, color="#dc2626", alpha=0.8, label="Vector Reranker")
ax.set_xlabel("Target Pool Size")
ax.set_ylabel("Avg Time per Query (s)")
ax.set_title("Latency vs Pool Size")
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("pool_comparison_sweep.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to pool_comparison_sweep.png")
print(f"Total time: {time.time()-t_start:.0f}s")
driver.close()
print("Done!")
