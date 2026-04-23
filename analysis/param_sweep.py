#!/usr/bin/env python3
"""
Parameter sweep: try different vector_limit and graph_k combinations,
report pool sizes and MRR@5/Recall@5 for each.
"""
import csv, os, time, sys, pathlib
csv.field_size_limit(10 * 1024 * 1024)
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
from collections import deque
from functools import lru_cache
from dotenv import load_dotenv

import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from neo4j import GraphDatabase

ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

MILVUS_DB  = str(ROOT / "RAG.db")
COLLECTION = "ingestion_v0"
NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

print("Loading models...")
embed_model = SentenceTransformer("jordyvl/scibert_scivocab_uncased_sentence_transformer")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def create_embedding(texts):
    return embed_model.encode(texts)

client = MilvusClient(MILVUS_DB)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# PaperBFS
class PaperBFS:
    def __init__(self, d):
        self.driver = d
    def _get_neighbors(self, session, label, node_id, rel_type):
        return session.run(
            f"MATCH (n:{label} {{id: $id}})-[:{rel_type}]-(m) RETURN labels(m)[0] AS label, m.id AS id",
            id=node_id)
    @lru_cache(maxsize=10_000)
    def bfs_nearest_papers(self, start_paper_id, k, max_hops=4):
        queue = deque([("Paper", start_paper_id, 0)])
        visited = {("Paper", start_paper_id)}
        found = []
        plan = [("Paper","CITES"),("Paper","HAS_FIELD"),("Paper","WROTE"),
                ("Paper","PUBLISHED_IN"),("Author","AFFILIATED_WITH")]
        with self.driver.session() as session:
            while queue and len(found) < k:
                label, nid, depth = queue.popleft()
                if depth >= max_hops: continue
                if label == "Paper" and nid != start_paper_id:
                    found.append({"paper_id": nid, "distance": depth})
                    if len(found) == k: break
                for sl, rel in plan:
                    if label != sl: continue
                    for r in self._get_neighbors(session, label, nid, rel):
                        key = (r["label"], r["id"])
                        if key not in visited:
                            visited.add(key)
                            queue.append((r["label"], r["id"], depth+1))
        return found
bfs = PaperBFS(driver)

# Load papers
all_papers = {}
with open("papers.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["id"] not in all_papers:
            all_papers[row["id"]] = {"title": row.get("title",""), "abstract": row.get("abstract","")}

# Load ground truth
relevance_by_query = {}
with open("ground_truth_relevance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qi = int(row["query_id"])
        if qi not in relevance_by_query: relevance_by_query[qi] = set()
        if int(row["relevant"]) == 1:
            relevance_by_query[qi].add(row["paperId"])

QUERIES = [
    "How can natural language processing be used to automate systematic literature reviews?",
    "What machine learning techniques are used for biomedical named entity recognition?",
    "How do knowledge graphs improve information retrieval in scientific research?",
    "What are the challenges of extracting structured data from unstructured scientific text?",
    "How can citation network analysis reveal emerging research trends?",
    "What deep learning architectures are effective for text classification of academic papers?",
    "How do embedding models capture semantic similarity between research documents?",
    "What methods exist for automated quality assessment of scientific evidence?",
    "How can graph neural networks be applied to bibliometric analysis?",
    "What are effective approaches for cross-lingual information extraction from scientific literature?",
]

def mrr_at_k(ranked, rel, k=5):
    for i, pid in enumerate(ranked[:k]):
        if pid in rel: return 1.0/(i+1)
    return 0.0

def recall_at_k(ranked, rel, k=5):
    if not rel: return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)

def do_vector_search(query, limit):
    res = client.search(collection_name=COLLECTION, data=[create_embedding([query])[0]],
                        limit=limit, output_fields=["paperId"])
    seen, result = set(), []
    for h in res[0]:
        pid = h["entity"]["paperId"]
        if pid not in seen: seen.add(pid); result.append(pid)
    return result

def do_graph_search(seed_ids, k, max_hops=10):
    results = {}
    for sid in seed_ids:
        try:
            results[sid] = [n["paper_id"] for n in bfs.bfs_nearest_papers(sid, k=k, max_hops=max_hops)]
        except: results[sid] = []
    return results

def do_rerank(paper_ids, query, top_k=5):
    if not paper_ids: return []
    pairs, valid = [], []
    for pid in paper_ids:
        p = all_papers.get(pid)
        if p:
            pairs.append((query, (p["title"]+" "+p["abstract"]).strip()))
            valid.append(pid)
    if not pairs: return []
    scores = reranker.predict(pairs)
    scored = sorted(zip(valid, scores), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored[:top_k]]

def unique_pool(vec_ids, graph_results):
    all_ids = list(vec_ids); seen = set(vec_ids)
    for sid in vec_ids:
        for gid in graph_results.get(sid, []):
            if gid not in seen: seen.add(gid); all_ids.append(gid)
    return all_ids

def interleave(vec_ids, graph_results):
    seen, ordered = set(), []
    for vid in vec_ids:
        if vid not in seen: seen.add(vid); ordered.append(vid)
    mx = max((len(v) for v in graph_results.values()), default=0)
    for ri in range(mx):
        for vid in vec_ids:
            nb = graph_results.get(vid, [])
            if ri < len(nb) and nb[ri] not in seen:
                seen.add(nb[ri]); ordered.append(nb[ri])
    return ordered

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

# Hybrid configs: (vec_limit, graph_k)
hybrid_params = [
    (5, 20),
    (5, 50),
    (10, 10),
    (10, 20),
    (10, 30),
    (15, 10),
    (15, 20),
    (20, 5),
    (20, 10),
    (20, 15),
    (20, 30),
]

# Vector-only configs: vec_limit (Milvus limit, not unique papers)
vector_limits = [50, 100, 150, 200, 300]

print(f"\n{'='*90}")
print("HYBRID SEARCH PARAMETER SWEEP (with reranker)")
print(f"{'='*90}")
print(f"{'vec_limit':>10} {'graph_k':>8} {'avg_pool':>9} {'avg_MRR@5':>10} {'avg_Rec@5':>10} {'time':>7}")
print("-"*60)

sweep_rows = []

for vlim, gk in hybrid_params:
    mrrs, recs, pools, times = [], [], [], []
    for qi, query in enumerate(QUERIES):
        rel = relevance_by_query.get(qi+1, set())
        t0 = time.time()
        vids = do_vector_search(query, limit=vlim)
        gres = do_graph_search(vids, k=gk)
        pool = unique_pool(vids, gres)
        top5 = do_rerank(pool, query, top_k=5)
        elapsed = time.time() - t0
        mrrs.append(mrr_at_k(top5, rel))
        recs.append(recall_at_k(top5, rel))
        pools.append(len(pool))
        times.append(elapsed)
    avg_pool = np.mean(pools)
    avg_mrr = np.mean(mrrs)
    avg_rec = np.mean(recs)
    avg_time = np.mean(times)
    print(f"{vlim:>10} {gk:>8} {avg_pool:>9.0f} {avg_mrr:>10.4f} {avg_rec:>10.4f} {avg_time:>6.1f}s")
    sweep_rows.append({
        "config": "hybrid_reranker", "vec_limit": vlim, "graph_k": gk,
        "avg_pool": round(avg_pool,1), "avg_MRR@5": round(avg_mrr,4),
        "avg_Recall@5": round(avg_rec,4), "avg_time_s": round(avg_time,2)
    })

print(f"\n{'='*90}")
print("HYBRID SEARCH PARAMETER SWEEP (no reranker, interleaved)")
print(f"{'='*90}")
print(f"{'vec_limit':>10} {'graph_k':>8} {'avg_pool':>9} {'avg_MRR@5':>10} {'avg_Rec@5':>10} {'time':>7}")
print("-"*60)

for vlim, gk in hybrid_params:
    mrrs, recs, pools, times = [], [], [], []
    for qi, query in enumerate(QUERIES):
        rel = relevance_by_query.get(qi+1, set())
        t0 = time.time()
        vids = do_vector_search(query, limit=vlim)
        gres = do_graph_search(vids, k=gk)
        pool = unique_pool(vids, gres)
        top5 = interleave(vids, gres)[:5]
        elapsed = time.time() - t0
        mrrs.append(mrr_at_k(top5, rel))
        recs.append(recall_at_k(top5, rel))
        pools.append(len(pool))
        times.append(elapsed)
    avg_pool = np.mean(pools)
    avg_mrr = np.mean(mrrs)
    avg_rec = np.mean(recs)
    avg_time = np.mean(times)
    print(f"{vlim:>10} {gk:>8} {avg_pool:>9.0f} {avg_mrr:>10.4f} {avg_rec:>10.4f} {avg_time:>6.1f}s")
    sweep_rows.append({
        "config": "hybrid_no_reranker", "vec_limit": vlim, "graph_k": gk,
        "avg_pool": round(avg_pool,1), "avg_MRR@5": round(avg_mrr,4),
        "avg_Recall@5": round(avg_rec,4), "avg_time_s": round(avg_time,2)
    })

print(f"\n{'='*90}")
print("VECTOR SEARCH PARAMETER SWEEP")
print(f"{'='*90}")
print(f"{'milvus_limit':>13} {'unique_papers':>14} {'avg_MRR@5':>10} {'avg_Rec@5':>10}  (with reranker)")
print(f"{'':>13} {'':>14} {'avg_MRR@5':>10} {'avg_Rec@5':>10}  (no reranker)")
print("-"*70)

for vlim in vector_limits:
    # With reranker
    mrrs_r, recs_r, mrrs_n, recs_n, pools = [], [], [], [], []
    for qi, query in enumerate(QUERIES):
        rel = relevance_by_query.get(qi+1, set())
        vids = do_vector_search(query, limit=vlim)
        pools.append(len(vids))
        # reranker
        top5_r = do_rerank(vids, query, top_k=5)
        mrrs_r.append(mrr_at_k(top5_r, rel))
        recs_r.append(recall_at_k(top5_r, rel))
        # no reranker
        mrrs_n.append(mrr_at_k(vids[:5], rel))
        recs_n.append(recall_at_k(vids[:5], rel))
    avg_pool = np.mean(pools)
    print(f"{vlim:>13} {avg_pool:>14.0f} {np.mean(mrrs_r):>10.4f} {np.mean(recs_r):>10.4f}  (reranker)")
    print(f"{'':>13} {'':>14} {np.mean(mrrs_n):>10.4f} {np.mean(recs_n):>10.4f}  (no reranker)")
    sweep_rows.append({
        "config": "vector_reranker", "vec_limit": vlim, "graph_k": 0,
        "avg_pool": round(avg_pool,1), "avg_MRR@5": round(np.mean(mrrs_r),4),
        "avg_Recall@5": round(np.mean(recs_r),4), "avg_time_s": 0
    })
    sweep_rows.append({
        "config": "vector_only", "vec_limit": vlim, "graph_k": 0,
        "avg_pool": round(avg_pool,1), "avg_MRR@5": round(np.mean(mrrs_n),4),
        "avg_Recall@5": round(np.mean(recs_n),4), "avg_time_s": 0
    })

# Save sweep results
with open("param_sweep_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
    writer.writeheader()
    writer.writerows(sweep_rows)
print(f"\nSaved param_sweep_results.csv ({len(sweep_rows)} rows)")

driver.close()
