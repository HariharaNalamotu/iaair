#!/usr/bin/env python3
"""
Hybrid no-reranker variant: rank papers by retrieval frequency.
Each paper gets +1 for appearing in vector results,
+1 for each graph seed that found it.
Papers found by BOTH vector and graph rank highest.
"""
import csv, time, sys, os, pathlib
csv.field_size_limit(10 * 1024 * 1024)
from collections import deque, Counter
from functools import lru_cache
from dotenv import load_dotenv

import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MILVUS_DB  = str(ROOT / "RAG.db")
COLLECTION = "ingestion_v0"
NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]
VEC_PAPERS = 50
GRAPH_K    = 10

# ── LOAD MODELS + CONNECT ────────────────────────────────────────────────────
print("Loading models...")
embed_model = SentenceTransformer("jordyvl/scibert_scivocab_uncased_sentence_transformer")
def create_embedding(texts):
    return embed_model.encode(texts)

print("Connecting to Milvus...")
client = MilvusClient(MILVUS_DB)
count = client.query(collection_name=COLLECTION, filter="id >= 0",
                     output_fields=["count(*)"])[0]["count(*)"]
print(f"  Milvus: {count} vectors")

print("Connecting to Neo4j...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ── PaperBFS ─────────────────────────────────────────────────────────────────
class PaperBFS:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    def _get_neighbors(self, session, label, node_id, rel_type):
        query = f"MATCH (n:{label} {{id: $id}})-[:{rel_type}]-(m) RETURN labels(m)[0] AS label, m.id AS id"
        return session.run(query, id=node_id)
    @lru_cache(maxsize=10_000)
    def bfs_nearest_papers(self, start_paper_id, k, max_hops=4):
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
                    records = self._get_neighbors(session, label, node_id, rel)
                    for r in records:
                        key = (r["label"], r["id"])
                        if key not in visited:
                            visited.add(key)
                            queue.append((r["label"], r["id"], depth + 1))
        return found_papers

bfs = PaperBFS(driver)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
print("\nLoading papers and ground truth...")
all_papers = {}
with open("papers.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pid = row["id"]
        if pid not in all_papers:
            all_papers[pid] = {
                "paperId": pid, "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
            }
print(f"  {len(all_papers)} papers loaded")

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

for qi in range(1, 11):
    print(f"  Q{qi}: {len(relevance_by_query.get(qi, set()))} relevant papers")

# ── METRICS ──────────────────────────────────────────────────────────────────
def mrr_at_k(ranked_ids, relevant_ids, k=5):
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(ranked_ids, relevant_ids, k=5):
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)

# ── RETRIEVAL HELPERS ────────────────────────────────────────────────────────
def do_vector_search(query, n_papers=5):
    chunk_limit = min(n_papers * 25, 16000)
    print(f"    [vector] fetching up to {n_papers} unique papers (chunk_limit={chunk_limit})...", flush=True)
    search_res = client.search(
        collection_name=COLLECTION,
        data=[create_embedding([query])[0]],
        limit=chunk_limit, output_fields=["paperId"],
    )
    seen = set()
    result = []
    for hit in search_res[0]:
        pid = hit["entity"]["paperId"]
        if pid not in seen:
            seen.add(pid)
            result.append(pid)
            if len(result) == n_papers:
                break
    print(f"    [vector] got {len(result)} unique papers", flush=True)
    return result

def do_graph_search(seed_ids, k=50, max_hops=10):
    results = {}
    n = len(seed_ids)
    for i, sid in enumerate(seed_ids):
        print(f"\r    [graph] seed {i+1}/{n} ({100*(i+1)//n}%)", end="", flush=True)
        try:
            neighbors = bfs.bfs_nearest_papers(sid, k=k, max_hops=max_hops)
            results[sid] = [n_["paper_id"] for n_ in neighbors]
        except Exception:
            results[sid] = []
    print()
    return results

# ── FREQUENCY-RANKED HYBRID (NO RERANKER) ───────────────────────────────────
def retrieval_hybrid_frequency(query):
    vec_ids = do_vector_search(query, n_papers=VEC_PAPERS)
    graph_results = do_graph_search(vec_ids, k=GRAPH_K, max_hops=10)

    # Count retrieval frequency for each paper
    freq = Counter()

    # +1 for each paper found by vector search
    vec_set = set(vec_ids)
    for pid in vec_ids:
        freq[pid] += 1

    # +1 for each graph seed that found this paper
    for sid, neighbors in graph_results.items():
        for pid in neighbors:
            freq[pid] += 1

    # Build full pool (for pool_size metric)
    all_ids = list(freq.keys())

    # Rank by frequency desc, break ties by vector-search order
    # (papers in vec_ids get tie-break priority)
    vec_rank = {pid: i for i, pid in enumerate(vec_ids)}

    # Filter to ground truth papers only (same as reranker config)
    gt_candidates = [pid for pid in freq if pid in ground_truth_pids]

    ranked = sorted(
        gt_candidates,
        key=lambda pid: (-freq[pid], vec_rank.get(pid, float('inf'))),
    )

    top5 = ranked[:5]

    # Print frequency breakdown for top results
    print(f"    [frequency] pool={len(all_ids)}, gt_candidates={len(gt_candidates)}")
    for i, pid in enumerate(top5):
        in_vec = "V" if pid in vec_set else " "
        graph_count = freq[pid] - (1 if pid in vec_set else 0)
        title = (all_papers.get(pid, {}).get("title", "?"))[:60]
        print(f"      #{i+1}: freq={freq[pid]:>3} [{in_vec}+G{graph_count:<2}] {title}")

    return all_ids, top5

# ── RUN ──────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Running HYBRID FREQUENCY-RANKED (no reranker)")
print("="*70)

results = {}
for qi, query in enumerate(QUERIES):
    relevant_ids = relevance_by_query.get(qi + 1, set())
    print(f"\n  Q{qi+1}/10: {query[:70]}...")
    t0 = time.time()
    pool, top5 = retrieval_hybrid_frequency(query)
    elapsed = time.time() - t0

    mrr5 = mrr_at_k(top5, relevant_ids, k=5)
    rec5 = recall_at_k(top5, relevant_ids, k=5)
    results[qi] = {
        "retrieved": top5, "pool_size": len(pool),
        "mrr5": mrr5, "recall5": rec5,
        "n_relevant": len(relevant_ids), "elapsed": elapsed,
    }

    hits = [pid[:12] for pid in top5 if pid in relevant_ids]
    print(f"  => MRR@5={mrr5:.3f}  Recall@5={rec5:.3f}  "
          f"pool={len(pool)} ({len(relevant_ids)} rel, {elapsed:.1f}s) hits={hits}")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY: hybrid_frequency_rank")
print("="*70)
mrr_vals  = [results[qi]["mrr5"] for qi in range(len(QUERIES))]
rec_vals  = [results[qi]["recall5"] for qi in range(len(QUERIES))]
time_vals = [results[qi]["elapsed"] for qi in range(len(QUERIES))]
pool_vals = [results[qi]["pool_size"] for qi in range(len(QUERIES))]

print(f"  Avg Pool Size : {np.mean(pool_vals):.1f}")
print(f"  Avg MRR@5     : {np.mean(mrr_vals):.4f}")
print(f"  Avg Recall@5  : {np.mean(rec_vals):.4f}")
print(f"  Avg Time      : {np.mean(time_vals):.2f}s")

print("\nPer-query breakdown:")
print(f"  {'Q':>3} {'MRR@5':>8} {'Recall@5':>10} {'Pool':>6} {'#Rel':>5} {'Time':>6}")
print(f"  {'---':>3} {'-----':>8} {'--------':>10} {'----':>6} {'----':>5} {'----':>6}")
for qi in range(len(QUERIES)):
    r = results[qi]
    print(f"  {qi+1:>3} {r['mrr5']:>8.3f} {r['recall5']:>10.3f} {r['pool_size']:>6} {r['n_relevant']:>5} {r['elapsed']:>5.1f}s")

driver.close()
print("\nDone!")
