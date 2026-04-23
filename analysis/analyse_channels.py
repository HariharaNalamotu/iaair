# -*- coding: utf-8 -*-
"""Analyse which BFS channels contribute relevant papers in hybrid retrieval."""
import functools, csv, sys, os, pathlib
from dotenv import load_dotenv
print = functools.partial(print, flush=True)
csv.field_size_limit(10 * 1024 * 1024)

from collections import deque, Counter, defaultdict
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# ---- Config (same as run_evaluation.py) ----
MILVUS_DB  = str(ROOT / "RAG.db")
COLLECTION = "ingestion_v0"
NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]
HYBRID_VEC = 20
GRAPH_K    = 50

# ---- Load models + DBs ----
print("Loading models...")
embed_model = SentenceTransformer("jordyvl/scibert_scivocab_uncased_sentence_transformer")
client = MilvusClient(MILVUS_DB)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ---- BFS with channel tracking ----
CHANNELS = {
    "cites":    [("Paper", "CITES")],
    "coauthor": [("Paper", "WROTE"), ("Author", "WROTE")],
    "venue":    [("Paper", "PUBLISHED_IN"), ("Venue", "PUBLISHED_IN")],
    "field":    [("Paper", "HAS_FIELD"), ("FieldOfStudy", "HAS_FIELD")],
}

def get_neighbors(session, label, node_id, rel_type):
    query = f"MATCH (n:{label} {{id: $id}})-[:{rel_type}]-(m) RETURN labels(m)[0] AS label, m.id AS id"
    return list(session.run(query, id=node_id))

def bfs_with_channel_tracking(start_paper_id, k, max_hops=4):
    """Returns (found_papers, paper_to_channel) where paper_to_channel maps paper_id -> channel name."""
    channel_results = {}
    paper_to_channel = {}  # paper_id -> first channel that found it

    with driver.session() as session:
        for ch_name, ch_plan in CHANNELS.items():
            queue = deque([("Paper", start_paper_id, 0)])
            visited = {("Paper", start_paper_id)}
            found = []
            while queue and len(found) < k:
                label, node_id, depth = queue.popleft()
                if depth >= max_hops:
                    continue
                if label == "Paper" and node_id != start_paper_id:
                    found.append(node_id)
                    if node_id not in paper_to_channel:
                        paper_to_channel[node_id] = ch_name
                for src_label, rel in ch_plan:
                    if label != src_label:
                        continue
                    records = get_neighbors(session, label, node_id, rel)
                    for r in records:
                        key = (r["label"], r["id"])
                        if key not in visited:
                            visited.add(key)
                            queue.append((r["label"], r["id"], depth + 1))
            channel_results[ch_name] = found

    # Round-robin interleave (same as run_evaluation.py)
    found_papers = []
    seen = set()
    max_len = max((len(v) for v in channel_results.values()), default=0)
    for i in range(max_len):
        for ch_name in CHANNELS:
            results = channel_results[ch_name]
            if i < len(results):
                pid = results[i]
                if pid not in seen:
                    seen.add(pid)
                    found_papers.append(pid)
                    if len(found_papers) == k:
                        return found_papers, paper_to_channel, channel_results
    return found_papers, paper_to_channel, channel_results

def do_vector_search(query, n_papers):
    BATCH = 500
    query_vec = embed_model.encode([query])[0]
    seen, result = set(), []
    offset = 0
    while len(result) < n_papers:
        limit = min(BATCH, 16000 - offset)
        if limit <= 0:
            break
        hits = client.search(
            collection_name=COLLECTION, data=[query_vec],
            limit=limit, offset=offset, output_fields=["paperId"],
        )[0]
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

# ---- Load ground truth ----
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

# ---- Run analysis ----
print("\n" + "="*80)
print("CHANNEL CONTRIBUTION ANALYSIS")
print("="*80)

# Aggregate counters
total_pool_by_source = Counter()       # "vector" / channel_name -> total papers across all queries
total_relevant_by_source = Counter()   # same but only relevant papers
total_gt_by_source = Counter()         # same but only ground truth papers
per_query_stats = []

for qi, query in enumerate(QUERIES, 1):
    relevant_ids = relevance_by_query.get(qi, set())
    print(f"\nQ{qi}: {query[:70]}...")
    print(f"  Relevant papers in GT: {len(relevant_ids)}")

    # Vector search
    vec_ids = do_vector_search(query, n_papers=HYBRID_VEC)
    print(f"  Vector seeds: {len(vec_ids)}")

    # Graph search with channel tracking
    pool_source = {}  # paper_id -> source ("vector" or channel name)
    for pid in vec_ids:
        pool_source[pid] = "vector"

    all_graph_ids = []
    for seed_i, sid in enumerate(vec_ids):
        if (seed_i + 1) % 20 == 0:
            print(f"    BFS seed {seed_i+1}/{len(vec_ids)}...")
        graph_papers, paper_channels, ch_results = bfs_with_channel_tracking(sid, k=GRAPH_K)
        for pid in graph_papers:
            if pid not in pool_source:
                pool_source[pid] = paper_channels.get(pid, "unknown")
            all_graph_ids.append(pid)

    pool = list(pool_source.keys())
    print(f"  Total pool: {len(pool)}")

    # Count by source
    source_counts = Counter(pool_source.values())
    source_relevant = Counter()
    source_gt = Counter()
    for pid, src in pool_source.items():
        if pid in ground_truth_pids:
            source_gt[src] += 1
        if pid in relevant_ids:
            source_relevant[src] += 1

    print(f"  {'Source':<12} {'Pool':>6} {'In GT':>6} {'Relevant':>8}")
    print(f"  {'-'*36}")
    for src in ["vector", "cites", "coauthor", "venue", "field"]:
        p = source_counts.get(src, 0)
        g = source_gt.get(src, 0)
        r = source_relevant.get(src, 0)
        print(f"  {src:<12} {p:>6} {g:>6} {r:>8}")
        total_pool_by_source[src] += p
        total_relevant_by_source[src] += r
        total_gt_by_source[src] += g

    per_query_stats.append({
        "query_id": qi,
        "pool_size": len(pool),
        "source_counts": dict(source_counts),
        "source_relevant": dict(source_relevant),
        "source_gt": dict(source_gt),
    })

# ---- Summary ----
print("\n" + "="*80)
print("AGGREGATE ACROSS ALL 10 QUERIES")
print("="*80)
print(f"{'Source':<12} {'Total Pool':>10} {'Total GT':>10} {'Total Rel':>10} {'Rel Rate':>10}")
print("-"*55)
for src in ["vector", "cites", "coauthor", "venue", "field"]:
    p = total_pool_by_source[src]
    g = total_gt_by_source[src]
    r = total_relevant_by_source[src]
    rate = f"{100*r/p:.1f}%" if p > 0 else "N/A"
    print(f"{src:<12} {p:>10} {g:>10} {r:>10} {rate:>10}")

total_p = sum(total_pool_by_source.values())
total_g = sum(total_gt_by_source.values())
total_r = sum(total_relevant_by_source.values())
rate = f"{100*total_r/total_p:.1f}%" if total_p > 0 else "N/A"
print("-"*55)
print(f"{'TOTAL':<12} {total_p:>10} {total_g:>10} {total_r:>10} {rate:>10}")

# Breakdown: what % of relevant papers come from each source?
print(f"\nRelevant paper source breakdown:")
for src in ["vector", "cites", "coauthor", "venue", "field"]:
    r = total_relevant_by_source[src]
    pct = f"{100*r/total_r:.1f}%" if total_r > 0 else "N/A"
    print(f"  {src:<12} {r:>4} relevant  ({pct})")

driver.close()
print("\nDone.")
