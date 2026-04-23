# -*- coding: utf-8 -*-
"""Sweep with constant total retrieval budget: VEC * (1 + GRAPH_K) = 1000."""
import functools, csv, sys, os, pathlib
from dotenv import load_dotenv
print = functools.partial(print, flush=True)
csv.field_size_limit(10 * 1024 * 1024)

from collections import deque, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# ---- Config ----
MILVUS_DB  = str(ROOT / "RAG.db")
COLLECTION = "ingestion_v0"
NEO4J_URI  = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]

# Constant budget: VEC * (1 + GRAPH_K) ~ 1200
# GRAPH_K = (1200 / VEC) - 1
TARGET = 1200
CONFIGS = [
    (300,   3),   # 300 * 4  = 1200
    (200,   5),   # 200 * 6  = 1200
    (150,   7),   # 150 * 8  = 1200
    (100,  11),   # 100 * 12 = 1200
    (80,   14),   # 80  * 15 = 1200
    (60,   19),   # 60  * 20 = 1200
    (40,   29),   # 40  * 30 = 1200
    (20,   59),   # 20  * 60 = 1200
    (10,  119),   # 10  * 120= 1200
    (5,   239),   # 5   * 240= 1200
]

# ---- Load models + DBs ----
print("Loading models...")
embed_model = SentenceTransformer("jordyvl/scibert_scivocab_uncased_sentence_transformer")
client = MilvusClient(MILVUS_DB)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ---- BFS ----
CHANNELS = {
    "cites":    [("Paper", "CITES")],
    "coauthor": [("Paper", "WROTE"), ("Author", "WROTE")],
    "venue":    [("Paper", "PUBLISHED_IN"), ("Venue", "PUBLISHED_IN")],
    "field":    [("Paper", "HAS_FIELD"), ("FieldOfStudy", "HAS_FIELD")],
}

def get_neighbors(session, label, node_id, rel_type):
    query = f"MATCH (n:{label} {{id: $id}})-[:{rel_type}]-(m) RETURN labels(m)[0] AS label, m.id AS id"
    return list(session.run(query, id=node_id))

_bfs_cache = {}

def bfs_with_tracking(start_paper_id, k, max_hops=4):
    cache_key = (start_paper_id, k)
    if cache_key in _bfs_cache:
        return _bfs_cache[cache_key]
    channel_results = {}
    paper_to_channel = {}
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
    # Round-robin interleave
    found_papers = []
    seen = set()
    max_len = max((len(v) for v in channel_results.values()), default=0)
    for i in range(max_len):
        for ch_name in CHANNELS:
            results = channel_results[ch_name]
            if i < len(results) and results[i] not in seen:
                seen.add(results[i])
                found_papers.append(results[i])
                if len(found_papers) == k:
                    break
        if len(found_papers) == k:
            break
    result = (found_papers, paper_to_channel)
    _bfs_cache[cache_key] = result
    return result

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

# ---- Pre-cache vector results ----
print("Pre-caching vector results...")
vec_cache = {}
max_vec = max(v for v, g in CONFIGS)
for qi, query in enumerate(QUERIES, 1):
    vec_cache[qi] = do_vector_search(query, n_papers=max_vec)
    print(f"  Q{qi}: cached {len(vec_cache[qi])} vector results")

# ---- Run sweep ----
sweep_results = []

for vec_n, graph_k in CONFIGS:
    budget = vec_n * (1 + graph_k)
    print(f"\n{'='*60}")
    print(f"CONFIG: VEC={vec_n}, GRAPH_K={graph_k}  (budget={budget})")
    print(f"{'='*60}")
    _bfs_cache.clear()

    totals = {"vector": 0, "cites": 0, "coauthor": 0, "venue": 0, "field": 0}
    total_relevant = 0
    total_pool = 0
    total_rel_by_src = {"vector": 0, "cites": 0, "coauthor": 0, "venue": 0, "field": 0}

    for qi, query in enumerate(QUERIES, 1):
        relevant_ids = relevance_by_query.get(qi, set())
        vec_ids = vec_cache[qi][:vec_n]

        pool_source = {}
        for pid in vec_ids:
            pool_source[pid] = "vector"

        for sid in vec_ids:
            graph_papers, paper_channels = bfs_with_tracking(sid, k=graph_k)
            for pid in graph_papers:
                if pid not in pool_source:
                    pool_source[pid] = paper_channels.get(pid, "unknown")

        pool = list(pool_source.keys())
        total_pool += len(pool)

        for pid, src in pool_source.items():
            totals[src] = totals.get(src, 0) + 1
            if pid in relevant_ids:
                total_rel_by_src[src] = total_rel_by_src.get(src, 0) + 1
                total_relevant += 1

        print(f"  Q{qi}: pool={len(pool)}, rel={sum(1 for p in pool if p in relevant_ids)}")

    avg_pool = total_pool / 10
    print(f"\n  Avg pool: {avg_pool:.0f}, Total relevant: {total_relevant}")
    for src in ["vector", "cites", "coauthor", "venue", "field"]:
        print(f"    {src}: {totals[src]} pool, {total_rel_by_src[src]} relevant")

    sweep_results.append({
        "vec": vec_n, "graph_k": graph_k, "budget": budget,
        "avg_pool": avg_pool, "total_relevant": total_relevant,
        "rel_vector": total_rel_by_src["vector"],
        "rel_cites": total_rel_by_src["cites"],
        "rel_coauthor": total_rel_by_src["coauthor"],
        "rel_venue": total_rel_by_src["venue"],
        "rel_field": total_rel_by_src["field"],
        "pool_vector": totals["vector"],
        "pool_cites": totals["cites"],
        "pool_coauthor": totals["coauthor"],
        "pool_venue": totals["venue"],
        "pool_field": totals["field"],
    })

# ---- Save CSV ----
import pandas as pd
df = pd.DataFrame(sweep_results)
df.to_csv("channel_sweep_1200_results.csv", index=False)
print("\nSaved channel_sweep_1200_results.csv")

# ---- Plot ----
labels = [f"V{r['vec']}/G{r['graph_k']}" for r in sweep_results]
x = np.arange(len(labels))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Constant Retrieval Budget (~1200): Shifting Vector to Graph", fontsize=14, fontweight="bold")

# Panel 1: Total relevant papers
ax = axes[0, 0]
ax.bar(x, [r["total_relevant"] for r in sweep_results], color="steelblue")
ax.set_ylabel("Total Relevant Found")
ax.set_title("Total Relevant Papers (across 10 queries)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
for i, r in enumerate(sweep_results):
    ax.text(i, r["total_relevant"] + 0.5, str(r["total_relevant"]), ha="center", fontsize=9)

# Panel 2: Relevant by source (stacked bar)
ax = axes[0, 1]
sources = ["vector", "cites", "coauthor", "venue", "field"]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
bottom = np.zeros(len(sweep_results))
for src, color in zip(sources, colors):
    vals = [r[f"rel_{src}"] for r in sweep_results]
    ax.bar(x, vals, bottom=bottom, label=src, color=color)
    bottom += np.array(vals)
ax.set_ylabel("Relevant Papers")
ax.set_title("Relevant Papers by Source")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(fontsize=8)

# Panel 3: Pool size (deduped) + budget line
ax = axes[1, 0]
bottom = np.zeros(len(sweep_results))
for src, color in zip(sources, colors):
    vals = [r[f"pool_{src}"] for r in sweep_results]
    ax.bar(x, vals, bottom=bottom, label=src, color=color)
    bottom += np.array(vals)
ax.set_ylabel("Deduped Pool Size (total across queries)")
ax.set_title("Pool Composition by Source")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(fontsize=8)

# Panel 4: Relevance rate by source
ax = axes[1, 1]
for src, color in zip(sources, colors):
    rates = []
    for r in sweep_results:
        pool = r[f"pool_{src}"]
        rel = r[f"rel_{src}"]
        rates.append(100 * rel / pool if pool > 0 else 0)
    ax.plot(x, rates, marker="o", label=src, color=color)
ax.set_ylabel("Relevance Rate (%)")
ax.set_title("Relevance Rate by Source")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("channel_sweep_1200_results.png", dpi=150)
print("Saved channel_sweep_1200_results.png")

driver.close()
print("Done.")
