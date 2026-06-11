# Hybrid Graph–Vector Retrieval for Scientific Papers
### A Budget-Controlled Evaluation of Whether a Knowledge Graph Improves Retrieval

**Author:** Hari Nalamotu

This repository contains the full corpus, code, and evaluation for a controlled study of
whether augmenting dense vector retrieval with a citation/authorship/venue/field knowledge
graph ("GraphRAG") improves retrieval for scientific papers, when compared against an
**equivalent** semantic baseline under matched conditions.

Everything needed to reproduce the dataset and the results is committed; no external API key
is required (see [Recreating the dataset](#recreating-the-dataset)).

---

## Research question

> *Does adding a citation / authorship / venue / field knowledge graph improve retrieval for
> scientific papers, when compared against an equivalent vector search under matched
> conditions — or not?*

Naive GraphRAG comparisons are confounded by three factors, all of which this study removes:
(1) a hybrid method usually returns a **larger** candidate pool, so it can contain more
relevant papers merely by being bigger; (2) hybrid pipelines usually add a **re-ranker** the
vector baseline lacks; and (3) the vector/graph **mix** is a free parameter, and reporting
only its best setting flatters the graph.

## Summary of findings

Over a 1,500-paper corpus, three retrievers — a pure-vector baseline, a query-blind
breadth-first graph hybrid (**Bfs**), and a query-aware meta-path best-first hybrid
(**MetaPath**) — are compared at matched candidate-pool sizes, under the **same** cross-encoder
re-ranker, averaged across all vector/graph mixes, and judged against a method-independent
random sample of 250 papers (2,500 binary relevance labels). Differences are tested with
bootstrap confidence intervals and paired Wilcoxon signed-rank tests over the 10 queries.

- **The query-aware graph does not significantly beat vector.** On the typical (mix-averaged)
  setting, MetaPath and vector are statistically indistinguishable on top-rank quality
  (MRR@5 0.896 vs. 0.903; paired Wilcoxon *p* = 0.89).
- **Vector leads on coverage at every budget** (Recall@10 0.308 vs. 0.284; Relevant-in-Pool
  6.93 vs. 5.71). The deficit vs. MetaPath is consistent but not individually significant
  (*p* = 0.09); the naive **Bfs** graph is significantly worse (*p* = 0.005).
- **The graph's perfect MRR@5 (1.000) is an oracle result** — it appears only at a
  test-selected vector/graph mix (mean best-mix 0.994), not at a deployable operating point.
- **The re-ranker carries the result.** Without the cross-encoder, every graph method collapses
  below the unranked vector floor (≈0.50 MRR@5).

**Conclusion.** Once the standard confounds are removed, a knowledge graph does not improve
retrieval over an equivalent semantic baseline for scientific papers. At best it trades
coverage for an oracle-conditional sharpening of the very top of the ranking.

---

## Repository structure

```
.
├── data/
│   ├── papers.csv                  # 1,500 papers (id, title, abstract, year, venue, full_text*)
│   ├── citations.csv               # CITES edges
│   ├── written_by.csv              # authorship (Author→Paper)
│   ├── write_together.csv          # co-authorship (Author–Author)
│   ├── written_for.csv             # PUBLISHED_IN (Paper→Venue)
│   ├── field_of_study.csv          # HAS_FIELD (Paper→Field)
│   ├── affiliations.csv            # AFFILIATED_WITH (Author→Institution)
│   ├── authors.csv / venues.csv / fields_of_study.csv / institutions.csv
│   ├── queries.csv                 # 10 evaluation queries
│   ├── ground_truth_papers_250.csv # 250 randomly-sampled judged candidates
│   ├── ground_truth_relevance.csv  # 2,500 binary relevance labels (250 × 10)
│   ├── vector_chunks.csv.gz        # frozen source text of all 56,379 embedded chunks
│   └── dataset_formation.ipynb     # how the judged set was constructed
│
├── ingestion/
│   └── run_ingestion.py            # original Semantic Scholar crawl (provenance; needs API keys)
│
├── scripts/
│   ├── build_graph.py              # committed CSVs → Neo4j graph
│   ├── build_vectors.py            # vector_chunks.csv.gz → FAISS index (deterministic)
│   ├── precompute_embeddings.py    # papers.csv → paper-embedding cache
│   ├── extract_chunks.py           # one-time: RAG.db → vector_chunks.csv.gz (provenance only)
│   └── visualize_results.py        # generate result figures
│
├── retrieval/
│   └── query_aware_graph.py        # MetaPathBestFirstGraph (+ greedy variant)
│
├── evaluation/
│   └── final_evaluation.py         # budget × mix sweep — the main results
│
├── results/
│   ├── final_evaluation_sweep.csv  # committed sweep results
│   └── figures/                    # committed figures
│
├── .env.example                    # required environment variables
└── requirements.txt
```
`*full_text` is present for the 139 open-access PDFs.

---

## Dataset

**Corpus.** 1,500 papers grown by breadth-first citation crawl from a single seed —
*Construction of the Literature Graph in Semantic Scholar* (Ammar et al., NAACL 2018,
`649def34f8be52c8b66281af98ae884c09aef38b`) — spanning RAG, knowledge graphs, NLP, information
extraction, and large language models. Because the corpus is grown by citation crawl, it is
more citation-connected than an arbitrary sample, and is therefore structurally *favourable* to
graph methods — a point worth bearing in mind when reading the (negative) result.

**Dual representation.** The same 1,500 papers are represented two ways:
- **Vectors** — abstracts and full text are token-chunked (size 128, overlap 26) and embedded
  with a sentence-transformer SciBERT (mean-sqrt-length pooling; *not* fine-tuned for
  retrieval), giving **56,379 chunk embeddings** in a flat FAISS inner-product index. A paper's
  query similarity is the score of its best-matching chunk.
- **Graph** — a Neo4j property graph of **10,199 nodes** (1,500 Papers, 8,071 Authors,
  618 Venues, 10 Fields) and ~240k edges (CITES 26,272; WROTE 9,366; WRITES_WITH 202,044;
  PUBLISHED_IN 1,387; HAS_FIELD 1,359). Co-author edges dominate by count but are concentrated
  in a few hyper-authored papers (one with 500 authors; the top five produce 83% of all
  co-author pairs).

**Relevance judgments (method-independent).** The 10 queries were written *first*, before any
paper was selected, each with a pre-registered rubric of the information facets a relevant
paper should cover. Only then were 250 candidates drawn from the corpus **programmatically at
random**. A single annotator judged each paper *binary* relevant to a query if its title and
abstract covered any facet in that query's rubric, never consulting any retrieval system's
output. This yields 2,500 labels (12–20 relevant per query, mean 16.3). Because the 250 are
sampled independently of every retrieval method, the between-method comparison is free of
pooling bias; the cost is that absolute recall is low by construction. Limitations:
single-annotator (no inter-annotator agreement), disjunctive (lenient) relevance, and
title/abstract-only judging.

---

## Recreating the dataset

The dataset is defined by **canonical committed artifacts** (the CSVs and one gzipped text
file). Every derived artifact — the Neo4j graph, the FAISS vector index, and the
paper-embedding cache — is a **deterministic function** of those inputs and is regenerated by
the steps below. No Semantic Scholar API key, no `RAG.db`, and no database dump are required.

### Prerequisites

- Python 3.11, an NVIDIA GPU (CPU works, slower), and a running **Neo4j 5** instance.
- Install dependencies: `pip install -r requirements.txt`
- Copy and fill credentials:
  ```bash
  cp .env.example .env       # set NEO4J_URI, NEO4J_USER, NEO4J_PASS
  ```

### Step 1 — Build the knowledge graph (from committed CSVs)

```bash
python scripts/build_graph.py --force
```
Reads the committed edge/node CSVs and (re)creates the exact 10,199-node graph in Neo4j. No API
key and no `transfer/neo4j.dump` needed.

### Step 2 — Build the FAISS vector index (from committed chunk text)

```bash
python scripts/build_vectors.py            # add --verify to compare against an existing index
```
Re-embeds the frozen chunk text in `data/vector_chunks.csv.gz` (all 56,379 chunks, including
their exact source text) to write `data/vectors.npy` (56,379 × 768 float32) and
`data/vector_paperids.json`. Verified to reproduce the original index to within floating point
(per-row cosine 1.000000).

### Step 3 — Pre-compute paper embeddings (from `papers.csv`)

```bash
python scripts/precompute_embeddings.py
```
Writes `data/paper_embeddings.npy` and `data/paper_ids.json` in FP32 (~7 s on an RTX 5080).

After these three steps the dataset is fully reconstructed and the evaluation can be run.

### Determinism

The pipeline is deterministic by construction: the evaluation contains no random seeds,
sampling, or shuffling; FAISS exact inner-product search, Neo4j traversal, and all rankers
break ties deterministically; embeddings are computed in **FP32** (the hardware-dependent FP16
fast path was removed). The only residual variation is floating-point neural inference across
different GPUs (~1e-6), which is below the granularity that changes any reported metric.

### Checksums of the canonical inputs

```
ae60ddbf734cf5f4856a6747d4be4900d60e7afb75b17bc1816c7a591291c233  data/papers.csv
193d0b4632c77768313dd49a01e3d13914afd37e6f70a98ac76f553b0e8871b8  data/citations.csv
cd9c2f1d063a999e466139db76722a37c9c50c1bc56a50291e2c6c3e85ceed7c  data/written_by.csv
ddb69161e8694ce468e1fd19022947e3a8a22cb8e1408898c26b6d2c52143b47  data/write_together.csv
cc8f57972cb90afe41ca405973370225fbd591220fb0668d4ce19d4eecbe0e5c  data/written_for.csv
9e49108e4efc67e039525a13d0344639ce1069686b370d20d80ff6190342d077  data/field_of_study.csv
109e54aac02790b8f5ea5b06e4700574e63c3014b7c35516535c652da0476b59  data/affiliations.csv
76be47b06925a6655dbe2bb9dd2a8ea73b49cc6c4d68d3222a4fe4f9899b908c  data/queries.csv
8912714c28e6220cd82eba87a6c1b7c4b4dd607cc768a75506dd4e39ba193d34  data/ground_truth_relevance.csv
25f1bc2c5e3e4cf275ad3b01f28f1275216bf6436a6e1cb90881f2df3b0fc475  data/ground_truth_papers_250.csv
d566c89d34dbeb732d141ce5d8b13e12b84901cf9ba7493f3645701e3dabb281  data/vector_chunks.csv.gz
```

### (Optional) Rebuilding from scratch via the live API

`ingestion/run_ingestion.py` documents how the corpus was originally grown by citation BFS from
the seed (it requires Semantic Scholar API keys). It is retained for provenance and **will not**
reproduce this dataset byte-for-byte: Semantic Scholar's data drifts continuously (citation
lists, abstracts, and open-access PDFs change) and the API does not guarantee citation ordering.
For the exact data used here, use the committed files and Steps 1–3 above.

---

## Running the evaluation

With the graph (Step 1), vector index (Step 2), and embeddings (Step 3) in place, and Neo4j
running:

```bash
python evaluation/final_evaluation.py      # → results/final_evaluation_sweep.csv
python scripts/visualize_results.py        # → results/figures/
```

The sweep matches each method's mean candidate pool to a target **budget** (swept 100–300 in
steps of 25) by binary-searching the graph fan-out `GRAPH_K` to within ±5% of the target, then
evaluates every vector/graph **mix** at that budget. Retrieval results are disk-cached to
`results/eval_cache/` (gitignored), so re-runs complete in seconds; delete that directory to
force a cold run.

### Methods

- **Vector** — FAISS top-*N* by cosine similarity (best-chunk-per-paper). The "equivalent
  semantic search."
- **Bfs** (query-blind hybrid) — from each vector seed, four breadth-first channels (citation,
  co-author, venue, field) round-robin interleaved.
- **MetaPath** (query-aware hybrid) — greedy best-first expansion over four undirected
  meta-paths (P–P via CITES, P–A–P via shared author, P–V–P via shared venue, P–F–P via shared
  field), with a single Papers-only frontier ordered by cosine similarity to the query.
- **Re-ranker** — a cross-encoder (`ms-marco-MiniLM-L-6-v2`), held constant across all methods;
  its input per candidate is the paper's **title + abstract** (truncated to the token limit).
  Two ranker-free controls (frequency, interleave) isolate retrieval from ranking.

### Metrics

- **MRR@5** — mean reciprocal rank at cutoff 5 (top-rank quality)
- **Recall@5 / Recall@10** — coverage
- **Relevant-in-Pool** — relevant papers in the pool *before* ranking

---

## Hardware

- CPU: Intel Ultra 9 275HX (32 cores) — parallel Neo4j queries via `ThreadPoolExecutor`
- GPU: NVIDIA RTX 5080 Mobile (16 GB VRAM) — SciBERT encoding, cross-encoder re-ranking
- RAM: 32 GB

## Dependencies

See `requirements.txt`. Key packages: `torch` (CUDA), `sentence-transformers`, `faiss-cpu`,
`neo4j`, `numpy`, `tqdm`, `matplotlib`. `pymilvus[milvus_lite]` (Linux/WSL2) is needed only for
the optional provenance script `extract_chunks.py` and the original API ingestion — not for
recreating the dataset from committed files.

---

## Data availability

| Artifact | In repo | How to obtain |
|---|---|---|
| `data/*.csv`, `data/vector_chunks.csv.gz` | ✅ | The canonical committed dataset |
| Neo4j graph | ❌ | `scripts/build_graph.py` (from CSVs) |
| `data/vectors.npy`, `vector_paperids.json` | ❌ | `scripts/build_vectors.py` (from `vector_chunks.csv.gz`) |
| `data/paper_embeddings.npy`, `paper_ids.json` | ❌ | `scripts/precompute_embeddings.py` (from `papers.csv`) |
| `RAG.db`, `transfer/neo4j.dump` | ❌ | Not needed — superseded by committed artifacts |
| `results/final_evaluation_sweep.csv`, `results/figures/` | ✅ | Committed; regenerate via the evaluation steps |
