# IAAIR — Information Access and AI Retrieval

A publishable research project evaluating **hybrid (graph + vector) RAG systems** for scientific paper retrieval. The system combines Milvus vector search (SciBERT embeddings) with Neo4j graph traversal over a 1500-paper citation corpus, then compares multiple retrieval configurations using manually annotated ground truth.

---

## Repository Structure

```
IAAIR/
├── RAG.db                          # Milvus Lite vector database (56,379 vectors)
├── semantic_scholar_corpus/pdfs/   # 380 full-text PDFs
│
├── ingestion/                      # Data collection & ingestion
│   ├── run_ingestion.py            # Main ingestion pipeline (Semantic Scholar → Milvus + Neo4j)
│   ├── recover_missing_chunks.py   # Re-embeds papers that lost vector chunks
│   ├── expand_ground_truth.py      # Ground truth dataset expansion utilities
│   └── ingestion_state.json        # Checkpoint: 1500 seen paper IDs, BFS queue state
│
├── retrieval/                      # Retrieval algorithm implementations
│   ├── query_aware_graph.py        # Two graph retrieval classes:
│   │                               #   MetaPathBestFirstGraph — greedy best-first, query-aware,
│   │                               #     expands via meta-paths (CITES, co-author, venue, field)
│   │                               #   GreedyBestFirstGraph   — CITES-only version (baseline)
│   └── run_hybrid_frequency.py     # Standalone frequency-based hybrid ranker
│
├── evaluation/                     # Evaluation scripts
│   ├── run_evaluation.py           # MAIN EVALUATION — runs all retrieval configs, outputs metrics
│   └── hybrid_sweep.py             # Sweeps VEC/GRAPH_K to calibrate pool sizes
│
├── analysis/                       # Parameter sweep & diagnostic scripts
│   ├── analyse_channels.py         # Tracks relevant paper contribution per BFS channel
│   ├── channel_sweep.py            # Sweeps VEC↓/GRAPH_K↑ at variable budget
│   ├── channel_sweep_constant.py   # Sweeps VEC↓/GRAPH_K↑ at constant pre-dedup budget (~1200)
│   ├── plot_constant_sweep.py      # Plots results from channel_sweep_constant
│   ├── pool_comparison_sweep.py    # Hybrid vs vector pool-size comparison sweep
│   ├── param_sweep.py              # General parameter sweep utility
│   └── test_vector_db.py           # Validates Milvus vector DB integrity
│
├── data/                           # All dataset files
│   ├── papers.csv                  # 1500 papers (id, title, abstract, year, …)
│   ├── queries.csv                 # 10 evaluation queries
│   ├── ground_truth_papers_250.csv # 250 manually selected candidate papers
│   ├── ground_truth_relevance.csv  # 2500 rows: 250 papers × 10 queries, binary relevance label
│   ├── authors.csv                 # Author metadata
│   ├── venues.csv                  # Venue metadata
│   ├── citations.csv               # Paper→Paper citation edges
│   ├── written_by.csv              # Paper→Author edges
│   ├── write_together.csv          # Author→Author co-author edges
│   ├── written_for.csv             # Paper→Venue edges
│   ├── field_of_study.csv          # Paper→FieldOfStudy edges
│   ├── affiliations.csv            # Author→Institution edges
│   ├── institutions.csv            # Institution nodes
│   └── dataset_formation.ipynb     # Notebook used to build & label ground truth dataset
│
└── results/                        # All output CSVs and plots
    ├── evaluation_results_summary.csv   # Per-config averaged metrics
    ├── evaluation_results_detail.csv    # Per-query per-config metrics (70+ rows)
    ├── hybrid_vec_sweep.csv             # VEC sweep results
    ├── channel_sweep_results.*          # Variable-budget channel sweep
    ├── channel_sweep_constant_results.* # Constant-budget channel sweep (~1000)
    ├── channel_sweep_1200_results.*     # Constant-budget channel sweep (~1200)
    ├── constant_budget_sweep.png        # Summary chart of budget sweep
    ├── hybrid_sweep_results.png         # Hybrid parameter sweep chart
    └── pool_comparison_sweep.csv        # Hybrid vs vector pool-size sweep
```

---

## Databases

### Milvus (`RAG.db`)
- **Type**: Milvus Lite (file-based, single-process lock)
- **Collection**: `ingestion_v0`
- **Vectors**: 56,379 total (papers embedded as chunks; papers with PDFs have ~70× more chunks than abstract-only papers)
- **Model**: `jordyvl/scibert_scivocab_uncased_sentence_transformer` (SciBERT, 768-dim)
- **Note**: Only one process can open `RAG.db` at a time. Kill any running script before starting another.

### Neo4j
- **URI**: `bolt://localhost:7687`
- **User**: `neo4j` / **Pass**: `Thammu123`
- **Node types**: `Paper`, `Author`, `Venue`, `FieldOfStudy`, `Institution`
- **Relationship types**: `CITES`, `WROTE`, `PUBLISHED_IN`, `HAS_FIELD`, `AFFILIATED_WITH`
- **Scale**: 1500 Paper nodes, 8071 Author nodes, 618 Venue nodes, 10 FieldOfStudy nodes
- **Relationship counts**: CITES=5878, WROTE=18732, PUBLISHED_IN=2774, HAS_FIELD=2718, AFFILIATED_WITH=890

---

## Corpus

- **Seed paper**: `649def34f8be52c8b66281af98ae884c09aef38b` — the primary RAG survey paper
- **Expansion**: BFS via Semantic Scholar API from seed, targeting 1500 papers
- **Topic**: Scientific RAG, knowledge graphs, information extraction, NLP, LLMs
- **Ground truth**: 250 papers manually selected and relevance-judged across 10 queries

---

## Evaluation

### How to run
```bash
/opt/anaconda3/envs/iaair2/bin/python3 evaluation/run_evaluation.py
```

### Python environment
```bash
# Conda env is broken (binary missing) but packages still work:
/opt/anaconda3/envs/iaair2/bin/python3   # always use this, never plain `python`
```

### Metrics
- **MRR@5** — Mean Reciprocal Rank at 5
- **Recall@5** — Recall at 5
- **Recall@10** — Recall at 10
- **Rel in Pool** — Total relevant papers present in the retrieval pool (before reranking)
- **Avg Pool** — Average deduped pool size per query

### Retrieval Configurations in `run_evaluation.py`

All configs share the same QUERIES (10) and ground truth (250 papers, binary labels per query).

| Config name | Method | VEC | GRAPH_K | Pre-dedup budget |
|---|---|---|---|---|
| `hybrid_reranker` | Vector + multi-channel BFS → CrossEncoder rerank | 60 | 15 | ~960 |
| `hybrid_freq_no_reranker` | Vector + BFS → frequency ranking | 60 | 15 | ~960 |
| `hybrid_interleave_no_reranker` | Vector + BFS → interleaved ranking | 60 | 15 | ~960 |
| `b800_hybrid_reranker` | Vector + BFS → CrossEncoder rerank | 80 | 14 | 1200 |
| `b800_hybrid_freq` | Vector + BFS → frequency ranking | 80 | 14 | 1200 |
| `b800_hybrid_interleave` | Vector + BFS → interleaved ranking | 80 | 14 | 1200 |
| `metapath_hybrid_reranker` | Vector + **query-aware** meta-path BFS → rerank | 60 | 15 | ~960 |
| `metapath_hybrid_freq` | Vector + query-aware BFS → frequency ranking | 60 | 15 | ~960 |
| `metapath_hybrid_interleave` | Vector + query-aware BFS → interleaved ranking | 60 | 15 | ~960 |
| `vector_reranker` | Vector only → CrossEncoder rerank | 60 | — | 60 |
| `vector_only` | Vector only, no reranking | 60 | — | 60 |
| `vector_poolmatch_reranker` | Vector (pool-matched to hybrid size) → rerank | ~300 | — | ~300 |
| `vector_poolmatch_only` | Vector (pool-matched), no reranking | ~300 | — | ~300 |

### Key tunable parameters (top of `run_evaluation.py`)
```python
HYBRID_VEC = 60   # vector seeds for standard hybrid
GRAPH_K    = 15   # BFS neighbors per seed (multi-channel, round-robin interleaved)
BUDGET_VEC = 80   # vector seeds for budget hybrid
BUDGET_GK  = 14   # BFS neighbors for budget hybrid
METAPATH_VEC = 60 # vector seeds for query-aware meta-path hybrid
METAPATH_GK  = 15 # neighbors per seed for meta-path hybrid
```

### Reranker
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranking is applied **only to ground truth papers** (250 judged papers) within the pool
- Returns top-10 after scoring

---

## Graph Retrieval Algorithms

### Standard BFS (`PaperBFS` in `run_evaluation.py`)
Multi-channel breadth-first search with round-robin interleaving across 4 channels:
1. **CITES** — direct citation edges between Papers
2. **Co-author** — Paper → WROTE → Author → WROTE → Paper
3. **Venue** — Paper → PUBLISHED_IN → Venue → PUBLISHED_IN → Paper
4. **Field** — Paper → HAS_FIELD → FieldOfStudy → HAS_FIELD → Paper

Each channel runs its own independent BFS so high-degree edges (CITES has ~430 neighbors for the seed paper) cannot starve other channels. Results are round-robin interleaved: 1 paper from each channel, rotating, until k papers are collected.

### Query-Aware Meta-Path BFS (`MetaPathBestFirstGraph` in `retrieval/query_aware_graph.py`)
Greedy best-first search guided by cosine similarity between candidate papers and the query:
- Uses a single max-heap (priority queue) keyed on cosine similarity
- At each expansion, collects all candidates reachable via any of the 4 meta-paths in one Cypher `UNION` query
- Always expands the globally most-similar candidate next
- Requires pre-computed paper embeddings (title + abstract, embedded once at startup)
- Equal weights across all meta-path types

---

## Known Issues / Experimental Notes

1. **PDF chunk bias**: 380 papers have full-text PDFs (up to 813 chunks); 1120 papers have only 1 abstract chunk. Vector search heavily favours PDF papers regardless of relevance.

2. **Field channel noise**: `FieldOfStudy` is too coarse (only 10 fields, mostly "Computer Science"). The field channel finds zero relevant papers in most queries and adds noise to the pool.

3. **Relevance rate by source** (from channel analysis at VEC=60, GRAPH_K=15):
   - Vector: 5.3% relevance rate — most efficient channel
   - Co-author: 2.8%
   - Venue: 2.4%
   - CITES: 1.9%
   - Field: 0.0%

4. **Optimal config from sweeps**: At constant pre-dedup budget ~1200, VEC=200/GRAPH_K=4 finds the most relevant papers (91 across 10 queries). Higher graph weight consistently loses relevant papers because vector precision (5.3%) far exceeds graph precision (~2%).

5. **Statistical significance**: Only 10 queries — results are directionally useful but not statistically significant for publication. Consider expanding to 50+ queries.

---

## Ingestion Pipeline

To re-run ingestion from scratch (or extend the corpus):
```bash
/opt/anaconda3/envs/iaair2/bin/python3 ingestion/run_ingestion.py
```

State is checkpointed in `ingestion/ingestion_state.json`. The pipeline:
1. BFS from seed paper via Semantic Scholar API
2. Chunks text (abstracts + PDF full-text) with tiktoken (128 tokens, 26 overlap)
3. Embeds chunks with SciBERT, inserts into Milvus
4. Exports CSVs to `data/`
5. Imports all nodes and relationships into Neo4j
