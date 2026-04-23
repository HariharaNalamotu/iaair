# IAAIR — Information Access and AI Retrieval

A publishable research project evaluating **hybrid (graph + vector) RAG systems** for scientific paper retrieval. The system combines FAISS vector search (SciBERT embeddings) with Neo4j graph traversal over a 1500-paper citation corpus, then compares multiple retrieval configurations using manually annotated ground truth.

---

## Platform

**Windows 11 native** with one WSL2 step (vector extraction, run once).

- **Python**: project venv at `.venv/` (Python 3.13, Windows native)
- **Vector search**: FAISS CPU (`data/vectors.npy`, extracted from `RAG.db` once)
- **Graph DB**: Neo4j 5 running as a Windows service (`bolt://localhost:7687`)
- **GPU**: RTX 5080 used for SciBERT embedding (PyTorch CUDA 12.8)
- **`RAG.db`**: kept as backup; not used at runtime

---

## Repository Structure

```
IAAIR/
├── .venv/                          # Project Python venv (Windows, gitignored)
├── RAG.db                          # Milvus Lite backup (56,379 vectors) — not used at runtime
├── semantic_scholar_corpus/pdfs/   # 380 full-text PDFs
│
├── ingestion/                      # Data collection & ingestion
│   ├── run_ingestion.py            # Main ingestion pipeline (Semantic Scholar → Milvus + Neo4j)
│   ├── recover_missing_chunks.py   # Re-embeds papers that lost vector chunks
│   ├── expand_ground_truth.py      # Ground truth dataset expansion utilities
│   └── ingestion_state.json        # Checkpoint: 1500 seen paper IDs, BFS queue state
│
├── retrieval/                      # Retrieval algorithm implementations
│   ├── query_aware_graph.py        # MetaPathBestFirstGraph + GreedyBestFirstGraph
│   └── run_hybrid_frequency.py     # Standalone frequency-based hybrid ranker
│
├── evaluation/                     # Evaluation scripts
│   ├── run_evaluation.py           # MAIN EVALUATION — runs all configs, outputs metrics
│   └── hybrid_sweep.py             # Sweeps VEC/GRAPH_K to calibrate pool sizes
│
├── analysis/                       # Parameter sweep & diagnostic scripts
│   ├── analyse_channels.py
│   ├── channel_sweep.py
│   ├── channel_sweep_constant.py
│   ├── plot_constant_sweep.py
│   ├── pool_comparison_sweep.py
│   ├── param_sweep.py
│   └── test_vector_db.py
│
├── scripts/                        # Setup & maintenance utilities
│   ├── setup_env.py                # ONE-TIME Windows setup (Java, Neo4j, venv, packages)
│   ├── extract_vectors.py          # ONE-TIME WSL2 step: RAG.db → data/vectors.npy
│   ├── precompute_embeddings.py    # GPU SciBERT encoding for all 1500 papers
│   └── verify_setup.py             # Sanity-check all components
│
├── data/                           # All dataset files
│   ├── papers.csv                  # 1500 papers (id, title, abstract, year, …)
│   ├── queries.csv                 # 10 evaluation queries
│   ├── ground_truth_papers_250.csv # 250 manually judged candidate papers
│   ├── ground_truth_relevance.csv  # 2500 rows: 250 papers × 10 queries, binary label
│   ├── vectors.npy                 # (56379, 768) float32 — FAISS search index data
│   ├── vector_paperids.json        # Paper ID for each row in vectors.npy
│   ├── paper_embeddings.npy        # (1500, 768) float32 — per-paper title+abstract embeds
│   ├── paper_ids.json              # Paper ID list matching paper_embeddings.npy rows
│   ├── authors.csv / venues.csv / citations.csv / ...
│   └── dataset_formation.ipynb
│
├── transfer/                       # One-time transfer (not needed after setup)
│   └── neo4j.dump
│
└── results/                        # Output CSVs and plots
    ├── evaluation_results_summary.csv
    ├── evaluation_results_detail.csv
    └── ...
```

---

## Setup (one-time)

### Step 1 — Windows setup

Run from a **Windows PowerShell** (as Administrator for the Neo4j service install):

```powershell
C:\Users\harih\anaconda3\python.exe scripts\setup_env.py
```

This installs Java 21 (winget), downloads Neo4j 5 for Windows, restores the database dump, starts Neo4j as a Windows service, creates `.venv/`, installs all Python packages, and pre-computes paper embeddings on the RTX 5080.

### Step 2 — Extract FAISS vectors (WSL2, once only)

Open a **WSL2 terminal** and run:

```bash
# Use any WSL2 Python that has pymilvus[milvus_lite]
python3 /mnt/c/Users/harih/hybrid-graphrag/scripts/extract_vectors.py
```

This reads `RAG.db` and writes `data/vectors.npy` + `data/vector_paperids.json`. After this, WSL2 is never needed again.

---

## Running Evaluation

From **Windows** (PowerShell or CMD), with Neo4j service running:

```powershell
.venv\Scripts\python.exe evaluation\run_evaluation.py
```

---

## Databases

### FAISS (`data/vectors.npy`)
- **56,379 vectors**, 768-dim float32 (SciBERT, cosine similarity via IndexFlatIP)
- Extracted once from `RAG.db`, runs natively on Windows
- Search time: ~5–20 ms per query on CPU

### Neo4j (Windows service)
- **URI**: `bolt://localhost:7687` / **User**: `neo4j` / **Pass**: `Thammu123`
- **Node types**: `Paper`, `Author`, `Venue`, `FieldOfStudy`, `Institution`
- **Relationship types**: `CITES`, `WROTE`, `PUBLISHED_IN`, `HAS_FIELD`, `AFFILIATED_WITH`
- **Scale**: 1500 Papers, 8071 Authors, 618 Venues, 10 FieldOfStudy nodes
- **Relationship counts**: CITES=5878, WROTE=18732, PUBLISHED_IN=2774, HAS_FIELD=2718

### RAG.db (backup)
- Original Milvus Lite database kept at project root
- Not used at runtime; restore with `scripts/extract_vectors.py` if needed

---

## Paper Embedding Cache

`data/paper_embeddings.npy` — 1500 × 768 float32, pre-computed SciBERT embeddings for all papers (title + abstract). Loaded at startup by `run_evaluation.py` for the `MetaPathBestFirstGraph`. Generated in ~3 s on the RTX 5080 by `scripts/precompute_embeddings.py`.

---

## Corpus

- **Seed paper**: `649def34f8be52c8b66281af98ae884c09aef38b`
- **Expansion**: BFS via Semantic Scholar API, 1500 papers
- **Topic**: Scientific RAG, knowledge graphs, NLP, LLMs
- **Ground truth**: 250 papers × 10 queries, binary relevance (2500 judgements)

---

## Evaluation

### Metrics
- **MRR@5**, **Recall@5**, **Recall@10**, **Rel in Pool**, **Avg Pool**

### Retrieval Configurations

| Config | Method | VEC | GRAPH_K |
|---|---|---|---|
| `hybrid_reranker` | Vector + BFS → CrossEncoder | 60 | 15 |
| `hybrid_freq_no_reranker` | Vector + BFS → frequency rank | 60 | 15 |
| `hybrid_interleave_no_reranker` | Vector + BFS → interleave | 60 | 15 |
| `b800_hybrid_reranker` | Vector + BFS → CrossEncoder | 80 | 14 |
| `b800_hybrid_freq` | Vector + BFS → frequency rank | 80 | 14 |
| `b800_hybrid_interleave` | Vector + BFS → interleave | 80 | 14 |
| `metapath_hybrid_reranker` | Vector + query-aware BFS → CrossEncoder | 200 | 20 |
| `metapath_hybrid_freq` | Vector + query-aware BFS → frequency | 200 | 20 |
| `metapath_hybrid_interleave` | Vector + query-aware BFS → interleave | 200 | 20 |
| `vector_reranker` | Vector only → CrossEncoder | 60 | — |
| `vector_only` | Vector only | 60 | — |
| `vector_poolmatch_reranker` | Vector (pool-matched) → CrossEncoder | ~300 | — |
| `vector_poolmatch_only` | Vector (pool-matched) | ~300 | — |

### Reranker
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Applied only to ground-truth papers (250) within the pool; returns top-10

---

## Graph Retrieval Algorithms

### Standard BFS (`PaperBFS`)
Four independent BFS channels round-robin interleaved: CITES, co-author, venue, field.

### Query-Aware Meta-Path BFS (`MetaPathBestFirstGraph`)
Greedy best-first search over all four meta-paths, scored by cosine similarity to the query. Uses pre-computed `data/paper_embeddings.npy` — loaded instantly at startup.

---

## Known Issues / Experimental Notes

1. **PDF chunk bias**: 380 PDF papers dominate vector search (up to 813 chunks vs 1 for abstract-only).
2. **Field channel noise**: `FieldOfStudy` is too coarse (10 fields, mostly "Computer Science") — finds 0 relevant papers in most queries.
3. **Relevance rate by source**: Vector 5.3% > Co-author 2.8% > Venue 2.4% > CITES 1.9% > Field 0.0%
4. **Optimal config**: At ~1200 pre-dedup budget, VEC=200/GRAPH_K=4 maximises relevant papers found.
5. **Statistical significance**: 10 queries — directional, not publication-ready. Consider expanding to 50+.

---

## Ingestion Pipeline

To extend the corpus:

```powershell
.venv\Scripts\python.exe ingestion\run_ingestion.py
```
