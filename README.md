# IAAIR — Hybrid Graph-Vector RAG Evaluation

**Information Access and AI Retrieval**
A peer-reviewed research evaluation of hybrid (graph + vector) retrieval-augmented generation systems for scientific paper search.

**Author:** Hari Nalamotu | Institute of Applied Artificial Intelligence and Robotics (IAAIR)

---

## Overview

This repository contains the full experimental pipeline for evaluating hybrid retrieval systems that combine **FAISS vector search** (SciBERT embeddings) with **Neo4j graph traversal** over a 1,500-paper scientific corpus. Results are compared across 13 retrieval configurations using 250 manually annotated ground-truth papers across 10 queries.

The key research question: *does augmenting vector retrieval with a citation/authorship/venue knowledge graph improve retrieval quality for scientific papers?*

**Main finding:** Query-aware metapath graph search with a CrossEncoder reranker consistently achieves perfect MRR@5 (1.0) across all tested pool sizes, while pure vector search with reranking maintains higher recall. The reranker is the decisive factor — without it, all graph-augmented methods underperform pure vector search.

---

## Repository Structure

```
IAAIR/
├── data/                           # Dataset (CSVs committed; large binaries gitignored)
│   ├── papers.csv                  # 1,500 papers with abstracts
│   ├── queries.csv                 # 10 evaluation queries
│   ├── ground_truth_papers_250.csv # 250 manually judged candidate papers
│   ├── ground_truth_relevance.csv  # 2,500 binary relevance labels (250 × 10 queries)
│   ├── authors.csv / venues.csv / citations.csv / ...  # Graph edge data
│   └── dataset_formation.ipynb    # Ground truth construction notebook
│
├── evaluation/
│   ├── run_evaluation.py           # Fixed-config evaluation (13 retrieval configurations)
│   └── final_evaluation.py        # Budget × VEC sweep evaluation
│
├── retrieval/
│   ├── query_aware_graph.py        # MetaPathBestFirstGraph + GreedyBestFirstGraph
│   └── run_hybrid_frequency.py    # Standalone frequency-based hybrid ranker
│
├── ingestion/
│   └── run_ingestion.py           # Data collection pipeline (Semantic Scholar API)
│
├── analysis/                       # Parameter sweep scripts
│
├── scripts/
│   ├── setup_env.py               # One-time environment setup (Windows)
│   ├── extract_vectors.py         # One-time FAISS vector extraction (WSL2)
│   ├── precompute_embeddings.py   # GPU-accelerated paper embedding cache
│   ├── train_ranker.py            # FFNN ranker training (optional)
│   └── visualize_results.py       # Generate all result figures
│
├── results/
│   ├── final_evaluation_sweep.csv  # Main sweep results (budget × VEC × config)
│   ├── evaluation_results_summary.csv
│   ├── evaluation_results_detail.csv
│   ├── ranker_*.json / *.pt / *.csv  # Trained ranker outputs
│   └── figures/                   # All generated plots
│
├── .env.example                   # Template for required environment variables
└── requirements.txt               # Python dependencies
```

---

## Corpus

- **Seed paper:** `649def34f8be52c8b66281af98ae884c09aef38b` (primary RAG survey)
- **Expansion:** BFS via Semantic Scholar API, 1,500 papers
- **Topics:** RAG systems, knowledge graphs, NLP, information extraction, LLMs
- **Ground truth:** 250 papers manually relevance-judged across 10 queries (2,500 labels)
- **Vector index:** 56,379 SciBERT chunk embeddings (abstracts + PDF full-text)
- **Graph:** 1,500 Papers, 8,071 Authors, 618 Venues, 10 Fields of Study

---

## Environment Setup

### Requirements

- Windows 11 with WSL2 (Ubuntu 24.04, systemd enabled)
- NVIDIA GPU with CUDA (tested: RTX 5080 Mobile, CUDA 13)
- 16+ GB RAM recommended for the full sweep

### Step 1 — Configure credentials

```bash
cp .env.example .env
# Edit .env with your Neo4j password and (optionally) Semantic Scholar API keys
```

### Step 2 — Windows setup (Neo4j + Python venv)

Run from a Windows PowerShell:
```powershell
C:\Users\<user>\anaconda3\python.exe scripts\setup_env.py
```

This installs Java 21, downloads Neo4j 5, imports the graph from CSVs, creates `.venv/`, and installs all Python packages.

### Step 3 — Extract FAISS vectors (WSL2, once only)

```bash
# WSL2 terminal — requires pymilvus[milvus_lite]
conda activate <env_with_milvus>
python /mnt/c/<path>/scripts/extract_vectors.py
```

Writes `data/vectors.npy` (56,379 × 768 float32) and `data/vector_paperids.json`.

### Step 4 — Pre-compute paper embeddings

```bash
# WSL2 terminal
conda activate torchtest
python /mnt/c/<path>/scripts/precompute_embeddings.py
```

Writes `data/paper_embeddings.npy` (~3 s on RTX 5080).

---

## Reproducing Results

### Fixed-config evaluation (13 configurations)

```bash
# WSL2 terminal (Neo4j must be running)
conda activate torchtest
python /mnt/c/<path>/evaluation/run_evaluation.py
```

Outputs:
- `results/evaluation_results_summary.csv`
- `results/evaluation_results_detail.csv`

To run a single configuration:
```bash
python evaluation/run_evaluation.py --only metapath_nn
```

### Budget × VEC sweep (main paper results)

```bash
conda activate torchtest
python /mnt/c/<path>/evaluation/final_evaluation.py
```

Sweeps 11 pool-size budgets (100–350, step 25) × all VEC fractions, auto-tuning GRAPH_K to hit each target pool size within ±5%. Results append incrementally to `results/final_evaluation_sweep.csv` — resumable after interruption.

### Generate figures

```bash
conda activate torchtest
python /mnt/c/<path>/scripts/visualize_results.py
```

Outputs 9 publication-ready figures to `results/figures/`.

### Train the neural ranker (optional)

```bash
conda activate torchtest
python /mnt/c/<path>/scripts/train_ranker.py
```

Requires the sweep to have been run first (uses cached scores). Outputs:
- `results/ranker_model.pt`
- `results/ranker_scaler.json`
- `results/ranker_loo_results.csv`

---

## Retrieval Configurations

### Fixed-config evaluation

| Config | Retrieval | Ranker | VEC | GRAPH_K |
|---|---|---|---|---|
| `hybrid_reranker` | Vector + multi-channel BFS | CrossEncoder | 60 | 9 |
| `hybrid_freq_no_reranker` | Vector + BFS | Frequency | 60 | 9 |
| `hybrid_interleave_no_reranker` | Vector + BFS | Interleave | 60 | 9 |
| `b800_hybrid_reranker` | Vector + BFS | CrossEncoder | 60 | 9 |
| `b800_hybrid_freq` | Vector + BFS | Frequency | 60 | 9 |
| `b800_hybrid_interleave` | Vector + BFS | Interleave | 60 | 9 |
| `metapath_hybrid_reranker` | Vector + MetaPath BFS | CrossEncoder | 200 | 5 |
| `metapath_hybrid_freq` | Vector + MetaPath BFS | Frequency | 200 | 5 |
| `metapath_hybrid_interleave` | Vector + MetaPath BFS | Interleave | 200 | 5 |
| `vector_reranker` | Vector only | CrossEncoder | 60 | — |
| `vector_only` | Vector only | None | 60 | — |
| `vector_poolmatch_reranker` | Vector (pool-matched) | CrossEncoder | 225 | — |
| `vector_poolmatch_only` | Vector (pool-matched) | None | 225 | — |

### Graph Retrieval Algorithms

**Standard BFS (`PaperBFS`):** Four independent channels (CITES, co-author, venue, field), round-robin interleaved to prevent high-degree edges dominating.

**Query-Aware MetaPath BFS (`MetaPathBestFirstGraph`):** Greedy best-first search over all four meta-path types, scored by cosine similarity to the query. Always expands the globally most query-relevant candidate.

### Evaluation Metrics

- **MRR@5** — Mean Reciprocal Rank at cutoff 5
- **Recall@5** — Fraction of relevant papers found in top 5
- **Recall@10** — Fraction of relevant papers found in top 10
- **Relevant in Pool** — Total relevant papers in the retrieval pool (before reranking)

---

## Key Results

All results are in `results/final_evaluation_sweep.csv` and `results/figures/`.

**At optimal configuration (best VEC fraction per method):**

| Method | MRR@5 | Recall@10 | Rel/Pool |
|---|---|---|---|
| metapath + CrossEncoder | **1.000** (all budgets) | Wins 6/9 budgets | Lower |
| vector + CrossEncoder | 0.857–0.950 | Wins 2–3/9 budgets | **Wins 9/9 budgets** |
| BFS + CrossEncoder | 0.875–0.950 | Mixed | Mixed |

**On average across all VEC fractions:**

| Method | MRR@5 | Recall@10 | Rel/Pool |
|---|---|---|---|
| metapath + CrossEncoder | Wins 5/9 budgets | Loses all | Loses all |
| vector + CrossEncoder | Wins 4/9 budgets | **Wins all** | **Wins all** |

**The reranker is the decisive factor.** Without CrossEncoder reranking, all graph-augmented methods score ~0.50 MRR@5, below the pure vector baseline.

---

## Caching

All retrieval results are cached to `results/eval_cache/` (gitignored). On first run, Neo4j and FAISS are queried. Subsequent runs load from cache — evaluation runs in seconds. Delete `results/eval_cache/` to force a full re-run.

---

## Hardware Used

- CPU: Intel Ultra 9 275HX (32 cores) — parallel Neo4j queries via `ThreadPoolExecutor`
- GPU: NVIDIA RTX 5080 Mobile (16 GB VRAM) — SciBERT encoding, CrossEncoder reranking
- RAM: 32 GB — full cache loaded into memory at startup

---

## Dependencies

See `requirements.txt`. Key packages:

```
torch>=2.11  (CUDA 12.8)
sentence-transformers>=5.0
faiss-cpu
neo4j>=5.0
pymilvus[milvus_lite]   (WSL2 only, for extract_vectors.py)
tqdm
matplotlib
```

Install in the project venv:
```bash
pip install -r requirements.txt
```

---

## Data Availability

| File | In repo | How to obtain |
|---|---|---|
| `data/*.csv` | ✅ Yes | Included |
| `data/vectors.npy` | ❌ Gitignored | Run `scripts/extract_vectors.py` |
| `data/paper_embeddings.npy` | ❌ Gitignored | Run `scripts/precompute_embeddings.py` |
| `RAG.db` (Milvus backup) | ❌ Gitignored | Original; needed for extract_vectors.py |
| `transfer/neo4j.dump` | ❌ Gitignored | Original; or import from CSVs via setup_env.py |
| `results/final_evaluation_sweep.csv` | ✅ Yes | Included |
| `results/figures/` | ✅ Yes | Included; regenerate with `scripts/visualize_results.py` |
