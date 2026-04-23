# IAAIR — Information Access and AI Retrieval

A publishable research project evaluating **hybrid (graph + vector) RAG systems** for scientific paper retrieval. The system combines FAISS vector search (SciBERT embeddings) with Neo4j graph traversal over a 1500-paper citation corpus, then compares multiple retrieval configurations using manually annotated ground truth.

---

## Platform

**Windows 11 + WSL2 (Ubuntu 24.04).**

| Component | Where it runs | Details |
|---|---|---|
| Evaluation Python | WSL2 — `conda activate torchtest` | PyTorch 2.11 + CUDA 13, faiss-cpu, neo4j, sentence-transformers |
| Neo4j | WSL2 — systemd user service | Rebuilt from CSVs; bolt on `0.0.0.0:7687` |
| FAISS index | Windows filesystem | `data/vectors.npy`, loaded by evaluation at startup |
| GPU (SciBERT / CrossEncoder) | RTX 5080 via CUDA WSL2 passthrough | CUDA 13 driver; PyTorch cu128 |
| RAG.db | Windows filesystem (backup only) | Original Milvus Lite DB; not used at runtime |

WSL2 mirrored networking is enabled (`~/.wslconfig`), so Neo4j at `bolt://localhost:7687` is also reachable from Windows Python if a firewall rule is added (see below).

---

## Repository Structure

```
IAAIR/
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
│   ├── run_evaluation.py           # MAIN — runs all 13 retrieval configs, outputs metrics
│   └── hybrid_sweep.py             # Sweeps VEC/GRAPH_K to calibrate pool sizes
│
├── analysis/                       # Parameter sweep & diagnostic scripts
│   ├── analyse_channels.py         # Relevant-paper contribution per BFS channel
│   ├── channel_sweep.py            # Variable-budget channel sweep
│   ├── channel_sweep_constant.py   # Constant-budget channel sweep (~1200)
│   ├── plot_constant_sweep.py      # Plots channel_sweep_constant results
│   ├── pool_comparison_sweep.py    # Hybrid vs vector pool-size comparison
│   ├── param_sweep.py              # General parameter sweep utility
│   └── test_vector_db.py           # Validates vector DB integrity
│
├── scripts/                        # Setup & maintenance utilities
│   ├── setup_env.py                # Windows setup: Java, Neo4j zip, .venv, packages
│   ├── extract_vectors.py          # WSL2 one-time: RAG.db → data/vectors.npy
│   ├── precompute_embeddings.py    # GPU SciBERT encoding for all 1500 papers
│   └── verify_setup.py             # Full sanity-check (GPU, FAISS, Neo4j, cache, CSVs)
│
├── data/                           # All dataset files (large generated files gitignored)
│   ├── papers.csv                  # 1500 papers (id, title, abstract, year, …)
│   ├── queries.csv                 # 10 evaluation queries
│   ├── ground_truth_papers_250.csv # 250 manually judged candidate papers
│   ├── ground_truth_relevance.csv  # 2500 rows: 250 papers × 10 queries, binary label
│   ├── vectors.npy                 # [gitignored] (56379, 768) float32 — FAISS data
│   ├── vector_paperids.json        # [gitignored] Paper ID per row in vectors.npy
│   ├── paper_embeddings.npy        # [gitignored] (1500, 768) float32 — per-paper embeds
│   ├── paper_ids.json              # [gitignored] Paper ID list for paper_embeddings.npy
│   ├── authors.csv                 # Author metadata
│   ├── venues.csv                  # Venue metadata
│   ├── citations.csv               # Paper→Paper citation edges
│   ├── written_by.csv              # Paper→Author edges
│   ├── write_together.csv          # Author→Author co-author edges
│   ├── written_for.csv             # Paper→Venue edges
│   ├── field_of_study.csv          # Paper→FieldOfStudy edges
│   ├── affiliations.csv            # Author→Institution edges
│   ├── institutions.csv            # Institution nodes
│   └── dataset_formation.ipynb     # Notebook used to build & label ground truth
│
├── transfer/                       # One-time transfer files (gitignored after setup)
│   └── neo4j.dump                  # Neo4j database dump from original machine
│
└── results/                        # Output CSVs and plots
    ├── evaluation_results_summary.csv
    ├── evaluation_results_detail.csv
    └── ...
```

---

## Setup (one-time, new machine)

### Prerequisites

- Windows 11 with WSL2 (Ubuntu 24.04, systemd enabled)
- WSL2 Anaconda with a `torchtest` conda environment containing PyTorch + CUDA
- `~/.wslconfig` containing:
  ```ini
  [wsl2]
  networkingMode=mirrored
  ```

### Step 1 — Extract FAISS vectors from RAG.db (WSL2, once only)

The original 56,379 SciBERT chunk embeddings are stored in `RAG.db` (Milvus Lite). `milvus-lite` has no Windows wheel, so this extraction runs in WSL2:

```bash
# WSL2 terminal
conda activate torchtest
python /mnt/c/Users/harih/hybrid-graphrag/scripts/extract_vectors.py
```

Outputs `data/vectors.npy` (56,379 × 768, float32) and `data/vector_paperids.json`. After this, WSL2 is not needed for vector search.

### Step 2 — Set up Neo4j in WSL2

Neo4j requires Java and runs as a WSL2 systemd user service. From a WSL2 terminal:

```bash
# Download Java 21 (direct CDN, no apt)
mkdir -p ~/java
wget -qO /tmp/jdk21.tar.gz "https://download.java.net/java/GA/jdk21.0.2/f2283984656d49d69e91c558476027ac/13/GPL/openjdk-21.0.2_linux-x64_bin.tar.gz"
tar -xzf /tmp/jdk21.tar.gz -C ~/
mv ~/jdk-21.0.2 ~/java && rm /tmp/jdk21.tar.gz
export JAVA_HOME=~/java && export PATH=~/java/bin:$PATH

# Download Neo4j 5.26.1 (direct CDN)
wget -qO /tmp/neo4j.tar.gz "https://dist.neo4j.org/neo4j-community-5.26.1-unix.tar.gz"
tar -xzf /tmp/neo4j.tar.gz -C ~/
mv ~/neo4j-community-5.26.1 ~/neo4j && rm /tmp/neo4j.tar.gz

# Configure bolt to listen on all interfaces (required for Windows access)
sed -i 's/#server.bolt.listen_address=:7687/server.bolt.listen_address=0.0.0.0:7687/' ~/neo4j/conf/neo4j.conf

# Set password and start
~/neo4j/bin/neo4j-admin dbms set-initial-password Thammu123
~/neo4j/bin/neo4j start
```

Then import the graph from the CSV files:

```bash
conda activate torchtest
pip install neo4j -q   # only needed once
python - <<'EOF'
import csv, pathlib
from neo4j import GraphDatabase
# ... (see ingestion/run_ingestion.py lines 460–580 for the full import queries)
EOF
```

> **Shortcut:** the import logic is extracted into a standalone script. Contact the project maintainer for `scripts/_setup_neo4j_wsl.py`.

Register Neo4j as a systemd user service so it survives reboots:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/neo4j.service <<'EOF'
[Unit]
Description=Neo4j Graph Database
After=network.target
[Service]
Type=forking
ExecStart=/home/harih/neo4j/bin/neo4j start
ExecStop=/home/harih/neo4j/bin/neo4j stop
PIDFile=/home/harih/neo4j/run/neo4j.pid
Environment=JAVA_HOME=/home/harih/java
Environment=PATH=/home/harih/java/bin:/usr/local/bin:/usr/bin:/bin
Restart=on-failure
[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable neo4j
systemctl --user start neo4j
```

### Step 3 — Pre-compute paper embeddings

```bash
# WSL2, torchtest env — uses RTX 5080 GPU
conda activate torchtest
python /mnt/c/Users/harih/hybrid-graphrag/scripts/precompute_embeddings.py
# Completes in ~3 s; writes data/paper_embeddings.npy
```

### Step 4 — Verify everything

```bash
conda activate torchtest
python /mnt/c/Users/harih/hybrid-graphrag/scripts/verify_setup.py
# All 6 checks should pass: GPU, SciBERT, FAISS, Neo4j, embedding cache, CSVs
```

---

## Running Evaluation

```bash
# WSL2 terminal — Neo4j must be running (systemd starts it automatically on WSL2 boot)
conda activate torchtest
python /mnt/c/Users/harih/hybrid-graphrag/evaluation/run_evaluation.py
```

### Optional: Run from Windows PowerShell

Neo4j is reachable at `bolt://localhost:7687` from Windows if WSL2 mirrored networking is active **and** a Windows Firewall inbound rule exists for port 7687. Add the rule once (requires Admin PowerShell):

```powershell
netsh advfirewall firewall add rule name="Neo4j Bolt (WSL2)" dir=in action=allow protocol=TCP localport=7687
```

Then evaluation can run from the Windows project venv:

```powershell
.venv\Scripts\python.exe evaluation\run_evaluation.py
```

---

## Databases

### FAISS (`data/vectors.npy`)
- **Type**: `faiss.IndexFlatIP` — exact inner-product (cosine) search, zero approximation
- **56,379 vectors**, 768-dim float32, extracted from `RAG.db` via `scripts/extract_vectors.py`
- Covers abstract chunks (all 1500 papers) + PDF full-text chunks (380 papers with valid PDFs)
- Search time: ~10 ms per query on CPU for the full 56K vectors
- **Not stored in git** (165 MB). Regenerate with `scripts/extract_vectors.py`.

### Neo4j (WSL2 systemd service)
- **URI**: `bolt://localhost:7687` / **User**: `neo4j` / **Pass**: `Thammu123`
- **Rebuilt from CSVs** — not from the dump (dump was made with a newer Neo4j version than available)
- **Node types**: `Paper`, `Author`, `Venue`, `FieldOfStudy`, `Institution`
- **Relationship types**: `CITES`, `WROTE`, `PUBLISHED_IN`, `HAS_FIELD`, `AFFILIATED_WITH`
- **Scale**: 1500 Papers, 8071 Authors, 618 Venues, 10 FieldOfStudy nodes
- Relationship counts in this rebuild: `CITES=2939, WROTE=9366, PUBLISHED_IN=1387, HAS_FIELD=1359, AFFILIATED_WITH=445`
  - Note: counts are half of the original README values because the original dump stored every edge bidirectionally; the CSV rebuild stores each edge once. Since all BFS traversal is **undirected** (`-[:CITES]-(m)`), results are functionally identical.
- Start/stop: `systemctl --user start|stop neo4j`

### RAG.db (backup)
- Original Milvus Lite database, 56,379 vectors, kept at project root
- Not used at runtime — `milvus-lite` has no Windows wheel and gRPC throttling makes bulk extraction unreliable at high offsets
- Source of truth for the FAISS vectors; use `scripts/extract_vectors.py` to re-extract if needed

---

## Paper Embedding Cache

`data/paper_embeddings.npy` — (1500, 768) float32. One embedding per paper (title + abstract concatenated), computed with SciBERT at batch size 256 on the RTX 5080 in ~3 s.

Used exclusively by `MetaPathBestFirstGraph` at startup. If the file is missing, `run_evaluation.py` falls back to computing embeddings on-the-fly (slower).

Regenerate:
```bash
conda activate torchtest
python /mnt/c/Users/harih/hybrid-graphrag/scripts/precompute_embeddings.py
```

**Not stored in git** (4.6 MB — excluded because it's a derived artifact).

---

## Corpus

- **Seed paper**: `649def34f8be52c8b66281af98ae884c09aef38b` — primary RAG survey paper
- **Expansion**: BFS via Semantic Scholar API from seed, targeting 1500 papers
- **Topic**: Scientific RAG, knowledge graphs, information extraction, NLP, LLMs
- **Ground truth**: 250 papers manually selected and relevance-judged across 10 queries (2500 binary labels)

---

## Evaluation

### Metrics
- **MRR@5** — Mean Reciprocal Rank at 5
- **Recall@5** — Recall at 5
- **Recall@10** — Recall at 10
- **Rel in Pool** — Total relevant papers in the retrieval pool before reranking
- **Avg Pool** — Average deduped pool size per query

### Retrieval Configurations

All 13 configurations share the same 10 queries and 250-paper ground truth with binary relevance labels.

| Config | Method | VEC | GRAPH_K | Pre-dedup budget |
|---|---|---|---|---|
| `hybrid_reranker` | Vector + multi-channel BFS → CrossEncoder | 60 | 15 | ~960 |
| `hybrid_freq_no_reranker` | Vector + BFS → frequency ranking | 60 | 15 | ~960 |
| `hybrid_interleave_no_reranker` | Vector + BFS → interleaved ranking | 60 | 15 | ~960 |
| `b800_hybrid_reranker` | Vector + BFS → CrossEncoder | 80 | 14 | 1200 |
| `b800_hybrid_freq` | Vector + BFS → frequency ranking | 80 | 14 | 1200 |
| `b800_hybrid_interleave` | Vector + BFS → interleaved ranking | 80 | 14 | 1200 |
| `metapath_hybrid_reranker` | Vector + query-aware BFS → CrossEncoder | 200 | 20 | ~4200 |
| `metapath_hybrid_freq` | Vector + query-aware BFS → frequency | 200 | 20 | ~4200 |
| `metapath_hybrid_interleave` | Vector + query-aware BFS → interleave | 200 | 20 | ~4200 |
| `vector_reranker` | Vector only → CrossEncoder | 60 | — | 60 |
| `vector_only` | Vector only, no reranking | 60 | — | 60 |
| `vector_poolmatch_reranker` | Vector (pool-matched to hybrid size) → CrossEncoder | ~300 | — | ~300 |
| `vector_poolmatch_only` | Vector (pool-matched), no reranking | ~300 | — | ~300 |

### Reranker
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Applied only to the 250 ground-truth papers within each pool; returns top-10

---

## Graph Retrieval Algorithms

### Standard BFS (`PaperBFS` in `run_evaluation.py`)
Four independent BFS channels, round-robin interleaved to prevent high-degree edges from starving others:
1. **CITES** — direct citation edges between Papers
2. **Co-author** — Paper → WROTE → Author → WROTE → Paper
3. **Venue** — Paper → PUBLISHED_IN → Venue → PUBLISHED_IN → Paper
4. **Field** — Paper → HAS_FIELD → FieldOfStudy → HAS_FIELD → Paper

### Query-Aware Meta-Path BFS (`MetaPathBestFirstGraph` in `retrieval/query_aware_graph.py`)
Greedy best-first search guided by cosine similarity to the query:
- Single max-heap keyed on similarity; always expands the globally most-similar candidate next
- All four meta-paths collected via a single Cypher `UNION` query per expansion
- Uses `data/paper_embeddings.npy` loaded at startup — no per-query embedding delay

---

## Known Issues / Experimental Notes

1. **PDF chunk bias**: 380 papers have full-text PDFs (up to 813 chunks); 1120 have only abstract chunks. Vector search heavily favours PDF papers regardless of topical relevance.

2. **Field channel noise**: `FieldOfStudy` is too coarse (10 fields, mostly "Computer Science"). Finds 0 relevant papers in most queries and only adds noise.

3. **Relevance rate by source** (VEC=60, GRAPH_K=15):
   - Vector: 5.3% — most efficient channel
   - Co-author: 2.8%
   - Venue: 2.4%
   - CITES: 1.9%
   - Field: 0.0%

4. **Optimal budget config**: At ~1200 pre-dedup budget, VEC=200/GRAPH_K=4 finds the most relevant papers. Higher graph weight consistently underperforms because vector precision (5.3%) far exceeds graph precision (~2%).

5. **Statistical significance**: Only 10 queries — results are directionally useful but not statistically significant for publication. Expanding to 50+ queries is strongly recommended.

---

## Ingestion Pipeline

To re-run ingestion from scratch or extend the corpus:

```bash
# WSL2, torchtest env
conda activate torchtest
python /mnt/c/Users/harih/hybrid-graphrag/ingestion/run_ingestion.py
```

State is checkpointed in `ingestion/ingestion_state.json`. The pipeline:
1. BFS from seed paper via Semantic Scholar API
2. Chunks text (abstracts + PDF full-text) with tiktoken (128 tokens, 26 overlap)
3. Embeds chunks with SciBERT, inserts into Milvus (`RAG.db`)
4. Exports CSVs to `data/`
5. Imports all nodes and relationships into Neo4j
