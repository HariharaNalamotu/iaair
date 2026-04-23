#!/usr/bin/env python3
"""
Quick sanity-check for the IAAIR environment.
Run after setup.ps1 to confirm everything is wired up correctly.
"""
import os, sys, json, pathlib
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

PASS = "✓"
FAIL = "✗"

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}" + (f"  ({result})" if result else ""))
        return True
    except Exception as e:
        print(f"  {FAIL}  {label}  ERROR: {e}")
        return False

results = []

# ── 1. GPU / PyTorch ──────────────────────────────────────────────────────────
def _gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    return torch.cuda.get_device_name(0)
results.append(check("PyTorch + CUDA", _gpu))

# ── 2. SciBERT model loads ────────────────────────────────────────────────────
def _model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("jordyvl/scibert_scivocab_uncased_sentence_transformer")
    v = m.encode(["test"])
    assert v.shape == (1, 768)
    return "768-dim embeddings OK"
results.append(check("SciBERT model", _model))

# ── 3. FAISS index ───────────────────────────────────────────────────────────
def _faiss():
    import faiss, numpy as np
    vecs = ROOT / "data" / "vectors.npy"
    pids = ROOT / "data" / "vector_paperids.json"
    assert vecs.exists(), f"Missing {vecs} — run scripts/extract_vectors.py in WSL2"
    assert pids.exists(), f"Missing {pids}"
    v = np.load(str(vecs))
    p = json.load(open(pids))
    assert v.shape[0] == len(p), "Row count mismatch"
    assert v.shape[1] == 768
    idx = faiss.IndexFlatIP(768)
    faiss.normalize_L2(v)
    idx.add(v)
    return f"{idx.ntotal} vectors, {v.shape[1]}-dim"
results.append(check("FAISS index", _faiss))

# ── 4. Neo4j connection ───────────────────────────────────────────────────────
def _neo4j():
    from neo4j import GraphDatabase
    uri  = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    pwd  = os.environ["NEO4J_PASS"]
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as s:
        n_papers = s.run("MATCH (p:Paper) RETURN count(p) AS n").single()["n"]
        assert n_papers > 0, "No Paper nodes found"
    driver.close()
    return f"{n_papers} Paper nodes"
results.append(check("Neo4j database", _neo4j))

# ── 5. Paper embedding cache ──────────────────────────────────────────────────
def _emb_cache():
    import numpy as np
    npy  = ROOT / "data" / "paper_embeddings.npy"
    ids  = ROOT / "data" / "paper_ids.json"
    assert npy.exists(),  f"Missing {npy} — run scripts/precompute_embeddings.py"
    assert ids.exists(),  f"Missing {ids}"
    embs = np.load(str(npy))
    pids = json.load(open(ids, encoding="utf-8"))
    assert embs.shape[0] == len(pids), "Row count mismatch between .npy and ids"
    assert embs.shape[1] == 768, f"Expected 768-dim, got {embs.shape[1]}"
    return f"{embs.shape[0]} embeddings, shape {embs.shape}"
results.append(check("Paper embedding cache", _emb_cache))

# ── 6. Data CSV files ─────────────────────────────────────────────────────────
import csv
csv.field_size_limit(10 * 1024 * 1024)
def _csvs():
    required = ["papers.csv", "queries.csv", "ground_truth_relevance.csv",
                "ground_truth_papers_250.csv"]
    missing = [f for f in required if not (ROOT / "data" / f).exists()]
    assert not missing, f"Missing: {missing}"
    return f"{len(required)} CSV files present"
results.append(check("Data CSVs", _csvs))

# ── Summary ───────────────────────────────────────────────────────────────────
print()
passed = sum(results)
total  = len(results)
if passed == total:
    print(f"All {total} checks passed. Environment is ready.")
else:
    print(f"{passed}/{total} checks passed. Fix the failing items above.")
    sys.exit(1)
