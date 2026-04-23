#!/usr/bin/env python3
"""
Pre-compute SciBERT embeddings for all 1500 papers (title + abstract).
Uses the RTX 5080 GPU with a large batch size for fast throughput.

Outputs:
  data/paper_embeddings.npy   — float32 array (N, 768)
  data/paper_ids.json         — ordered list of paper IDs matching row order

Run:  python scripts/precompute_embeddings.py
Re-run at any time to refresh the cache.
"""
import csv, json, os, sys, time, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PAPERS_CSV   = ROOT / "data" / "papers.csv"
EMB_NPY      = ROOT / "data" / "paper_embeddings.npy"
IDS_JSON     = ROOT / "data" / "paper_ids.json"
MODEL_NAME   = "jordyvl/scibert_scivocab_uncased_sentence_transformer"
BATCH_SIZE   = 256   # RTX 5080 has 16 GB VRAM — large batches are fast
csv.field_size_limit(10 * 1024 * 1024)

def main():
    # ── Load papers ───────────────────────────────────────────────────────────
    print(f"Loading papers from {PAPERS_CSV} ...")
    papers = {}
    with open(PAPERS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            if pid not in papers:
                title    = row.get("title", "") or ""
                abstract = row.get("abstract", "") or ""
                papers[pid] = (title + " " + abstract).strip()

    paper_ids   = list(papers.keys())
    paper_texts = [papers[pid] for pid in paper_ids]
    print(f"  {len(paper_ids)} unique papers loaded.")

    # ── Load model onto GPU ───────────────────────────────────────────────────
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
    else:
        print("  WARNING: CUDA not available — running on CPU (will be slow).")

    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    # Enable FP16 on GPU for ~2x throughput with no quality loss on embeddings
    if device == "cuda":
        model = model.half()

    # ── Encode ────────────────────────────────────────────────────────────────
    print(f"Encoding {len(paper_texts)} papers (batch_size={BATCH_SIZE}) ...")
    t0 = time.time()
    embeddings = model.encode(
        paper_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    elapsed = time.time() - t0

    # Cast back to float32 for storage regardless of compute dtype
    embeddings = embeddings.astype(np.float32)
    print(f"  Done in {elapsed:.1f}s  ({len(paper_texts)/elapsed:.1f} papers/s)")
    print(f"  Embedding matrix: {embeddings.shape}  dtype={embeddings.dtype}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(str(EMB_NPY), embeddings)
    with open(IDS_JSON, "w", encoding="utf-8") as f:
        json.dump(paper_ids, f)

    print(f"\nSaved:")
    print(f"  {EMB_NPY}  ({os.path.getsize(EMB_NPY) / 1e6:.1f} MB)")
    print(f"  {IDS_JSON}")

if __name__ == "__main__":
    main()
