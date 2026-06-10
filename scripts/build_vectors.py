#!/usr/bin/env python3
"""
Regenerate the FAISS vector index deterministically from committed text.

Reads data/vector_chunks.csv.gz (the frozen source text of all 56,379 chunks,
produced once by scripts/extract_chunks.py) and re-embeds every chunk with
SciBERT in FP32, reproducing:

    data/vectors.npy           (N, 768) float32   -- raw, un-normalised embeddings
    data/vector_paperids.json  list[str] of N paper IDs (row-aligned)

This removes any dependency on RAG.db: the vector index is now a deterministic
function of committed data. Embeddings are computed in FP32 (no half precision)
for hardware-stability; tiny cross-device floating-point differences (~1e-6) are
expected and do not change FAISS nearest-neighbour results in practice.

Run:
    python scripts/build_vectors.py            # rebuild the index
    python scripts/build_vectors.py --verify   # rebuild, then compare to the
                                               # existing vectors.npy (cosine)
"""
import argparse, csv, gzip, json, pathlib, sys, time
import numpy as np

ROOT       = pathlib.Path(__file__).parent.parent
CHUNKS_GZ  = ROOT / "data" / "vector_chunks.csv.gz"
OUT_NPY    = ROOT / "data" / "vectors.npy"
OUT_PIDS   = ROOT / "data" / "vector_paperids.json"
MODEL_NAME = "jordyvl/scibert_scivocab_uncased_sentence_transformer"
BATCH_SIZE = 256
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_chunks():
    ids, pids, texts = [], [], []
    with gzip.open(CHUNKS_GZ, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["id"]))
            pids.append(row["paperId"])
            texts.append(row["chunk_text"])
    # Rows are written in id order, but sort defensively to guarantee alignment.
    order = np.argsort(ids, kind="stable")
    pids  = [pids[i] for i in order]
    texts = [texts[i] for i in order]
    return pids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="compare rebuilt vectors to the existing vectors.npy")
    args = ap.parse_args()

    if not CHUNKS_GZ.exists():
        sys.exit(f"{CHUNKS_GZ} not found. Run scripts/extract_chunks.py first "
                 "(needs RAG.db, one-time).")

    print(f"Loading chunks from {CHUNKS_GZ} ...")
    pids, texts = load_chunks()
    print(f"  {len(texts)} chunks.")

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = SentenceTransformer(MODEL_NAME, device=device)   # FP32 (no .half())
    print(f"Encoding {len(texts)} chunks (batch_size={BATCH_SIZE}, FP32) ...")
    t0 = time.time()
    vecs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=False)
    vecs = vecs.astype(np.float32)
    print(f"  Done in {time.time()-t0:.0f}s.  Matrix {vecs.shape} {vecs.dtype}")

    if args.verify and OUT_NPY.exists():
        ref = np.load(str(OUT_NPY))
        ref_pids = json.load(open(OUT_PIDS))
        ok_pid = (ref_pids == pids)
        if ref.shape != vecs.shape:
            print(f"  VERIFY: shape differs {ref.shape} vs {vecs.shape}")
        else:
            a = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            b = ref  / (np.linalg.norm(ref,  axis=1, keepdims=True) + 1e-12)
            cos = (a * b).sum(axis=1)
            print(f"  VERIFY: paperId order identical = {ok_pid}")
            print(f"  VERIFY: per-row cosine vs committed vectors.npy  "
                  f"min={cos.min():.6f}  mean={cos.mean():.6f}  "
                  f"frac>0.9999={np.mean(cos > 0.9999):.4f}")
        print("  (Not overwriting in --verify mode.)")
        return

    np.save(str(OUT_NPY), vecs)
    json.dump(pids, open(OUT_PIDS, "w"))
    print(f"\nSaved:\n  {OUT_NPY}  ({OUT_NPY.stat().st_size/1e6:.1f} MB)\n  {OUT_PIDS}")


if __name__ == "__main__":
    main()
