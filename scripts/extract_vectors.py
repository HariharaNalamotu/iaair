#!/usr/bin/env python3
"""
Build FAISS vector data by re-embedding all paper chunks from source text.
Runs entirely on Windows using the project .venv — no WSL2 needed.

    .venv\Scripts\python.exe scripts\extract_vectors.py

Uses the exact same model and chunking parameters as the original ingestion
(SciBERT, 128-token chunks, 26-token overlap), so results are identical to
what is stored in RAG.db.

Outputs:
    data/vectors.npy           — (N, 768) float32, one row per chunk
    data/vector_paperids.json  — list of N paper IDs (row i → paper)
"""
import csv, json, pathlib, sys, time, warnings
import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")  # suppress pypdf/huggingface noise

ROOT       = pathlib.Path(__file__).parent.parent
PAPERS_CSV = ROOT / "data" / "papers.csv"
PDF_DIR    = ROOT / "semantic_scholar_corpus" / "pdfs"
OUT_VECS   = ROOT / "data" / "vectors.npy"
OUT_PIDS   = ROOT / "data" / "vector_paperids.json"

MODEL_NAME    = "jordyvl/scibert_scivocab_uncased_sentence_transformer"
CHUNK_SIZE    = 128   # tokens
CHUNK_OVERLAP = 26    # tokens
EMBED_BATCH   = 256   # larger = faster on GPU
csv.field_size_limit(10 * 1024 * 1024)

def chunk_text(text: str, enc) -> list[str]:
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        chunks.append(enc.decode(tokens[start : start + CHUNK_SIZE]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def main():
    import torch
    import tiktoken
    from pypdf import PdfReader
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: CUDA not available — using CPU (slow).", flush=True)

    print(f"Loading {MODEL_NAME} ...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)

    enc = tiktoken.encoding_for_model("text-embedding-3-small")

    # ── Load papers ───────────────────────────────────────────────────────────
    print(f"Loading papers from {PAPERS_CSV} ...", flush=True)
    papers = {}
    with open(PAPERS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["id"]
            if pid not in papers:
                papers[pid] = {
                    "abstract": row.get("abstract") or "",
                    "title":    row.get("title") or "",
                }
    print(f"  {len(papers)} papers", flush=True)

    # ── Build chunk list ──────────────────────────────────────────────────────
    print("Building chunks (abstract + PDF where available) ...", flush=True)
    chunk_texts = []
    chunk_pids  = []
    t0 = time.time()

    for i, (pid, p) in enumerate(papers.items()):
        # Abstract chunks
        abstract = p["abstract"].strip()
        if abstract:
            for chunk in chunk_text(abstract, enc):
                chunk_texts.append(chunk)
                chunk_pids.append(pid)

        # PDF chunks (if PDF exists)
        pdf_path = PDF_DIR / f"{pid}.pdf"
        if pdf_path.exists():
            try:
                reader = PdfReader(str(pdf_path))
                full_text = "".join(page.extract_text() or "" for page in reader.pages)
                if full_text.strip():
                    for chunk in chunk_text(full_text, enc):
                        chunk_texts.append(chunk)
                        chunk_pids.append(pid)
            except Exception:
                pass  # skip unreadable PDFs

        if (i + 1) % 100 == 0:
            print(f"\r  {i+1}/{len(papers)} papers -> {len(chunk_texts)} chunks ...",
                  end="", flush=True)

    print(f"\r  {len(papers)} papers -> {len(chunk_texts)} chunks total", flush=True)

    # ── Embed all chunks ──────────────────────────────────────────────────────
    print(f"Embedding {len(chunk_texts)} chunks (batch={EMBED_BATCH}) ...", flush=True)
    t1 = time.time()
    embeddings = model.encode(
        chunk_texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.1f}s  ({len(chunk_texts)/elapsed:.0f} chunks/s)", flush=True)

    vecs = embeddings.astype(np.float32)
    print(f"  Matrix: {vecs.shape}  dtype={vecs.dtype}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(str(OUT_VECS), vecs)
    with open(OUT_PIDS, "w", encoding="utf-8") as f:
        json.dump(chunk_pids, f)

    print(f"\nSaved:")
    print(f"  {OUT_VECS}  ({OUT_VECS.stat().st_size / 1e6:.1f} MB)")
    print(f"  {OUT_PIDS}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
