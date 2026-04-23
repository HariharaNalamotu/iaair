#!/usr/bin/env python3
"""
recover_missing_chunks.py
Re-inserts the abstract chunks and PDF chunks that were lost due to
overlapping IDs from a previous ingestion run.

Run this once to restore the DB to the correct state.
New vectors are assigned IDs starting from 22508 (after the current max).
"""
import csv
import json
import os
import sys
from pathlib import Path

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import tiktoken

# ── Configuration (matches data_ingestion.ipynb) ──────────────────────────────
MILVUS_DB   = "/Users/Hari/IAAIR/RAG.db"
COLLECTION  = "ingestion_v0"
PDF_DIR     = "/Users/Hari/IAAIR/semantic_scholar_corpus/pdfs/"
STATE_FILE  = "/Users/Hari/IAAIR/ingestion_state.json"
PAPERS_CSV  = "/Users/Hari/IAAIR/papers.csv"
EMBED_MODEL = "jordyvl/scibert_scivocab_uncased_sentence_transformer"
CHUNK_SIZE    = 128
CHUNK_OVERLAP = 26
START_ID      = 22508   # first free ID (current max is 22507)
PDF_BATCH     = 500     # flush to Milvus every N pdf chunks

csv.field_size_limit(sys.maxsize)


def chunk_text(text: str) -> list[str]:
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        chunks.append(enc.decode(tokens[start : start + CHUNK_SIZE]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_paper_ids_in_db(client, source: str) -> set:
    """Return set of paperIds that already have chunks of the given source in DB."""
    seen = set()
    with open(PAPERS_CSV, newline="", encoding="utf-8") as f:
        all_ids = [row["id"] for row in csv.DictReader(f)]

    batch_size = 50
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i : i + batch_size]
        ids_str = ", ".join(f'"{pid}"' for pid in batch)
        rows = client.query(
            collection_name=COLLECTION,
            filter=f'paperId in [{ids_str}] && chunk_source == "{source}"',
            output_fields=["paperId"],
            limit=16000,
        )
        for r in rows:
            seen.add(r["paperId"])
    return seen


def main() -> int:
    print("=" * 60)
    print("  Recovery: re-inserting missing abstract & PDF chunks")
    print("=" * 60)

    # ── Connect ──────────────────────────────────────────────────────────
    client = MilvusClient(MILVUS_DB)
    total_before = client.get_collection_stats(COLLECTION)["row_count"]
    print(f"\nVectors in DB before recovery: {total_before}")

    # ── Load embedding model ──────────────────────────────────────────────
    print("\nLoading embedding model …")
    model = SentenceTransformer(EMBED_MODEL)

    id_counter = START_ID

    # ── 1. Recover abstract chunks ────────────────────────────────────────
    print("\n[1/2] Recovering abstract chunks …")

    with open(STATE_FILE) as f:
        seen_ids = set(json.load(f)["seen_paper_ids"])

    already_have_abstract = get_paper_ids_in_db(client, "abstract")
    print(f"      Papers already with abstract chunks: {len(already_have_abstract)}")

    papers_needing_abstract = []
    with open(PAPERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] in seen_ids and row["id"] not in already_have_abstract:
                if (row.get("abstract") or "").strip():
                    papers_needing_abstract.append(row)

    print(f"      Papers needing re-insertion:         {len(papers_needing_abstract)}")

    abstract_inserted = 0
    for paper in papers_needing_abstract:
        chunks = chunk_text(paper["abstract"])
        embeddings = model.encode(chunks)
        datapoints = [
            {
                "id":           id_counter + i,
                "paperId":      paper["id"],
                "title":        paper["title"],
                "abstract":     paper["abstract"],
                "chunk_text":   chunk,
                "chunk_index":  i,
                "chunk_source": "abstract",
                "vector":       emb,
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        client.insert(collection_name=COLLECTION, data=datapoints)
        id_counter += len(datapoints)
        abstract_inserted += len(datapoints)

    print(f"      Re-inserted {abstract_inserted} abstract vectors  "
          f"(IDs {START_ID}–{id_counter - 1})")

    # ── 2. Recover PDF chunks ─────────────────────────────────────────────
    print("\n[2/2] Recovering PDF chunks …")

    already_have_pdf = get_paper_ids_in_db(client, "pdf")
    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))
    missing_pdf_files = [p for p in pdf_files if p.stem not in already_have_pdf]
    print(f"      PDFs on disk:            {len(pdf_files)}")
    print(f"      Already in DB:           {len(already_have_pdf)}")
    print(f"      Need re-embedding:       {len(missing_pdf_files)}")

    from pypdf import PdfReader

    pdf_buffer = []
    pdf_inserted = 0
    pdf_id_start = id_counter

    for pdf_path in missing_pdf_files:
        paper_id = pdf_path.stem
        try:
            reader = PdfReader(str(pdf_path))
            text = "".join(page.extract_text() or "" for page in reader.pages)
            chunks = chunk_text(text)
            embeddings = model.encode(chunks)
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                pdf_buffer.append({
                    "id":           id_counter,
                    "paperId":      paper_id,
                    "chunk_text":   chunk,
                    "chunk_index":  i,
                    "chunk_source": "pdf",
                    "vector":       emb,
                })
                id_counter += 1
        except Exception as e:
            print(f"      SKIP {paper_id}: {e}")

        if len(pdf_buffer) >= PDF_BATCH:
            res = client.insert(collection_name=COLLECTION, data=pdf_buffer)
            pdf_inserted += res["insert_count"]
            pdf_buffer.clear()

    if pdf_buffer:
        res = client.insert(collection_name=COLLECTION, data=pdf_buffer)
        pdf_inserted += res["insert_count"]
        pdf_buffer.clear()

    print(f"      Re-inserted {pdf_inserted} PDF vectors  "
          f"(IDs {pdf_id_start}–{id_counter - 1})")

    # ── Summary ───────────────────────────────────────────────────────────
    total_after = client.get_collection_stats(COLLECTION)["row_count"]
    print(f"\nVectors in DB after recovery:  {total_after}")
    print(f"Net vectors added:             {total_after - total_before}")
    print(f"\nNext id_num to use:            {id_counter}")
    print(f"  → Update ingestion_state.json if you plan to run more ingestion.\n")

    # Optionally update state file id_num
    with open(STATE_FILE) as f:
        state = json.load(f)
    state["id_num"] = id_counter
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    print(f"Updated {STATE_FILE}  id_num → {id_counter}")

    print("\n  Run python test_vector_db.py to verify.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
