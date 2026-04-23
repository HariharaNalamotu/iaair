#!/usr/bin/env python3
"""
test_vector_db.py
Verifies that all abstract chunks and PDF chunks from the ingestion pipeline
are present in the Milvus vector database (ingestion_v0 collection).

Usage:
    python test_vector_db.py

Exit codes:
    0 — all chunks present
    1 — one or more chunks missing
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ── Configuration (matches data_ingestion.ipynb) ──────────────────────────────
MILVUS_DB   = "/Users/Hari/IAAIR/RAG.db"
COLLECTION  = "ingestion_v0"
PDF_DIR     = "/Users/Hari/IAAIR/semantic_scholar_corpus/pdfs/"
STATE_FILE  = "/Users/Hari/IAAIR/ingestion_state.json"
PAPERS_CSV  = "/Users/Hari/IAAIR/papers.csv"

QUERY_PAGE_SIZE = 2000  # vectors per Milvus query page


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_seen_paper_ids() -> set:
    """Return the set of paper IDs ingested into Milvus (from BFS state file)."""
    with open(STATE_FILE) as f:
        return set(json.load(f)["seen_paper_ids"])


def load_papers(seen_ids: set) -> dict:
    """Return {paperId: row_dict} for every paper in papers.csv that was ingested."""
    csv.field_size_limit(sys.maxsize)  # papers.csv has large full_text fields
    papers = {}
    with open(PAPERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] in seen_ids:
                papers[row["id"]] = row
    return papers


def get_pdf_paper_ids() -> set:
    """Return set of paper IDs that have a PDF file on disk."""
    return {p.stem for p in Path(PDF_DIR).glob("*.pdf")}


def fetch_db_vectors_for_papers(client, paper_ids: set) -> dict:
    """
    Query Milvus in batches of paper IDs and return:
        {paperId: set_of_chunk_sources}
    Uses paperId-in batching to stay within Milvus Lite's limit+offset <= 16384.
    Each batch of 50 papers produces at most ~2800 results (50 × ~56 chunks),
    comfortably under the cap.
    """
    index: dict = defaultdict(set)
    ids_list = sorted(paper_ids)
    batch_size = 50

    for i in range(0, len(ids_list), batch_size):
        batch = ids_list[i : i + batch_size]
        ids_str = ", ".join(f'"{pid}"' for pid in batch)
        page = client.query(
            collection_name=COLLECTION,
            filter=f"paperId in [{ids_str}]",
            output_fields=["id", "paperId", "chunk_source"],
            limit=16000,  # max safe limit for Milvus Lite
        )
        for row in page:
            pid = row.get("paperId") or ""
            src = row.get("chunk_source") or ""
            index[pid].add(src)

    return dict(index)


def print_missing(label: str, missing: list, papers: dict):
    if not missing:
        return
    preview = missing[:20]
    for pid in preview:
        title = (papers.get(pid) or {}).get("title") or ""
        suffix = f"  ({title[:60]})" if title else ""
        print(f"      - {pid}{suffix}")
    if len(missing) > 20:
        print(f"      … and {len(missing) - 20} more")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        from pymilvus import MilvusClient
    except ImportError:
        print("ERROR: pymilvus not installed. Activate the iaair2 conda env first.")
        return 1

    print("=" * 64)
    print("  Vector DB Completeness Check")
    print("=" * 64)

    # ── Step 1: Ground truth from CSV and state file ──────────────────────
    print("\n[1/4] Loading ingestion state …")
    seen_ids = load_seen_paper_ids()
    print(f"      seen_paper_ids:           {len(seen_ids)}")

    print("\n[2/4] Loading papers.csv (ingested papers only) …")
    papers = load_papers(seen_ids)
    print(f"      Matched in papers.csv:    {len(papers)}")

    not_in_csv = seen_ids - set(papers)
    if not_in_csv:
        print(f"      WARNING: {len(not_in_csv)} seen IDs not found in papers.csv — "
              "they will be skipped.")

    papers_with_abstract    = {pid for pid, p in papers.items()
                                if (p.get("abstract") or "").strip()}
    papers_without_abstract = {pid for pid, p in papers.items()
                                if not (p.get("abstract") or "").strip()}
    print(f"      With non-empty abstract:  {len(papers_with_abstract)}")
    print(f"      Without abstract:         {len(papers_without_abstract)}")

    print("\n[3/4] Scanning PDF directory …")
    pdf_ids = get_pdf_paper_ids()
    ingested_with_pdf = seen_ids & pdf_ids
    print(f"      PDF files on disk:        {len(pdf_ids)}")
    print(f"      Ingested papers with PDF: {len(ingested_with_pdf)}")

    # ── Step 2: Query Milvus ──────────────────────────────────────────────
    print("\n[4/4] Querying Milvus (this may take a moment) …")
    try:
        client = MilvusClient(MILVUS_DB)
    except Exception as e:
        if "opened by another program" in str(e) or "Open local milvus failed" in str(e):
            print(f"\nERROR: RAG.db is locked by another process (Jupyter kernel?).")
            print("       Shut down the notebook kernel and retry:\n"
                  "         Kernel > Shut Down All Kernels  (or restart the kernel)\n"
                  "       Then re-run:  python test_vector_db.py")
        else:
            print(f"\nERROR connecting to Milvus: {e}")
        return 1
    total_in_db = client.get_collection_stats(COLLECTION).get("row_count", 0)
    print(f"      Total vectors in DB:      {total_in_db}")

    db_index = fetch_db_vectors_for_papers(client, seen_ids)
    print(f"      Ingested paper IDs found: {len(db_index)}")

    # ── Step 3: Cross-reference ───────────────────────────────────────────
    missing_abstract = sorted(
        pid for pid in papers_with_abstract
        if "abstract" not in db_index.get(pid, set())
    )
    missing_pdf = sorted(
        pid for pid in ingested_with_pdf
        if "pdf" not in db_index.get(pid, set())
    )

    abstract_ok = len(papers_with_abstract) - len(missing_abstract)
    pdf_ok      = len(ingested_with_pdf) - len(missing_pdf)

    # ── Step 4: Report ────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)

    # Abstract summary
    ab_status = "OK" if not missing_abstract else "FAIL"
    print(f"\n  Abstract chunks  [{ab_status}]")
    print(f"    Expected (papers with abstract): {len(papers_with_abstract)}")
    print(f"    Present in DB:                   {abstract_ok}")
    print(f"    Missing:                         {len(missing_abstract)}")
    print_missing("abstract", missing_abstract, papers)

    # PDF summary
    pdf_status = "OK" if not missing_pdf else "FAIL"
    print(f"\n  PDF chunks  [{pdf_status}]")
    print(f"    Expected (PDFs on disk):         {len(ingested_with_pdf)}")
    print(f"    Present in DB:                   {pdf_ok}")
    print(f"    Missing:                         {len(missing_pdf)}")
    print_missing("pdf", missing_pdf, papers)

    # DB overview
    papers_in_db = len(db_index)
    print(f"\n  DB overview")
    print(f"    Total vectors in collection:     {total_in_db}")
    print(f"    Ingested papers with any chunk:  {papers_in_db} / {len(seen_ids)}")

    # Chunk-source breakdown
    source_counts: dict = defaultdict(int)
    for sources in db_index.values():
        for src in sources:
            source_counts[src] += 1
    print(f"\n  Chunk-source breakdown (papers that have each source type)")
    for src, count in sorted(source_counts.items()):
        print(f"    {src:<20} {count}")

    # Final verdict
    passed = not missing_abstract and not missing_pdf
    print("\n" + "=" * 64)
    if passed:
        print("  OVERALL: PASS  — all abstracts and PDFs are in the vector DB")
    else:
        parts = []
        if missing_abstract:
            parts.append(f"{len(missing_abstract)} abstract gap(s)")
        if missing_pdf:
            parts.append(f"{len(missing_pdf)} PDF gap(s)")
        print(f"  OVERALL: FAIL  — {', '.join(parts)}")
    print("=" * 64)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
