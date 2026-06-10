#!/usr/bin/env python3
"""
One-time extraction of the *source text* of every embedded chunk from RAG.db.

This freezes the exact text behind all 56,379 vectors --- including chunks from
papers that are cited by the corpus but not part of the final 1,500-paper set
("orphan" chunks, which live only in RAG.db) --- into a single committed,
gzipped CSV. Once this file exists, `scripts/build_vectors.py` can regenerate
`data/vectors.npy` and `data/vector_paperids.json` deterministically from
committed data alone, and RAG.db is no longer needed by anyone.

Run once from a WSL2 terminal (milvus-lite is Linux-only):

    conda activate torchtest
    python /mnt/c/Users/harih/hybrid-graphrag/scripts/extract_chunks.py

Output (committed to the repo):
    data/vector_chunks.csv.gz   columns: id,paperId,chunk_source,chunk_index,chunk_text
                                ordered by id (== row order of vectors.npy)

The script verifies that the extracted (id -> paperId) sequence exactly matches
the committed data/vector_paperids.json before writing.
"""
import csv, gzip, json, pathlib, shutil, sys, tempfile, time
import warnings
warnings.filterwarnings("ignore")

WIN_PROJECT = pathlib.Path("/mnt/c/Users/harih/hybrid-graphrag")
SRC_DB      = WIN_PROJECT / "RAG.db"
OUT_GZ      = WIN_PROJECT / "data" / "vector_chunks.csv.gz"
REF_PIDS    = WIN_PROJECT / "data" / "vector_paperids.json"
COLLECTION  = "ingestion_v0"
BATCH_SIZE  = 1000

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def open_client(db_path: str):
    from pymilvus import MilvusClient
    return MilvusClient(db_path)


def main():
    if not SRC_DB.exists():
        sys.exit(f"RAG.db not found at {SRC_DB}")

    # Copy to WSL2-native fs for fast I/O (10x faster than /mnt/c).
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    tmp_db  = tmp_dir / "RAG.db"
    print(f"Copying RAG.db to {tmp_dir} ...", flush=True)
    shutil.copy2(str(SRC_DB), str(tmp_db))
    db_path = str(tmp_db)

    client = open_client(db_path)
    total  = client.query(collection_name=COLLECTION, filter="id >= 0",
                          output_fields=["count(*)"])[0]["count(*)"]
    del client
    print(f"  Total chunks in RAG.db: {total}", flush=True)

    rows_by_id: dict[int, dict] = {}
    t0 = time.time()
    cur = 0
    while cur < total:
        batch_ids = list(range(cur, min(cur + BATCH_SIZE, total)))
        client = open_client(db_path)          # fresh client avoids gRPC throttling
        try:
            rows = client.get(collection_name=COLLECTION, ids=batch_ids,
                              output_fields=["paperId", "chunk_text",
                                             "chunk_index", "chunk_source"])
        except Exception as e:
            print(f"\n  get() failed at id={cur}: {e}; retry in 10s", flush=True)
            del client; time.sleep(10); continue
        del client
        for r in rows:
            rows_by_id[int(r["id"])] = r
        cur += len(batch_ids)
        rate = cur / (time.time() - t0 + 1e-9)
        print(f"\r  {cur}/{total} ({100*cur/total:.1f}%)  {rate:.0f} chunk/s", end="", flush=True)
        time.sleep(0.2)
    print(flush=True)

    if len(rows_by_id) != total:
        sys.exit(f"\nExpected {total} chunks, got {len(rows_by_id)} unique ids")

    # Verify (id -> paperId) order matches the committed vector_paperids.json.
    ref_pids = json.load(open(REF_PIDS))
    if len(ref_pids) != total:
        sys.exit(f"vector_paperids.json has {len(ref_pids)} ids, RAG.db has {total}")
    mismatch = sum(1 for i in range(total) if rows_by_id[i]["paperId"] != ref_pids[i])
    if mismatch:
        sys.exit(f"\nPAPERID ORDER MISMATCH in {mismatch} rows --- aborting, will not write")
    print("  Verified: id->paperId order matches vector_paperids.json exactly.", flush=True)

    print(f"Writing {OUT_GZ} ...", flush=True)
    with gzip.open(OUT_GZ, "wt", newline="", encoding="utf-8") as gz:
        w = csv.writer(gz)
        w.writerow(["id", "paperId", "chunk_source", "chunk_index", "chunk_text"])
        for i in range(total):
            r = rows_by_id[i]
            w.writerow([i, r["paperId"], r["chunk_source"], r["chunk_index"],
                        r["chunk_text"]])

    shutil.rmtree(str(tmp_dir))
    size_mb = OUT_GZ.stat().st_size / 1e6
    print(f"\nDone. {total} chunks -> {OUT_GZ}  ({size_mb:.1f} MB gzipped)", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
