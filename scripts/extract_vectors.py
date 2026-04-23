#!/usr/bin/env python3
"""
Extract all vectors from RAG.db with exact data fidelity.
Run once from a WSL2 terminal:

    python3 /mnt/c/Users/harih/hybrid-graphrag/scripts/extract_vectors.py

Strategy:
- Uses client.get() with explicit ID batches (avoids the 16,384 offset ceiling
  that makes client.query() fail beyond row 16,384)
- Reconnects the MilvusClient per batch (avoids gRPC keepalive throttling)
- Copies RAG.db to WSL2 native filesystem first (10x faster I/O than /mnt/c/)
- Writes a checkpoint file so the run can be resumed if interrupted

Outputs (on the Windows filesystem):
    data/vectors.npy           -- (56379, 768) float32
    data/vector_paperids.json  -- list[str] of 56379 paper IDs
"""
import json, pathlib, shutil, sys, tempfile, time
import numpy as np

WIN_PROJECT  = pathlib.Path("/mnt/c/Users/harih/hybrid-graphrag")
OUT_NPY      = WIN_PROJECT / "data" / "vectors.npy"
OUT_PIDS     = WIN_PROJECT / "data" / "vector_paperids.json"
CHECKPOINT   = WIN_PROJECT / "data" / "_extract_checkpoint.json"
COLLECTION   = "ingestion_v0"
BATCH_SIZE   = 1000   # well under any Milvus limit

def open_client(db_path: str):
    from pymilvus import MilvusClient
    return MilvusClient(db_path)

def main():
    src_db = WIN_PROJECT / "RAG.db"
    if not src_db.exists():
        sys.exit(f"RAG.db not found at {src_db}")

    # Copy to WSL2 native fs for fast I/O
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    tmp_db  = tmp_dir / "RAG.db"
    print(f"Copying RAG.db to WSL2 temp ({tmp_dir}) ...", flush=True)
    shutil.copy2(str(src_db), str(tmp_db))
    db_path = str(tmp_db)
    print("  Done.", flush=True)

    # Get total count
    client = open_client(db_path)
    total  = client.query(
        collection_name=COLLECTION,
        filter="id >= 0",
        output_fields=["count(*)"],
    )[0]["count(*)"]
    del client
    print(f"  Total vectors in RAG.db: {total}", flush=True)

    # Load checkpoint if resuming
    start_id  = 0
    all_vecs  = []
    all_pids  = []
    if CHECKPOINT.exists():
        cp = json.loads(CHECKPOINT.read_text())
        start_id = cp["next_id"]
        all_vecs = [np.array(v, dtype=np.float32) for v in cp["vecs_so_far"]]
        all_pids = cp["pids_so_far"]
        print(f"  Resuming from ID {start_id} ({len(all_pids)} already extracted)", flush=True)

    t0 = time.time()
    current_id = start_id

    while current_id < total:
        batch_ids = list(range(current_id, min(current_id + BATCH_SIZE, total)))

        # Fresh client per batch to avoid gRPC keepalive throttling
        client = open_client(db_path)
        try:
            rows = client.get(
                collection_name=COLLECTION,
                ids=batch_ids,
                output_fields=["paperId", "vector"],
            )
        except Exception as e:
            print(f"\n  Warning: get() failed at id={current_id}: {e}", flush=True)
            print("  Retrying after 10s ...", flush=True)
            del client
            time.sleep(10)
            continue
        del client

        for r in rows:
            all_vecs.append(np.array(r["vector"], dtype=np.float32))
            all_pids.append(r["paperId"])

        current_id += len(batch_ids)
        elapsed = time.time() - t0
        rate    = (current_id - start_id) / elapsed if elapsed > 0 else 0
        eta     = (total - current_id) / rate if rate > 0 else 0
        print(
            f"\r  {current_id}/{total} ({100*current_id/total:.1f}%)  "
            f"{rate:.0f} vec/s  ETA {eta:.0f}s",
            end="", flush=True,
        )
        time.sleep(0.5)   # brief pause; prevents rapid-fire gRPC connections

    print(f"\n  Extraction complete: {len(all_vecs)} vectors", flush=True)

    # Stack and save
    vecs = np.stack(all_vecs, axis=0)   # (N, 768) float32
    print(f"  Matrix: {vecs.shape}  dtype={vecs.dtype}", flush=True)
    np.save(str(OUT_NPY), vecs)
    with open(OUT_PIDS, "w") as f:
        json.dump(all_pids, f)

    # Clean up
    shutil.rmtree(str(tmp_dir))
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print(f"\nSaved:")
    print(f"  {OUT_NPY}  ({OUT_NPY.stat().st_size / 1e6:.1f} MB)")
    print(f"  {OUT_PIDS}")
    print(f"  Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
