"""Reconstruct the worked example for the paper's qualitative section.

Rebuilds, from the evaluation cache, the exact retrieval pools and re-ranked
lists for one sweep cell, and verifies them against the recorded metrics in
results/final_evaluation_sweep.csv before printing the trace:

  Query 4, budget 200:
    metapath  vec_n=50  graph_k=150   (MRR@5 1.0,  Recall@10 0.176, 4 rel in pool)
    vector    vec_n=200               (MRR@5 0.5,  Recall@10 0.294, 8 rel in pool)

Requires results/eval_cache/ (populated by evaluation/final_evaluation.py)
and a GPU/CPU CrossEncoder for the re-ranking step.
"""
import csv, hashlib, json, pathlib, pickle, sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ROOT      = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "results" / "eval_cache"

QUERY_ID  = 4
QUERY     = ("How can NLP tools support researchers in conducting "
             "systematic literature reviews and evidence synthesis?")
BUDGET    = 200
META_VECN = 50
META_GK   = 150
META_HOPS = 4

def _ckey(*args) -> str:
    return hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()

def cache_get(ns, *args):
    p = CACHE_DIR / f"{ns}_{_ckey(*args)}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"cache miss: {ns} {args[:2]}...")
    return pickle.load(open(p, "rb"))

# ── Ground truth and paper metadata ───────────────────────────────────────────
relevant: set[str] = set()
judged:   set[str] = set()
with open(ROOT / "data" / "ground_truth_relevance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        judged.add(row["paperId"])
        if int(row["query_id"]) == QUERY_ID and int(row["relevant"]) == 1:
            relevant.add(row["paperId"])

papers: dict[str, dict] = {}
with open(ROOT / "data" / "papers.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["id"] not in papers:
            papers[row["id"]] = {"title": row.get("title", ""),
                                 "abstract": row.get("abstract", "")}

# ── Rebuild pools from cache (identical to final_evaluation.py) ───────────────
def unique_pool(vec_ids, graph):
    pool, seen = list(vec_ids), set(vec_ids)
    for sid in vec_ids:
        for pid in graph.get(sid, []):
            if pid not in seen:
                seen.add(pid); pool.append(pid)
    return pool

meta_seeds = cache_get("vs", QUERY, META_VECN)
meta_graph = {sid: cache_get("meta", QUERY, sid, META_GK, META_HOPS)
              for sid in meta_seeds}
meta_pool  = unique_pool(meta_seeds, meta_graph)
vec_pool   = cache_get("vs", QUERY, BUDGET)

print(f"metapath pool: {len(meta_pool)} papers, "
      f"{len(set(meta_pool) & relevant)} relevant  (CSV: 224 / 4)")
print(f"vector   pool: {len(vec_pool)} papers, "
      f"{len(set(vec_pool) & relevant)} relevant  (CSV: 200 / 8)")

# ── Re-rank both pools (identical to rerank_all) ──────────────────────────────
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(pool):
    # As in final_evaluation.py: only judged candidates are ranked (closed pool)
    cands  = [p for p in pool if p in judged]
    pairs  = [(QUERY, (papers[p]["title"] + " " + papers[p]["abstract"]).strip())
              for p in cands if p in papers]
    pids   = [p for p in cands if p in papers]
    scores = reranker.predict(pairs, batch_size=512, show_progress_bar=False)
    ranked = sorted(zip(scores, pids), reverse=True)
    return [(pid, float(sc)) for sc, pid in ranked[:10]]

def mrr5(ranked):
    for i, (pid, _) in enumerate(ranked[:5]):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0

def recall10(ranked):
    return len({pid for pid, _ in ranked[:10]} & relevant) / len(relevant)

for name, pool, want in (("METAPATH", meta_pool, (1.0, 0.1765)),
                         ("VECTOR",   vec_pool,  (0.5, 0.2941))):
    top = rerank(pool)
    print(f"\n=== {name}  MRR@5={mrr5(top):.4f} (CSV {want[0]})  "
          f"Recall@10={recall10(top):.4f} (CSV {want[1]})  "
          f"|relevant|={len(relevant)} ===")
    for i, (pid, sc) in enumerate(top, 1):
        mark = "REL" if pid in relevant else "   "
        print(f"  {i:2d}. [{mark}] {sc:+.3f}  {papers[pid]['title'][:90]}")
