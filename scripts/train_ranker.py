#!/usr/bin/env python3
"""
Train a neural network to predict paper relevance from three retrieval signals.

Features  : min-max scaled (vec_score, graph_score, rerank_score)
Target    : 1 = relevant, 0 = not relevant

Evaluation: leave-one-query-out cross-validation (10 folds).

Run AFTER the evaluation with metapath_weighted has populated the score caches:

    conda activate torchtest
    python /mnt/c/Users/harih/hybrid-graphrag/scripts/train_ranker.py

Outputs:
    results/ranker_model.pt         -- trained neural network weights
    results/ranker_scaler.json      -- min/max values used for scaling
    results/ranker_loo_results.csv  -- per-fold LOO-CV metrics
"""
import csv, hashlib, json, pathlib, pickle, sys
import numpy as np
import torch
import torch.nn as nn

ROOT      = pathlib.Path(__file__).parent.parent
CACHE_DIR = ROOT / "results" / "eval_cache"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

csv.field_size_limit(10 * 1024 * 1024)
print(f"Device: {DEVICE}" +
      (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

# ── Config (must match evaluation/run_evaluation.py) ─────────────────────────
METAPATH_VEC  = 200
METAPATH_GK   = 5
METAPATH_HOPS = 4

QUERIES = [
    "How do graph-based and multi-step retrieval methods enhance retrieval-augmented generation systems?",
    "How are knowledge graphs constructed and applied to organize scholarly and scientific information?",
    "What methods are used for automated information extraction from scientific and biomedical text?",
    "How can NLP tools support researchers in conducting systematic literature reviews and evidence synthesis?",
    "What bibliometric and citation analysis methods reveal research trends and collaboration patterns?",
    "What techniques reduce factual hallucination in large language model outputs?",
    "What deep learning and language model approaches are used for classifying, tagging, or organizing scientific text?",
    "How are pretrained language models and embedding techniques used for semantic understanding of scientific text?",
    "What benchmarks, evaluation methods, and datasets are used to assess RAG and information retrieval system performance?",
    "How can AI systems assist in peer review, quality assessment, and research impact evaluation of scholarly work?",
]

# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_key(*args):
    return hashlib.md5(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()

def _load(namespace, *args):
    p = CACHE_DIR / f"{namespace}_{_cache_key(*args)}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None

# ── Load relevance labels ─────────────────────────────────────────────────────
print("Loading relevance labels...")
relevance_by_query: dict[int, set] = {}
gt_pids: set[str] = set()
with open(ROOT / "data" / "ground_truth_relevance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qi = int(row["query_id"])
        gt_pids.add(row["paperId"])
        relevance_by_query.setdefault(qi, set())
        if int(row["relevant"]) == 1:
            relevance_by_query[qi].add(row["paperId"])

# ── Collect raw scores from evaluation cache ──────────────────────────────────
def collect_query(qi: int, query: str) -> list[tuple]:
    """
    Returns list of (paper_id, vec_score, graph_score, rerank_score, label)
    for every ground-truth paper reached during the metapath_weighted run.
    Papers missing a particular score receive 0.0 (imputed pre-scaling minimum).
    """
    data = _load("vec_scores", query, METAPATH_VEC)
    if data is None:
        return []
    vec_ids, vec_score_map = data

    # Aggregate metapath similarity scores (max across seeds)
    graph_score_map: dict[str, float] = {}
    all_graph_ids: set[str] = set()
    for sid in vec_ids:
        d = _load("meta_scores", query, sid, METAPATH_GK, METAPATH_HOPS)
        if d:
            ids, sm = d
            all_graph_ids.update(ids)
            for pid, sim in sm.items():
                if sim > graph_score_map.get(pid, -1.0):
                    graph_score_map[pid] = sim

    # Reconstruct the reranker cache key (gt papers ∩ pool, sorted)
    pool = set(vec_ids) | all_graph_ids
    candidates = tuple(sorted(pid for pid in pool if pid in gt_pids))
    d = _load("rerank_scores", query, candidates, 10)
    rerank_score_map: dict[str, float] = d[1] if d else {}

    relevant = relevance_by_query.get(qi, set())
    scored_gt = gt_pids & (set(vec_ids) | set(graph_score_map) | set(rerank_score_map))

    return [
        (pid,
         vec_score_map.get(pid, 0.0),
         graph_score_map.get(pid, 0.0),
         rerank_score_map.get(pid, 0.0),
         1 if pid in relevant else 0)
        for pid in scored_gt
    ]


print("Collecting scores from cache...")
query_data: list[tuple[int, str, list]] = []   # (qi, query, rows)
for qi, query in enumerate(QUERIES, start=1):
    rows = collect_query(qi, query)
    if rows:
        n_pos = sum(r[4] for r in rows)
        query_data.append((qi, query, rows))
        print(f"  Q{qi:2d}: {len(rows):3d} papers  {n_pos:2d} relevant")
    else:
        print(f"  Q{qi:2d}: no cache  —  run evaluation/run_evaluation.py first")

if not query_data:
    sys.exit("No cached scores found. Run the evaluation first.")

# ── Fit global min-max scaler on ALL data ─────────────────────────────────────
all_feats = np.array(
    [r[1:4] for _, _, rows in query_data for r in rows], dtype=np.float32
)                                              # shape: (N, 3)
feat_min   = all_feats.min(axis=0)
feat_max   = all_feats.max(axis=0)
feat_range = np.where(feat_max > feat_min, feat_max - feat_min, 1.0)

def scale(X: np.ndarray) -> np.ndarray:
    """Apply the global min-max scaler; clamp to [0, 1] for unseen extremes."""
    return np.clip((X - feat_min) / feat_range, 0.0, 1.0)

print(f"\nGlobal feature ranges (raw values):")
print(f"  vec_score    [{feat_min[0]:.4f}, {feat_max[0]:.4f}]")
print(f"  graph_score  [{feat_min[1]:.4f}, {feat_max[1]:.4f}]")
print(f"  rerank_score [{feat_min[2]:.4f}, {feat_max[2]:.4f}]")

scaler_path = ROOT / "results" / "ranker_scaler.json"
json.dump({"min": feat_min.tolist(), "max": feat_max.tolist()},
          open(scaler_path, "w"), indent=2)

# ── Neural network ────────────────────────────────────────────────────────────
class RelevanceNN(nn.Module):
    """3 → 16 → 8 → 1  (ReLU hidden layers, Sigmoid output)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),  nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def make_tensors(rows_list: list[list]) -> tuple[torch.Tensor, torch.Tensor]:
    all_rows = [r for rows in rows_list for r in rows]
    X = scale(np.array([r[1:4] for r in all_rows], dtype=np.float32))
    y = np.array([r[4]        for r in all_rows], dtype=np.float32)
    return (torch.tensor(X, device=DEVICE),
            torch.tensor(y, device=DEVICE))

def train(X: torch.Tensor, y: torch.Tensor, epochs: int = 500) -> RelevanceNN:
    model = RelevanceNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Up-weight positives to handle class imbalance
    n_pos = max(y.sum().item(), 1)
    n_neg = len(y) - n_pos
    pos_w = torch.tensor([n_neg / n_pos], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    # Use pre-sigmoid logits: remove Sigmoid from last layer temporarily
    # Re-build model without sigmoid for BCEWithLogitsLoss
    model2 = nn.Sequential(
        nn.Linear(3, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 1),
    ).to(DEVICE)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-4)
    model2.train()
    for _ in range(epochs):
        opt2.zero_grad()
        logits = model2(X).squeeze(-1)
        loss   = criterion(logits, y)
        loss.backward()
        opt2.step()
    # Copy weights into the sigmoid model for inference
    for (name, p_src), (_, p_dst) in zip(
            model2.named_parameters(), model.named_parameters()):
        p_dst.data.copy_(p_src.data)
    return model

# ── Metrics ───────────────────────────────────────────────────────────────────
def mrr_at_5(ranked: list, relevant: set) -> float:
    for i, pid in enumerate(ranked[:5]):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0

def evaluate(model: RelevanceNN, qi: int, query: str) -> dict | None:
    rows = collect_query(qi, query)
    if not rows:
        return None
    pids  = [r[0] for r in rows]
    X     = torch.tensor(scale(np.array([r[1:4] for r in rows], dtype=np.float32)),
                         device=DEVICE)
    model.eval()
    with torch.no_grad():
        scores = model(X).cpu().numpy()
    ranked   = [pids[i] for i in np.argsort(scores)[::-1]]
    relevant = relevance_by_query.get(qi, set())
    return {
        "mrr5":     mrr_at_5(ranked, relevant),
        "recall5":  recall_at_k(ranked, relevant, 5),
        "recall10": recall_at_k(ranked, relevant, 10),
    }

# ── Leave-one-query-out cross-validation ─────────────────────────────────────
print("\n" + "="*60)
print("Leave-one-query-out cross-validation")
print("="*60)

loo_records = []
for hold_idx in range(len(query_data)):
    hold_qi, hold_query, _ = query_data[hold_idx]
    train_rows = [query_data[i][2] for i in range(len(query_data))
                  if i != hold_idx]
    X_tr, y_tr = make_tensors(train_rows)
    m = train(X_tr, y_tr, epochs=500)
    metrics = evaluate(m, hold_qi, hold_query)
    if metrics:
        loo_records.append({"query_id": hold_qi, **metrics})
        print(f"  Q{hold_qi:2d}  MRR@5={metrics['mrr5']:.3f}  "
              f"Recall@5={metrics['recall5']:.3f}  "
              f"Recall@10={metrics['recall10']:.3f}")

if loo_records:
    avg_mrr5    = float(np.mean([r["mrr5"]     for r in loo_records]))
    avg_recall5 = float(np.mean([r["recall5"]  for r in loo_records]))
    avg_recall10= float(np.mean([r["recall10"] for r in loo_records]))
    print(f"\n  Average   MRR@5={avg_mrr5:.4f}  "
          f"Recall@5={avg_recall5:.4f}  Recall@10={avg_recall10:.4f}")

# ── Train final model on all queries ─────────────────────────────────────────
print("\nTraining final model on all queries (epochs=1000)...")
X_all, y_all = make_tensors([qd[2] for qd in query_data])
final_model  = train(X_all, y_all, epochs=1000)

# ── Save ──────────────────────────────────────────────────────────────────────
model_path = ROOT / "results" / "ranker_model.pt"
torch.save(final_model.state_dict(), model_path)

loo_path = ROOT / "results" / "ranker_loo_results.csv"
with open(loo_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["query_id", "mrr5", "recall5", "recall10"])
    w.writeheader()
    w.writerows(loo_records)
    if loo_records:
        w.writerow({"query_id": "avg",
                    "mrr5":     round(avg_mrr5,   4),
                    "recall5":  round(avg_recall5, 4),
                    "recall10": round(avg_recall10,4)})

print(f"\nSaved:")
print(f"  {model_path}")
print(f"  {scaler_path}")
print(f"  {loo_path}")
