#!/usr/bin/env python3
"""
Train a FFNN to learn optimal combination weights for the metapath weighted ranker.

Run AFTER the evaluation has been executed at least once so the score caches
(vec_scores, meta_scores, rerank_scores) are populated.

    conda activate torchtest
    python /mnt/c/Users/harih/hybrid-graphrag/scripts/train_ranker.py

What it does:
  1. Loads cached (vec_score, graph_score, rerank_score) for every ground-truth
     paper that was retrieved, across all 10 queries.
  2. Pairs scores with binary relevance labels.
  3. Trains two models with leave-one-query-out (LOOQ) cross-validation:
       Linear  — 3 → 1, no activation  (equivalent to finding optimal α, β, γ)
       FFNN    — 3 → 16 → 8 → 1, ReLU + Sigmoid
  4. Reports MRR@5 and Recall@5 for each model vs the fixed-weight baseline.
  5. Saves:
       results/ranker_linear_weights.json  — optimal α, β, γ
       results/ranker_model.pt             — full FFNN state dict

Usage after training:
  Update WEIGHT_ALPHA/BETA/GAMMA in evaluation/run_evaluation.py with the
  values printed by this script, then re-run the evaluation.
"""

import csv, hashlib, json, os, pathlib, pickle, sys, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "retrieval"))

CACHE_DIR = ROOT / "results" / "eval_cache"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'})")

csv.field_size_limit(10 * 1024 * 1024)

# ── Config (must match evaluation/run_evaluation.py) ─────────────────────────
METAPATH_VEC = 200
METAPATH_GK  = 5
METAPATH_HOPS = 4

WEIGHT_ALPHA_FIXED = 0.4
WEIGHT_BETA_FIXED  = 0.2
WEIGHT_GAMMA_FIXED = 0.4

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

# ── Cache utilities (mirrors run_evaluation.py) ───────────────────────────────
def _cache_key(*args):
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()

def _cache_get(fn, *args):
    p = CACHE_DIR / f"{fn}_{_cache_key(*args)}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return True, pickle.load(f)
    return False, None

# ── Load relevance labels ─────────────────────────────────────────────────────
print("Loading relevance labels...")
relevance_by_query = {}
gt_pids = set()
with open(ROOT / "data" / "ground_truth_relevance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        qi = int(row["query_id"])
        gt_pids.add(row["paperId"])
        if qi not in relevance_by_query:
            relevance_by_query[qi] = set()
        if int(row["relevant"]) == 1:
            relevance_by_query[qi].add(row["paperId"])
print(f"  {len(gt_pids)} ground-truth papers, {sum(len(v) for v in relevance_by_query.values())} relevant labels")

# ── Collect scores from cache for each query ─────────────────────────────────
print("\nCollecting scores from evaluation cache...")

def collect_query_scores(qi, query):
    """
    Returns list of (vec_score, graph_score, rerank_score, relevant) tuples
    for every ground-truth paper retrieved for this query.
    Scores default to 0.0 if the paper was not reached by that source.
    """
    # Vector scores
    hit, data = _cache_get("vec_scores", query, METAPATH_VEC)
    if not hit:
        print(f"  [Q{qi}] vec_scores cache MISS — run evaluation/run_evaluation.py first")
        return []
    vec_ids, vec_scores = data

    # Metapath graph scores (aggregate across all seeds)
    graph_scores: dict[str, float] = {}
    for sid in vec_ids:
        hit, data = _cache_get("meta_scores", query, sid, METAPATH_GK, METAPATH_HOPS)
        if hit:
            _, score_map = data
            for pid, sim in score_map.items():
                if pid not in graph_scores or sim > graph_scores[pid]:
                    graph_scores[pid] = sim

    # Reranker scores — keyed on the full candidate set, so we need to know what
    # was passed to do_rerank_with_scores. It was called with all_ids from
    # _unique_pool(vec_ids, graph_results). We reconstruct all_ids from the cache.
    all_ids_set = set(vec_ids)
    for sid in vec_ids:
        hit, data = _cache_get("meta_scores", query, sid, METAPATH_GK, METAPATH_HOPS)
        if hit:
            ids, _ = data
            all_ids_set.update(ids)
    all_ids = list(all_ids_set)  # order doesn't matter for rerank key

    # The rerank cache key uses the gt-filtered candidate set (gt_pids ∩ all_ids)
    candidates = sorted(pid for pid in all_ids if pid in gt_pids)
    hit, data = _cache_get("rerank_scores", query, tuple(candidates), 10)
    rerank_scores: dict[str, float] = {}
    if hit:
        _, rerank_scores = data

    relevant_set = relevance_by_query.get(qi, set())

    # Build per-paper feature rows for every gt paper that has at least one score
    rows = []
    scored_gt = gt_pids & (set(vec_ids) | set(graph_scores) | set(rerank_scores))
    for pid in scored_gt:
        v = vec_scores.get(pid, 0.0)
        g = graph_scores.get(pid, 0.0)
        r = rerank_scores.get(pid, 0.0)
        rows.append((v, g, r, 1 if pid in relevant_set else 0))

    return rows

# Per-query data: list of (n_papers × 4) arrays
query_data = []
for qi, query in enumerate(QUERIES, start=1):
    rows = collect_query_scores(qi, query)
    if not rows:
        print(f"  Q{qi}: no data — skipping")
        query_data.append(None)
    else:
        arr = np.array(rows, dtype=np.float32)
        n_pos = int(arr[:, 3].sum())
        query_data.append(arr)
        print(f"  Q{qi}: {len(rows)} papers, {n_pos} relevant")

valid_queries = [(qi, d) for qi, d in enumerate(query_data) if d is not None]
print(f"\n{len(valid_queries)} queries with cached data.")

if len(valid_queries) == 0:
    sys.exit("No cached scores found. Run the evaluation first.")

# ── Per-query min-max normalisation (mirrors _weighted_rank) ─────────────────
def minmax_normalise(arr):
    """Normalise each of the 3 score columns independently to [0, 1]."""
    out = arr.copy()
    for col in range(3):
        lo, hi = arr[:, col].min(), arr[:, col].max()
        span = hi - lo if hi != lo else 1.0
        out[:, col] = (arr[:, col] - lo) / span
    return out

query_data_norm = []
for qi, d in valid_queries:
    query_data_norm.append((qi, minmax_normalise(d)))

# ── Models ────────────────────────────────────────────────────────────────────
class LinearRanker(nn.Module):
    """Single linear layer — equivalent to α·v + β·g + γ·r (plus bias)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1, bias=True)
        nn.init.constant_(self.fc.weight, 1/3)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class FFNNRanker(nn.Module):
    """Small FFNN: 3 → 16 → 8 → 1 with ReLU activations and sigmoid output."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),  nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ── Training utilities ────────────────────────────────────────────────────────
def build_pairs(data_list):
    """
    Build pairwise training data from a list of (qi, normalised_array) tuples.
    For each query, form all (positive, negative) paper pairs.
    Returns (pos_features, neg_features) tensors on DEVICE.
    """
    pos_rows, neg_rows = [], []
    for qi, arr in data_list:
        feats = arr[:, :3]
        labels = arr[:, 3]
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        for pi in pos_idx:
            for ni in neg_idx:
                pos_rows.append(feats[pi])
                neg_rows.append(feats[ni])
    if not pos_rows:
        return None, None
    return (torch.tensor(np.array(pos_rows), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(neg_rows), dtype=torch.float32, device=DEVICE))


def bpr_loss(pos_scores, neg_scores):
    """Bayesian Personalised Ranking loss: maximise P(pos > neg)."""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


def train_model(model, train_data, epochs=500, lr=1e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos, neg = build_pairs(train_data)
    if pos is None:
        return model
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = bpr_loss(model(pos), model(neg))
        loss.backward()
        opt.step()
    return model


def score_papers(model, arr):
    """Score all papers in arr (shape N×4) and return scores as numpy array."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(arr[:, :3], dtype=torch.float32, device=DEVICE)
        return model(x).cpu().numpy()


def fixed_weight_scores(arr, alpha, beta, gamma):
    return alpha * arr[:, 0] + beta * arr[:, 1] + gamma * arr[:, 2]

# ── Metrics ───────────────────────────────────────────────────────────────────
def mrr_at_5(ranked_ids, relevant_ids):
    for i, pid in enumerate(ranked_ids[:5]):
        if pid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)

def evaluate_on_query(scores_arr, gt_pids_for_query, relevant_set):
    """Given per-paper scores and a list of pids, compute MRR@5 and Recall@5/10."""
    # scores_arr is aligned with valid_queries[i] rows, which are scored_gt papers
    # We need the paper IDs for this query — reconstruct them
    return scores_arr  # caller handles ranking

def evaluate_query(qi_query, arr_norm, scores_np):
    qi, query = qi_query
    relevant = relevance_by_query.get(qi + 1, set())
    # Reconstruct pid list by re-running cache lookups (fast: already cached)
    hit, data = _cache_get("vec_scores", query, METAPATH_VEC)
    if not hit:
        return None
    vec_ids, vec_scores = data
    all_ids_set = set(vec_ids)
    for sid in vec_ids:
        hit2, data2 = _cache_get("meta_scores", query, sid, METAPATH_GK, METAPATH_HOPS)
        if hit2:
            ids2, _ = data2
            all_ids_set.update(ids2)
    candidates = sorted(pid for pid in all_ids_set if pid in gt_pids)
    hit3, data3 = _cache_get("rerank_scores", query, tuple(candidates), 10)
    rerank_scores = data3[1] if hit3 else {}

    graph_scores_q: dict[str, float] = {}
    for sid in vec_ids:
        hit4, data4 = _cache_get("meta_scores", query, sid, METAPATH_GK, METAPATH_HOPS)
        if hit4:
            _, sm = data4
            for pid, sim in sm.items():
                if pid not in graph_scores_q or sim > graph_scores_q[pid]:
                    graph_scores_q[pid] = sim

    # Build scored_gt list (same order as collect_query_scores)
    scored_gt = gt_pids & (set(vec_ids) | set(graph_scores_q) | set(rerank_scores))
    scored_gt_list = list(scored_gt)

    # Map pid → score index
    pid_to_idx = {pid: i for i, pid in enumerate(scored_gt_list)}
    ranked = sorted(scored_gt_list, key=lambda p: scores_np[pid_to_idx[p]], reverse=True)

    return (mrr_at_5(ranked, relevant),
            recall_at_k(ranked, relevant, 5),
            recall_at_k(ranked, relevant, 10))


# ── Leave-one-query-out cross-validation ─────────────────────────────────────
print("\n" + "="*60)
print("Leave-one-query-out cross-validation")
print("="*60)

loo_results = {"linear": [], "ffnn": [], "fixed": []}
best_linear_weights = None  # from all-data training

for hold_out_idx in range(len(valid_queries)):
    hold_qi, hold_arr = valid_queries[hold_out_idx]
    hold_norm = query_data_norm[hold_out_idx][1]
    train_data = [query_data_norm[i] for i in range(len(valid_queries))
                  if i != hold_out_idx]

    # Train linear model
    linear = train_model(LinearRanker(), train_data, epochs=1000, lr=5e-3)
    # Train FFNN
    ffnn   = train_model(FFNNRanker(),   train_data, epochs=1000, lr=1e-3)

    # Score the held-out query
    lin_scores   = score_papers(linear, hold_norm)
    ffnn_scores  = score_papers(ffnn,   hold_norm)
    fixed_scores = fixed_weight_scores(
        hold_norm, WEIGHT_ALPHA_FIXED, WEIGHT_BETA_FIXED, WEIGHT_GAMMA_FIXED
    )

    # Evaluate — need pid list for the held-out query
    qi_query = (hold_qi, QUERIES[hold_qi])
    r_lin   = evaluate_query(qi_query, hold_norm, lin_scores)
    r_ffnn  = evaluate_query(qi_query, hold_norm, ffnn_scores)
    r_fixed = evaluate_query(qi_query, hold_norm, fixed_scores)

    if r_lin:
        loo_results["linear"].append(r_lin)
        loo_results["ffnn"].append(r_ffnn)
        loo_results["fixed"].append(r_fixed)
        print(f"  Q{hold_qi+1:2d}  "
              f"fixed=MRR:{r_fixed[0]:.3f}/R5:{r_fixed[1]:.3f}  "
              f"linear=MRR:{r_lin[0]:.3f}/R5:{r_lin[1]:.3f}  "
              f"ffnn=MRR:{r_ffnn[0]:.3f}/R5:{r_ffnn[1]:.3f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LOO-CV Summary (averaged across held-out queries)")
print("="*60)
for name in ["fixed", "linear", "ffnn"]:
    r = np.array(loo_results[name])
    if len(r):
        print(f"  {name:8s}  MRR@5={r[:,0].mean():.4f}  "
              f"Recall@5={r[:,1].mean():.4f}  Recall@10={r[:,2].mean():.4f}")

# ── Train final models on ALL queries ────────────────────────────────────────
print("\nTraining final models on all queries...")
linear_final = train_model(LinearRanker(), query_data_norm, epochs=2000, lr=5e-3)
ffnn_final   = train_model(FFNNRanker(),   query_data_norm, epochs=2000, lr=1e-3)

# Extract linear weights (α, β, γ)
w = linear_final.fc.weight.data.cpu().numpy().flatten()
b = float(linear_final.fc.bias.data.cpu())

# Normalise to sum to 1 (drop bias for interpretability)
w_pos = np.abs(w)
w_norm = w_pos / w_pos.sum()
alpha, beta, gamma = float(w_norm[0]), float(w_norm[1]), float(w_norm[2])

print(f"\nLinear model learned weights (raw):  {w.tolist()}")
print(f"Linear model weights (L1-normalised): α={alpha:.4f}  β={beta:.4f}  γ={gamma:.4f}")
print(f"  (was: α={WEIGHT_ALPHA_FIXED}  β={WEIGHT_BETA_FIXED}  γ={WEIGHT_GAMMA_FIXED})")

# ── Save results ──────────────────────────────────────────────────────────────
results_dir = ROOT / "results"
weights_path = results_dir / "ranker_linear_weights.json"
model_path   = results_dir / "ranker_model.pt"

with open(weights_path, "w") as f:
    json.dump({
        "WEIGHT_ALPHA": alpha,
        "WEIGHT_BETA":  beta,
        "WEIGHT_GAMMA": gamma,
        "raw_weights":  w.tolist(),
        "bias":         b,
        "loo_cv": {
            name: {"avg_mrr5": float(np.mean(np.array(v)[:, 0])),
                   "avg_recall5": float(np.mean(np.array(v)[:, 1]))}
            for name, v in loo_results.items() if v
        },
    }, f, indent=2)

torch.save(ffnn_final.state_dict(), model_path)

print(f"\nSaved:")
print(f"  {weights_path}")
print(f"  {model_path}")
print(f"\nTo apply: update WEIGHT_ALPHA/BETA/GAMMA in evaluation/run_evaluation.py:")
print(f"  WEIGHT_ALPHA = {alpha:.4f}")
print(f"  WEIGHT_BETA  = {beta:.4f}")
print(f"  WEIGHT_GAMMA = {gamma:.4f}")
