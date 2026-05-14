"""
Visualise final_evaluation_sweep.csv results.

Main comparison uses AVERAGE across all vec_n mixes (vec_n < budget)
rather than optimal, to show typical rather than best-case performance.
Optimal-mix plots are included separately for reference.

Run:
    conda activate torchtest
    python /mnt/c/Users/harih/hybrid-graphrag/scripts/visualize_results.py

Outputs all figures to results/figures/
"""

import csv, pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = pathlib.Path(__file__).parent.parent
CSV     = ROOT / "results" / "final_evaluation_sweep.csv"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "mrr5":               "MRR@5",
    "recall5":            "Recall@5",
    "recall10":           "Recall@10",
    "n_relevant_in_pool": "Relevant Papers in Pool",
}

PALETTE = {
    "vector/reranker":     "#2196F3",
    "vector/none":         "#90CAF9",
    "bfs/reranker":        "#FF9800",
    "bfs/freq":            "#FFCC80",
    "bfs/interleave":      "#FFE0B2",
    "metapath/reranker":   "#4CAF50",
    "metapath/freq":       "#A5D6A7",
    "metapath/interleave": "#C8E6C9",
}
MARKERS = {
    "vector/reranker": "o", "vector/none": "s",
    "bfs/reranker": "^",    "bfs/freq": "v",    "bfs/interleave": "<",
    "metapath/reranker": "D","metapath/freq": "P","metapath/interleave": "X",
}

CONFIGS = [
    ("vector","reranker"), ("vector","none"),
    ("bfs","reranker"),    ("bfs","freq"),    ("bfs","interleave"),
    ("metapath","reranker"),("metapath","freq"),("metapath","interleave"),
]
KEY_CONFIGS = [("vector","reranker"), ("bfs","reranker"), ("metapath","reranker")]

def lbl(rt, rk): return f"{rt}/{rk}"
def save(fig, name):
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p.name}")

# ── Load and aggregate ────────────────────────────────────────────────────────
print("Loading CSV…")
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print(f"  {len(rows)} rows")

# Average per (budget, vec_n, type, ranker) over 10 queries
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (int(r["budget"]), int(r["vec_n"]), r["retrieval_type"], r["ranker"])
    for m in METRICS:
        agg[key][m].append(float(r[m]))

def avg(lst): return sum(lst)/len(lst)
pts = {k: {m: avg(v) for m, v in vals.items()} for k, vals in agg.items()}

budgets  = sorted(set(int(r["budget"]) for r in rows))
all_vecn = sorted(set(int(r["vec_n"]) for r in rows))

def vec_bl(b):  return pts.get((b, b, "vector", "reranker"))

# ── Average across ALL mixes (vec_n < budget) per (budget, config) ────────────
def avg_across_mixes(b, rt, rk, metric):
    """Mean metric value across every mix point (vec_n < budget)."""
    vals = [pts[(b,vn,rt,rk)][metric]
            for vn in all_vecn
            if vn < b and (b,vn,rt,rk) in pts]
    return avg(vals) if vals else None

# ── Best vec_n per (budget, config) by MRR@5 ─────────────────────────────────
def best_mix(b, rt, rk, metric="mrr5"):
    vns = [vn for vn in all_vecn if vn < b and (b,vn,rt,rk) in pts]
    if not vns: return None, None
    best_vn = max(vns, key=lambda vn: pts[(b,vn,rt,rk)][metric])
    return best_vn, pts[(b,best_vn,rt,rk)]

# =============================================================================
# Fig 1 — AVERAGE across all mixes: all 4 metrics, key reranker configs
# =============================================================================
print("\nFig 1: Average-across-all-mixes — 4 metrics, reranker configs…")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (metric, mlabel) in zip(axes.flatten(), METRICS.items()):
    for rt, rk in KEY_CONFIGS:
        l = lbl(rt, rk)
        xs, ys = [], []
        for b in budgets:
            if rt == "vector":
                v = pts.get((b, b, rt, rk))
                if v: xs.append(b); ys.append(v[metric])
            else:
                y = avg_across_mixes(b, rt, rk, metric)
                if y is not None: xs.append(b); ys.append(y)
        ax.plot(xs, ys, marker=MARKERS[l], color=PALETTE[l],
                label=l, linewidth=2, markersize=7)
    ax.set_title(mlabel, fontsize=12, fontweight="bold")
    ax.set_xlabel("Budget (pool size)"); ax.set_ylabel(mlabel)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(budgets)

fig.suptitle("Average Performance Across All Vec–Graph Mixes per Budget\n"
             "(graph configs averaged over all vec_n values, not cherry-picked optimal)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig1_avg_mix_4metrics.png")

# =============================================================================
# Fig 2 — AVERAGE: delta vs vector baseline
# =============================================================================
print("Fig 2: Average delta vs vector/reranker…")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (metric, mlabel) in zip(axes, list(METRICS.items())[:3]):
    for rt, rk in [("bfs","reranker"), ("metapath","reranker")]:
        l = lbl(rt, rk)
        xs, ys = [], []
        for b in budgets:
            bl = vec_bl(b)
            y  = avg_across_mixes(b, rt, rk, metric)
            if bl and y is not None:
                xs.append(b); ys.append(y - bl[metric])
        ax.plot(xs, ys, marker=MARKERS[l], color=PALETTE[l],
                label=l, linewidth=2, markersize=7)
        ax.fill_between(xs, ys, 0,
                        alpha=0.15, color=PALETTE[l])
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--",
               label="vector baseline")
    ax.set_title(f"Δ {mlabel}", fontweight="bold")
    ax.set_xlabel("Budget"); ax.set_ylabel(f"Δ {mlabel}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(budgets)

fig.suptitle("Average Delta vs Pure Vector — Across All Vec–Graph Mixes",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig2_avg_delta_vs_vector.png")

# =============================================================================
# Fig 3 — OPTIMAL mix: 4 metrics (for reference)
# =============================================================================
print("Fig 3: Optimal-mix — 4 metrics (for reference)…")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (metric, mlabel) in zip(axes.flatten(), METRICS.items()):
    for rt, rk in KEY_CONFIGS:
        l = lbl(rt, rk)
        xs, ys = [], []
        for b in budgets:
            if rt == "vector":
                v = pts.get((b, b, rt, rk))
                if v: xs.append(b); ys.append(v[metric])
            else:
                _, v = best_mix(b, rt, rk)
                if v: xs.append(b); ys.append(v[metric])
        ax.plot(xs, ys, marker=MARKERS[l], color=PALETTE[l],
                label=l, linewidth=2, markersize=7)
    ax.set_title(mlabel, fontsize=12, fontweight="bold")
    ax.set_xlabel("Budget"); ax.set_ylabel(mlabel)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(budgets)

fig.suptitle("Optimal-Mix Performance per Budget (best vec_n per config)\n"
             "— shown for reference only",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig3_optimal_mix_4metrics.png")

# =============================================================================
# Fig 4 — Full heatmaps: MRR@5 across (budget, vec_n)
# =============================================================================
print("Fig 4: MRR@5 heatmaps…")
for rt, rk in [("bfs","reranker"), ("metapath","reranker")]:
    l = lbl(rt, rk)
    vns_used = sorted(set(vn for (b,vn,t,r) in pts if t==rt and r==rk and vn<b))
    mat = np.full((len(vns_used), len(budgets)), np.nan)
    for bi, b in enumerate(budgets):
        for vi, vn in enumerate(vns_used):
            v = pts.get((b,vn,rt,rk))
            if v: mat[vi,bi] = v["mrr5"]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", cmap="YlGn", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels(budgets)
    ax.set_yticks(range(len(vns_used))); ax.set_yticklabels(vns_used)
    ax.set_xlabel("Budget"); ax.set_ylabel("vec_n")

    # Mark best per budget
    for bi, b in enumerate(budgets):
        best_vn, _ = best_mix(b, rt, rk)
        if best_vn and best_vn in vns_used:
            ax.plot(bi, vns_used.index(best_vn), "r*", markersize=14)

    # Mark average vec_n per budget (white circle)
    for bi, b in enumerate(budgets):
        vns = [vn for vn in vns_used if (b,vn,rt,rk) in pts]
        if vns:
            avg_vn = sum(vns)/len(vns)
            ax.axhline(vns_used.index(min(vns_used, key=lambda x: abs(x-avg_vn))),
                       color="white", alpha=0.3, linewidth=0.5)

    plt.colorbar(im, ax=ax, label="MRR@5")
    ax.set_title(f"MRR@5: {l}  (★ = best per budget, brighter = better)",
                 fontweight="bold")
    plt.tight_layout()
    save(fig, f"fig4_heatmap_{rt}_{rk}.png")

# =============================================================================
# Fig 5 — Winner heatmap: avg-mix comparison per metric per budget
# =============================================================================
print("Fig 5: Winner heatmap (avg mix)…")
THRESH = 0.005
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (rt, rk) in zip(axes, [("bfs","reranker"), ("metapath","reranker")]):
    l = lbl(rt, rk)
    mk_labels = list(METRICS.values())
    mk_keys   = list(METRICS.keys())
    mat = np.zeros((len(mk_keys), len(budgets)))
    for bi, b in enumerate(budgets):
        bl = vec_bl(b)
        if not bl: continue
        for mi, mk in enumerate(mk_keys):
            y = avg_across_mixes(b, rt, rk, mk)
            if y is None: continue
            diff = y - bl[mk]
            mat[mi,bi] = 1 if diff > THRESH else (-1 if diff < -THRESH else 0)

    cmap = matplotlib.colors.ListedColormap(["#E53935","#BDBDBD","#43A047"])
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels(budgets, rotation=45)
    ax.set_yticks(range(len(mk_keys))); ax.set_yticklabels(mk_labels)
    ax.set_title(f"{l} vs vector/reranker\n(avg across all mixes)\n"
                 "green=graph wins  grey=tie  red=vector wins",
                 fontweight="bold", fontsize=10)

    for mi in range(len(mk_keys)):
        for bi, b in enumerate(budgets):
            bl = vec_bl(b)
            y  = avg_across_mixes(b, rt, rk, mk_keys[mi])
            if bl and y:
                diff = y - bl[mk_keys[mi]]
                ax.text(bi, mi, f"{diff:+.3f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(diff)>0.04 else "black")

fig.suptitle("Graph Config vs Vector: Winner at Each Budget\n(averaged over all vec–graph mixes)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "fig5_winner_heatmap_avg.png")

# =============================================================================
# Fig 6 — MRR@5 vs vec fraction per budget (shape of the curve)
# =============================================================================
print("Fig 6: MRR@5 vs vec fraction per budget…")
ncols = 3
nrows = (len(budgets) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
axes = axes.flatten()
for ax, b in zip(axes, budgets):
    bl = vec_bl(b)
    for rt, rk in [("bfs","reranker"), ("metapath","reranker")]:
        l = lbl(rt, rk)
        vns = sorted(vn for (bb,vn,t,r) in pts if bb==b and t==rt and r==rk and vn<b)
        xs = [100*vn/b for vn in vns]
        ys = [pts[(b,vn,rt,rk)]["mrr5"] for vn in vns]
        ax.plot(xs, ys, marker=MARKERS[l], color=PALETTE[l],
                label=l, linewidth=1.5, markersize=5)
    if bl:
        ax.axhline(bl["mrr5"], color=PALETTE["vector/reranker"],
                   linestyle="--", linewidth=1.5, label="vector/reranker")
    ax.set_title(f"Budget = {b}", fontweight="bold")
    ax.set_xlabel("Vector fraction (%)"); ax.set_ylabel("MRR@5")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0, 100)
for ax in axes[len(budgets):]:
    ax.set_visible(False)
fig.suptitle("MRR@5 vs Vector–Graph Mix Fraction per Budget",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig6_mrr5_vs_vecfrac.png")

# =============================================================================
# Fig 7 — All configs grouped bar: AVERAGE MRR@5
# =============================================================================
print("Fig 7: All configs grouped bar — average MRR@5…")
fig, ax = plt.subplots(figsize=(16, 6))
n_cfg    = len(CONFIGS)
bw       = 0.8 / n_cfg
x        = np.arange(len(budgets))

for i, (rt, rk) in enumerate(CONFIGS):
    l = lbl(rt, rk)
    ys = []
    for b in budgets:
        if rt == "vector":
            v = pts.get((b, b, rt, rk))
            ys.append(v["mrr5"] if v else 0)
        else:
            y = avg_across_mixes(b, rt, rk, "mrr5")
            ys.append(y if y else 0)
    offset = (i - n_cfg/2 + 0.5) * bw
    ax.bar(x + offset, ys, bw*0.9, label=l,
           color=PALETTE[l], edgecolor="white", linewidth=0.5)

ax.set_xticks(x); ax.set_xticklabels(budgets)
ax.set_xlabel("Budget"); ax.set_ylabel("MRR@5")
ax.set_title("Average MRR@5 Across All Mixes — All Configs per Budget\n"
             "(graph configs averaged over all vec_n values)",
             fontweight="bold")
ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 1.1)
plt.tight_layout()
save(fig, "fig7_all_configs_avg_mrr5_bar.png")

# =============================================================================
# Fig 8 — Recall@10 vs Relevant/pool scatter (avg mix)
# =============================================================================
print("Fig 8: Recall@10 vs Rel/pool scatter (avg mix)…")
fig, ax = plt.subplots(figsize=(9, 6))
for rt, rk in KEY_CONFIGS:
    l = lbl(rt, rk)
    xs, ys, labels = [], [], []
    for b in budgets:
        if rt == "vector":
            v = pts.get((b, b, rt, rk))
            if v: xs.append(v["n_relevant_in_pool"]); ys.append(v["recall10"]); labels.append(str(b))
        else:
            r10 = avg_across_mixes(b, rt, rk, "recall10")
            rp  = avg_across_mixes(b, rt, rk, "n_relevant_in_pool")
            if r10 and rp: xs.append(rp); ys.append(r10); labels.append(str(b))
    ax.scatter(xs, ys, c=PALETTE[l], marker=MARKERS[l], s=80, label=l, zorder=3)
    for x_, y_, lb in zip(xs, ys, labels):
        ax.annotate(lb, (x_, y_), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color=PALETTE[l])

ax.set_xlabel("Avg Relevant Papers in Pool")
ax.set_ylabel("Recall@10")
ax.set_title("Recall@10 vs Relevant Papers in Pool (avg across all mixes)\n"
             "Each point = one budget, labelled", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
save(fig, "fig8_recall_vs_relpool_avg.png")

print(f"\nAll figures saved to: {FIG_DIR}")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024} KB)")
