"""
Visualise final_evaluation_sweep.csv results.

Run:
    conda activate torchtest
    python /mnt/c/Users/harih/hybrid-graphrag/scripts/visualize_results.py

Outputs all figures to results/figures/
"""

import csv, pathlib, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
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

# Colours consistent across all plots
PALETTE = {
    "vector/reranker":     "#2196F3",   # blue
    "vector/none":         "#90CAF9",   # light blue
    "bfs/reranker":        "#FF9800",   # orange
    "bfs/freq":            "#FFCC80",   # light orange
    "bfs/interleave":      "#FFE0B2",   # very light orange
    "metapath/reranker":   "#4CAF50",   # green
    "metapath/freq":       "#A5D6A7",   # light green
    "metapath/interleave": "#C8E6C9",   # very light green
}

MARKERS = {
    "vector/reranker":     "o",
    "vector/none":         "s",
    "bfs/reranker":        "^",
    "bfs/freq":            "v",
    "bfs/interleave":      "<",
    "metapath/reranker":   "D",
    "metapath/freq":       "P",
    "metapath/interleave": "X",
}

def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading CSV…")
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print(f"  {len(rows)} rows")

agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    key = (int(r["budget"]), int(r["vec_n"]), r["retrieval_type"], r["ranker"])
    for m in METRICS:
        agg[key][m].append(float(r[m]))
    agg[key]["pool"].append(float(r["actual_avg_pool"]))

def avg(lst): return sum(lst) / len(lst)
pts = {k: {m: avg(v) for m, v in vals.items()} for k, vals in agg.items()}

budgets  = sorted(set(int(r["budget"]) for r in rows))
all_vecn = sorted(set(int(r["vec_n"]) for r in rows))

CONFIGS = [
    ("vector",   "reranker"),
    ("vector",   "none"),
    ("bfs",      "reranker"),
    ("bfs",      "freq"),
    ("bfs",      "interleave"),
    ("metapath", "reranker"),
    ("metapath", "freq"),
    ("metapath", "interleave"),
]

def label(rt, rk): return f"{rt}/{rk}"
def vec_baseline(b): return pts.get((b, b, "vector", "reranker"))

# Best vec_n per (budget, config) by MRR@5
def best_for(b, rt, rk, metric="mrr5"):
    vns = [vn for vn in all_vecn if (b, vn, rt, rk) in pts and vn < b]
    if not vns: return None, None
    best_vn = max(vns, key=lambda vn: pts[(b, vn, rt, rk)][metric])
    return best_vn, pts[(b, best_vn, rt, rk)]

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Best metric values per budget — one subplot per metric
# (only the three reranker configs + vector baseline, for clarity)
# ─────────────────────────────────────────────────────────────────────────────
print("\nFig 1: Best metric per budget (reranker configs)…")
KEY_CONFIGS = [
    ("vector",   "reranker"),
    ("bfs",      "reranker"),
    ("metapath", "reranker"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (metric, metric_label) in zip(axes, METRICS.items()):
    for rt, rk in KEY_CONFIGS:
        lbl = label(rt, rk)
        ys, xs = [], []
        for b in budgets:
            # include the pure-vector point (vec_n == budget)
            if rt == "vector":
                v = pts.get((b, b, rt, rk))
                if v: xs.append(b); ys.append(v[metric])
            else:
                _, v = best_for(b, rt, rk)
                if v: xs.append(b); ys.append(v[metric])
        ax.plot(xs, ys, marker=MARKERS[lbl], color=PALETTE[lbl],
                label=lbl, linewidth=2, markersize=7)
    ax.set_title(metric_label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Budget (deduped pool size)")
    ax.set_ylabel(metric_label)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(budgets)

fig.suptitle("Best Performance per Budget — Reranker Configs vs Pure Vector",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save(fig, "fig1_best_metric_per_budget.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Optimal vec_n and vec fraction per budget
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: Optimal vec_n and vec fraction per budget…")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for rt, rk in [("bfs","reranker"), ("metapath","reranker")]:
    lbl = label(rt, rk)
    vecns, fracs, xs = [], [], []
    for b in budgets:
        vn, _ = best_for(b, rt, rk)
        if vn:
            xs.append(b); vecns.append(vn); fracs.append(100 * vn / b)
    ax1.plot(xs, vecns,  marker=MARKERS[lbl], color=PALETTE[lbl],
             label=lbl, linewidth=2, markersize=7)
    ax2.plot(xs, fracs, marker=MARKERS[lbl], color=PALETTE[lbl],
             label=lbl, linewidth=2, markersize=7)

ax1.axhline(150, color="grey", linestyle="--", alpha=0.5, label="vec_n=150")
ax1.set_title("Optimal vec_n (absolute) per Budget", fontweight="bold")
ax1.set_xlabel("Budget"); ax1.set_ylabel("Optimal vec_n")
ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xticks(budgets)

ax2.set_title("Optimal Vector Fraction per Budget", fontweight="bold")
ax2.set_xlabel("Budget"); ax2.set_ylabel("Vector fraction (%)")
ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_xticks(budgets)

fig.suptitle("Optimal Vector–Graph Mix per Budget", fontsize=13,
             fontweight="bold")
plt.tight_layout()
save(fig, "fig2_optimal_mix_per_budget.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Full sweep heatmaps — MRR@5 across (budget, vec_n) for each config
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: Heatmaps of MRR@5 across budget × vec_n…")
for rt, rk in [("bfs","reranker"), ("metapath","reranker")]:
    lbl = label(rt, rk)
    # Build matrix: rows = vec_n, cols = budget
    vns_used = sorted(set(vn for (b,vn,t,r) in pts if t==rt and r==rk and vn<b))
    mat = np.full((len(vns_used), len(budgets)), np.nan)
    for bi, b in enumerate(budgets):
        for vi, vn in enumerate(vns_used):
            v = pts.get((b, vn, rt, rk))
            if v: mat[vi, bi] = v["mrr5"]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", cmap="YlGn", vmin=0, vmax=1,
                   origin="lower")
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels(budgets)
    ax.set_yticks(range(len(vns_used))); ax.set_yticklabels(vns_used)
    ax.set_xlabel("Budget"); ax.set_ylabel("vec_n")

    # Mark the best vec_n per budget with a red star
    for bi, b in enumerate(budgets):
        best_vn, _ = best_for(b, rt, rk)
        if best_vn and best_vn in vns_used:
            vi = vns_used.index(best_vn)
            ax.plot(bi, vi, "r*", markersize=14)

    plt.colorbar(im, ax=ax, label="MRR@5")
    ax.set_title(f"MRR@5 Heatmap: {lbl}  (★ = best vec_n per budget)",
                 fontweight="bold")
    plt.tight_layout()
    save(fig, f"fig3_heatmap_mrr5_{rt}_{rk}.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Delta vs vector baseline — all reranker configs at best mix
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: Delta vs vector/reranker at best mix…")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
delta_metrics = [("mrr5","MRR@5"), ("recall10","Recall@10"), ("n_relevant_in_pool","Rel/pool")]

for ax, (metric, mlabel) in zip(axes, delta_metrics):
    for rt, rk in [("bfs","reranker"), ("metapath","reranker")]:
        lbl = label(rt, rk)
        xs, ys = [], []
        for b in budgets:
            bl = vec_baseline(b)
            _, v = best_for(b, rt, rk, metric)
            if bl and v:
                xs.append(b)
                ys.append(v[metric] - bl[metric])
        ax.plot(xs, ys, marker=MARKERS[lbl], color=PALETTE[lbl],
                label=lbl, linewidth=2, markersize=7)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", label="vector baseline")
    ax.fill_between(budgets, 0, 0, alpha=0)
    ax.set_title(f"Δ {mlabel} vs vector/reranker", fontweight="bold")
    ax.set_xlabel("Budget"); ax.set_ylabel(f"Δ {mlabel}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xticks(budgets)

fig.suptitle("Performance Delta vs Pure Vector Baseline at Optimal Mix",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig4_delta_vs_vector.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Winner per metric per budget — heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: Winner heatmap per metric per budget…")
COMPARE = [
    ("bfs",      "reranker"),
    ("metapath", "reranker"),
]
THRESHOLD = 0.005   # must beat by at least this to count as a win

fig, axes = plt.subplots(1, len(COMPARE), figsize=(14, 5))
for ax, (rt, rk) in zip(axes, COMPARE):
    lbl = label(rt, rk)
    metric_labels = list(METRICS.values())
    metric_keys   = list(METRICS.keys())
    mat = np.zeros((len(metric_keys), len(budgets)))  # 1=graph wins, -1=vec wins, 0=tie

    for bi, b in enumerate(budgets):
        bl = vec_baseline(b)
        _, v = best_for(b, rt, rk)
        if not (bl and v): continue
        for mi, mk in enumerate(metric_keys):
            diff = v[mk] - bl[mk]
            if diff > THRESHOLD:   mat[mi, bi] =  1
            elif diff < -THRESHOLD: mat[mi, bi] = -1
            else:                   mat[mi, bi] =  0

    cmap = matplotlib.colors.ListedColormap(["#E53935","#BDBDBD","#43A047"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels(budgets, rotation=45)
    ax.set_yticks(range(len(metric_keys))); ax.set_yticklabels(metric_labels)
    ax.set_title(f"{lbl}\n(green=graph wins, red=vector wins, grey=tie)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Budget")

    # Add text values
    for mi in range(len(metric_keys)):
        for bi, b in enumerate(budgets):
            bl = vec_baseline(b)
            _, v = best_for(b, rt, rk)
            if bl and v:
                diff = v[metric_keys[mi]] - bl[metric_keys[mi]]
                ax.text(bi, mi, f"{diff:+.3f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(diff)>0.05 else "black")

fig.suptitle("Graph Config vs Vector Baseline — Which Wins at Each Budget?",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "fig5_winner_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: MRR@5 vs vec fraction — one subplot per budget
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6: MRR@5 vs vec fraction per budget…")
n_budgets = len(budgets)
ncols = 3
nrows = (n_budgets + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
axes = axes.flatten()

for ax, b in zip(axes, budgets):
    bl = vec_baseline(b)
    for rt, rk in [("bfs","reranker"), ("metapath","reranker")]:
        lbl = label(rt, rk)
        vns = sorted(vn for (bb,vn,t,r) in pts if bb==b and t==rt and r==rk and vn<b)
        xs = [100*vn/b for vn in vns]
        ys = [pts[(b,vn,rt,rk)]["mrr5"] for vn in vns]
        ax.plot(xs, ys, marker=MARKERS[lbl], color=PALETTE[lbl],
                label=lbl, linewidth=1.5, markersize=5)
    if bl:
        ax.axhline(bl["mrr5"], color=PALETTE["vector/reranker"],
                   linestyle="--", linewidth=1.5, label="vector/reranker")
    ax.set_title(f"Budget = {b}", fontweight="bold")
    ax.set_xlabel("Vector fraction (%)")
    ax.set_ylabel("MRR@5")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

# Hide unused subplots
for ax in axes[len(budgets):]:
    ax.set_visible(False)

fig.suptitle("MRR@5 vs Vector–Graph Mix Fraction per Budget",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "fig6_mrr5_vs_vecfrac_per_budget.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: All 8 configs at each budget — grouped bar chart for MRR@5
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7: All configs grouped bar chart — MRR@5…")
fig, ax = plt.subplots(figsize=(16, 6))
n_configs = len(CONFIGS)
bar_width = 0.8 / n_configs
x = np.arange(len(budgets))

for i, (rt, rk) in enumerate(CONFIGS):
    lbl = label(rt, rk)
    ys = []
    for b in budgets:
        if rt == "vector":
            v = pts.get((b, b, rt, rk))
            ys.append(v["mrr5"] if v else 0)
        else:
            _, v = best_for(b, rt, rk)
            ys.append(v["mrr5"] if v else 0)
    offset = (i - n_configs/2 + 0.5) * bar_width
    ax.bar(x + offset, ys, bar_width*0.9, label=lbl,
           color=PALETTE[lbl], edgecolor="white", linewidth=0.5)

ax.set_xticks(x); ax.set_xticklabels(budgets)
ax.set_xlabel("Budget"); ax.set_ylabel("MRR@5")
ax.set_title("MRR@5 — All Configs at Best Mix per Budget", fontweight="bold")
ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 1.1)
plt.tight_layout()
save(fig, "fig7_all_configs_mrr5_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Recall@10 vs Rel/pool scatter — does more pool = more recall?
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 8: Recall@10 vs Rel/pool scatter…")
fig, ax = plt.subplots(figsize=(9, 6))
for rt, rk in KEY_CONFIGS:
    lbl = label(rt, rk)
    xs, ys, bud_labels = [], [], []
    for b in budgets:
        if rt == "vector":
            v = pts.get((b, b, rt, rk))
        else:
            _, v = best_for(b, rt, rk)
        if v:
            xs.append(v["n_relevant_in_pool"]); ys.append(v["recall10"])
            bud_labels.append(str(b))
    sc = ax.scatter(xs, ys, c=PALETTE[lbl], marker=MARKERS[lbl],
                    s=80, label=lbl, zorder=3)
    for x_, y_, bl in zip(xs, ys, bud_labels):
        ax.annotate(bl, (x_, y_), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color=PALETTE[lbl])

ax.set_xlabel("Avg Relevant Papers in Pool"); ax.set_ylabel("Recall@10")
ax.set_title("Recall@10 vs Relevant Papers in Pool\n(each point = one budget, labeled)",
             fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
save(fig, "fig8_recall_vs_relpool_scatter.png")

print(f"\nAll figures saved to: {FIG_DIR}")
print("Files:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.name}  ({f.stat().st_size//1024} KB)")
