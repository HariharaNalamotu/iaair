# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

configs = ["V200/G4", "V150/G6", "V100/G9", "V80/G12", "V60/G16", "V40/G24", "V20/G49", "V10/G99", "V5/G199"]
avg_pool = [372, 354, 314, 309, 302, 306, 331, 357, 446]
total_rel = [91, 86, 73, 75, 77, 74, 66, 57, 79]

# Relevant by source
rel_vector =   [70, 62, 45, 39, 32, 26, 15,  8,  4]
rel_cites =    [12, 13, 20, 23, 26, 29, 33, 30, 47]
rel_coauthor = [ 5,  5,  2,  5,  6,  7,  6,  4,  1]
rel_venue =    [ 4,  6,  6,  8, 13, 12, 10, 12, 17]
rel_field =    [ 0,  0,  0,  0,  0,  0,  2,  3, 10]

x = np.arange(len(configs))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Constant Retrieval Budget (~1000): Vector vs Graph Trade-off", fontsize=15, fontweight="bold")

# Panel 1: Total relevant (bar + line)
ax = axes[0, 0]
bars = ax.bar(x, total_rel, color="steelblue", edgecolor="white", zorder=2)
ax.plot(x, total_rel, color="darkblue", marker="o", linewidth=2, zorder=3)
ax.set_ylabel("Total Relevant Found", fontsize=11)
ax.set_title("Total Relevant Papers (10 queries)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=9)
for i, v in enumerate(total_rel):
    ax.text(i, v + 1.2, str(v), ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)

# Panel 2: Relevant by source (stacked bar)
ax = axes[0, 1]
sources = ["vector", "cites", "coauthor", "venue", "field"]
data = [rel_vector, rel_cites, rel_coauthor, rel_venue, rel_field]
colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
bottom = np.zeros(len(configs))
for src, vals, color in zip(sources, data, colors):
    ax.bar(x, vals, bottom=bottom, label=src, color=color, edgecolor="white")
    bottom += np.array(vals)
ax.set_ylabel("Relevant Papers", fontsize=11)
ax.set_title("Relevant Papers by Source Channel", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=9)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Panel 3: Avg pool size
ax = axes[1, 0]
ax.bar(x, avg_pool, color="#78909C", edgecolor="white")
ax.plot(x, avg_pool, color="#37474F", marker="s", linewidth=2)
ax.set_ylabel("Avg Deduped Pool Size", fontsize=11)
ax.set_title("Average Pool Size per Query", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=9)
for i, v in enumerate(avg_pool):
    ax.text(i, v + 5, str(v), ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Panel 4: Relevance rate (relevant / pool) per config
ax = axes[1, 1]
rel_rate = [100 * r / (p * 10) for r, p in zip(total_rel, avg_pool)]  # per-query avg
ax.plot(x, rel_rate, color="steelblue", marker="o", linewidth=2.5, markersize=8)
ax.fill_between(x, rel_rate, alpha=0.15, color="steelblue")
ax.set_ylabel("Relevance Rate (%)", fontsize=11)
ax.set_title("Pool Relevance Rate (relevant / pool)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=9)
for i, v in enumerate(rel_rate):
    ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("constant_budget_sweep.png", dpi=150)
print("Saved constant_budget_sweep.png")
