import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SLOTS = ["#2a78d6", "#eb6834"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight"})
files, lp_table, out = sys.argv[1:4], sys.argv[4], sys.argv[5]
cur = pd.concat([pd.read_csv(f) for f in files])
lp = pd.read_csv(lp_table)
DS = ["pneumoniamnist", "breastmnist", "octmnist"]; NICE = {"pneumoniamnist": "PneumoniaMNIST (2 classes)", "breastmnist": "BreastMNIST (2 classes)", "octmnist": "OCTMNIST (4 classes)"}
LAB = {"zscore": "original blend (z-scores)", "logprob": "log-probability blend"}
fig, axes = plt.subplots(3, 3, figsize=(8.6, 7.0), sharex=True)
summary = []
for i, d in enumerate(DS):
    for j, k in enumerate([1, 4, 16]):
        ax = axes[i, j]; sub = cur[(cur.dataset == d) & (cur.K == k)]
        lo_all, hi_all = [], []
        for b, col in zip(["zscore", "logprob"], SLOTS):
            per = sub[sub.blend == b].groupby(["prompts", "draw", "eta"]).bacc.mean().unstack("eta")  # rows: prompt set x draw
            m = per.mean(); q1, q3 = per.quantile(0.25), per.quantile(0.75)
            ax.fill_between(m.index, q1, q3, color=col, alpha=0.15, lw=0)
            ax.plot(m.index, m.values, color=col, lw=2, label=LAB[b])
            lo_all.append(q1.min()); hi_all.append(q3.max())
            summary.append([d, k, b, m.idxmax(), m.max(), m.loc[0.0], m.loc[1.0]])
        pe = lp[(lp.dataset == d) & (lp.K == k) & (lp.rule == "PARA [zscore]")].eta.mean()
        ax.axvline(pe, color=INK, lw=1, ls=(0, (4, 3)))
        if i == 0: ax.set_title(f"K = {k}", loc="left")
        if j == 0: ax.set_ylabel(f"{NICE[d]}\nBAcc (%)")
        if i == 2: ax.set_xlabel("step η")
        pad = 0.08 * (max(hi_all) - min(lo_all)); ax.set_ylim(min(lo_all) - pad, max(hi_all) + 2.5 * pad)
        ax.text(pe, ax.get_ylim()[1], " PARA", fontsize=7, color=INK2, va="top", ha="left")
fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.965))
fig.suptitle("On 2-class data the original blend is a switch at η = 0.5 (flat on each side); the log-prob blend gives a smooth curve",
             x=0.01, ha="left", fontsize=9)
fig.text(0.01, -0.01, "Lines: mean over 10 support draws (and both prompt sets where available); bands: interquartile range across draws. Dashed: PARA's η at that K.\nη = 0 is the anchor alone, η = 1 the trained branch alone. The dip at exactly η = 0.5 (PneumoniaMNIST, K = 1) is a tie between the two branches.",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"F3_eta_curves.{ext}"), dpi=300)
s = pd.DataFrame(summary, columns=["dataset", "K", "blend", "best_eta", "best_bacc", "bacc_eta0", "bacc_eta1"])
print(s.round(2).to_string(index=False))
