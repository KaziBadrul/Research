"""Paper figures F1, F2, F4, F5 from the saved PARA results (no training).
Inputs: the extracted para_results folder and logprob_table.csv. Outputs: PNG (300 dpi) + PDF per figure."""
import glob, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = sys.argv[1]            # para_results folder
LP_TABLE = sys.argv[2]       # logprob_table.csv
OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)

# ---- style: reference palette (validated light mode), recessive axes, text in ink tokens
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#9a9994"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "text.color": INK, "axes.titlecolor": INK, "savefig.bbox": "tight",
})

DATASETS = ["dermamnist", "octmnist", "bloodmnist", "pneumoniamnist", "breastmnist", "organamnist"]
NICE = {"dermamnist": "DermaMNIST", "octmnist": "OCTMNIST", "bloodmnist": "BloodMNIST",
        "pneumoniamnist": "PneumoniaMNIST", "breastmnist": "BreastMNIST", "organamnist": "OrganAMNIST"}
HELD = {"pneumoniamnist", "breastmnist", "organamnist"}
COLOR = {d: SLOTS[i] for i, d in enumerate(DATASETS)}          # color follows the dataset, fixed order
MARK = {"dermamnist": "o", "octmnist": "s", "bloodmnist": "D", "pneumoniamnist": "^", "breastmnist": "v", "organamnist": "P"}
LAMBDA = 0.9

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("saved", name)

# ---------------------------------------------------------------- parse the 14.2 geometry reports
def parse_geometry(path):
    txt = open(path, encoding="utf-8").read()
    out = {}
    for space, block in zip(["raw", "centered"], re.split(r"######## CENTERED", txt)):
        d = re.search(r"own-text delta ([\d.]+) vs other-text delta ([\d.]+)", block)
        p = re.search(r"phi: real classes ([\d.]+) \(max possible (\d+)\) \| random natural mix ([\d.]+) \| random uniform over classes ([\d.]+)", block)
        c = re.search(r"mean cosine between support images: same class ([-\d.]+) \| different class ([-\d.]+)", block)
        out[space] = dict(delta_own=float(d[1]), delta_other=float(d[2]), phi_real=float(p[1]), phi_max=int(p[2]),
                          phi_nat=float(p[3]), phi_uni=float(p[4]), cos_same=float(c[1]), cos_diff=float(c[2]))
    return out

geo = {d: parse_geometry(os.path.join(RES, d, "14.2_geometry_check.txt")) for d in DATASETS}

# ---------------------------------------------------------------- F1: PARA's eta vs K, measured vs predicted
eta_rows = []
for d in DATASETS:
    for f in glob.glob(os.path.join(RES, d, "sweep_*_eta.csv")):
        e = pd.read_csv(f)
        eta_rows.append(e[e.quantity == "eta(S)"])
eta = pd.concat(eta_rows)
delta_bar = np.mean([geo[d]["raw"]["delta_own"] for d in DATASETS])
cos_bar = np.mean([(geo[d]["raw"]["cos_same"] + geo[d]["raw"]["cos_diff"]) / 2 for d in DATASETS])
Kc = np.linspace(1, 16, 200)
a = delta_bar * (1 + (Kc - 1) * cos_bar)
pred = LAMBDA * a / (1 + a)
pred_at = {k: LAMBDA * (delta_bar * (1 + (k - 1) * cos_bar)) / (1 + delta_bar * (1 + (k - 1) * cos_bar)) for k in (1, 4, 16)}

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(Kc, pred, color=INK, lw=2, label=f"predicted from two model constants\n(δ ≈ {delta_bar:.2f}, mean image cosine ≈ {cos_bar:.2f})", zorder=2)
offsets = np.linspace(-0.16, 0.16, len(DATASETS))      # small horizontal dodge in log2 space so points don't hide each other
for off, d in zip(offsets, DATASETS):
    g = eta[eta.dataset == d].groupby("K").value
    ks = np.array(sorted(g.groups))
    m, lo, hi = g.mean().values, g.min().values, g.max().values
    x = ks * 2 ** off
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt=MARK[d], ms=6, color=COLOR[d], mec="white", mew=1, elinewidth=1,
                capsize=0, label=NICE[d] + (" (held-out)" if d in HELD else ""), zorder=3)
ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16]); ax.set_xticklabels(["1", "4", "16"])
ax.set_xlim(0.8, 20); ax.set_ylim(0, 1)
ax.set_xlabel("shots per class K"); ax.set_ylabel("PARA's step η(S)")
ax.set_title("PARA's step is the same curve in K on every dataset", loc="left")
ax.legend(fontsize=7, loc="lower right", handlelength=1.5)
ax.text(0.02, 0.97, "markers: mean over 10 support draws (range bars are smaller than the markers)",
        transform=ax.transAxes, fontsize=7, color=INK2, va="top")
save(fig, "F1_para_eta_vs_K")
print("F1 predicted:", {k: round(v, 3) for k, v in pred_at.items()},
      "| measured:", eta.groupby("K").value.mean().round(3).to_dict(), "| max SD across draws:",
      round(eta.groupby(["dataset", "prompts", "K"]).value.std().max(), 4))

# ---------------------------------------------------------------- F2: geometry check, raw vs centered
fig, axes = plt.subplots(2, 2, figsize=(9.0, 4.8), sharex=True)
x = np.arange(len(DATASETS)); w = 0.36
for col, space in enumerate(["raw", "centered"]):
    ax = axes[0, col]
    own = [geo[d][space]["delta_own"] for d in DATASETS]
    oth = [geo[d][space]["delta_other"] for d in DATASETS]
    ax.bar(x - w / 2 - 0.01, own, w, color=SLOTS[0], label="images vs their own class text", edgecolor="white", linewidth=1)
    ax.bar(x + w / 2 + 0.01, oth, w, color=SLOTS[1], label="images vs other classes' text", edgecolor="white", linewidth=1)
    ax.set_ylabel("discrepancy δ")
    ax.set_title(("Raw frozen space (what PARA uses)" if space == "raw" else "After centering each modality"), loc="left")
    ax = axes[1, col]
    real = [geo[d][space]["phi_real"] for d in DATASETS]
    rnd = [geo[d][space]["phi_uni"] for d in DATASETS]
    ax.bar(x - w / 2 - 0.01, real, w, color=SLOTS[0], label="16 images of one class", edgecolor="white", linewidth=1)
    ax.bar(x + w / 2 + 0.01, rnd, w, color=SLOTS[1], label="16 random images, mixed classes", edgecolor="white", linewidth=1)
    ax.axhline(16, color=MUTED, lw=0.8, ls="--")
    ax.text(len(DATASETS) - 0.5, 16, "max = K = 16", fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_ylabel("agreement φ"); ax.set_ylim(0, 17.5)
    ax.set_xticks(x); ax.set_xticklabels(["Derma", "OCT", "Blood", "Pneu.", "Breast", "OrganA"], rotation=0, fontsize=8)
axes[0, 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0)); axes[1, 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
for ax in axes[0]:
    ax.set_ylim(0, 1.25)
fig.suptitle("Raw space: δ and φ barely separate a real class from other classes' text or from random images (K = 16, draw 0)",
             x=0.01, ha="left", fontsize=9, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "F2_geometry_check")

# ---------------------------------------------------------------- shared: logprob table
lp = pd.read_csv(LP_TABLE)
def val(d, k, rule, col="BAcc"):
    r = lp[(lp.dataset == d) & (lp.K == k) & (lp.rule == rule)]
    return float(r[col].iloc[0]) if len(r) else np.nan

# ---------------------------------------------------------------- F4: best eta vs relative anchor quality
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
KCOL = {1: SLOTS[0], 4: SLOTS[1], 16: SLOTS[2]}
para_eta = {k: lp[(lp.K == k) & (lp.rule == "PARA [zscore]")].eta.mean() for k in (1, 4, 16)}
for ax, blend, title in [(axes[0], "zscore", "Original blend (per-image z-scores)"), (axes[1], "logprob", "Log-probability blend")]:
    for k in (1, 4, 16):
        ax.axhline(para_eta[k], color=KCOL[k], lw=1, ls=(0, (4, 3)), zorder=1)
        for d in DATASETS:
            gap = val(d, k, "Zero-shot") - val(d, k, "Visual only")
            best = val(d, k, f"Oracle BAcc (cheat) [{blend}]", "eta")
            binary = d in ("pneumoniamnist", "breastmnist")
            ax.scatter(gap, best, s=42, marker=MARK[d], color=KCOL[k] if not (binary and blend == "zscore") else "white",
                       edgecolor=KCOL[k], linewidth=1.2, zorder=3)
    ax.set_title(title, loc="left"); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("zero-shot BAcc minus visual-only BAcc (points)")
axes[0].set_ylabel("best step η in hindsight")
fig.text(0.01, -0.02, "Left of 0: the trained branch is better than the anchor.  Hollow (left panel): 2-class datasets, where the z-score blend is a switch at η = 0.5, so the best η is not unique.",
             fontsize=7, color=INK2)
k_handles = [Line2D([], [], marker="o", ls="", color=KCOL[k], label=f"K = {k}") for k in (1, 4, 16)]
k_handles += [Line2D([], [], color=MUTED, ls=(0, (4, 3)), label="PARA's η at that K (same on all datasets)")]
d_handles = [Line2D([], [], marker=MARK[d], ls="", color=INK2, mfc="none", label=NICE[d]) for d in DATASETS]
axes[1].legend(handles=k_handles + d_handles, fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.suptitle("The best step tracks how good the anchor is relative to the trained branch; PARA's step (dashed) does not",
             x=0.01, ha="left", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "F4_best_eta_vs_relative_anchor")

# ---------------------------------------------------------------- F5: worst-case BAcc at K = 1
RULES = [("Zero-shot", "Zero-shot (anchor only)", NEUTRAL), ("Visual only", "Visual only", SLOTS[0]),
         ("PARA [zscore]", "PARA, original blend", SLOTS[1]), ("PARA [logprob]", "PARA, log-prob blend", SLOTS[2]),
         ("Shot-only [zscore]", "Shot-only step", SLOTS[6])]
order = ["octmnist", "pneumoniamnist", "breastmnist", "dermamnist", "organamnist", "bloodmnist"]   # by relative anchor quality
fig, ax = plt.subplots(figsize=(7.4, 3.4))
n = len(RULES); w = 0.15; x = np.arange(len(order))
for i, (rule, label, col) in enumerate(RULES):
    vals = [val(d, 1, rule, "BAcc_CVaR20") for d in order]
    ax.bar(x + (i - (n - 1) / 2) * (w + 0.01), vals, w, color=col, label=label, edgecolor="white", linewidth=0.8)
orc = [val(d, 1, "Oracle BAcc (cheat) [logprob]", "BAcc_CVaR20") for d in order]
ax.scatter(x + (n / 2) * (w + 0.01) - 0.02, orc, marker="_", s=160, color=INK, linewidth=2, zorder=4,
           label="best η in hindsight (log-prob blend)")
ax.set_xticks(x); ax.set_xticklabels([NICE[d].replace("MNIST", "") + ("*" if d in HELD else "") for d in order], fontsize=8)
ax.set_ylabel("worst-case BAcc (CVaR20, %)")
ax.set_title("K = 1 worst case: anchoring protects good-anchor datasets (left) and hurts weak-anchor ones (right)",
             loc="left", fontsize=9)
ax.text(0.0, -0.17, "* held-out dataset.  Datasets ordered from 'anchor better than trained branch' to 'trained branch much better'.",
        transform=ax.transAxes, fontsize=7, color=INK2)
ax.legend(fontsize=7, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.0))
ax.set_ylim(0, 92)
save(fig, "F5_worst_case_K1")
