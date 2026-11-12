"""Read-only checks of numbers in para-replication-findings.md that verify_claims.py does not cover.

Usage: python analysis/check_notes.py [para_results]
Scope of that note: the three design datasets (Blood, Derma, OCT), both prompt sets, 18 settings.
"""
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = sys.argv[1] if len(sys.argv) > 1 else "para_results"
DESIGN = ["bloodmnist", "dermamnist", "octmnist"]


def cvar20(v):
    v = np.sort(np.asarray(v))
    return v[: max(1, math.ceil(0.2 * len(v)))].mean()


def load(p):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, p)))], ignore_index=True)


rs = load("*/rule_scores.csv")
rs = rs[rs.dataset.isin(DESIGN)]
sw = load("*/sweep_*_eta.csv")
sw = sw[sw.dataset.isin(DESIGN)]
key = ["dataset", "prompts", "K", "draw"]

print("== line 55: shot-only minus PARA BAcc per dataset")
w = rs.pivot_table(index=key, columns="rule", values="bacc")
print((w["Shot-only"] - w["PARA"]).groupby(level=0).mean().round(2).to_dict())

print("== line 59: Spearman(r_a, oracle-BAcc eta) over the 18 settings")
ra = rs.groupby(["dataset", "prompts", "K"]).r_anchor.mean()
orc = sw[sw.quantity == "oracle BAcc"].groupby(["dataset", "prompts", "K"]).value.mean()
m = pd.concat([ra, orc], axis=1, keys=["ra", "orc"]).dropna()
r = spearmanr(m.ra, m.orc)
print(f"all: rho {r.statistic:.2f} p {r.pvalue:.4f} (n={len(m)})")
for k, g in m.groupby(level=2):
    print(f"K={k}: rho {spearmanr(g.ra, g.orc).statistic:.2f}")

print("== line 61: PARA minus visual-only, CVaR20 BAcc, K=1, per prompt set")
for (d, p), g in rs[rs.K == 1].groupby(["dataset", "prompts"]):
    a = cvar20(g[g.rule == "PARA"].bacc) - cvar20(g[g.rule == "Visual only"].bacc)
    print(f"{d} {p}: {a:+.2f}")

print("== line 65: Derma K=1 best eta by Acc vs by BAcc (mean over draws, per prompt set)")
for p, g in sw[(sw.dataset == "dermamnist") & (sw.K == 1)].groupby("prompts"):
    print(p, {q: round(g[g.quantity == q].value.mean(), 3) for q in ["oracle Acc", "oracle BAcc"]})

print("== line 69: frozen r_v range (K>=4)")
rv = sw[(sw.quantity == "r_visual") & (sw.K > 1)].groupby(["dataset", "K"]).value.mean()
print(rv.round(3).to_dict())

print("== lines 78-79: pooled over the 18 design settings, mean BAcc / mean CVaR20")
rows = {}
for rule, g in rs.groupby("rule"):
    cv = g.groupby(["dataset", "prompts", "K"]).bacc.apply(cvar20).mean()
    rows[rule] = (round(g.bacc.mean(), 1), round(cv, 1))
for k in ["PARA", "Shot-only", "Visual only", "Anchor-aware c=4", "Anchor-aware c=8", "Oracle BAcc (cheat)"]:
    print(f"  {k}: {rows[k]}")
o = rs[rs.dataset == "octmnist"]
print("  OCT mean CVaR20:", {c: round(o[o.rule == f"Anchor-aware c={c}"].groupby(["prompts", "K"]).bacc.apply(cvar20).mean(), 1)
                             for c in [4, 8, 16]})
print("== line 80: OCT K=1 BAcc, c=8 and c=16 vs oracle")
ok = o[o.K == 1].groupby("rule").bacc.mean()
print({k: round(ok[k], 2) for k in ["Anchor-aware c=8", "Anchor-aware c=16", "Oracle BAcc (cheat)"]})
print("== line 81: HOSO-style eta range")
print(rs[rs.rule == "HOSO-style"].groupby(["dataset", "K"]).eta.mean().round(3).to_dict())
