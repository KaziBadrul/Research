"""Read-only verification of the chat's claims against files in para_results/.

Usage: python analysis/verify_claims.py [para_results]
Never writes into para_results/. Prints one block per claim.
"""
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = sys.argv[1] if len(sys.argv) > 1 else "para_results"
DESIGN = ["bloodmnist", "dermamnist", "octmnist"]
HELD = ["breastmnist", "organamnist", "pneumoniamnist"]
ALL = DESIGN + HELD
BINARY = ["breastmnist", "pneumoniamnist"]


def cvar20(v):
    v = np.sort(np.asarray(v))
    return v[: max(1, math.ceil(0.2 * len(v)))].mean()


def load(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, pattern)))], ignore_index=True)


def hdr(s):
    print("\n" + "=" * 8 + " " + s)


def paired_counts(df, a, b, key=("dataset", "prompts", "K", "draw"), metric="bacc", col="rule"):
    """a minus b, per-setting Wilcoxon over draws (same rule as the notebook)."""
    w = df.pivot_table(index=list(key), columns=col, values=metric)[[a, b]].dropna()
    d = w[a] - w[b]
    better = worse = n = 0
    rows = []
    for name, dd in d.groupby(level=[0, 1, 2]):
        n += 1
        p = wilcoxon(dd).pvalue if (np.any(dd != 0) and len(dd) >= 5) else 1.0
        better += int(p < 0.05 and dd.mean() > 0)
        worse += int(p < 0.05 and dd.mean() < 0)
        rows.append((*name, dd.mean(), p))
    return d.mean(), better, worse, n, int((d > 0).sum()), int((d < 0).sum()), rows


# ---------------------------------------------------------------- 14.2 geometry
geo = {}
for ds in ALL:
    txt = open(os.path.join(ROOT, ds, "14.2_geometry_check.txt"), encoding="utf-8").read()
    raw, cen = txt.split("CENTERED")
    g = {}
    for tag, part in [("raw", raw), ("cen", cen)]:
        m = re.search(r"own-text delta ([\d.]+) vs other-text delta ([\d.]+) \(gap ([+-][\d.]+)\)", part)
        g[tag + "_own"], g[tag + "_other"], g[tag + "_gap"] = map(float, m.groups())
        m = re.search(r"phi: real classes ([\d.]+).*random natural mix ([\d.]+) \| random uniform over classes ([\d.]+)", part)
        g[tag + "_phi_real"], g[tag + "_phi_mix"], g[tag + "_phi_unif"] = map(float, m.groups())
        m = re.search(r"same class ([-\d.]+) \| different class ([-\d.]+)", part)
        g[tag + "_cos_same"], g[tag + "_cos_diff"] = map(float, m.groups())
        g[tag + "_eta"] = float(re.search(r"eta\(S\) = ([\d.]+)", part).group(1))
    geo[ds] = g
geo = pd.DataFrame(geo).T
hdr("Claims 8-11: 14.2 geometry (K=16, draw 0)")
pd.set_option("display.width", 200)
print(geo[["raw_own", "raw_other", "raw_gap", "raw_phi_real", "raw_phi_mix", "raw_phi_unif", "raw_cos_same", "raw_cos_diff"]])
print(geo[["cen_own", "cen_other", "cen_gap", "cen_phi_real", "cen_phi_mix", "cen_phi_unif", "cen_cos_same", "cen_cos_diff"]])
print("raw gap range:", geo.raw_gap.min(), geo.raw_gap.max())
print("raw phi real/max(random) ratio:", (geo.raw_phi_real / geo[["raw_phi_mix", "raw_phi_unif"]].max(axis=1)).round(3).to_dict())
cos = pd.concat([geo.raw_cos_same, geo.raw_cos_diff])
print("raw cosine range:", cos.min(), cos.max())
print("centered phi real range:", geo.cen_phi_real.min(), geo.cen_phi_real.max(),
      "| random range:", geo[["cen_phi_mix", "cen_phi_unif"]].min().min(), geo[["cen_phi_mix", "cen_phi_unif"]].max().max())
print("centered own < other on all:", bool((geo.cen_own < geo.cen_other).all()))

# ---------------------------------------------------------------- sweep eta files
sw = load("*/sweep_*_eta.csv")
eS = sw[sw.quantity == "eta(S)"]
hdr("Claim 12: measured mean eta(S) by K")
print("pooled over all rows:", eS.groupby("K").value.mean().round(4).to_dict())
print("mean of dataset means:", eS.groupby(["K", "dataset"]).value.mean().groupby("K").mean().round(4).to_dict())
print(eS.groupby(["K", "dataset"]).value.mean().unstack().round(3))

hdr("Claim 13: predicted curve")
for name, dbar, cbar in [("files", geo.raw_own.mean(), pd.concat([geo.raw_cos_same, geo.raw_cos_diff]).mean()),
                         ("chat new", 0.55, 0.87), ("chat old", 0.52, 0.9)]:
    pred = {K: round(0.9 * (a := dbar * (1 + (K - 1) * cbar)) / (1 + a), 4) for K in (1, 4, 16)}
    print(f"{name:9s} dbar={dbar:.4f} cbar={cbar:.4f} -> {pred}")

hdr("Claim 14: SD of eta(S) across draws")
sd = eS.groupby(["dataset", "prompts", "K"]).value.std(ddof=1)
print("max SD (ddof=1):", sd.max().round(5), "at", sd.idxmax())
sd0 = eS.groupby(["dataset", "prompts", "K"]).value.std(ddof=0)
print("max SD (ddof=0):", sd0.max().round(5), "at", sd0.idxmax())
print(sd.unstack().round(4))

hdr("Claim 15: eta(S) vs per-draw oracle-BAcc eta")
ob = sw[sw.quantity == "oracle BAcc"]
m = eS.merge(ob, on=["dataset", "prompts", "K", "draw"], suffixes=("_eta", "_orc"))
rs = []
for k, g in m.groupby(["dataset", "prompts", "K"]):
    r = spearmanr(g.value_eta, g.value_orc).statistic if g.value_orc.nunique() > 1 else np.nan
    rs.append((*k, r))
rs = pd.DataFrame(rs, columns=["dataset", "prompts", "K", "rho"])
print(rs.round(3).to_string())
print("mean within-setting Spearman:", round(rs.rho.mean(), 3), "| n settings", rs.rho.notna().sum())
# pooled after removing K (within-K demeaned)
mm = m.copy()
for c in ["value_eta", "value_orc"]:
    mm[c] = mm[c] - mm.groupby(["dataset", "prompts", "K"])[c].transform("mean")
print("pooled within-setting Spearman:", round(spearmanr(mm.value_eta, mm.value_orc).statistic, 3),
      "p=", spearmanr(mm.value_eta, mm.value_orc).pvalue)

# ---------------------------------------------------------------- rule scores
rsc = load("*/rule_scores.csv")
hdr("Claim 16: shot-only minus PARA (rule_scores.csv)")
for grp, dss in [("design", DESIGN), ("held-out", HELD)]:
    for metric in ["bacc", "macro_f1"]:
        md, b, w, n, won, lost, _ = paired_counts(rsc[rsc.dataset.isin(dss)], "Shot-only", "PARA", metric=metric)
        print(f"{grp:9s} {metric:9s} mean {md:+.3f} | won {won} lost {lost} | sig better {b} / worse {w} of {n}")

hdr("Claim 17: binary rows, PARA-type rows equal zero-shot (eta<0.5) or visual-only (eta>0.5)")
b = rsc[rsc.dataset.isin(BINARY)]
key = ["dataset", "prompts", "K", "draw"]
ref = b[b.rule.isin(["Zero-shot", "Visual only"])].pivot_table(index=key, columns="rule", values="bacc").reset_index()
cand = b[~b.rule.isin(["Zero-shot", "Visual only"])].merge(ref, on=key)
for label, sub in [("all non-ZS/VO rows", cand), ("excluding oracle", cand[cand.rule != "Oracle BAcc (cheat)"])]:
    s = sub[sub.eta != 0.5]
    exp = np.where(s.eta < 0.5, s["Zero-shot"], s["Visual only"])
    ok = np.isclose(s.bacc, exp, atol=1e-9)
    print(f"{label}: rows={len(sub)}, eta==0.5 rows={int((sub.eta == 0.5).sum())}, tested={len(s)}, match={ok.sum()}")
    if (~ok).any():
        print(s[~ok][key + ["rule", "eta", "bacc", "Zero-shot", "Visual only"]].head(10).to_string())
e05 = cand[cand.eta == 0.5]
print("rows with eta exactly 0.5:\n", e05[key + ["rule", "bacc", "Zero-shot", "Visual only"]].to_string())

hdr("Claim 18: eta_curves z-score, one value below 0.5 and one above")
ec = load("*/eta_curves.csv")
for ds in ["breastmnist", "pneumoniamnist", "octmnist"]:
    z = ec[(ec.dataset == ds) & (ec.blend == "zscore")]
    lo = z[z.eta < 0.5].groupby(["prompts", "K", "draw"]).bacc.nunique()
    hi = z[z.eta > 0.5].groupby(["prompts", "K", "draw"]).bacc.nunique()
    print(f"{ds}: runs={len(lo)}, runs with 1 value below 0.5: {(lo == 1).sum()}, above 0.5: {(hi == 1).sum()}")

hdr("Claim 19: anchor-aware c=8 eta recomputed from r_anchor")
aa = rsc[rsc.rule == "Anchor-aware c=8"]
re_eta = aa.K / (aa.K + 8 * aa.r_anchor / np.maximum(1e-6, 1 - aa.r_anchor))
print("max |eta_file - K/(K+k0)|:", float(np.abs(aa.eta - re_eta).max()))
# Which c does LODO pick under CVaR20 vs mean BAcc (design datasets)?
aac = rsc[rsc.rule.str.startswith("Anchor-aware c=") & rsc.dataset.isin(DESIGN)]
cv = aac.groupby(["dataset", "prompts", "K", "rule"]).bacc.apply(cvar20).reset_index()
for ds in DESIGN:
    o = cv[cv.dataset != ds].groupby("rule").bacc.mean()
    o2 = aac[aac.dataset != ds].groupby("rule").bacc.mean()
    print(f"LODO leaving {ds}: best by CVaR20 -> {o.idxmax()} ({o.max():.2f}); best by mean -> {o2.idxmax()}")
allcv = cv.groupby("rule").bacc.mean().sort_values(ascending=False)
print("all 3 design datasets, mean CVaR20 by c:", allcv.round(2).to_dict())

hdr("Claim 20: frozen c=8 minus each rule on held-out (recomputed)")
h = rsc[rsc.dataset.isin(HELD)]
for other in ["PARA", "Shot-only", "Visual only"]:
    md, b_, w_, n, won, lost, _ = paired_counts(h, "Anchor-aware c=8", other)
    print(f"vs {other:12s} mean {md:+.3f} won {won} lost {lost} | {b_}/{w_} of {n}")

hdr("Claims 21-22: r_a and oracle eta")
print("mean r_anchor:", rsc.groupby("dataset").r_anchor.mean().round(3).to_dict())
lt = pd.read_csv(os.path.join(ROOT, "logprob_table.csv"))
orc = lt[lt.rule.str.startswith("Oracle")]
print(orc.pivot_table(index=["dataset", "K"], columns="rule", values="eta").round(3))
print("oracle eta range (both blends):", orc.eta.min().round(4), orc.eta.max().round(4))
print("oracle eta range (z-score only):", orc[orc.rule.str.contains("zscore")].eta.agg(["min", "max"]).round(4).to_dict())

# ---------------------------------------------------------------- logprob
lp = load("*/logprob_scores.csv")
lp["name"] = lp.rule + np.where(lp.blend == "-", "", " [" + lp.blend + "]")
hdr("Claim 23: log-prob minus z-score at the same eta")
for grp, dss in [("binary", BINARY), ("multi-class", [d for d in ALL if d not in BINARY])]:
    for rule in ["PARA", "Shot-only"]:
        for metric in ["bacc", "macro_f1"]:
            md, b_, w_, n, won, lost, _ = paired_counts(lp[lp.dataset.isin(dss)], f"{rule} [logprob]", f"{rule} [zscore]",
                                                        metric=metric, col="name")
            print(f"{grp:11s} {rule:9s} {metric:8s} mean {md:+.3f} won {won} lost {lost} | {b_}/{w_} of {n}")

hdr("Claims 24-25: logprob_table values")
lt2 = lt.set_index(["dataset", "K", "rule"])
for ds in ["bloodmnist", "organamnist", "pneumoniamnist", "breastmnist", "octmnist"]:
    print(f"{ds:15s} K=1 CVaR20 PARA z {lt2.loc[(ds, 1, 'PARA [zscore]'), 'BAcc_CVaR20']:.3f} -> "
          f"logprob {lt2.loc[(ds, 1, 'PARA [logprob]'), 'BAcc_CVaR20']:.3f}")
o = lt[lt.rule.str.startswith("Oracle")].pivot_table(index=["dataset", "K"], columns="rule", values="BAcc")
o["lp_minus_z"] = o["Oracle BAcc (cheat) [logprob]"] - o["Oracle BAcc (cheat) [zscore]"]
print(o.round(3))
print("settings where log-prob oracle < z-score oracle:", o[o.lp_minus_z < 0].index.tolist())

hdr("Claim 26: eta_curves best mean BAcc (mean over draws and prompt sets)")
for ds in ["breastmnist", "pneumoniamnist", "octmnist"]:
    e = ec[ec.dataset == ds]
    curve = e.groupby(["blend", "K", "prompts", "eta"]).bacc.mean().groupby(["blend", "K", "eta"]).mean()
    for K in (1, 4, 16):
        out = []
        for bl in ["zscore", "logprob"]:
            c = curve.loc[(bl, K)]
            out.append(f"{bl} {c.max():.2f} @ eta {c.idxmax():.3f}")
        print(f"{ds:15s} K={K:2d}: " + " | ".join(out))
