"""Every number quoted in the paper draft, computed from saved files (read-only) and written to paper/numbers.json.

Usage: python analysis/paper_numbers.py [para_results] [paper/numbers.json]
Each entry: {"value": float, "source": str}. analysis/check_paper_numbers.py checks the .tex files against this file.
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = sys.argv[1] if len(sys.argv) > 1 else "para_results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("paper", "numbers.json")
DESIGN = ["bloodmnist", "dermamnist", "octmnist"]
HELD = ["breastmnist", "organamnist", "pneumoniamnist"]
ALL = DESIGN + HELD
LAMBDA = 0.9
N = {}


def put(name, value, source):
    N[name] = {"value": float(value), "source": source}


def load(p):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, p)))], ignore_index=True)


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


# ------------------------------------------------------------------ setup
nb = json.load(open("PARA.ipynb", encoding="utf-8"))
tau0 = next(float(m.group(1)) for c in nb["cells"] for o in c.get("outputs", [])
            for m in [re.search(r"tau0 = ([\d.]+)", "".join(o.get("text", [])))] if m)
put("tau0", tau0, "PARA.ipynb cell 8 output")
cfg = json.load(open(os.path.join(ROOT, "dermamnist", "config.json")))
for k in ["lora_rank", "lora_alpha", "epochs", "batch_size", "lr_lora", "lr_scale", "weight_decay", "lambda_"]:
    put(f"cfg_{k}", cfg[k], "config.json (identical across datasets)")
pdraw = load("dermamnist/sweep_50_GPT_4_per_draw.csv")
put("derma_gpt_zs_acc", pdraw[pdraw.method == "Zero-shot"].acc.iloc[0], "dermamnist/sweep_50_GPT_4_per_draw.csv")
put("derma_gpt_zs_bacc", pdraw[pdraw.method == "Zero-shot"].bacc.iloc[0], "dermamnist/sweep_50_GPT_4_per_draw.csv")
put("derma_gpt_zs_acc_paper", 53.78, "Kang et al. 2026, Table A3, DermaMNIST, BiomedCLIP (zero-shot) column, p. 15")
put("frozen_c", 8, "PARA_followup_5.ipynb FROZEN_C")
put("alpha", 0.05, "significance threshold (CLAUDE.md rule)")
put("p_report_floor", 0.001, "reporting convention: p below this is written p < 0.001")
put("crop_scale_min", 0.5, "train_tf RandomResizedCrop(224, scale=(0.5, 1.0)) in the notebooks (not specified by the paper)")
put("wilcoxon_min_p", wilcoxon(np.arange(1, 11)).pvalue, "scipy: smallest two-sided Wilcoxon p with n = 10 (2/1024)")
rs = load("*/rule_scores.csv")
aa = rs[rs.rule.str.startswith("Anchor-aware c=") & rs.dataset.isin(DESIGN)]
cv = aa.groupby(["dataset", "prompts", "K", "rule"]).bacc.apply(
    lambda v: np.sort(v)[: max(1, math.ceil(0.2 * len(v)))].mean()).reset_index().groupby("rule").bacc.mean()
put("c8_pooled_cvar", cv["Anchor-aware c=8"], "rule_scores.csv, design datasets, pooled CVaR20")
put("c4_pooled_cvar", cv["Anchor-aware c=4"], "rule_scores.csv, design datasets, pooled CVaR20")
put("c16_pooled_cvar", cv["Anchor-aware c=16"], "rule_scores.csv, design datasets, pooled CVaR20")

# ------------------------------------------------------------------ 14.2 geometry
geo = {}
for ds in ALL:
    txt = open(os.path.join(ROOT, ds, "14.2_geometry_check.txt"), encoding="utf-8").read()
    raw, cen = txt.split("CENTERED")
    g = {}
    for tag, part in [("raw", raw), ("cen", cen)]:
        m = re.search(r"own-text delta ([\d.]+) vs other-text delta ([\d.]+) \(gap ([+-][\d.]+)\); images closest "
                      r"to their own text for (\d+)/(\d+)", part)
        g[tag + "_own"], g[tag + "_other"], g[tag + "_gap"] = map(float, m.groups()[:3])
        g[tag + "_closest"], g[tag + "_C"] = int(m.group(4)), int(m.group(5))
        m = re.search(r"real classes ([\d.]+).*natural mix ([\d.]+) \| random uniform over classes ([\d.]+)", part)
        g[tag + "_phi_real"], g[tag + "_phi_mix"], g[tag + "_phi_unif"] = map(float, m.groups())
        m = re.search(r"same class ([-\d.]+) \| different class ([-\d.]+)", part)
        g[tag + "_cos_same"], g[tag + "_cos_diff"] = map(float, m.groups())
    geo[ds] = g
geo = pd.DataFrame(geo).T
src = "*/14.2_geometry_check.txt (K=16, draw 0, your prompts)"
for ds in ALL:
    put(f"gap_{ds}", geo.loc[ds, "raw_gap"], src)
    put(f"own_{ds}", geo.loc[ds, "raw_own"], src)
    put(f"other_{ds}", geo.loc[ds, "raw_other"], src)
    put(f"phi_real_{ds}", geo.loc[ds, "raw_phi_real"], src)
    put(f"phi_mix_{ds}", geo.loc[ds, "raw_phi_mix"], src)
    put(f"phi_unif_{ds}", geo.loc[ds, "raw_phi_unif"], src)
    put(f"cos_same_{ds}", geo.loc[ds, "raw_cos_same"], src)
    put(f"cos_diff_{ds}", geo.loc[ds, "raw_cos_diff"], src)
    put(f"closest_{ds}", geo.loc[ds, "raw_closest"], src)
    put(f"C_{ds}", geo.loc[ds, "raw_C"], src)
five = [d for d in ALL if d != "organamnist"]
put("gap_min", geo.raw_gap.min(), src)
put("gap_max_five", geo.loc[five, "raw_gap"].max(), src + "; all but OrganAMNIST")
ratio = geo.raw_phi_real / geo[["raw_phi_mix", "raw_phi_unif"]].max(axis=1)
put("phi_ratio_dev_max_five_pct", 100 * (ratio[five] - 1).abs().max(), src + "; |real/max(random) - 1|, all but Organ")
put("phi_ratio_organ_pct", 100 * (ratio["organamnist"] - 1), src)
cos = pd.concat([geo.raw_cos_same, geo.raw_cos_diff])
put("cos_min", cos.min(), src)
put("cos_max", cos.max(), src)
put("cos_min_five", pd.concat([geo.loc[five, "raw_cos_same"], geo.loc[five, "raw_cos_diff"]]).min(), src)
put("delta_bar", geo.raw_own.mean(), src + "; mean own-text delta")
put("c_bar_pooled", cos.mean(), src + "; mean of same- and different-class cosine")
put("c_bar_same", geo.raw_cos_same.mean(), src + "; mean same-class cosine")
put("cen_phi_real_min", geo.cen_phi_real.min(), src)
put("cen_phi_real_max", geo.cen_phi_real.max(), src)
put("cen_phi_rand_min", geo[["cen_phi_mix", "cen_phi_unif"]].min().min(), src)
put("cen_phi_rand_max", geo[["cen_phi_mix", "cen_phi_unif"]].max().max(), src)
put("cen_own_min", geo.cen_own.min(), src)
put("cen_own_max", geo.cen_own.max(), src)
put("cen_other_min", geo.cen_other.min(), src)
put("cen_other_max", geo.cen_other.max(), src)


def pred(K, d, c):
    a = d * (1 + (K - 1) * c)
    return LAMBDA * a / (1 + a)


for K in (1, 4, 16):
    put(f"pred_pooled_K{K}", pred(K, N["delta_bar"]["value"], N["c_bar_pooled"]["value"]), "derived: 14.2 constants")
    put(f"pred_same_K{K}", pred(K, N["delta_bar"]["value"], N["c_bar_same"]["value"]), "derived: 14.2 constants")

# ------------------------------------------------------------------ eta(S) across draws
sw = load("*/sweep_*_eta.csv")
eS = sw[sw.quantity == "eta(S)"]
src = "*/sweep_*_eta.csv, quantity eta(S)"
for K, v in eS.groupby("K").value.mean().items():
    put(f"eta_meas_K{K}", v, src + ", pooled")
per = eS.groupby(["K", "dataset"]).value.mean()
for K in (1, 4, 16):
    put(f"eta_ds_min_K{K}", per.loc[K].min(), src + ", per-dataset mean")
    put(f"eta_ds_max_K{K}", per.loc[K].max(), src + ", per-dataset mean")
for K in (1, 4, 16):
    put(f"eta_organ_K{K}", per.loc[(K, "organamnist")], src + ", OrganAMNIST mean")
put("pred_diff_max", max(abs(N[f"pred_same_K{K}"]["value"] - N[f"pred_pooled_K{K}"]["value"]) for K in (1, 4, 16)),
    "derived: same-class vs pooled cosine prediction")
sd = eS.groupby(["dataset", "prompts", "K"]).value.std(ddof=1)
put("eta_sd_max", sd.max(), src + f"; at {sd.idxmax()}")
put("eta_sd_max_design", sd[sd.index.get_level_values(0).isin(DESIGN)].max(), src)
m = eS.merge(sw[sw.quantity == "oracle BAcc"], on=["dataset", "prompts", "K", "draw"], suffixes=("_e", "_o"))
for c in ["value_e", "value_o"]:
    m[c] = m[c] - m.groupby(["dataset", "prompts", "K"])[c].transform("mean")
r = spearmanr(m.value_e, m.value_o)
put("spearman_eta_oracle", r.statistic, src + " vs oracle BAcc, within-setting demeaned, pooled")
put("spearman_eta_oracle_p", r.pvalue, src)
cen = sw[sw.quantity == "centered"].groupby("K").value.mean()
for K, v in cen.items():
    put(f"eta_centered_K{K}", v, "*/sweep_*_eta.csv, quantity centered")
pdall = load("*/sweep_*_per_draw.csv")
dm = pdall[pdall.dataset.isin(DESIGN)].groupby("method").bacc.mean()
put("bacc_design_para", dm["PARA"], "*/sweep_*_per_draw.csv, design, mean BAcc")
put("bacc_design_para_centered", dm["PARA centered eta"], "*/sweep_*_per_draw.csv, design, mean BAcc")
byds = pdall.groupby(["dataset", "method"]).bacc.mean().unstack()
hm = pdall[pdall.dataset.isin(HELD)].groupby("method").bacc.mean()
put("bacc_held_para", hm["PARA"], "*/sweep_*_per_draw.csv, held-out, mean BAcc")
put("bacc_held_reliability_eta", hm["Reliability eta"], "*/sweep_*_per_draw.csv, held-out, mean BAcc, frozen-space reliability step")
put("n_ds_centered_worse", int((byds["PARA centered eta"] < byds["PARA"]).sum()), "*/sweep_*_per_draw.csv")

# ------------------------------------------------------------------ shot-only vs PARA
key = ["dataset", "prompts", "K", "draw"]
for grp, dss in [("design", DESIGN), ("heldout", HELD), ("organ", ["organamnist"])]:
    w = rs[rs.dataset.isin(dss)].pivot_table(index=key, columns="rule", values="bacc")
    d = w["Shot-only"] - w["PARA"]
    ps, means = [], []
    for _, dd in d.groupby(level=[0, 1, 2]):
        ps.append(wilcoxon(dd).pvalue if (np.any(dd != 0) and len(dd) >= 5) else 1.0)
        means.append(dd.mean())
    ps, means = np.array(ps), np.array(means)
    s = f"*/rule_scores.csv, shot-only minus PARA BAcc, {grp}"
    put(f"shot_{grp}_mean", d.mean(), s)
    put(f"shot_{grp}_settings", len(ps), s)
    put(f"shot_{grp}_sig_better", ((ps < 0.05) & (means > 0)).sum(), s)
    put(f"shot_{grp}_sig_worse", ((ps < 0.05) & (means < 0)).sum(), s)
    put(f"shot_{grp}_holm_better", ((holm(ps) < 0.05) & (means > 0)).sum(), s)
    put(f"shot_{grp}_won", (d > 0).sum(), s)
    put(f"shot_{grp}_lost", (d < 0).sum(), s)
bin_ = rs[rs.dataset.isin(["breastmnist", "pneumoniamnist"])].pivot_table(index=key, columns="rule", values="bacc")
put("shot_binary_nonzero", int(((bin_["Shot-only"] - bin_["PARA"]) != 0).sum()), "*/rule_scores.csv, binary datasets")
for K in (1, 4, 16):
    put(f"shot_eta_K{K}", LAMBDA * K / (1 + K), "lambda K / (1 + K)")

# ------------------------------------------------------------------ Finding 2: binary switch
BIN = ["breastmnist", "pneumoniamnist"]
b = rs[rs.dataset.isin(BIN)]
ref = b[b.rule.isin(["Zero-shot", "Visual only"])].pivot_table(index=key, columns="rule", values="bacc").reset_index()
cand = b[~b.rule.isin(["Zero-shot", "Visual only"])].merge(ref, on=key)
for tag, sub in [("nonoracle", cand[cand.rule != "Oracle BAcc (cheat)"]), ("all", cand)]:
    s = sub[sub.eta != 0.5]
    match = np.isclose(s.bacc, np.where(s.eta < 0.5, s["Zero-shot"], s["Visual only"]), atol=1e-9)
    src = f"breast+pneumonia rule_scores.csv, rule rows ({tag}), eta != 0.5"
    put(f"switch_rows_{tag}", len(s), src)
    put(f"switch_match_{tag}", match.sum(), src)
put("switch_rules_n", cand[cand.rule != "Oracle BAcc (cheat)"].rule.nunique(), "distinct non-oracle step rules on binary data")
para_b = b[b.rule == "PARA"]
put("para_bin_eta_max_K1", para_b[para_b.K == 1].eta.max(), "breast+pneumonia rule_scores.csv, PARA eta, K=1, max over draws")
put("para_bin_eta_min_K4", para_b[para_b.K == 4].eta.min(), "breast+pneumonia rule_scores.csv, PARA eta, K=4, min over draws")
ec = load("*/eta_curves.csv")
for ds in ["breastmnist", "pneumoniamnist", "octmnist"]:
    z = ec[(ec.dataset == ds) & (ec.blend == "zscore")]
    lo = z[z.eta < 0.5].groupby(["prompts", "K", "draw"]).bacc.nunique()
    hi = z[z.eta > 0.5].groupby(["prompts", "K", "draw"]).bacc.nunique()
    src = f"{ds}/eta_curves.csv, z-score blend, 41-point grid"
    put(f"curves_runs_{ds}", len(lo), src)
    put(f"curves_switch_{ds}", int(((lo == 1) & (hi == 1)).sum()), src + ": runs with one BAcc below and one above 0.5")
put("eta_grid_n", ec.eta.nunique(), "eta_curves.csv grid size")
lt = pd.read_csv(os.path.join(ROOT, "logprob_table.csv"))
for ds in BIN:
    for K in (1, 4, 16):
        for rule, tag in [("Zero-shot", "zs"), ("Visual only", "vo"), ("PARA [zscore]", "para"),
                          ("Oracle BAcc (cheat) [zscore]", "oraclez")]:
            r = lt[(lt.dataset == ds) & (lt.K == K) & (lt.rule == rule)].iloc[0]
            put(f"{ds}_K{K}_{tag}_bacc", r.BAcc, "logprob_table.csv")
            put(f"{ds}_K{K}_{tag}_cvar", r.BAcc_CVaR20, "logprob_table.csv")

# ------------------------------------------------------------------ Finding 3: relative reliability, pre-specified rule
src_lt = "logprob_table.csv"
for ds in ALL:
    for K in (1, 4, 16):
        g = lt[(lt.dataset == ds) & (lt.K == K)].set_index("rule")
        for rule, tag in [("Zero-shot", "zs"), ("Visual only", "vo"), ("PARA [zscore]", "para")]:
            put(f"{ds}_K{K}_{tag}_bacc", g.loc[rule, "BAcc"], src_lt)
            put(f"{ds}_K{K}_{tag}_cvar", g.loc[rule, "BAcc_CVaR20"], src_lt)
        put(f"{ds}_K{K}_relq", g.loc["Zero-shot", "BAcc"] - g.loc["Visual only", "BAcc"], src_lt + ": zero-shot minus visual-only BAcc")
        for bl in ["zscore", "logprob"]:
            put(f"{ds}_K{K}_oracle_{bl}_eta", g.loc[f"Oracle BAcc (cheat) [{bl}]", "eta"], src_lt)
            put(f"{ds}_K{K}_oracle_{bl}_bacc", g.loc[f"Oracle BAcc (cheat) [{bl}]", "BAcc"], src_lt)
rel = pd.DataFrame([(ds, K, N[f"{ds}_K{K}_relq"]["value"], N[f"{ds}_K{K}_oracle_logprob_eta"]["value"],
                     N[f"{ds}_K{K}_oracle_zscore_eta"]["value"]) for ds in ALL for K in (1, 4, 16)],
                   columns=["ds", "K", "relq", "orc_lp", "orc_z"])
r = spearmanr(rel.relq, rel.orc_lp)
put("rho_relq_oracle_lp", r.statistic, src_lt + ": Spearman over 18 (dataset, K), log-prob oracle eta")
put("rho_relq_oracle_lp_p", r.pvalue, src_lt)
mc = rel[~rel.ds.isin(BIN)]
r = spearmanr(mc.relq, mc.orc_z)
put("rho_relq_oracle_z_mc", r.statistic, src_lt + ": Spearman over 12 multi-class (dataset, K), z-score oracle eta")
put("rho_relq_oracle_z_mc_p", r.pvalue, src_lt)
put("oracle_lp_eta_min", rel.orc_lp.min(), src_lt)
put("oracle_lp_eta_max", rel.orc_lp.max(), src_lt)
put("oracle_z_eta_min_mc", mc.orc_z.min(), src_lt + ", multi-class")
put("oracle_z_eta_max_mc", mc.orc_z.max(), src_lt + ", multi-class")
for ds in ALL:
    put(f"ra_{ds}", rs[rs.dataset == ds].r_anchor.mean(), "*/rule_scores.csv, mean r_anchor")
ra_set = rs.groupby(["dataset", "prompts", "K"]).r_anchor.mean()
orc_set = sw[sw.quantity == "oracle BAcc"].groupby(["dataset", "prompts", "K"]).value.mean()
mm = pd.concat([ra_set, orc_set], axis=1, keys=["ra", "orc"]).dropna()
des = mm[mm.index.get_level_values(0).isin(DESIGN)]
r = spearmanr(des.ra, des.orc)
put("rho_ra_oracle_design", r.statistic, "rule_scores r_anchor vs sweep oracle BAcc eta, 18 design settings")
put("rho_ra_oracle_design_p", r.pvalue, "same")
mcs = mm[~mm.index.get_level_values(0).isin(BIN)]
r = spearmanr(mcs.ra, mcs.orc)
put("rho_ra_oracle_mc_all", r.statistic, "same, all multi-class settings incl. OrganAMNIST")
put("n_ra_oracle_mc_all", len(mcs), "same")
# pre-specified rule vs baselines (cross_dataset_table.csv per-dataset rows; paired stats as in T4)
cdt = pd.read_csv(os.path.join(ROOT, "cross_dataset_table.csv"))
for ds, K in [("organamnist", 1), ("pneumoniamnist", 4), ("breastmnist", 1), ("octmnist", 1)]:
    g = cdt[(cdt.dataset == ds) & (cdt.K == K)].set_index("rule")
    for rule, tag in [("Anchor-aware frozen c=8", "c8"), ("PARA", "paratab"), ("Visual only", "votab"),
                      ("Oracle BAcc (cheat)", "orctab"), ("HOSO-style", "hoso")]:
        if rule in g.index:
            put(f"{ds}_K{K}_{tag}_eta", g.loc[rule, "eta"], "cross_dataset_table.csv")
            put(f"{ds}_K{K}_{tag}_bacc", g.loc[rule, "BAcc"], "cross_dataset_table.csv")
            put(f"{ds}_K{K}_{tag}_cvar", g.loc[rule, "BAcc_CVaR20"], "cross_dataset_table.csv")
for ds, K in [("organamnist", 4), ("organamnist", 16), ("pneumoniamnist", 16)]:
    g = cdt[(cdt.dataset == ds) & (cdt.K == K)].set_index("rule")
    for rule, tag in [("HOSO-style", "hoso"), ("Oracle BAcc (cheat)", "orctab"), ("PARA", "paratab")]:
        put(f"{ds}_K{K}_{tag}_bacc", g.loc[rule, "BAcc"], "cross_dataset_table.csv")
        put(f"{ds}_K{K}_{tag}_cvar", g.loc[rule, "BAcc_CVaR20"], "cross_dataset_table.csv")
for ds, K in [("organamnist", 4), ("organamnist", 16)]:
    put(f"{ds}_K{K}_hoso_gap", N[f"{ds}_K{K}_orctab_bacc"]["value"] - N[f"{ds}_K{K}_hoso_bacc"]["value"],
        "cross_dataset_table.csv: oracle minus HOSO-style BAcc")
c8 = rs[rs.rule == "Anchor-aware c=8"]
pk4 = c8[(c8.dataset == "pneumoniamnist") & (c8.K == 4)]
put("pneu_K4_c8_below", (pk4.eta < 0.5).sum(), "pneumoniamnist rule_scores.csv, c=8, K=4: draws with eta < 0.5")
put("pneu_K4_c8_n", len(pk4), "same")
bk1 = c8[(c8.dataset == "breastmnist") & (c8.K == 1)]
put("breast_K1_c8_above", (bk1.eta > 0.5).sum(), "breastmnist rule_scores.csv, c=8, K=1: draws with eta > 0.5")
put("breast_K1_c8_n", len(bk1), "same")
put("bacc_design_reliability_eta", dm["Reliability eta"], "*/sweep_*_per_draw.csv, design, mean BAcc, frozen-space reliability step")
t4 = pd.read_csv(os.path.join("tables", "T4_preregistered_rule.csv"))
for _, r in t4.iterrows():
    tag = {"Design, in-sample (frozen c=8)": "ins", "Design, leave-one-dataset-out": "lodo", "Held-out (frozen c=8)": "held"}[r.Setting]
    vs = {"PARA": "para", "Shot-only": "shot", "Visual only": "vo"}[r.vs]
    put(f"t4_{tag}_{vs}_dbacc", r.dBAcc, "tables/T4_preregistered_rule.csv")
    put(f"t4_{tag}_{vs}_dcvar", r.dCVaR20, "tables/T4_preregistered_rule.csv")
    put(f"t4_{tag}_{vs}_df1", r.dF1, "tables/T4_preregistered_rule.csv")

# ------------------------------------------------------------------ Finding 4: log-probability blend
for ds in ALL:
    for K in (1, 4, 16):
        g = lt[(lt.dataset == ds) & (lt.K == K)].set_index("rule")
        put(f"{ds}_K{K}_paralp_bacc", g.loc["PARA [logprob]", "BAcc"], src_lt)
        put(f"{ds}_K{K}_paralp_cvar", g.loc["PARA [logprob]", "BAcc_CVaR20"], src_lt)
o = lt[lt.rule.str.startswith("Oracle")].pivot_table(index=["dataset", "K"], columns="rule", values="BAcc")
d_or = o["Oracle BAcc (cheat) [logprob]"] - o["Oracle BAcc (cheat) [zscore]"]
put("oracle_lp_ge_z_n", int((d_or >= 0).sum()), src_lt + ": settings where log-prob oracle >= z-score oracle")
put("oracle_lp_lt_z_derma16", d_or.loc[("dermamnist", 16)], src_lt)
t3 = pd.read_csv(os.path.join("tables", "T3_paired_tests.csv"))
for _, r in t3[t3.Comparison.str.startswith("(b)")].iterrows():
    g = r.group.replace("-", "")
    put(f"t3b_{g}_dbacc", r.dBAcc, "tables/T3_paired_tests.csv")
    put(f"t3b_{g}_df1", r.dF1, "tables/T3_paired_tests.csv")
    put(f"t3b_{g}_dcvar", r.dCVaR20, "tables/T3_paired_tests.csv")
for ds in ["breastmnist", "pneumoniamnist", "octmnist"]:
    e = ec[ec.dataset == ds]
    curve = e.groupby(["blend", "K", "prompts", "eta"]).bacc.mean().groupby(["blend", "K", "eta"]).mean()
    for K in (1, 4, 16):
        for bl in ["zscore", "logprob"]:
            c = curve.loc[(bl, K)]
            put(f"curve_{ds}_K{K}_{bl}_best", c.max(), f"{ds}/eta_curves.csv, mean over draws and prompt sets")
            put(f"curve_{ds}_K{K}_{bl}_argbest", c.idxmax(), f"{ds}/eta_curves.csv, eta of the best mean BAcc")
        put(f"curve_{ds}_K{K}_gain", curve.loc[("logprob", K)].max() - curve.loc[("zscore", K)].max(), f"{ds}/eta_curves.csv")

# ------------------------------------------------------------------ additions from the 2026-11-12 review
# phi of a real class against the uniform random baseline that Figure 2 plots
ru = geo.raw_phi_real / geo.raw_phi_unif
put("phi_unif_dev_min_five_pct", 100 * (ru[five] - 1).abs().min(), "14.2; |real/uniform random - 1|, all but Organ (as in Figure 2)")
put("phi_unif_dev_max_five_pct", 100 * (ru[five] - 1).abs().max(), "14.2; |real/uniform random - 1|, all but Organ (as in Figure 2)")
put("phi_unif_organ_pct", 100 * (ru["organamnist"] - 1), "14.2; real/uniform random - 1, OrganAMNIST")
for ds in ALL:
    put(f"cen_phi_real_{ds}", geo.loc[ds, "cen_phi_real"], "14.2, centered space")
    put(f"cen_phi_unif_{ds}", geo.loc[ds, "cen_phi_unif"], "14.2, centered space (as in Figure 2)")
# anchor reliability per (dataset, K), averaged over draws and prompt sets
for (ds, K), v in rs.groupby(["dataset", "K"]).r_anchor.mean().items():
    put(f"ra_{ds}_K{K}", v, "*/rule_scores.csv, mean r_anchor at this K")
# eta(S) vs per-draw oracle eta, multi-class settings only (the z-score oracle eta is not unique on binary data)
mmc = m[~m.dataset.isin(BIN)]
r = spearmanr(mmc.value_e, mmc.value_o)
put("spearman_eta_oracle_mc", r.statistic, "sweep eta files, within-setting demeaned, multi-class settings only")
put("spearman_eta_oracle_mc_p", r.pvalue, "same")
# frozen-space reliability step (Appendix): prior epsilon
put("rel_eps", cfg["rel_eps"], "config.json rel_eps (REL_EPS in PARA_followup_2/3/5)")
# HOSO-style eta on OrganAMNIST
hs = rs[(rs.rule == "HOSO-style") & (rs.dataset == "organamnist")]
for K in (4, 16):
    put(f"hoso_organ_K{K}_eta_min", hs[hs.K == K].eta.min(), "organamnist rule_scores.csv, HOSO-style eta, min over draws")
# DermaMNIST majority-class share of the test set
scr = open(os.path.join(ROOT, "16_screening.txt"), encoding="utf-8").read()
put("derma_majority_pct", float(re.search(r"dermamnist\s+templates\s+reference(?:\s+\S+){6}\s+([\d.]+)", scr).group(1)),
    "16_screening.txt, dermamnist majority%")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(N, open(OUT, "w", encoding="utf-8"), indent=1)
print(f"wrote {len(N)} numbers to {OUT}")
for k, v in N.items():
    print(f"  {k:32s} {v['value']:.4f}")
