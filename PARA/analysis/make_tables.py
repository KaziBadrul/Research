"""Paper tables T1 to T4 from the saved PARA results (read-only; no training).

Usage: python analysis/make_tables.py [para_results] [tables]
Writes <name>.csv and <name>.tex (booktabs) into the output folder, prints each table, and compares
every recomputed number with the saved summary files (logprob_table.csv, cross_dataset_table.csv,
cross_dataset_summary.txt, logprob_summary.txt).
"""
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = sys.argv[1] if len(sys.argv) > 1 else "para_results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tables"
os.makedirs(OUT, exist_ok=True)

DESIGN = ["dermamnist", "octmnist", "bloodmnist"]
HELD = ["breastmnist", "pneumoniamnist", "organamnist"]
ORDER = DESIGN + HELD
NICE = {"dermamnist": "DermaMNIST", "octmnist": "OCTMNIST", "bloodmnist": "BloodMNIST",
        "breastmnist": "BreastMNIST", "pneumoniamnist": "PneumoniaMNIST", "organamnist": "OrganAMNIST"}
FROZEN_C = 8
TOL = 0.005          # report saved-vs-recomputed differences above this (percentage points / eta units)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def cvar20(v):
    v = np.sort(np.asarray(v))
    return v[: max(1, math.ceil(0.2 * len(v)))].mean()


def load(pattern):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, pattern)))], ignore_index=True)


def summarize(df):
    """mean BAcc / Macro-F1 over prompt sets and draws; CVaR20 per prompt set, then averaged (as the notebook)."""
    return pd.Series({"eta": df.eta.mean(), "BAcc": df.bacc.mean(),
                      "CVaR20": np.mean([cvar20(g.bacc) for _, g in df.groupby("prompts")]),
                      "MacroF1": df.macro_f1.mean()})


def to_latex(df, path, caption, label, bold=None, fmt="{:.2f}"):
    """Minimal booktabs writer. bold: set of (row_idx, col) cells to bold."""
    bold = bold or set()
    cols = list(df.columns)
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for i, (_, r) in enumerate(df.iterrows()):
        cells = []
        for c in cols:
            v = r[c]
            s = (fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                 else ("--" if (isinstance(v, float) and pd.isna(v)) else esc(v)))
            cells.append(rf"\textbf{{{s}}}" if (i, c) in bold else s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def save(df, name, caption, bold=None, fmt="{:.2f}"):
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    to_latex(df, os.path.join(OUT, f"{name}.tex"), caption, f"tab:{name}", bold, fmt)
    print(f"\n######## {name}: {caption}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def report_diffs(name, pairs):
    """pairs: list of (label, recomputed, saved)."""
    bad = [(l, a, b) for l, a, b in pairs if not (pd.isna(a) and pd.isna(b)) and abs(a - b) > TOL]
    print(f"-- {name}: compared {len(pairs)} numbers with saved files; {len(bad)} differ by more than {TOL}")
    for l, a, b in bad:
        print(f"   {l}: recomputed {a:.4f} vs saved {b:.4f}")


def paired(df, a, b, col, key=("dataset", "prompts", "K", "draw")):
    """a minus b per setting (dataset, prompts, K); Wilcoxon over draws. Returns per-setting frame."""
    rows = []
    for setting, g in df[df[col].isin([a, b])].groupby(list(key[:3])):
        w = g.pivot_table(index="draw", columns=col, values=["bacc", "macro_f1"]).dropna()
        if len(w) == 0:
            continue
        rec = dict(zip(key[:3], setting))
        for m in ["bacc", "macro_f1"]:
            d = w[(m, a)] - w[(m, b)]
            p = wilcoxon(d).pvalue if (np.any(d != 0) and len(d) >= 5) else 1.0
            rec[f"d_{m}"], rec[f"p_{m}"], rec[f"won_{m}"], rec[f"lost_{m}"] = d.mean(), p, int((d > 0).sum()), int((d < 0).sum())
            rec[f"n_{m}"] = len(d)
        rec["d_cvar20"] = cvar20(w[("bacc", a)]) - cvar20(w[("bacc", b)])
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate(per, label):
    """mean over all paired draws (= mean of setting means, since every setting has 10 draws)."""
    out = {"group": label, "settings": len(per)}
    for m, tag in [("bacc", "BAcc"), ("macro_f1", "F1")]:
        out[f"d{tag}"] = np.average(per[f"d_{m}"], weights=per[f"n_{m}"])
        out[f"won/lost {tag}"] = f"{per[f'won_{m}'].sum()}/{per[f'lost_{m}'].sum()}"
        out[f"sig +/- {tag}"] = f"{int(((per[f'p_{m}'] < 0.05) & (per[f'd_{m}'] > 0)).sum())}/" \
                                f"{int(((per[f'p_{m}'] < 0.05) & (per[f'd_{m}'] < 0)).sum())}"
    out["dCVaR20"] = per.d_cvar20.mean()
    return out


rules = load("*/rule_scores.csv")
lp = load("*/logprob_scores.csv")
lp["name"] = lp.rule + np.where(lp.blend == "-", "", " [" + lp.blend + "]")
classes = lp.groupby("dataset").classes.first().to_dict()
binary = [d for d in ORDER if classes[d] == 2]
cfg = {d: pd.read_json(os.path.join(ROOT, d, "config.json"), typ="series") for d in ORDER}
screen = pd.read_csv(os.path.join(ROOT, "16_screening.csv"))
lp_table = pd.read_csv(os.path.join(ROOT, "logprob_table.csv"))

# ================================================================ T1 datasets
t1 = []
for d in ORDER:
    zs = lp[(lp.dataset == d) & (lp.rule == "Zero-shot")]
    zsb = zs.groupby("prompts").bacc.agg(["mean", "std"])
    assert (zsb["std"].fillna(0) < 1e-9).all(), "zero-shot should not depend on the support draw"
    full = screen[screen.dataset == d].n_test_total.iloc[0]
    t1.append({"Dataset": NICE[d], "Classes": classes[d], "Test (eval)": int(cfg[d].n_test), "Test (full)": int(full),
               "Image px": int(cfg[d].medmnist_size), "Prompt sets": ", ".join(sorted(zsb.index, reverse=True)),
               "ZS BAcc (your prompts)": zsb.loc["your prompts", "mean"],
               "ZS BAcc (GPT-4)": zsb["mean"].get("50 GPT-4", np.nan),
               "Split": "design" if d in DESIGN else "held-out"})
t1 = pd.DataFrame(t1)
save(t1, "T1_datasets", "Datasets. Zero-shot BAcc (\\%) on the evaluated test subset, per prompt set.")
# compare zero-shot with 16_screening (which used the full test split) and logprob_table
pairs = []
for _, r in t1.iterrows():
    d = [k for k, v in NICE.items() if v == r.Dataset][0]
    s = screen[screen.dataset == d].set_index("prompts").bacc
    if "templates" in s:
        pairs.append((f"{d} your prompts ZS vs 16_screening", r["ZS BAcc (your prompts)"], s["templates"]))
    if "BiomedCoOp GPT-4" in s and not pd.isna(r["ZS BAcc (GPT-4)"]):
        pairs.append((f"{d} GPT-4 ZS vs 16_screening", r["ZS BAcc (GPT-4)"], s["BiomedCoOp GPT-4"]))
    zrow = lp_table[(lp_table.dataset == d) & (lp_table.rule == "Zero-shot")].BAcc.iloc[0]
    pairs.append((f"{d} ZS mean over prompt sets vs logprob_table",
                  np.nanmean([r["ZS BAcc (your prompts)"], r["ZS BAcc (GPT-4)"]]), zrow))
report_diffs("T1", pairs)
print("   note: 16_screening.csv scores the full test split; the sweeps use a fixed subset of n_test images "
      "(config.json), so differences there are expected.")

# ================================================================ T2 main results
METHODS = [("Zero-shot", "Zero-shot"), ("Visual only", "Visual only"), ("PARA (z)", "PARA [zscore]"),
           ("PARA (log-prob)", "PARA [logprob]"), ("Shot-only (z)", "Shot-only [zscore]"),
           ("Oracle (log-prob)*", "Oracle BAcc (cheat) [logprob]")]
t2, t2f, pairs, bold, boldf = [], [], [], set(), set()
for d in ORDER:
    for k in (1, 4, 16):
        row, rowf = {"Dataset": NICE[d], "K": k}, {"Dataset": NICE[d], "K": k}
        for short, name in METHODS:
            s = summarize(lp[(lp.dataset == d) & (lp.K == k) & (lp.name == name)])
            row[f"{short} BAcc"], row[f"{short} CVaR"] = s.BAcc, s.CVaR20
            rowf[f"{short} F1"] = s.MacroF1
            saved = lp_table[(lp_table.dataset == d) & (lp_table.K == k) & (lp_table.rule == name)].iloc[0]
            for col, sc in [("BAcc", "BAcc"), ("CVaR20", "BAcc_CVaR20"), ("MacroF1", "MacroF1"), ("eta", "eta")]:
                pairs.append((f"{d} K={k} {name} {col}", s[col], saved[sc]))
        i = len(t2)
        for metric in ["BAcc", "CVaR"]:
            cands = {f"{m} {metric}": row[f"{m} {metric}"] for m, _ in METHODS if not m.startswith("Oracle")}
            best = max(cands.values())
            bold |= {(i, c) for c, v in cands.items() if np.isclose(v, best, atol=1e-9)}
        candf = {f"{m} F1": rowf[f"{m} F1"] for m, _ in METHODS if not m.startswith("Oracle")}
        boldf |= {(i, c) for c, v in candf.items() if np.isclose(v, max(candf.values()), atol=1e-9)}
        t2.append(row); t2f.append(rowf)
t2, t2f = pd.DataFrame(t2), pd.DataFrame(t2f)
save(t2, "T2_main_results", "Mean BAcc and CVaR20 BAcc (\\%) over 10 support draws (and prompt sets). Bold: best "
     "non-oracle method per row and metric. *Oracle picks $\\eta$ on the test set and is not achievable.", bold)
save(t2f, "T2b_main_results_macroF1", "Mean Macro-F1 (\\%), same runs as Table T2. Bold: best non-oracle. "
     "*Oracle is not achievable.", boldf)
report_diffs("T2 vs logprob_table.csv", pairs)
# cross-check: z-score PARA / shot-only in logprob_scores vs rule_scores (two independent files, same runs)
x = []
for d in ORDER:
    for k in (1, 4, 16):
        for rr, ln in [("PARA", "PARA [zscore]"), ("Shot-only", "Shot-only [zscore]"), ("Visual only", "Visual only")]:
            a = rules[(rules.dataset == d) & (rules.K == k) & (rules.rule == rr)].bacc.mean()
            b = lp[(lp.dataset == d) & (lp.K == k) & (lp.name == ln)].bacc.mean()
            x.append((f"{d} K={k} {rr} rule_scores vs logprob_scores", a, b))
report_diffs("T2 cross-check rule_scores vs logprob_scores", x)

# ================================================================ T3 paired tests
per_a = paired(rules, "Shot-only", "PARA", "rule")
per_b = paired(lp, "PARA [logprob]", "PARA [zscore]", "name")
per_a.to_csv(os.path.join(OUT, "T3a_per_setting.csv"), index=False)
per_b.to_csv(os.path.join(OUT, "T3b_per_setting.csv"), index=False)
t3 = []
for comp, per in [("(a) Shot-only minus PARA", per_a), ("(b) Log-prob minus z-score at PARA eta", per_b)]:
    groups = [("binary", per.dataset.isin(binary)), ("multi-class", ~per.dataset.isin(binary)), ("all", per.dataset.notna())]
    if comp.startswith("(a)"):
        groups += [("design", per.dataset.isin(DESIGN)), ("held-out", per.dataset.isin(HELD))]
    for g, mask in groups:
        t3.append({"Comparison": comp, **aggregate(per[mask], g)})
t3 = pd.DataFrame(t3)
save(t3, "T3_paired_tests", "Paired differences over support draws. Wilcoxon signed-rank per (dataset, prompt set, K) "
     "setting; counts of settings significantly better/worse at $p<0.05$. $\\Delta$CVaR20 is the mean over settings.")
# compare (b) with logprob_summary.txt; (a) design/held-out with the values recorded during verification
txt = open(os.path.join(ROOT, "logprob_summary.txt"), encoding="utf-8").read()
pairs = []
for m in re.finditer(r"^\s+(binary|multi-class)\s+PARA\s+(bacc|macro_f1)\s+([+-][\d.]+)\s+(\d+)\s+(\d+) \| (\d+) / (\d+) of (\d+)",
                     txt, re.M):
    g, met, md, won, lost, sb, sw, n = m.groups()
    tag = "BAcc" if met == "bacc" else "F1"
    r = t3[(t3.Comparison.str.startswith("(b)")) & (t3.group == g)].iloc[0]
    pairs.append((f"(b) {g} {met} mean diff", r[f"d{tag}"], float(md)))
    rw, rl = map(int, r[f"won/lost {tag}"].split("/")); rb, rwz = map(int, r[f"sig +/- {tag}"].split("/"))
    for lab, a, b in [("won", rw, won), ("lost", rl, lost), ("sig better", rb, sb), ("sig worse", rwz, sw), ("settings", r.settings, n)]:
        pairs.append((f"(b) {g} {met} {lab}", a, int(b)))
report_diffs("T3(b) vs logprob_summary.txt", [(l, round(a, 2) if isinstance(a, float) else a, b) for l, a, b in pairs])
print("   (a) has no saved summary file; its design/held-out rows reproduce verify_claims.py (claim 16).")

# ================================================================ T4 pre-registered rule
aa = rules[rules.rule.str.startswith("Anchor-aware c=")]
frozen = aa[aa.rule == f"Anchor-aware c={FROZEN_C}"].assign(rule="Frozen c=8")
lodo_parts, lodo_c = [], {}
for d in DESIGN:      # as in the notebook: c with best mean BAcc on the other design datasets
    other = aa[(aa.dataset != d) & aa.dataset.isin(DESIGN)].groupby("rule").bacc.mean()
    lodo_c[d] = other.idxmax().replace("Anchor-aware c=", "")
    lodo_parts.append(aa[(aa.dataset == d) & (aa.rule == other.idxmax())].assign(rule="LODO"))
print(f"\nLODO choice of c (mean-BAcc criterion, as in cross_dataset_summary.txt): {lodo_c}")
full = pd.concat([rules, frozen, *lodo_parts], ignore_index=True)
t4 = []
for block, rule, dss in [("Design, in-sample (frozen c=8)", "Frozen c=8", DESIGN),
                         ("Design, leave-one-dataset-out", "LODO", DESIGN),
                         ("Held-out (frozen c=8)", "Frozen c=8", HELD)]:
    for other in ["PARA", "Shot-only", "Visual only"]:
        per = paired(full[full.dataset.isin(dss)], rule, other, "rule")
        t4.append({"Setting": block, "vs": other, **{k: v for k, v in aggregate(per, "").items() if k != "group"}})
t4 = pd.DataFrame(t4)
save(t4, "T4_preregistered_rule", "Anchor-aware rule minus each baseline. Wilcoxon per setting over 10 draws; counts "
     "of settings significantly better/worse ($p<0.05$). LODO picks $c$ by mean BAcc on the other design datasets.")
# compare with cross_dataset_summary.txt
txt = open(os.path.join(ROOT, "cross_dataset_summary.txt"), encoding="utf-8").read()
blocks = {"Held-out (frozen c=8)": r"HELD-OUT datasets.*?<-- the main test\n(.*?)\n\n",
          "Design, leave-one-dataset-out": r"LODO minus each rule\n(.*?)\n\n",
          "Design, in-sample (frozen c=8)": r"in-sample, for reference only\)\n(.*?)\n\n"}
pairs = []
for block, pat in blocks.items():
    body = re.search(pat, txt, re.S).group(1)
    for m in re.finditer(r"^\s+(PARA|Shot-only|Visual only)\s+(bacc|macro_f1)\s+([+-][\d.]+)\s+(\d+)\s+(\d+) \| (\d+) / (\d+) of (\d+)",
                         body, re.M):
        other, met, md, won, lost, sb, sw, n = m.groups()
        tag = "BAcc" if met == "bacc" else "F1"
        r = t4[(t4.Setting == block) & (t4.vs == other)].iloc[0]
        rw, rl = map(int, r[f"won/lost {tag}"].split("/")); rb, rwz = map(int, r[f"sig +/- {tag}"].split("/"))
        pairs += [(f"{block} vs {other} {met} mean diff", round(r[f"d{tag}"], 2), float(md)),
                  (f"{block} vs {other} {met} won", rw, int(won)), (f"{block} vs {other} {met} lost", rl, int(lost)),
                  (f"{block} vs {other} {met} sig better", rb, int(sb)), (f"{block} vs {other} {met} sig worse", rwz, int(sw)),
                  (f"{block} vs {other} {met} settings", r.settings, int(n))]
report_diffs("T4 vs cross_dataset_summary.txt", pairs)
# per-dataset frozen-rule means vs cross_dataset_table.csv
cdt = pd.read_csv(os.path.join(ROOT, "cross_dataset_table.csv"))
pairs = []
for d in ORDER:
    for k in (1, 4, 16):
        s = summarize(frozen[(frozen.dataset == d) & (frozen.K == k)])
        sv = cdt[(cdt.dataset == d) & (cdt.K == k) & (cdt.rule == "Anchor-aware frozen c=8")].iloc[0]
        pairs += [(f"{d} K={k} frozen {c}", s[c], sv[sc]) for c, sc in
                  [("eta", "eta"), ("BAcc", "BAcc"), ("CVaR20", "BAcc_CVaR20"), ("MacroF1", "MacroF1")]]
report_diffs("T4 per-dataset frozen rule vs cross_dataset_table.csv", pairs)
print(f"\nWrote tables to {os.path.abspath(OUT)}")
