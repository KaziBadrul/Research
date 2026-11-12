"""Check that every decimal number in the paper draft traces to paper/numbers.json.

Usage: python analysis/check_paper_numbers.py [paper]
A number written with d decimals passes if some value in numbers.json rounds to it at d decimals
(sign ignored, so "$-0.12$" matches -0.1179). Integers are not checked; LaTeX comments, \\label, \\ref,
\\cite, \\input and \\includegraphics arguments are skipped. Exit code 1 if anything is unmatched.
"""
import glob
import json
import os
import re
import sys

PAPER = sys.argv[1] if len(sys.argv) > 1 else "paper"
nums = json.load(open(os.path.join(PAPER, "numbers.json"), encoding="utf-8"))
values = [(k, v["value"]) for k, v in nums.items()]

SKIP = re.compile(r"\\(label|ref|eqref|cite[tp]?|input|includegraphics(\[[^\]]*\])?|graphicspath|bibliography\w*)\{[^}]*\}")
NUM = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")

bad, total = [], 0
for path in sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex"))):
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        line = re.sub(r"(?<!\\)%.*", "", line)          # drop comments
        line = SKIP.sub("", line)
        for m in NUM.finditer(line):
            s = m.group(1)
            d = len(s.split(".")[1])
            x = float(s)
            hits = [k for k, v in values if round(abs(v), d) == x]
            total += 1
            if not hits:
                bad.append((path, ln, s, line.strip()[:110]))
print(f"checked {total} decimal numbers in {PAPER}/sections/*.tex against {len(values)} traced values")
for path, ln, s, ctx in bad:
    print(f"  UNTRACED {os.path.basename(path)}:{ln}: {s}   | {ctx}")
print("all numbers trace to numbers.json" if not bad else f"{len(bad)} untraced number(s)")

# cross-references: every \ref / \eqref must have a \label in the sections or in an \input table
texts = [open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(PAPER, "sections", "*.tex"))]
inputs = re.findall(r"\\input\{([^}]*)\}", "".join(texts))
tables = [open(os.path.normpath(os.path.join(PAPER, i if i.endswith(".tex") else i + ".tex")), encoding="utf-8").read()
          for i in inputs if os.path.exists(os.path.normpath(os.path.join(PAPER, i if i.endswith(".tex") else i + ".tex")))]
labels = set(re.findall(r"\\label\{([^}]*)\}", "".join(texts + tables)))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", "".join(texts)))
missing = sorted(refs - labels)
print("all cross-references resolve" if not missing else f"undefined references: {missing}")
# citations: every \cite key must exist in refs.bib
keys = set(re.findall(r"@\w+\{([^,]+),", open(os.path.join(PAPER, "refs.bib"), encoding="utf-8").read()))
cites = {k.strip() for t in texts for g in re.findall(r"\\cite[tp]?\{([^}]*)\}", t) for k in g.split(",")}
missing_bib = sorted(cites - keys)
print("all citations are in refs.bib" if not missing_bib else f"citations missing from refs.bib: {missing_bib}")
if keys - cites:
    print(f"note: bib entries not cited: {sorted(keys - cites)}")
missing += missing_bib
# hard-coded references ("Appendix~B", "Table 3") bypass \ref and can point at nothing
hard = [(os.path.basename(p), m.group(0)) for p in glob.glob(os.path.join(PAPER, "sections", "*.tex"))
        for m in re.finditer(r"\b(Appendix|Section|Table|Figure|Eq\.)~?\s?[A-Z0-9]+\b(?!\})(?!\s+of\s+\\cite)",
                             re.sub(r"(?<!\\)%.*", "", open(p, encoding="utf-8").read()))]
print("no hard-coded references" if not hard else f"hard-coded references: {hard}")
missing += [h[1] for h in hard]
emdash = [os.path.basename(p) for p, t in zip(glob.glob(os.path.join(PAPER, "sections", "*.tex")), texts) if "---" in t or "\u2014" in t]
if emdash:
    print(f"em dashes found in: {emdash}")
sys.exit(1 if (bad or missing or emdash) else 0)
