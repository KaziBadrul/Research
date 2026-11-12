# PARA analysis paper

## Goal
An analysis paper on PARA (Prompt-Anchored Residual Adaptation; Kang, Yue, Liu, Qin; arXiv:2609.33701v1, 27 Sep 2026;
no official code). We reimplemented it (BiomedCLIP ViT-B/16 + LoRA) and study why its residual step
$\eta = \lambda \cdot \text{mean}_c\left[\frac{\delta_c \phi_c}{1+\delta_c \phi_c}\right]$ behaves as it does. Main findings:
1. The support geometry carries almost no information, so $\eta$ is close to a fixed curve in $K$.
2. The z-score blend acts as a switch on binary tasks.
3. The right step depends on how reliable the anchor is relative to the trained branch. The pre-registered rule did not beat the baselines on held-out data.
4. A log-probability blend helps on multi-class tasks.

## Layout
- `PARA.ipynb`: the original reimplementation. `PARA_followup.ipynb` and `PARA_followup_2.ipynb` to `PARA_followup_5.ipynb`: the follow-ups. Sections 14 to 17 are in `_5`.
- `para_results/<dataset>/`: per-dataset outputs:
  - `config.json`, `14.2_geometry_check.txt`, `14.3_zero_shot_prompts.txt` (not present for organ or pneumonia)
  - `rule_scores.csv`, `sweep_*_eta.csv`, `sweep_*_per_draw.csv`, `15.3_rules_this_dataset.txt`, `hoso_eta.json`, `logprob_scores.csv`
  - `eta_curves.csv` (pneumonia, breast and OCT only), `zS_cache.pt`
- `para_results/` top level: `cross_dataset_summary.txt`, `cross_dataset_table.csv`, `16_screening.{txt,csv}`,
  `logprob_table.csv`, `logprob_summary.txt`.
- `para_figures/`: F1 to F5 (PNG and PDF) and the scripts that make them (`make_figs.py`, `make_f3.py`, `cell_17_3_eta_curves.py`).
- `analysis/`: read-only analysis scripts. `verify_claims.py` checks every claim; `make_tables.py` builds T1 to T4.
- `tables/`: the T1 to T4 tables, as LaTeX (booktabs) and CSV.
- `docs/findings_verified.md`: every verified number with its source file and a list of corrections. **Use these values.**
- `para-paper-outline.md`, `para-replication-findings.md`: earlier chat notes. These are not verified; check numbers against `docs/`.

## Datasets
| dataset | classes | image size | prompt sets | split |
|---|---|---|---|---|
| DermaMNIST | 7 | 224 | your prompts, 50 GPT-4 | design |
| OCTMNIST | 4 | 128 | your prompts, 50 GPT-4 | design |
| BloodMNIST | 8 | 128 | your prompts, 50 GPT-4 | design |
| BreastMNIST | 2 | 128 | your prompts, 50 GPT-4 (BUSI prompts: malignant vs benign + normal) | held-out |
| PneumoniaMNIST | 2 | 128 | your prompts | held-out |
| OrganAMNIST | 11 | 128 | your prompts | held-out |

Protocol for every dataset:
- Support set: $K \in \{1, 4, 16\}$, 10 support draws per $K$, 1 training seed. Support draws are shared across rules and prompt sets.
- Training: LoRA $r=8$, $\alpha=16$, 80 epochs.
- Step and anchor: $\lambda = 0.9$, $\tau_0 = 85.23$.

## Rules
- Never retrain, and never write into or overwrite `para_results/`. Analysis scripts go in `analysis/` and only read the saved files.
- BAcc is the primary metric. Always also report CVaR20 (the mean of the worst 20% of support draws; with 10 draws, the worst 2)
  and Macro-F1. CVaR20 is computed per prompt set and then averaged over prompt sets.
- The anchor-aware rule is frozen at c = 8. Do not re-tune it. The rule is
  $\eta = K/(K+k_0)$, $k_0 = c\, r_a/(1-r_a)$, where $r_a$ is the rescaled mean true-class softmax probability of the anchor on the support set.
- Paired tests use a Wilcoxon signed-rank test over the 10 draws in each (dataset, prompts, K) setting, with significance at p < 0.05.
  A setting whose differences are all zero counts as not significant.
- Writing style: no em dashes; formulas in LaTeX.
- Every number that goes into the paper must trace to a file in `para_results/` and a script in `analysis/`.

## Regenerate
```bash
python para_figures/make_figs.py para_results para_results/logprob_table.csv para_figures/out
python para_figures/make_f3.py para_results/pneumoniamnist/eta_curves.csv para_results/breastmnist/eta_curves.csv para_results/octmnist/eta_curves.csv para_results/logprob_table.csv para_figures/out
python analysis/verify_claims.py para_results
python analysis/make_tables.py para_results tables
```
