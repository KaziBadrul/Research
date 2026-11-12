# Verified findings

Checked 2026-11-12 with `analysis/verify_claims.py` (read-only). File paths are relative to `para_results/`.
Design datasets are Blood, Derma and OCT; held-out datasets are Breast, Organ and Pneumonia.

## Corrections to the chat's claims
- **Claim 8.** The raw own-text vs other-text $\delta$ gap at K=16, draw 0 ranges from **0.004 (Blood) to 0.043 (Organ: 0.604 vs 0.647)**, not 0.002 to 0.014.
  Without Organ the maximum is 0.018 (Pneumonia).
- **Claim 9.** Raw $\phi$ of a real class is within 1.3% of random mixed-class $\phi$ on every dataset except Organ. Organ is 13.01 vs 11.05 (natural mix) and 11.19 (uniform), 16% higher.
- **Claim 10.** Mean pairwise cosine between support images ranges from **0.671 to 0.962**. Organ is 0.800 same-class and 0.671 different-class; Breast is 0.839 different-class.
- **Claim 13.** The files support $\bar\delta = 0.550$ and $\bar c = 0.867$. The older values ($\bar\delta = 0.52$, $\bar c = 0.9$) are not supported.
- **Claim 14.** The maximum SD of $\eta$ across draws is **0.0108** (Breast, your prompts, K=1). 0.0035 is the maximum over the design datasets only.
- **Claim 15.** $\eta(S)$ is **weakly negatively** correlated with the per-draw best $\eta$: within-setting Spearman $\rho$ averages −0.13 over 27 settings; pooled within-setting $\rho$ = −0.12, p = 0.04.
  Say "no useful (positive) relation", not "no correlation".
- **Claim 16.** The source is `*/rule_scores.csv`, not `cross_dataset_summary.txt`.
- **Claim 19.** c = 8 maximises **pooled in-sample** CVaR20 BAcc over the three design datasets (56.07; c=4: 55.71; c=16: 55.61).
  Leave-one-dataset-out by CVaR20 chooses c = 16 (leaving Blood out), 8 (leaving Derma out) and 4 (leaving OCT out).
  The "Anchor-aware (LODO)" rows in `cross_dataset_summary.txt` choose c by mean BAcc instead (4, 4, 2).
- **Claim 25.** The log-prob oracle is **not** at least the z-score oracle everywhere: Derma K=16 is 58.62 vs 58.64 (−0.02). It is higher in the other 17 settings.
- **Claim 26.** At Pneumonia K=4 the log-prob curve peaks inside the range, at 82.81 with $\eta$ = 0.975, vs 82.75 for z-score (+0.06). K=1 and K=16 peak at the endpoints.
- `logprob_summary.txt` lists binary PARA as "won 51"; a recount gives 50. All other counts match.
- `16_screening.csv` gives different zero-shot BAcc from the sweeps for Derma (template prompts 24.80 vs "your prompts" 20.93; GPT-4 22.03 vs 21.39),
  Blood and Organ. The screening scores the full test split, while the sweeps score a fixed subset of `n_test` images.
  For Derma, the subset (2000 images) is almost the whole split (2005), so the template-prompt gap is likely a different prompt set, not the subset. Unresolved.
  Report zero-shot numbers from the sweep files (Table T1), not from the screening.

## Tables
`python analysis/make_tables.py para_results tables` builds T1 to T4 and T2b (Macro-F1). Every T2 and T4 number matches
`logprob_table.csv`, `cross_dataset_table.csv` and `cross_dataset_summary.txt` (within 0.005).

## A. Setup
| # | Verified value | Source |
|---|---|---|
| 1 | BiomedCLIP `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`; LoRA r=8, alpha=16; 80 epochs; $\lambda$=0.9; $\tau_0$ = 85.23 | `*/config.json`; notebook cell 8 output |
| 2 | The training loss uses leave-one-out prototypes $p_{c,-i}$, which include $t_c$. At test time, prototypes use all support images. | `train_visual_branch`, PARA_followup_5 |
| 3 | 10 draws, K in {1,4,16}, 1 seed. Draws come from `draw_support(k, draw_seed=d)`, which does not depend on the prompt set. | config.json; code |
| 4 | Derma 224 px; all other datasets 128 px | `medmnist_size` in config.json |
| 5 | Design: Blood, Derma, OCT. Held-out: Breast, Organ, Pneumonia. | `ORIGINAL_DATASETS`, cross_dataset_summary.txt |
| 6 | DermaMNIST GPT-4 zero-shot Acc 53.55 (paper 53.78); BAcc 21.39 | `dermamnist/sweep_50_GPT_4_per_draw.csv`; paper PDF |
| 7 | Both prompt sets for Derma, OCT, Blood and Breast; "your prompts" only for Organ and Pneumonia (in every file, including eta_curves). Breast GPT-4 prompts: "malignant tumor" vs "benign tumor" and "normal scan" pooled. | all CSVs; notebook |

## B. Geometry carries little information; $\eta$ is a fixed curve in K
| # | Verified value | Source |
|---|---|---|
| 8 | Raw gap: Blood 0.004, Breast 0.006, Derma 0.009, OCT 0.012, Pneumonia 0.018, Organ 0.043 | `*/14.2_geometry_check.txt` |
| 9 | Raw $\phi$ real/random ratio is 0.994 to 1.013 on five datasets and 1.163 on Organ | 14.2 |
| 10 | Cosine: same-class 0.800 to 0.962, different-class 0.671 to 0.945 | 14.2 |
| 11 | Centered $\phi$: real 1.62 to 6.73, random 0.94 to 1.74. Centered own $\delta$ (0.764 to 0.926) is below other $\delta$ (1.010 to 1.130) on all six datasets. | 14.2 |
| 12 | Mean $\eta(S)$ = 0.328 / 0.603 / 0.798 at K = 1 / 4 / 16 (pooled over all rows; per-dataset means 0.316 to 0.353, 0.587 to 0.610, 0.790 to 0.802) | `*/sweep_*_eta.csv`, quantity `eta(S)` |
| 13 | $\eta(K) = 0.9\,a/(1+a)$ with $a = \bar\delta\,(1+(K-1)\bar c)$, $\bar\delta$ = 0.550, $\bar c$ = 0.867, gives 0.320 / 0.598 / 0.797 | 14.2 |
| 14 | Maximum SD 0.0108 (Breast, your prompts, K=1); 0.0035 over design datasets only (Derma GPT-4, K=4) | sweep eta files |
| 15 | Spearman($\eta(S)$, per-draw oracle $\eta$): mean within-setting −0.13; pooled −0.12 (p = 0.04) | sweep eta files, quantity `oracle BAcc` |
| 16 | Shot-only $\lambda K/(1+K)$ minus PARA, BAcc: design +2.65, 13 significantly better / 0 worse of 18; held-out +0.82, 3 / 0 of 12 (won 30, lost 0) | `*/rule_scores.csv` |

## C. The z-score blend is a switch on binary tasks
| # | Verified value | Source |
|---|---|---|
| 17 | 1,046 of 1,046 binary rows (non-oracle, $\eta \ne 0.5$): BAcc equals zero-shot when $\eta<0.5$ and visual-only when $\eta>0.5$. Including oracle rows: 1,128 of 1,128. | breast and pneumonia `rule_scores.csv` |
| 18 | z-score curve takes one value for all $\eta<0.5$ and one for all $\eta>0.5$ in 60 of 60 Breast runs and 30 of 30 Pneumonia runs | `eta_curves.csv` |

## D. Relative anchor reliability decides; the pre-registered rule failed
| # | Verified value | Source |
|---|---|---|
| 19 | `anchor_aware_eta`: $k_0 = c\,r_a/\max(10^{-6}, 1-r_a)$, $\eta = K/(K+k_0)$. $r_a$ = mean true-class softmax of $\tau_0 u T^\top$, rescaled as $(p-1/C)/(1-1/C)$ and clipped to [0,1]. Frozen at c = 8 on 2026-11-11. The file etas match the formula to within 7e-16. | PARA_followup_5 sections 14.4 and 15.1; rule_scores |
| 20 | Held-out, c=8 minus PARA −0.68 BAcc (sig. 1/1 of 12); minus shot-only −1.50 (1/3); minus visual-only −2.20 (0/3) | rule_scores; cross_dataset_summary.txt |
| 21 | $r_a$: Organ 0.196 > OCT 0.164. Organ oracle $\eta$ (z-score) 0.908 / 0.963 / 0.955 | rule_scores; logprob_table.csv |
| 22 | Oracle $\eta$ ranges from 0.166 (OCT K=1, log-prob) to 0.981 (Blood K=16, z-score); z-score only: 0.210 to 0.981 | logprob_table.csv |

## E. The log-probability blend
| # | Verified value | Source |
|---|---|---|
| 23 | Same $\eta$, log-prob minus z-score, BAcc: multi-class +4.86, 14 / 0 of 21; binary +0.36, 2 / 0 of 9 | `*/logprob_scores.csv`; logprob_summary.txt |
| 24 | K=1 CVaR20, PARA z-score to log-prob: Blood 27.0 to 44.0; Organ 34.6 to 43.0; Pneumonia 67.4 to 53.6; Breast 62.3 to 50.5; OCT 47.7 to 43.5 | logprob_table.csv |
| 25 | Log-prob oracle is higher than the z-score oracle in 17 of 18 settings (exception: Derma K=16, −0.02); Breast K=1 70.4 vs 67.7 | logprob_table.csv |
| 26 | Best mean BAcc (z-score vs log-prob, with $\eta$): Breast K=1 62.3 vs 64.6 (0.075); K=4 65.5 vs 69.3 (0.35); K=16 73.3 vs 74.3 (0.525). OCT K=1 53.8 (0.55) vs 54.9 (0.15); K=4 66.8 (0.80) vs 67.2 (0.40); K=16 81.0 (0.875) vs 81.9 (0.55). Pneumonia: K=1 67.35 vs 67.35 (both at 0); K=4 82.75 vs 82.81 (0.975); K=16 89.40 vs 89.40 (1.0). | `eta_curves.csv` |

## F. Citations (checked on the web, 2026-11-12)
| # | Verified |
|---|---|
| 27 | Liang, Zhang, Kwon, Yeung, Zou, "Mind the Gap", NeurIPS 2022, arXiv 2203.02053. The authors' code repo states a default CLIP gap distance of 0.82. |
| 28 | ProLIP: Fahes et al., WACV 2026, arXiv 2410.05270. Current title: "CLIP's Visual Embedding Projector is a Few-shot Cornucopia". |
| 29 | HOSO-Adapter: Vorster, Maniparambil, O'Connor, Murphy, Molloy, "Hold-One-Shot-Out (HOSO) for Validation-Free Few-Shot CLIP Adapters", arXiv 2603.04341 |
| 30 | vMFCoOp: Shao et al., AAAI-26 (oral), vol. 40, no. 11, pp. 8851 to 8859; arXiv 2511.09540 |
| 31 | PARA: Jingxuan Kang, Qianying Yue, Che Liu, Chen Qin; arXiv:2609.33701; v1 only (27 Sep 2026); no official code |
