# Paper outline: what PARA's adaptive step really does (draft, updated 2026-11-12)

Working title options:
- "Does the Support Set Steer the Step? An Analysis of Prompt-Anchored Residual Adaptation for Biomedical VLMs"
- "When Geometry Says Nothing: Re-examining Adaptive Anchoring in Few-Shot Biomedical Vision-Language Models"

Type: analysis / evaluation paper (workshop or evaluation/reproducibility track). Evidence: claude/para-replication-findings.md.

## One-paragraph abstract (draft)
Few-shot adaptation of biomedical vision-language models is unstable across support sets. PARA addresses this by blending a frozen
text-prompt prediction with a trained visual prediction, using a step size computed from the geometry of the frozen support embeddings.
We reimplement PARA on BiomedCLIP and study it on six MedMNIST datasets with a repeated-support, worst-case (CVaR20) protocol and balanced
metrics. We find that (i) the geometric signals carry almost no class information in BiomedCLIP's frozen space, so the step reduces to the
same shot-count curve on every dataset, predictable from two model constants, and a plain shot schedule matches or beats it; (ii) the
per-image standardized blend becomes a hard anchor-or-visual switch on binary tasks; (iii) whether anchoring helps depends on the anchor's
reliability relative to the trained branch, which PARA does not measure; a pre-registered anchor-only step rule improved in-sample but failed
on held-out datasets; and (iv) blending log-probabilities instead of z-scores, with no new parameters, significantly improves PARA on
multi-class tasks but exposes an overconfident few-shot branch that erodes worst-case protection when the anchor is good.

## 1. Introduction
- Few-shot medical adaptation; support-set variance; why worst-case matters clinically.
- PARA's idea and claim (frozen anchor + residual; step from frozen support geometry).
- Our question: does the support geometry actually steer the step? Contributions (i)-(iv) above, plus the evaluation notes.

## 2. Background and related work
- PARA (Eq. 1-10 summary). BiomedCLIP. BiomedCoOp benchmark and prompts.
- Modality gap and cone effect (Liang et al., NeurIPS 2022): cite, do not claim.
- Blending ratios: Tip-Adapter / TaskRes (tuned), ProLIP (shot-dependent, validation-free), HOSO-Adapter (one held-out shot). vMFCoOp.

## 3. Setup
- Reimplementation details (LoRA r=8, alpha 16, 80 epochs, leave-one-out prototypes, lambda 0.9), verified against the paper:
  DermaMNIST GPT-4 zero-shot 53.55 vs paper 53.78 Acc.
- Datasets: Table 1. Protocol: 10 shared support draws x K in {1, 4, 16}, 1 seed; BAcc, Macro-F1, CVaR20; paired Wilcoxon per setting.
- Design vs held-out split and the pre-registration (dates, frozen rule, prompt policy, dataset choice from zero-shot screening only).
- Why BAcc: DermaMNIST accuracy rewards predicting "nevus" (Appendix A).

## 4. Finding 1: the geometry is uninformative, so the step is a fixed curve in K
- delta own vs other text (gap 0.002-0.014); phi of real classes = phi of random mixes; pairwise cosine 0.85-0.96.
- Derivation: phi = 1 + (K-1) * mean cosine -> predicted eta 0.32 / 0.60 / 0.80 (delta 0.55, cosine 0.87, averaged over 6 datasets)
  vs measured 0.33 / 0.60 / 0.80 on all 6 datasets (Figure 1).
- eta SD across draws <= 0.011; no correlation with the best eta per draw.
- Shot-only schedule vs PARA: 13 / 0 settings significantly better (design), +0.82 held-out.
- Centering restores signal (phi real 1.5-6.7 vs random ~1.0-1.7) but does not by itself give a better step (Appendix B).

## 5. Finding 2: the standardized blend is a switch on binary tasks
- Short proof: 2-class z-scores are (+1, -1); (1-eta) a + eta b keeps a's sign iff eta < 0.5.
- Empirical check: 1,046 / 1,046 binary rows equal zero-shot (eta < 0.5) or visual-only (eta > 0.5). Confirmed again on the full
  41-point eta grid (Figure 3): every binary run takes exactly one BAcc value below 0.5 and one above.
- Consequence: PARA = zero-shot at K=1 and visual-only at K>=4 on binary tasks.

## 6. Finding 3: anchor reliability relative to the trained branch decides whether anchoring helps
- Worst case at K=1: anchoring helps (OCT, Pneumonia, Breast anchor-protected) vs hurts (Blood, Organ) (Figure 5).
- Best eta orders by relative anchor quality (Figure 4): roughly 0.17 to 0.98. PARA's step is identical across these cases.
- Pre-registered test: anchor-aware rule (c = 8) in-sample gains vs held-out failure (Table 4); OrganAMNIST shows why (r_a higher than OCT,
  yet best eta 0.91-0.96 because the visual branch is far stronger). HOSO-style as the reliability-measuring baseline: matches oracle on Organ,
  noisy on binary.

## 7. Finding 4: blend space matters
- Log-prob blend, same eta, no new parameters: multi-class +4.86 BAcc (14 / 0 settings), binary +0.36 (2 / 0).
- Gains concentrate where the anchor is weak (Blood K=1 CVaR 27.0 -> 44.0; Organ 34.6 -> 43.0).
- Trade-off: overconfident few-shot branch erodes worst-case protection with good anchors (Pneumonia K=1 CVaR 67.4 -> 53.6; Breast 62.3 -> 50.5; OCT 47.7 -> 43.5).
- Figure 3 (eta-grid curves, mean BAcc): log-prob turns the binary switch into a smooth curve.
  - BreastMNIST: interior optimum that beats both branches (K=1 64.6 at eta 0.08 vs 62.3; K=4 69.3 at 0.35 vs 65.5; K=16 74.3 at 0.52 vs 73.3).
  - PneumoniaMNIST: no gain; best is an endpoint (anchor at K=1, visual at K=4/16). Only "your prompts" available for Pneumonia.
  - OCTMNIST: small gains (+0.4 to +1.2) but the log-prob optimum sits much lower (0.15 / 0.40 / 0.55 vs z-score 0.55 / 0.80 / 0.88):
    the same eta means different things in the two blends, so PARA's schedule does not transfer.
- State honestly: motivated by the binary flaw seen on held-out data; no parameters fitted.

## 8. Discussion
- What a good step needs: reliability of the trained branch relative to the anchor, and calibrated confidences.
- Practical recommendations: report BAcc + CVaR; include binary tasks; compare against shot-only and visual-only baselines; check
  whether adaptive signals actually vary across support draws.
- Limitations: 1 seed; K in {1, 4, 16}; MedMNIST 128 px and splits (absolute numbers not comparable to the paper); one backbone;
  one multi-class held-out dataset; BreastMNIST test set small.
- Future work: estimating trained-branch reliability relative to the anchor without test labels (r_a measured only the anchor side);
  calibrated blending (support-set temperature scaling); more held-out datasets.

## Figures (all made 2026-11-12, PNG + PDF, scripts make_figs.py and make_f3.py)
- **F1** PARA's eta vs K on all 6 datasets + predicted curve from two constants. Data: sweep_*_eta.csv.
- **F2** Geometry check: delta own vs other text; phi real vs random; raw vs centered. Data: 14.2_geometry_check.txt.
- **F3** BAcc vs eta (3 x 3 grid: Pneumonia, Breast, OCT x K = 1, 4, 16), z-score vs log-prob, IQR bands. Data: eta_curves.csv (cell 17.3).
- **F4** Best eta (oracle) vs zero-shot minus visual-only BAcc, both blends, PARA's eta as dashed lines. Data: logprob_table.csv.
- **F5** Worst-case BAcc (CVaR20) at K=1 per dataset ordered by relative anchor quality. Data: logprob_table.csv.

## Tables
- **T1** Datasets: classes, test size, image size, prompt sets, zero-shot BAcc and anchor quality, design vs held-out. Data: 16_screening.txt.
- **T2** Main results: mean / CVaR20 BAcc at K = 1, 4, 16 for zero-shot, visual, PARA, shot-only, PARA log-prob, oracle. Data: logprob_table.csv.
- **T3** Paired tests: shot-only vs PARA; log-prob vs z-score (binary / multi-class). Data: cross_dataset_summary.txt, logprob_summary.txt.
- **T4** Pre-registered anchor-aware rule: design (in-sample, LODO) vs held-out. Data: cross_dataset_summary.txt.

## Appendix
- A. DermaMNIST accuracy vs BAcc; per-class recall of both prompt sets.
- B. Centered geometry; failed frozen-reliability rule; c sensitivity (4 vs 8).
- C. HOSO-style details and results.
- D. Pre-registration record (dates, rule, prompt policy, screening table, predicted eta).
- E. Implementation details and hyperparameters; code and prompt files released.

## To do before submission
1. Tables T1-T4 (from existing files).
2. Supervisor review of outline, figures and findings; choose venue.
3. Optional, depending on venue: 3 seeds; K = 2, 8; second backbone (plain CLIP) to show Finding 1 is not BiomedCLIP-specific.
4. Re-check arXiv for a PARA v2 and code release; verify every related-work citation directly.
5. Write, release code (notebook) and the BUSI prompt mapping.

Rejected idea (2026-11-12): Gemini's "PARA + PRiSM regularizer + NeuFS active support selection to maximize CVaR20". CVaR20 is over support
draws and cannot be optimized in one run; active selection removes the random-draw protocol; does not address relative reliability.
