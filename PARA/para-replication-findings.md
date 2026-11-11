# PARA replication: findings so far (DermaMNIST, OCTMNIST, BloodMNIST)

Paper: *Prompt-Anchored Residual Adaptation for Biomedical Vision-Language Models* (Kang et al., arXiv, Sep 2026).
Setup: own reimplementation on BiomedCLIP ViT-B/16, LoRA r=8, 80 epochs, lambda=0.9. MedMNIST official splits
(DermaMNIST 224 px, 2,000 test images; OCTMNIST 128 px, 1,000 balanced test images; BloodMNIST 128 px, 2,000 test images).
K = 1, 4, 16; 10 support draws; 1 seed. Two anchor sets: own prompts, and BiomedCoOp's 50 GPT-4 descriptions per class (the paper's).
Metrics: Acc, Macro-F1, balanced accuracy (BAcc). Paired Wilcoxon tests across draws. 18 settings = 3 datasets x 2 anchor sets x 3 K.

## FROZEN RULE (decided 2026-11-11, before any held-out dataset was run)
**Anchor-aware eta = K / (K + k0), k0 = c * r_a / (1 - r_a), with c = 8.**
r_a = anchor's mean probability of the true class on the labelled support images (frozen encoder), rescaled so chance = 0, perfect = 1.
c = 8 was chosen by worst-case (CVaR20) BAcc under leave-one-dataset-out on DermaMNIST, OCTMNIST and BloodMNIST, because worst-case
reliability is the paper's primary metric. c = 4 (best mean BAcc) is reported as a sensitivity check.
Every new (held-out) dataset is evaluated with c = 8 unchanged. Numbers for c = 8 on the original three datasets are in-sample and must be labelled so.
Do not change c, the rule's form, or the selection criterion after seeing held-out results; any later change must be tested on yet more new data.

## Prompt policy for held-out datasets (decided 2026-11-11)
- Main condition on every held-out dataset: the notebook's 5 template prompts (fully reproducible; same as "your prompts" on OCT and Blood).
- Second condition only where BiomedCoOp has an exact source match: **BreastMNIST = BUSI** with benign + normal merged. Class 0 (malignant)
  uses BiomedCoOp's "malignant tumor" prompts; class 1 (normal, benign) pools "benign tumor" + "normal scan" (100 prompts). Report the pooling.
- Approximate mappings (PneumoniaMNIST via COVID_19, PathMNIST via CHMNIST) are NOT used. Open-weight-LLM descriptions are kept in reserve
  (greedy decoding, fixed model version and instruction, generated once and published) in case reviewers ask for rich prompts on more datasets.

## HELD-OUT DATASETS (chosen 2026-11-11 from the zero-shot screening table only, before any training on them)
Screening (zero-shot, 128 px, up to 1,000 test images; quality = (BAcc - chance) / (100 - chance); r_a on 16 test images per class):
| dataset | prompts | classes | BAcc | quality | r_a | note |
|---|---|---|---|---|---|---|
| PneumoniaMNIST | templates | 2 | 67.4 | 0.347 | 0.417 | strongest r_a seen so far; anchor over-predicts "normal" (pneumonia recall 35%) |
| BreastMNIST | BiomedCoOp GPT-4 (BUSI) | 2 | 70.9 | 0.419 | 0.243 | paper's own prompt source; only 156 test images |
| BreastMNIST | templates | 2 | 53.6 | 0.073 | 0.119 | near chance: predicts "malignant" 98% |
| OrganAMNIST | templates | 11 | 26.3 | 0.189 | 0.175 | moderate anchor, many classes |
Reference rows (same screening): OCT templates 0.324 / r_a 0.257; Derma templates 0.123 / 0.077; Blood templates 0.028 / 0.007.
**Chosen: PneumoniaMNIST, BreastMNIST (both prompt sets), OrganAMNIST.** Together they give held-out anchors from weak (Breast templates)
through moderate (Organ) to strong (Breast GPT-4, Pneumonia), i.e. the good-anchor range the original three datasets lacked.
RetinaMNIST and PathMNIST were not screened (not downloaded) and are not used.
Predicted frozen eta from screening r_a (K = 1 / 4 / 16), for reference: Pneumonia 0.15 / 0.41 / 0.74; Breast GPT-4 0.28 / 0.61 / 0.86;
Organ 0.37 / 0.70 / 0.90; Breast templates 0.48 / 0.79 / 0.94. (Actual r_a is measured on each support draw.)
Run order: PneumoniaMNIST first (fast; send its 15.3 output as a sanity check), then BreastMNIST, then OrganAMNIST (slowest, 11 classes).

## Findings

1. **delta and phi are degenerate in BiomedCLIP's frozen space** (all 3 datasets, both anchor sets).
   - Own-text vs other-text delta differ by only 0.002 to 0.014; images are closest to their own class text for 1-2 classes only.
   - phi of a real class equals phi of 16 random mixed images (Derma 14.23 vs 14.32; OCT 14.85 vs 14.69; Blood 15.43 vs 15.23).
   - Mean cosine between any two images: 0.85 to 0.96 (Blood tightest: 0.962 same class vs 0.945 different class).
   - After centering each modality, both become informative (real phi 2.6 to 5.3 vs random ~1.0).
   - Derivation: phi = ||sum u_i||^2 / K = 1 + (K - 1) * (mean pairwise cosine), so with mean cosine ~0.9 and delta ~0.52 (both properties
     of the pretrained model) PARA's eta is predicted as 0.31 / 0.59 / 0.79 at K = 1 / 4 / 16, matching the measured 0.32 / 0.61 / 0.80 on all datasets.

2. **eta(S) collapses to a fixed function of K.** Across draws its SD is at most 0.0035 in all 18 settings, it is the same on every
   dataset (about 0.32 / 0.61 / 0.80 at K = 1 / 4 / 16), and it does not track the best eta from draw to draw.

3. **A geometry-free schedule beats PARA's eta.** eta = lambda * K / (1 + K):
   BAcc wins 150 of 180 paired draws vs PARA (mean +2.65); significantly better in 13 of 18 settings, never significantly worse.
   Macro-F1 +2.84 (12 better, 1 worse). Per dataset: Blood +5.56, Derma +1.59, OCT +0.80 BAcc.

4. **Whether anchoring helps depends on anchor quality, and PARA's eta ignores it.**
   - Anchor quality (BAcc of zero-shot): OCT ~50% (decent), Derma ~21% (chance for 7 classes), Blood 13-15% (chance for 8 classes).
   - Support-set anchor reliability r_a vs best eta in hindsight (BAcc oracle): Spearman -0.71 over the 18 settings (p = 0.001;
     -0.77 at K=1, -0.94 at K=4, -0.54 at K=16). Settings cluster within 3 datasets, so treat this as 3 independent data points.
   - PARA vs visual only, CVaR20 BAcc: OCT K=1 +10.5 to +10.8 (anchoring protects the worst case); Blood K=1 -21.8 to -25.6.
   - This replicates the paper's own CHMNIST failure mode on a second dataset.

5. **DermaMNIST accuracy is a majority-class effect.** GPT-4 anchors reach 53.55% Acc (paper: 53.78%), but BAcc is 21.39% vs 20.93% for own prompts:
   the gain comes from predicting "nevus" for ~65% of images. The best eta by Acc vs by BAcc can differ hugely (0.15 vs 0.72 at K = 1).

6. **Two candidate fixes failed.** Centered eta and a frozen-space reliability eta (r_v from frozen leave-one-out prototypes) both set eta badly:
   frozen-space reliability underestimates the trained LoRA branch (frozen r_v 0.14-0.27 while the adapted branch reaches 80-91%).
   r_a works where r_v failed because the anchor branch IS the frozen model, so measuring it in frozen space measures exactly what is used at test time.

7. **Section 15: anchor-aware eta** (no extra training), c chosen by leave-one-dataset-out (LODO, criterion: mean BAcc).
   - LODO picked c = 4 (Blood), 4 (Derma), 2 (OCT).
   - vs PARA: +4.94 BAcc, better in 140/180 draws, significantly better in 10/18 settings, never worse (Macro-F1 +5.51, 9 better / 3 worse).
   - vs shot-only: +2.29 BAcc, 6 better / 0 worse (Macro-F1 +2.67, 6 better / 4 worse).
   - vs HOSO-style: +0.58 BAcc, 0 / 0 of 12. vs visual only: +0.40 BAcc, 3 better / 1 worse. Its eta is 0.75-0.99, so it mostly turns anchoring off.
   - **Weakness:** on OCTMNIST (the only decent-anchor dataset) it gives up the worst-case protection: K=1 CVaR 41.8 vs PARA 47.7 (visual 37.1, zero-shot 50.4);
     K=4 CVaR 47.6 vs PARA 52.6. LODO with 2 weak-anchor datasets out of 3 biases c low.
   - Pooled over all settings (in-sample): mean / CVaR BAcc: PARA 57.5 / 52.0, shot-only 60.2 / 53.9, visual 62.1 / 54.2, anchor-aware c=4 62.6 / 55.7,
     c=8 62.3 / 56.1, oracle 64.0 / 57.9. OCT CVaR: c=4 55.2, c=8 56.9, c=16 57.8. No rule is best on every dataset.
   - Even c = 8-16 does not reach the OCT K=1 oracle (56.6 vs ~54), so a single c may not fit both weak and good anchors; the held-out test will show this.
   - HOSO-style (one held-out image per class, grid on held-out cross-entropy): eta 0.78-1.0, close to visual only; also loses OCT worst case (K=4 CVaR 47.7).

## Caveats
- 1 seed, 10 draws, 3 datasets, 3 shot levels; pooled counts treat the two anchor sets as separate settings although they share support draws.
- OCT and Blood use 128 px MedMNIST images and MedMNIST splits, so absolute numbers are not comparable to the paper (OCT zero-shot 49-52% here vs 72.11% in the paper).
- The anchor-quality vs best-eta relationship rests on only 3 datasets, 2 of them with near-chance anchors.
- Held-out: BreastMNIST has only 156 test images (noisy); PneumoniaMNIST's anchor is biased toward "normal", so BAcc and Acc may disagree.

## Related-work check (Nov 11, 2026)
Verified to exist:
- **Liang et al., "Mind the Gap", NeurIPS 2022**: modality gap and cone effect; CLIP gap distance 0.82 at default temperature. Explains finding 1; cite, don't claim as new.
- **ProLIP (Fahes et al., WACV 2026)**: https://arxiv.org/abs/2410.05270 ; validation-free; scales its regularization inversely with shots per class. Prior art for shot-dependent schedules, so "shot-only eta" is a baseline, not a contribution.
- **HOSO-Adapter (Vorster et al., arXiv 2603.04341, CVPR 2026 Findings)**: https://arxiv.org/abs/2603.04341 ; learns the blending ratio on one held-out support shot. Prior art for hold-out eta, so it is used as a baseline ("HOSO-style").
- **vMFCoOp (Shao et al., AAAI 2026 oral)**: hyperspherical/vMF prompt learning for biomedical VLMs; related work.
Not verified (do not cite without checking): Gemini report's claims about the PARA authors' other CVPR 2026 papers and PARA's code status.
Gemini report errors to avoid repeating: (a) it predicts eta saturates near lambda; measured eta does not saturate, it loses all draw-to-draw variance, and phi (ignored by Gemini) is the bigger cause; (b) its "Euclidean vs hypersphere" critique does not apply: delta = 1 - cos is a monotone function of the angle, phi is the vMF mean-resultant statistic, and PARA blends standardized logits, not embedding vectors.

## Possible paper framing
Analysis / evaluation contribution: in biomedical CLIP spaces, frozen cross-modal geometry carries almost no class information, so PARA's eta(S)
reduces to the same shot-count schedule on every dataset (predictable from two model constants); a simpler schedule beats it, and it cannot react
to anchor quality, which is what decides whether anchoring helps (OCT) or hurts badly (Blood). Anchor reliability measured on the support set
predicts the best eta; the frozen anchor-aware schedule (c = 8) is tested on held-out datasets.
Still needed: held-out results, a second backbone, 3 seeds, K = 1, 2, 4, 8, 16.
