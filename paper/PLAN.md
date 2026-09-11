# Paper plan: claims, figures, and the cells each needs

Working folder for the ICLR paper's evidence. Nothing is written here yet; this
file fixes WHAT we claim, WHICH figure carries each claim, and WHICH benchmark
cells that figure reads. `python paper/inventory.py` compares this plan with
`outputs/benchmark` and regenerates `paper/TODO.md` (never hand-edit TODO.md).
The theorem numbering follows `paper/Pruning.pdf`.

Decisions (2026-09-10): OSSCAR is the major baseline and keeps ITS OWN repair
throughout (no OSSCAR-under-our-repair arm). For every other arm "repaired"
means the empirical least-squares repair at ridge 1e-2 (OSSCAR's damping
strength; the 1e-8 guard is an appendix ablation). OPT-2.7b/6.7b run a
reduced set (`paper_big`): no Gaussian ablation, no medoid variants, no
mash_certified. Three seeds everywhere; OPT-2.7b/6.7b
start at one seed and get three if the budget allows. The delta_f score's
measure is SAMPLED by default; Gaussian moments are the `mash_gaussian_*`
ablation and exist only on fully-connected layers.

## Models

| entry | family | note |
|---|---|---|
| mnist_lenet | mixed (conv+fc) | trained here; Gaussian arms skipped (conv slots) |
| cifar10_resnet20, cifar10_resnet56 | conv + BN | merge dictionary refused; medoid only |
| imagenet_resnet18, imagenet_resnet50, imagenet_mobilenetv2 | conv + BN | medoid only; 11-point grid from 1% |
| imagenet_vit_b16 | fc (GELU) | both dictionaries; ReLU-response Grams are misspecified here |
| wikitext_opt125m, 350m, 1.3b, 2.7b, 6.7b | fc | both dictionaries; 10-point grid at 1.3b+ |

## Arm set (the "paper" tier in configs/benchmark/arms.yaml)

Baselines: `random`, `random_ridge`, `magnitude_mass`, `magnitude_mass_ridge`,
`osscar_norepair`, `osscar` (its own repair).

MASH, sampled delta_f: `mash_medoid_none_delta_f` (delete), `mash_medoid_sum_delta_f`,
`mash_merge_sum_delta_f` (merge + sum, data-free), `mash_ridge_medoid_empirical_delta_f`,
`mash_ridge_merge_empirical_delta_f`, and -- decided 2026-09-11 -- the headline
repaired MASH: `mash_full_medoid_empirical_delta_f` = medoid survivors, sum-rule
transfer, then the ridge repair fitted on EVERY calibration token (the same
~65k rows OSSCAR's Hessian uses; the other repaired arms subsample 20000) and
centred on the sum-rule column. `mash_full_medoid_empirical_cylinder` is its
certificate-score twin. Both reuse the existing plans.

MASH, Gaussian delta_f (FC only): `mash_gaussian_medoid_none_delta_f`,
`mash_gaussian_medoid_sum_delta_f`, `mash_gaussian_merge_sum_delta_f`,
`mash_ridge_gaussian_medoid_empirical_delta_f`, `mash_ridge_gaussian_merge_empirical_delta_f`.

Cylinder certificate: `mash_merge_sum_cylinder` (the certificate tier: no data),
`mash_medoid_sum_cylinder` (BN models), `mash_ridge_merge_empirical_cylinder` and
`mash_ridge_medoid_empirical_cylinder` (NEW ridge grid on the cylinder score),
`mash_certified` (tolerance-driven width).

## Claims -> figures -> cells

### C1. MASH removes a DIFFERENT set of neurons than magnitude and OSSCAR, and under the same repair the performance is not harmed
- **F1a overlap.** Removed-set overlap vs chance, per model, MASH-vs-OSSCAR,
  MASH-vs-magnitude, OSSCAR-vs-magnitude, at every swept width. Source:
  `removals.json` of `mash_medoid_none_delta_f`, `osscar_norepair`, `magnitude_mass`
  via `src.analysis.pruning_detail.overlap_from_removals`. (Removal sets do not
  depend on the repair, so the no-repair cells are the canonical source.)
- **F1b same repair.** `mash_full_medoid_empirical_delta_f` (the headline),
  with `mash_ridge_{medoid,merge}_empirical_delta_f` (20000-row subsample) as
  the row-budget ablation, `random_ridge`, `magnitude_mass_ridge`, against
  `osscar` with its own repair. The 20000-row repair added nothing over the sum
  rule on OPT-350m while OSSCAR's repair gained 5 ppl on an 80%-overlapping
  set; the full-row arm tests whether the row budget is that gap.
- Models: all. Seeds: 3.

### C2. Without any repair of the downstream layer, MASH performs better, because it replaces a useless neuron by a merged one
- **F2 no repair.** `random`, `magnitude_mass`, `osscar_norepair`,
  `mash_medoid_none_delta_f` (MASH deleted outright -- shows deletion is the
  wrong reading of MASH) and `mash_merge_sum_delta_f` (the merge; medoid+sum on
  BN models). The claim is that the green sum-rule curve sits above every
  deletion curve.
- Models: all. Seeds: 3.

### C3. For overall performance, OSSCAR is optimal
- **F3 repaired.** Every arm at its best: `osscar`, MASH ridge (merge where
  allowed, medoid otherwise), `random_ridge`, `magnitude_mass_ridge`. Plus a
  capacity table (first crossing at -1pt / -5pt / -10% ppl) per model.
- Also the cost table: plan + solve seconds per cell from `report.json`
  (MASH one plan for all widths vs OSSCAR's per-width Hessian).
- Models: all. Seeds: 3.

### C4. The measure matters on LLMs
- **F4 measure.** Sampled vs Gaussian delta_f, same emission, on the FC models:
  `mash_merge_sum_delta_f` vs `mash_gaussian_merge_sum_delta_f` (data-free) and
  `mash_ridge_merge_empirical_delta_f` vs `mash_ridge_gaussian_merge_empirical_delta_f`
  (repaired). Panel per FC model: ViT, OPT-125m/350m/1.3b(/2.7b).
- Seeds: 3 (1 at 2.7b).

### C5. Theorem 1: the representation-error hierarchy holds and the certificate is tight enough to use
- Eq. (8): E*_adapt(d-1) <= E_rec(psi) <= D_lambda(i,j) <= U_lambda(i,j) <= C_lambda(i,j),
  and eq. (10): U_mw <= min(U_avg, U_med).
- **F5a chain.** Per candidate merge on real layers: oracle / exact damage D /
  pre-activation Ward U / cylinder C, scatter or sorted curves, showing each
  inequality and its tightness. Source: `studies/score_vs_oracle/run.py`
  (chain.csv) -- port to `paper/figs/`, run on the paper's models at seed 0.
- **F5b certificate along a pass.** Corollary 1 / eq. (16): E_rec(t) <= Dup(t)
  <= W_t along the greedy pass, held-out. Source:
  `studies/score_vs_oracle/ward_certificate.py`, same porting.
- **F5c certified width.** `mash_certified` and `mash_merge_sum_cylinder`:
  realised error vs the certificate at every width, per model. From the
  benchmark cells (report.json carries the accepted certificate).
- Models: lenet, resnet20, resnet56, resnet18, vit, opt125m, opt350m. Seed 0
  for F5a/b (analysis, not a benchmark cell); 3 seeds for F5c.

## Code changes (done 2026-09-10 unless noted)
1. `mash_ridge` grid now covers the cylinder score too.
2. `paper` and `paper_big` tiers in arms.yaml; per-model manifests
   `configs/benchmark/manifest_paper_<entry>.txt` from
   `generate_benchmark_configs.py --tier paper` / `--tier paper_big`.
3. resnet50 and vit_b16 at 3 seeds; 2.7b/6.7b at 1.
4. TODO: port F5a/F5b scripts from studies/score_vs_oracle into paper/figs.
