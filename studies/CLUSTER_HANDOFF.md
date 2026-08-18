# Cluster handoff: merge-pruning studies → full ImageNet

Written 2026-08-18 on the laptop, for a fresh Claude session on the cluster.
Read this + `studies/DESIGN_SPACE.md` (the option registry with evidence tags
E1–E5) before running anything. Everything referenced here is committed.

## Where the project stands

**Method (settled through Phase A of DESIGN_SPACE.md):** neurons/filters are
parameterized as `(u = w/‖w‖, γ = (wᵀx₀+b)/‖w‖, α = ‖w‖, c = outgoing column)`
with merge weight `a = α‖c‖`; merging is covector addition (associative)
realized as the loudness-weighted mean hyperplane + sum-rule fan-out surgery;
pairs are picked greedily by a scoring metric; stopping is a budget on the
certified Lipschitz bound (NOT a pairwise similarity threshold).

**Phase A verdict (E4, E5):**
- Scoring: `func_matched` (expected damage under Gaussian matched to layer
  moments from ~128 unlabeled inputs) if calibration data is allowed; `ward`
  (Euclidean Ward on box-centered covectors) as the certified data-free
  fallback. All data-free repairs of the functional metric (isotropic,
  ellipsoid 1A2, box-CLT 1B3) are FALSIFIED — the required information is the
  feature *correlation* structure, hence the ~128 samples.
- On MLPs: `func_matched` beats ward 27–2–1 on layer-local error; at the
  end-to-end accuracy level the arms are at parity in deep layers.
- On pretrained ResNet-18 (Imagenette, laptop, 1 seed): block-internal
  filters have ~NO weight-space redundancy (first merge already ≈1% layer
  error; 5%-budget capacity 1–3% of channels, all arms; zero saturated
  filters — BN suppression). Yet accuracy capacity on Imagenette was large
  (19–76%/block at −1pt) — pure TASK over-provisioning of a 1000-class
  backbone on 10 classes.

Laptop filter counts at −1pt Imagenette acc, per block (K=64,64,128,128,256,
256,512,512): ward 35/35/71/71/127/143/287/95 (45% total), func_matched
47/19/63/47/95/127/255/287 (49% total). Per-slot independently, single seed.

**THE open question the cluster answers:** on the REAL 1000-class ImageNet
task the over-provisioning excuse disappears. How much accuracy capacity
survives? Hypothesis from E5: very little at strict budgets — which would
mean weight-space merging on well-trained backbones needs the data-using
surgery (Phase B / 4F4) or task-subset framing to be useful. Either outcome
is a headline result.

## Code map (all under `studies/gram_stability/`)

- `merge.py` — parameterization, `IterativeMerge` (ward), `EllipsoidMerge`,
  IBP boxes for MLPs, certified bound, self-tests (`python merge.py`).
- `functional.py` — closed-form Gaussian kernel (Owen's-T based, no
  quadrature; validated vs MC and the exact arc-cosine/ball forms),
  `FunctionalMerge` (isotropic), `GaussianMeasureMerge(mu, cov)` (box-CLT or
  matched). Self-tests (`python functional.py`) include an MC check of the
  exact expected merge damage under N(mu, C).
- `compare_metrics.py` — Phase A harness for MLP configs (loads run_study
  checkpoints; arms registry; capacity reports).
- `resnet_phase_a.py` — the ResNet harness: BN folding (exact, eval mode),
  patch-space treatment (one covector per filter over im2col patches),
  calibration boxes/moments, exact realization back into the block (BN →
  identity-with-offset trick, γ=√(1+eps)), V-error eval + full-model val
  accuracy on GPU. Step-0 assert guarantees the folded realization is exact.
- `run_study.py` / `analyze.py` / `gram.py` — the earlier Gram-stability
  study (E1): which BᵀB properties are stable/predictive under merging.
- Baselines elsewhere in repo: `src/pruning/methods/{data_free_merge,
  redundant, osscar}.py`. NOTE: `redundant.py:37` has a known sign bug
  (`abs()` accepts antiparallel neurons) — fix before benchmarking it.

`src/models/resnet.py` exposes the prunable protocol (block-internal channels;
BasicBlock 1 slot, Bottleneck 2 slots; `outgoing_weights` patch-major;
`prune_layer` surgery). Pretrained: torchvision weights, 1000-way head kept
when output_dim==1000, sliced for known subsets (Imagenette).

## How to run on the cluster

```bash
# sanity: engines' self-tests (seconds)
python studies/gram_stability/merge.py && python studies/gram_stability/functional.py

# Phase A on real ImageNet, all slots, log to file (NOT through tail -- it buffers)
nohup python studies/gram_stability/resnet_phase_a.py \
    --config configs/experiments/imagenet/resnet/imagenet_resnet18_pretrained_cluster.yaml \
    > outputs_resnetA_cluster.log 2>&1 &
# options: --slots 0 7 (subset), --arms ward func_matched, --acc-every N
```

Laptop timing reference (RTX 4090): dataset build ~10–15 min (one-time,
CPU-bound ImageFolder decode), 8 slots x 4 arms sweep+eval ~13 min. Engines
are sub-second per sweep; per-step full-model accuracy evals dominate.

## Known gotchas / first engineering tasks

1. **The `imagenet` builder holds images in RAM as tensors** (`n_samples` +
   `n_test` × 0.6 MB each). 10k+10k ≈ 12 GB — fine on cluster nodes; for
   full-val (50k) evaluation write a streaming DataLoader path first, or
   accept a 10k-image val estimate (±0.5% acc noise; the −1pt budget needs
   ~2k+ val images to be meaningful — do NOT use a 400-image val like the
   laptop run).
2. `resnet_phase_a.py` evaluates accuracy on `bundle.val_ds` (a split of
   n_samples train-dir images). With 1000 classes, raise n_samples so the val
   split has ≥2000 images, or repoint accuracy eval at test_ds.
3. Multi-seed: pretrained weights are fixed; seeds vary the calibration/eval
   sampling (`seeds: [0,1,2]` in the config → run per seed; the harness
   currently loops config.seeds only in compare_metrics, resnet_phase_a uses
   seeds[0] — extend the loop or run 3× with edited configs).
4. Accuracy snapshots are quantized (`--acc-every`, stride = K/64): capacity
   numbers on 512-wide slots have ±32-filter granularity. Tighten for finals.
5. ResNet-50 (Bottleneck, 32 slots) should work unchanged via the protocol —
   verify the step-0 assert on one slot before a full run.

## Priority queue on the cluster

- **P0** — sanity: self-tests, then `--slots 0 7 --arms ward func_matched` on
  the cluster config; confirm baseline top-1 (~69.8% for torchvision
  resnet18 V1) and step-0 exactness.
- **P1** — Phase A confirm at scale: all 8 slots × {ward, func_matched},
  3 calibration seeds, ≥2k val images. Key readout: the −1pt (and −0.1pt)
  accuracy capacities on the REAL task vs Imagenette's 19–76%.
- **P2** — joint all-slot pruning with a shared global budget (design-doc 2S9;
  greedy across layers by marginal certified cost, output-normalized) → one
  deployable pruned checkpoint + its measured top-1. This is "our side of the
  table" for any external comparison.
- **P3** — Phase D vs OSSCAR at matched widths AND matched data budgets
  (give OSSCAR the same 128 images; also its full-calibration ceiling), plus
  the hybrid arm: our merged-channel structure + least-squares outgoing
  repair on calibration activations (design-doc 4F4 — expected biggest gain,
  `repair_test` prototype showed ~190x for the always-on fold).
- **P4** — Phase B surgery grid (4F2 projection — one-flag change; 4F5 bias
  fold), ResNet-34/50, and the Phase-A cleanup ablations (weighting
  a=α‖c‖ vs cascaded importance; greedy vs one-shot matching, 1C7).

## What NOT to redo (settled, see DESIGN_SPACE.md Evidence)

E1: stopping = certified-bound budget (Spearman 0.99–1.0 vs true error);
anchor-Gram meters falsified. E2/E4: isotropic/product-measure functional
metrics falsified on structured data; matched-Gaussian wins; ellipsoid and
box-CLT falsified. E3: engine exactness/associativity/conservation verified.
E5: laptop ResNet-18 numbers above. Raw outputs live only on the laptop
(`studies/gram_stability/outputs/`, gitignored) — the numbers that matter are
summarized here and in DESIGN_SPACE.md.
