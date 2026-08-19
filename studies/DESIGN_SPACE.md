# Merge-pruning method: design space & experiment plan

The method has four decision slots. This document enumerates the options for
each, so experiments can reference them by ID (e.g. `1B4 + 3M1 + 4F2 + 2S1`).

Status tags: **[impl]** implemented in `studies/gram_stability/`,
**[repo]** exists elsewhere in the repo, **[derived]** math done / not coded,
**[idea]** proposed only, **[✓]** empirically validated, **[✗]** empirically
falsified (see Evidence notes).

Data-requirement tiers: **DF** = fully data-free (weights + input format only),
**DOM** = needs the input domain (per-feature ranges; one streaming pass or
free for images), **DL** = data-light (~100 calibration samples for moments),
**DD** = data-dependent (full calibration set / forward passes).

Per-neuron notation: `u = w/‖w‖`, `γ = (wᵀx₀+b)/‖w‖` (signed distance of the
boundary from the box center), `α = ‖w‖`, `c` = outgoing column,
`a = α‖c‖` (loudness), `q̃ = [R₀u; γ]`.

---

## 1. Scoring rule (which pair/cluster to merge next)

Three sub-axes: the **geometry/measure**, the **weighting**, and the
**search procedure** that consumes the scores.

### 1A. Worst-case / geometric family (Euclidean on covectors)

| ID | option | tier | status | notes |
|---|---|---|---|---|
| 1A1 | Ward cost on `q̃` with sphere radius `R₀` | DOM | [impl][✓] | current default; robust on all datasets (see Evidence E1, E3) |
| 1A2 | axis-aligned **ellipsoid**: `q̃ = [r⊙u; γ]`, per-feature half-ranges `r` | DOM | [derived] | one-line change; predicted big win on images (border pixels have `r≈0`) |
| 1A3 | exact box metric: `Σ_j r_j\|Δu_j\| + \|Δγ\|` (weighted L1) | DOM | [idea] | tighter than 1A2 but non-Euclidean → breaks Ward/parallel-axis; pair with greedy search only |
| 1A4 | origin-centered ablation: `ρ` instead of `γ` | DF | [impl] | expected to fail with depth (beta-saturation finding); keep as sanity arm |
| 1A5 | linear vs squared cost: bound-greedy `(a_ia_j/(a_i+a_j))·d` vs Ward `·d²` | DOM | [idea] | linear = literal Theorem-2 increment; squared = Ward. Cheap ablation |

### 1B. Average-case / functional family (exact expected damage under a measure)

The kernel `K̃_ij = (c_iᵀc_j)·E[h_i h_j]` gives the exact expected squared
layer-output error of any merge — the sub-axis is **which measure** defines E.

| ID | measure | tier | status | notes |
|---|---|---|---|---|
| 1B1 | isotropic Gaussian `N(x₀, σ²I)`, `σ = R₀/√(d+2)` | DOM | [impl][✗ high-d] | wins on d≤2, collapses on MNIST layer 0 (E2): real projection spread ≠ isotropic model (2.3× off at layer 0, 3–10× the other way deep) |
| 1B2 | exact uniform ball (incomplete-beta quadrature) | DOM | [impl, validation only] | same prior as 1B1, 10³× slower; keep as oracle for the closed form |
| 1B3 | box-product measure via CLT: per-pair Gaussian with `var_i = Σ_j u_{ij}²r_j²/3`, `cov = Σ_j u_{ij}u_{kj}r_j²/3` | DOM | [derived] | the **data-free ellipsoid** measure; same closed-form G, per-pair moments; fixes part of 1B1's mismatch at zero data cost |
| 1B4 | **matched Gaussian** `N(μ, Σ)` from ~100 calibration samples | DL | [derived] | same G with per-pair standardization; `pca_test`/`calib_test` already justify (μ,Σ) sufficiency; predicted winner of the family |
| 1B5 | empirical measure (score = error on calibration activations) | DD | [idea] | ceiling/baseline; converges to OSSCAR's currency |
| 1B6 | effective-dimension correction: `σ = R₀/√d_eff`, `d_eff` from a weights-only proxy (e.g. stable rank of W) | DF | [idea] | cheap patch for 1B1; test only if 1B3/1B4 disappoint |

**Sub-decisions inside 1B** (apply to every measure):
- **Include α and c in the integrand?** Yes for damage ranking — the exact
  formula `E‖Σc_iα_iσ_i − c̄σ̄‖²` requires both (α scales the activation, c
  scales how it is read). Ablations: drop c (α only), drop both (pure-shape
  kernel). Prediction: full weighting wins; ablations quantify each factor.
- **Dimension scaling (`d_F ~ R/√d`)**: irrelevant *within* a layer (ranking is
  invariant to monotone rescaling); matters only for **cross-layer budget
  allocation**. Options: raw; dimension-normalized `D_F² = (d+2)/R²·d_F²`;
  output-normalized `E‖ΔZ‖²/E‖Z‖²` (recommended — comparable across layers by
  construction).
- **Pointwise norm**: L² (current), L¹, sup. Note sup recovers the worst-case
  family — 1A and 1B are the two ends of one dial.

### 1C. Other criteria & search procedures

| ID | option | tier | status | notes |
|---|---|---|---|---|
| 1C1 | Srinivas–Babu saliency (`⟨a²⟩·ε²`) | DF | [repo] | baseline (`data_free_merge`) |
| 1C2 | angle+offset thresholds | DF | [repo] | baseline (`redundant`; has the `abs()` sign bug — fix before benchmarking) |
| 1C3 | activation-pattern overlap (Hamming distance of masks on calibration batch) | DD | [idea] | classic data-dependent baseline |
| 1C4 | OBS/Hessian saliency | DD | [idea] | baseline only if OSSCAR comparison needs it |
| 1C5 | joint two-layer spectral perturbation (score = Gram damage of layer ℓ **and** ℓ+1 from the surgery) | DOM | [idea] | addresses c-direction blindness of 1A |
| 1C6 | **global row selection**: leverage scores / BSS spectral sparsification of the weighted covector matrix, or of `K̃` | DOM/DL | [idea] | the non-greedy contender; picks a width-K subset with spectral guarantee instead of a merge sequence |
| 1C7 | search procedure axis: greedy agglomerative (current) vs weighted k-means at fixed k vs k-medoids vs spectral clustering on `K̃` vs dendrogram + optimal cut | — | [impl: greedy] | same metric, different optimizer; k-medoids composes with 3M6 (no new hyperplanes) |

---

## 2. Stopping rule (how far to merge)

Key property: **every stopping rule is a post-hoc cut of the recorded merge
trajectory** — one sweep per (scoring × merge × fan-out) combo evaluates ALL
stopping rules offline. Never rerun sweeps to compare stops.

The stated requirement — "don't disturb the general placement of boundaries,
not just test loss" — is measured by S6/S2/S10 (geometry meters), while
S1/S7 are error meters. Report both families for every arm.

| ID | rule | tier | status | notes |
|---|---|---|---|---|
| 2S1 | certified Lipschitz budget: `Σ_C A_C(R₀ε_u + ε_γ) ≤ τ` | DOM | [impl][✓] | Spearman 0.99–1.0 vs true error (E1); only rule with error units + all-inputs guarantee; composes end-to-end via Lipschitz products |
| 2S2 | weighted-Gram drift tripwire: `wq_fro` / `wq_stable_rank` beyond threshold | DOM | [impl][✓] | best-aligned snapshot property (E1); `wq_trace` [✗ blind] (orientation block mechanically conserved), `anchor_*` [✗ false alarms] |
| 2S3 | first-moment drift `m1_real_ratio` | DOM | [impl][✓] | cheapest (no SVD); validated well-aligned |
| 2S4 | dendrogram elbow (first sustained jump in merge cost) | DF/DOM | [impl, analysis] | zero knobs; the principled version of Srinivas–Babu's histogram cutoff |
| 2S5 | fixed width / fixed fraction | — | [impl] | baseline; also needed for matched-width method comparisons |
| 2S6 | **arrangement fingerprint**: crossing patterns of boundaries along random lines/2-planes through the box preserved within tolerance | DOM | [idea] | the most literal "boundary placement" meter; ignores saturated boundaries automatically; connects to linear-region counting |
| 2S7 | expected-error budget under measure: cumulative `E‖ΔZ‖²/E‖Z‖² ≤ τ` | per 1B tier | [derived] | closed form from `K̃`; the average-case sibling of 2S1 |
| 2S8 | val-loss early stop | DD | [impl, eval only] | data-dependent baseline / oracle |
| 2S9 | **cross-layer allocation**: global budget spent greedily across layers by marginal (normalized) cost, vs per-layer budgets | — | [idea] | interacts with the normalization decision in 1B; likely free capacity |
| 2S10 | kernel-Gram (`K̃`) spectrum drift | per 1B tier | [idea] | the v2 spectral object; stable-by-construction under harmless merges — test whether it beats 2S2 |

---

## 3. Merge rule (incoming update: the new neuron)

Controls the **disagreement-region** error channel. Suboptimality here is
first-order in cluster dispersion — this slot matters more than slot 4.

| ID | rule | status | notes |
|---|---|---|---|
| 3M1 | loudness-weighted covector mean (`ξ̄ = Σξ_i`, renormalized) | [impl][✓] | current; exact for coincident boundaries; associative (order-free) |
| 3M2 | keep-survivor (no new hyperplane; Srinivas–Babu) | [repo] | baseline; strictly dominated by 3M1 in theory — verify empirically |
| 3M3 | medoid (best existing member represents the cluster) | [idea] | no new directions → the only rule that ports directly to conv/BN layers; pairs with k-medoids (1C7) |
| 3M4 | rank-1 SVD of the cluster contribution matrix `Σ(α_ic_i)q_iᵀ` | [derived] | jointly optimal with slot 4 in the co-active region; guard: only inside geometrically tight clusters (else it trades affine error for activation error) |
| 3M5 | measure-optimal neuron: `argmin E‖cluster − c̄σ(ūᵀx−ρ̄)‖²` via kernel closed forms | [idea] | nonconvex; optimal `ū` lies in span of member `u`s → small subproblem; initialize at 3M1 |
| 3M6 | disagreement-aware offset: keep `ū` from 3M1, 1-D search on `ρ̄` minimizing disagreement-slab volume ∩ box | [idea] | cheap refinement targeting exactly the first-order error term |
| 3M7 | rank-r replacement: cluster of k → r>1 neurons (top-r SVD or inner k-means) | [idea] | generalizes merging; changes width accounting — evaluate on the accuracy-vs-width curve only |
| 3M8 | weighting ablation inside the mean: `a = α‖c‖` (current) vs `α` vs `‖c‖` vs uniform vs downstream-cascaded importance | [impl: α‖c‖] | same ablation grid as scoring weights; run once, share conclusion across slots |

Invariants any candidate must pass (already in self-tests): exact on duplicate
covectors with arbitrary outgoing vectors; associativity (or declared
order-dependence); `Σα_ic_i ≈ 0` degenerates to deletion.

---

## 4. Fan-out update (outgoing columns of the next layer)

Controls the **affine-match** channel; suboptimality is second-order in
dispersion. Cheap to get right — F2 is closed-form and strictly better than F1.

| ID | rule | tier | status | notes |
|---|---|---|---|---|
| 4F1 | plain sum `c̄ = Σα_ic_i` (survivor column only) | DF | [impl][✓] | current; overshoot `A_C/‖g_C‖` logged as diagnostic (hit 7–40 only in end-of-dendrogram garbage merges) |
| 4F2 | least-squares projection `c̄ᾱ = Tq̃̄/‖q̃̄‖²` | DOM | [derived] | optimal given the merged hyperplane; = the paper's slope-matched rule at m=1; flag-level change |
| 4F3 | joint rank-1 SVD (with 3M4) | DOM | [derived] | one decision with 3M4, not a separate arm |
| 4F4 | **global repair**: re-solve ALL surviving columns, projecting removed contributions onto the span of survivors — normal equations in the kernel `K̃` (data-free-under-measure) or on calibration activations (= `repair_test` joint regression / OSSCAR-lite) | DOM/DL/DD | [idea / repo-prototype] | the biggest candidate gain in this slot; `repair_test` showed ~190× over pairwise on always-on folds |
| 4F5 | bias compensation: fold residual midpoint (certified) or mean (measure) into next-layer bias | DOM/DL | [derived] | orthogonal toggle; halves sup error / zero-means the residual; test as on/off with every arm |
| 4F6 | OSSCAR least squares on full calibration set | DD | [repo] | ceiling |
| 4F7 | no update (naive delete) | — | [repo] | context baseline |
| 4F8 | brief fine-tune after pruning | DD | [repo trainer] | out of scope for method search; final-table option only |

---

## Evidence so far (what's already settled — don't re-run)

- **E1** (`gram_stability` main run): certified bound tracks true error with
  Spearman 0.99–1.0 everywhere; `wq_fro`/`wq_stable_rank`/`m1_real` drift
  aligned with error onset; `wq_trace` blind (orientation block conserved);
  raw anchor/covector Grams false-alarm ~8× early. First moment conserved to
  2e-16. → fixes 2S1–2S3 as validated, kills anchor-Gram stopping.
- **E2** (`compare_metrics` run): functional metric with isotropic prior (1B1)
  wins 9/12 on d≤2, loses 16/18 on d=784 (layer-0 capacity 0.06 vs 0.37).
  Diagnosed: real projection spread 2.3σ at layer 0 (structured data), 0.1–0.3σ
  deep (IBP blowup); 36% declared Gaussian-dead of which 61% empirically
  active. → the *measure* is the failure point, not the metric; motivates
  1B3/1B4; establishes 1A1 as the robust default.
- **E3** (engine self-tests): duplicate-exactness incl. arbitrary fan-out,
  associativity, dead-absorption under 1B, Gaussian closed form ≡ arc-cosine
  ≡ ball MC (d=50, 784).

- **E4** (Phase A, five arms, four MLP configs, 20260818): on LAYER-LOCAL
  error, `func_matched` (1B4) beats ward 27-2-1 and repairs the MNIST layer-0
  collapse (cap_err1 0.376 vs 0.141; cap_acc1pt 0.769 vs 0.616). Both
  data-free fixes FALSIFIED: 1A2 ellipsoid 6-5-19, 1B3 box-CLT 7-0-23 — the
  missing ingredient is feature *correlation*, not per-feature scale, so full
  covariance (~128 samples) is the minimum sufficient repair. On ACCURACY
  capacity the arms are at parity in deep layers (1B4 wins decisively only at
  layer 0): accuracy tolerates so much layer error that all arms exhaust real
  clumps long before the accuracy budget binds. → Selection is solved;
  accuracy-level gains must come from Phase B (surgery: 4F2/4F4) and
  loss-aware weighting, not better selection.
- **E5** (Phase A on pretrained ImageNet ResNet-18 / Imagenette, block-internal
  channels, calibration boxes + patch moments): layer-local merge capacity is
  NEAR-ZERO for every arm (cap_err5 ≈ 1.6-4.8% of channels; the first merge
  already costs ~1% layer error) — a well-trained ImageNet backbone has
  essentially no clump structure in block-internal filters, and no saturated
  filters (BN suppression, confirming the cnn_sigmafold finding; sat_absorbed
  = 0 everywhere). Yet ACCURACY capacity on Imagenette is large (19-76% of
  channels per block at -1pt): the backbone is over-provisioned for the
  10-class task, not redundant in weight space. `func_matched` is best where
  arms differ (slot 0: 0.76 vs 0.57; slot 7: 0.56 vs 0.19) and never
  catastrophic. → On large pretrained models, capacity is TASK over-capacity;
  exploiting it needs loss/task information (4F4-with-data, 4F6, fine-tune) —
  certified weight-space merging alone yields only a few percent there.

- **E6** (Phase C1, stopping rules post-hoc on all recorded trajectories,
  `stopping_rules.py`; oracle = last step within layer-error tol; one global
  knob calibrated on MNIST at <=5% violations, tested frozen on
  fashion/sine/shape2d): **within-dataset, rules transfer well across seeds
  and layers** (bound/wq_fro/m1 capture 0.6-0.77 of oracle capacity at zero
  violations). **Across datasets, no smart rule transfers safely at high
  capture**: bound(2S1) and the drift rules (2S2/2S3) violate on 43-71% of
  test runs (the bound's looseness factor and the drift->error scale are
  dataset-dependent -- E1's Spearman ~1 is an ordering fact, not a
  calibration fact). The only safe cross-dataset rule is normalized
  cumulative predicted damage (2S7): ~0.50 capture at 0-3% violations,
  beating fixed-fraction at tol=1%. Elbow (2S4) is weak everywhere.
  -> Operating rule: calibrate the certified-bound threshold once per
  (architecture, dataset) domain -- cheap and then reliable; if zero
  calibration is allowed, use cum-damage with the global theta and accept
  ~50% of oracle capacity.

- **E7** (Phase C2, cross-layer allocation, `phase_c2.py`, mnist+fashion, 3
  seeds, all layers pruned JOINTLY): with `func_matched` selection, **64-70%
  of ALL hidden units are removable at -1pt val acc**, and the allocation
  strategy barely matters — equal fractions, raw greedy, and normalized
  greedy all land within ~5pts of the brute-force grid oracle (mnist: equal
  0.647, greedy_norm 0.650, grid 0.700; fashion: greedy_raw 0.645 ≈ grid
  0.640). Sequential recomputation of dendrograms on the pruned model does
  NOT beat intact-model partitions (parity for func_matched; much worse for
  ward on mnist, 0.31 vs 0.48) → **intact-model dendrograms are sufficient;
  the cheap one-pass deployment recipe is sound**. WARD COSTS MUST NOT drive
  cross-layer greedy (0.41-0.48 vs its own equal-frac 0.55-0.62): worst-case
  geometric costs aren't comparable across layers even normalized — in the
  data-free tier allocate by equal fractions. Raw-vs-normalized greedy is
  inconclusive on MLPs (norm better on mnist, raw on fashion); keep
  normalized as the principled default. CAVEAT: MLP layers here have similar
  per-layer capacities, so allocation had little to exploit — on ResNet the
  per-slot spread is large (19-76%), so re-test allocation there (cluster
  P2) before trusting "equal fractions is enough" beyond MLPs.

- **E8** (Phase B, surgery grid, `phase_b.py`, mnist+fashion, joint
  equal-fraction pruning, func_matched partitions fixed across variants):
  **kernel global repair (4F4) is the single largest gain in the whole
  search**: capacity@-1pt jumps 0.618→0.772 (mnist) and 0.547→0.772
  (fashion) over the sum surgery — +15 to +22 pts of the WHOLE network's
  hidden units, using zero data beyond the same 128-sample moments (the
  normal equations G_kk C_new = G_ko C are closed-form kernel evaluations).
  It even beats C2's allocation grid-ORACLE under the old surgery (0.700) —
  surgery dominates allocation, as E4 predicted. Full hierarchy at -1pt:
  survivor+sum (3M2, Srinivas-Babu-style) < mean+sum (3M1: the weighted-mean
  hyperplane is worth +7 to +12 pts over keep-survivor) < mean+proj (4F2,
  +4 to +8) < mean+global (4F4). Bias compensation (4F5) adds +4 pts to
  sum-based surgery but nothing on top of global (its LS residual is already
  ~zero-mean). Both datasets reach 0.810 at -2pt. → **Operating surgery:
  3M1 + 4F4 (+4F5 free). Update the OSSCAR comparison plan: our best arm is
  now mean+global at the same 128-sample budget.**

- **E9** (Phase D vs OSSCAR, `phase_d.py`, mnist+fashion, joint
  equal-fraction pruning, 12-point width grid => +-0.08 resolution): **at the
  matched 128-input budget our pipeline and OSSCAR are at parity** — every
  difference is exactly one grid step, in both directions (OSSCAR +1 step on
  mnist@-1pt: 0.790 vs 0.712; ours +1 step on fashion@-0.5pt: 0.790 vs
  0.712). Full-calibration OSSCAR (2000 inputs) leads only at the extremes
  (mnist@-2pt 0.868 vs 0.790) and gains nothing on fashion. **hybrid128
  (our structure + empirical LS on the same 128 samples) == ours in every
  cell**: at this budget the Gaussian moments capture everything the raw
  samples know — the parametric kernel repair loses nothing (consistent with
  pca_test sufficiency). Reading: accuracy parity at matched data, while our
  side additionally provides the anytime dendrogram (one pass serves all 12
  widths; OSSCAR re-ran its search 36x per config), the certified data-free
  tier, and the merged-unit dictionary. Caveats: MLP scale; refine the width
  grid before quoting final numbers; ImageNet-scale P3 may separate the
  methods where redundancy is scarce.

## Experiment protocol

**Fixed harness for every arm** (already built in `compare_metrics.py`, extend):
capacity at layer-error {1,5,10}% and val-acc −1pt; AUC of acc-vs-width curve;
certified capacity (width at bound ≤ τ); geometry disturbance at matched width
(2S6 fingerprint distance + `wq_fro` drift); runtime; data budget. Medians over
≥3 seeds; win/tie/loss matrices per (seed, layer).

**Search order** (coordinate-wise from base `1A1 + 3M1 + 4F1`, never full
factorial — the slots are near-separable except (3M, 4F)):

1. **Phase A — scoring** (rest fixed): 1A1 vs 1A2 vs 1B3 vs 1B4, baselines
   1C1/1C2. Stopping rules cost nothing extra (post-hoc cuts) — evaluate all
   of 2S* on every Phase-A trajectory.
2. **Phase B — merge × fan-out jointly** (scoring = Phase A winner): the
   coupled grid {3M1, 3M2, 3M3, 3M4} × {4F1, 4F2, 4F4} + 4F5 toggle. 3M4
   pairs only with 4F3.
3. **Phase C — stopping + cross-layer allocation**: pick the operating rule
   from the (already-recorded) trajectories; test 2S9 global allocation vs
   per-layer.
4. **Phase D — external comparison**: winner (DF/DOM tier) and 1B4 winner
   (DL tier) vs OSSCAR at matched widths AND matched data budgets (give
   OSSCAR the same ~100 samples for the DL comparison).

**Priors** (to be confirmed/killed, cheapest decisive test first):
1A2 > 1A1 on images; 1B4 > everything within DL; 4F2 ≥ 4F1 always (free);
4F4 is the largest single gain in slot 4; 3M1 ≈ 3M4 inside tight clusters;
2S1 remains the operating stop, 2S6 the headline geometry meter.
