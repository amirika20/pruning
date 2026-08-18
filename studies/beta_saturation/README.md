# Beta-saturation study

**Hypothesis.** Neurons that are *always* activated across all training points, or
*never* activated across all training points, have the largest hyperplane magnitude

```
|beta| = |b| / ||w||
```

**Geometric intuition.** A ReLU neuron's activation boundary is the hyperplane
`w·x + b = 0`, whose signed distance from the origin is `-b/||w|| = -beta`. If that
hyperplane sits far outside the data cloud (|beta| large relative to the data scale),
every training point lands on the same side of it — so the neuron is either always on
or always off. Always-off neurons are prunable for free (the `silent` method); always-on
neurons are affine on the data and can be folded into the next layer. If |beta| alone
predicts saturation, it is a **data-free** pruning score.

**Deep-layer follow-up.** |beta| is distance from the *origin*, but layer 1+ inputs are
post-ReLU and live in the nonnegative orthant, far from the origin — so |beta| loses its
meaning with depth. What actually determines saturation is whether the pre-activation
distribution `z = w·x + b` crosses zero. The study therefore compares a **ladder of
candidate scores**, ordered by how much data they need:

| score | formula | needs |
|---|---|---|
| `abs_beta` | `\|b\| / ‖w‖` | nothing (the original hypothesis) |
| `ibp_score` | interval-bound margin / ‖w‖ | only the input box (pixel range) — comes with a *proof* |
| `dist_mean` | `\|w·μ + b\| / ‖w‖` | one vector: the mean layer input μ |
| `z_margin` | `\|E z\| / std(z)` | pre-activation moments (statistical upper bound) |
| `abs_b`, `w_norm` | — | naive baselines |

## What is measured

For every hidden neuron of a trained network:

| column | meaning |
|---|---|
| `act_freq` | fraction of training points with pre-activation > 0 (for conv: fraction of sample × spatial-position pairs) |
| `beta`, `abs_beta` | `b / ||w||` from `src.pruning.geometry.compute_hyperplane_params` |
| `dist_mean` | `\|w·μ + b\| / ‖w‖` — hyperplane distance from the layer-input centroid |
| `z_mean`, `z_std`, `z_margin` | pre-activation moments over the train set; `z_margin = \|z_mean\|/z_std` |
| `cos_w_mu` | cosine(w, μ) — alignment with the mean input direction (dead: expect < 0; always-on: > 0) |
| `frac_pos_w` | fraction of positive weights (deep inputs are nonnegative, so sign structure controls w·x) |
| `ibp_lo`, `ibp_hi`, `ibp_dead`, `ibp_always`, `ibp_score` | interval-bound-propagation bounds on z over the input box; `ibp_dead/always` are data-free *certificates* (plain MLPs only) |
| `w_norm`, `b` | raw ingredients, for follow-up analysis |
| `saturation` | `|2·act_freq − 1|` — 0 = fires on half the data, 1 = always/never |
| `category` | `never` (p ≤ eps), `always` (p ≥ 1−eps), `mixed` — eps defaults to 0 (strict) |

Pre-activations are read where the ReLU reads them: the prunable layer's output, or the
paired BatchNorm's output when the model has one (matching how `compute_hyperplane_params`
takes `b` from the BN bias).

## Hypothesis tests (per layer — beta scales differ across layers, so layers are never pooled)

- **AUROC** of each candidate score separating saturated from mixed neurons. 1.0 = the
  score perfectly ranks saturated neurons above mixed ones; 0.5 = no signal; < 0.5 = inverted.
- **precision@k**: with k = number of saturated neurons, the fraction of the top-k
  neurons by |beta| that are actually saturated — the literal "highest |beta|" claim.
- **Spearman rho** between `saturation` and `abs_beta` — the graded, monotone version.
- **Mann-Whitney U** p-value (saturated > mixed, one-sided).
- **IBP certificate counts** vs actual saturated counts — how much of the saturation is
  *provable* data-free, and how that decays with depth.
- **Category profile**: median of every feature for never / always / mixed neurons —
  the descriptive "what is special about useless deep neurons" table.
- Plots: scatter (act_freq vs |beta|), rank plot (neurons sorted by |beta| with saturated
  ones marked — hypothesis true ⇒ colored points cluster left), box plots per category,
  `score_auroc.png` (the candidate-score ladder per layer), and `deep_profile` scatter
  (cos(w, μ) vs z_margin — dead neurons should sit left of 0, always-on right of 0).

## Running

From the repo root, with the project venv:

```bash
/home/amirabbas-kazeminia/Projects/.ml/bin/python studies/beta_saturation/run_study.py \
    --config studies/beta_saturation/configs          # all four cases

# a single case, or a quick smoke run:
/home/amirabbas-kazeminia/Projects/.ml/bin/python studies/beta_saturation/run_study.py \
    --config studies/beta_saturation/configs/sine_mlp.yaml --epochs 50

# tolerant saturation threshold (e.g. "active on ≤1% / ≥99% of points"):
... run_study.py --config ... --eps 0.01
```

Configs are ordinary `ExperimentConfig` yamls; only `data` / `model` / `training` /
`seeds` are used (nothing is pruned — trained networks are only measured).

## Cases

1. `sine_mlp.yaml` — 1-D regression, MLP [64, 64]: fully visualizable geometry.
2. `shape2d_mlp.yaml` — 2-D circle classification, MLP [64, 64].
3. `mnist_mlp.yaml` — flattened MNIST, MLP [256, 128, 64]: the real target setting.
4. `fashion_mnist_mlp.yaml` — same architecture, harder data.

Each runs 3 seeds. Outputs land in `studies/beta_saturation/outputs/<name>_<timestamp>/`
(gitignored): `neurons.csv`, `summary.json`, `report.txt`, and `plots/`.

## Prototypes

`prototypes/` holds the follow-up experiments that turned the study's findings into a
method design (each is standalone; they load checkpoints from `outputs/*/models/`, so
run the study first):

- `threshold_rules.py` — how many layer-0 neurons each decision rule removes: data
  radius vs per-feature box (IBP) vs directional support `min/max of ŵ·x`, + κ margin.
- `subset_test.py` — why "just the high-norm points" fails: the argmax of `ŵ·x` is
  direction-specific.
- `calib_test.py` / `calib_test2.py` — LLM-style small calibration sets: support
  (min/max) does not concentrate; the moment rule `E[z] + c·std(z) < 0` is safe from
  ~100 points.
- `pca_test.py` / `pca_deep_test.py` — (μ, Σ) is a sufficient statistic for the moment
  rule; truncated PCA + PPCA residual floor reproduces exact decisions at every layer.
- `repair_test.py` — removing always-on neurons: joint regression repair
  (`h_R ≈ Cᵀh_S + c₀`, folded into the next layer's weights and bias) vs naive /
  bias-fold / pairwise merge. ~190× lower val error than pairwise; predicted residual
  matches held-out MSE within ~16%.
- `cnn_sigmafold_test.py` — the full pipeline on the Conv→BN→ReLU CNN (trains its own
  checkpoint into `outputs/cnn_sigmafold/`). Validates: BN folding into (w_eff, b_eff)
  is exact (predicted E[z] matches empirical ≤1e-3); σ-margin from im2col patch moments
  makes no unsafe calls; the channel fold works through next-conv kernels + BN
  running_mean and (losslessly) through the GAP head. Key finding: **BatchNorm
  suppresses saturation** — no filter is even 1%-dead or 99%-on (t ∈ ±1.4), so on BN
  networks the saturation stage finds nothing and the redundancy stage (predicted-cost
  ranking + fold) carries the method: removing the cheapest 25% of filters per block,
  the fold beats naive removal by 37×/3.8× in val loss (blocks 0/1) and is exactly
  lossless at the last block.
- `cnn_nobn_sigmafold_test.py` — the same architecture WITHOUT BatchNorm, confirming
  the BN finding: saturation reappears in deep blocks (block 2: 37/128 filters strictly
  dead, 29%; weight decay collapses their norms). Dead-deletes are exactly free;
  redundancy-fold of 25% per block is lossless (blocks 1/2) or near-lossless
  (block 0: −0.25 acc pts) while naive removal collapses the model. Detection nuance:
  weight-collapsed dead filters (w≈0, b≈0) have t ≈ 0/0 rather than t « 0 — the
  σ-margin misses them, but the predicted-cost criterion (E[h²]·‖A‖² ≈ 0) catches
  them; the method should pair the margin test with this zero-energy guard.

## Reading the results

The hypothesis is supported where AUROC and prec@k are near 1 across seeds and
`med|B| saturated ≫ mixed` in `report.txt`. Caveats to keep in mind:

- |beta| is distance from the **origin**, but the data cloud is not centered at the
  origin (MNIST-normalized inputs, and layer 1+ inputs are nonnegative post-ReLU).
  A neuron can be saturated with small |beta| if the data sits far from the origin in
  its direction `w/||w||`. If AUROC is good but not perfect, compare against the refined
  score `|beta + mean(x)·w/||w|||` (distance from the data *mean*) — `neurons.csv` plus
  the orientation from `compute_hyperplane_params` is enough to compute it.
- Strict eps=0 saturation depends on the finite training sample; re-run with `--eps 0.01`
  to check the conclusion is not knife-edge.
- For deeper layers the input distribution is itself learned, so the data-scale caveat
  grows with depth — check whether AUROC decays across layers.
