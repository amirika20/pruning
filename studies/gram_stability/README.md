# Gram-stability study

**Question.** Iteratively merge the most-similar neuron pairs of a layer, all the
way down to a single unit, and watch the Gram matrix `BᵀB` of the layer's
neuron representation at every step: **which properties of the Gram are stable
under merging, and do the ones that finally move actually predict the damage?**
If a Gram property is both stable-while-safe and moving-when-harmful, it is a
data-free stopping criterion for merge-based pruning.

## Method under test (phase 1, settled in discussion)

Per neuron of a hidden Linear layer: orientation `u = w/‖w‖` (sign from `w` —
the anchor `β = ρu` is sign-blind, so `u, ρ` are the primitives), signed offset
`ρ = −b/‖w‖`, gain `α = ‖w‖`, outgoing column `c`, merge weight `a = α‖c‖`.

* **Merge rule** = covector addition `S_u += a·u, S_ρ += a·ρ, S_c += α·c`,
  realized as `ū = S_u/‖S_u‖, ρ̄ = S_ρ/‖S_u‖`, outgoing column `S_c`
  (sum-rule fan-out surgery). Associative → the trajectory is a dendrogram.
* **Pair selection** = Ward linkage in box-centered covector space
  `q̃ = [R₀u ; uᵀx₀ − ρ]`, cost `(A_kA_l/(A_k+A_l))‖q̃_k − q̃_l‖²`, with the
  layer-input box `(x₀, R₀)` propagated data-free by IBP.
* **Certificate** per cluster: `Σᵢ aᵢ(R₀‖uᵢ−ū‖ + |(uᵢ−ū)ᵀx₀ − (ρᵢ−ρ̄)|)`.

`merge.py` self-tests (run `python studies/gram_stability/merge.py`): singleton
realization is exact; identical covectors merge exactly for **arbitrary**
outgoing vectors; merging is associative; the first moment is conserved.

## What is tracked (steps.csv, one row per merge step)

Five B-matrix definitions, rows = the layer's current (realized) units:

| def | rows | why |
|---|---|---|
| `anchor` | `ρ̄ū` `[K,d]` | the original hypothesis: anchors of activation boundaries |
| `cov` | `[ū; ρ̄]` `[K,d+1]` | unweighted covectors (pure geometry, no gains) |
| `wcov` | `√A·[ū; ρ̄]` | Gram = gain-weighted 2nd moment `M = ΣA ppᵀ` |
| `xi` | `A·[ū; ρ̄]` | covector-sum rows (their column sum = conserved 1st moment) |
| `wq` | `√A·q̃` | box-centered weighted — theory-native: Ward cost = its trace loss |

Per definition: `trace, opnorm, fro, stable_rank, eff_rank, eig1..eig5`, and
`aff3` = top-3 eigenspace affinity vs the unmerged layer ("general positioning
of the betas in space"). Zero-norm (`‖w‖≈0`) units are frozen out of merging
and excluded from the Gram (they have no hyperplane).

Alongside, per step: per-step + cumulative Ward cost, certified error bound,
overshoot `A_C/‖g_C‖`, first-moment conservation (`m1_raw_*` exact,
`m1_real_*` measures the 2nd-order deviation), **actual** layer output error
(relative L2 of next-layer pre-activations on the val set), and full-model val
loss/accuracy.

## Predictions (falsifiable)

1. `m1_raw` is conserved to machine precision (asserted at runtime).
2. `wq_trace` decays by ≈ the accumulated Ward cost (parallel-axis theorem;
   `ward_trace_ratio ≈ 1` in the report), so the *weighted, centered* Gram is
   the object whose perturbation the merge provably controls.
3. Weighted definitions (`wq`, `wcov`) are stable while error is small and
   drift when error grows → high `rho_err`. The unweighted `anchor` Gram is
   dominated by saturated (large `|ρ|`) units — the least functionally
   relevant — so it should be stable-but-blind or noisy (low `rho_err`).
4. The certified bound rises monotonically with actual error
   (`bound_vs_err_spearman ≈ 1`).

## Running

```bash
/home/amirabbas-kazeminia/Projects/.ml/bin/python studies/gram_stability/run_study.py \
    --config studies/gram_stability/configs            # all four cases

# quick smoke run
/home/amirabbas-kazeminia/Projects/.ml/bin/python studies/gram_stability/run_study.py \
    --config studies/gram_stability/configs/sine_mlp.yaml --epochs 40 --stride 4
```

Cases mirror the beta-saturation study: `sine_mlp` (1-D regression, [64,64]),
`shape2d_mlp` (2-D classification), `mnist_mlp` / `fashion_mnist_mlp`
([256,128,64]). 3 seeds each; each hidden layer is merged independently from
the same trained checkpoint, down to one unit. Outputs land in
`studies/gram_stability/outputs/<name>_<timestamp>/` (gitignored): `steps.csv`,
`summary.json`, `report.txt`, `plots/`, `models/`.

## Reading the results

`report.txt` ranks every property by `frac@5%` (how much of the layer merges
before the property drifts 5% — stability) and `rho_err` (Spearman of its
drift with actual error — predictiveness). The useful invariants sit in the
**stable AND predictive** corner; stable-but-blind properties (e.g. a Gram
dominated by saturated neurons) certify nothing. `plots/drift_vs_error.png` is
the same question drawn: a good property's scatter hugs a monotone curve.
