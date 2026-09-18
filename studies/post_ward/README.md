# Post-activation Ward for delta_f

Question (2026-09-17): the sampled `delta_f` score forms its cluster centroid in
pre-activation space (summed covector, pushed through the ReLU) and rescores
pairs against that synthesized neuron after every merge. Would exact Ward on the
post-activation response rows, propagated by Lance--Williams, select better
clusters, and drop the per-step response recompute?

## Files

- `post_ward.py`: `PostWardEngine`, a subclass of the stock `MashEngine` that
  keeps the stock singleton matrix and flips the engine into its existing
  Lance--Williams branch. Self-test: `python studies/post_ward/post_ward.py`
  checks the Lance--Williams matrix against direct recomputation from
  response-space centroids at every step (agrees to 1e-14) and confirms the
  trajectory diverges from stock delta_f after merges.
- `run.py`: ResNet-20 sweep, both arms, repairs `none` and `ridge`, 3 seeds.
  Writes `RESULTS.md`, `results.csv`, `outputs/`.

Nothing in `src/` is modified. The study arm swaps the engine class looked up by
`MASH.plan` inside a context manager.

## Finding (ResNet-20, 3 seeds, 16-point grid, test split)

| repair | stock AUC | post_ward AUC | ΔAUC per seed |
|---|---|---|---|
| none | 0.278 ± 0.006 | 0.294 ± 0.003 | +0.018, +0.026, +0.006 |
| ridge 1e-2 | 0.456 ± 0.006 | 0.454 ± 0.009 | +0.001, +0.005, −0.011 |

Without repair, post-activation Ward wins at every mid width (about +3.5 to
+5.4 points from 18% to 42% removed), consistently across seeds. With the ridge
repair the two are indistinguishable: the repair absorbs the selection
difference, which matches the repair-tier finding that the lead lives in
repair, not selection. The stock rows reproduce the `weight` gauge of
`studies/gauge_ablation` exactly.

Plan time is 0.3 s vs 0.4 s per model here; the speed argument does not bite at
ResNet-20 widths.

## Promoted (2026-09-17)

The knob is `centroid: post` on the stock `mash` / `mash_certified` methods
(src/pruning/methods/mash.py, module docstring CENTROID; default `pre` keeps
every recorded number). In a benchmark cell config:

```yaml
pruning:
  methods:
  - kind: mash
    params: {score: delta_f, dictionary: medoid, repair: empirical, centroid: post}
```

`run.py --arms post_src` runs that configuration through the same harness and
`run.py --check` asserts it reproduces the study engine's six ResNet-20 curves
bit for bit (it does). The engine self-tests in mash.py cover the
Lance--Williams identity for the post centroid, the parameter validation, the
plan-key rule (present only when not `pre`) and the functional (GELU) path.
