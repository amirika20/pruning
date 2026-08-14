# pruning

A small lab for studying **structured neural-network pruning**: train a model,
apply one or more pruning methods to its hidden layers, fine-tune, and measure
the accuracy/efficiency impact — across datasets, architectures, and seeds.

## Running an experiment

Experiments are defined by YAML configs and run with:

```bash
python scripts/run_experiment.py --config configs/experiments/mnist/mlp/mnist_flat_mlp.yaml
python scripts/run_experiment.py --config configs/experiments/mnist        # every yaml under a directory
```

Each run writes a fresh timestamped folder grouped by dataset:

```
outputs/<dataset>/<name>_<YYYYMMDD_HHMMSS>/
    config.yaml               exact config that produced the run
    metadata.json             git commit, timestamp, torch/device info
    run.log                   full log
    seeds/seed_<s>/           per-seed results.json + pruning_summary.png
    aggregated/results.json   mean ± std across seeds
    plots/curves.png          aggregate loss/accuracy curves
```

## Config schema

```yaml
name: mnist_flat_mlp
notes: "free-form description"

data:
  kind: mnist                 # registered in src/data/ (mnist, fashion_mnist, sine, shape2d, modular_add)
  params: {flatten: true, n_samples: 2000, train_ratio: 0.8}

model:
  kind: mlp                   # registered in src/models/ (mlp, resmlp, cnn, rescnn, transformer)
  params: {hidden_sizes: [512, 256, 128]}

training:  {optimizer: sgd, epochs: 500, lr: 1.0e-3, batch_size: 64, weight_decay: 1.0e-4, log_every: 10}

pruning:
  methods:                    # applied in order to every prunable layer;
    - kind: silent            # later methods never re-select earlier picks
    - kind: redundant
      params: {orientation_threshold: 0.1, magnitude_threshold: 0.1}

finetune:  {optimizer: adam, epochs: 500, lr: 1.0e-4, batch_size: 64, weight_decay: 1.0e-4, log_every: 10}

seeds: [0, 1, 2]              # one full train→prune→finetune run per seed, results aggregated
```

## Repo layout

```
configs/experiments/<dataset>/<arch>/*.yaml   experiment definitions
scripts/run_experiment.py                     entry point
src/
  config.py            YAML → ExperimentConfig dataclasses
  data/                dataset registry + builders (DatasetBundle)
  models/              model registry + PrunableModel protocol
  pruning/             pruning-method registry, selection methods, surgery loop
  training/trainer.py  train loop / optimizer builder
  analysis/            efficiency metrics (params/FLOPs/latency) + plots
  experiments/         runner (per-seed pipeline) + cross-seed aggregation
arxiv/                 archived results from the pre-restructure layout
```

## Adding a new pruning method

Pruning methods only *select* which neurons/filters to remove; the structural
surgery is each model's own responsibility, so a new method automatically
works on every architecture.

1. Create `src/pruning/methods/my_method.py`:

   ```python
   from src.models.registry import PrunableModel
   from src.pruning.registry import PruneContext, PruningMethod, register_pruning_method

   @register_pruning_method("my_method")
   class MyMethod(PruningMethod):
       def __init__(self, some_threshold: float = 0.5):   # kwargs come from YAML params
           self.some_threshold = some_threshold

       def select(self, model: PrunableModel, layer_idx: int, ctx: PruneContext) -> list[int]:
           layer = model.prunable_layer(layer_idx)         # the Linear/Conv2d to score
           # ctx.train_inputs, ctx.bundle, ctx.already_selected are available
           return [...]                                    # indices to remove
   ```

2. Import it in `src/pruning/methods/__init__.py`.
3. Reference it from any config:

   ```yaml
   pruning:
     methods:
       - kind: my_method
         params: {some_threshold: 0.3}
   ```

Methods that go beyond plain removal can return a `PruneDecision` instead of
a list — its `merges` field holds ordered `MergeOp(removed, survivor, scale)`
entries, applied via the model's `merge_outgoing` (outgoing-weight transfer)
before the neurons are deleted.

Built-in methods:

- **silent** — removes units that never produce a positive pre-ReLU activation
  on any training input.
- **redundant** — removes units whose hyperplanes nearly coincide with an
  earlier unit's (parallel normals within `orientation_threshold`, offsets
  within `magnitude_threshold`).
- **data_free_merge** — Srinivas & Babu, *Data-free Parameter Pruning for Deep
  Neural Networks* (BMVC 2015). Weight-sets are unit-normalized (ReLU
  homogeneity), pairs are ranked by the saliency
  `s_ij = ⟨a_j²⟩ · ε_ij²` (angular + relative-bias similarity, weighted by the
  removed neuron's outgoing magnitude), and the minimum-saliency pair is
  greedily merged — the removed neuron's outgoing weights are folded into its
  survivor's, so the network's function barely changes even before
  fine-tuning. Needs no training data. Budget via `prune_fraction` /
  `n_remove`, or the paper's data-free histogram-mode cutoff by default
  (optionally scaled by `cutoff_fraction`). Fully-connected style layers only
  (`mlp`, `resmlp`, `transformer`).
- **osscar** — Meng et al., *OSSCAR: One-Shot Structured Pruning in Vision and
  Language Models with Combinatorial Optimization* (ICML 2024), ported from
  the official code (verified bit-exact on the dense-layer case). Minimizes
  the layer-wise reconstruction loss `‖X·W_dense − X·W‖²` on calibration
  activations `X` over which neurons to keep *and* the surviving consumer
  weights: greedy OBS-style elimination with closed-form inverse downdates
  (`fastprune`), an optional swap local search, then an exact least-squares
  re-solve on the final support (returned via `PruneDecision.new_outgoing` /
  the model's `set_outgoing_weights`). One-shot by design — pair with
  `finetune: {epochs: 0}`. Budgeted only: set `prune_fraction` or `n_remove`;
  key params `lambda2` (ridge damping, default 1e-2), `update_iter`,
  `local_search`/`local_iter`/`local_swap`. Fully-connected style layers only
  (`mlp`, `resmlp`, `transformer`).

## Adding a new dataset or model

- **Dataset**: add a builder in `src/data/` returning a `DatasetBundle`
  (declares its own `input_shape`/`output_dim`/`task`), decorate it with
  `@register_dataset("name")`, and import the module in `src/data/__init__.py`.
- **Model**: add a `PrunableModel` subclass in `src/models/` implementing
  `n_prunable_layers` / `prunable_layer` / `prunable_bn` (if BatchNorm-paired)
  / `prune_layer`, register its builder with `@register_model("name")`, and
  import the module in `src/models/__init__.py`. Nothing in `src/pruning`
  needs to change.
