# What to submit next (written 2026-08-30, from the laptop)

State as of the last rsync of `outputs/benchmark`:

| tier | seed-runs done | missing | notes |
|---|---|---|---|
| small / modular | all | 0 | |
| medium | 350 / 372 | 22 | all `wikitext_opt350m` MASH arms, killed by the wall |
| large (imagenet) | 12 / 68 | 56 | dataset caches warm, cheap — ready to go |
| large (wikitext) | 0 / 65 | 65 | **blocked on planner cost, see §3** |

---

## 1. Medium resume — 22 seed-runs

Job `42560713` lost 10 of 16 array tasks to `DUE TO TIME LIMIT`. Seeds run
sequentially inside a cell, so the losses skew late (seed 0: 3, seed 1: 8,
seed 2: 11). `run_sweep.py` has no per-seed resume — rerunning a cell redoes
every seed — but `_slurm_body.sh` honours a `SEED` env var, so three per-seed
manifests run exactly the 22 missing seeds and redo nothing:

```bash
sbatch --export=ALL,SEED=0 --array=1-3  scripts/slurm_medium.sh configs/benchmark/manifest_medium_resume_seed0.txt
sbatch --export=ALL,SEED=1 --array=1-8  scripts/slurm_medium.sh configs/benchmark/manifest_medium_resume_seed1.txt
sbatch --export=ALL,SEED=2 --array=1-11 scripts/slurm_medium.sh configs/benchmark/manifest_medium_resume_seed2.txt
```

One cell per task (`--array` size == manifest length, and the shard is strided).
The default 8h wall is enough: the worst single opt350m seed measured 15086s
(4.2h). It was three seeds in one task at ~12.6h that overran, not any one seed.

Cost: ~60–80 GPU-h across 22 tasks.

Add `--exclude=$(sort -u logs/bad_nodes.txt | paste -sd,)` if that file is
non-empty.

## 2. Large / ImageNet — 56 seed-runs, ready

The `manifest_large_imagenet` array `42127863` was `scancel`ed 2 minutes in on
08-26 (all 16 tasks *and* the parent got SIGTERM at 20:25:12 — not a time
limit), so this tier never really started.

Nothing blocks a resubmission:

- The uint8 tensor cache in `$PRUNING_SCRATCH/cache/datasets` is warm for all
  three seeds — the smoke run paid it (5870s for seed 0, ~1400s each for seeds
  1 and 2). All 36 configs share identical `data.params`, and the cache key is
  `(root, n_samples, n_test, image_size, seed)`, so every cell hits it.
- Cost anchor: `imagenet_vit_b16__mash_merge_kernel_delta_f` seed 0 = 626s plan
  + 1091s solve. ViT-B/16 is 12 x 3072 = 36864 prunable units, the same shape as
  opt125m, whose planning for that arm was 757s — the two agree, so the cost
  model in §3 transfers across architectures. The conv models are far smaller
  and their cheap arms ran in 270–370s per seed.

```bash
sbatch --array=1-36 scripts/slurm_large.sh configs/benchmark/manifest_large_imagenet.txt
```

Estimate ~10–15 GPU-h total. `slurm_large.sh` gives 16h / 128G, which is ample.

**Before spending it, decide on the width grid.** Every ImageNet cell finished so
far reports `cap005 = cap01 = 0.0`, and all but one report `cap02 = 0.0`. The
grid is `np.linspace(0.95/16, 0.95, 16)`, so the finest point already removes
5.9% of units — and on the real 1000-class task even that exceeds a 5% drop. As
manifested, the headline ImageNet table will be a column of zeros and the arms
will be separable only by AUC. (That is E5's prediction arriving, not a bug, but
it is not a readable result.) `run_sweep.py` takes `--widths` for an explicit
grid; `run_manifest.py` does not forward it. Either plumb it through and sweep
something like 0.01…0.10, or accept AUC-only comparisons on this tier.

## 3. Large / WikiText — do NOT submit as manifested

Cost was fitted from the medium logs, which have every arm at both opt125m and
opt350m. For MASH planning specifically the fit is tight: across 32 arms,
`median(350m) / median(125m) = 5.98` (range 4.7–8.7). With
opt125m at 12 layers x 3072 and opt350m at 24 x 4096, that is 2x the layers and
3.0x per layer at 1.33x the width.

That factor is not uniform across arms, so each arm gets its own exponent from
its own 125m→350m ratio `r`, via `r = 2 * (4096/3072)^p`, and is then projected
by `(width ratio)^p * (layer ratio)` onto each size. Wall clock **per seed**,
anchored on the measured opt350m `total_seconds`:

| arm | 350m | p | 1.3b | 2.7b | 6.7b |
|---|---|---|---|---|---|
| `random` | 18s | 1.95 | 0.0h | 0.0h | 0.1h |
| `magnitude_mass` | 32s | 2.05 | 0.0h | 0.1h | 0.2h |
| `saturated_dead_margin` | 2037s | 1.82 | 2.0h | 4.0h | 9.3h |
| `saturated_dead_empirical` | 1660s | 2.24 | 2.2h | 4.8h | 13.7h |
| `osscar` | 1951s | 2.63 | 3.4h | 8.1h | **27.8h** |
| `mash_merge_sum_cylinder` | 939s | 3.26 | 2.5h | 6.9h | **31.9h** |
| `leo_pp` | 971s | 3.39 | 2.8h | 8.1h | **39.7h** |
| `mash_certified` | 1028s | 3.91 | 4.3h | 13.7h | **85.6h** |
| `mash_merge_kernel_delta_f` | 8690s | 2.48 | 13.5h | **31.3h** | **100.3h** |
| `mash_merge_empirical_delta_f` | 8787s | 2.52 | 14.0h | **32.6h** | **106.3h** |
| `data_free_merge` | 8190s | 3.33 | 22.9h | **64.3h** | **308.3h** |

`mash_medoid_empirical_delta_f` has no opt350m datapoint yet (it is one of the
22 in §1); treat it as ~`mash_merge_empirical_delta_f`.

Tier totals: **opt1.3b ~203 GPU-h** (3 seeds), **opt2.7b ~174 GPU-h**,
**opt6.7b ~723 GPU-h** — about 1100 GPU-h for the 65 seed-runs, over half of it
opt6.7b, and most of that in `data_free_merge` and the two `delta_f` arms.

Bolded cells exceed the 24h `slurm_xlarge.sh` wall as a single seed and cannot
be run as manifested. Note it is not only MASH: the non-MASH arms have
`plan_seconds = 0`, but their *solve* scales just as steeply — `data_free_merge`
is the single most expensive arm in the tier.

Feasibility, per cell, against a 24h wall:

- **opt1.3b** — everything fits, but only if split by seed (`data_free_merge` is
  23h for one seed, ~69h for the cell). Use the §1 `SEED=` trick throughout.
- **opt2.7b** — fits except `data_free_merge` and the two `delta_f` arms.
- **opt6.7b** — only `random`, `magnitude_mass` and the two `saturated_*` arms
  fit. Everything interesting is out of reach.

This is a two-point extrapolation, and the exponent is the shaky part: 125m→350m
changes layers, FFN width and `d_model` together, so I cannot tell which drives
the per-layer factor, and `p` swings from 1.8 to 3.9 across arms. **opt1.3b is
the decisive probe** — same 24 layers as 350m, exactly 2x the width and
`d_model` — so one cell measures the width exponent with no layer confound:

```bash
# warm the weights from a login node first (compute nodes are HF_HUB_OFFLINE=1)
python scripts/warm_caches.py --entries wikitext_opt1.3b wikitext_opt2.7b wikitext_opt6.7b

sbatch --export=ALL,SEED=0 scripts/slurm_large.sh configs/benchmark/generated/wikitext_opt1.3b/mash_merge_sum_cylinder.yaml
sbatch --export=ALL,SEED=0 scripts/slurm_large.sh configs/benchmark/generated/wikitext_opt1.3b/mash_merge_empirical_delta_f.yaml
```

Predicted 2.5h and 14.0h. If they land there the table holds, and the bolded
cells need one of:

1. **Trim the arm set at the big sizes** — carry the full set to 1.3b, drop
   `data_free_merge` and the `delta_f` scores at 2.7b, and take only the cheap
   arms plus one MASH representative to 6.7b. Costs the headline
   "MASH scales to 6.7b" claim.
2. **Split by seed** — necessary for opt1.3b, no help at 2.7b/6.7b, which are
   already single-seed by config.
3. **Make the solve cheaper.** A per-layer cost growing as `width^2.5–3.9` is
   the real finding here; at 16384 units a layer it *is* the experiment. Worth
   an afternoon of profiling before buying 700 GPU-hours around it — a 3x
   speedup is the difference between opt6.7b being in the paper and not.

Meanwhile the cheap end of the tier is unblocked and worth submitting now:
`random`, `magnitude_mass` and both `saturated_*` arms fit comfortably at every
size (~30 GPU-h for all 12 cells) and establish the dense baselines the rest of
the tier is scored against.

---

## 4. Scale tier (decided 2026-09-08) — what the big models run

`arms.yaml` now has a `scale:` set and `suite.yaml` puts every large/xlarge
entry on it (`arms_tier: scale`), with an explicit `sweep_fractions` grid per
entry. The generator writes one manifest per big model,
`configs/benchmark/manifest_scale_<entry>.txt`, and
`submit_benchmark.sh --tier scale` deliberately refuses: these cells are
budgeted one model and one seed at a time.

**Arms (9 on FC, 5 on BN conv):** `random`, `magnitude_mass`, `osscar`, and
MASH under the empirical repair only, crossing dictionary {merge, medoid} x
score {cylinder = pre-activation Ward, delta_f = post-activation Ward} plus
`mash_sampled_{merge,medoid}_empirical_delta_f` (delta_f from sample Grams
instead of the Gaussian closed form; FC only, conv already scores that way).
On ResNet18/50 and MobileNetV2 the merge dictionary is refused by BN, so only
the medoid arms run there; ViT-B/16 carries the merge-vs-medoid comparison on
ImageNet. Dropped at scale: `mash_certified`, `hope`, `leo_pp`,
`data_free_merge`.

**Grids.** ImageNet: 0.01 0.02 0.03 0.05 0.07 0.10 0.15 0.20 0.30 0.40 0.50
(the default grid's first point, 5.9%, was already past the tolerance on every
finished cell). OPT 1.3b+: 0.02 0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.65 0.80.
Capacities are first-crossing and grid-independent; AUC compares only within a
grid, so do not put the 1.3b AUC next to the 350m one.

**What every cell stores** (per `<cell>/seed_<s>/`): `curve.csv`, `report.json`,
`removals.json` (removed unit indices PER LAYER at EVERY swept width, dense-layer
numbering, for every arm including OSSCAR and magnitude) and, for MASH arms,
`dendrogram.json` (the full merge sequence per layer with cost, certificate and
unit masses, so any width can be cut offline). The OSSCAR-vs-MASH overlap is
then `src.analysis.pruning_detail.overlap_from_removals({"osscar": dir,
"mash_...": dir})` -- one row per (fraction, layer) with `excess` over chance.
Verified on a synthetic MLP: the medoid removed set is reconstructed exactly
from `dendrogram.json`, and merge clusters agree. None of the 914 finished seed
dirs has `removals.json` (the recording landed after they ran), so the medium
tier's OSSCAR/MASH overlap needs a rerun of just those two arms if wanted.

**SLURM.** All four job scripts now request `--cpus-per-task=2` and the body
pins OMP/MKL to that count: Job Defense Shield measured every task of the 4-
and 8-core runs at exactly one core busy. Check `jobstats` on the first scale
job before asking for more.

### Submission order

```bash
# 0. weights, from a login node (compute nodes are HF_HUB_OFFLINE=1)
python scripts/warm_caches.py --entries wikitext_opt1.3b wikitext_opt2.7b wikitext_opt6.7b wikitext_opt13b

# 1. ImageNet -- cheap, everything fits; one task per cell
for m in imagenet_resnet18 imagenet_mobilenetv2 imagenet_resnet50; do
  sbatch --array=1-5 scripts/slurm_large.sh configs/benchmark/manifest_scale_$m.txt
done
sbatch --array=1-9 scripts/slurm_large.sh configs/benchmark/manifest_scale_imagenet_vit_b16.txt

# 2. the opt1.3b PROBE -- one delta_f seed; predicted ~14h on the old 16-pt
#    grid, so ~10h on the 10-pt one. If it lands there, §3's cost table holds.
sbatch --export=ALL,SEED=0 scripts/slurm_large.sh configs/benchmark/generated/wikitext_opt1.3b/mash_merge_empirical_delta_f.yaml

# 3. opt1.3b, all 9 arms, split by seed (three seeds in one task overran at 350m)
for s in 0 1 2; do
  sbatch --export=ALL,SEED=$s --array=1-9 scripts/slurm_large.sh configs/benchmark/manifest_scale_wikitext_opt1.3b.txt
done

# 4. opt2.7b, single seed. Predicted: cheap arms minutes, osscar ~5h, cylinder
#    arms ~5h, the four delta_f arms ~26h EACH on the 10-pt grid -- over the
#    24h wall. Submit the five that fit now; the delta_f arms need a longer
#    partition limit or a 6-point grid (FRACTIONS="0.02 0.05 0.1 0.2 0.3 0.5").
sbatch --array=1-9 --time=24:00:00 scripts/slurm_large.sh configs/benchmark/manifest_scale_wikitext_opt2.7b.txt

# 5. opt6.7b: only random, magnitude_mass and osscar fit (osscar ~17h on the
#    10-pt grid). Every MASH arm is projected at 60-100h per seed; do not queue them.
grep -E 'random|magnitude_mass|osscar' configs/benchmark/manifest_scale_wikitext_opt6.7b.txt > configs/benchmark/manifest_scale_wikitext_opt6.7b_cheap.txt
sbatch --array=1-3 --time=24:00:00 scripts/slurm_large.sh configs/benchmark/manifest_scale_wikitext_opt6.7b_cheap.txt

# 6. opt13b: random and magnitude_mass only, on an 80GB card
grep -E 'random|magnitude_mass' configs/benchmark/manifest_scale_wikitext_opt13b.txt > configs/benchmark/manifest_scale_wikitext_opt13b_cheap.txt
sbatch --gres=gpu:h100:1 --array=1-2 scripts/slurm_xlarge.sh configs/benchmark/manifest_scale_wikitext_opt13b_cheap.txt
```

Add `--exclude=$(sort -u logs/bad_nodes.txt | paste -sd,)` to each if that file
is non-empty. Budget: ImageNet ~15 GPU-h, opt1.3b ~90 GPU-h over 27 tasks,
opt2.7b ~20 GPU-h for the five that fit, opt6.7b ~20 GPU-h, opt13b ~1 GPU-h.

---

## 5. Repair tier (decided 2026-09-09) — selection vs repair, and the ridge

Two experiments, one arm set (`arms.yaml` `repair:`), generated with
`--tier repair` for the nine models below. The generator writes
`configs/benchmark/manifest_repair_<entry>.txt` per model and does NOT touch
`manifest.txt` or the class manifests.

**Experiment 1 — with vs without repair.** random, magnitude, OSSCAR, MASH
(delta_f Ward score, both the Gaussian-moment and the sample-Gram measure).
"Without" is the arm's own removed set deleted outright: `random`,
`magnitude_mass`, `osscar_norepair` (new: OSSCAR's selection, no consumer
rewrite), `mash_medoid_none_delta_f` / `mash_sampled_medoid_none_delta_f`
(new `repair: none`: medoid survivors keep their original columns). "With" is
`*_empirical`, `osscar`, `mash_{medoid,merge}_empirical_delta_f` and the
`mash_sampled_*` versions.

**Experiment 2 — the ridge.** The empirical repair at the 1e-8 guard vs
OSSCAR's 1e-2, relative to the mean diagonal of the Gram, for random,
magnitude and MASH: `random_ridge`, `magnitude_mass_ridge`,
`mash_ridge_{medoid,merge}_empirical_delta_f`, `mash_ridge_sampled_*`. MASH's
ridge is centred on the unrepaired columns (lam -> inf returns repair=none /
the sum rule), which is the same limit OSSCAR's damped target and the
baselines' transfer have; a zero-centred ridge would have driven the columns
to zero.

Arms per model: 18 on the FC entries (ViT, the OPTs), 11 on the BN ResNets
and MobileNetV2 (no merge, no sampled measure -- conv already scores
empirically). 127 cells, 323 seed-runs. `random`, `magnitude_mass`, `osscar`
and `mash_{medoid,merge}_empirical_delta_f` overlap the scale/ablation tiers:
finished seed dirs are reused by the plots, so skip those lines when a cell is
already on disk (`ls outputs/benchmark/<cell>/seed_*`).

**Cost, and how it is kept down.** Three things changed after the cluster's
efficiency report on the OPT-1.3b probe (5% GPU utilization, 7% of 128G):

1. **Plan reuse.** A MASH cell's cost is its planning pass (10 h at 1.3b, CPU
   only, GPU idle). Arms that differ only in repair -- none / empirical /
   ridge -- cut the same dendrogram, so `run_sweep` now looks for a sibling
   cell's `dendrogram.json` with an identical `plan_key` (score, measure,
   dictionary, gauge, radius, n_calib) and reuses it; the report records
   `plan_reused_from`. That turns the 10 MASH plans at 1.3b into 4. It only
   works if the planning arms FINISH FIRST, hence the two stages below.
2. **PARALLEL cells per GPU.** `PARALLEL=k` in the job body runs k cells of a
   task's share side by side on one GPU (each its own process, finer strided
   sub-shards), so the card is busy while the others plan or load. Pair it
   with `--cpus-per-task=2k`. k=3 fits any ImageNet model or OPT-1.3b on a
   40GB card.
3. **Memory** requests are right-sized: medium 32G, large 48G (6.7b: pass
   `--mem=96G`).

Per seed, after these: CIFAR ResNets and ImageNet convs minutes per arm;
OPT-125m ~15 min per MASH plan; OPT-350m ~1.1 h per MASH plan, ~0.5 h OSSCAR;
OPT-1.3b ~10 h per MASH plan (4 plans), ~3.5 h OSSCAR (2 arms). OPT-1.3b is
~55 GPU-h per seed with reuse and PARALLEL=3 against ~110 without. Everything
else together is under 40 GPU-h.

```bash
# CIFAR + OPT-125m/350m: medium class, 3 cells per GPU, ~4 tasks per model
for m in cifar10_resnet20 cifar10_resnet56 wikitext_opt125m wikitext_opt350m; do
  n=$(wc -l < configs/benchmark/manifest_repair_$m.txt)
  sbatch --export=ALL,PARALLEL=3 --cpus-per-task=6 --array=1-$(( (n + 2) / 3 )) \
         scripts/slurm_medium.sh configs/benchmark/manifest_repair_$m.txt
done
# ImageNet: large class, 3 cells per GPU
for m in imagenet_resnet18 imagenet_resnet50 imagenet_mobilenetv2 imagenet_vit_b16; do
  n=$(wc -l < configs/benchmark/manifest_repair_$m.txt)
  sbatch --export=ALL,PARALLEL=3 --cpus-per-task=6 --array=1-$(( (n + 2) / 3 )) \
         scripts/slurm_large.sh configs/benchmark/manifest_repair_$m.txt
done
# OPT-1.3b, seed 0. Stage 1 = the 4 planning arms + random/magnitude/osscar
# (7 cells, 3 per GPU, 3 tasks); stage 2 = the 11 repair variants, 6 of which
# reuse stage 1's plans, held until stage 1 has finished. Seeds 1-2: repeat
# with SEED=1,2.
j=$(sbatch --parsable --export=ALL,SEED=0,PARALLEL=3 --cpus-per-task=6 --array=1-3 \
      scripts/slurm_large.sh configs/benchmark/manifest_repair_wikitext_opt1.3b_stage1.txt)
sbatch --dependency=afterany:$j --export=ALL,SEED=0,PARALLEL=3 --cpus-per-task=6 --array=1-4 \
      scripts/slurm_large.sh configs/benchmark/manifest_repair_wikitext_opt1.3b_stage2.txt
```

Stage 1 of OPT-1.3b is a superset of §4 step 4's MASH delta_f arms (the two
cylinder arms are the only scale-tier cells it lacks), so submit §4 step 4 as
stage 1's sibling or skip it. Add `--exclude=$(sort -u logs/bad_nodes.txt |
paste -sd,)` if that file is non-empty. Figures:
`python studies/paper/figs/make_repair_figures.py` writes `fig_repair_effect`,
`fig_repair_ridge` and `fig_repair_measure` from whatever cells exist. The
best configuration from these then goes to OPT-2.7b/6.7b as §4 describes.
