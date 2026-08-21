# Shared body for every scripts/slurm_<class>.sh. Sourced, not executed.
#
# ONE TASK RUNS MANY CELLS. A benchmark cell (one arm x one model, swept over
# widths) can take seconds on an MLP, so one SLURM task per cell would spend
# more time queueing and importing torch than working. Instead each array task
# claims a STRIDED share of the manifest:
#
#     task i of n  ->  manifest lines i, i+n, i+2n, ...
#
# so `--array=1-16` over a 184-line manifest gives 16 tasks of ~12 cells each,
# matching a 16-GPU allocation with no further arithmetic. A stride rather than
# a contiguous block because the manifest is ordered by model, so contiguous
# chunks would pile all the expensive models into one task.
#
# A cell that fails does NOT abort its task -- the remaining cells still run and
# the task exits non-zero at the end with a list, so one bad arm costs one cell
# rather than a twelfth of the sweep.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

TARGET="${1:?usage: sbatch scripts/slurm_<class>.sh <manifest.txt|config.yaml>}"

# ── scratch: caches and results both live off /n/netscratch ──────────────────
# Home directories are small and often quota'd; model checkpoints (13b OPT is
# ~52GB) and per-cell outputs go to lab scratch. Override PRUNING_SCRATCH to
# relocate everything at once.
export PRUNING_SCRATCH="${PRUNING_SCRATCH:-/n/netscratch/pehlevan_lab/Lab/akazeminia/pruning}"
export TORCH_HOME="$PRUNING_SCRATCH/cache/torch"          # torchvision + torch.hub
export HF_HOME="$PRUNING_SCRATCH/cache/huggingface"       # OPT weights + tokenizers
export HF_DATASETS_CACHE="$PRUNING_SCRATCH/cache/huggingface/datasets"
RESULTS_ROOT="${RESULTS_ROOT:-$PRUNING_SCRATCH/results}"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$RESULTS_ROOT"

STAMP="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
mv -f "logs/slurm_${SLURM_JOB_ID}.log" \
      "logs/$(basename "${TARGET%.*}")_${STAMP}.log" 2>/dev/null || true

module load python
module load cuda/12.9.1-fasrc01
mamba activate ML

echo "host:     $(hostname)"
echo "python:   $(which python) ($(python --version 2>&1))"
echo "scratch:  $PRUNING_SCRATCH"
echo "results:  $RESULTS_ROOT"
echo "caches:   TORCH_HOME=$TORCH_HOME  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail fast with a clear message rather than a traceback from inside the runner:
# a SLURM retry costs a queue wait, not just a rerun.
python -c "import torch, src.pruning.methods" \
    || { echo "repo deps missing in this env (need torch + the src package importable from the repo root)" >&2; exit 1; }

# Compute nodes are frequently network-isolated, and every pretrained entry
# downloads weights on first use. Warm the caches from a login node first:
#   python scripts/warm_caches.py
# HF_HUB_OFFLINE makes a cold cache fail loudly here instead of hanging on a
# blocked connection. Set HF_HUB_OFFLINE=0 to allow in-job downloads.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# ── run this task's share ───────────────────────────────────────────────────
# The loop itself lives in scripts/run_manifest.py, so a local run and a cluster
# task execute identical code (and a subprocess per cell means an OOM kill costs
# one cell, not the task). Striding is by --shard i/n.
GRID="${GRID:-16}"
ARGS=(--grid "$GRID" --out "$RESULTS_ROOT")
[[ -n "${SEED:-}" ]] && ARGS+=(--seed "$SEED")

if [[ "$TARGET" == *.yaml ]]; then
    ARGS+=(--config "$TARGET")
else
    ARGS+=(--manifest "$TARGET"
           --shard "${SLURM_ARRAY_TASK_ID:-1}/${SLURM_ARRAY_TASK_COUNT:-1}")
fi

python scripts/run_manifest.py "${ARGS[@]}"
