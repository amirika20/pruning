# Shared body for every scripts/slurm_<class>.sh. Sourced, not executed.
#
# Resolves one manifest line into a config path (array mode) or takes a config
# path directly (single mode), sets up the environment, and runs it.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

TARGET="${1:?usage: sbatch scripts/slurm_<class>.sh <manifest.txt|config.yaml>}"

if [[ "$TARGET" == *.yaml ]]; then
    CONFIG="$TARGET"
else
    : "${SLURM_ARRAY_TASK_ID:?a manifest needs an array: sbatch --array=1-N ...}"
    CONFIG="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TARGET")"
    [[ -n "$CONFIG" ]] || { echo "manifest $TARGET has no line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
fi
[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 1; }

# #SBATCH --output is parsed before the script runs, so it opens a job-id-only
# placeholder; rename it now that CONFIG is known. The open file descriptor
# survives the rename (same inode), so output from before and after lands in
# the one file.
STAMP="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
mv -f "logs/slurm_${SLURM_JOB_ID}.log" \
      "logs/$(basename "${CONFIG%.yaml}")_${STAMP}.log" 2>/dev/null || true

module load python
module load cuda/12.9.1-fasrc01
mamba activate ML

echo "host:    $(hostname)"
echo "config:  $CONFIG"
echo "python:  $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail fast with a clear message rather than a traceback from inside the runner:
# a SLURM retry costs a queue wait, not just a rerun.
python -c "import torch, src.pruning.methods" \
    || { echo "repo deps missing in this env (need torch + the src package importable from the repo root)" >&2; exit 1; }

# Compute nodes are often network-isolated. Every pretrained entry in the suite
# downloads weights on first use, so warm the caches on a login node once:
#   python scripts/warm_caches.py --config configs/benchmark/manifest.txt
# HF_HUB_OFFLINE makes a cold cache fail loudly here instead of hanging on a
# blocked connection.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# One benchmark cell = an accuracy-versus-width sweep for one arm. NOT
# run_experiment.py, which prunes once at a fixed width and so cannot produce a
# capacity number. GRID/SEED are overridable from the submit environment:
#   sbatch --export=ALL,GRID=24 ... scripts/slurm_small.sh <manifest>
python scripts/run_sweep.py --config "$CONFIG" --grid "${GRID:-16}" ${SEED:+--seed "$SEED"}
