# Shared body for every scripts/slurm_<class>.sh. Sourced, not executed.
#
# HOW TO TEST A CHANGE TO THE SOURCING. SLURM copies the batch script into the
# node's spool directory, so "$0" inside it is /var/slurmd/.../slurm_script and
# anything resolved relative to $0 misses this file. Running `bash
# scripts/slurm_small.sh ...` directly does NOT reproduce that. Copy it first:
#
#   d=$(mktemp -d); cp scripts/slurm_small.sh $d/slurm_script
#   (cd / && SLURM_SUBMIT_DIR=$PWD SLURM_JOB_ID=1 bash $d/slurm_script <manifest>)
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

# The attempt number belongs in the log name. Without it a requeued attempt
# reopens the same path and OVERWRITES its predecessor, so the first attempt's
# diagnosis is lost -- which is why the modular run looked like the GPU preflight
# had failed to requeue: the requeue had happened, the retry landed on the same
# contended node, and attempt 1's log (with RESTART_COUNT=1, so no requeue line)
# had replaced attempt 0's.
STAMP="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
[[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]] && STAMP="${STAMP}_try$((SLURM_RESTART_COUNT + 1))"
# :- defaults throughout: SLURM always sets these, but `set -u` would otherwise
# abort a local run of this script at line 1, defeating the test recipe above.
mv -f "logs/slurm_${SLURM_JOB_ID:-}.log" \
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

# ── GPU preflight ───────────────────────────────────────────────────────────
# A node can be allocated with its GPU unusable (cudaErrorDevicesUnavailable),
# which took out 12 cells of the first small run: torch imports fine, every cell
# fails at the first .to(device), and the task burns its slot failing 12 times.
# Test the GPU with a real allocation and matmul -- cuda.is_available() alone
# returns True on a node whose GPU is held by a dying process -- and requeue the
# task onto a different node instead of grinding through the manifest. Requeued
# at most once (SLURM_RESTART_COUNT), so a cluster-wide fault fails rather than
# looping.
# RETRY, PATIENTLY, THEN STOP. The modular run lost 7 tasks to
# cudaErrorDevicesUnavailable, all on one node that also ran 8 tasks
# successfully -- the GPU was CONTENDED, not broken: another process still held
# it. So waiting is the right response, and 5 minutes of it is far cheaper than a
# trip through the queue.
#
# WHAT DOES NOT WORK: requeueing. `scontrol update ExcNodeList` is REJECTED on a
# running job ("Job is no longer pending execution"), so the earlier version's
# exclusion silently failed and the bare requeue handed the task straight back to
# the same node -- three attempts, same node, three wasted queue slots. A job
# cannot edit its own allocation, so the exclusion has to come from the submit
# line. This records the node and prints that command instead of pretending.
gpu_ok=0
for attempt in $(seq 1 10); do
    if python -c "
import torch, sys
if not torch.cuda.is_available(): sys.exit('no CUDA device visible')
x = torch.randn(256, 256, device='cuda'); (x @ x).sum().item()
torch.cuda.synchronize()
print(f'gpu ok: {torch.cuda.get_device_name(0)}')
"; then gpu_ok=1; break; fi
    echo "GPU probe $attempt/10 failed on $(hostname -s); waiting 30s" >&2
    sleep 30
done

if [[ "$gpu_ok" -ne 1 ]]; then
    BAD="$(hostname -s)"
    echo "GPU UNUSABLE on $BAD after 10 probes over 5 minutes" >&2
    # A deduplicated list the next submission can consume directly.
    touch logs/bad_nodes.txt
    grep -qxF "$BAD" logs/bad_nodes.txt || echo "$BAD" >> logs/bad_nodes.txt
    echo "recorded in logs/bad_nodes.txt -- resubmit excluding it:" >&2
    echo "    sbatch --exclude=$(paste -sd, logs/bad_nodes.txt) <script> $TARGET" >&2
    echo "and report $BAD to FASRC: a GPU held by a stuck process" >&2
    exit 1
fi

# Fail fast, and check what THIS manifest actually needs. A missing optional
# package is the worst cluster failure mode: the array starts, and only the cells
# that need the package fail -- `datasets` absent takes out every OPT cell while
# the CIFAR cells succeed beside them and the run looks half-healthy.
if [[ "$TARGET" == *.yaml ]]; then
    python scripts/check_env.py || exit 1
else
    python scripts/check_env.py --manifest "$TARGET" || exit 1
fi

# Compute nodes are frequently network-isolated, and every pretrained entry
# downloads weights on first use. Warm the caches from a login node first:
#   python scripts/warm_caches.py
# HF_HUB_OFFLINE makes a cold cache fail loudly here instead of hanging on a
# blocked connection. Set HF_HUB_OFFLINE=0 to allow in-job downloads.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
# torch-only: stop transformers probing the TensorFlow/Flax backends, which
# costs seconds of startup per cell and buries the log in absl/oneDNN notices.
export TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_FLAX=1 USE_TF=0 USE_FLAX=0
export TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0

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
