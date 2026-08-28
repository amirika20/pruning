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
if ! mkdir -p "$TORCH_HOME" "$HF_HOME" "$RESULTS_ROOT" 2>/dev/null; then
    echo "FATAL: cannot create directories under PRUNING_SCRATCH=$PRUNING_SCRATCH" >&2
    echo "  $(mkdir -p "$RESULTS_ROOT" 2>&1 | head -1)" >&2
    echo "  Nothing can be written, so the manifest is not started. Check the lab" >&2
    echo "  scratch mount from this node (ls -ld $(dirname "$PRUNING_SCRATCH")), or" >&2
    echo "  set PRUNING_SCRATCH to a writable path." >&2
    exit 1
fi

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

# RE-INITIALISE LMOD RATHER THAN TRUSTING THE INHERITED FUNCTION. `module` is a
# shell function lmod defines, and sbatch's default --export=ALL ships the
# SUBMITTING shell's copy to the compute node. Submitting from a VS Code remote
# terminal sent one pointing at /n/sw/helmod-rocky8/..., which the nodes no
# longer have, so every task failed identically with
#   environment: line 17: /n/sw/helmod-rocky8/.../lmod: No such file or directory
# ("environment" is what bash calls the source of an exported function). Drop a
# stale definition and source the node's own init instead.
if [[ -n "${LMOD_CMD:-}" && ! -x "${LMOD_CMD}" ]]; then
    echo "note: inherited LMOD_CMD=$LMOD_CMD is not executable here; re-initialising" >&2
    unset -f module 2>/dev/null || true
    unset LMOD_CMD MODULESHOME LMOD_PKG 2>/dev/null || true
fi
if ! command -v module >/dev/null 2>&1 && [[ -z "${LMOD_CMD:-}" ]]; then
    for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
                /usr/share/lmod/lmod/init/bash /n/sw/helmod-rocky9/apps/lmod/lmod/init/bash; do
        [[ -r "$init" ]] && source "$init" && break
    done
fi

# `module` and `mamba` failing used to be survivable-looking: each printed its
# own error, the script continued, and the first thing to actually die was the
# GPU probe -- with "python: command not found", which the probe then reported as
# GPU UNUSABLE and wrote a perfectly healthy node into bad_nodes.txt. Diagnose
# the environment here instead, where the message can name the real cause.
module load python 2>/dev/null || echo "note: module load python failed" >&2
module load cuda/12.9.1-fasrc01 2>/dev/null || echo "note: module load cuda failed" >&2
if command -v mamba >/dev/null 2>&1; then
    mamba activate ML || echo "note: mamba activate ML failed" >&2
elif command -v conda >/dev/null 2>&1; then
    echo "note: mamba absent, trying conda" >&2
    conda activate ML || echo "note: conda activate ML failed" >&2
else
    echo "note: neither mamba nor conda on PATH" >&2
fi

if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: no python on PATH after environment setup on $(hostname -s)." >&2
    echo "  This is an ENVIRONMENT failure, not a GPU or code failure -- the" >&2
    echo "  module system or the conda env did not load. Common cause: the node" >&2
    echo "  is missing the lmod install the login node has" >&2
    echo "  (/n/sw/helmod-rocky8/apps/lmod/...), so `module` and `mamba` are" >&2
    echo "  both unavailable." >&2
    echo "  Check: srun --pty -p ${SLURM_JOB_PARTITION:-kempner} bash -c 'module load python; which python'" >&2
    echo "  Do NOT add this node to bad_nodes.txt -- its GPU was never tested." >&2
    exit 78
fi

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
    # sort -u at print time: concurrent array tasks append without locking, so the
    # grep guard above loses races and the list can hold a host twice.
    echo "    sbatch --exclude=$(sort -u logs/bad_nodes.txt | paste -sd,) <script> $TARGET" >&2
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
