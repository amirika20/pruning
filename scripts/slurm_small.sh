#!/bin/bash
#SBATCH --job-name=prune_small
#SBATCH --output=logs/slurm_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akazeminia@g.harvard.edu
#
# Resource class "small" of the benchmark. Submit from the repo root.
#
#   sbatch --array=1-N scripts/slurm_small.sh configs/benchmark/manifest_small.txt
#   sbatch            scripts/slurm_small.sh configs/benchmark/generated/<e>/<a>.yaml
#
# cpus-per-task=2: Job Defense Shield measured every task of the earlier 4-
# and 8-core runs at exactly ONE core busy (CPU-Util = 100%/cores) -- the work
# is GPU-side torch with no DataLoader workers, and the host-side numpy/scipy
# planning is scalar-bound. Two cores = the python main thread plus the CUDA
# driver/allocator threads; _slurm_body.sh pins OMP/MKL to the same count.
# Check with `jobstats <jobid>` before asking for more.
#
# Override anything without editing this file:
#   sbatch --time=02:00:00 --mem=16G --array=1-10 scripts/slurm_small.sh <manifest>
# Locate the shared body. SLURM COPIES this script into the node's spool
# directory before running it, so "$0" is /var/slurmd/.../slurm_script and
# dirname "$0" is the spool dir -- which does not contain _slurm_body.sh. That is
# why sourcing relative to $0 works when the script is run directly but fails
# under sbatch, and why local testing could not catch it. Prefer
# SLURM_SUBMIT_DIR (where sbatch was invoked, documented above as the repo
# root), then fall back to the script's own directory for a direct run.
_BODY=""
for _d in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "$0")" 2>/dev/null && pwd)" "$(pwd)"; do
    [[ -z "$_d" ]] && continue
    for _c in "$_d/scripts/_slurm_body.sh" "$_d/_slurm_body.sh"; do
        [[ -f "$_c" ]] && { _BODY="$_c"; break 2; }
    done
done
if [[ -z "$_BODY" ]]; then
    echo "cannot find scripts/_slurm_body.sh -- submit from the repo root so" >&2
    echo "SLURM_SUBMIT_DIR points at it (got '${SLURM_SUBMIT_DIR:-unset}')" >&2
    exit 1
fi
source "$_BODY"
