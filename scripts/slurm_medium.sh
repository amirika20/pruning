#!/bin/bash
#SBATCH --job-name=prune_medium
#SBATCH --output=logs/slurm_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akazeminia@g.harvard.edu
#
# Resource class "medium" of the benchmark. Submit from the repo root.
#
#   sbatch --array=1-N scripts/slurm_medium.sh configs/benchmark/manifest_medium.txt
#   sbatch            scripts/slurm_medium.sh configs/benchmark/generated/<e>/<a>.yaml
#
# cpus-per-task=8: unlike the reparam repo's in-memory GPU pipeline, this
# one uses a torch DataLoader and the pruning maths is numpy/scipy on the host
# (Owen's-T kernel evaluations, LAPACK solves), so there IS real CPU work here.
# Verify with `jobstats <jobid>` and lower it if CPU-Util says otherwise.
#
# Override anything without editing this file:
#   sbatch --time=02:00:00 --mem=16G --array=1-10 scripts/slurm_medium.sh <manifest>
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
