#!/bin/bash
#
# Submit the whole benchmark: one SLURM array per resource class, sized from the
# manifests scripts/generate_benchmark_configs.py wrote.
#
#   bash scripts/submit_benchmark.sh                 # submit everything
#   bash scripts/submit_benchmark.sh --dry-run       # print the sbatch lines
#   bash scripts/submit_benchmark.sh small medium    # only these classes
#
# Classes get their own array because #SBATCH lines are parsed before the script
# runs, so one array cannot carry per-entry memory and time limits.
#
# %20 throttles each array to 20 concurrent tasks -- polite on a shared
# partition, and it keeps a bad config from burning the whole allocation before
# anyone notices. Raise it once a class has run clean.

set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0
CLASSES=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        -*) echo "unknown flag $a" >&2; exit 1 ;;
        *) CLASSES+=("$a") ;;
    esac
done
[[ ${#CLASSES[@]} -gt 0 ]] || CLASSES=(small medium large xlarge)

for cls in "${CLASSES[@]}"; do
    man="configs/benchmark/manifest_${cls}.txt"
    if [[ ! -f "$man" ]]; then
        echo "skip ${cls}: no ${man} (run scripts/generate_benchmark_configs.py)" >&2
        continue
    fi
    n=$(grep -c . "$man")
    [[ "$n" -gt 0 ]] || { echo "skip ${cls}: manifest empty" >&2; continue; }
    cmd=(sbatch "--array=1-${n}%20" "scripts/slurm_${cls}.sh" "$man")
    if [[ "$DRY" == 1 ]]; then
        echo "${cmd[*]}"
    else
        echo "submitting ${cls}: ${n} tasks"
        "${cmd[@]}"
    fi
done
