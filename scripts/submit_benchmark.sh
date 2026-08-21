#!/bin/bash
#
# Submit the benchmark as one SLURM array per resource class, sized for a
# 16-GPU allocation.
#
#   bash scripts/submit_benchmark.sh                 # all classes
#   bash scripts/submit_benchmark.sh --dry-run
#   bash scripts/submit_benchmark.sh small medium    # only these
#   CONCURRENCY=8 bash scripts/submit_benchmark.sh   # fewer GPUs
#
# ARRAY SIZING. Each task claims a strided share of its manifest (see
# _slurm_body.sh), so the array size decides cells-per-task. With 16 GPUs:
#
#   small / medium   16 tasks -> ~11-12 cells each, one wave, no requeueing.
#                    These are MNIST/Fashion/CIFAR/OPT-125m-scale cells that
#                    take seconds to a few minutes, so batching them is the
#                    difference between 12 queue waits and one.
#   large            32 tasks throttled to 16 -> ~8 cells each, two waves.
#                    ImageNet forward passes dominate, so smaller batches keep
#                    any single task inside the time limit.
#   xlarge           one task per cell, throttled to 16. OPT-6.7b/13b load
#                    ~27-52GB of weights per cell; batching them would just
#                    reload the same checkpoint repeatedly and risk the wall
#                    clock.

set -euo pipefail
cd "$(dirname "$0")/.."

CONCURRENCY="${CONCURRENCY:-16}"
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
    n=$(grep -c '[^[:space:]]' "$man" || true)
    [[ "${n:-0}" -gt 0 ]] || { echo "skip ${cls}: manifest empty" >&2; continue; }

    case "$cls" in
        small|medium) tasks=$CONCURRENCY ;;
        large)        tasks=$((CONCURRENCY * 2)) ;;
        xlarge)       tasks=$n ;;
        *)            tasks=$CONCURRENCY ;;
    esac
    [[ "$tasks" -le "$n" ]] || tasks=$n
    per=$(( (n + tasks - 1) / tasks ))

    cmd=(sbatch "--array=1-${tasks}%${CONCURRENCY}" "scripts/slurm_${cls}.sh" "$man")
    if [[ "$DRY" == 1 ]]; then
        printf '%-8s %4d cells -> %3d tasks (~%d cells/task): %s\n' \
               "$cls" "$n" "$tasks" "$per" "${cmd[*]}"
    else
        echo "submitting ${cls}: ${n} cells over ${tasks} tasks (~${per} each)"
        "${cmd[@]}"
    fi
done
