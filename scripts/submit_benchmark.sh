#!/bin/bash
#
# Submit the benchmark as one SLURM array per resource class, sized for a
# 16-GPU allocation.
#
#   bash scripts/submit_benchmark.sh                        # everything
#   bash scripts/submit_benchmark.sh --tier headline        # Tables 2/3 first
#   bash scripts/submit_benchmark.sh --tier ablation        # then Table 4
#   bash scripts/submit_benchmark.sh --dry-run
#   bash scripts/submit_benchmark.sh small medium           # only these classes
#   CONCURRENCY=8 bash scripts/submit_benchmark.sh          # fewer GPUs
#
# --tier headline is the one to run first: it is the ~13 arms that answer "how
# much can this remove, against the baselines" on every model including the big
# ones. --tier ablation is the rest of the design grid, which only the cheap
# entries carry (see the tier note in configs/benchmark/arms.yaml).
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
TIER=""
CLASSES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1 ;;
        --tier) TIER="${2:?--tier needs headline|ablation}"; shift ;;
        --tier=*) TIER="${1#*=}" ;;
        -*) echo "unknown flag $1" >&2; exit 1 ;;
        *) CLASSES+=("$1") ;;
    esac
    shift
done
case "$TIER" in
    ""|headline|ablation) ;;
    *) echo "--tier must be headline or ablation" >&2; exit 1 ;;
esac
SUFFIX="${TIER:+_$TIER}"
[[ ${#CLASSES[@]} -gt 0 ]] || CLASSES=(small medium large xlarge)

for cls in "${CLASSES[@]}"; do
    man="configs/benchmark/manifest_${cls}${SUFFIX}.txt"
    if [[ ! -f "$man" ]]; then
        # A class can legitimately have no cells in a tier -- the expensive
        # entries carry headline arms only, so their ablation manifest is absent.
        echo "skip ${cls}${SUFFIX:+ (${TIER})}: no ${man}" >&2
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
        printf '%-8s %-9s %4d cells -> %3d tasks (~%d cells/task): %s\n' \
               "$cls" "${TIER:-all}" "$n" "$tasks" "$per" "${cmd[*]}"
    else
        echo "submitting ${cls} (${TIER:-all}): ${n} cells over ${tasks} tasks (~${per} each)"
        "${cmd[@]}"
    fi
done
