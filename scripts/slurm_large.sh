#!/bin/bash
#SBATCH --job-name=prune_large
#SBATCH --output=logs/slurm_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --partition=kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akazeminia@g.harvard.edu
#
# Resource class "large" of the benchmark. Submit from the repo root.
#
#   sbatch --array=1-N scripts/slurm_large.sh configs/benchmark/manifest_large.txt
#   sbatch            scripts/slurm_large.sh configs/benchmark/generated/<e>/<a>.yaml
#
# cpus-per-task=8: unlike the reparam repo's in-memory GPU pipeline, this
# one uses a torch DataLoader and the pruning maths is numpy/scipy on the host
# (Owen's-T kernel evaluations, LAPACK solves), so there IS real CPU work here.
# Verify with `jobstats <jobid>` and lower it if CPU-Util says otherwise.
#
# Override anything without editing this file:
#   sbatch --time=02:00:00 --mem=16G --array=1-10 scripts/slurm_large.sh <manifest>
source "$(dirname "$0")/_slurm_body.sh"
