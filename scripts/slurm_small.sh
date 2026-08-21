#!/bin/bash
#SBATCH --job-name=prune_small
#SBATCH --output=logs/slurm_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=kempner
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akazeminia@g.harvard.edu
#
# Resource class "small" of the benchmark. Submit from the repo root.
#
#   sbatch --array=1-N scripts/slurm_small.sh configs/benchmark/manifest_small.txt
#   sbatch            scripts/slurm_small.sh configs/benchmark/generated/<e>/<a>.yaml
#
# cpus-per-task=4: unlike the reparam repo's in-memory GPU pipeline, this
# one uses a torch DataLoader and the pruning maths is numpy/scipy on the host
# (Owen's-T kernel evaluations, LAPACK solves), so there IS real CPU work here.
# Verify with `jobstats <jobid>` and lower it if CPU-Util says otherwise.
#
# Override anything without editing this file:
#   sbatch --time=02:00:00 --mem=16G --array=1-10 scripts/slurm_small.sh <manifest>
source "$(dirname "$0")/_slurm_body.sh"
