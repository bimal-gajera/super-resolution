#!/bin/bash
# Short AdcSR check (3k iterations) that also tests resuming across jobs, as short_runs.sh does for the other models:
# part A is killed by its walltime mid-run, part B (--dependency=afterany:A) continues with --auto_resume.
# AdcSR needs ~20 GB (student + Adam 8.5 GB, frozen SD discriminator / OSEDiff teacher / RAM / text encoder ~11 GB):
# v100-32 or h100-80, not v100-16.
#
#   bash scripts/slurm/short_runs_adcsr.sh [h100-80|v100-32]     # from the repo root
#   bash scripts/slurm/check_short_runs.sh short_AdcSR_x4
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p slurm_logs
GPU=${1:-h100-80}
# the environment must be passed explicitly: Bridges-2 sessions may set SBATCH_EXPORT=NONE
X=--export=ALL,SRBENCH_ENV=/ocean/projects/cis250179p/purohit/super-res/envs/adcsr
T=scripts/slurm/train.sbatch
C="options/train/AdcSR/train_AdcSR_x4.yml --force_yml name=short_AdcSR_x4 train:total_iter=3000 val:val_freq=500 logger:save_checkpoint_freq=250 logger:print_freq=50"

a=$(sbatch --parsable $X --gpus=$GPU:1 -J short-adcsr-A -t 00:12:00 $T $C)
b=$(sbatch --parsable $X --gpus=$GPU:1 -J short-adcsr-B -t 01:00:00 --dependency=afterany:$a $T $C)
echo "AdcSR  A=$a B=$b  ($GPU)"
echo "$a $b" > slurm_logs/short_runs_adcsr_jobids.txt
