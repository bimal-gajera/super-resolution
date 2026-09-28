#!/bin/bash
# Short Slurm runs (~2 epochs = 5k iterations of train_clean) that check training AND resuming across jobs.
# Each model runs as two jobs: part A is killed by its walltime mid-run (state TIMEOUT), part B
# (--dependency=afterany:A) restarts the same command, --auto_resume picks up the latest training state and finishes.
# ESRGAN (2k iterations) starts from the smoke RRDBNet result. Bicubic is the real (full, 5k-iteration) baseline run.
#
#   bash scripts/slurm/smoke_test.sh          # from the repo root; prints the job ids
#   bash scripts/slurm/check_smoke.sh         # progress / verification summary
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p slurm_logs
T=scripts/slurm/train.sbatch
sub() { sbatch --parsable "$@"; }

R="options/train/ESRGAN/train_RRDBNet_PSNR_x4.yml --force_yml name=smoke_RRDBNet_PSNR_x4 train:total_iter=5000 val:val_freq=1000 logger:save_checkpoint_freq=1000"
S="options/train/SwinIR/train_SwinIR_SRx4_scratch.yml --force_yml name=smoke_SwinIR_SRx4 train:total_iter=5000 val:val_freq=1000 logger:save_checkpoint_freq=1000"
E="options/train/ESRGAN/train_ESRGAN_x4.yml --force_yml name=smoke_ESRGAN_x4 train:total_iter=2000 val:val_freq=500 logger:save_checkpoint_freq=500 path:pretrain_network_g=experiments/smoke_RRDBNet_PSNR_x4/models/net_g_latest.pth"
B="options/train/Baseline/train_Bicubic_x4.yml"

# RRDBNet: ~0.33 s/iter -> 5k iters ~30 min; A stops after 20 min (~3k iters)
r1=$(sub -J smoke-rrdb-A -t 00:20:00 $T $R)
r2=$(sub -J smoke-rrdb-B -t 00:30:00 --dependency=afterany:$r1 $T $R)
# SwinIR: ~0.51 s/iter -> 5k iters ~45 min; A stops after 25 min (~2.5k iters)
s1=$(sub -J smoke-swinir-A -t 00:25:00 $T $S)
s2=$(sub -J smoke-swinir-B -t 00:45:00 --dependency=afterany:$s1 $T $S)
# ESRGAN from the finished smoke RRDBNet: ~0.52 s/iter -> 2k iters ~20 min; A stops after 12 min (~1k iters)
e1=$(sub -J smoke-esrgan-A -t 00:12:00 --dependency=afterok:$r2 $T $E)
e2=$(sub -J smoke-esrgan-B -t 00:25:00 --dependency=afterany:$e1 $T $E)
# Bicubic baseline (full run, 5k iters, a few minutes)
b1=$(sub -J bicubic-x4 -t 00:30:00 $T $B)

echo "RRDBNet  A=$r1 B=$r2"
echo "SwinIR   A=$s1 B=$s2"
echo "ESRGAN   A=$e1 B=$e2 (after RRDBNet B)"
echo "Bicubic  $b1"
echo "$r1 $r2 $s1 $s2 $e1 $e2 $b1" > slurm_logs/smoke_jobids.txt
