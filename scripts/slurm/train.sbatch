#!/bin/bash
# Train one config. Submit from the repo root:
#   sbatch scripts/slurm/train.sbatch options/train/ESRGAN/train_RRDBNet_PSNR_x4.yml
# Multi-GPU (DDP via torchrun), e.g. 2 GPUs:
#   sbatch --gpus=v100-32:2 scripts/slurm/train.sbatch options/train/SwinIR/train_SwinIR_SRx4_scratch.yml
# Always runs with --auto_resume: resubmitting the same config continues from the latest training state, so
# runs longer than the 48 h walltime can be chained:
#   jid=$(sbatch --parsable scripts/slurm/train.sbatch CFG); sbatch --dependency=afterany:$jid scripts/slurm/train.sbatch CFG
# AdcSR runs in its own environment (requirements-adcsr.txt):
#   sbatch --export=ALL,SRBENCH_ENV=/ocean/projects/cis250179p/purohit/super-res/envs/adcsr scripts/slurm/train.sbatch options/train/AdcSR/train_AdcSR_x4.yml
#SBATCH -J srbench-train
#SBATCH -A cis250179p
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x-%j.out

set -euo pipefail
CFG=${1:?usage: sbatch scripts/slurm/train.sbatch <options/train/...yml> [extra train.py args]}
shift
ENV=${SRBENCH_ENV:-/ocean/projects/cis250179p/purohit/super-res/envs/srbench}  # AdcSR: SRBENCH_ENV=<...>/envs/adcsr
export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

NGPU=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "host $(hostname)  gpus $NGPU  cfg $CFG  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$NGPU" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NGPU" srbench/train.py -opt "$CFG" --launcher pytorch --auto_resume "$@"
else
    python srbench/train.py -opt "$CFG" --auto_resume "$@"
fi
