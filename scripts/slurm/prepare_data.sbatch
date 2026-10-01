#!/bin/bash
# One-time data preparation (~1 h with 5 cores, I/O bound; resumable -> just resubmit if it dies).
# Add x16 / x32 targets (gt_x16.npy 88 GB, gt_x32.npy 354 GB) to an existing cache:
#   sbatch scripts/slurm/prepare_data.sbatch --add-scales 16 32
# The allocation only covers GPU partitions, so this takes one shared GPU slot for its 5 CPU cores.
#   sbatch scripts/slurm/prepare_data.sbatch
#SBATCH -J srbench-prep
#SBATCH -A cis250179p
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/%x-%j.out

set -euo pipefail
ENV=/ocean/projects/cis250179p/purohit/super-res/envs/srbench
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
$ENV/bin/python scripts/prepare_s2maxar.py \
    --src /ocean/projects/cis250179p/purohit/Bimal/visual_small --out datasets/s2maxar --scales 4 8 --workers 5 "$@"
