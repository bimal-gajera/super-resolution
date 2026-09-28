#!/bin/bash
# One-time data preparation (~1 h with 5 cores, I/O bound; resumable -> just resubmit if it dies).
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
