#!/bin/bash
# Evaluate one test config (results/<name>/: metrics_<set>.csv per tile, log with averages, images).
#   sbatch scripts/slurm/test.sbatch options/test/ESRGAN/test_RRDBNet_PSNR_x4.yml
#SBATCH -J srbench-test
#SBATCH -A cis250179p
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 02:00:00
#SBATCH -o slurm_logs/%x-%j.out

set -euo pipefail
CFG=${1:?usage: sbatch scripts/slurm/test.sbatch <options/test/...yml> [extra test.py args]}
shift
ENV=${SRBENCH_ENV:-/ocean/projects/cis250179p/purohit/super-res/envs/srbench}  # AdcSR: SRBENCH_ENV=<...>/envs/adcsr
export PATH=$ENV/bin:$PATH
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
python srbench/test.py -opt "$CFG" "$@"
