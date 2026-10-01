#!/bin/bash
# Native-resolution benchmark on test-mini (500 evenly spaced test_clean tiles, scored at 1584 px with the 7 metrics of
# scripts/compute_metrics.py). Uses each model's test YAML, i.e. the full runs' experiments/<name>/models/net_g_latest.pth.
#   sbatch scripts/slurm/benchmark.sbatch x4         # ESRGAN, EDSR (EDSR-L), SwinIR -> results/benchmark/test_mini_x4/results/test_mini_x4.csv
#   sbatch scripts/slurm/benchmark.sbatch x8          # also x16, x32
#   sbatch scripts/slurm/benchmark.sbatch x4 AdcSR=options/test/AdcSR/test_AdcSR_x4.yml    # extra ROW=YML rows
# Re-running regenerates the model rows; ground truth / bicubic reference are kept once <root>/tiles.txt exists.
#SBATCH -J srbench-benchmark
#SBATCH -A cis250179p
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 02:00:00
#SBATCH -o slurm_logs/%x-%j.out

set -euo pipefail
S=${1:-x4}
shift || true
# envs/adcsr has torchmetrics + torch-fidelity + transformers (and diffusers for an AdcSR row); srbench runs unchanged in it
ENV=/ocean/projects/cis250179p/purohit/super-res/envs/adcsr
export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

case $S in
    x4) ROWS="ESRGAN=options/test/ESRGAN/test_ESRGAN_x4.yml EDSR=options/test/EDSR/test_EDSR_Lx4.yml SwinIR=options/test/SwinIR/test_SwinIR_SRx4.yml" ;;
    x8|x16|x32) ROWS="ESRGAN=options/test/ESRGAN/test_ESRGAN_$S.yml EDSR=options/test/EDSR/test_EDSR_L$S.yml SwinIR=options/test/SwinIR/test_SwinIR_SR$S.yml" ;;
    *) echo "usage: sbatch scripts/slurm/benchmark.sbatch x4|x8|x16|x32 [ROW=YML ...]"; exit 1 ;;
esac
ROWS="$ROWS $*"
R=results/benchmark/test_mini_$S
echo "host $(hostname)  root $R  rows $ROWS  $(date)"

[ -f $R/tiles.txt ] || python scripts/generate_sr.py --root $R --reference --num-tiles 500
names=()
for spec in $ROWS; do
    python scripts/generate_sr.py --root $R --row "${spec%%=*}" -opt "${spec#*=}"
    names+=("${spec%%=*}")
done
python scripts/compute_metrics.py --root $R --models "${names[@]}" --workers 4
