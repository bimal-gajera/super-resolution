#!/bin/bash
# Native-resolution benchmark on test-mini (500 evenly spaced test_clean tiles, scored at 1584 px with the 7 metrics of
# scripts/compute_metrics.py). Uses each model's test YAML, i.e. the full runs' experiments/<name>/models/net_g_latest.pth.
#   sbatch scripts/slurm/benchmark.sbatch x4         # ESRGAN, EDSR (EDSR-L), SwinIR -> results/benchmark/test_mini_x4/results/test_mini_x4.csv
#   sbatch scripts/slurm/benchmark.sbatch x8          # also x16, x32
#   sbatch scripts/slurm/benchmark.sbatch x4 pretrained   # the _pretrained runs -> .../results_pretrained/test_mini_x4_pretrained.csv
#   sbatch scripts/slurm/benchmark.sbatch x4 both         # scratch and pretrained rows in one table (results_both/)
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
INIT=scratch
case ${1:-} in scratch|pretrained|both) INIT=$1; shift ;; esac
# envs/adcsr has torchmetrics + torch-fidelity + transformers (and diffusers for an AdcSR row); srbench runs unchanged in it
ENV=/ocean/projects/cis250179p/purohit/super-res/envs/adcsr
export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

case $S in
    x4|x8|x16|x32) ;;
    *) echo "usage: sbatch scripts/slurm/benchmark.sbatch x4|x8|x16|x32 [scratch|pretrained|both] [ROW=YML ...]"; exit 1 ;;
esac
rows() {  # ESRGAN, EDSR (EDSR-L), SwinIR; $1 = suffix of row name and test config ("" or _pretrained)
    echo "ESRGAN$1=options/test/ESRGAN/test_ESRGAN_$S$1.yml EDSR$1=options/test/EDSR/test_EDSR_L$S$1.yml" \
         "SwinIR$1=options/test/SwinIR/test_SwinIR_SR$S$1.yml"
}
case $INIT in
    scratch) ROWS=$(rows "") ;;
    pretrained) ROWS=$(rows _pretrained) ;;
    both) ROWS="$(rows "") $(rows _pretrained)" ;;
esac
ROWS="$ROWS $*"
R=results/benchmark/test_mini_$S
OUT=$R/results TABLE=$R/results/test_mini_$S.csv
[ $INIT = scratch ] || { OUT=$R/results_$INIT; TABLE=$OUT/test_mini_${S}_$INIT.csv; }
echo "host $(hostname)  root $R  rows $ROWS  $(date)"

[ -f $R/tiles.txt ] || python scripts/generate_sr.py --root $R --reference --num-tiles 500
names=()
for spec in $ROWS; do
    python scripts/generate_sr.py --root $R --row "${spec%%=*}" -opt "${spec#*=}"
    names+=("${spec%%=*}")
done
python scripts/compute_metrics.py --root $R --models "${names[@]}" --workers 4 --out-dir $OUT --table-csv $TABLE
