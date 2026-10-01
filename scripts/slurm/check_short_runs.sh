#!/bin/bash
# Summarise the short runs: job states, resume points, loss trend, validation metrics.
#   bash scripts/slurm/check_short_runs.sh [run names under experiments/ ...]
cd "$(dirname "$0")/../.."
if [ -f slurm_logs/short_runs_jobids.txt ]; then
    sacct -X -j "$(tr ' ' ',' < slurm_logs/short_runs_jobids.txt)" --format=JobID,JobName%16,State%12,Elapsed,NodeList%10
fi
RUNS=${*:-short_RRDBNet_PSNR_x4 short_SwinIR_SRx4 short_ESRGAN_x4 Bicubic_x4_S2Maxar_linearcolor_5k_B16G1}
for n in $RUNS; do
    d=experiments/$n
    echo; echo "=== $n"
    [ -d "$d" ] || { echo "  not started"; continue; }
    echo "  training states: $(ls "$d/training_states" 2>/dev/null | sort -n | tr '\n' ' ')"
    grep -h "Start training from\|Resuming training" "$d"/train_*.log | sed 's/^.*INFO: /  /'
    echo "  loss (mean over each 100-iteration interval, every 500 iterations):"
    grep -hE "iter:\s+[0-9,]+, lr" "$d"/train_*.log | sed -E 's/.*iter:\s+([0-9,]+),.*time \(data\): ([0-9.]+) \(([0-9.]+)\)\] (.*)/\1 \2 \3 \4/' |
        awk '{gsub(",", "", $1); if ($1 % 500 == 0) {printf "    iter %5d  %.2f s/it (data %.3f)  ", $1, $2, $3; for (i = 4; i <= NF; i++) printf "%s ", $i; print ""}}' |
        cut -c1-170
    echo "  validation (in order, one value per val_freq):"
    for L in "$d"/train_*.log; do
        for m in psnr ssim lpips; do
            vals=$(grep -hE "# $m:" "$L" | sed -E "s/^\s+# $m: ([0-9.]+).*/\1/" | tr '\n' ' ')
            [ -n "$vals" ] && echo "    $(basename "$L" .log | sed 's/.*_\([0-9]*_[0-9]*\)$/\1/')  $m: $vals"
        done
    done
    grep -hE "Traceback|Error|error:" "$d"/train_*.log | head -3 | sed 's/^/  !! /'
    grep -h "End of training" "$d"/train_*.log | sed 's/^.*INFO: /  /'
done
