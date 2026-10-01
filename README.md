# srbench — Sentinel-2 → Maxar super-resolution benchmark

A [BasicSR](https://github.com/XPixelGroup/BasicSR)-style codebase (same package layout, registries, YAML options,
`train.py` / `test.py`) for benchmarking SR models on 50,000 paired tiles: **Sentinel-2 L2A** (48 × 48 px, 12 bands,
10 m) → **Maxar RGB** (1584 × 1584 px, 0.303 m; native ratio ×33). The HR target is the Maxar tile area-downsampled
to 48·S px.

## Models

| model | type | notes |
|---|---|---|
| Bicubic baseline | learned band→RGB colour map + bicubic | the interpolation reference |
| RRDBNet → ESRGAN | CNN, L1 → GAN + perceptual | ESRGAN stage 2 starts from stage 1 |
| SwinIR | transformer (classical SR), L1 | |
| EDSR-L / EDSR-M | CNN, L1 | full model / small baseline |
| AdcSR | one-step diffusion (SD 2.1), pretrained | comparison model; RGB input, ×4 only, own env |

RRDBNet/ESRGAN, SwinIR and EDSR train either **from scratch** or from the **official pretrained RGB weights**
(`_pretrained` configs: same data and recipe, only the initialisation differs). The L1 models share the same budget
(batch 16 × 300k iterations).

## Scales

| S | HR | GSD | training | targets |
|---|---|---|---|---|
| 4 (default) | 192 px | 2.5 m | whole tiles | `gt_x4.npy` |
| 8 | 384 px | 1.25 m | whole tiles | `gt_x8.npy` |
| 16 | 768 px | 0.625 m | random crops, LR 32 → HR 512 | `gt_x16.npy` |
| 32 | 1536 px | 0.3125 m | random crops, LR 16 → HR 512 | `gt_x32.npy` |

Powers of 2 only, which every architecture supports natively; ×32 stands in for the native ×33.

## Setup

```bash
conda activate /ocean/projects/cis250179p/purohit/super-res/envs/srbench      # AdcSR: envs/adcsr (docs/07 §7.1)
python -m pytest tests                                                        # ~20 s, no GPU
python scripts/download_pretrained.py      # official RGB weights for the _pretrained configs (~360 MB)
```

The data cache `datasets/s2maxar/` exists in this checkout. To rebuild it, or to add ×16 / ×32 targets:

```bash
python scripts/prepare_s2maxar.py --out datasets/s2maxar --scales 4 8 --workers 5   # ~1 h, resumable
sbatch scripts/slurm/prepare_data.sbatch --add-scales 16 32                           # +442 GB, ~45 min
```

## Run a scale

```bash
S=4     # 4 | 8 | 16 | 32
P=      # from scratch;  P=_pretrained: start from the official RGB weights (docs/02 §2.2)
python srbench/train.py -opt options/train/Baseline/train_Bicubic_x$S.yml
python srbench/train.py -opt options/train/ESRGAN/train_RRDBNet_PSNR_x$S$P.yml --auto_resume    # ESRGAN stage 1
python srbench/train.py -opt options/train/ESRGAN/train_ESRGAN_x$S$P.yml --auto_resume          # stage 2
python srbench/train.py -opt options/train/SwinIR/train_SwinIR_SRx$S${P:-_scratch}.yml --auto_resume
python srbench/train.py -opt options/train/EDSR/train_EDSR_Lx$S$P.yml --auto_resume             # or EDSR_Mx$S$P

python srbench/test.py -opt options/test/SwinIR/test_SwinIR_SRx$S$P.yml     # test configs: same names, test_ prefix
```

On Slurm (Bridges-2, 48 h jobs, always auto-resume; chain long runs with `--dependency=afterany:<jobid>`):

```bash
sbatch scripts/slurm/train.sbatch options/train/SwinIR/train_SwinIR_SRx${S}_scratch.yml
sbatch --export=ALL,SRBENCH_ENV=/ocean/projects/cis250179p/purohit/super-res/envs/adcsr --gpus=h100-80:1 \
       scripts/slurm/train.sbatch options/train/AdcSR/train_AdcSR_x4.yml                 # AdcSR (~27 GB GPU)
```

Outputs: `experiments/<name>/` (checkpoints, log, S2 | SR | Maxar panels), `tb_logger/<name>`, and
`results/<name>/metrics_*.csv` for tests.

## Benchmark table

```bash
sbatch scripts/slurm/benchmark.sbatch x$S      # ESRGAN, EDSR, SwinIR; x$S pretrained | both: the _pretrained runs
```

Scores 500 `test_clean` tiles at the native 1584 px (outputs upsampled bicubically) with SSIM, PSNR, LPIPS, MSE,
Inception Score, CLIP score and FID → `results/benchmark/test_mini_x$S/results/test_mini_x$S.csv`.

## Documentation

| | |
|---|---|
| [docs/01_data.md](docs/01_data.md) | data, pairing, QA filter, spatial split, cache |
| [docs/02_methods.md](docs/02_methods.md) | framework and changes vs BasicSR, models, training recipes, evaluation |
| [docs/03_validation.md](docs/03_validation.md) | tests, checks, speed and memory |
| [docs/04_decisions.md](docs/04_decisions.md) | decision log |
| [docs/05_caveats_and_open_items.md](docs/05_caveats_and_open_items.md) | limitations, next steps |
| [docs/06_adcsr.md](docs/06_adcsr.md) | AdcSR comparison model |
| [docs/07_usage.md](docs/07_usage.md) | usage reference: environments, all configs, options, Slurm, W&B, outputs |
| [docs/NEXT_SESSION.md](docs/NEXT_SESSION.md) | handoff note: running jobs, open decisions |

Ported BasicSR code: Apache-2.0 (`LICENSE.BasicSR.txt`); AdcSR / RAM code: `srbench/third_party/`.
