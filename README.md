# srbench — Sentinel-2 → Maxar super-resolution benchmark

A [BasicSR](https://github.com/XPixelGroup/BasicSR)-style codebase (same package layout, registries, YAML
options, `train.py`/`test.py` entry points, experiment folders) for benchmarking SR models on paired
**Sentinel-2 L2A (10 m, 12 bands)** → **Maxar RGB** imagery.

| model | configs | notes |
|---|---|---|
| Bicubic baseline | `options/{train,test}/Baseline/*Bicubic_x4.yml` | learned linear band→RGB map + bicubic upsampling (the "interpolation" row) |
| RRDBNet (ESRGAN stage 1, PSNR) | `options/{train,test}/ESRGAN/*RRDBNet_PSNR_x4.yml` | L1 |
| ESRGAN (stage 2, GAN) | `options/{train,test}/ESRGAN/*ESRGAN_x4.yml` | init from stage 1; L1 + VGG19 perceptual + relativistic GAN |
| SwinIR (classical SR) | `options/{train,test}/SwinIR/*SwinIR_SRx4*.yml` | L1, embed 180, 6×6 RSTB, window 8 |
| EDSR (EDSR-L; EDSR-M = baseline size) | `options/{train,test}/EDSR/*EDSR_{L,M}x4.yml` | L1, 32 blocks × 256 feat, res_scale 0.1 (M: 16 × 64) |
| AdcSR (one-step diffusion, CVPR 2025) | `options/{train,test}/AdcSR/*AdcSR_x4*.yml` | RGB input; SD 2.1-based student distilled from OSEDiff + adversarial loss; ×4 only; own env (below) |

**Scales:** ×4 (default), ×8, ×16 and ×32: powers of 2, which all architectures support natively; ×32 (0.3125 m)
stands in for the native ×33 (0.303 m). The ×16/×32 configs are generated from the ×4 ones by
`scripts/make_scale_configs.py` and train on random crops (LR 32 / 16 px → HR 512); their targets are cached with
`prepare_s2maxar.py --add-scales 16 32`. AdcSR is ×4 only.

The PSNR-oriented models (RRDBNet, SwinIR, EDSR) share the same data budget (batch 16 × 300k iterations) so
they are directly comparable; LR schedules follow the shape of the respective BasicSR configs.

## Environment

```bash
conda activate /ocean/projects/cis250179p/purohit/super-res/envs/srbench   # Python 3.11, torch 2.14 + CUDA 12.6
# to recreate elsewhere:
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 && pip install -e .
python -m pytest tests      # ~20 s, synthetic data, no GPU needed
```

The cu126 wheels cover V100 (sm_70), L40S and H100. `pip install -e .` is required (like BasicSR's
`python setup.py develop`) so that `python srbench/train.py` can import the package.

**AdcSR** uses a second environment, because its prompt extractor (RAM/DAPE) needs the old transformers/timm versions
pinned by the AdcSR repo. srbench itself runs unchanged in it (all tests pass):

```bash
conda create -p /ocean/projects/cis250179p/purohit/super-res/envs/adcsr python=3.10 -y
envs/adcsr/bin/pip install -r requirements-adcsr.txt --extra-index-url https://download.pytorch.org/whl/cu121
envs/adcsr/bin/pip install -e . --no-deps
envs/adcsr/bin/python scripts/download_adcsr_weights.py   # ~17 GB -> experiments/pretrained_models/{AdcSR,stable-diffusion-2-1-base}
```

## Data

Source: `/ocean/projects/cis250179p/purohit/Bimal/visual_small` — 50,000 pairs. The prepared cache
(`datasets/s2maxar`, 29 GB) already exists in this checkout.

| | Maxar | Sentinel-2 |
|---|---|---|
| file | `Maxar/<key>.tif` | `Sentinel-2/<key with MAXAR→S2>.tif` |
| array | 1584 × 1584 × 3, uint8 RGB | 48 × 48 × 12, uint16 (L2A reflectance × 10⁴; B01–B12 without B10) |
| GSD | 0.303 m | 10 m |

Both tiles cover the same 480 m footprint, i.e. the native ratio is **×33**. The benchmark therefore defines the
HR target as the Maxar tile area-downsampled to `48 × scale` px: **×4 → 192 px @ 2.5 m** (default configs) and
**×8 → 384 px @ 1.25 m** (cached as well).

Two file-naming schemes occur (`<catalog>_<quadkey>_<idx>_MAXAR.tif` and `<catalog>_<quadkey>_MAXAR_<idx>.tif`);
the key is the full Maxar file stem, so 464 `(catalog, quadkey, idx)` triples that exist under both schemes
stay separate pairs (they are mostly different tiles).

### Preparation (one time, ~1 h, resumable)

```bash
python scripts/prepare_s2maxar.py --out datasets/s2maxar --scales 4 8 --workers 5
# or: sbatch scripts/slurm/prepare_data.sbatch        (mkdir -p slurm_logs first)
```

`datasets/s2maxar/` then contains

| file | content |
|---|---|
| `keys.txt` | all pair keys; row *i* of every array belongs to line *i* |
| `lq_s2.npy` | (N, 48, 48, 12) uint16 raw S2 (memory-mapped by the dataset) |
| `gt_x4.npy`, `gt_x8.npy` | (N, 192/384, 192/384, 3) uint8 Maxar, `cv2.INTER_AREA` |
| `qa.csv` | per-pair QA: nodata, S2↔Maxar correlation, texture, B02 brightness, residual shift |
| `meta_info/{train,val,test}.txt` | spatial split 90/5/5 |
| `meta_info/{train,val,test}_clean.txt` | the same after the QA filter (default in all configs) |
| `stats.json` | per-band S2 mean/std on `train_clean` (input z-scoring) + split summary |

**Spatial split.** The second token of every key is a level-12 quadkey of a Maxar ARD-style grid defined per UTM
zone (≈ 5 km cells holding 11 × 11 chips of 480 m). Whole level-10 prefix groups (≈ 20 km cells; 535 groups) are
assigned to one split, so repeated acquisitions of the same area (1,822 quadkeys
have several Maxar catalogs, some tiles are near-duplicates) never leak between train and val/test.

**QA filter** (`is_clean` in the prep script; thresholds are CLI flags, re-run cheaply with `--lists-only`):
a pair is dropped when S2 does not show what Maxar shows — mostly clouds/haze in S2, sometimes strong
temporal change: grey-level correlation of Maxar (downsampled to 10 m) and S2 RGB `< 0.2`, unless both tiles are
flat (water, uniform fields: coefficient of variation `< 0.1`, where correlation is meaningless); or median
B02 `> 4000` (cloud/snow); or > 5 % nodata.

| split | pairs | QA-clean | quadkey groups |
|---|---|---|---|
| train | 45,000 | 39,603 | 485 |
| val | 2,500 | 2,056 | 26 |
| test | 2,500 | 2,130 | 24 |

All 50,000 pairs are readable (none dropped); 12.4 % are flagged by the QA filter.

**Dataset options** (`S2MaxarDataset`, see its docstring): `lq_bands` (default all 12; `[3, 2, 1]` = RGB only —
remember to set `num_in_ch`/`in_chans`), `lq_norm` (`meanstd` | `reflectance`), `gt_size`, `use_hflip`/`use_rot`,
`max_samples`, `io_backend: {type: disk, maxar_dir, s2_dir}` to read raw GeoTIFFs on the fly (slow, but no
cache needed), and `lq_source: bicubic` (synthetic LR from the GT, a pipeline sanity check).

## Training

```bash
# ESRGAN = stage 1 (PSNR) then stage 2 (GAN; loads experiments/<stage-1 name>/models/net_g_latest.pth)
python srbench/train.py -opt options/train/ESRGAN/train_RRDBNet_PSNR_x4.yml --auto_resume
python srbench/train.py -opt options/train/ESRGAN/train_ESRGAN_x4.yml --auto_resume
python srbench/train.py -opt options/train/SwinIR/train_SwinIR_SRx4_scratch.yml --auto_resume
python srbench/train.py -opt options/train/EDSR/train_EDSR_Lx4.yml --auto_resume        # or train_EDSR_Mx4.yml
envs/adcsr/bin/python srbench/train.py -opt options/train/AdcSR/train_AdcSR_x4.yml --auto_resume   # ~27 GB GPU memory
python srbench/train.py -opt options/train/Baseline/train_Bicubic_x4.yml

# Slurm (Bridges-2, 48 h max; --auto_resume is always on, so chaining jobs continues the run)
mkdir -p slurm_logs
sbatch scripts/slurm/train.sbatch options/train/ESRGAN/train_RRDBNet_PSNR_x4.yml
jid=$(sbatch --parsable scripts/slurm/train.sbatch CFG); sbatch --dependency=afterany:$jid scripts/slurm/train.sbatch CFG
sbatch --gpus=v100-32:2 scripts/slurm/train.sbatch CFG                  # 2 GPUs -> torchrun DDP
# AdcSR: its env must be passed explicitly (sessions here may set SBATCH_EXPORT=NONE); v100-32 or h100-80
sbatch --export=ALL,SRBENCH_ENV=/ocean/projects/cis250179p/purohit/super-res/envs/adcsr --gpus=h100-80:1 \
       scripts/slurm/train.sbatch options/train/AdcSR/train_AdcSR_x4.yml
```

**Weights & Biases** (optional, mirrors tensorboard: losses, val metrics, LR|SR|GT panels): `wandb login` once, then
set `logger.wandb.project` in the YAML or on the command line, e.g.
`--force_yml logger:wandb:project=s2maxar-sr`. Jobs continued with `--auto_resume` append to the same W&B run
(id stored in `experiments/<name>/wandb_id.txt`); `--debug` runs are never logged. On nodes without internet use
`export WANDB_MODE=offline` and upload later with `wandb sync experiments/<name>/wandb/offline-run-*`.

Everything BasicSR offers works the same way: `--debug` (tiny intervals, name prefixed `debug_`),
`--force_yml key:sub=value` overrides, `--launcher pytorch` for DDP, `auto_resume`, EMA (`ema_decay`), tensorboard
(`tb_logger/<name>`). Outputs go to `experiments/<name>/`: `models/net_g_<iter>.pth` (`params` + `params_ema`),
`models/net_g_best.pth` (best `val.save_best` metric), `training_states/`, the log, and
`visualization/<tile>/<tile>_<iter>.png` panels **S2 RGB | SR | Maxar GT** for the first `max_save_img` val tiles.

Measured on one V100-16GB (batch 16, ×4, fp32): RRDBNet 0.33 s/iter (10.6 GB), SwinIR 0.51 s/iter (14.3 GB
allocated, ~15.5 GB reserved — it fits, but the sbatch default `v100-32` leaves headroom; for ×8 or larger batches
use `use_checkpoint: true` or a 32/80 GB GPU), EDSR-L 0.35 s/iter (5.4 GB), EDSR-M 0.04 s/iter (0.9 GB). Data
loading (memory-mapped cache) is ~1 ms/iter once warm.

×8 (`*_x8.yml`, 384 px targets): RRDBNet bs16 needs 14.6 GB and SwinIR bs16 does not fit 16 GB → use a 32 GB
GPU (sbatch default); ESRGAN ×8 defaults to batch 8; EDSR-L ×8 bs16 needs 12.2 GB + cuDNN workspace and ~1 s/iter
(~83 h → two chained 48 h jobs).

### Pipeline validation (2026-09-27)

Short runs on a V100 with ~12k cached pairs and a temporary split. They show that every model learns; they
are **not benchmark numbers**, because that temporary split is not the final one. Validation uses the EMA weights on
500 val tiles. Logs: `experiments/check_*`; curves: `tensorboard --logdir tb_logger`.

| run | iters | training loss | val PSNR (dB) / SSIM | val LPIPS |
|---|---|---|---|---|
| Bicubic baseline | 3k | L1 0.34 → 0.09 | 17.06 → **18.83** / 0.376 | – |
| RRDBNet (PSNR) | 3.5k | L1 0.089 → 0.074 | 13.88 → **19.72** / 0.400 | – |
| SwinIR | 2.5k | L1 0.098 → 0.083 | 17.09 → **19.63** / 0.379 | – |
| ESRGAN (from the RRDBNet run) | 1.5k | perceptual 1.37 → 1.31, D balanced | 19.83 → 19.65 / 0.397 | 0.683 → **0.459** |

EDSR (added 2026-09-29, checked on the final split, same 500 `val_clean` tiles): EDSR-M 2k iterations, L1
0.092 → 0.071, PSNR 16.86 → **19.99** / 0.378, resumed from iteration 1k; EDSR-L 1k iterations, L1 0.093 → 0.074,
PSNR 17.38 → **18.74** / 0.343. Logs: `experiments/short_EDSR_{M,L}x4`.

## Testing

```bash
python srbench/test.py -opt options/test/ESRGAN/test_RRDBNet_PSNR_x4.yml     # or sbatch scripts/slurm/test.sbatch ...
envs/adcsr/bin/python srbench/test.py -opt options/test/AdcSR/test_AdcSR_x4.yml           # trained AdcSR
envs/adcsr/bin/python srbench/test.py -opt options/test/AdcSR/test_AdcSR_x4_official.yml  # released AdcSR, zero-shot
```

Evaluates `test_clean` (headline numbers) and `test` (incl. QA-flagged pairs) with PSNR / SSIM (RGB, 4 px
border crop, as in BasicSR) and LPIPS (AlexNet). Writes `results/<name>/metrics_<set>.csv` (per tile, e.g. to
slice by QA columns), the averages in the log, SR PNGs and LR|SR|GT panels.

### Benchmark table (7 metrics, native resolution)

The comparison table is computed with the project's metric script, ported unchanged as
`scripts/compute_metrics.py`. It scores at the **native Maxar resolution (1584 px)**: each model's ×4/×8 output is
upsampled bicubically to 1584 px and compared with the original Maxar tile. **test-mini** = 500 evenly spaced
`test_clean` tiles. Metrics: SSIM, PSNR, LPIPS, MSE (per tile, averaged), Inception Score, CLIP score
(prediction-vs-GT image similarity, CLIP ViT-L/14-336), FID (FID/IS on native-resolution 299 px patches). Runs in
`envs/adcsr` (torchmetrics, torch-fidelity, transformers).

```bash
sbatch scripts/slurm/benchmark.sbatch x4      # -> results/benchmark/test_mini_x4/results/test_mini_x4.csv
sbatch scripts/slurm/benchmark.sbatch x8
# by hand: python scripts/generate_sr.py --root R --reference; ... --root R --row ESRGAN -opt options/test/...yml;
#          python scripts/compute_metrics.py --root R --models ESRGAN EDSR SwinIR
```

Output `test_mini_x4.csv`: rows = models, columns `SSIM,PSNR,LPIPS,MSE,Inception_Score,CLIP_SCORE,FID`; plus
`summary.md/csv/json` (95 % CIs, whole-tile FID/IS, bicubic reference row) and `per_image.csv`.

Detailed documentation of the data analysis, methods, validation and decisions: [`docs/`](docs/README.md).

## Differences from BasicSR

Ported (Apache-2.0, see `LICENSE.BasicSR.txt`): registries, option parsing, logger, `SRModel`, `SRGANModel`,
`ESRGANModel`, `SwinIRModel`, losses, PSNR/SSIM, schedulers, `RRDBNet`, `SwinIR`, `EDSR`, `VGGStyleDiscriminator`.
Changes:

* `S2MaxarDataset` (multispectral uint16 LR, memory-mapped cache, spatial split lists); batched validation.
* `RRDBNet`: ×8 (third nearest+conv stage; ×4 is unchanged, so official weights still load).
  `SwinIR`: `out_chans` (12 bands in, RGB out). `SwinIR`/`EDSR`: the DIV2K RGB mean shift is applied only to
  3-channel inputs/outputs (z-scored S2 bands are already zero-mean). `VGGStyleDiscriminator`: any input size divisible by 32 (192).
  `BicubicBaseline` arch; LPIPS metric.
* `AdcSR` / `AdcSRDiscriminator` archs and `AdcSRModel` ported from the AdcSR repo (not part of BasicSR; RAM/DAPE code
  vendored in `srbench/third_party/`); `logger.keep_last: N` keeps only the N newest checkpoints (multi-GB models).
* Fixes: SwinIR `use_checkpoint=True` crashed (`x_size` not passed); `test_selfensemble` without EMA; auto-resume
  looked for states relative to the CWD; `torch.load` with `weights_only` for torch ≥ 2.6; torchvision
  `pretrained=` removal; the training loop kept iterating over leftover epochs after `total_iter`.
* `net_g_best.pth` + best-metric record stored in the training state (survives resumes; validation runs before
  checkpointing); training stops on non-finite losses; relative paths in YAMLs resolve against the repo root.

## Known data caveats

* **Radiometry is mixed**: some S2 tiles carry the +1000 DN offset of L2A processing baseline ≥ 04.00 (Jan 2022+)
  and others do not; acquisition dates are not in the files, so it cannot be corrected here. Per-band z-scoring
  reduces but does not remove this.
* **Residual misregistration** between S2 and Maxar content: see `docs/01_data.md` §1.6 and
  `datasets/s2maxar/registration.csv` (`scripts/registration.py`). The `shift_*` columns of `qa.csv` are deprecated
  (that estimator is quantised to whole S2 pixels). Absolute PSNR is low for every method; compare methods against
  each other and against the Bicubic baseline.
