# Field-boundary downstream task: step by step

How to rerun the field-boundary study in `downstream/fields/`: Sentinel-2 chips of Fields of The World (FTW) go
through a 3 m super-resolution (SR) model, and the SR images are used by the downstream model of Fields of the Planet
(FTP, arXiv 2607.04449) in place of PlanetScope. The commands below are the ones used for row "3" (the 3 m LDM); the
times are those measured on Bridges-2.

```
FTW chip list (FTP's sets) ──► 12-band FTW chips ──► 3 m SR (LDM) ──► + NIR, + labels ──► paper's data layout
                                (FTW's procedure)    853 x 853 px       (stack_nir,          │
                                256 x 256 px, 2 dates                    FTW polygons)       ▼
                                                                         paper's U-Net + EfficientNet-B3 (its code)
                                                                         ──► PQ, SQ, RQ, object F1, boundary error
```

All commands are run from `downstream/fields/` unless noted; `SUPERRES` is the folder that holds `super-resolution/`,
`envs/`, the SR models and their code (`/ocean/projects/cis250179p/purohit/super-res` here). Data, runs, results,
logs and notes are git-ignored and stay in `downstream/fields/`.

## 0. Setup (once)

**Third-party code**, cloned into `downstream/fields/third_party/` (git-ignored):

```bash
git clone https://github.com/taylor-geospatial/fields-of-the-planet third_party/fields-of-the-planet
git -C third_party/fields-of-the-planet checkout 56893bc        # the paper's training and evaluation code
git clone https://github.com/fieldsoftheworld/ftw-dataset-creation third_party/ftw-dataset-creation
git -C third_party/ftw-dataset-creation checkout 812de88        # FTW's chip-creation code
```

**Environments** (in `$SUPERRES/envs/`):

| env | used for | how it was made |
|---|---|---|
| `ftp` | the paper's code, chip lists, 12-band chips, data layout, training, scoring | `python3.13 -m venv --system-site-packages envs/ftp` (on top of `env_sr`), then `pip install -e super-resolution/downstream/fields/third_party/fields-of-the-planet` and `pip install torchgeo==0.9.0 timm==1.0.27` (the versions in the paper's `uv.lock`; torchgeo 0.10 breaks `ftw_tools`) |
| `ldm` | the 3 m LDM | `python3.10 -m venv envs/ldm`; `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124`; `pip install pytorch-lightning==1.9.5 torchmetrics==0.11.4 omegaconf==2.3.0 einops==0.8.0 numpy==1.26.4 setuptools==69.5.1 rasterio==1.4.4 tqdm`; clone `CompVis/taming-transformers` at 3ba01b2 into `envs/ldm/src/`; put it and `$SUPERRES/latent-diffusion` on the path with `.pth` files in site-packages (the versions listed in the LDM code's `README_BENCH.md`) |
| `env_sr` | chip selection for the first check, PixelDiT | existing environment (Python 3.13) |

**SR model and its code** (from the SR benchmark, not in this repo): `$SUPERRES/inference/` (`predict_ldm.py`,
`sr_tiling.py`, `stack_nir.py`, `download_s2.py`, README), `$SUPERRES/ldm/` (`ldm_3m.ckpt`, `config.yaml`,
`model.json`), `$SUPERRES/latent-diffusion/` (CompVis code with small pytorch-lightning 1.9 changes). The unchanged
`predict_ldm.py` reproduces the benchmark's test outputs here (correlation >= 0.9995).

Home is at its quota on Bridges-2, so caches go to `/ocean`: `export TORCH_HOME=$SUPERRES/envs/.cache/torch
HF_HOME=$SUPERRES/envs/.cache/hf`. The allocation has no CPU partition: CPU work runs in the interactive session or
in a small-GPU job (GPU-shared, `--gpus=v100-16:1`, 5 cores, at most 48 h).

## 1. Data download

**1.1 FTW split tables and FTP's index** (small):

```bash
mkdir -p data/ftw_chips data/ftp_index
for c in austria belgium brazil cambodia corsica croatia denmark estonia finland france germany india kenya latvia \
         lithuania luxembourg netherlands portugal rwanda slovakia slovenia south_africa spain sweden vietnam; do
  curl -s -o data/ftw_chips/chips_$c.parquet \
    https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/kerner-lab/fields-of-the-world/$c/chips_$c.parquet
done
curl -sL -o data/ftp_index/index.parquet \
  https://huggingface.co/datasets/taylor-geospatial/ftw-planet/resolve/main/index.parquet
```

`chips_<country>.parquet`: FTW's chip footprints and train / val / test split. `index.parquet`: FTP's patch list
with `usable_pair` (PlanetScope imagery itself is not published).

**1.2 The paper's chip sets:**

```bash
$SUPERRES/envs/ftp/bin/python scripts/make_chip_lists.py
```

`data/selections/`: `ftp_train.txt` (41,615 chips, all 25 regions), `ftp_val.txt` (393, Austria), `ftp_test.txt`
(5,648), `ftp_test_dense10.txt` (2,544, the ten countries the paper averages), `ftp_all.txt` (47,656).

**1.3 12-band FTW chips with FTW's own procedure** (CPU and internet, ~4 h with 12 workers on 5 cores):

```bash
$SUPERRES/envs/ftp/bin/python -u scripts/ftw12_chips.py --select data/selections/ftp_all.txt --workers 12
```

FTW publishes only R, G, B, NIR and no list of the Sentinel-2 scenes it used. The script runs FTW's chip-creation
procedure (`third_party/ftw-dataset-creation/download_s2_chips.py`, FTW's functions for the scene search and the
in-chip cloud check) with the band list extended to B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12: per chip and
date, the least cloudy Planetary Computer L2A scene of the season window (`data_config_<country>.json`, downloaded
on the way) with < 10 % cloud inside the chip, loaded onto EPSG:4326 at 0.00009009 deg and interpolated to 256 x 256
px, as FTW does. It also downloads the published FTW chip and compares the four shared bands.

Outputs: `data/ftw12/<country>/s2_images/<window_a|window_b>/<aoi_id>.tif` (12 bands, uint16, raw L2A values),
`data/ftw/<country>/s2_images/...` (the published 4-band chips), `data/ftw12/scenes/*.csv` (scene id, cloud share,
`equal_share` = share of identical values). Resumable: written chips are skipped.

**1.4 Repair pass** (~0.7 h with 12 workers):

```bash
$SUPERRES/envs/ftp/bin/python -u scripts/ftw12_repair.py --workers 12
```

About 11 % of chip-dates do not reproduce the published chip after 1.3, because Planetary Computer has since added
reprocessed copies of some scenes and updated cloud values. For those, the script goes through FTW's own candidates
in FTW's order and takes the scene whose four bands equal the published chip. Output:
`data/ftw12/scenes_repair/*.csv`.

**1.5 Final chip lists:**

```bash
$SUPERRES/envs/ftp/bin/python scripts/finalize_ftw12.py
```

Keeps chips whose two dates are FTW's exact chips (94,816 of 95,312 chip-dates); moves the files of the other 248
chips to `data/ftw12_inexact/`. Writes `data/ftw12/scenes_final.csv` and `data/selections/*_exact.txt`: train 41,368,
val 393, test 5,647, dense-10 test 2,543.

## 2. Data preparation: polygons and labels

**2.1 FTW polygons** (~0.5 GB):

```bash
$SUPERRES/envs/ftp/bin/python scripts/download_polygons.py      # -> data/ftw_polygons/boundaries_<country>_<year>.parquet
```

**2.2 Labels** are made per chip in step 3.2 (`scripts/make_sr3m_layout.py`), as FTP describes (Sec. 3): the FTW
polygons are clipped to the chip footprint (`chips_<country>.parquet`) and rasterised on the SR image's grid with
three classes, 0 background, 1 field, 2 boundary; the boundary class is each polygon outline buffered by one pixel
("buffering 3 m (about 1 pixel per side)").

## 3. SR image generation

**3.1 3 m SR with the LDM** (GPU; ~11 s per chip-date on a V100, ~290 V100-hours for all chips):

```bash
for window in window_a window_b; do
  $SUPERRES/envs/ldm/bin/python -u $SUPERRES/inference/predict_ldm.py \
      --input data/ftw12/<country>/s2_images/$window --output data/ldm3m/sr_s36/<country>/$window/ \
      --weights $SUPERRES/ldm --repo $SUPERRES/latent-diffusion --stride 36
done
```

What happens to one chip and date (`predict_ldm.py`, `sr_tiling.py`, unchanged): the 12 bands are clipped to the
training 1st-99th percentile and scaled to [-1, 1]; the 256 x 256 chip (treated as 10 m / px, as the FTW paper
states; no resampling) is padded to 264 x 264 and cut into 48 x 48 tiles every 36 px (7 x 7 = 49 tiles, 12 px
overlap); each tile becomes a 160 x 160 RGB tile (100 DDIM steps, noise seeded by the tile's name); the tiles are
blended into an 853 x 853 px image (256 x 10/3). Output: `data/ldm3m/sr_s36/<country>/<window>/<aoi_id>_SR3m.tif`
(RGB, uint8). `--stride 36` was chosen after comparing 24 (default, 100 tiles), 36 and 42 on 10 test chips: tile
seams at 36 were close to the default at half the time; 42 showed visible tile blocks. Resumable: written images are
skipped. On Bridges-2 the run was split by country into 10 V100 jobs of 30-40 h.

**3.2 NIR, labels and the paper's data layout** (CPU, ~6 s per chip):

```bash
$SUPERRES/envs/ftp/bin/python -u scripts/make_sr3m_layout.py --sr data/ldm3m/sr_s36 --name ldm3m_s36 \
    --train data/selections/ftp_train_exact.txt --val data/selections/ftp_val_exact.txt \
    --test data/selections/ftp_test_exact.txt --workers 4 [--countries <country> --no-index] [--panels 2]
```

Per chip and date, the user's `stack_nir.py` (unchanged) adds Sentinel-2 B08 of the 12-band chip to the SR image
(bilinear onto the SR grid): 4 bands, uint16 (SR R, G, B as 0-255 values, B08 raw L2A), used by the paper's code as it
is. Labels as in 2.2. Output (what the paper's `FTWPlanet` dataset reads):
`data/ftp_layout/ldm3m_s36/planet/<country>/<window_a|window_b|labels>/<aoi_id>.tif`, `planet/index.parquet`,
`ftw/<country>/chips_<country>.parquet`. With `--countries ... --no-index` only the per-chip files are written (run
after each country inside the SR jobs); a final run without them writes the tables. `--panels N`: input panels
(Sentinel-2 | SR | SR with label outlines | labels) in `results/ldm3m_s36_inputs/`.

## 4. Downstream task

**4.1 Training config** = the paper's PRUE+ config (`third_party/fields-of-the-planet/configs/ftw_planet_efnet3_512_prueplus.yaml`) with only the data
location, countries, epochs and logging changed (model, loss, augmentation, crop size, learning rate unchanged):

```bash
$SUPERRES/envs/ftp/bin/python scripts/make_config.py --name ldm3m_s36_b3 --root data/ftp_layout/ldm3m_s36 \
    --epochs 10 --num-workers 10 --val-countries austria --wandb-project ftw-sr-downstream \
    --countries austria belgium brazil cambodia corsica croatia denmark estonia finland france germany india kenya \
                latvia lithuania luxembourg netherlands portugal rwanda slovakia slovenia south_africa spain sweden vietnam
```

Writes `configs/ldm3m_s36_b3.yaml` (git-ignored). `--backbone b7` for EfficientNet-B7.

**4.2 Train** with the paper's launcher (H100; the config uses bf16):

```bash
$SUPERRES/envs/ftp/bin/ftw model fit -c configs/ldm3m_s36_b3.yaml [--ckpt_path runs/ldm3m_s36_b3/checkpoints/last.ckpt]
```

Checkpoints in `runs/ldm3m_s36_b3/checkpoints/` (`last.ckpt` every epoch), CSV log in `runs/ldm3m_s36_b3/log/`, W&B
run `ldm3m_s36_b3` in project `ftw-sr-downstream`. To train further, regenerate the config with a larger `--epochs`
and pass `--ckpt_path .../last.ckpt`: the paper's schedule (AdamW, cosine annealing with T_max = 100 epochs) does not
depend on `max_epochs`, so training in pieces equals one run.

**4.3 Score** with the paper's evaluation (marker-controlled watershed, D4 test-time augmentation, polygon metrics)
on its ten dense-label test countries, plus prediction panels:

```bash
$SUPERRES/envs/ftp/bin/python -u scripts/evaluate_ftp.py --ckpt runs/ldm3m_s36_b3/checkpoints/last.ckpt \
    --root data/ftp_layout/ldm3m_s36 --out results/ldm3m_s36_b3 --panels 2 \
    --countries belgium cambodia croatia germany latvia lithuania portugal slovenia south_africa sweden
```

Outputs: `results/ldm3m_s36_b3/metrics.csv` (one row per country: PQ, SQ, RQ, F1[.5:.95], object F1, pixel IoU,
polygon-count error, boundary error), the macro-average printed at the end, and
`results/ldm3m_s36_b3/panels/<country>/<aoi_id>.jpg` (Sentinel-2 chip | SR input | labels | predicted classes |
predicted fields with the label outlines; the 256 px Sentinel-2 chip is enlarged for display only). The paper's
Sentinel-2 and PlanetScope rows are taken from its Table 1, not rerun.

## 5. Choices to know when reading the results

- FTW's chips are fed to the SR model as delivered and treated as 10 m / px, as the FTW paper states; FTW stores them
  on a ~6 m x 3-6 m lat / lon grid (FTW paper: 1536 m patches resized to 256 x 256 px). The SR output is called 3 m on
  the same terms, and the paper's evaluation converts boundary errors with 3 m per pixel.
- Chips stay in FTW's orientation (rows stored south-up, as FTW writes them); nothing in the papers or code says to
  flip them.
- SR RGB is stored as 0-255 values and divided by 10,000 by the paper's code, unchanged (the code has no option for
  8-bit images); NIR is raw L2A / 10,000.
- 248 chips are left out because their exact FTW scene could not be found (1.5); the dense-10 test set has 2,543 of
  the paper's 2,544 chips.
- Other 3 m SR models with the same input and output format (e.g. `predict_pixeldit.py`) go through steps 3-4 the
  same way, with their own output folder and layout name.
